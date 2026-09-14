/**
 * Parser for Git diffs to extract newly added/modified code blocks
 * Identifies function-level changes and code hunks
 */

import * as vscode from 'vscode';
import { DiffHunk, DiffLine, CodeBlock } from './types';

export class DiffParser {
  /**
   * Extract all function/class blocks from a document.
   * Used when the caller wants full-file attribution coverage.
   */
  public extractAllFunctionBlocks(
    document: vscode.TextDocument,
    filePath: string
  ): CodeBlock[] {
    const blocks: CodeBlock[] = [];
    const language = this.detectLanguage(filePath);
    const docLines = document.getText().split('\n');

    for (let i = 0; i < docLines.length; i++) {
      const line = docLines[i];
      if (!this.isFunctionHeader(line)) {
        continue;
      }

      const functionName = this.extractFunctionName(line, language);
      const endLine = this.findFunctionEnd(docLines, i);
      const blockContent = this.extractBlockContent(document, i, endLine);
      const hasExistingTagAbove = this.hasAuthorshipTagAbove(docLines, i);

      blocks.push({
        filePath,
        functionName,
        startLine: i,
        endLine,
        language,
        content: blockContent,
        hunks: [],
        isNewFunction: false,
        isModifiedFunction: false,
        hasExistingTagAbove,
        context: blockContent
      });

      // Skip to end of current function to avoid duplicate nested captures.
      i = endLine;
    }

    return blocks;
  }

  /**
   * Parse a unified diff and return hunks
   */
  public parseDiff(diffContent: string, filePath: string): DiffHunk[] {
    const hunks: DiffHunk[] = [];
    const lines = diffContent.split('\n');

    let currentHunk: DiffLine[] = [];
    let hunkHeader: string = '';
    let oldStart = 0, oldCount = 0, newStart = 0, newCount = 0;
    let oldLineNum = 0, newLineNum = 0;

    for (const line of lines) {
      // Detect hunk header @@ -old_start,old_count +new_start,new_count @@
      const hunkMatch = line.match(/@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)/);
      
      if (hunkMatch) {
        // Save previous hunk
        if (currentHunk.length > 0) {
          hunks.push({
            filePath,
            oldStart,
            oldCount,
            newStart,
            newCount,
            lines: currentHunk,
            header: hunkHeader
          });
        }

        // Start new hunk
        oldStart = parseInt(hunkMatch[1]);
        oldCount = parseInt(hunkMatch[2]) || 1;
        newStart = parseInt(hunkMatch[3]);
        newCount = parseInt(hunkMatch[4]) || 1;
        hunkHeader = hunkMatch[5];
        currentHunk = [];
        oldLineNum = oldStart;
        newLineNum = newStart;
      } else if (line.startsWith('+') && !line.startsWith('+++')) {
        // Added line
        currentHunk.push({
          type: 'add',
          content: line.substring(1),
          newLineNumber: newLineNum,
          isFunctionHeader: this.isFunctionHeader(line.substring(1))
        });
        newLineNum++;
      } else if (line.startsWith('-') && !line.startsWith('---')) {
        // Removed line
        currentHunk.push({
          type: 'remove',
          content: line.substring(1),
          oldLineNumber: oldLineNum
        });
        oldLineNum++;
      } else if (line.startsWith(' ')) {
        // Context line (unchanged)
        currentHunk.push({
          type: 'context',
          content: line.substring(1),
          oldLineNumber: oldLineNum,
          newLineNumber: newLineNum
        });
        oldLineNum++;
        newLineNum++;
      }
    }

    // Save last hunk
    if (currentHunk.length > 0) {
      hunks.push({
        filePath,
        oldStart,
        oldCount,
        newStart,
        newCount,
        lines: currentHunk,
        header: hunkHeader
      });
    }

    return hunks;
  }

  /**
   * Extract code blocks from diff hunks
   * Groups related hunks into logical code blocks (functions, classes, etc.)
   */
  public extractCodeBlocks(
    hunks: DiffHunk[],
    document: vscode.TextDocument,
    filePath: string
  ): CodeBlock[] {
    const blocks: CodeBlock[] = [];
    const language = this.detectLanguage(filePath);

    for (const hunk of hunks) {
      // Find function boundaries
      const addedLines = hunk.lines.filter(l => l.type === 'add');
      
      if (addedLines.length === 0) continue;

      // Try to find the function this hunk belongs to
      const functionInfo = this.findContainingFunction(
        document,
        hunk.newStart,
        language
      );

      if (functionInfo) {
        // Create a code block for this function
        const blockContent = this.extractBlockContent(document, functionInfo.startLine, functionInfo.endLine);
        const newCode = addedLines.map(l => l.content).join('\n');

        blocks.push({
          filePath,
          functionName: functionInfo.name,
          startLine: functionInfo.startLine,
          endLine: functionInfo.endLine,
          language,
          content: blockContent,
          hunks: [hunk],
          isNewFunction: functionInfo.isNew,
          isModifiedFunction: !functionInfo.isNew,
          context: newCode
        });
      } else {
        // Create a block for the hunk itself (at file level)
        const blockStartLine = hunk.newStart;
        const blockEndLine = hunk.newStart + hunk.newCount - 1;
        const blockContent = this.extractBlockContent(document, blockStartLine, blockEndLine);

        blocks.push({
          filePath,
          functionName: `HunkAt${blockStartLine}`,
          startLine: blockStartLine,
          endLine: blockEndLine,
          language,
          content: blockContent,
          hunks: [hunk],
          isNewFunction: true,
          isModifiedFunction: false,
          context: blockContent
        });
      }
    }

    return blocks;
  }

  /**
   * Calculate similarity between two code blocks
   * Used for deduplication of blocks
   */
  public areBlocksSimilar(block1: CodeBlock, block2: CodeBlock, threshold = 0.8): boolean {
    if (block1.functionName === block2.functionName && block1.filePath === block2.filePath) {
      return true;
    }

    const similarity = this.calculateContentSimilarity(block1.content, block2.content);
    return similarity >= threshold;
  }

  /**
   * Check if a line is a function header
   */
  private isFunctionHeader(line: string): boolean {
    return /^\s*(async\s+)?(def|function|class|const|let|var|public|private|protected)\s+\w+/.test(line);
  }

  /**
   * Detect programming language from file extension
   */
  private detectLanguage(filePath: string): string {
    const ext = filePath.split('.').pop()?.toLowerCase() || '';
    const languageMap: Record<string, string> = {
      'py': 'python',
      'js': 'javascript',
      'ts': 'typescript',
      'jsx': 'javascript',
      'tsx': 'typescript',
      'java': 'java',
      'c': 'c',
      'cpp': 'cpp',
      'cs': 'csharp',
      'go': 'go',
      'rs': 'rust',
      'rb': 'ruby',
      'php': 'php',
      'sh': 'bash'
    };
    return languageMap[ext] || 'unknown';
  }

  /**
   * Find the function that a line belongs to
   */
  private findContainingFunction(
    document: vscode.TextDocument,
    lineNumber: number,
    language: string
  ): { name: string; startLine: number; endLine: number; isNew: boolean } | null {
    const docLines = document.getText().split('\n');
    
    // Search backwards from the line to find function definition
    for (let i = lineNumber; i >= Math.max(0, lineNumber - 50); i--) {
      const line = docLines[i];
      
      if (this.isFunctionHeader(line)) {
        const functionName = this.extractFunctionName(line, language);
        const endLine = this.findFunctionEnd(docLines, i);
        
        return {
          name: functionName,
          startLine: i,
          endLine: endLine,
          isNew: false // Will be determined by context
        };
      }
    }

    return null;
  }

  /**
   * Extract function name from a function definition line
   */
  private extractFunctionName(line: string, language: string): string {
    // Python: def function_name(
    const pythonMatch = line.match(/def\s+(\w+)/);
    if (pythonMatch) return pythonMatch[1];

    // JavaScript/TypeScript: function name( or const name =
    const jsMatch = line.match(/(?:function\s+(\w+)|(?:const|let|var)\s+(\w+))/);
    if (jsMatch) return jsMatch[1] || jsMatch[2];

    // Class: class ClassName
    const classMatch = line.match(/class\s+(\w+)/);
    if (classMatch) return classMatch[1];

    return 'Unknown';
  }

  /**
   * Find the end line of a function (next function definition or EOF)
   */
  private findFunctionEnd(lines: string[], startLine: number): number {
    const startIndent = lines[startLine].match(/^\s*/)?.[0].length || 0;

    for (let i = startLine + 1; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      // Skip empty lines and comments
      if (trimmed.length === 0 || trimmed.startsWith('#') || trimmed.startsWith('//')) {
        continue;
      }

      const lineIndent = line.match(/^\s*/)?.[0].length || 0;

      // If we find a line with same or less indentation (that's not empty/comment), function ends
      if (lineIndent <= startIndent && this.isFunctionHeader(line)) {
        return i - 1;
      }
    }

    return lines.length - 1;
  }

  /**
   * Extract content of a code block from document
   */
  private extractBlockContent(document: vscode.TextDocument, startLine: number, endLine: number): string {
    const lines = document.getText().split('\n');
    return lines.slice(Math.max(0, startLine), Math.min(lines.length, endLine + 1)).join('\n');
  }

  /**
   * Calculate basic similarity between two code blocks
   */
  private calculateContentSimilarity(code1: string, code2: string): number {
    const lines1 = new Set(code1.split('\n').map(l => l.trim()).filter(l => l));
    const lines2 = new Set(code2.split('\n').map(l => l.trim()).filter(l => l));

    if (lines1.size === 0 && lines2.size === 0) return 1;
    if (lines1.size === 0 || lines2.size === 0) return 0;

    const intersection = [...lines1].filter(l => lines2.has(l)).length;
    const union = new Set([...lines1, ...lines2]).size;

    return intersection / union;
  }

  /**
   * Check whether an authorship tag already exists directly above function header.
   */
  private hasAuthorshipTagAbove(lines: string[], functionStartLine: number): boolean {
    if (functionStartLine <= 0) {
      return false;
    }

    for (let offset = 1; offset <= 2; offset++) {
      const idx = functionStartLine - offset;
      if (idx < 0) {
        break;
      }

      const text = lines[idx].trim();
      if (!text) {
        continue;
      }

      return /(@Authorship:|\[AUTHORSHIP:)/i.test(text);
    }

    return false;
  }

  /**
   * Get only the newly added code from a hunk
   */
  public getAddedCode(hunk: DiffHunk): string {
    return hunk.lines
      .filter(l => l.type === 'add')
      .map(l => l.content)
      .join('\n');
  }

  /**
   * Get only the modified code (both removed and added)
   */
  public getModifiedCode(hunk: DiffHunk): string {
    return hunk.lines
      .map(l => `${l.type[0].toUpperCase()} ${l.content}`)
      .join('\n');
  }
}
