/**
 * Inserts language-aware authorship tags into source code
 * Handles different comment syntax for different languages
 */

import * as vscode from 'vscode';
import { AttributionResult, AuthorshipTag } from './types';

export class CodeTagInserter {
  /**
   * Generate authorship tags for attribution results
   */
  public generateTags(attributions: AttributionResult[]): AuthorshipTag[] {
    const tags: AuthorshipTag[] = [];

    for (const attribution of attributions) {
      const tag = this.createTag(attribution);
      tags.push(tag);
    }

    return tags;
  }

  /**
   * Create a single authorship tag
   */
  private createTag(attribution: AttributionResult): AuthorshipTag {
    const language = attribution.codeBlock.language || 'unknown';
    const commentPrefix = this.getCommentPrefix(language);
    const label = attribution.label;
    const lineSummary = this.buildLineWiseSummary(attribution);

    // Build tag content
    const tagParts = [
      `@Authorship: ${lineSummary || label}`,
      `timestamp=${attribution.timestamp.toISOString().split('T')[0]}`,
    ];

    const tagContent = tagParts.join(' | ');
    const fullTag = `${commentPrefix} ${tagContent}`;

    return {
      codeBlock: attribution.codeBlock,
      label: attribution.label,
      confidence: attribution.confidence,
      lineNumber: attribution.codeBlock.startLine,
      insertBeforeLine: attribution.codeBlock.startLine,
      tagContent: fullTag,
      language,
      sourceMessage: attribution.evidence?.firstSourceMessage
    };
  }

  /**
   * Get comment syntax for language
   */
  private getCommentPrefix(language: string): string {
    const commentMap: Record<string, string> = {
      'python': '#',
      'javascript': '//',
      'typescript': '//',
      'java': '//',
      'c': '//',
      'cpp': '//',
      'csharp': '//',
      'go': '//',
      'rust': '//',
      'php': '//',
      'ruby': '#',
      'shell': '#',
      'bash': '#',
      'yaml': '#',
      'toml': '#',
      'sql': '--',
      'haskell': '--',
      'lua': '--'
    };

    return commentMap[language] || '//';
  }

  /**
   * Insert tags into a document
   * Modifies the actual file
   */
  public async insertTagsIntoDocument(
    document: vscode.TextDocument,
    tags: AuthorshipTag[]
  ): Promise<void> {
    if (tags.length === 0) {
      return;
    }

    // Sort tags by line number (descending) to insert from bottom to top
    // This prevents line number shifts during insertion
    const sortedTags = [...tags].sort((a, b) => b.insertBeforeLine - a.insertBeforeLine);

    const workspaceEdit = new vscode.WorkspaceEdit();

    for (const tag of sortedTags) {
      this.insertTag(workspaceEdit, document, tag);
    }

    await vscode.workspace.applyEdit(workspaceEdit);
  }

  /**
   * Insert a single tag into a document (used within editBuilder)
   */
  private insertTag(
    workspaceEdit: vscode.WorkspaceEdit,
    document: vscode.TextDocument,
    tag: AuthorshipTag
  ): void {
    const lineIndex = this.normalizeToDocumentLine(tag.insertBeforeLine, document);
    const line = document.lineAt(lineIndex);
    const indent = this.getLineIndentation(line.text);
    const existingTagLineIndex = this.findExistingTagLineAbove(document, lineIndex);

    if (existingTagLineIndex !== null) {
      const existingLine = document.lineAt(existingTagLineIndex).text;
      const existingSignature = this.getStableTagSignature(existingLine);
      const newSignature = this.getStableTagSignature(tag.tagContent);

      // Unchanged authorship summary/confidence -> keep previous tag.
      if (existingSignature && newSignature && existingSignature === newSignature) {
        return;
      }

      // Function changed -> replace previous tag in-place.
      workspaceEdit.replace(
        document.uri,
        document.lineAt(existingTagLineIndex).range,
        `${indent}${tag.tagContent}`
      );
      return;
    }

    // Create the tag line with proper indentation
    const tagLine = `${indent}${tag.tagContent}\n`;
    
    // Insert at the beginning of the line
    workspaceEdit.insert(
      document.uri,
      new vscode.Position(lineIndex, 0),
      tagLine
    );
  }

  /**
   * Normalize line numbers from analyzers (often 1-based) to VS Code document lines (0-based).
   */
  private normalizeToDocumentLine(insertBeforeLine: number, document: vscode.TextDocument): number {
    const lastLineIndex = Math.max(0, document.lineCount - 1);
    const clamped = Math.max(0, Math.min(insertBeforeLine, lastLineIndex));
    const clampedMinusOne = Math.max(0, Math.min(insertBeforeLine - 1, lastLineIndex));

    // Prefer the candidate that points to an actual function/class definition.
    if (this.isLikelyFunctionHeader(document.lineAt(clamped).text)) {
      return clamped;
    }

    if (this.isLikelyFunctionHeader(document.lineAt(clampedMinusOne).text)) {
      return clampedMinusOne;
    }

    return clamped;
  }

  /**
   * Find existing tag line directly above the function.
   */
  private findExistingTagLineAbove(document: vscode.TextDocument, lineIndex: number): number | null {
    if (lineIndex <= 0) {
      return null;
    }

    // Check up to 2 lines above (allow one empty spacer line).
    for (let offset = 1; offset <= 2; offset++) {
      const candidate = lineIndex - offset;
      if (candidate < 0) {
        break;
      }

      const candidateText = document.lineAt(candidate).text;
      if (candidateText.trim().length === 0) {
        continue;
      }

      if (this.isAuthorshipTag(candidateText)) {
        return candidate;
      }

      break;
    }

    return null;
  }

  /**
   * Update existing tags
   * Finds and replaces existing authorship tags
   */
  public async updateTagsInDocument(
    document: vscode.TextDocument,
    newTags: AuthorshipTag[]
  ): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    
    if (!editor || editor.document !== document) {
      console.error('Active editor does not match document');
      return;
    }

    const docLines = document.getText().split('\n');
    const updates: Array<{ lineNum: number; newTag: string }> = [];

    // Find existing tags and mark for update
    for (let i = 0; i < docLines.length; i++) {
      const line = docLines[i];
      
      if (this.isAuthorshipTag(line)) {
        // Find corresponding new tag
        const matchingNew = newTags.find(t => {
          // Fuzzy match: same function area
          return Math.abs(t.insertBeforeLine - i) < 5;
        });

        if (matchingNew) {
          updates.push({ lineNum: i, newTag: matchingNew.tagContent });
        }
      }
    }

    // Apply updates
    await editor.edit(editBuilder => {
      for (const update of updates) {
        const line = document.lineAt(update.lineNum);
        editBuilder.replace(line.range, update.newTag);
      }
    });
  }

  /**
   * Remove all authorship tags from a document
   */
  public async removeAllTags(document: vscode.TextDocument): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    
    if (!editor || editor.document !== document) {
      console.error('Active editor does not match document');
      return;
    }

    const docLines = document.getText().split('\n');
    const linesToDelete: number[] = [];

    for (let i = 0; i < docLines.length; i++) {
      if (this.isAuthorshipTag(docLines[i])) {
        linesToDelete.push(i);
      }
    }

    // Delete in reverse order
    await editor.edit(editBuilder => {
      for (let i = linesToDelete.length - 1; i >= 0; i--) {
        const lineNum = linesToDelete[i];
        const line = document.lineAt(lineNum);
        editBuilder.delete(line.rangeIncludingLineBreak);
      }
    });
  }

  /**
   * Check if a line is an authorship tag
   */
  private isAuthorshipTag(line: string): boolean {
    return /(\[AUTHORSHIP:\s*(LLM_GENERATED|HUMAN_PROMPT_ORIGIN|HUMAN_WRITTEN|MIXED|UNCERTAIN)\])|(@Authorship:)/i.test(line);
  }

  /**
   * Extract authorship information from a tag
   */
  public parseTag(tagLine: string): {
    label: string;
    confidence: number;
    origin?: string;
    timestamp?: string;
  } | null {
    const labelMatch = tagLine.match(/\[AUTHORSHIP:\s*(\w+)\]|@Authorship:\s*(\w+)/i);
    const originMatch = tagLine.match(/origin=(\w+)/);
    const timestampMatch = tagLine.match(/timestamp=([^\s|]+)/);

    if (!labelMatch) return null;

    const label = labelMatch[1] || labelMatch[2];

    return {
      label,
      confidence: 0,
      origin: originMatch?.[1],
      timestamp: timestampMatch?.[1]
    };
  }

  /**
   * Get indentation of a line
   */
  private getLineIndentation(lineText: string): string {
    const match = lineText.match(/^(\s*)/);
    return match ? match[1] : '';
  }

  /**
   * Format a tag for display/logging
   */
  public formatTagForDisplay(tag: AuthorshipTag): string {
    return `${tag.tagContent}\n  (Function: ${tag.codeBlock.functionName}, lines ${tag.codeBlock.startLine}-${tag.codeBlock.endLine})`;
  }

  /**
   * Validate tag format
   */
  public isValidTag(tagLine: string): boolean {
    return /(\[AUTHORSHIP:\s*(LLM_GENERATED|HUMAN_PROMPT_ORIGIN|HUMAN_WRITTEN|MIXED|UNCERTAIN)\])|(@Authorship:)/i.test(tagLine);
  }

  /**
   * Build compact line-wise summary like:
   * human - line 1,4-6; llm - line 2-3
   */
  private buildLineWiseSummary(attribution: AttributionResult): string {
    const lines = attribution.lineAttributions || [];
    if (lines.length === 0) {
      return attribution.label;
    }

    const humanLines: number[] = [];
    const llmLines: number[] = [];
    const functionStartLine = attribution.codeBlock.startLine;
    const postInsertionOffset = attribution.codeBlock.hasExistingTagAbove ? 0 : 1;

    for (const line of lines) {
      // Convert function-relative line numbers (1-based) to file-relative line numbers (1-based).
      const fileLineNumber = functionStartLine + line.lineNumber + postInsertionOffset;

      if (line.label === 'LLM_GENERATED') {
        llmLines.push(fileLineNumber);
      } else {
        humanLines.push(fileLineNumber);
      }
    }

    const parts: string[] = [];
    if (humanLines.length > 0) {
      parts.push(`human - file lines ${this.formatLineRanges(humanLines)}`);
    }
    if (llmLines.length > 0) {
      parts.push(`llm - file lines ${this.formatLineRanges(llmLines)}`);
    }

    return parts.join('; ');
  }

  /**
   * Convert [1,2,3,5,6,9] -> "1-3,5-6,9".
   */
  private formatLineRanges(lineNumbers: number[]): string {
    const uniqueSorted = [...new Set(lineNumbers)].sort((a, b) => a - b);
    if (uniqueSorted.length === 0) {
      return '';
    }

    const ranges: string[] = [];
    let rangeStart = uniqueSorted[0];
    let previous = uniqueSorted[0];

    for (let i = 1; i < uniqueSorted.length; i++) {
      const current = uniqueSorted[i];
      if (current === previous + 1) {
        previous = current;
        continue;
      }

      ranges.push(rangeStart === previous ? `${rangeStart}` : `${rangeStart}-${previous}`);
      rangeStart = current;
      previous = current;
    }

    ranges.push(rangeStart === previous ? `${rangeStart}` : `${rangeStart}-${previous}`);
    return ranges.join(',');
  }

  /**
   * Stable tag signature excluding timestamp so idempotent updates still work.
   */
  private getStableTagSignature(tagLine: string): string | null {
    const authorshipMatch = tagLine.match(/@Authorship:\s*([^|]+)/i);

    if (!authorshipMatch) {
      return null;
    }

    return authorshipMatch[1].trim();
  }

  /**
   * Heuristic function/class header detection.
   */
  private isLikelyFunctionHeader(line: string): boolean {
    return /^\s*(def|class|function|async\s+def|async\s+function|public\s+|private\s+|protected\s+|const\s+\w+\s*=\s*\(|let\s+\w+\s*=\s*\(|var\s+\w+\s*=\s*\()/i.test(line);
  }

  /**
   * Get all authorship tags from a document
   */
  public extractTagsFromDocument(document: vscode.TextDocument): string[] {
    const docLines = document.getText().split('\n');
    const tags: string[] = [];

    for (const line of docLines) {
      if (this.isAuthorshipTag(line)) {
        tags.push(line.trim());
      }
    }

    return tags;
  }

  /**
   * Create extended tag format with detailed reasoning
   */
  public createDetailedTag(attribution: AttributionResult): string {
    const language = attribution.codeBlock.language || 'unknown';
    const commentPrefix = this.getCommentPrefix(language);
    
    const basicTag = `[AUTHORSHIP: ${attribution.label}]`;
    const confidenceTag = `confidence=${Math.round(attribution.confidence * 100)}%`;
    const reasoningTag = `reason="${attribution.reasoning.substring(0, 60)}..."`;
    const timestampTag = `timestamp=${attribution.timestamp.toISOString().split('T')[0]}`;
    
    const fullLine = `${commentPrefix} ${basicTag} | ${confidenceTag} | ${reasoningTag} | ${timestampTag}`;
    return fullLine;
  }
}
