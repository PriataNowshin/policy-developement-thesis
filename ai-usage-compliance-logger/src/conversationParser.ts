/**
 * Parses chatbot conversation history and extracts code snippets
 * Analyzes the FULL conversation history without artificial limits
 */

import { ConversationMessage, ParsedConversation, CodeSnippet } from './types';

export class ConversationParser {
  /**
   * Parse full conversation history and extract all code snippets
   */
  public parseConversation(messages: any[]): ParsedConversation {
    const conversationMessages: ConversationMessage[] = messages.map((msg, index) => ({
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp),
      modelName: msg.modelName,
      messageIndex: index
    }));

    const codeSnippets = this.extractCodeSnippets(conversationMessages);

    return {
      messages: conversationMessages,
      codeSnippets,
      totalMessages: conversationMessages.length,
      startTime: conversationMessages.length > 0 ? conversationMessages[0].timestamp : new Date(),
      endTime: conversationMessages.length > 0 
        ? conversationMessages[conversationMessages.length - 1].timestamp 
        : new Date()
    };
  }

  /**
   * Extract ALL code snippets from conversation (no limits)
   */
  private extractCodeSnippets(messages: ConversationMessage[]): CodeSnippet[] {
    const snippets: CodeSnippet[] = [];

    messages.forEach((msg, index) => {
      // Extract from code blocks (```)
      const codeBlocks = this.extractCodeBlocks(msg.content);
      codeBlocks.forEach(block => {
        snippets.push({
          code: block.code,
          language: block.language,
          messageIndex: index,
          messageRole: msg.role,
          context: msg.content,
          timestamp: msg.timestamp
        });
      });

      // Extract inline code patterns
      const inlinePatterns = this.extractInlineCodePatterns(msg.content);
      inlinePatterns.forEach(pattern => {
        snippets.push({
          code: pattern,
          messageIndex: index,
          messageRole: msg.role,
          context: msg.content,
          timestamp: msg.timestamp
        });
      });

      // Extract function/class definitions
      const functionDefs = this.extractFunctionDefinitions(msg.content);
      functionDefs.forEach(def => {
        snippets.push({
          code: def,
          messageIndex: index,
          messageRole: msg.role,
          context: msg.content,
          timestamp: msg.timestamp
        });
      });

      // Extract import statements
      const imports = this.extractImportStatements(msg.content);
      imports.forEach(imp => {
        snippets.push({
          code: imp,
          messageIndex: index,
          messageRole: msg.role,
          context: msg.content,
          timestamp: msg.timestamp
        });
      });

      // Extract code provided as context in user prompts
      if (msg.role === 'user') {
        const contextCode = this.extractContextCode(msg.content);
        contextCode.forEach(code => {
          snippets.push({
            code,
            messageIndex: index,
            messageRole: msg.role,
            context: msg.content,
            timestamp: msg.timestamp
          });
        });
      }
    });

    return snippets;
  }

  /**
   * Extract code blocks marked with triple backticks
   */
  private extractCodeBlocks(content: string): Array<{ code: string; language: string }> {
    const blocks: Array<{ code: string; language: string }> = [];
    
    // Match ```language...code...```
    const regex = /```([\w]*)\n?([\s\S]*?)```/g;
    let match;

    while ((match = regex.exec(content)) !== null) {
      const language = match[1] || 'unknown';
      const code = match[2].trim();
      
      if (code.length > 0) {
        blocks.push({ code, language });
      }
    }

    return blocks;
  }

  /**
   * Extract inline code patterns that look like actual code
   */
  private extractInlineCodePatterns(content: string): string[] {
    const patterns: string[] = [];
    
    // Single-line code patterns
    const codePatterns = [
      /def\s+\w+\s*\([^)]*\):[^\n]*/g,          // Python functions
      /class\s+\w+[^\n]*/g,                      // Class definitions
      /function\s+\w+\s*\([^)]*\)\s*\{[^}]*\}/g, // JS functions
      /const\s+\w+\s*=\s*[^;]+;?/g,             // Const assignments
      /let\s+\w+\s*=\s*[^;]+;?/g,               // Let assignments
      /import\s+[^\n]+from\s+['\"][^'\"]+['\"]/g, // Imports
      /async\s+(function|def)\s+[^\n]+/g,        // Async
      /await\s+[^\n]+/g,                        // Await
    ];

    for (const pattern of codePatterns) {
      let match;
      while ((match = pattern.exec(content)) !== null) {
        const code = match[0].trim();
        if (code.length > 0 && !patterns.includes(code)) {
          patterns.push(code);
        }
      }
    }

    return patterns;
  }

  /**
   * Extract function and class definitions
   */
  private extractFunctionDefinitions(content: string): string[] {
    const definitions: string[] = [];
    
    // Match multi-line function/class definitions
    const functionRegex = /((?:async\s+)?(?:def|function|class)\s+\w+[^]*?(?=\n(?:def|function|class|$)))/g;
    let match;

    while ((match = functionRegex.exec(content)) !== null) {
      const def = match[0].trim();
      if (def.length > 0 && !definitions.includes(def)) {
        definitions.push(def);
      }
    }

    return definitions;
  }

  /**
   * Extract import statements
   */
  private extractImportStatements(content: string): string[] {
    const imports: string[] = [];
    
    const importRegex = /(^|\n)(import\s+.+|from\s+.+\s+import\s+.+)/gm;
    let match;

    while ((match = importRegex.exec(content)) !== null) {
      const imp = match[2].trim();
      if (imp.length > 0 && !imports.includes(imp)) {
        imports.push(imp);
      }
    }

    return imports;
  }

  /**
   * Extract code provided as context (user showing their current file)
   */
  private extractContextCode(content: string): string[] {
    const code: string[] = [];
    
    // Look for common context markers
    const contextMarkers = {
      'Current file:': content.match(/Current file[:\s]*```?([\s\S]*?)```/i),
      'My code:': content.match(/My code[:\s]*```?([\s\S]*?)```/i),
      'Here\'s my code:': content.match(/Here'?s my code[:\s]*```?([\s\S]*?)```/i),
      'Code:': content.match(/^[Cc]ode[:\s]*```?([\s\S]*?)```/m),
    };

    for (const [, match] of Object.entries(contextMarkers)) {
      if (match && match[1]) {
        const extractedCode = match[1].trim();
        if (extractedCode.length > 0 && !code.includes(extractedCode)) {
          code.push(extractedCode);
        }
      }
    }

    // Also extract indented code blocks (4+ spaces)
    const lines = content.split('\n');
    let inIndentedBlock = false;
    let blockContent = '';

    for (const line of lines) {
      if (line.match(/^(\s{4,}|\t+)\S/)) {
        blockContent += line.trim() + '\n';
        inIndentedBlock = true;
      } else if (inIndentedBlock && line.trim().length > 0) {
        if (blockContent.length > 0) {
          code.push(blockContent.trim());
        }
        blockContent = '';
        inIndentedBlock = false;
      }
    }

    if (blockContent.length > 0) {
      code.push(blockContent.trim());
    }

    return code;
  }

  /**
   * Get all user prompts (for context)
   */
  public getUserPrompts(parsed: ParsedConversation): ConversationMessage[] {
    return parsed.messages.filter(msg => msg.role === 'user');
  }

  /**
   * Get all assistant responses
   */
  public getAssistantResponses(parsed: ParsedConversation): ConversationMessage[] {
    return parsed.messages.filter(msg => msg.role === 'assistant');
  }

  /**
   * Find code snippets from a specific role (user or assistant)
   */
  public getSnippetsByRole(parsed: ParsedConversation, role: 'user' | 'assistant'): CodeSnippet[] {
    return parsed.codeSnippets.filter(snippet => snippet.messageRole === role);
  }

  /**
   * Get code snippets in temporal order (earliest first)
   */
  public getSnippetsInOrder(parsed: ParsedConversation): CodeSnippet[] {
    return [...parsed.codeSnippets].sort((a, b) => a.messageIndex - b.messageIndex);
  }

  /**
   * Find the first appearance of similar code in conversation
   * Returns: { messageIndex, role, timestamp }
   */
  public findFirstAppearance(
    parsed: ParsedConversation, 
    targetSnippet: string, 
    similarityFn: (a: string, b: string) => number
  ): { messageIndex: number; role: 'user' | 'assistant'; timestamp: Date } | null {
    const orderedSnippets = this.getSnippetsInOrder(parsed);
    
    for (const snippet of orderedSnippets) {
      const score = similarityFn(targetSnippet, snippet.code);
      if (score > 0.7) { // 70% similarity threshold
        return {
          messageIndex: snippet.messageIndex,
          role: snippet.messageRole,
          timestamp: snippet.timestamp
        };
      }
    }

    return null;
  }

  /**
   * Get conversation summary stats
   */
  public getStats(parsed: ParsedConversation) {
    return {
      totalMessages: parsed.messages.length,
      totalCodeSnippets: parsed.codeSnippets.length,
      userMessages: parsed.messages.filter(m => m.role === 'user').length,
      assistantMessages: parsed.messages.filter(m => m.role === 'assistant').length,
      userCodeSnippets: parsed.codeSnippets.filter(s => s.messageRole === 'user').length,
      assistantCodeSnippets: parsed.codeSnippets.filter(s => s.messageRole === 'assistant').length,
      conversationDuration: (parsed.endTime.getTime() - parsed.startTime.getTime()) / 1000 // seconds
    };
  }
}
