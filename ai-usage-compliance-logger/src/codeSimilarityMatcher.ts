/**
 * Multi-level code similarity matcher
 * Implements exact, whitespace-normalized, fuzzy, and structural comparisons
 */

import { SimilarityResult, CodeSnippet } from './types';

export class CodeSimilarityMatcher {
  private readonly exactMatchThreshold = 100;
  private readonly whitespacMatchThreshold = 95;
  private readonly normalizedMatchThreshold = 85;
  private readonly fuzzyMatchThreshold = 70;

  /**
   * Find similar code in a list of snippets
   * Returns all matching results in descending order of similarity
   */
  public findSimilarCode(
    targetCode: string,
    snippets: CodeSnippet[],
    minScore = 60
  ): Array<{ snippet: CodeSnippet; result: SimilarityResult }> {
    const results: Array<{ snippet: CodeSnippet; result: SimilarityResult }> = [];

    for (const snippet of snippets) {
      const result = this.compareCodes(targetCode, snippet.code);
      
      if (result.similarityScore >= minScore) {
        results.push({ snippet, result });
      }
    }

    // Sort by similarity score descending
    return results.sort((a, b) => b.result.similarityScore - a.result.similarityScore);
  }

  /**
   * Multi-level code comparison
   * Returns the best match type and overall score (0-100)
   */
  public compareCodes(code1: string, code2: string): SimilarityResult {
    // Level 1: Exact match
    if (code1 === code2) {
      return {
        matchType: 'exact',
        similarityScore: 100,
        explanation: 'Exact character-by-character match'
      };
    }

    // Level 2: Whitespace-insensitive match
    const whitespaceNorm1 = this.normalizeWhitespace(code1);
    const whitespaceNorm2 = this.normalizeWhitespace(code2);
    if (whitespaceNorm1 === whitespaceNorm2) {
      return {
        matchType: 'whitespace',
        similarityScore: 95,
        explanation: 'Match after normalizing whitespace and formatting'
      };
    }

    // Level 3: Syntax normalized match
    const syntaxNorm1 = this.normalizeSyntax(code1);
    const syntaxNorm2 = this.normalizeSyntax(code2);
    if (syntaxNorm1 === syntaxNorm2) {
      return {
        matchType: 'normalized',
        similarityScore: 85,
        explanation: 'Match after normalizing syntax (variable names, etc.)'
      };
    }

    // Level 4: Fuzzy line-based matching
    const fuzzyScore = this.calculateFuzzyLineMatch(code1, code2);
    if (fuzzyScore >= this.fuzzyMatchThreshold) {
      return {
        matchType: 'fuzzy',
        similarityScore: fuzzyScore,
        explanation: `Fuzzy match based on line-by-line similarity (${Math.round(fuzzyScore)}%)`
      };
    }

    // Level 5: Structural analysis (AST-like)
    const structuralScore = this.calculateStructuralSimilarity(code1, code2);
    if (structuralScore >= 60) {
      return {
        matchType: 'structural',
        similarityScore: structuralScore,
        explanation: `Structural similarity: same logic patterns and flow (${Math.round(structuralScore)}%)`
      };
    }

    return {
      matchType: 'none',
      similarityScore: 0,
      explanation: 'No significant similarity detected'
    };
  }

  /**
   * Normalize whitespace and formatting
   */
  private normalizeWhitespace(code: string): string {
    return code
      .replace(/\s+/g, ' ')        // Multiple spaces to single space
      .replace(/\n\s*/g, '\n')      // Remove leading spaces after newlines
      .trim();
  }

  /**
   * Normalize syntax variations
   * - Variable name normalization
   * - Comment removal (optionally)
   * - Consistent spacing around operators
   */
  private normalizeSyntax(code: string): string {
    let normalized = this.normalizeWhitespace(code);

    // Remove comments
    normalized = normalized
      .replace(/\/\/.*$/gm, '')      // Remove // comments
      .replace(/\/\*[\s\S]*?\*\//g, '') // Remove /* */ comments
      .replace(/#.*$/gm, '');         // Remove # comments

    // Normalize variable names to generic placeholders
    // This is basic - for full AST analysis would need proper parser
    normalized = normalized
      .replace(/\b[a-zA-Z_]\w*\b/g, 'VAR')  // Replace identifiers
      .replace(/VAR\s*:/g, 'VAR:');         // But keep structure

    // Normalize string literals
    normalized = normalized
      .replace(/'[^']*'/g, '"STR"')  // Single-quoted strings
      .replace(/"[^"]*"/g, '"STR"')  // Double-quoted strings
      .replace(/`[^`]*`/g, '"STR"'); // Template literals

    // Normalize numbers
    normalized = normalized.replace(/\b\d+\b/g, 'NUM');

    return normalized;
  }

  /**
   * Fuzzy line-based matching
   * Compares line-by-line similarity allowing for some variations
   */
  private calculateFuzzyLineMatch(code1: string, code2: string): number {
    const lines1 = code1.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    const lines2 = code2.split('\n').map(l => l.trim()).filter(l => l.length > 0);

    if (lines1.length === 0 && lines2.length === 0) return 100;
    if (lines1.length === 0 || lines2.length === 0) return 0;

    let matches = 0;
    const maxLen = Math.max(lines1.length, lines2.length);
    const minLen = Math.min(lines1.length, lines2.length);

    for (let i = 0; i < minLen; i++) {
      if (this.areLinesSimilar(lines1[i], lines2[i])) {
        matches++;
      }
    }

    // Calculate percentage match
    const percentage = (matches / maxLen) * 100;
    return Math.min(100, Math.max(0, percentage));
  }

  /**
   * Check if two lines are similar
   * Allows small variations in variable names, formatting
   */
  private areLinesSimilar(line1: string, line2: string): boolean {
    // Exact match
    if (line1 === line2) return true;

    // Whitespace normalized
    if (this.normalizeWhitespace(line1) === this.normalizeWhitespace(line2)) return true;

    // Check if structure is same by normalizing identifiers
    const struct1 = line1.replace(/\b[a-zA-Z_]\w*\b/g, 'VAR');
    const struct2 = line2.replace(/\b[a-zA-Z_]\w*\b/g, 'VAR');
    if (struct1 === struct2) return true;

    // Fuzzy token match
    const tokens1 = line1.match(/\b\w+\b|[^\w\s]/g) || [];
    const tokens2 = line2.match(/\b\w+\b|[^\w\s]/g) || [];

    return this.calculateTokenSimilarity(tokens1, tokens2) > 0.7;
  }

  /**
   * Calculate similarity between two token arrays
   */
  private calculateTokenSimilarity(tokens1: string[], tokens2: string[]): number {
    if (tokens1.length === 0 && tokens2.length === 0) return 1;
    if (tokens1.length === 0 || tokens2.length === 0) return 0;

    const maxLen = Math.max(tokens1.length, tokens2.length);
    let matches = 0;

    for (let i = 0; i < Math.min(tokens1.length, tokens2.length); i++) {
      if (tokens1[i] === tokens2[i]) {
        matches++;
      }
    }

    return matches / maxLen;
  }

  /**
   * Structural similarity analysis
   * Checks for similar code patterns and logic flow
   */
  private calculateStructuralSimilarity(code1: string, code2: string): number {
    let score = 0;
    let checks = 0;

    // Check for similar control flow structures
    const structures = ['if', 'for', 'while', 'return', 'function', 'def', 'class'];
    for (const struct of structures) {
      const count1 = (code1.match(new RegExp(`\\b${struct}\\b`, 'g')) || []).length;
      const count2 = (code2.match(new RegExp(`\\b${struct}\\b`, 'g')) || []).length;
      
      if (count1 > 0 || count2 > 0) {
        if (count1 === count2) {
          score += 10;
        } else {
          score += Math.max(0, 10 - Math.abs(count1 - count2));
        }
        checks++;
      }
    }

    // Check line count similarity
    const lines1 = code1.split('\n').length;
    const lines2 = code2.split('\n').length;
    const lineSimilarity = 1 - Math.abs(lines1 - lines2) / Math.max(lines1, lines2);
    score += lineSimilarity * 30;
    checks++;

    // Check for similar operators and patterns
    const operators = /[+\-*/=<>!&|]/g;
    const ops1 = (code1.match(operators) || []).length;
    const ops2 = (code2.match(operators) || []).length;
    
    if (ops1 > 0 || ops2 > 0) {
      const opSimilarity = 1 - Math.abs(ops1 - ops2) / Math.max(ops1, ops2) / 2;
      score += opSimilarity * 30;
      checks++;
    }

    return checks > 0 ? score / checks : 0;
  }

  /**
   * Check if code block is a partial match of another
   * Useful for detecting when a user copy-pasted only part of a chatbot response
   */
  public isPartialMatch(targetCode: string, fullCode: string): boolean {
    // Check if target is contained in full
    if (fullCode.includes(targetCode)) return true;

    // Check if major lines of target appear in full
    const targetLines = targetCode.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    const fullLines = fullCode.split('\n').map(l => l.trim());

    const matches = targetLines.filter(tl => fullLines.includes(tl)).length;
    const percentage = matches / targetLines.length;

    return percentage > 0.7; // 70% of lines match
  }

  /**
   * Get normalized version of code for storage/comparison
   */
  public normalizeForStorage(code: string): string {
    return this.normalizeWhitespace(code);
  }

  /**
   * Find best match from multiple snippets
   */
  public findBestMatch(
    targetCode: string, 
    snippets: CodeSnippet[]
  ): { snippet: CodeSnippet; result: SimilarityResult } | null {
    const matches = this.findSimilarCode(targetCode, snippets, 50);
    return matches.length > 0 ? matches[0] : null;
  }
}
