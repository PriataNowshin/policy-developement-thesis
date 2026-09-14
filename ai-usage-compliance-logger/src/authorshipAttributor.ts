/**
 * Core attribution engine
 * Determines code authorship based on first appearance in conversation history
 * Implements temporal ordering: user prompt → chatbot response → human written
 */

import { 
  CodeBlock, 
  ParsedConversation, 
  AttributionResult, 
  AuthorshipLabel,
  ProvenanceEvidence,
  ProvenanceMatch,
  LineAttribution
} from './types';
import { ConversationParser } from './conversationParser';
import { CodeSimilarityMatcher } from './codeSimilarityMatcher';

export class AuthorshipAttributor {
  private parser: ConversationParser;
  private matcher: CodeSimilarityMatcher;

  constructor() {
    this.parser = new ConversationParser();
    this.matcher = new CodeSimilarityMatcher();
  }

  /**
   * Attribute a code block to its origin
   * CORE LOGIC: Checks conversation history in temporal order
   * 1. Did user introduce it first in a prompt?
   * 2. Did chatbot generate it first?
   * 3. Is it new (not in conversation)?
   */
  public attributeCodeBlock(
    codeBlock: CodeBlock,
    conversation: ParsedConversation
  ): AttributionResult {
    const timestamp = new Date();
    const lineAttributions = this.attributeLines(codeBlock, conversation);

    // Find all potential matches from the full conversation
    const allMatches = this.findAllMatches(codeBlock.content, conversation);

    if (allMatches.length === 0) {
      // Code not found in conversation → human written
      const labelFromLines = this.getLabelFromLineAttributions(lineAttributions);

      return {
        codeBlock,
        label: labelFromLines,
        confidence: 0.95,
        lineAttributions,
        evidence: null,
        reasoning: 'Code does not appear in conversation history. Likely written directly by human.',
        firstAppearance: null,
        timestamp
      };
    }

    // Sort matches by message index (temporal order)
    const sortedMatches = allMatches.sort((a, b) => a.messageIndex - b.messageIndex);
    const firstAppearanceMatch = sortedMatches[0];

    // Determine label based on first appearance
    if (firstAppearanceMatch.messageRole === 'user') {
      // First appearance is in user prompt
      const confidence = this.calculateConfidence('HUMAN_PROMPT_ORIGIN', firstAppearanceMatch, allMatches);
      
      const result: AttributionResult = {
        codeBlock,
        label: AuthorshipLabel.HUMAN_PROMPT_ORIGIN,
        confidence,
        lineAttributions,
        evidence: this.buildProvenanceEvidence(codeBlock, firstAppearanceMatch, allMatches),
        reasoning: `Code first appeared in user prompt (message ${firstAppearanceMatch.messageIndex}). ` +
                   `Chatbot may have refined or reused it. Human originated the concept.`,
        firstAppearance: {
          messageIndex: firstAppearanceMatch.messageIndex,
          messageRole: firstAppearanceMatch.messageRole,
          timestamp: firstAppearanceMatch.timestamp
        },
        timestamp
      };

      return this.applyLineWiseOverrides(result);
    } else {
      // First appearance is in chatbot response
      // Check if it was in a user prompt before that response
      const userPromptBefore = this.findUserPromptBefore(
        conversation.messages,
        firstAppearanceMatch.messageIndex
      );

      if (userPromptBefore && this.codeIsInMessage(codeBlock.content, userPromptBefore)) {
        // Code was in user prompt, chatbot just echoed it
        const result: AttributionResult = {
          codeBlock,
          label: AuthorshipLabel.HUMAN_PROMPT_ORIGIN,
          confidence: 0.9,
          lineAttributions,
          evidence: this.buildProvenanceEvidence(codeBlock, firstAppearanceMatch, allMatches),
          reasoning: 'Code originated in user prompt (before chatbot response). User is the originator.',
          firstAppearance: {
            messageIndex: userPromptBefore.messageIndex,
            messageRole: 'user',
            timestamp: userPromptBefore.timestamp
          },
          timestamp
        };

        return this.applyLineWiseOverrides(result);
      } else {
        // First appearance is genuinely in chatbot response
        const confidence = this.calculateConfidence('LLM_GENERATED', firstAppearanceMatch, allMatches);
        
        const result: AttributionResult = {
          codeBlock,
          label: AuthorshipLabel.LLM_GENERATED,
          confidence,
          lineAttributions,
          evidence: this.buildProvenanceEvidence(codeBlock, firstAppearanceMatch, allMatches),
          reasoning: `Code first appeared in chatbot response (message ${firstAppearanceMatch.messageIndex}). ` +
                     `Attributed to the assistant response unless modified by human later.`,
          firstAppearance: {
            messageIndex: firstAppearanceMatch.messageIndex,
            messageRole: firstAppearanceMatch.messageRole,
            timestamp: firstAppearanceMatch.timestamp
          },
          timestamp
        };

        return this.applyLineWiseOverrides(result);
      }
    }
  }

  /**
   * Attribute each meaningful line by first appearance in conversation.
   */
  private attributeLines(codeBlock: CodeBlock, conversation: ParsedConversation): LineAttribution[] {
    const lines = codeBlock.content.split('\n');
    const lineAttributions: LineAttribution[] = [];

    for (let i = 0; i < lines.length; i++) {
      const rawLine = lines[i];
      const trimmed = rawLine.trim();

      // Skip blank lines but keep stable indexing for output readability.
      if (!trimmed) {
        continue;
      }

      // Never attribute previously inserted authorship tags.
      if (this.isAuthorshipTagLine(trimmed)) {
        continue;
      }

      const firstAppearance = this.findFirstAppearanceForLine(trimmed, conversation);

      let label = AuthorshipLabel.HUMAN_WRITTEN;
      if (firstAppearance?.messageRole === 'assistant') {
        label = AuthorshipLabel.LLM_GENERATED;
      } else if (firstAppearance?.messageRole === 'user') {
        label = AuthorshipLabel.HUMAN_PROMPT_ORIGIN;
      }

      lineAttributions.push({
        lineNumber: i + 1,
        label
      });
    }

    return lineAttributions;
  }

  /**
   * Detect existing authorship tag lines.
   */
  private isAuthorshipTagLine(line: string): boolean {
    return /(@Authorship:|\[AUTHORSHIP:)/i.test(line);
  }

  /**
   * Find earliest message containing a matching line.
   */
  private findFirstAppearanceForLine(
    targetLine: string,
    conversation: ParsedConversation
  ): { messageIndex: number; messageRole: 'user' | 'assistant' } | null {
    const normalizedTarget = this.normalizeForComparison(targetLine);

    // Avoid noisy matches for punctuation-only lines.
    if (normalizedTarget.length < 3 || !/[a-zA-Z0-9_]/.test(normalizedTarget)) {
      return null;
    }

    // Deterministic rule: first exact normalized line appearance in chronological messages.
    for (const message of conversation.messages) {
      const normalizedLines = this.extractNormalizedLines(message.content);
      if (normalizedLines.has(normalizedTarget)) {
        return {
          messageIndex: message.messageIndex,
          messageRole: message.role
        };
      }
    }

    return null;
  }

  /**
   * Extract normalized non-empty lines from message content.
   */
  private extractNormalizedLines(content: string): Set<string> {
    const lines = content.split('\n');
    const result = new Set<string>();

    for (const line of lines) {
      const normalized = this.normalizeForComparison(line);
      if (!normalized) {
        continue;
      }
      result.add(normalized);
    }

    return result;
  }

  /**
   * Resolve block-level label using line-level labels.
   */
  private getLabelFromLineAttributions(lineAttributions: LineAttribution[]): AuthorshipLabel {
    const hasLLM = lineAttributions.some(line => line.label === AuthorshipLabel.LLM_GENERATED);
    const hasHuman = lineAttributions.some(
      line => line.label === AuthorshipLabel.HUMAN_WRITTEN || line.label === AuthorshipLabel.HUMAN_PROMPT_ORIGIN
    );

    if (hasLLM && hasHuman) {
      return AuthorshipLabel.MIXED;
    }

    if (hasLLM) {
      return AuthorshipLabel.LLM_GENERATED;
    }

    if (lineAttributions.some(line => line.label === AuthorshipLabel.HUMAN_PROMPT_ORIGIN)) {
      return AuthorshipLabel.HUMAN_PROMPT_ORIGIN;
    }

    return AuthorshipLabel.HUMAN_WRITTEN;
  }

  /**
   * Keep block label consistent with line-level attribution.
   */
  private applyLineWiseOverrides(result: AttributionResult): AttributionResult {
    if (!result.lineAttributions || result.lineAttributions.length === 0) {
      return result;
    }

    const lineWiseLabel = this.getLabelFromLineAttributions(result.lineAttributions);
    if (lineWiseLabel === result.label) {
      return result;
    }

    return {
      ...result,
      label: lineWiseLabel,
      reasoning: `${result.reasoning} Line-wise analysis adjusted label to ${lineWiseLabel}.`
    };
  }

  /**
   * Attribute multiple code blocks at once
   */
  public attributeCodeBlocks(
    codeBlocks: CodeBlock[],
    conversation: ParsedConversation
  ): AttributionResult[] {
    return codeBlocks.map(block => this.attributeCodeBlock(block, conversation));
  }

  /**
   * Find all matches in conversation history
   * Uses multi-level similarity matching
   */
  private findAllMatches(
    targetCode: string,
    conversation: ParsedConversation,
    minScore = 60
  ): ProvenanceMatch[] {
    const matches: ProvenanceMatch[] = [];

    for (const snippet of conversation.codeSnippets) {
      const similarity = this.matcher.compareCodes(targetCode, snippet.code);

      if (similarity.similarityScore >= minScore) {
        const message = conversation.messages[snippet.messageIndex];
        
        matches.push({
          messageIndex: snippet.messageIndex,
          messageRole: snippet.messageRole,
          timestamp: message.timestamp,
          matchScore: similarity.similarityScore,
          matchType: similarity.matchType,
          codeSnippet: snippet
        });
      }
    }

    return matches;
  }

  /**
   * Find the user prompt before a given message index
   */
  private findUserPromptBefore(messages: any[], messageIndex: number): any | null {
    for (let i = messageIndex - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        return { ...messages[i], messageIndex: i };
      }
    }
    return null;
  }

  /**
   * Check if target code appears in a message (any form)
   */
  private codeIsInMessage(targetCode: string, message: any): boolean {
    // Check direct inclusion
    if (message.content.includes(targetCode)) return true;

    // Check normalized
    const normalTarget = this.normalizeForComparison(targetCode);
    const normalMsg = this.normalizeForComparison(message.content);
    
    if (normalMsg.includes(normalTarget)) return true;

    // Check by lines
    const targetLines = targetCode.split('\n').map((l: string) => l.trim()).filter((l: string) => l);
    const msgLines = message.content.split('\n').map((l: string) => l.trim());

    const matchedLines = targetLines.filter(tl => msgLines.includes(tl)).length;
    return matchedLines / targetLines.length > 0.7;
  }

  /**
   * Build provenance evidence object
   */
  private buildProvenanceEvidence(
    codeBlock: CodeBlock,
    firstMatch: ProvenanceMatch,
    allMatches: ProvenanceMatch[]
  ): ProvenanceEvidence {
    return {
      firstSourceMessage: {
        role: firstMatch.messageRole,
        content: firstMatch.codeSnippet.context,
        timestamp: firstMatch.timestamp,
        messageIndex: firstMatch.messageIndex
      },
      sourceRole: firstMatch.messageRole,
      sourceMessageIndex: firstMatch.messageIndex,
      similarityResults: allMatches.map(m => ({
        matchType: (m.matchType as any),
        similarityScore: m.matchScore,
        explanation: `Match in message ${m.messageIndex} (${m.messageRole})`
      })) as any,
      bestMatchScore: allMatches[0].matchScore,
      allMatches
    };
  }

  /**
   * Calculate confidence score (0-1)
   */
  private calculateConfidence(
    label: string,
    primaryMatch: ProvenanceMatch,
    allMatches: ProvenanceMatch[]
  ): number {
    let confidence = 0.5;

    // High match score increases confidence
    if (primaryMatch.matchScore >= 90) {
      confidence += 0.4;
    } else if (primaryMatch.matchScore >= 80) {
      confidence += 0.3;
    } else if (primaryMatch.matchScore >= 70) {
      confidence += 0.2;
    } else if (primaryMatch.matchScore >= 60) {
      confidence += 0.1;
    }

    // Exact or whitespace match types increase confidence
    if (primaryMatch.matchType === 'exact' || primaryMatch.matchType === 'whitespace') {
      confidence += 0.1;
    }

    // Multiple matching messages reduce confidence (ambiguous)
    if (allMatches.length > 3) {
      confidence -= 0.05;
    }

    // Ensure between 0 and 1
    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Normalize code for comparison
   */
  private normalizeForComparison(code: string): string {
    return code
      .replace(/\s+/g, ' ')
      .toLowerCase()
      .trim();
  }

  /**
   * Detect mixed authorship
   * When function contains both assistant-response and human-written portions
   */
  public detectMixedAuthorship(
    codeBlock: CodeBlock,
    conversation: ParsedConversation
  ): boolean {
    // Split into logical sections
    const sections = this.splitCodeIntoSections(codeBlock.content);
    
    if (sections.length < 2) return false;

    let hasLLMCode = false;
    let hasHumanCode = false;

    for (const section of sections) {
      const attribution = this.attributeCodeBlock(
        { ...codeBlock, content: section },
        conversation
      );

      if (attribution.label === AuthorshipLabel.LLM_GENERATED) {
        hasLLMCode = true;
      } else if (attribution.label === AuthorshipLabel.HUMAN_WRITTEN || 
                attribution.label === AuthorshipLabel.HUMAN_PROMPT_ORIGIN) {
        hasHumanCode = true;
      }
    }

    return hasLLMCode && hasHumanCode;
  }

  /**
   * Split code into logical sections
   * Simple heuristic: split by function/class definitions
   */
  private splitCodeIntoSections(code: string): string[] {
    const sections: string[] = [];
    let currentSection = '';

    const lines = code.split('\n');
    for (const line of lines) {
      if (/^\s*(def|function|class|async)\s+/.test(line) && currentSection.trim()) {
        sections.push(currentSection);
        currentSection = line + '\n';
      } else {
        currentSection += line + '\n';
      }
    }

    if (currentSection.trim()) {
      sections.push(currentSection);
    }

    return sections.length === 0 ? [code] : sections;
  }

  /**
   * Get attribution summary
   */
  public getSummary(attributions: AttributionResult[]) {
    const summary = {
      total: attributions.length,
      llmGenerated: attributions.filter(a => a.label === AuthorshipLabel.LLM_GENERATED).length,
      humanPrompt: attributions.filter(a => a.label === AuthorshipLabel.HUMAN_PROMPT_ORIGIN).length,
      humanWritten: attributions.filter(a => a.label === AuthorshipLabel.HUMAN_WRITTEN).length,
      mixed: attributions.filter(a => a.label === AuthorshipLabel.MIXED).length,
      uncertain: attributions.filter(a => a.label === AuthorshipLabel.UNCERTAIN).length,
      averageConfidence: attributions.reduce((sum, a) => sum + a.confidence, 0) / attributions.length
    };

    return summary;
  }
}
