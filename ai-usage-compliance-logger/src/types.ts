/**
 * Core types and interfaces for the code authorship tracking system
 */

/**
 * Authorship labels determined by the attribution engine
 */
export enum AuthorshipLabel {
  LLM_GENERATED = 'LLM_GENERATED',
  HUMAN_PROMPT_ORIGIN = 'HUMAN_PROMPT_ORIGIN',
  HUMAN_WRITTEN = 'HUMAN_WRITTEN',
  MIXED = 'MIXED',
  UNCERTAIN = 'UNCERTAIN'
}

/**
 * Conversation message with metadata
 */
export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  modelName?: string;
  messageIndex: number;
}

/**
 * Code snippet extracted from conversation
 */
export interface CodeSnippet {
  code: string;
  language?: string;
  messageIndex: number;
  messageRole: 'user' | 'assistant';
  context: string;
  timestamp: Date;
  normalization?: {
    original: string;
    whitespaceNormalized: boolean;
    syntaxNormalized: boolean;
  };
}

/**
 * Parsed conversation with indexed code snippets
 */
export interface ParsedConversation {
  messages: ConversationMessage[];
  codeSnippets: CodeSnippet[];
  totalMessages: number;
  startTime: Date;
  endTime: Date;
}

/**
 * Git diff hunk (contiguous change)
 */
export interface DiffHunk {
  filePath: string;
  oldStart: number;
  oldCount: number;
  newStart: number;
  newCount: number;
  lines: DiffLine[];
  header?: string;
}

/**
 * Individual line in a diff
 */
export interface DiffLine {
  type: 'add' | 'remove' | 'context';
  content: string;
  oldLineNumber?: number;
  newLineNumber?: number;
  isFunctionHeader?: boolean;
  functionName?: string;
}

/**
 * Code block extracted from diff (represents new/modified code)
 */
export interface CodeBlock {
  filePath: string;
  functionName: string;
  startLine: number;
  endLine: number;
  language: string;
  content: string;
  hunks: DiffHunk[];
  isNewFunction: boolean;
  isModifiedFunction: boolean;
  hasExistingTagAbove?: boolean;
  context?: string;
}

/**
 * Result of similarity comparison
 */
export interface SimilarityResult {
  matchType: 'exact' | 'whitespace' | 'normalized' | 'fuzzy' | 'structural' | 'none';
  similarityScore: number; // 0-100
  matchedSnippet?: CodeSnippet;
  explanation: string;
}

/**
 * Provenance evidence from conversation
 */
export interface ProvenanceEvidence {
  firstSourceMessage: ConversationMessage;
  sourceRole: 'user' | 'assistant';
  sourceMessageIndex: number;
  similarityResults: SimilarityResult[];
  bestMatchScore: number;
  allMatches: ProvenanceMatch[];
}

/**
 * Individual provenance match
 */
export interface ProvenanceMatch {
  messageIndex: number;
  messageRole: 'user' | 'assistant';
  timestamp: Date;
  matchScore: number;
  matchType: string;
  codeSnippet: CodeSnippet;
}

/**
 * Authorship attribution result
 */
export interface AttributionResult {
  codeBlock: CodeBlock;
  label: AuthorshipLabel;
  confidence: number; // 0-1
  lineAttributions?: LineAttribution[];
  evidence: ProvenanceEvidence | null;
  reasoning: string;
  firstAppearance: {
    messageIndex: number;
    messageRole: 'user' | 'assistant';
    timestamp: Date;
  } | null;
  timestamp: Date;
  commitHash?: string;
}

/**
 * Line-wise authorship attribution inside a function/code block.
 * lineNumber is relative to the function (1-based).
 */
export interface LineAttribution {
  lineNumber: number;
  label: AuthorshipLabel;
}

/**
 * Authorship tag to be inserted into source code
 */
export interface AuthorshipTag {
  codeBlock: CodeBlock;
  label: AuthorshipLabel;
  confidence: number;
  lineNumber: number;
  insertBeforeLine: number;
  tagContent: string;
  language: string;
  sourceMessage?: ConversationMessage;
}

/**
 * Complete analysis result for a commit
 */
export interface CommitAnalysisResult {
  commitHash: string;
  timestamp: Date;
  filesAnalyzed: string[];
  codeBlocksFound: CodeBlock[];
  attributions: AttributionResult[];
  tags: AuthorshipTag[];
  logPath?: string;
  statistics: {
    totalNewLines: number;
    analyzedLines: number;
    llmGeneratedLines: number;
    humanPromptLines: number;
    humanWrittenLines: number;
    mixedLines: number;
    uncertainLines: number;
  };
}

/**
 * Structured log entry for audit trail
 */
export interface AttributionLogEntry {
  commitHash: string;
  timestamp: string;
  filePath: string;
  functionName: string;
  codeBlockLineRange: string;
  detectedLabel: AuthorshipLabel;
  confidenceScore: number;
  matchingConversationSegment: {
    messageIndex: number;
    messageRole: 'user' | 'assistant';
    timestamp: string;
    snippet: string;
  } | null;
  firstMatchSource: 'user_prompt' | 'chatbot_response' | 'not_found';
  reasoning: string;
  similarityMatches: Array<{
    type: string;
    score: number;
    messageIndex: number;
  }>;
}

/**
 * Configuration for the authorship tracking system
 */
export interface AuthorshipConfig {
  minSimilarityThreshold: number; // 0-100, default 75
  fuzzyMatchThreshold: number; // 0-100, default 60
  analyzeFullConversation: boolean; // default true
  insertTags: boolean; // default true
  createLogs: boolean; // default true
  logOutputPath?: string;
  languageCommentMap?: Record<string, string>;
}

/**
 * Code normalization options for similarity matching
 */
export interface NormalizationOptions {
  removeWhitespace: boolean;
  removeComments: boolean;
  normalizeIdentifiers: boolean;
  normalizeFormatting: boolean;
  structuralAnalysis: boolean;
}

/**
 * AST-based structural comparison result
 */
export interface StructuralComparisonResult {
  isSyntacticallyEqual: boolean;
  isFunctionallyEqual: boolean;
  structureMatch: number; // 0-100
  differences: string[];
}
