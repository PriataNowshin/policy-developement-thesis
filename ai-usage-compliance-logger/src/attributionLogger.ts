/**
 * Structured logging for attribution results
 * Creates JSON audit trails for every analyzed commit
 */

import * as fs from 'fs';
import * as path from 'path';
import { AttributionResult, CommitAnalysisResult, AttributionLogEntry, AuthorshipLabel } from './types';

export class AttributionLogger {
  private logDir: string;

  constructor(logDir?: string) {
    this.logDir = logDir || path.join(process.cwd(), '.authorship-logs');
    this.ensureLogDirExists();
  }

  /**
   * Create log directory if it doesn't exist
   */
  private ensureLogDirExists(): void {
    if (!fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
    }
  }

  /**
   * Log a complete analysis result
   */
  public logAnalysis(analysisResult: CommitAnalysisResult): string {
    const filename = `${analysisResult.commitHash}_${Date.now()}.json`;
    const filepath = path.join(this.logDir, filename);

    const logData = {
      metadata: {
        commitHash: analysisResult.commitHash,
        timestamp: analysisResult.timestamp.toISOString(),
        filesAnalyzed: analysisResult.filesAnalyzed,
        totalCodeBlocks: analysisResult.codeBlocksFound.length
      },
      statistics: analysisResult.statistics,
      attributions: analysisResult.attributions.map(a => this.formatAttributionForLog(a)),
      tags: analysisResult.tags.map(t => ({
        functionName: t.codeBlock.functionName,
        filePath: t.codeBlock.filePath,
        lineRange: `${t.codeBlock.startLine}-${t.codeBlock.endLine}`,
        tagContent: t.tagContent,
        label: t.label,
        confidence: t.confidence
      }))
    };

    fs.writeFileSync(filepath, JSON.stringify(logData, null, 2), 'utf-8');
    this.writeLatestViewFiles(logData);
    return filepath;
  }

  /**
   * Write stable "latest" files in workspace root for easy viewing in Explorer.
   */
  private writeLatestViewFiles(logData: any): void {
    const latestJsonPath = path.join(process.cwd(), 'AUTHORSHIP_LATEST_LOG.json');
    fs.writeFileSync(latestJsonPath, JSON.stringify(logData, null, 2), 'utf-8');

    const summaryLines: string[] = [];
    summaryLines.push('# Latest Authorship Analysis');
    summaryLines.push('');
    summaryLines.push(`- Commit: ${logData.metadata.commitHash}`);
    summaryLines.push(`- Timestamp: ${logData.metadata.timestamp}`);
    summaryLines.push(`- Files Analyzed: ${logData.metadata.filesAnalyzed.join(', ')}`);
    summaryLines.push('');
    summaryLines.push('## Statistics');
    summaryLines.push('');
    summaryLines.push(`- Total Blocks: ${logData.metadata.totalCodeBlocks}`);
    summaryLines.push(`- LLM Generated: ${logData.statistics.llmGeneratedLines}`);
    summaryLines.push(`- Human Prompt: ${logData.statistics.humanPromptLines}`);
    summaryLines.push(`- Human Written: ${logData.statistics.humanWrittenLines}`);
    summaryLines.push(`- Mixed: ${logData.statistics.mixedLines}`);
    summaryLines.push(`- Uncertain: ${logData.statistics.uncertainLines}`);
    summaryLines.push('');
    summaryLines.push('## Per-Function');
    summaryLines.push('');

    for (const attr of logData.attributions) {
      summaryLines.push(`- ${attr.filePath} :: ${attr.functionName} :: ${attr.detectedLabel} (${(attr.confidenceScore * 100).toFixed(1)}%)`);
    }

    const latestMdPath = path.join(process.cwd(), 'AUTHORSHIP_LATEST_REPORT.md');
    fs.writeFileSync(latestMdPath, summaryLines.join('\n'), 'utf-8');
  }

  /**
   * Format attribution result for JSON logging
   */
  private formatAttributionForLog(attribution: AttributionResult): AttributionLogEntry {
    return {
      commitHash: attribution.codeBlock.filePath, // Will be updated by caller
      timestamp: attribution.timestamp.toISOString(),
      filePath: attribution.codeBlock.filePath,
      functionName: attribution.codeBlock.functionName,
      codeBlockLineRange: `${attribution.codeBlock.startLine}-${attribution.codeBlock.endLine}`,
      detectedLabel: attribution.label,
      confidenceScore: attribution.confidence,
      matchingConversationSegment: attribution.evidence ? {
        messageIndex: attribution.evidence.sourceMessageIndex,
        messageRole: attribution.evidence.sourceRole,
        timestamp: attribution.evidence.firstSourceMessage.timestamp.toISOString(),
        snippet: attribution.evidence.firstSourceMessage.content.substring(0, 200)
      } : null,
      firstMatchSource: attribution.evidence 
        ? (attribution.evidence.sourceRole === 'user' ? 'user_prompt' : 'chatbot_response')
        : 'not_found',
      reasoning: attribution.reasoning,
      similarityMatches: attribution.evidence?.allMatches.map(m => ({
        type: m.matchType,
        score: m.matchScore,
        messageIndex: m.messageIndex
      })) || []
    };
  }

  /**
   * Log per-function attribution
   */
  public logFunctionAttribution(
    commitHash: string,
    filePath: string,
    attribution: AttributionResult
  ): void {
    const logEntry: AttributionLogEntry = {
      commitHash,
      timestamp: attribution.timestamp.toISOString(),
      filePath,
      functionName: attribution.codeBlock.functionName,
      codeBlockLineRange: `${attribution.codeBlock.startLine}-${attribution.codeBlock.endLine}`,
      detectedLabel: attribution.label,
      confidenceScore: attribution.confidence,
      matchingConversationSegment: attribution.evidence ? {
        messageIndex: attribution.evidence.sourceMessageIndex,
        messageRole: attribution.evidence.sourceRole,
        timestamp: attribution.evidence.firstSourceMessage.timestamp.toISOString(),
        snippet: attribution.evidence.firstSourceMessage.content.substring(0, 200)
      } : null,
      firstMatchSource: attribution.evidence
        ? (attribution.evidence.sourceRole === 'user' ? 'user_prompt' : 'chatbot_response')
        : 'not_found',
      reasoning: attribution.reasoning,
      similarityMatches: attribution.evidence?.allMatches.map(m => ({
        type: m.matchType,
        score: m.matchScore,
        messageIndex: m.messageIndex
      })) || []
    };

    const logPath = path.join(this.logDir, 'function-attributions.jsonl');
    const logLine = JSON.stringify(logEntry) + '\n';
    
    fs.appendFileSync(logPath, logLine, 'utf-8');
  }

  /**
   * Get summary of attributions
   */
  public getSummaryReport(attributions: AttributionResult[]): {
    totalFunctions: number;
    llmGenerated: number;
    humanPrompt: number;
    humanWritten: number;
    mixed: number;
    uncertain: number;
    averageConfidence: number;
    byLabel: Record<string, number>;
  } {
    const summary = {
      totalFunctions: attributions.length,
      llmGenerated: 0,
      humanPrompt: 0,
      humanWritten: 0,
      mixed: 0,
      uncertain: 0,
      averageConfidence: 0,
      byLabel: {} as Record<string, number>
    };

    let totalConfidence = 0;

    for (const attr of attributions) {
      summary.byLabel[attr.label] = (summary.byLabel[attr.label] || 0) + 1;
      totalConfidence += attr.confidence;

      switch (attr.label) {
        case AuthorshipLabel.LLM_GENERATED:
          summary.llmGenerated++;
          break;
        case AuthorshipLabel.HUMAN_PROMPT_ORIGIN:
          summary.humanPrompt++;
          break;
        case AuthorshipLabel.HUMAN_WRITTEN:
          summary.humanWritten++;
          break;
        case AuthorshipLabel.MIXED:
          summary.mixed++;
          break;
        case AuthorshipLabel.UNCERTAIN:
          summary.uncertain++;
          break;
      }
    }

    summary.averageConfidence = attributions.length > 0 ? totalConfidence / attributions.length : 0;

    return summary;
  }

  /**
   * Create human-readable report
   */
  public createReport(analysisResult: CommitAnalysisResult): string {
    const summary = this.getSummaryReport(analysisResult.attributions);
    
    let report = `
================================================================================
                    CODE AUTHORSHIP ATTRIBUTION REPORT
================================================================================

Commit: ${analysisResult.commitHash}
Timestamp: ${analysisResult.timestamp.toISOString()}
Files Analyzed: ${analysisResult.filesAnalyzed.join(', ')}

SUMMARY
-------
Total Code Blocks Analyzed: ${summary.totalFunctions}
LLM Generated: ${summary.llmGenerated} (${this.percentage(summary.llmGenerated, summary.totalFunctions)}%)
Human Prompt Origin: ${summary.humanPrompt} (${this.percentage(summary.humanPrompt, summary.totalFunctions)}%)
Human Written: ${summary.humanWritten} (${this.percentage(summary.humanWritten, summary.totalFunctions)}%)
Mixed (LLM + Human): ${summary.mixed} (${this.percentage(summary.mixed, summary.totalFunctions)}%)
Uncertain: ${summary.uncertain} (${this.percentage(summary.uncertain, summary.totalFunctions)}%)
Average Confidence: ${(summary.averageConfidence * 100).toFixed(1)}%

STATISTICS
----------
Total New Lines: ${analysisResult.statistics.totalNewLines}
Analyzed Lines: ${analysisResult.statistics.analyzedLines}
LLM Generated Lines: ${analysisResult.statistics.llmGeneratedLines}
Human Prompt Lines: ${analysisResult.statistics.humanPromptLines}
Human Written Lines: ${analysisResult.statistics.humanWrittenLines}
Mixed Lines: ${analysisResult.statistics.mixedLines}
Uncertain Lines: ${analysisResult.statistics.uncertainLines}

DETAILED ATTRIBUTIONS
---------------------
`;

    // Sort by confidence descending
    const sorted = [...analysisResult.attributions].sort((a, b) => b.confidence - a.confidence);

    for (const attr of sorted) {
      report += `
File: ${attr.codeBlock.filePath}
Function: ${attr.codeBlock.functionName}
Lines: ${attr.codeBlock.startLine}-${attr.codeBlock.endLine}
Label: ${attr.label}
Confidence: ${(attr.confidence * 100).toFixed(1)}%
Reasoning: ${attr.reasoning}
${attr.firstAppearance ? `First Appearance: Message ${attr.firstAppearance.messageIndex} (${attr.firstAppearance.messageRole})` : 'First Appearance: Not found in conversation'}
---
`;
    }

    report += `
================================================================================
Generated: ${new Date().toISOString()}
================================================================================
`;

    return report;
  }

  /**
   * Helper to calculate percentage
   */
  private percentage(value: number, total: number): string {
    if (total === 0) return '0';
    return ((value / total) * 100).toFixed(1);
  }

  /**
   * List all log files
   */
  public listLogs(): string[] {
    if (!fs.existsSync(this.logDir)) {
      return [];
    }
    return fs.readdirSync(this.logDir).filter(f => f.endsWith('.json') || f.endsWith('.jsonl'));
  }

  /**
   * Read a specific log file
   */
  public readLog(filename: string): any {
    const filepath = path.join(this.logDir, filename);
    if (!fs.existsSync(filepath)) {
      throw new Error(`Log file not found: ${filepath}`);
    }
    const content = fs.readFileSync(filepath, 'utf-8');
    return JSON.parse(content);
  }

  /**
   * Export logs to CSV for spreadsheet analysis
   */
  public exportToCSV(attributions: AttributionResult[]): string {
    const headers = [
      'File Path',
      'Function Name',
      'Line Range',
      'Label',
      'Confidence',
      'First Appearance',
      'Source Role',
      'Reasoning'
    ];

    const rows = attributions.map(attr => [
      attr.codeBlock.filePath,
      attr.codeBlock.functionName,
      `${attr.codeBlock.startLine}-${attr.codeBlock.endLine}`,
      attr.label,
      `${(attr.confidence * 100).toFixed(1)}%`,
      attr.firstAppearance ? `Message ${attr.firstAppearance.messageIndex}` : 'Not Found',
      attr.firstAppearance?.messageRole || 'N/A',
      `"${attr.reasoning}"`
    ]);

    let csv = headers.join(',') + '\n';
    for (const row of rows) {
      csv += row.join(',') + '\n';
    }

    return csv;
  }

  /**
   * Get log directory path
   */
  public getLogDir(): string {
    return this.logDir;
  }

  /**
   * Clear old logs (keep last N)
   */
  public clearOldLogs(keepCount = 10): void {
    const logs = this.listLogs();
    
    if (logs.length > keepCount) {
      const toDelete = logs
        .sort()
        .slice(0, logs.length - keepCount);
      
      for (const file of toDelete) {
        fs.unlinkSync(path.join(this.logDir, file));
      }
    }
  }
}
