/**
 * Git change tracker and authorship attribution orchestrator
 * Listens for file save events, analyzes diffs, and attributes code authorship
 */

import * as vscode from 'vscode';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { ChatbotPanel } from './chatbotPanel';
import { ConversationParser } from './conversationParser';
import { DiffParser } from './diffParser';
import { CodeSimilarityMatcher } from './codeSimilarityMatcher';
import { AuthorshipAttributor } from './authorshipAttributor';
import { CodeTagInserter } from './codeTagInserter';
import { AttributionLogger } from './attributionLogger';
import { CommitAnalysisResult, AuthorshipConfig } from './types';

export class GitChangeTracker {
  private context: vscode.ExtensionContext;
  private conversationParser: ConversationParser;
  private diffParser: DiffParser;
  private attributor: AuthorshipAttributor;
  private tagInserter: CodeTagInserter;
  private logger: AttributionLogger;
  private config: AuthorshipConfig;
  private lastAnalyzedCommit: string | null = null;
  private git: any = null;
  private repositorySubscriptions = new Map<string, vscode.Disposable>();
  private execFileAsync = promisify(execFile);

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.conversationParser = new ConversationParser();
    this.diffParser = new DiffParser();
    this.attributor = new AuthorshipAttributor();
    this.tagInserter = new CodeTagInserter();
    this.logger = new AttributionLogger();

    // Default configuration
    this.config = {
      minSimilarityThreshold: 75,
      fuzzyMatchThreshold: 60,
      analyzeFullConversation: true,
      insertTags: true,
      createLogs: true
    };
  }

  /**
   * Activate the git change tracker
   */
  public async activate(): Promise<void> {
    const isMonitoring = await this.setupGitCommitListener();

    if (isMonitoring) {
      console.log('[AuthorshipTracker] Git change tracker activated - monitoring commits');
    } else {
      console.log('[AuthorshipTracker] Git change tracker activated - waiting for repository');
    }
  }

  /**
   * Setup listener for git commit events
   */
  private async setupGitCommitListener(): Promise<boolean> {
    try {
      this.git = await this.getGitExtension();
      if (!this.git) {
        console.log('[AuthorshipTracker] Git not available - tracker inactive');
        return false;
      }

      // Attach to any repositories already open.
      for (const repo of this.git.repositories) {
        await this.attachRepositoryListener(repo);
      }

      // Also handle repositories that open after startup.
      if (typeof this.git.onDidOpenRepository === 'function') {
        const openRepoDisposable = this.git.onDidOpenRepository((repo: any) => {
          void this.attachRepositoryListener(repo).catch((error: any) => {
            console.error('[AuthorshipTracker] Failed attaching newly opened repository:', error);
          });
        });
        this.context.subscriptions.push(openRepoDisposable);
      }

      return this.repositorySubscriptions.size > 0;
    } catch (error) {
      console.error('[AuthorshipTracker] Failed to setup git listener:', error);
      return false;
    }
  }

  /**
   * Attach commit listener to a repository if not already attached.
   */
  private async attachRepositoryListener(repo: any): Promise<void> {
    const repoPath = repo?.rootUri?.fsPath;
    if (!repoPath || this.repositorySubscriptions.has(repoPath)) {
      return;
    }

    // Keep latest commit per currently attached repository.
    this.lastAnalyzedCommit = await this.getCurrentCommitHash(repo);
    console.log(`[AuthorshipTracker] Repository detected: ${repoPath}`);
    console.log(`[AuthorshipTracker] Initial commit: ${this.lastAnalyzedCommit}`);

    const disposable = this.createRepositoryChangeSubscription(repo, repoPath);

    this.repositorySubscriptions.set(repoPath, disposable);
    this.context.subscriptions.push(disposable);
  }

  /**
   * Subscribe to repository changes across multiple Git API versions.
   */
  private createRepositoryChangeSubscription(repo: any, repoPath: string): vscode.Disposable {
    const disposables: vscode.Disposable[] = [];

    const onRepoChange = () => {
      void this.checkForNewCommit(repo).catch((error: any) => {
        console.error(`[AuthorshipTracker] Error while processing repository change for ${repoPath}:`, error);
      });
    };

    if (typeof repo.onDidChangeRepository === 'function') {
      disposables.push(repo.onDidChangeRepository(onRepoChange));
    } else if (repo?.state && typeof repo.state.onDidChange === 'function') {
      disposables.push(repo.state.onDidChange(onRepoChange));
    } else {
      console.log(`[AuthorshipTracker] Repository change event unavailable for ${repoPath}; using polling fallback`);
    }

    // Polling ensures commit detection even when Git API events are inconsistent.
    const interval = setInterval(onRepoChange, 3000);
    disposables.push(new vscode.Disposable(() => clearInterval(interval)));

    return new vscode.Disposable(() => {
      for (const disposable of disposables) {
        disposable.dispose();
      }
    });
  }

  /**
   * Check if there's a new commit and analyze it
   */
  private async checkForNewCommit(repo: any): Promise<void> {
    try {
      const currentCommitHash = await this.getCurrentCommitHash(repo);

      if (!currentCommitHash) {
        return;
      }

      // Only analyze if commit has changed
      if (currentCommitHash === this.lastAnalyzedCommit) {
        return;
      }

      console.log(`\n[AuthorshipTracker] New commit detected: ${currentCommitHash}`);
      this.lastAnalyzedCommit = currentCommitHash;

      // Analyze the new commit
      await this.analyzeCommitChanges(repo, currentCommitHash);
    } catch (error) {
      console.error('[AuthorshipTracker] Error checking for new commit:', error);
    }
  }

  /**
   * Analyze changes in a commit
   * Gets all files from the commit and analyzes them
   */
  private async analyzeCommitChanges(repo: any, commitHash: string): Promise<void> {
    try {
      console.log(`[AuthorshipTracker] Analyzing commit: ${commitHash}`);

      const conversationHistory = this.getConversationHistoryForAnalysis();
      if (conversationHistory.length === 0) {
        console.log('[AuthorshipTracker] No conversation history found - continuing with HUMAN_WRITTEN defaults');
      }

      // Get files changed in this commit
      const committedFiles = await this.getCommitFiles(repo, commitHash);
      if (!committedFiles || committedFiles.length === 0) {
        console.log('[AuthorshipTracker] No files changed in commit');
        return;
      }

      console.log(`[AuthorshipTracker] Files in commit: ${committedFiles.length}`);

      // ========== ATTRIBUTION PIPELINE ==========

      // 1. Parse full conversation history
      console.log('[AuthorshipTracker] Step 1: Parsing conversation history...');
      const parsedConversation = this.conversationParser.parseConversation(conversationHistory);
      console.log(`  ✓ Found ${parsedConversation.totalMessages} messages with ${parsedConversation.codeSnippets.length} code snippets`);

      // 2. Analyze each file in the commit
      console.log('[AuthorshipTracker] Step 2: Analyzing commit diff...');
      const allAttributions: any[] = [];
      const allCodeBlocks: any[] = [];

      for (const fileInfo of committedFiles) {
        const { filePath, status } = fileInfo;

        // Skip deleted files
        if (status === 'D') {
          console.log(`  - Skipping deleted file: ${filePath}`);
          continue;
        }

        try {
          // Get current version
          const currentContent = await this.getFileFromCommit(repo, commitHash, filePath);
          const previousContent = await this.getFileFromCommit(repo, `${commitHash}^`, filePath);

          if (!currentContent) {
            console.log(`  - Skipping ${filePath} (not accessible)`);
            continue;
          }

          // Calculate diff
          const diffContent = this.calculateDiff(previousContent || '', currentContent);
          const hunks = this.diffParser.parseDiff(diffContent, filePath);
          const virtualDocument = await vscode.workspace.openTextDocument({ content: currentContent });
          const changedCodeBlocks = this.diffParser.extractCodeBlocks(hunks, virtualDocument, filePath);
          const allFunctionBlocks = this.diffParser.extractAllFunctionBlocks(virtualDocument, filePath);

          const codeBlocks = allFunctionBlocks.length > 0 ? allFunctionBlocks : changedCodeBlocks;

          if (codeBlocks.length > 0) {
            if (allFunctionBlocks.length > 0) {
              console.log(`  ✓ ${filePath}: ${codeBlocks.length} function block(s) (full-file coverage)`);
            } else {
              console.log(`  ✓ ${filePath}: ${codeBlocks.length} code block(s) (diff coverage)`);
            }
            allCodeBlocks.push(...codeBlocks);
          }
        } catch (error) {
          console.log(`  - Error analyzing ${filePath}:`, error);
        }
      }

      if (allCodeBlocks.length === 0) {
        console.log('[AuthorshipTracker] No code blocks to analyze');
        return;
      }

      // 3. Attribute authorship
      console.log('[AuthorshipTracker] Step 3: Attributing code authorship...');
      const attributions = this.attributor.attributeCodeBlocks(allCodeBlocks, parsedConversation);

      const summary = this.attributor.getSummary(attributions);
      console.log(`  ✓ Results: ${summary.llmGenerated} LLM, ${summary.humanPrompt} Human-Prompted, ${summary.humanWritten} Human-Written`);
      console.log(`  ✓ Average Confidence: ${(summary.averageConfidence * 100).toFixed(1)}%`);

      if (this.config.insertTags) {
        console.log('[AuthorshipTracker] Step 4: Inserting authorship tags into files...');
        await this.insertAuthorshipTags(repo, attributions);
      }

      // 4. Log results
      if (this.config.createLogs) {
        console.log('[AuthorshipTracker] Step 5: Creating structured logs...');
        const analysisResult: CommitAnalysisResult = {
          commitHash: commitHash,
          timestamp: new Date(),
          filesAnalyzed: committedFiles.map(f => f.filePath),
          codeBlocksFound: allCodeBlocks,
          attributions,
          tags: this.tagInserter.generateTags(attributions),
          statistics: {
            totalNewLines: allCodeBlocks.reduce((sum, b) => sum + (b.endLine - b.startLine), 0),
            analyzedLines: allCodeBlocks.length,
            llmGeneratedLines: summary.llmGenerated,
            humanPromptLines: summary.humanPrompt,
            humanWrittenLines: summary.humanWritten,
            mixedLines: summary.mixed,
            uncertainLines: summary.uncertain
          }
        };

        const logPath = this.logger.logAnalysis(analysisResult);
        console.log(`  ✓ Log created at: ${logPath}`);
      }

      // 5. Show summary
      vscode.window.showInformationMessage(
        `Authorship Analysis: ${summary.llmGenerated} LLM, ${summary.humanPrompt} Human-Prompted, ${summary.humanWritten} Human-Written`,
        'View Logs'
      ).then(selection => {
        if (selection === 'View Logs') {
          const logDir = this.getLogsDirectory();
          vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(logDir));
        }
      });

      console.log('[AuthorshipTracker] Commit analysis complete! ✓');

    } catch (error) {
      console.error('[AuthorshipTracker] Error during commit analysis:', error);
      vscode.window.showErrorMessage(`Authorship tracking failed: ${error}`);
    }
  }

  /**
   * Get conversation history from active chatbot panel when available,
   * otherwise fall back to persisted extension state.
   */
  private getConversationHistoryForAnalysis(): any[] {
    const chatbot = ChatbotPanel.getCurrentPanel();
    if (chatbot) {
      return chatbot.getConversationHistory();
    }

    const stored = this.context.globalState.get<any[]>('conversationHistory');
    if (stored && stored.length > 0) {
      return stored;
    }

    return [];
  }

  /**
   * Insert generated authorship tags into changed files.
   */
  private async insertAuthorshipTags(repo: any, attributions: any[]): Promise<void> {
    const tags = this.tagInserter.generateTags(attributions);

    if (tags.length === 0) {
      console.log('  - No tags generated');
      return;
    }

    const tagsByFile = new Map<string, typeof tags>();
    for (const tag of tags) {
      const fileTags = tagsByFile.get(tag.codeBlock.filePath) || [];
      fileTags.push(tag);
      tagsByFile.set(tag.codeBlock.filePath, fileTags);
    }

    for (const [relativeFilePath, fileTags] of tagsByFile.entries()) {
      try {
        const fileUri = vscode.Uri.joinPath(repo.rootUri, relativeFilePath);
        const document = await vscode.workspace.openTextDocument(fileUri);

        await this.tagInserter.insertTagsIntoDocument(document, fileTags);
        await document.save();

        console.log(`  ✓ Inserted ${fileTags.length} tag(s) into ${relativeFilePath}`);
      } catch (error) {
        console.error(`  - Failed to insert tags into ${relativeFilePath}:`, error);
      }
    }
  }

  /**
   * Get all files changed in a commit
   */
  private async getCommitFiles(repo: any, commitHash: string): Promise<Array<{filePath: string, status: string}> | null> {
    const repoPath = repo?.rootUri?.fsPath;

    // Prefer git CLI for consistency across VS Code Git API versions.
    if (repoPath) {
      const output = await this.runGitCommand(repoPath, ['show', '--pretty=format:', '--name-status', commitHash]);
      if (output !== null) {
        const files: Array<{filePath: string, status: string}> = [];
        const lines = output.split('\n').map(line => line.trim()).filter(Boolean);

        for (const line of lines) {
          const parts = line.split(/\s+/);
          if (parts.length < 2) {
            continue;
          }

          const status = parts[0].toUpperCase();
          const filePath = parts.slice(1).join(' ');
          files.push({ filePath, status: status[0] });
        }

        return files.length > 0 ? files : null;
      }
    }

    try {
      // Get commit diff to see what files changed
      const diff = await repo.show(commitHash);
      const files: Array<{filePath: string, status: string}> = [];

      const lines = diff.split('\n');
      for (const line of lines) {
        if (line.startsWith('diff --git')) {
          // Extract file path from: diff --git a/path/to/file b/path/to/file
          const match = line.match(/^diff --git a\/(.*) b\/(.*)/);
          if (match) {
            const filePath = match[2];
            files.push({ filePath, status: 'M' }); // Assume modified
          }
        } else if (line.startsWith('new file mode')) {
          // File was added
          const lastFile = files[files.length - 1];
          if (lastFile) lastFile.status = 'A';
        } else if (line.startsWith('deleted file mode')) {
          // File was deleted
          const lastFile = files[files.length - 1];
          if (lastFile) lastFile.status = 'D';
        }
      }

      return files.length > 0 ? files : null;
    } catch (error) {
      console.log('[AuthorshipTracker] Could not get commit files:', error);
      return null;
    }
  }

  /**
   * Get file content from a specific commit
   */
  private async getFileFromCommit(repo: any, commitHash: string, filePath: string): Promise<string | null> {
    const repoPath = repo?.rootUri?.fsPath;
    if (repoPath) {
      const content = await this.runGitCommand(repoPath, ['show', `${commitHash}:${filePath}`]);
      if (content !== null) {
        return content;
      }

      // Parent commit may not exist on initial commit.
      if (commitHash.endsWith('^')) {
        const baseCommit = commitHash.slice(0, -1);
        const parentCheck = await this.runGitCommand(repoPath, ['rev-parse', '--verify', `${baseCommit}^`]);
        if (parentCheck === null) {
          return '';
        }
      }
    }

    try {
      const content = await repo.show(`${commitHash}:${filePath}`);
      return content;
    } catch (error) {
      return null;
    }
  }

  /**
   * Calculate diff between old and new content
   */
  private calculateDiff(oldContent: string, newContent: string): string {
    const oldLines = oldContent.split('\n');
    const newLines = newContent.split('\n');
    let diff = '';
    let oldLineNum = 1;
    let newLineNum = 1;

    diff += `--- a/file\n`;
    diff += `+++ b/file\n`;
    diff += `@@ -1,${oldLines.length} +1,${newLines.length} @@\n`;

    // Simple diff generation (unified format)
    const maxLines = Math.max(oldLines.length, newLines.length);
    for (let i = 0; i < maxLines; i++) {
      const oldLine = oldLines[i] || '';
      const newLine = newLines[i] || '';

      if (oldLine === newLine) {
        diff += ` ${oldLine}\n`;
        oldLineNum++;
        newLineNum++;
      } else {
        if (oldLine) {
          diff += `-${oldLine}\n`;
          oldLineNum++;
        }
        if (newLine) {
          diff += `+${newLine}\n`;
          newLineNum++;
        }
      }
    }

    return diff;
  }

  /**
   * Get git extension
   */
  private async getGitExtension(): Promise<any> {
    const gitExtension = vscode.extensions.getExtension('vscode.git');
    if (!gitExtension) {
      vscode.window.showErrorMessage('Git extension not found');
      return null;
    }

    const gitApi = gitExtension.isActive
      ? gitExtension.exports
      : await gitExtension.activate();

    return gitApi.getAPI(1);
  }

  /**
   * Get the current git repository
   */
  private getGitRepository(git: any): any {
    if (git.repositories.length === 0) {
      vscode.window.showErrorMessage('No git repository found');
      return null;
    }
    return git.repositories[0];
  }

  /**
   * Get the current commit hash
   */
  private async getCurrentCommitHash(repo: any): Promise<string | null> {
    const headCommit = repo?.state?.HEAD?.commit;
    if (headCommit) {
      return headCommit;
    }

    try {
      const commit = await repo.getCommit('HEAD');
      return commit.hash;
    } catch (error) {
      const repoPath = repo?.rootUri?.fsPath;
      if (repoPath) {
        const hash = await this.runGitCommand(repoPath, ['rev-parse', 'HEAD']);
        return hash ? hash.trim() : null;
      }
      return null;
    }
  }

  /**
   * Run a git command in repository root. Returns null when command fails.
   */
  private async runGitCommand(repoPath: string, args: string[]): Promise<string | null> {
    try {
      const { stdout } = await this.execFileAsync('git', args, { cwd: repoPath });
      return stdout;
    } catch {
      return null;
    }
  }

  /**
   * Get logs directory
   */
  public getLogsDirectory(): string {
    return this.logger.getLogDir();
  }

  /**
   * Update configuration
   */
  public setConfig(config: Partial<AuthorshipConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get configuration
   */
  public getConfig(): AuthorshipConfig {
    return this.config;
  }
}
