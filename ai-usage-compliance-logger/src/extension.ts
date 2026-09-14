import * as vscode from 'vscode';
import { GitChangeTracker } from './gitChangeTracker';
import { ChatbotPanel } from './chatbotPanel';

export async function activate(context: vscode.ExtensionContext) {
    // Keep git change tracker as-is
    const gitChangeTracker = new GitChangeTracker(context);
    await gitChangeTracker.activate();

    // Create status bar item for chatbot
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    
    statusBarItem.text = "$(comment-discussion) AI Assistant";
    statusBarItem.tooltip = "Open AI Assistant";
    statusBarItem.command = 'ai-usage-compliance-logger.toggleChat';
    statusBarItem.show();
    
    context.subscriptions.push(statusBarItem);

    // Register chatbot toggle command
    const toggleChatCommand = vscode.commands.registerCommand('ai-usage-compliance-logger.toggleChat', () => {
        ChatbotPanel.togglePanel(context);
    });

    context.subscriptions.push(toggleChatCommand);
}

export function deactivate() {}
