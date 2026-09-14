import * as vscode from 'vscode';
import OpenAI from 'openai';
import { loadEnvFile } from './envLoader';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

export class ChatbotPanel {
    private static currentPanel: ChatbotPanel | undefined;
    private readonly panel: vscode.WebviewPanel;
    private conversationHistory: Message[] = [];
    private disposables: vscode.Disposable[] = [];
    private openai!: OpenAI;
    private selectedModel: string = "openai/gpt-oss-20b:free";
    private includeFileContext: boolean = true;
    private lastActiveEditor: vscode.TextEditor | undefined; // Add this to cache the last editor

    private constructor(panel: vscode.WebviewPanel, private readonly context: vscode.ExtensionContext) {
        this.panel = panel;
        
        // Cache the current active editor before opening the panel
        this.lastActiveEditor = vscode.window.activeTextEditor;
        
        // Initialize OpenAI immediately
        loadEnvFile(this.context.extensionPath);
        const apiKey = process.env.OPENROUTER_API_KEY;
        if (!apiKey) {
            vscode.window.showErrorMessage(
                'OPENROUTER_API_KEY is not set. Add it to a .env file in the extension root.'
            );
        }
        this.openai = new OpenAI({
            baseURL: "https://openrouter.ai/api/v1",
            apiKey: apiKey ?? "",
        });
        
        // Load conversation history from storage FIRST
        this.loadConversationHistory().then(() => {
            // THEN set HTML
            this.panel.webview.html = this.getHtmlContent();
            
            this.panel.webview.onDidReceiveMessage(
                message => this.handleMessage(message),
                null,
                this.disposables
            );

            this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
            
            // Restore conversation AFTER webview is ready
            setTimeout(() => {
                this.restoreConversationInUI();
                // Send file context after webview is ready
                this.sendFileContextToWebview();
            }, 100);
        });
        
        // Listen for active editor changes
        vscode.window.onDidChangeActiveTextEditor((editor) => {
            if (editor) {
                this.lastActiveEditor = editor;
                this.sendFileContextToWebview();
            }
        }, null, this.disposables);
    }

    public static createOrShow(context: vscode.ExtensionContext) {
        if (ChatbotPanel.currentPanel) {
            ChatbotPanel.currentPanel.panel.reveal(vscode.ViewColumn.Beside);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'aiChatbot',
            'AI Assistant',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true
            }
        );

        ChatbotPanel.currentPanel = new ChatbotPanel(panel, context);
    }

    public static togglePanel(context: vscode.ExtensionContext) {
        if (ChatbotPanel.currentPanel) {
            // If panel exists, close it
            ChatbotPanel.currentPanel.dispose();
            ChatbotPanel.currentPanel = undefined;
        } else {
            // If panel doesn't exist, create it
            ChatbotPanel.createOrShow(context);
        }
    }

    private async loadConversationHistory(): Promise<void> {
        const saved = this.context.globalState.get<Message[]>('conversationHistory');
        if (saved) {
            // Convert timestamp strings back to Date objects
            this.conversationHistory = saved.map(msg => ({
                ...msg,
                timestamp: new Date(msg.timestamp)
            }));
        }
    }

    private async saveConversationHistory() {
        await this.context.globalState.update('conversationHistory', this.conversationHistory);
    }

    private restoreConversationInUI() {
        // Send all previous messages to the webview
        this.conversationHistory.forEach(msg => {
            if (msg.role === 'user') {
                this.panel.webview.postMessage({
                    type: 'userMessage',
                    content: msg.content
                });
            } else {
                this.panel.webview.postMessage({
                    type: 'assistantMessage',
                    content: msg.content
                });
            }
        });
    }

    private async handleMessage(message: any) {
        switch (message.type) {
            case 'sendMessage':
                await this.handleUserMessage(message.text, message.includeFile);
                break;
            case 'insertCode':
                await this.insertCodeToEditor(message.code);
                break;
            case 'modelChanged':
                this.selectedModel = message.model;
                console.log('Model changed to:', this.selectedModel);
                vscode.window.showInformationMessage(`Model switched to: ${this.selectedModel}`);
                break;
            case 'requestFileContext':
                this.sendFileContextToWebview();
                break;
            case 'removeFileContext':
                this.includeFileContext = false;
                // Update the webview to show no context
                this.panel.webview.postMessage({
                    type: 'fileContext',
                    fileInfo: null
                });
                break;
            case 'clearChat':
                console.log('Before clear:', this.conversationHistory.length, 'messages');
                this.conversationHistory = [];
                await this.saveConversationHistory();
                console.log('After clear:', this.conversationHistory.length, 'messages');
                const stored = this.context.globalState.get<Message[]>('conversationHistory');
                console.log('Storage now contains:', stored?.length || 0, 'messages');
                this.panel.webview.postMessage({
                    type: 'clearMessages'
                });
                vscode.window.showInformationMessage('Chat history cleared');
                break;
        }
    }

    private sendFileContextToWebview() {
        // Use lastActiveEditor instead of activeTextEditor
        const editor = this.lastActiveEditor;
        if (editor) {
            const document = editor.document;
            const fileName = document.fileName.split('/').pop() || 'Unknown';
            this.panel.webview.postMessage({
                type: 'fileContext',
                fileInfo: {
                    fileName: fileName,
                    fullPath: document.fileName,
                    language: document.languageId
                }
            });
            this.includeFileContext = true;
        } else {
            this.panel.webview.postMessage({
                type: 'fileContext',
                fileInfo: null
            });
            this.includeFileContext = false;
        }
    }

    private async handleUserMessage(userMessage: string, includeFile: boolean = true) {
        // Use lastActiveEditor instead of activeTextEditor
        const editor = this.lastActiveEditor;
        let contextMessage = userMessage;
        
        if (includeFile && this.includeFileContext && editor) {
            const document = editor.document;
            const fileName = document.fileName;
            const fileContent = document.getText();
            const selection = editor.selection;
            const selectedText = document.getText(selection);
            const language = document.languageId;
            
            let context = `[File Context]\n`;
            context += `File: ${fileName}\n`;
            context += `Language: ${language}\n`;
            
            if (selectedText && !selection.isEmpty) {
                context += `Selected Code:\n\`\`\`${language}\n${selectedText}\n\`\`\`\n`;
            } else if (fileContent.length < 4000) {
                context += `File Content:\n\`\`\`${language}\n${fileContent}\n\`\`\`\n`;
            } else {
                const truncatedContent = fileContent.substring(0, 4000);
                context += `File Content (truncated to first 4000 chars):\n\`\`\`${language}\n${truncatedContent}\n\`\`\`\n`;
            }
            
            context += `\nUser Question: ${userMessage}`;
            contextMessage = context;
        }
        
        this.conversationHistory.push({
            role: 'user',
            content: contextMessage,
            timestamp: new Date()
        });
        
        await this.saveConversationHistory();

        this.panel.webview.postMessage({
            type: 'userMessage',
            content: userMessage
        });

        this.panel.webview.postMessage({
            type: 'typing',
            isTyping: true
        });

        const response = await this.getLLMResponse(contextMessage);
        
        this.panel.webview.postMessage({
            type: 'typing',
            isTyping: false
        });
        
        this.conversationHistory.push({
            role: 'assistant',
            content: response,
            timestamp: new Date()
        });
        
        await this.saveConversationHistory();

        const codeBlocks = this.extractCodeBlocks(response);

        this.panel.webview.postMessage({
            type: 'assistantMessage',
            content: response,
            codeBlocks: codeBlocks
        });
    }

    private async getLLMResponse(prompt: string): Promise<string> {
        // Limit context size to reduce provider throttling risk.
        const recentMessages = this.conversationHistory.slice(-3);
        const messages = recentMessages.map(msg => ({
            role: msg.role,
            content: msg.content
        }));

        // The user message has already been pushed to conversationHistory by handleUserMessage.
        // Do not push prompt again here, or the request duplicates tokens.

        const fallbackModels = this.getFallbackModels(this.selectedModel);
        const maxAttempts = 3;
        let delayMs = 1000;
        let modelForAttempt = this.selectedModel;

        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                const completion = await this.openai.chat.completions.create({
                    model: modelForAttempt,
                    messages
                });

                if (modelForAttempt !== this.selectedModel) {
                    this.selectedModel = modelForAttempt;
                    this.panel.webview.postMessage({
                        type: 'modelAutoSwitched',
                        model: modelForAttempt
                    });
                }

                return completion.choices[0].message.content || "No response received";
            } catch (error: any) {
                const status = error?.status ?? error?.code;
                const isRateLimit = status === 429;
                const isLastAttempt = attempt === maxAttempts;

                if (isRateLimit && !isLastAttempt) {
                    const fallbackModel = fallbackModels.shift();
                    if (fallbackModel) {
                        modelForAttempt = fallbackModel;
                    }
                    await new Promise(resolve => setTimeout(resolve, delayMs));
                    delayMs *= 2;
                    continue;
                }

                console.error('LLM Error:', error);

                if (isRateLimit) {
                    return 'Error: Provider rate limit hit (429). Please wait ~30-60 seconds, disable file context for large files, or switch model/provider and try again.';
                }

                return `Error: Unable to get response from LLM. ${error}`;
            }
        }

        return 'Error: Unable to get response from LLM after retries.';
    }

    private getFallbackModels(primaryModel: string): string[] {
        const candidates = [
            'openai/gpt-oss-20b:free',
            'meta-llama/llama-3.2-3b-instruct:free',
            'mistralai/mistral-7b-instruct:free'
        ];

        return candidates.filter(model => model !== primaryModel);
    }

    private extractCodeBlocks(text: string): Array<{language: string, code: string}> {
        const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
        const codeBlocks: Array<{language: string, code: string}> = [];
        let match;

        while ((match = codeBlockRegex.exec(text)) !== null) {
            codeBlocks.push({
                language: match[1] || 'plaintext',
                code: match[2].trim()
            });
        }

        return codeBlocks;
    }

    private async insertCodeToEditor(code: string) {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            editor.edit(editBuilder => {
                editBuilder.insert(editor.selection.active, code);
            });
        }
    }

    public getConversationHistory(): Message[] {
        return this.conversationHistory;
    }

    public getSelectedModel(): string {
        return this.selectedModel || 'openai/gpt-oss-20b:free';
    }

    private getHtmlContent(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Assistant</title>
    <style>
        * {
            box-sizing: border-box;
        }
        
        body {
            padding: 0;
            margin: 0;
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
        }
        
        #chat-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        
        #model-selector {
            padding: 12px 16px;
            background-color: var(--vscode-sideBar-background);
            border-bottom: 1px solid var(--vscode-panel-border);
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        #model-selector label {
            font-size: 12px;
            font-weight: 600;
            opacity: 0.8;
        }
        
        #model-select {
            flex: 1;
            padding: 6px 12px;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
        }
        
        #model-select:focus {
            outline: none;
            border-color: var(--vscode-focusBorder);
        }
        
        #context-section {
            padding: 12px 16px;
            background-color: var(--vscode-sideBar-background);
            border-bottom: 1px solid var(--vscode-panel-border);
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        
        #context-label {
            font-size: 11px;
            font-weight: 600;
            opacity: 0.7;
            text-transform: uppercase;
        }
        
        .context-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 8px;
            background-color: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        
        .context-tag-icon {
            font-size: 10px;
            opacity: 0.8;
        }
        
        .context-tag-name {
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .context-tag-remove {
            cursor: pointer;
            padding: 2px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.2s ease;
            opacity: 0.7;
        }
        
        .context-tag-remove:hover {
            background-color: rgba(255, 255, 255, 0.1);
            opacity: 1;
        }
        
        #no-context {
            font-size: 11px;
            opacity: 0.5;
            font-style: italic;
        }
        
        #messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        
        .message {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 8px;
            line-height: 1.6;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }
        
        .assistant-message {
            background-color: var(--vscode-input-background);
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            border-left: 3px solid var(--vscode-focusBorder);
        }
        
        .message-header {
            font-size: 11px;
            opacity: 0.7;
            margin-bottom: 8px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .typing-indicator {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 8px;
            background-color: var(--vscode-input-background);
            align-self: flex-start;
            border-bottom-left-radius: 4px;
            border-left: 3px solid var(--vscode-focusBorder);
            animation: fadeIn 0.3s ease-in;
        }
        
        .typing-dots {
            display: flex;
            gap: 6px;
            align-items: center;
        }
        
        .typing-dots span {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--vscode-foreground);
            opacity: 0.4;
            animation: typing 1.4s infinite;
        }
        
        .typing-dots span:nth-child(1) {
            animation-delay: 0s;
        }
        
        .typing-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 60%, 100% {
                opacity: 0.4;
                transform: scale(1);
            }
            30% {
                opacity: 1;
                transform: scale(1.2);
            }
        }
        
        .code-block {
            background-color: var(--vscode-textCodeBlock-background);
            padding: 16px;
            margin: 12px 0;
            border-radius: 6px;
            position: relative;
            border: 1px solid var(--vscode-panel-border);
        }
        
        .code-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        
        .code-language {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            opacity: 0.7;
            letter-spacing: 0.5px;
        }
        
        .code-block pre {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', Consolas, monospace;
            font-size: 13px;
            line-height: 1.5;
        }
        
        .insert-button {
            padding: 6px 12px;
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 600;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .insert-button:hover {
            background-color: var(--vscode-button-hoverBackground);
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .insert-button:active {
            transform: translateY(0);
        }
        
        #input-section {
            padding: 16px;
            background-color: var(--vscode-sideBar-background);
            border-top: 1px solid var(--vscode-panel-border);
        }
        
        #input-wrapper {
            position: relative;
            display: flex;
            align-items: flex-end;
        }
        
        #user-input {
            flex: 1;
            padding: 10px 50px 10px 12px;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            border-radius: 6px;
            resize: none;
            font-family: var(--vscode-font-family);
            font-size: 13px;
            line-height: 1.5;
            transition: border-color 0.2s ease;
            max-height: 200px;
            min-height: 38px;
        }
        
        #user-input:focus {
            outline: none;
            border-color: var(--vscode-focusBorder);
        }
        
        #send-button {
            position: absolute;
            right: 8px;
            bottom: 8px;
            padding: 6px;
            background-color: transparent;
            color: var(--vscode-foreground);
            border: none;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.2s ease;
            width: 28px;
            height: 28px;
        }
        
        #send-button:hover:not(:disabled) {
            background-color: var(--vscode-toolbar-hoverBackground);
        }
        
        #send-button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }
        
        #send-button svg {
            width: 16px;
            height: 16px;
            fill: currentColor;
        }
        
        #button-row {
            display: flex;
            gap: 8px;
            margin-top: 8px;
            justify-content: flex-end;
        }
        
        .action-button {
            padding: 4px 8px;
            background-color: transparent;
            color: var(--vscode-foreground);
            border: 1px solid var(--vscode-button-border);
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: all 0.2s ease;
            opacity: 0.8;
        }
        
        .action-button:hover {
            background-color: var(--vscode-toolbar-hoverBackground);
            opacity: 1;
        }
        
        strong {
            font-weight: 700;
            color: var(--vscode-textLink-foreground);
        }
        
        em {
            font-style: italic;
            opacity: 0.9;
        }
        
        code {
            background-color: var(--vscode-textCodeBlock-background);
            padding: 3px 6px;
            border-radius: 3px;
            font-family: 'Courier New', Consolas, monospace;
            font-size: 0.9em;
            border: 1px solid var(--vscode-panel-border);
        }
        
        h3 {
            margin: 16px 0 8px 0;
            font-size: 16px;
            font-weight: 600;
            color: var(--vscode-textLink-foreground);
        }
        
        ul, ol {
            margin: 8px 0;
            padding-left: 24px;
        }
        
        li {
            margin: 4px 0;
            line-height: 1.6;
        }
        
        p {
            margin: 8px 0;
        }
        
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--vscode-editor-background);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--vscode-scrollbarSlider-background);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--vscode-scrollbarSlider-hoverBackground);
        }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="model-selector">
            <label for="model-select">Model:</label>
            <select id="model-select">
                <option value="openai/gpt-oss-20b:free" selected>GPT OSS 20B (Free)</option>
                <option value="meta-llama/llama-3.2-3b-instruct:free">Llama 3.2 3B (Free)</option>
                <option value="mistralai/mistral-7b-instruct:free">Mistral 7B (Free)</option>
            </select>
        </div>
        <div id="context-section">
            <span id="context-label">Context:</span>
            <div id="context-tags"></div>
            <span id="no-context">No file open</span>
        </div>
        <div id="messages"></div>
        <div id="input-section">
            <div id="input-wrapper">
                <textarea id="user-input" placeholder="Ask me anything..." rows="1"></textarea>
                <button id="send-button" title="Send message">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">
                        <path d="M2 2l12 6-12 6V9l8-3-8-3V2z"/>
                    </svg>
                </button>
            </div>
            <div id="button-row">
                <button id="clear-button" class="action-button">Clear Chat</button>
            </div>
        </div>
    </div>
    <script>
        const vscode = acquireVsCodeApi();
        const messagesDiv = document.getElementById('messages');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');
        const clearButton = document.getElementById('clear-button');
        const modelSelect = document.getElementById('model-select');
        const contextTags = document.getElementById('context-tags');
        const noContext = document.getElementById('no-context');
        let typingIndicatorElement = null;
        let currentFile = null;

        // Load saved model selection
        const savedState = vscode.getState();
        if (savedState && savedState.selectedModel) {
            modelSelect.value = savedState.selectedModel;
        }

        // Sync backend model with current UI model on load.
        vscode.postMessage({
            type: 'modelChanged',
            model: modelSelect.value
        });

        // Request current file context on load
        vscode.postMessage({ type: 'requestFileContext' });

        // Save model selection when changed
        modelSelect.addEventListener('change', () => {
            vscode.setState({ selectedModel: modelSelect.value });
            vscode.postMessage({
                type: 'modelChanged',
                model: modelSelect.value
            });
        });

        function updateContextDisplay(fileInfo) {
            if (fileInfo) {
                currentFile = fileInfo;
                noContext.style.display = 'none';
                
                const tag = document.createElement('div');
                tag.className = 'context-tag';
                tag.innerHTML = 
                    '<span class="context-tag-icon">📄</span>' +
                    '<span class="context-tag-name" title="' + fileInfo.fullPath + '">' + fileInfo.fileName + '</span>' +
                    '<span class="context-tag-remove" title="Remove from context">✕</span>';
                
                tag.querySelector('.context-tag-remove').addEventListener('click', () => {
                    currentFile = null;
                    contextTags.innerHTML = '';
                    noContext.style.display = 'inline';
                    vscode.postMessage({ type: 'removeFileContext' });
                });
                
                contextTags.innerHTML = '';
                contextTags.appendChild(tag);
            } else {
                currentFile = null;
                contextTags.innerHTML = '';
                noContext.style.display = 'inline';
            }
        }

        // Auto-resize textarea
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });

        function formatMarkdown(text) {
            text = text.replace(/^### (.+)$/gm, '<h3>$1</h3>');
            text = text.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
            text = text.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');
            text = text.replace(/\`([^\`]+)\`/g, '<code>$1</code>');
            text = text.replace(/\\n\\n/g, '</p><p>');
            text = text.replace(/\\n/g, '<br>');
            text = '<p>' + text + '</p>';
            return text;
        }

        function showTypingIndicator(show) {
            if (show) {
                if (typingIndicatorElement) {
                    typingIndicatorElement.remove();
                }
                
                typingIndicatorElement = document.createElement('div');
                typingIndicatorElement.className = 'typing-indicator';
                typingIndicatorElement.innerHTML = 
                    '<div class="message-header">Working...</div>' +
                    '<div class="typing-dots">' +
                    '<span></span>' +
                    '<span></span>' +
                    '<span></span>' +
                    '</div>';
                
                messagesDiv.appendChild(typingIndicatorElement);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            } else {
                if (typingIndicatorElement) {
                    typingIndicatorElement.remove();
                    typingIndicatorElement = null;
                }
            }
        }

        function addMessage(content, isUser) {
            const messageDiv = document.createElement('div');
            messageDiv.className = isUser ? 'message user-message' : 'message assistant-message';
            
            if (!isUser) {
                const header = document.createElement('div');
                header.className = 'message-header';
                header.textContent = 'AI Assistant';
                messageDiv.appendChild(header);
            }
            
            let processedContent = content;
            const codeBlocks = [];
            let codeBlockIndex = 0;
            
            processedContent = processedContent.replace(/\`\`\`(\\w+)?\\n([\\s\\S]*?)\`\`\`/g, (match, lang, code) => {
                const language = lang || 'code';
                const placeholder = '___CODE_BLOCK_' + codeBlockIndex + '___';
                codeBlocks.push({
                    placeholder: placeholder,
                    language: language,
                    code: code.trim()
                });
                codeBlockIndex++;
                return placeholder;
            });
            
            processedContent = formatMarkdown(processedContent);
            
            codeBlocks.forEach(block => {
                const codeHtml = '<div class="code-block">' +
                    '<div class="code-header">' +
                    '<span class="code-language">' + block.language + '</span>' +
                    '<button class="insert-button" data-code-index="' + codeBlocks.indexOf(block) + '">Insert Code</button>' +
                    '</div>' +
                    '<pre><code>' + escapeHtml(block.code) + '</code></pre>' +
                    '</div>';
                processedContent = processedContent.replace(block.placeholder, codeHtml);
            });
            
            const contentDiv = document.createElement('div');
            contentDiv.innerHTML = processedContent;
            messageDiv.appendChild(contentDiv);
            
            messagesDiv.appendChild(messageDiv);
            
            messageDiv.querySelectorAll('.insert-button').forEach(button => {
                button.addEventListener('click', function() {
                    const index = parseInt(this.getAttribute('data-code-index'));
                    insertCode(codeBlocks[index].code);
                    this.textContent = '✓ Inserted';
                    this.style.backgroundColor = 'var(--vscode-testing-iconPassed)';
                    setTimeout(() => {
                        this.textContent = 'Insert Code';
                        this.style.backgroundColor = '';
                    }, 2000);
                });
            });
            
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function insertCode(code) {
            vscode.postMessage({
                type: 'insertCode',
                code: code
            });
        }

        function sendMessage() {
            const text = userInput.value.trim();
            if (text) {
                sendButton.disabled = true;
                vscode.postMessage({
                    type: 'sendMessage',
                    text: text,
                    includeFile: currentFile !== null
                });
                userInput.value = '';
                userInput.style.height = 'auto';
            }
        }

        clearButton.addEventListener('click', () => {
            if (confirm('Are you sure you want to clear the conversation history?')) {
                vscode.postMessage({
                    type: 'clearChat'
                });
            }
        });

        sendButton.addEventListener('click', sendMessage);
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        window.addEventListener('message', event => {
            const message = event.data;
            if (message.type === 'userMessage') {
                addMessage(message.content, true);
            } else if (message.type === 'assistantMessage') {
                addMessage(message.content, false);
                sendButton.disabled = false;
            } else if (message.type === 'typing') {
                showTypingIndicator(message.isTyping);
            } else if (message.type === 'clearMessages') {
                messagesDiv.innerHTML = '';
            } else if (message.type === 'fileContext') {
                updateContextDisplay(message.fileInfo);
            } else if (message.type === 'modelAutoSwitched') {
                modelSelect.value = message.model;
                vscode.setState({ selectedModel: modelSelect.value });
            }
        });
    </script>
</body>
</html>`;
    }

    private dispose() {
        ChatbotPanel.currentPanel = undefined;
        this.panel.dispose();
        while (this.disposables.length) {
            const disposable = this.disposables.pop();
            if (disposable) {
                disposable.dispose();
            }
        }
    }

    public checkAssistantSourcedContent(content: string): {
        isAssistantSourced: boolean;
        matchedMessages: Array<{
            assistantMessage: string;
            similarity: number;
            matchedPortion: string;
        }>;
        overallSimilarity: number;
    } {
        const matches: Array<{
            assistantMessage: string;
            similarity: number;
            matchedPortion: string;
        }> = [];

        // Get all assistant messages from conversation history
        const assistantMessages = this.conversationHistory
            .filter(msg => msg.role === 'assistant')
            .map(msg => msg.content);

        // Check each assistant message for similarity
        assistantMessages.forEach(assistantMsg => {
            // Extract code blocks from assistant message
            const codeBlocks = this.extractCodeBlocks(assistantMsg);
            
            if (codeBlocks.length > 0) {
                // Check against each code block individually
                codeBlocks.forEach(block => {
                    const similarity = this.calculateSimilarity(content, block.code);
                    
                    if (similarity > 0.7) {
                        matches.push({
                            assistantMessage: block.code,
                            similarity: similarity,
                            matchedPortion: block.code
                        });
                    }
                });

                // Also check if the content matches a portion within the code blocks
                const fullCodeBlock = codeBlocks.map(b => b.code).join('\n\n');
                const partialMatches = this.findPartialCodeMatches(content, fullCodeBlock);
                
                partialMatches.forEach(match => {
                    if (match.similarity > 0.7) {
                        matches.push({
                            assistantMessage: match.matchedPortion,
                            similarity: match.similarity,
                            matchedPortion: match.matchedPortion
                        });
                    }
                });
            } else {
                // No code blocks, check text similarity
                const textSimilarity = this.calculateSimilarity(content, assistantMsg);
                if (textSimilarity > 0.7) {
                    const matchedText = this.findBestMatchingSubstring(content, assistantMsg);
                    matches.push({
                        assistantMessage: matchedText,
                        similarity: textSimilarity,
                        matchedPortion: matchedText
                    });
                }
            }
        });

        // Remove duplicate matches and keep only the best
        const uniqueMatches = this.deduplicateMatches(matches);

        // Calculate overall similarity
        const overallSimilarity = uniqueMatches.length > 0
            ? Math.max(...uniqueMatches.map(m => m.similarity))
            : 0;

        return {
            isAssistantSourced: uniqueMatches.length > 0 && overallSimilarity > 0.7,
            matchedMessages: uniqueMatches.sort((a, b) => b.similarity - a.similarity),
            overallSimilarity: overallSimilarity
        };
    }

    private findPartialCodeMatches(content: string, fullCode: string): Array<{
        similarity: number;
        matchedPortion: string;
    }> {
        const matches: Array<{ similarity: number; matchedPortion: string }> = [];
        
        // Split the full code into functions/classes/blocks
        const codeSegments = this.splitCodeIntoSegments(fullCode);
        
        codeSegments.forEach(segment => {
            const similarity = this.calculateSimilarity(content, segment);
            if (similarity > 0.7) {
                matches.push({
                    similarity: similarity,
                    matchedPortion: segment
                });
            }
        });

        return matches;
    }

    private splitCodeIntoSegments(code: string): string[] {
        const segments: string[] = [];
        const lines = code.split('\n');
        
        let currentSegment: string[] = [];
        let inBlock = false;
        let baseIndent = 0;
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const trimmedLine = line.trim();
            const currentIndent = line.length - line.trimStart().length;
            
            // Skip empty lines when not in a block
            if (trimmedLine.length === 0 && !inBlock) {
                continue;
            }
            
            // Detect function/class/block definition
            const isBlockStart = 
                trimmedLine.startsWith('def ') || 
                trimmedLine.startsWith('class ') ||
                trimmedLine.startsWith('function ') ||
                trimmedLine.startsWith('const ') && (trimmedLine.includes('= function') || trimmedLine.includes('=> ')) ||
                trimmedLine.startsWith('let ') && (trimmedLine.includes('= function') || trimmedLine.includes('=> ')) ||
                trimmedLine.startsWith('var ') && (trimmedLine.includes('= function') || trimmedLine.includes('=> ')) ||
                trimmedLine.startsWith('async ') ||
                trimmedLine.startsWith('export function') ||
                trimmedLine.startsWith('export const') && (trimmedLine.includes('= function') || trimmedLine.includes('=> '));
            
            if (isBlockStart) {
                // Save previous segment if exists
                if (currentSegment.length > 0) {
                    segments.push(currentSegment.join('\n').trim());
                }
                
                // Start new segment
                currentSegment = [line];
                inBlock = true;
                baseIndent = currentIndent;
            } else if (inBlock) {
                // Check if this line is at base indent or less and is actual code (not empty/comment)
                const isCodeAtBaseIndent = trimmedLine.length > 0 && 
                                          currentIndent <= baseIndent && 
                                          !trimmedLine.startsWith('#') &&
                                          !trimmedLine.startsWith('//') &&
                                          !trimmedLine.startsWith('"""') &&
                                          !trimmedLine.startsWith("'''");
                
                if (isCodeAtBaseIndent && currentSegment.length > 1) {
                    // This line is at base indent and is code, so previous block has ended
                    segments.push(currentSegment.join('\n').trim());
                    currentSegment = [];
                    inBlock = false;
                    
                    // Don't add this line, it belongs to next segment or standalone
                    if (isBlockStart) {
                        currentSegment = [line];
                        inBlock = true;
                        baseIndent = currentIndent;
                    } else if (trimmedLine.length > 0) {
                        segments.push(line.trim());
                    }
                } else {
                    // Still part of current block
                    currentSegment.push(line);
                }
            } else {
                // Not in a block, treat as standalone segment or start of new block
                if (trimmedLine.length > 0) {
                    segments.push(line.trim());
                }
            }
        }
        
        // Add last segment if exists
        if (currentSegment.length > 0) {
            segments.push(currentSegment.join('\n').trim());
        }
        
        // Filter out very short segments (likely comments or single lines)
        return segments.filter(s => s.trim().length > 10);
    }

    private deduplicateMatches(matches: Array<{
        assistantMessage: string;
        similarity: number;
        matchedPortion: string;
    }>): Array<{
        assistantMessage: string;
        similarity: number;
        matchedPortion: string;
    }> {
        if (matches.length === 0) {
            return matches;
        }

        // Group similar matches
        const groups: Array<Array<typeof matches[0]>> = [];
        
        matches.forEach(match => {
            let addedToGroup = false;
            
            for (const group of groups) {
                const representative = group[0];
                const similarity = this.calculateSimilarity(
                    match.matchedPortion, 
                    representative.matchedPortion
                );
                
                // If they're very similar, group them together
                if (similarity > 0.9) {
                    group.push(match);
                    addedToGroup = true;
                    break;
                }
            }
            
            if (!addedToGroup) {
                groups.push([match]);
            }
        });

        // From each group, keep only the best match (highest similarity)
        return groups.map(group => {
            return group.reduce((best, current) => {
                return current.similarity > best.similarity ? current : best;
            });
        });
    }

    private calculateSimilarity(str1: string, str2: string): number {
        // Normalize strings: remove whitespace, convert to lowercase
        const normalize = (str: string) => str
            .replace(/\s+/g, ' ')
            .trim()
            .toLowerCase();

        const normalized1 = normalize(str1);
        const normalized2 = normalize(str2);

        // If strings are very short, require exact match
        if (normalized1.length < 20 || normalized2.length < 20) {
            return normalized1 === normalized2 ? 1.0 : 0.0;
        }

        // **NEW: Check if str1 is a subset of str2 (file code ⊆ chatbot code)**
        // This handles cases where user removed parts of assistant-sourced code
        if (this.isSubsetMatch(normalized1, normalized2)) {
            return 1.0; // 100% - file code is entirely from chatbot
        }

        if (this.isSubsetMatch(normalized2, normalized1)) {
            return 1.0; // 100% - chatbot code is entirely in file
        }

        // **NEW: Check for structural similarity (ignoring variable names)**
        const structuralSimilarity = this.calculateStructuralSimilarity(str1, str2);
        if (structuralSimilarity > 0.9) {
            return structuralSimilarity;
        }

        // Check for substring containment
        const longerLength = Math.max(normalized1.length, normalized2.length);
        const shorterLength = Math.min(normalized1.length, normalized2.length);

        if (normalized1.includes(normalized2)) {
            return shorterLength / longerLength;
        }
        
        if (normalized2.includes(normalized1)) {
            return shorterLength / longerLength;
        }

        // Calculate Levenshtein distance-based similarity for similar-length strings
        if (Math.abs(normalized1.length - normalized2.length) / longerLength < 0.5) {
            const distance = this.levenshteinDistance(normalized1, normalized2);
            const maxLength = Math.max(normalized1.length, normalized2.length);
            return maxLength === 0 ? 1 : 1 - (distance / maxLength);
        }

        // For very different lengths, check overlap
        return this.calculateOverlapSimilarity(normalized1, normalized2);
    }

    /**
     * Check if str1 is a subset of str2 by comparing all significant lines
     * This handles cases where user removed comments/docstrings from assistant-sourced code
     */
    private isSubsetMatch(str1: string, str2: string): boolean {
        // Extract significant lines (ignore comments, docstrings, empty lines)
        const getSignificantLines = (text: string): string[] => {
            return text
                .split(/\n/)
                .map(line => line.trim())
                .filter(line => {
                    // Keep only actual code lines
                    return line.length > 0 &&
                           !line.startsWith('#') &&
                           !line.startsWith('//') &&
                           !line.startsWith('"""') &&
                           !line.startsWith("'''") &&
                           line !== '"""' &&
                           line !== "'''";
                })
                .map(line => line.replace(/\s+/g, ' ').toLowerCase());
        };

        const lines1 = getSignificantLines(str1);
        const lines2 = getSignificantLines(str2);

        if (lines1.length === 0 || lines2.length === 0) {
            return false;
        }

        // Check if all lines from str1 exist in str2 (in any order)
        let matchedLines = 0;
        for (const line1 of lines1) {
            for (const line2 of lines2) {
                // Allow fuzzy match (handles minor spacing differences)
                if (line2.includes(line1) || line1.includes(line2)) {
                    matchedLines++;
                    break;
                }
            }
        }

        // If 90%+ of lines match, consider it a subset
        return (matchedLines / lines1.length) >= 0.9;
    }

    /**
     * Calculate structural similarity ignoring variable/parameter names
     * This handles cases where user renamed variables
     */
    private calculateStructuralSimilarity(str1: string, str2: string): number {
        // Extract structure by removing identifiers
        const getStructure = (code: string): string => {
            return code
                // Remove string literals
                .replace(/"[^"]*"/g, '""')
                .replace(/'[^']*'/g, "''")
                // Replace identifiers with placeholder (but keep keywords)
                .replace(/\b(?!def|class|if|else|elif|for|while|return|import|from|as|with|try|except|finally|pass|break|continue|function|const|let|var|async|await)\w+\b/g, 'ID')
                // Normalize whitespace
                .replace(/\s+/g, ' ')
                .trim()
                .toLowerCase();
        };

        const struct1 = getStructure(str1);
        const struct2 = getStructure(str2);

        // Check if structures match
        if (struct1 === struct2) {
            return 1.0;
        }

        // Calculate similarity of structures
        if (struct1.includes(struct2) || struct2.includes(struct1)) {
            const shorterLength = Math.min(struct1.length, struct2.length);
            const longerLength = Math.max(struct1.length, struct2.length);
            return shorterLength / longerLength;
        }

        // Use Levenshtein distance on structures
        const distance = this.levenshteinDistance(struct1, struct2);
        const maxLength = Math.max(struct1.length, struct2.length);
        return maxLength === 0 ? 1 : 1 - (distance / maxLength);
    }

    private levenshteinDistance(str1: string, str2: string): number {
        // Optimize for large strings
        if (Math.abs(str1.length - str2.length) > 1000) {
            return Math.max(str1.length, str2.length);
        }

        const matrix: number[][] = [];

        for (let i = 0; i <= str2.length; i++) {
            matrix[i] = [i];
        }

        for (let j = 0; j <= str1.length; j++) {
            matrix[0][j] = j;
        }

        for (let i = 1; i <= str2.length; i++) {
            for (let j = 1; j <= str1.length; j++) {
                if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1, // substitution
                        matrix[i][j - 1] + 1,     // insertion
                        matrix[i - 1][j] + 1      // deletion
                    );
                }
            }
        }

        return matrix[str2.length][str1.length];
    }

    private calculateOverlapSimilarity(str1: string, str2: string): number {
        const words1 = str1.split(/\s+/);
        const words2 = str2.split(/\s+/);
        
        const set1 = new Set(words1);
        const set2 = new Set(words2);
        
        let commonWords = 0;
        set1.forEach(word => {
            if (set2.has(word)) {
                commonWords++;
            }
        });
        
        const totalUniqueWords = Math.max(set1.size, set2.size);
        return commonWords / totalUniqueWords;
    }

    private findBestMatchingSubstring(content: string, text: string): string {
        const normalize = (str: string) => str.replace(/\s+/g, ' ').trim().toLowerCase();
        const normalizedContent = normalize(content);
        const normalizedText = normalize(text);

        // Split text into sentences or paragraphs
        const sentences = text.split(/[.!?\n]+/).filter(s => s.trim().length > 10);
        
        let bestMatch = '';
        let bestSimilarity = 0;

        // Find the sentence/paragraph with highest similarity
        sentences.forEach(sentence => {
            const similarity = this.calculateSimilarity(content, sentence);
            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatch = sentence.trim();
            }
        });

        // If we found a good match, return it with context
        if (bestMatch && bestMatch.length > 0) {
            return bestMatch.length > 300 ? bestMatch.substring(0, 300) + '...' : bestMatch;
        }

        // Fallback: try to find direct substring match
        if (normalizedText.includes(normalizedContent)) {
            const startIndex = normalizedText.indexOf(normalizedContent);
            const originalStartIndex = this.findOriginalIndex(text, normalizedText, startIndex);
            const matchLength = content.length;
            return text.substring(originalStartIndex, originalStartIndex + matchLength + 50) + '...';
        }

        return text.substring(0, 300) + '...';
    }

    private findOriginalIndex(original: string, normalized: string, normalizedIndex: number): number {
        let originalIndex = 0;
        let currentNormalizedIndex = 0;
        
        for (let i = 0; i < original.length && currentNormalizedIndex < normalizedIndex; i++) {
            if (!original[i].match(/\s/)) {
                currentNormalizedIndex++;
            }
            originalIndex++;
        }
        
        return originalIndex;
    }

    public static getCurrentPanel(): ChatbotPanel | undefined {
        return ChatbotPanel.currentPanel;
    }
}