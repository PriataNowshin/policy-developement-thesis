# AI Usage Compliance Logger

VS Code extension that tracks whether code came from you or an AI
assistant. It runs a built-in chat panel and watches Git commits; on each
commit it compares the committed code against chat history and labels every
function human-, LLM-, or mixed-authored. Labels get written as a comment
above the function and logged to disk.

## Requirements

VS Code `^1.104.0`, Node.js + npm, Git, and an OpenRouter API key
(free-tier models work).

## Setup

```bash
git clone <this-repository-url>
cd ai-usage-compliance-logger
npm ci
```

Add a `.env` file in the project root (gitignored):

```
OPENROUTER_API_KEY=your_actual_key_here
```

```bash
npm run compile
```

Open the folder in VS Code and press F5, which launches a second window
(the Extension Development Host) with the extension loaded. Open a Git
repo with at least one commit in that window.

Click **AI Assistant** in the status bar to chat, copy the generated code
into a file, commit it, and check the file a few seconds later for an
inserted `@Authorship` comment and a new `.authorship-logs/` folder.

## Labels

| Label | Meaning |
| --- | --- |
| `LLM_GENERATED` | First appeared in the AI's response. |
| `HUMAN_PROMPT_ORIGIN` | You typed it into the chat before the AI produced anything similar. |
| `HUMAN_WRITTEN` | No match in chat history. |
| `MIXED` | Part AI, part you. |
| `UNCERTAIN` | A possible match, not confident enough to commit to. |

Whoever's text appears first in the conversation gets the label, even if
the other side repeats or edits it afterward:

```ts
// @Authorship: human - file lines 10-12; llm - file lines 13-16 | timestamp=2026-09-05
function example() {
  return true;
}
```

Tags are inserted after the commit completes, so they land in the working
tree uncommitted.

## Output

- `.authorship-logs/<commit>_<timestamp>.json`: full analysis for one commit
- `.authorship-logs/function-attributions.jsonl`: one line per function
- `AUTHORSHIP_LATEST_LOG.json`, `AUTHORSHIP_LATEST_REPORT.md`: most recent analysis

## Architecture

| File | Role |
| --- | --- |
| `src/extension.ts` | Entry point; status bar button, starts the Git watcher. |
| `src/chatbotPanel.ts` | Chat UI and conversation history. |
| `src/envLoader.ts` | Reads `.env` for the API key. |
| `src/gitChangeTracker.ts` | Detects new commits, triggers the attribution pipeline. |
| `src/conversationParser.ts` | Extracts code snippets from chat history. |
| `src/diffParser.ts` | Breaks a commit's diff into function-sized blocks. |
| `src/codeSimilarityMatcher.ts` | Scores similarity between two code blocks. |
| `src/authorshipAttributor.ts` | Decides who introduced each block first. |
| `src/codeTagInserter.ts` | Writes the `@Authorship` comment. |
| `src/attributionLogger.ts` | Writes the JSON/JSONL/Markdown logs. |

Commit triggers `gitChangeTracker`, which hands off to `conversationParser`
to index chat history, then `diffParser` extracts the changed functions,
`authorshipAttributor` compares each against history, `codeTagInserter`
writes the comment, and `attributionLogger` saves the log.

## Commands

- `npm ci`: install locked dependencies
- `npm run compile`: build `src/` to `out/`
- `npm run watch`: compile on save
- `npm run lint`: ESLint
- `npm test`: extension test suite

## Notes

Attribution is based on text and structure similarity, not real
understanding of the code, so heavily rewritten or renamed code can confuse
it. Conversation history lives in VS Code's own storage, not this repo, so
switching machines or clearing extension storage loses it. The extension
only reacts to commits made after it starts watching; the first commit it
sees on startup is recorded as a baseline and isn't analyzed. No settings UI
yet; thresholds are set in code via `gitChangeTracker.setConfig(...)`.
