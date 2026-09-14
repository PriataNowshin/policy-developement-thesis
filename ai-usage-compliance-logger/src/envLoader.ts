import * as fs from 'fs';
import * as path from 'path';

/**
 * Minimal .env loader. Reads KEY=VALUE lines from a .env file at the
 * extension root and applies them to process.env without overwriting
 * variables that are already set (e.g. by the host environment).
 */
export function loadEnvFile(extensionPath: string): void {
    const envPath = path.join(extensionPath, '.env');

    let contents: string;
    try {
        contents = fs.readFileSync(envPath, 'utf8');
    } catch {
        return;
    }

    for (const line of contents.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith('#')) {
            continue;
        }

        const separatorIndex = trimmed.indexOf('=');
        if (separatorIndex === -1) {
            continue;
        }

        const key = trimmed.slice(0, separatorIndex).trim();
        let value = trimmed.slice(separatorIndex + 1).trim();

        if (
            (value.startsWith('"') && value.endsWith('"')) ||
            (value.startsWith("'") && value.endsWith("'"))
        ) {
            value = value.slice(1, -1);
        }

        if (key && !(key in process.env)) {
            process.env[key] = value;
        }
    }
}
