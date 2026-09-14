# LLM-Function-Generation

Generate a dataset for function-change examples with two LLM steps:

1) Generate a functional requirement comment from (old function code, new function code)
2) Regenerate new function code from (old function code, requirement)

## Input

- Default input file: `data/changed_functions.json` (a JSON array of objects)

Note: The input filename is configured in `src/constants.py` via `INPUT_FILENAME`.

The tool uses these fields from each record:

- `full_old_function_code`
- `full_new_function_code`

## Output

Writes a single JSON file (default: `output/llm_generated_dataset.json`) containing an array of objects.

## Run

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Set the environment variable in a .env file for an OpenAI-compatible chat completions endpoint:

- `API_KEY` (required)

The base URL is configured in `src/constants.py` via `BASE_URL` (default points at OpenRouter).

The supported models are defined in `src/constants.py` in `MODEL_MAP`. Model identifier must be passed via the required `--llm` argument:
- When the value matches a key in `MODEL_MAP`, the mapped model identifier is used.

Run:

```bash
python3 main.py --llm model-identifier
```

for example:

```bash
python3 main.py --llm gpt-oss
```
