# Policy Development for LLM-Assisted Bug Fixing

Code and data for a thesis on governing LLM-assisted software development:
what practitioners want from disclosure and oversight policies, and
technical support for putting those policies into practice.

## Structure

- `llm-assisted-bug-fixing-policies/`, Study 1 & 2: a two-phase practitioner
  survey on LLM-usage concerns, disclosure/oversight preferences, and
  organizational policy trade-offs, including a Nash-equilibrium analysis of
  policy choices. Survey instruments in `survey-design/`, data and thematic
  analysis in `survey-findings/`.
- `ai-usage-compliance-logger/`, Study 3 Part 1: a VS Code extension that
  links a developer's LLM chat history to later commits and labels each
  function as human, LLM, or mixed-authored, producing disclosure records
  automatically without developers having to self-report.
- `performance-prediction-model/`, Study 3 Part 2: predicting, before an LLM
  generates any code, how closely its update to a function will
  structurally match what a human actually did. Three prediction pathways
  (requirement-only, AST-only, fusion) plus a synthetic mutation follow-up.

Each folder has its own README with setup and reproduction detail.

## Motivation

Studies 1 and 2 find that practitioners want LLM use disclosed and risky
changes reviewed, but disclosure and review overhead is itself a barrier to
adoption. Study 3 responds with two technical components: a logger that
removes the burden of manually tracking LLM involvement, and a predictor
that estimates review-worthiness before code is generated, so review effort
can go where it's actually needed.
