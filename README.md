# agent-review

A maintainer-gated agentic pull request reviewer for GitHub Actions. Comment an exact command on an open, same-repository PR:

| Command | Passes |
| --- | --- |
| `/review` | Two independent Sonnet passes: bundle first, then callers first |
| `/review deep` | One Opus pass |
| `/review codex` | One GPT-6.1 Terra pass at high effort |
| `/review codex deep` | One GPT-6.1 Sol pass at medium effort |

The engine builds a merge-base diff with full changed files, falling back to changed regions when the configured input budget requires it. The reviewer can read related source. Each pass reports concrete introduced defects with a head location, trigger, consequence, repair, and confidence. Findings are merged by location and mechanism. Each pass can return up to six candidates; there is no evidence-challenge veto.

Inference jobs have read-only GitHub permissions. A separate publisher validates the result and transcript-derived receipts, rechecks the captured commits and original command, then publishes resolvable inline threads and a short summary table. Findings on overlapping lines share one thread. After a full review, verified earlier summaries are hidden as outdated. Related-source findings stay in the summary when no changed line can anchor them. Immutable completion records support duplicate detection, incremental coverage, and verified follow-up replies.

## Required operator policy

The central [.github/pr-agent.toml](.github/pr-agent.toml) retains its filename for compatibility. Each of `review_depth.default`, `review_depth.deep`, `review_depth.codex`, and `review_depth.codex-deep` requires `max_turns` and `max_budget_usd`. The code has no defaults; the shipped policy sets 40 turns, USD 2 per default pass, USD 4 per deep pass, USD 2 per codex pass, and USD 2 per codex-deep pass, so reviews run and incur provider charges once the provider secrets are configured. Removing or invalidating either setting fails before inference with the exact config key to set.

The USD allowance applies **per pass**. A review's configured allowance is its pass count multiplied by that limit. These are client-side cost estimates and can overshoot on an in-flight request; provider account limits remain the billing backstop. The shipped Claude limits come from measured evaluation runs. Codex limits are provisional values from a partial evaluation run, as described in `.github/pr-agent.toml`.

## Adoption

1. Install [.github/workflows/agent-review.yaml](.github/workflows/agent-review.yaml), using the exact maintained caller template. External callers must pin the reusable workflow to a full merged commit SHA with the `# main` tracking comment. [The example caller](examples/pipelock/agent-review.yaml) provides that form.
2. Add `.github/agent-review.toml` with only `[review] instructions` and `[ignore] glob`. Profiles cannot change models, tools, budgets, admission, credentials, or publication.
3. Configure the centrally approved per-depth limits before adopting the engine.
4. Map `ANTHROPIC_API_KEY` for Claude and/or `OPENAI_API_KEY` for Codex as named secrets. The existing second Anthropic secret mapping remains accepted for caller compatibility, but the engine never silently retries or changes provider. Optional `ORG_MEMBERS_TOKEN` supplies Members:read for private organization membership checks on replies.
5. Merge the caller and profile to the default branch, then post a command.

Only the designated maintainer's original, unedited command is admitted. Fork and closed PRs are explicitly refused before provider work. Reruns, bots, edited events, and unsupported command arguments are ineligible. Models and repository text have no authority over publication.

## Development

CI uses Python 3.12 on Linux x86_64 with hash-pinned dependencies:

```bash
pip install --require-hashes -r .github/requirements-runtime.txt -r .github/requirements-test.txt
python -m unittest discover -s scripts -p 'test_*.py'
python -m unittest -v scripts.test_review_engine scripts.test_review_jobs
```

All tests run offline with fixture credentials and provider transcripts. The argv harness proves argument forwarding and credential separation. A native CLI dry run also checks tool restrictions, scoped reads, skipped PR hooks/MCP, schema completion, and turn/budget stops against a local fixture API. It does not establish real provider behavior or hosted-runner isolation. See [the operating contract and vendor evidence](docs/agent-review.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
