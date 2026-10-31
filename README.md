# agent-review

A maintainer-gated agentic pull request reviewer for GitHub Actions. Comment an exact command on an open, same-repository PR:

| Command | Passes |
| --- | --- |
| `/review` | Two independent Sonnet passes: bundle first, then callers first |
| `/review deep` | One Opus pass |
| `/review codex` | One GPT-6.1 Terra pass at high effort |
| `/review codex deep` | One GPT-6.1 Sol pass at medium effort |

The engine builds a merge-base diff with full changed files, falling back to changed regions when the configured input budget requires it. The reviewer can read related source. Each pass reports concrete introduced defects with a head location, trigger, consequence, repair, and confidence. Findings are merged by location and mechanism. Each pass can return up to six candidates; there is no evidence-challenge veto.

Inference jobs have read-only GitHub permissions. A separate publisher validates the result and transcript-derived receipts, rechecks the captured commits and original command, then publishes resolvable inline threads and a short summary table. Findings on overlapping lines share one thread. Earlier verified success status lines are hidden as outdated; earlier summaries stay visible. Related-source findings stay in the summary when no changed line can anchor them. Immutable completion records support duplicate detection, incremental coverage, and verified follow-up replies.

When a review runs inference, it can also verify open threads backed by its completion records. One batched pass uses that review's provider, model, read-only tools, and per-pass limits. Only threads whose code changed (GitHub marks them outdated) are checked. Fixed evidence must cover every finding and include each finding's recorded path; evidence only from related files leaves the thread open. Present, unclear, unowned, edited, or unverifiable findings stay open. Silence in a later review never resolves anything. A verification failure leaves threads open without failing the review. Unchanged snapshots skip inference and thread verification.

Resolution remains a model judgment, as with CodeRabbit's documented [Auto-Resolve](https://docs.coderabbit.ai/changelog). The pull request author controls both the code the model reads and which threads become outdated, so automatic resolution is a convenience, not a security sign-off; turn follow-ups off where findings gate merges. Resolved threads remain in the PR and can be reopened. Once the bot posts a completed verified decision or records a resolution attempt for a root, it never auto-resolves that root again, even at the same head or another review depth. A failed resolve needs a person or a new `/resolve` command. Model text cannot supply the bot's state markers.

Eligible inline replies read the head checkout with restricted file tools. They can answer earlier-head threads and resolve an outdated thread when the human reports a fix that the code confirms. When the code appears to disprove a finding, the answer says so and leaves the thread open for a maintainer. Disputed findings that remain valid stay open with an explanation. Membership, event binding, provenance, and publication checks still apply.

To resolve one owned thread explicitly, reply with exactly `/resolve` (case-sensitive; surrounding whitespace is allowed). The admitted command skips model inference, acknowledges the request, and asks GitHub to resolve only that thread. It can resolve live or previously reopened threads. An already-resolved thread produces no action. A reply from a non-member, an edited comment, or an unowned root never authorizes the command. Only the pull request author or someone with write access, the people GitHub itself lets [resolve a conversation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/commenting-on-a-pull-request#resolving-conversations), can have the bot resolve a thread, by `/resolve` or by a reply reporting a fix; anyone else gets an answer and the thread stays open. Other text uses the ordinary reply path. The acknowledgment claims success only after GitHub confirms resolution; ambiguous failures stay visible.

## Required operator policy

The central [.github/pr-agent.toml](.github/pr-agent.toml) retains its filename for compatibility. Each of `review_depth.default`, `review_depth.deep`, `review_depth.codex`, and `review_depth.codex-deep` requires `max_turns` and `max_budget_usd`. The code has no defaults; the shipped policy sets 40 turns, USD 2 per default pass, USD 4 per deep pass, USD 2 per codex pass, and USD 2 per codex-deep pass, so reviews run and incur provider charges once the provider secrets are configured. Removing or invalidating either setting fails before inference with the exact config key to set.

The USD allowance applies **per pass**. A review's configured allowance is its review pass count multiplied by that limit, plus one pass when thread verification runs. Verification that cannot fit the existing limits is skipped with a fixed notice. These are client-side cost estimates and can overshoot on an in-flight request; provider account limits remain the billing backstop. The shipped Claude limits come from measured evaluation runs. Codex limits are provisional values from a partial evaluation run, as described in `.github/pr-agent.toml`.

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
