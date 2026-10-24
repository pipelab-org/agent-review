# Historical review replay

`scripts/review_replay.py` runs the review adapter against a saved snapshot of a pull request, so an operator can measure a reviewer revision without posting anything to GitHub. It runs `scripts/pr_agent_review.py` unmodified: the same prompts, model selection, budgets, single attempt per call, billing-only fallback, source discovery, finding challenge and validators. Only the pull request provider is replaced.

It measures nothing by itself. One replay is one sample of a model's output, not a precision or recall figure, and a replay that used a transport fixture says nothing about review quality.

## How it works

PR-Agent builds its provider through `get_git_provider_with_context`, the same seam the runtime tests patch. The replay returns a subclass of the pinned `GithubProvider` that skips the parent constructor, so no GitHub client, token or REST requester ever exists. Description, commit messages, diff files, languages, requested source and prior review threads come from the snapshot. Every write the adapter attempts (status comments, the summary, inline reviews) lands in an in-memory store and is reported in the result. Anything the replay does not model raises and is listed in `unmodeled_provider_access`; nothing returns a mock.

Use the digest-pinned PR-Agent image for comparable runs. The adapter needs the installed `pr_agent` package; the CLI does not attest its container image. From the engine checkout under test:

```bash
docker run --rm --user 1001:1001 --cap-drop=ALL --security-opt=no-new-privileges \
  -e TIKTOKEN_CACHE_DIR=/tokenizers -v "$TOKENIZER_CACHE":/tokenizers:ro \
  -v "$PWD":/work:ro -v "$SNAPSHOT_DIR":/snapshots:ro -w /work --entrypoint python \
  pragent/pr-agent:0.47.0@sha256:7d98954c29289846a08faf00aaea0f1f1375b1676cd8a2ed303e0af07010af7a \
  -B scripts/review_replay.py --snapshot /snapshots/case.json --snapshot-sha256 "$SNAPSHOT_SHA256" \
  --reviewer-sha "$(git rev-parse HEAD)" --inference fixture --fixture /snapshots/fixture.json
```

Exit status is 0 when the adapter reached its own completion transition, 1 when it aborted (the result is still printed), and 2 when nothing ran because an input was invalid. `--output FILE` writes the result instead of printing it. `--profile FILE --profile-sha SHA` selects a profile other than the engine's. `--depth deep` selects the deep policy. `--record-prompts` adds the exact model messages to each call, which contain snapshot text.

Prepare the tokenizer data first, as the CI runtime job does, and mount it read-only. The replay cannot fetch it: outside the configured model gateways, network access is denied.

## Exporting local history

`scripts/review_replay_export.py` reads full base and head commit SHAs from a local Git repository. It exports changed file content and text hunks plus unchanged UTF-8 source. It never reads the working tree, fetches from a remote, or imports the source it captures. Title and description are neutral, commit messages and discussion are empty, and `languages` is empty. This is a code-only comparison, not an exact recreation of an earlier hosted review.

Pass `--repo`, `--repository`, `--number`, `--base`, `--head`, and a new `--output` path. The exporter prints the snapshot SHA-256 and refuses to overwrite an existing file. Optional repeated `--absent` paths are checked against the head tree before being declared absent. Changed binary files, symlinks, submodules, and changes without text hunks are unsupported and abort export. Unchanged binary files are omitted; requests for uncaptured files still abort replay rather than imply that the source does not exist. Export loads repository blobs into memory, so use a disposable process with capacity for the repository size.

## Inference modes

`--inference fixture --fixture FILE` makes no network request. The fixture is an ordered list of responses that stand in for the model gateway, consumed one per completion call:

```json
{"schema": "agent-review-replay-transport-fixture", "version": 1, "responses": [
  {"text": "source_requests: []\nreview: {...}", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
  {"error": {"status": 402, "body": {"error": {"type": "billing_error", "message": "..."}}}},
  {"error": {"timeout": true}}
]}
```

Fixtures exercise the adapter, its transport handling and the CLI. The result says so in `inference.note` and sets `inference.quality_evidence` to false. The fixture path populates every credential slot with a label so the adapter admits its configured chain; no request is made with them.

`--inference real` uses the existing handler and whatever model credentials the environment already holds. The replay never reads credential files, prints no credential, and refuses to start when no configured provider credential is present. In real mode the guard allows the documented API hosts of the configured models' providers over HTTPS and nothing else, and ambient GitHub tokens are removed from the process environment before the adapter runs. The credential names and values come from the same variables the Actions job sets; this change does not provide a way to obtain them.

The network guard is in-process defense behind the provider, which already has no GitHub path. It patches name resolution, socket connects, `httpx` and `requests`. It is not a firewall: it does not constrain other processes, and a hosted runner has no egress filter of its own. A configured proxy is permitted by the standard proxy variables; that path has not been exercised against a live proxy.

## Snapshot contract

A new schema is needed because `FilePatchInfo` is an in-memory dataclass and the completion artifact records a review's output, not its input. Neither can be loaded to re-run a review. The document carries what the adapter reads from GitHub in one review, and nothing else:

| Field | Content |
|---|---|
| `schema`, `version` | `agent-review-replay-snapshot`, `1` |
| `repository`, `number`, `base`, `head` | Same identity checks as the adapter: `owner/name`, positive integer, full 40-character commit SHAs |
| `title`, `branch`, `description`, `commits` | Pull request text as the provider supplies it; an absent body is `""` |
| `languages` | Repository languages as bytes per language |
| `files` | One object per changed file, using `FilePatchInfo` names: `filename`, `base_file`, `head_file`, `patch`, `edit_type` (`ADDED`, `DELETED`, `MODIFIED`, `RENAMED`), optional `old_filename`, `num_plus_lines`, `num_minus_lines` |
| `sources` | Repository-relative path to its text at the head, for unchanged files a model may request |
| `absent` | Paths known not to exist at the head, so a request for one reads as determinately unavailable |
| `discussion` | `as_of` (UTC `YYYY-MM-DDTHH:MM:SSZ`) and `threads`, each with `id`, `in_reply_to_id`, `path`, `body` |

The schema is closed, so unknown metadata fields are rejected. This does not detect answers embedded in description, discussion, or source text. The snapshot exporter must omit later fixes and discussion, and keep expected outcomes in a separate file. Paths are checked by the same validator that checks model source requests. A snapshot's identity is the SHA-256 of its exact bytes; `--snapshot-sha256` pins it. Symlinks in the path, non-regular files, duplicate JSON keys and invalid UTF-8 are refused. No byte ceiling is applied to the file, because any number would be invented. What reaches a model is bounded by the adapter's existing token-budget refusal.

The replay presents the snapshot as an open pull request in its own repository at the captured commits, because the adapter refuses anything else. Prior completion comments are never presented, so a replay always reviews the full diff and cannot take the adapter's reuse or incremental paths. The replay serves source reads at the captured head only. A requested path that is neither in `sources`, nor `absent`, nor a changed file is not evidence that the file is missing, so the run aborts with `uncaptured_source_requests` naming the path and `comparable: false`. Capture the files a reviewer will ask for, or the comparison is not valid.

## Result document

The result is one JSON document with `replay: true` and `genuine_github_completion: false`. It records the snapshot identity, the asserted reviewer and profile SHAs, a SHA-256 of every engine file that executed (`executed_files`), the effective models, effort and budgets, the credential route by slot name, each call's model, parameters and message hashes, token usage, the findings and the challenge's decisions, and the captured publications.

Captured comments have their `agent-review-` markers and the replay's run id rewritten, so they parse as neither a completion record nor a status line, and the document is not a comment or a completion artifact. A replay run id is a sentinel, not an Actions run, and `review_followups.py` accepts a completion only with the immutable archive of a real run.

To compare two reviewer revisions, replay the same snapshot, by the same SHA-256, at the same depth, once from each revision's checkout in real mode, and compare `findings`, `quality.decisions`, `calls` and `executed_files`. Treat results with `comparable: false` or `adapter_outcome: aborted` as failed runs, not as a reviewer finding nothing.

## Running it in Actions

Not enabled. The existing review workflow cannot run a replay, and adding one needs these decisions first.

- Snapshots drawn from private repositories are private data. They must come from a private repository over a read-only credential, never from the public engine repository or a public artifact. Results can quote that source, so they need the same handling.
- The revision under test must be an already merged commit on the default branch, resolved to a full SHA, with `workflow_dispatch` limited to the same maintainer gate as `/review`. An unmerged revision would run proposed code in a job that holds model credentials. Measuring an unmerged fix needs a protected environment with a required reviewer, or a fixture-only run.
- Each job would use empty top-level permissions, `contents: read` only, and a checkout with `persist-credentials: false`. The replay step needs the provider secrets and no GitHub token; no job needs `issues: write` or `pull-requests: write`.
- Both revisions would run in the same pinned container with the same hardening options, one snapshot at a time, with the existing 25-minute job limit. A replay makes at most the adapter's three completion calls.
- The tokenizer cache must be prepared before the step that has secrets, as the runtime job does.
- A hosted runner cannot filter egress, so the in-process guard is the only network control there.

## Adapter seam

No change to `pr_agent_review.py` is needed. The provider factory is upstream's seam, and the adapter already reads its inputs from environment variables the replay sets. Tying the replay to `PRReviewer`'s import of that factory means a pin bump can break it, which the replay tests catch because they run in the pinned image.
