# agent-review

A maintainer-gated pull request review command for GitHub Actions. A maintainer comments `/review` on a pull request, and a reusable workflow reviews the static diff with a language model, posts findings as resolvable inline threads, and preserves each summary plus a completion record naming the exact commits, reviewer source, profile, models, and token usage.

Runs [PR-Agent](https://github.com/The-PR-Agent/pr-agent) (MIT) from its published container image.

## What it does

Only the designated maintainer can start a review with an original, unedited `/review` or `/review deep` comment on an open, same-repository pull request. Normal review uses Sonnet 5.5 at medium effort; deep review uses Opus 5.5 at high effort. Both retain the configured OpenAI fallback. Forks, closed PRs, edited events, reruns, bots, and unsupported commands can't start a review. Normal and deep completions have separate identities. The command runs from the default branch, so a pull request can't change the workflow that reviews it.

The review engine loads from this repository at a full commit SHA, and a credential-free helper checks that GitHub loaded that revision before anything else runs. The calling repository's profile and caller are read as bounded data from an immutable default-branch revision, and only Python from this repository executes. The model gets no tools, no environment access, and no authority over credentials or where results are published.

Provider fallback happens only on an explicit billing rejection. Timeouts, rate limits, authentication errors, malformed output, and truncated completions fail without another paid attempt. When a review can't finish, a host-only step posts a fixed status comment so a failed run is never mistaken for a clean one. Repeating the command on an unchanged, verified snapshot reuses its completion chain. New commits can use an immutable delta from the last verified review; missing proof, changed policy, or unsafe ancestry requires a full review. Earlier findings and threads remain for the maintainer's sweep.

Josh or a current organization member can ask a follow-up question in this workflow's verified inline thread. GitHub membership, the unchanged triggering comment, the current PR snapshot, and the original root's workflow artifact must all verify before inference. Replies use the normal model policy and remain static text in the same thread. The model has no tools or configuration access.

Candidate findings receive a separate evidence challenge before publication within the existing call ceiling. Repeated scenarios and held decisions remain visible without another inline thread.

Model choice is operator policy, not a quality guarantee. Findings are advice and can be wrong. A completed run doesn't establish review quality or recall.

## Adopting it in a repository

1. Add `.github/workflows/agent-review.yaml`. It must match the maintained caller template exactly, with `uses:` pointing at `pipelab-org/agent-review/.github/workflows/agent-review-reusable.yaml` at a full 40-character merged commit SHA followed by ` # main`. [examples/pipelock/agent-review.yaml](examples/pipelock/agent-review.yaml) is a complete caller; replace the SHA placeholder. Never use a branch or tag reference.
2. Add `.github/agent-review.toml` with exactly two sections: `[review] instructions` for repository-specific review guidance, and `[ignore] glob` for paths to exclude. Anything else is rejected before inference. [examples/pipelock/agent-review.toml](examples/pipelock/agent-review.toml) is a working profile.
3. Add at least one provider key as a repository or organization secret: `ANTHROPIC_API_KEY`, `ANTHROPIC_API_KEY_2` (used only after the first is rejected for billing), or `OPENAI_API_KEY`. Don't use `secrets: inherit` or environment secrets. GitHub repository operations use the caller's own `github.token`. For private organization membership, optionally provide `ORG_MEMBERS_TOKEN` with GitHub's Members:read permission. That token is used only for the membership API; unavailable membership fails closed.
4. Merge both files to the default branch, then comment `/review` on a pull request.

The caller template fixes who can trigger a review: one maintainer account, matched by login and numeric ID, commenting as an organization member. Changing that identity is a change to this repository's validator and reusable workflow, not to a caller.

Models, budgets, retries, publishing, and the shared review rubric live in [.github/pr-agent.toml](.github/pr-agent.toml). A profile can add instructions and exclusions but can't change any of those.

## Updating a caller

A merge here doesn't move any caller. To adopt a newer engine, update the caller's SHA to a merged commit after it has passed CI and a smoke review. Renovate's GitHub Actions manager can track the SHA through the `# main` comment.

## Development

```bash
pip install --require-hashes -r .github/requirements-test.txt
python3 -m unittest discover -s scripts -p 'test_*.py'
```

The runtime tests skip without PR-Agent installed. CI runs them inside the same digest-pinned image the review job uses. [docs/agent-review.md](docs/agent-review.md) covers the trust model, spending limits, failure reporting, and maintenance in detail.

## License

Apache-2.0. See [LICENSE](LICENSE).
