# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""Trusted PR-Agent adapter: explicit review, bounded calls, billing-only fallback."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch

if __package__:
    from .agent_review_contract import load_profile
else:
    from agent_review_contract import load_profile

PROFILE = Path(__file__).resolve().parents[1] / ".github" / "agent-review.toml"
CONFIG = Path(__file__).resolve().parents[1] / ".github" / "pr-agent.toml"


class ReviewAbort(BaseException):
    """Bypass upstream's broad retry/partial-result handlers after an unsafe failure."""


def billing_rejection(error: Exception) -> bool:
    """Accept explicit provider billing signals, never a timeout or bare HTTP 429.

    Anthropic's SDK defines billing_error; PR-Agent recognizes the legacy credit
    message. OpenAI uses insufficient_quota. Never inspect formatted log strings.
    LiteLLM 1.103.0 drops Anthropic bodies during mapping, so inspect the original
    HTTP response in the exception chain (http_handler.MaskedHTTPStatusError).
    """
    seen, errors = set(), []
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        # A timeout remains ambiguous even if another exception is attached.
        if (isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower()
                or getattr(error, "status_code", None) in (408, 504)):
            return False
        errors.append(error)
        error = error.__cause__ or error.__context__
    for error in errors:
        response = getattr(error, "response", None)
        status = getattr(error, "status_code", getattr(response, "status_code", None))
        if status not in (400, 402, 429):
            continue
        body = getattr(error, "body", None)
        if not isinstance(body, dict) and response is not None:
            try:
                body = response.json()
            except (ValueError, RuntimeError):
                continue
        if not isinstance(body, dict):
            continue
        detail = body.get("error", body)
        if not isinstance(detail, dict):
            continue
        if detail.get("type") == "billing_error" or detail.get("code") == "insufficient_quota":
            return True
        if (detail.get("type") == "invalid_request_error"
                and "credit balance is too low" in str(detail.get("message", "")).lower()):
            return True
    return False


async def deny_transport_retry(*args, **kwargs):
    # LiteLLM's HTTP handler retries RemoteProtocolError/ConnectError independently
    # of num_retries. This pinned method is called only from those retry branches.
    raise ReviewAbort("Connection failed; ambiguous requests are never retried")


def credential_chain(models: list[str], credentials: dict[str, str]) -> list[tuple[str, str]]:
    """Skip absent providers before inference; slot two requires slot one rejection."""
    chain = []
    for model in models:
        if model.startswith("anthropic/"):
            if credentials.get("ANTHROPIC_API_KEY"):
                chain.append((model, "ANTHROPIC_API_KEY"))
                if credentials.get("ANTHROPIC_API_KEY_2"):
                    chain.append((model, "ANTHROPIC_API_KEY_2"))
        elif model.startswith(("gpt-", "openai/")):
            if credentials.get("OPENAI_API_KEY"):
                chain.append((model, "OPENAI_API_KEY"))
        else:
            raise ValueError("Configure an Anthropic or OpenAI model for this adapter")
    return list(dict.fromkeys(chain))


def check_pull_request(pr, repo, head, base):
    if (pr.state != "open" or pr.head.repo.full_name != repo or pr.head.repo.fork
            or pr.base.repo.full_name != repo or pr.head.sha != head or pr.base.sha != base):
        raise ReviewAbort("Pull request moved or became ineligible; post a new command")


def check_complete(reviewer, calls, failed):
    if (failed or not reviewer.prediction or not calls
            or reviewer.review_failed_chunk_count or reviewer.remaining_files_list):
        raise ReviewAbort("Review incomplete; inspect the coverage footer and workflow log")


class CompletionChain:
    def __init__(self, chain, complete, max_calls):
        self.chain = list(chain)
        self.complete = complete
        self.max_calls = max_calls
        self.calls = 0
        self.index = 0
        self.models_used = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.rejections = 0
        self.usage_observed = False
        self.aborted = False
        self.lock = asyncio.Lock()

    def record_usage(self, response):
        # The pinned handler discards response metadata at its public boundary.
        # Observe only integer usage fields before that boundary, never messages.
        usage = getattr(response, "usage", None)
        for field in self.usage:
            value = getattr(usage, field, None)
            if type(value) is int and value >= 0:
                self.usage_observed = True
                self.usage[field] += value

    async def run(self, *args):
        # Upstream chunks run concurrently. Serialize so rejected slots stay retired
        # and the aggregate call limit cannot race. Upstream gather does not cancel
        # sibling chunks, so the first failure must stop every queued request.
        async with self.lock:
            if self.aborted:
                raise ReviewAbort("Review already failed; no further provider requests")
            try:
                return await self._run_locked(*args)
            except BaseException:
                self.aborted = True
                raise

    async def _run_locked(self, *args):
        if self.calls >= self.max_calls:
            raise ReviewAbort("Completion budget exceeded")
        while self.index < len(self.chain):
            model, slot = self.chain[self.index]
            try:
                result = await self.complete(model, slot, *args)
            except Exception as error:
                if billing_rejection(error):
                    self.rejections += 1
                    self.index += 1
                    continue
                raise ReviewAbort("Provider failed; ambiguous requests are never retried") from None
            if result[1] not in ("stop", "end_turn"):
                raise ReviewAbort("Provider returned an incomplete completion")
            self.calls += 1
            if model not in self.models_used:
                self.models_used.append(model)
            print(f"Completed review call {self.calls} with {model}")
            return result
        raise ReviewAbort("Configured provider credits exhausted")


def review_depth():
    depth = os.environ.get("REVIEW_DEPTH", "default")
    if depth not in ("default", "deep"):
        raise ReviewAbort("Invalid review depth")
    return depth


def configure():
    profile, _ = load_profile(Path(os.environ.get("REVIEW_PROFILE_PATH", PROFILE)))
    # Set before importing LiteLLM: its default import can fetch mutable metadata.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    import litellm
    from pr_agent.config_loader import get_settings
    from pr_agent.log import get_logger

    # Upstream exception logging may include request/response text. Emit only
    # our fixed status messages and allowlisted metadata, never upstream logs.
    get_logger().remove()
    settings = get_settings()
    data = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    for section, entries in data.items():
        for key, value in entries.items():
            settings.set(f"{section}.{key}", value)
    settings.set("pr_reviewer.extra_instructions",
                 data["pr_reviewer"]["extra_instructions"].rstrip() + "\n\n" + profile["review"]["instructions"])
    depth = review_depth()
    if depth == "deep":
        settings.set("config.reasoning_effort", "high")
        settings.set("pr_reviewer.extra_instructions",
                     settings.pr_reviewer.extra_instructions +
                     "\nTrace changed behavior through callers and failure paths. "
                     "Check malformed inputs, stale state, concurrency, and rollback. "
                     "Prioritize concrete security and correctness defects over breadth.")
    settings.set("ignore.glob", profile["ignore"]["glob"])
    # Provider registration supplies routing metadata only. The operator's token
    # ceiling above controls budgeting; it is not the model's advertised context.
    litellm.register_model({"gpt-6-luna": {
        "litellm_provider": "openai", "mode": "chat",
        "supports_system_messages": True,
    }})
    litellm.telemetry = False
    litellm.suppress_debug_info = True
    litellm.redact_messages_in_exceptions = True
    return settings, litellm


def print_settings(settings, litellm):
    models = [settings.config.model, *settings.config.fallback_models]
    # Allowlisted non-secret fields only; never dump the full Dynaconf settings.
    print(json.dumps({
        "model": models[0], "fallback_models": models[1:],
        "routes": {m: litellm.get_llm_provider(m)[:2] for m in models},
        "inline_key_issues": settings.pr_reviewer.inline_key_issues,
        "persistent_comment": settings.pr_reviewer.persistent_comment,
        "security_review": settings.pr_reviewer.require_security_review,
        "max_findings": settings.pr_reviewer.num_max_findings,
        "max_model_tokens": settings.config.max_model_tokens,
        "max_output_tokens": settings.config.max_output_tokens,
        "max_calls": settings.pr_reviewer.max_number_of_calls,
        "repo_settings": settings.config.use_repo_settings_file,
        "telemetry": settings.otel.is_enabled,
    }, sort_keys=True))


STATUS_MARKER = "agent-review-status:v1"
RESULT_MARKER = "agent-review-result:v1"


def bot_owned(comment):
    user = getattr(comment, "user", None)
    return (getattr(user, "login", None) == "github-actions[bot]"
            and getattr(user, "type", None) == "Bot")


def result_manifest(evidence):
    """Encode findings without a digest field. The digest covers the comment that contains this."""
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode()


def result_comment(manifest):
    return f"<!-- {RESULT_MARKER} {manifest} -->"


def stored_findings_ok(findings):
    if not isinstance(findings, list):
        return False
    for item in findings:
        if not isinstance(item, dict):
            return False
        title, body = item.get("title"), item.get("body")
        if not isinstance(title, str) or not title.strip() or not isinstance(body, str) or not body.strip():
            return False
        if item.get("path") == "unknown" and item.get("line") is None and item.get("end_line") is None:
            continue
        path, line, end = item.get("path"), item.get("line"), item.get("end_line")
        if (not isinstance(path, str) or not path.strip() or path == "unknown"
                or type(line) is not int or line < 1 or type(end) is not int or end < line):
            return False
    return True


def exact_positive_int(value):
    # bool is an int subclass, and JSON true must not compare equal to pull request 1.
    return type(value) is int and value > 0


def parse_result(body):
    """Use the last marker. Model text can precede the marker the publisher appends."""
    matches = list(re.finditer(rf"<!-- {RESULT_MARKER} ([A-Za-z0-9+/=]+) -->", body))
    if not matches:
        return None
    token = matches[-1].group(1)
    try:
        data = json.loads(base64.b64decode(token, validate=True))
    except (ValueError, json.JSONDecodeError):
        return None
    if (not isinstance(data, dict) or not exact_positive_int(data.get("number"))
            or not exact_positive_int(data.get("summary"))
            or not stored_findings_ok(data.get("findings"))):
        return None
    return token, data


def completed_review(comments, key, repository, number, head, base, depth):
    """Reuse only a bot success record whose summary digest covers the same findings."""
    comments = list(comments)
    summaries = {c.id: c for c in comments if bot_owned(c) and isinstance(c.body, str)}
    pattern = rf"<!-- {STATUS_MARKER} state=success key={key} summary=([1-9][0-9]*) digest=([0-9a-f]{{64}}) -->"
    for comment in summaries.values():
        match = re.fullmatch(pattern, comment.body.split("\n", 1)[0])
        if not match:
            continue
        summary = summaries.get(int(match[1]))
        if summary is None or hashlib.sha256(summary.body.encode()).hexdigest() != match[2]:
            continue
        published = parse_result(summary.body)
        recorded = parse_result(comment.body)
        if published is None or recorded is None or published[0] != recorded[0]:
            continue
        data = published[1]
        if (data.get("key") != key or data.get("summary") != int(match[1])
                or data.get("repository") != repository or data.get("number") != int(number)
                or data.get("head") != head or data.get("base") != base or data.get("depth") != depth):
            continue
        return comment
    return None


def review_binding(settings, repo, number, head, base):
    source, profile_sha = os.environ["REVIEW_SOURCE_SHA"], os.environ["REVIEW_PROFILE_SHA"]
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (source, profile_sha)):
        raise ReviewAbort("Invalid reviewer or profile source identity")
    profile, profile_hash = load_profile(Path(os.environ.get("REVIEW_PROFILE_PATH", PROFILE)))
    models = [settings.config.model, *settings.config.fallback_models]
    binding = {"repository": repo, "number": number, "head": head, "base": base,
               "reviewer": source, "profile_hash": profile_hash, "models": models,
               "common_config": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
               "depth": review_depth()}
    key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return key, source, profile_sha, profile_hash, profile["ignore"]["glob"]


def provenance(source, profile_sha, profile_hash, exclusions):
    # Escape code-span delimiters as JSON Unicode escapes, preserving each glob.
    shown_exclusions = json.dumps(exclusions, ensure_ascii=True).replace("`", "\\u0060")
    return (f"Reviewer source `{source}`; caller profile `{profile_sha}` "
            f"(SHA-256 `{profile_hash}`).\n\n"
            f"Declared exclusions: `{shown_exclusions}`.\n\nReview depth: `{review_depth()}`.")


def result_findings(data):
    """Preserve PR-Agent's review fields for consumers without parsing rendered HTML.

    Source: the pinned pr_reviewer_prompts.toml Review/KeyIssues schema. Its
    security_concerns contract specifies the exact English literal 'No'.
    """
    result = data.get("review") if isinstance(data, dict) else None
    if not isinstance(result, dict):
        raise ReviewAbort("Review result has no structured review")
    issues = result.get("key_issues_to_review")
    concerns = result.get("security_concerns")
    if not isinstance(issues, list) or not isinstance(concerns, str) or not concerns.strip():
        raise ReviewAbort("Review result has incomplete structured findings")
    findings = []
    for issue in issues:
        # The pinned prompt's example uses literal block scalars. PyYAML keeps
        # the trailing newline, and the publisher strips it before anchoring.
        text = {}
        for key in ("relevant_file", "issue_header", "issue_content"):
            value = issue.get(key) if isinstance(issue, dict) else None
            text[key] = value.strip() if isinstance(value, str) else ""
        if (not isinstance(issue, dict) or any(not text[key] for key in text)
                or type(issue.get("start_line")) is not int or issue["start_line"] < 1
                or type(issue.get("end_line")) is not int or issue["end_line"] < issue["start_line"]):
            raise ReviewAbort("Review result has invalid structured findings")
        findings.append({"path": text["relevant_file"], "line": issue["start_line"],
                         "end_line": issue["end_line"], "title": text["issue_header"],
                         "body": text["issue_content"]})
    # Exact 'No' after the same strip. 'no' and 'None' stay findings.
    concern = concerns.strip()
    if concern != "No":
        findings.append({"path": "unknown", "line": None, "end_line": None,
                         "title": "Security concerns", "body": concern})
    return findings


async def review(settings):
    started = time.monotonic()
    from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
    from pr_agent.algo.run_details import command_failed
    from pr_agent.tools.pr_reviewer import PRReviewer
    from tenacity import stop_after_attempt
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    credentials = {name: os.environ.pop(name, "") for name in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2", "OPENAI_API_KEY",
    )}
    models = [settings.config.model, *settings.config.fallback_models]
    chain = credential_chain(models, credentials)
    settings.set("github.user_token", os.environ.pop("GITHUB_TOKEN"))
    settings.set("github.deployment_type", "user")
    repo = os.environ["GITHUB_REPOSITORY"]
    number = os.environ["PR_NUMBER"]
    head, base = os.environ["REVIEW_HEAD"], os.environ["REVIEW_BASE"]
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
            or not re.fullmatch(r"[1-9][0-9]*", number)
            or not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (head, base))):
        raise ReviewAbort("Invalid admitted pull request identity")

    async def complete(model, slot, system, user, temperature, image):
        if image:
            raise ReviewAbort("Images are outside the static-diff review contract")
        settings.set("anthropic.key", credentials[slot] if slot.startswith("ANTHROPIC") else "")
        settings.set("openai.key", credentials[slot] if slot == "OPENAI_API_KEY" else "")
        handler = LiteLLMAIHandler()
        # Tenacity's supported retry_with API disables the upstream handler's
        # same-model retries; config.num_retries disables completion-client retries.
        once = handler._chat_completion_with_retry.retry_with(stop=stop_after_attempt(1))
        get_completion = handler._get_completion

        async def measured_completion(**kwargs):
            result = await get_completion(**kwargs)
            router.record_usage(result[2])
            return result

        with (patch.object(AsyncHTTPHandler, "single_connection_post_request", deny_transport_retry),
              patch.object(handler, "_get_completion", measured_completion)):
            return await once(handler, model, system, user, temperature, image,
                              configured_deployment_id=handler.deployment_id)

    # Fingerprint the configured chain, not the credential-dependent first route.
    key, source, profile_sha, profile_hash, exclusions = review_binding(settings, repo, number, head, base)
    settings.set("config.model", chain[0][0] if chain else models[0])
    settings.set("config.fallback_models", [])
    record = provenance(source, profile_sha, profile_hash, exclusions)
    run_id = os.environ["GITHUB_RUN_ID"]
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ReviewAbort("Invalid review run identity")
    run_url = f"https://github.com/{repo}/actions/runs/{run_id}"
    notice = f"<!-- agent-review-notice:v1 run={run_id} -->"

    router = CompletionChain(chain, complete, settings.pr_reviewer.max_number_of_calls)

    class Handler:
        main_pr_language = ""

        async def chat_completion(self, model, system, user, temperature=0.2, img_path=None):
            return await router.run(system, user, temperature, img_path)

    class Reviewer(PRReviewer):
        def check_binding(self, fresh=False):
            provider = self.git_provider
            pr = provider.github_client.get_repo(repo).get_pull(int(number)) if fresh else provider.pr
            check_pull_request(pr, repo, head, base)

        def _prepare_pr_review(self):
            self.check_binding(fresh=True)
            data = self.prediction_data if self.prediction_data is not None else self._load_review_yaml(self.prediction)
            if not self._validate_review_schema(data):
                raise ReviewAbort("Provider returned an invalid review schema")
            self.result_findings = result_findings(data)
            result = super()._prepare_pr_review()
            if not result:
                raise ReviewAbort("Provider returned an invalid review")
            return result + f"\n\nReviewed base `{base}` and head `{head}`.\n\n{record}"

    reviewer = Reviewer(f"https://github.com/{repo}/pull/{number}", ai_handler=Handler)
    reviewer.check_binding()
    previous = completed_review(reviewer.git_provider.get_issue_comments(), key,
                                repo, int(number), head, base, review_depth())
    if previous is not None:
        reviewer.check_binding(fresh=True)
        reviewer.git_provider.pr.create_issue_comment(
            f"{notice}\nThis snapshot already has a completed diff review. "
            f"[Completed review]({previous.html_url}); no new inference was requested.\n\n"
            f"[Actions run]({run_url}).")
        print(f"Reused completed review: {previous.html_url}")
        return
    if not chain:
        raise ReviewAbort("No configured provider credential is available")
    pr = reviewer.git_provider.pr
    # The review job's repository/PR concurrency serializes this status pattern.
    # A running or failed record is never evidence of successful completion.
    status = pr.create_issue_comment(
        f"{notice}\n<!-- {STATUS_MARKER} state=running key={key} -->\n"
        f"Diff review started for base `{base}` and head `{head}`.\n\n{record}\n\n"
        f"[Actions run]({run_url}).")
    published = []
    provider = reviewer.git_provider

    def require_publication(original):
        def publish(*args, **kwargs):
            result = original(*args, **kwargs)
            if result is None or result is False:
                raise ReviewAbort("Review summary publication failed")
            published.append(result)
            return result
        return publish

    try:
        with (patch.object(provider, "publish_persistent_comment",
                           require_publication(provider.publish_persistent_comment)),
              patch.object(provider, "publish_persistent_comment_full",
                           require_publication(provider.publish_persistent_comment_full))):
            await reviewer.run()
    except Exception as error:
        # Upstream raises generic exceptions, for example after an unparseable
        # prediction. Report only the class; the message can carry model output.
        raise ReviewAbort(f"PR-Agent review failed ({type(error).__name__})") from None
    check_complete(reviewer, router.calls, command_failed())
    if not published:
        raise ReviewAbort("Review summary was not published")
    reviewer.check_binding(fresh=True)
    summary = published[-1]
    if (not bot_owned(summary) or type(summary.id) is not int or summary.id <= 0
            or not isinstance(summary.body, str)):
        raise ReviewAbort("Review summary identity could not be verified")
    # The digest is the hash of the summary after the findings comment is appended.
    # Keeping the digest inside that comment would make the hash depend on itself.
    evidence = {"repository": repo, "number": int(number), "base": base, "head": head,
                "depth": review_depth(), "key": key, "summary": summary.id,
                "findings": reviewer.result_findings}
    manifest = result_manifest(evidence)
    line = result_comment(manifest)
    edit = getattr(summary, "edit", None)
    if not callable(edit):
        raise ReviewAbort("Review summary could not record structured findings")
    bound = summary.body + "\n" + line
    edit(bound)
    if summary.body != bound:
        summary.body = bound
    digest = hashlib.sha256(summary.body.encode()).hexdigest()
    usage = router.usage if router.usage_observed else None
    runtime_seconds = int(time.monotonic() - started)
    # Only this post-publication transition permits identical-snapshot reuse.
    status.edit(f"<!-- {STATUS_MARKER} state=success key={key} summary={summary.id} digest={digest} -->\n"
                f"Completed diff review for base `{base}` and head `{head}`.\n\n{record}\n\n"
                f"Models used: `{json.dumps(router.models_used)}`. "
                f"Successful calls: {router.calls}; billing rejections: {router.rejections}; "
                f"runtime: {runtime_seconds} seconds.\n\n"
                f"Reported token usage: `{json.dumps(usage, sort_keys=True)}`.\n\n"
                f"[Review summary]({summary.html_url}) · [Actions run]({run_url}).\n\n"
                f"{line}")
    print(json.dumps({"models": router.models_used, "successful_calls": router.calls,
                      "billing_rejections": router.rejections, "reported_tokens": usage,
                      "runtime_seconds": runtime_seconds}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    try:
        settings, litellm = configure()
        print_settings(settings, litellm)
        if not args.check_config:
            asyncio.run(review(settings))
    except (ReviewAbort, Exception) as error:
        # Provider exception messages can echo requests or credentials. Report a
        # fixed failure class; only our own ReviewAbort messages are safe to show.
        message = str(error) if isinstance(error, ReviewAbort) else type(error).__name__
        print(f"Review failed: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
