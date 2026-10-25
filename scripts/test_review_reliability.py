# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""GitHub request policy and failure diagnostics, including the pinned PR-Agent runtime."""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import importlib.util
import inspect
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import yaml

from scripts import pr_agent_review as review
from scripts import review_followups as followups
from scripts import review_reliability as reliability

ROOT = Path(__file__).resolve().parents[1]
NOW = 1_800_000_000
SECRET = "SECRET-sentinel-ghp_0123456789"
SECONDARY = {"message": "You have exceeded a secondary rate limit. Please wait a few minutes before you try again."}
PRIMARY = {"message": "API rate limit exceeded for installation ID 1."}


class GH(Exception):
    """Duck-typed stand-in for a PyGithub exception: status, headers, data, message."""

    def __init__(self, status, headers=None, data=None, message=None):
        super().__init__(f"{message}: {status} {data}")
        self.status, self.headers, self.data, self.message = status, headers, data, message


def clock():
    return patch.object(reliability, "_now", lambda: NOW)


class RateHeaderTests(unittest.TestCase):
    def test_well_formed_headers_become_integers_with_a_reset_countdown(self):
        with clock():
            rate = reliability.rate_headers({
                "X-RateLimit-Limit": "1000", "x-ratelimit-remaining": "0", "X-RateLimit-Used": "1000",
                "x-ratelimit-reset": str(NOW + 90), "x-ratelimit-resource": "core", "Retry-After": "7"})
        self.assertEqual({"limit": 1000, "remaining": 0, "used": 1000, "reset": NOW + 90, "resource": "core",
                          "reset_in_seconds": 90, "retry_after_seconds": 7}, rate)

    def test_malformed_values_are_named_never_echoed_or_trusted(self):
        for bad in ("abc", "-1", "+5", "1e3", "1.5", "", " 5", "5 ", "١٢", "5_0", "0x10", "1" * 13,
                    "Wed, 21 Oct 2026 07:28:00 GMT", SECRET):
            with self.subTest(value=bad), clock():
                rate = reliability.rate_headers({"retry-after": bad, "x-ratelimit-remaining": bad,
                                                 "x-ratelimit-reset": bad, "x-ratelimit-limit": bad})
                self.assertEqual(["retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"],
                                 rate["malformed"])
                self.assertEqual({"malformed"}, set(rate))
                self.assertNotIn(SECRET, json.dumps(rate))
        with clock():
            for bad in ("Core", "core data", "core\n", "1core", "c" * 33, SECRET):
                self.assertEqual({"malformed": ["x-ratelimit-resource"]},
                                 reliability.rate_headers({"x-ratelimit-resource": bad}), bad)

    def test_missing_odd_or_hostile_header_containers_are_empty_not_errors(self):
        for headers in (None, {}, [], 5, "x", {"retry-after": 5}, {"retry-after": None}, {5: "7"}):
            with self.subTest(headers=headers), clock():
                self.assertEqual({}, reliability.rate_headers(headers))

    def test_only_allowlisted_headers_survive_so_credentials_never_travel(self):
        mapped = reliability.header_map({"Authorization": "Bearer " + SECRET, "Set-Cookie": SECRET,
                                         "X-GitHub-Request-Id": SECRET, "Retry-After": "7"})
        self.assertEqual({"retry-after": "7"}, mapped)


class ClassifyTests(unittest.TestCase):
    def test_categories_from_status_headers_and_the_librarys_rate_limit_wording(self):
        zero = {"x-ratelimit-remaining": "0"}
        cases = [
            (403, zero, None, "rate_limit_primary"),
            (403, {}, "API rate limit exceeded for user ID 1.", "rate_limit_primary"),
            (403, {}, "You have exceeded a secondary rate limit.", "rate_limit_secondary"),
            (403, {}, "Something. Please wait a few minutes before you try again.", "rate_limit_secondary"),
            (403, {"retry-after": "60"}, "Resource not accessible", "rate_limit_unknown"),
            (403, {"retry-after": "tomorrow"}, None, "rate_limit_unknown"),
            (429, {}, None, "rate_limit_unknown"),
            (429, zero, None, "rate_limit_primary"),
            (403, {}, "Resource not accessible by integration", "forbidden"),
            (401, {}, "Bad credentials", "authentication"), (404, {}, None, "not_found"),
            (422, {}, None, "validation"), (500, {}, None, "server_error"), (502, {}, None, "server_error"),
            (418, {}, None, "client_error"), (409, {}, None, "client_error"),
            (500, zero, "API rate limit exceeded", "server_error"),
        ]
        for status, headers, message, expected in cases:
            with self.subTest(status=status, headers=headers, message=message):
                self.assertEqual(expected, reliability.classify(status, headers, message))

    def test_a_message_is_inspected_but_never_returned(self):
        entry = reliability.diagnostic(GH(403, {}, {"message": SECRET}, SECRET), "context")
        self.assertEqual("forbidden", entry["category"])
        self.assertNotIn(SECRET, json.dumps(entry))


class PlanWaitTests(unittest.TestCase):
    def test_retry_after_then_reset_then_one_minute_doubling(self):
        self.assertEqual((7.0, "retry-after"), reliability.plan_wait({"retry-after": "7"}, 1, NOW))
        self.assertEqual((0.0, "retry-after"), reliability.plan_wait({"retry-after": "0"}, 1, NOW))
        # Retry-After outranks an exhausted primary limit's reset.
        self.assertEqual((7.0, "retry-after"), reliability.plan_wait(
            {"retry-after": "7", "x-ratelimit-remaining": "0", "x-ratelimit-reset": str(NOW + 999)}, 1, NOW))
        self.assertEqual((31.0, "ratelimit-reset"), reliability.plan_wait(
            {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(NOW + 30)}, 1, NOW))
        self.assertEqual((1.0, "ratelimit-reset"), reliability.plan_wait(
            {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(NOW - 500)}, 1, NOW))
        for attempt, seconds in ((1, 60.0), (2, 120.0), (3, 240.0)):
            self.assertEqual((seconds, "default-backoff"), reliability.plan_wait({}, attempt, NOW))
        # A non-exhausted primary limit says nothing about this throttle: one-minute floor.
        self.assertEqual((60.0, "default-backoff"), reliability.plan_wait(
            {"x-ratelimit-remaining": "12", "x-ratelimit-reset": str(NOW + 5)}, 1, NOW))

    def test_unusable_headers_are_unbounded_never_a_guessed_shorter_wait(self):
        reset = str(NOW + 30)
        for headers, code in (({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, "retry-after-malformed"),
                              ({"retry-after": "-1", "x-ratelimit-reset": reset}, "retry-after-malformed"),
                              ({"x-ratelimit-remaining": "0"}, "reset-missing-or-malformed"),
                              ({"x-ratelimit-remaining": "0", "x-ratelimit-reset": "soon"}, "reset-missing-or-malformed"),
                              ({"x-ratelimit-remaining": "none", "x-ratelimit-reset": reset}, "remaining-malformed")):
            with self.subTest(headers=headers), self.assertRaises(reliability.Unbounded) as caught:
                reliability.plan_wait(headers, 1, NOW)
            self.assertEqual(code, caught.exception.code)


class BudgetTests(unittest.TestCase):
    def test_reservations_stop_exactly_at_the_budget_and_failed_ones_spend_nothing(self):
        budget = reliability.WaitBudget(300)
        self.assertTrue(budget.reserve(60) and budget.reserve(120))
        self.assertFalse(budget.reserve(121))
        self.assertTrue(budget.reserve(120))
        self.assertFalse(budget.reserve(1))
        self.assertEqual(300.0, budget.spent)
        self.assertFalse(reliability.WaitBudget(0).reserve(0.5))

    def test_budget_is_one_provider_timeout_and_fits_the_review_job_limit(self):
        config = reliability.tomllib.loads((ROOT / ".github/pr-agent.toml").read_text())
        workflow = yaml.safe_load((ROOT / ".github/workflows/agent-review-reusable.yaml").read_text())
        wait, timeout, calls = reliability.budget_seconds(), config["config"]["ai_timeout"], \
            config["pr_reviewer"]["max_number_of_calls"]
        self.assertEqual(timeout, wait)
        for job in ("review", "reply"):
            # Worst case: every completion call times out and every wait is spent.
            self.assertLess(wait + calls * timeout, workflow["jobs"][job]["timeout-minutes"] * 60, job)
        self.assertEqual(1, config["github"]["seconds_between_writes"])
        self.assertEqual(0, config["github"]["api_retries"])
        self.assertEqual(1, config["github"]["ratelimit_retries"])


class EndpointKindTests(unittest.TestCase):
    def test_labels_come_from_structure_never_from_names_or_paths(self):
        for url, kind in (("/repos/o/r/pulls/1/files?per_page=30&page=2", "repos/pulls/files"),
                          (f"/repos/o/r/contents/src/{SECRET}.py?ref={SECRET}", "repos/contents"),
                          ("/repos/o/r/compare/aaa...bbb", "repos/compare"),
                          ("/repos/o/r/actions/runs/9/artifacts", "repos/actions/runs/artifacts"),
                          ("/orgs/acme/memberships/someone", "orgs/memberships"),
                          ("/rate_limit", "rate_limit"), (f"/{SECRET}/x", "other"), ("", "other")):
            with self.subTest(url=url):
                self.assertEqual(kind, reliability.endpoint_kind(url))
        self.assertEqual("other", reliability.endpoint_kind(None))


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        reliability.begin(300)

    def test_phase_contexts_restore_and_the_innermost_failing_phase_wins(self):
        reliability.advance("context")
        with reliability.phase("publication"):
            self.assertEqual("publication", reliability.OBSERVER.phase)
        self.assertEqual("context", reliability.OBSERVER.phase)
        with self.assertRaises(RuntimeError):
            with reliability.phase("review"):
                with reliability.phase("inference"):
                    raise RuntimeError(SECRET)
        self.assertEqual("context", reliability.OBSERVER.phase)

    def test_a_recovered_failure_does_not_mislabel_a_later_one(self):
        # The billing fallback catches a rejection that crossed the inference phase.
        reliability.advance("review")
        try:
            with reliability.phase("inference"):
                raise RuntimeError("recovered by the next credential slot")
        except RuntimeError:
            pass
        reliability.advance("completion")
        later = review.ReviewAbort("Completion evidence is incomplete")
        self.assertEqual("completion", reliability.diagnostic(later)["phase"])
        try:
            with reliability.phase("publication"):
                raise OSError("write failed")
        except OSError as error:
            self.assertEqual("publication", reliability.diagnostic(error)["phase"])
        # The innermost phase an exception crosses is its phase, even through a wrapper.
        try:
            with reliability.phase("review"):
                try:
                    with reliability.phase("inference"):
                        raise RuntimeError(SECRET)
                except RuntimeError:
                    raise review.ReviewAbort("Provider failed; ambiguous requests are never retried") from None
        except review.ReviewAbort as error:
            self.assertEqual("review", reliability.failing_phase(error))
            self.assertEqual("review", reliability.diagnostic(error)["phase"])
        self.assertIsNone(reliability.failing_phase(RuntimeError()))

    def test_a_status_on_a_model_call_error_is_never_read_as_github(self):
        # litellm wraps aiohttp errors that carry .status and .headers like GitHub's own.
        class ClientResponseError(Exception):
            status, headers, message = 429, {"retry-after": "5"}, SECRET

        try:
            with reliability.phase("inference"):
                try:
                    raise ClientResponseError()
                except Exception:
                    raise RuntimeError("provider wrapper") from None
        except RuntimeError as error:
            caught = error
        entry = reliability.diagnostic(caught)
        self.assertEqual(("inference", 429), (entry["phase"], entry["provider_status"]))
        for key in ("status", "rate_limit", "decision"):
            self.assertNotIn(key, entry)
        self.assertNotIn("rate_limit", entry["category"])
        self.assertFalse(reliability.throttled(caught))
        self.assertNotIn(SECRET, json.dumps(entry))

    def test_github_error_is_found_through_the_adapters_wrapping(self):
        reliability.advance("provider_init")
        try:
            try:
                raise GH(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(NOW + 5),
                               "authorization": SECRET}, PRIMARY, "API rate limit exceeded")
            except GH:
                raise review.ReviewAbort(f"PR-Agent review failed ({SECRET})") from None
        except review.ReviewAbort as error:
            caught = error
        with clock():
            entry = reliability.diagnostic(caught)
        self.assertEqual({"phase": "provider_init", "category": "rate_limit_primary", "status": 403,
                          "exception": "GH", "rate_limit": {"remaining": 0, "reset": NOW + 5, "reset_in_seconds": 5}}, entry)
        self.assertNotIn(SECRET, json.dumps(entry))
        self.assertTrue(reliability.throttled(caught))

    def test_throttled_is_false_for_every_other_failure(self):
        for error in (GH(404), GH(403, {}, {"message": "Resource not accessible"}), GH(500),
                      RuntimeError("API rate limit exceeded"), ConnectionError(), review.ReviewAbort("x"),
                      ValueError()):
            with self.subTest(error=error):
                self.assertFalse(reliability.throttled(error))
        self.assertTrue(reliability.throttled(GH(429)))
        self.assertTrue(reliability.throttled(reliability.Throttled(403, {"x-ratelimit-remaining": "0"})))

    def test_non_github_failures_get_fixed_categories_without_messages(self):
        self.assertEqual("network", reliability.diagnostic(ConnectionError(SECRET))["category"])
        self.assertEqual("interrupted", reliability.diagnostic(KeyboardInterrupt())["category"])
        unexpected = reliability.diagnostic(ValueError(SECRET))
        self.assertEqual(("unexpected", "ValueError"), (unexpected["category"], unexpected["exception"]))
        self.assertNotIn(SECRET, json.dumps(unexpected))
        abort = reliability.diagnostic(review.ReviewAbort("fixed \x1b[31mred\n" + "x" * 500))
        self.assertEqual("review_abort", abort["category"])
        self.assertNotIn("detail", abort)
        foreign_abort = type("ProviderAbort", (Exception,), {})(SECRET)
        self.assertNotIn(SECRET, json.dumps(reliability.diagnostic(foreign_abort)))

    def test_a_model_providers_status_is_kept_apart_from_github_status(self):
        provider = type("APIStatusError", (Exception,), {})(SECRET)
        provider.status_code = 429
        try:
            with reliability.phase("inference"):
                try:
                    raise provider
                except Exception:
                    raise review.ReviewAbort("Provider failed; ambiguous requests are never retried") from None
        except review.ReviewAbort as error:
            caught = error
        entry = reliability.diagnostic(caught)
        self.assertEqual(("inference", "review_abort", 429), (entry["phase"], entry["category"], entry["provider_status"]))
        self.assertNotIn("status", entry)
        self.assertFalse(reliability.throttled(caught))

    def test_nothing_sensitive_reaches_the_outcome_summary_or_log_line(self):
        error = GH(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "bad", "authorization": SECRET,
                         "x-ratelimit-resource": SECRET, "set-cookie": SECRET},
                   {"message": SECRET, "documentation_url": f"https://x.example/?key={SECRET}"}, SECRET)
        with tempfile.TemporaryDirectory() as directory:
            env = {"GITHUB_RUN_ID": "99", "GITHUB_EVENT_NAME": "issue_comment",
                   "REVIEW_OUTCOME_PATH": f"{directory}/o/outcome.json", "GITHUB_STEP_SUMMARY": f"{directory}/s.md"}
            stream = io.StringIO()
            result = reliability.report(error, env, stream)
            written = Path(env["REVIEW_OUTCOME_PATH"]).read_text()
            summary = Path(env["GITHUB_STEP_SUMMARY"]).read_text()
        self.assertEqual(result, json.loads(written))
        for text in (written, summary, stream.getvalue(), json.dumps(result)):
            for hostile in (SECRET, "x.example", "ghp_", "authorization"):
                self.assertNotIn(hostile, text)
        self.assertIn("Review diagnostics: ", stream.getvalue())
        self.assertEqual(["x-ratelimit-reset", "x-ratelimit-resource"], result["failure"]["rate_limit"]["malformed"])
        self.assertEqual((99, "issue_comment"), (result["run_id"], result["event"]))


class OutcomeFileTests(unittest.TestCase):
    def setUp(self):
        reliability.begin(300)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = {"REVIEW_OUTCOME_PATH": f"{self.directory.name}/nested/outcome.json",
                    "GITHUB_STEP_SUMMARY": f"{self.directory.name}/summary.md"}

    def test_success_records_waits_without_a_failure_or_log_line(self):
        reliability.OBSERVER.throttle("GET", "/repos/o/r/pulls/1", 403, {"retry-after": "7"}, "wait", 7.0)
        stream = io.StringIO()
        result = reliability.report(None, self.env, stream)
        self.assertEqual("succeeded", result["result"])
        self.assertNotIn("failure", result)
        self.assertEqual({"count": 1, "seconds": 7}, result["github"]["waits"])
        self.assertEqual(300, result["github"]["wait_budget_seconds"])
        self.assertEqual("", stream.getvalue())
        self.assertEqual(0o600, Path(self.env["REVIEW_OUTCOME_PATH"]).stat().st_mode & 0o777)
        self.assertIn("1 waits, 7 s of 300 s budget", Path(self.env["GITHUB_STEP_SUMMARY"]).read_text())

    def test_summary_is_a_fixed_table_with_no_markup_injection(self):
        error = review.ReviewAbort("a | b\n# heading `code` <script>")
        reliability.report(error, self.env, io.StringIO())
        lines = Path(self.env["GITHUB_STEP_SUMMARY"]).read_text().strip().splitlines()
        self.assertEqual("### Agent review outcome", lines[0])
        self.assertTrue(all(line.startswith("| ") for line in lines[2:]), lines)
        text = "\n".join(lines)
        for markup in ("`", "<", "#heading"):
            self.assertNotIn(markup, text.split("\n", 1)[1])

    def test_unwritable_sinks_never_raise_and_events_and_run_ids_are_allowlisted(self):
        os.mkdir(f"{self.directory.name}/blocked")
        env = {"REVIEW_OUTCOME_PATH": f"{self.directory.name}/blocked", "GITHUB_STEP_SUMMARY": f"{self.directory.name}/blocked",
               "GITHUB_RUN_ID": "0", "GITHUB_EVENT_NAME": "workflow_dispatch"}
        result = reliability.report(RuntimeError(), env, io.StringIO())
        self.assertNotIn("run_id", result)
        self.assertNotIn("event", result)
        self.assertEqual("failed", result["result"])
        for run_id in ("007", "-1", "1e3", "٣", "", "1" * 20):
            self.assertNotIn("run_id", reliability.outcome(None, {"GITHUB_RUN_ID": run_id}), run_id)


class AdapterMainTests(unittest.TestCase):
    """main() records the outcome and keeps its exit status and fixed stderr message."""

    def run_main(self, fake, argv=("pr_agent_review.py",)):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = {"REVIEW_OUTCOME_PATH": f"{self.directory.name}/outcome.json",
               "GITHUB_STEP_SUMMARY": f"{self.directory.name}/summary.md", "GITHUB_RUN_ID": "5"}
        stderr = io.StringIO()
        with (patch.dict(os.environ, env), patch.object(sys, "argv", list(argv)), patch("sys.stderr", stderr),
              patch.object(review, "configure", lambda: (NS(), NS())), patch.object(review, "print_settings"),
              patch.object(review.reliability, "runtime", contextlib.nullcontext), patch.object(review, "review", fake)):
            try:
                code = review.main()
            except KeyboardInterrupt:
                code = "interrupted"
        outcome = json.loads(Path(env["REVIEW_OUTCOME_PATH"]).read_text()) if Path(env["REVIEW_OUTCOME_PATH"]).exists() else None
        return code, stderr.getvalue(), outcome

    def test_github_throttle_during_provider_init_is_diagnosed_without_the_raw_error(self):
        async def fake(settings):
            reliability.advance("provider_init")
            raise GH(403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(NOW + 3000)}, PRIMARY, SECRET)

        code, stderr, outcome = self.run_main(fake)
        self.assertEqual(1, code)
        self.assertTrue(stderr.startswith("Review failed: GH\n"), stderr)
        self.assertEqual(("failed", "provider_init", "rate_limit_primary", 403),
                         (outcome["result"], outcome["failure"]["phase"], outcome["failure"]["category"],
                          outcome["failure"]["status"]))
        self.assertNotIn(SECRET, stderr + json.dumps(outcome))
        self.assertIn("rate_limit_primary", Path(self.directory.name, "summary.md").read_text())

    def test_abort_keeps_its_fixed_message_and_adds_the_phase(self):
        async def fake(settings):
            with reliability.phase("publication"):
                raise review.ReviewAbort("Review summary publication failed")

        code, stderr, outcome = self.run_main(fake)
        self.assertEqual(1, code)
        self.assertTrue(stderr.startswith("Review failed: Review summary publication failed\n"), stderr)
        self.assertEqual(("publication", "review_abort"), (outcome["failure"]["phase"], outcome["failure"]["category"]))

    def test_success_and_interruption_are_recorded_too(self):
        async def fine(settings):
            return None

        code, stderr, outcome = self.run_main(fine)
        self.assertEqual((0, "", "succeeded"), (code, stderr, outcome["result"]))
        code, stderr, outcome = self.run_main(fine, argv=("pr_agent_review.py", "--check-config"))
        self.assertIsNone(outcome)

        async def interrupted(settings):
            raise KeyboardInterrupt

        code, _, outcome = self.run_main(interrupted)
        self.assertEqual(("interrupted", "interrupted"), (code, outcome["failure"]["category"]))

    def test_an_unwritable_outcome_never_changes_the_exit_status(self):
        async def fake(settings):
            raise RuntimeError(SECRET)

        stderr = io.StringIO()
        with (patch.object(reliability, "report", side_effect=OSError(SECRET)), patch.object(sys, "argv", ["x"]),
              patch("sys.stderr", stderr), patch.object(review, "configure", lambda: (NS(), NS())),
              patch.object(review, "print_settings"), patch.object(review.reliability, "runtime", contextlib.nullcontext),
              patch.object(review, "review", fake)):
            self.assertEqual(1, review.main())
        self.assertIn("Review failed: RuntimeError", stderr.getvalue())
        self.assertIn("Review outcome could not be recorded", stderr.getvalue())
        self.assertNotIn(SECRET, stderr.getvalue())


class FollowupThrottleTests(unittest.TestCase):
    """A rate limit is never mistaken for missing proof, which would spend inference."""

    def test_unreadable_comments_fall_back_but_a_throttle_propagates(self):
        def pr_raising(error):
            return NS(get_issue_comments=lambda: (_ for _ in ()).throw(error))

        self.assertEqual({}, followups.verified_records(None, pr_raising(RuntimeError()), {}, None))
        with self.assertRaises(GH):
            followups.verified_records(None, pr_raising(GH(429)), {}, None)

    def test_comparison_failures_fall_back_to_full_review_but_a_throttle_propagates(self):
        current = {"base": "b" * 40, "head": "a" * 40}
        records = {1: {"scope": {"kind": "full", "parent": None}, "head": "c" * 40}}

        def repo_raising(error):
            def compare(*args):
                raise error
            return NS(compare=compare)

        self.assertEqual("full", followups.scope_for(repo_raising(RuntimeError()), current, records)["kind"])
        with self.assertRaises(GH):
            followups.scope_for(repo_raising(GH(403, {"x-ratelimit-remaining": "0"})), current, records)

    def test_artifact_secondary_limit_preserves_classification_without_server_text(self):
        artifact = NS(archive_download_url="https://api.github.com/repos/o/r/actions/artifacts/1/zip", raw_data={})
        message = "You have exceeded a secondary rate limit. private-response-marker"
        requests = NS(get=lambda *a, **k: NS(status_code=403, headers={}, json=lambda: {"message": message}))
        with patch.dict(sys.modules, {"requests": requests}), self.assertRaises(reliability.Throttled) as caught:
            followups.download_record(artifact, "token", 5)
        self.assertTrue(reliability.throttled(caught.exception))
        self.assertEqual("rate_limit_secondary", reliability.diagnostic(caught.exception)["category"])
        self.assertNotIn("private-response-marker", repr(vars(caught.exception)))
        self.assertNotIn("private-response-marker", str(caught.exception))

    def test_artifact_error_body_cannot_hide_header_throttle(self):
        artifact = NS(archive_download_url="https://api.github.com/repos/o/r/actions/artifacts/1/zip", raw_data={})
        def invalid_json():
            raise ValueError("invalid JSON")
        for decode in (invalid_json, lambda: [], lambda: {"message": 42}):
            for status, headers, expected in ((403, {}, followups.FollowupAbort),
                                              (429, {}, reliability.Throttled),
                                              (403, {"retry-after": "5"}, reliability.Throttled)):
                with self.subTest(status=status, decode=decode):
                    response = NS(status_code=status, headers=headers, json=decode)
                    with patch.dict(sys.modules, {"requests": NS(get=lambda *a, **k: response)}), self.assertRaises(expected):
                        followups.download_record(artifact, "token", 5)

    def test_artifact_download_distinguishes_throttle_from_unavailable(self):
        artifact = NS(archive_download_url="https://api.github.com/repos/o/r/actions/artifacts/1/zip", raw_data={})
        for status, headers, expected in ((429, {}, reliability.Throttled),
                                          (403, {"x-ratelimit-remaining": "0"}, reliability.Throttled),
                                          (403, {"retry-after": "5"}, reliability.Throttled),
                                          (403, {}, followups.FollowupAbort), (404, {}, followups.FollowupAbort),
                                          (200, {}, followups.FollowupAbort)):
            with self.subTest(status=status, headers=headers):
                requests = NS(get=lambda *a, **k: NS(status_code=status, headers=headers, json=lambda: {"message": "Forbidden"}))
                with patch.dict(sys.modules, {"requests": requests}), self.assertRaises(expected):
                    followups.download_record(artifact, "token", 5)


def _pinned_runtime():
    return importlib.util.find_spec("pr_agent") is not None


class FakeGitHub:
    """Loopback GitHub API: scripted responses, request log, and a concurrency high-water mark."""

    def __init__(self, respond):
        self.log, self.inflight, self.max_inflight, self.lock = [], 0, 0, threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def handle_any(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                with outer.lock:
                    outer.inflight += 1
                    outer.max_inflight = max(outer.max_inflight, outer.inflight)
                    outer.log.append((self.command, self.path.split("?")[0], time.monotonic(), body))
                    attempt = len(outer.log)
                try:
                    threading.Event().wait(0.02)  # widen any overlap; unaffected by patching time.sleep
                    status, headers, payload = respond(self.command, self.path.split("?")[0], attempt)
                    data = json.dumps(payload).encode()
                    self.send_response(status)
                    for name, value in headers.items():
                        self.send_header(name, value)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                finally:
                    with outer.lock:
                        outer.inflight -= 1

            do_GET = do_POST = do_PATCH = do_PUT = do_DELETE = do_HEAD = handle_any

        class Quiet(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # a client closing a keep-alive connection is not a test failure

        self.server = Quiet(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def methods(self):
        return [(method, path) for method, path, _, _ in self.log]


@unittest.skipUnless(_pinned_runtime(), "requires disposable pr-agent==0.47.0 environment")
class PinnedGitHubPolicyTests(unittest.TestCase):
    """The real PyGithub 2.10.0 requester, retry and exception classes against a loopback server."""

    def setUp(self):
        from github import Auth
        self.auth = Auth.Token("fixture-token")
        self.sleeps = []
        for item in (patch("urllib3.util.retry.time.sleep", side_effect=self.sleeps.append), clock()):
            item.start()
            self.addCleanup(item.stop)

    def client(self, server, budget=300):
        reliability.begin(budget)
        return reliability.github_client(self.auth, server.url, seconds_between_requests=0, seconds_between_writes=1)

    def throttling(self, *responses, then=(200, {}, {"full_name": "o/r", "id": 1})):
        queue = list(responses)
        return lambda method, path, attempt: queue.pop(0) if queue else then

    def get_repo(self, client):
        return client.get_repo("o/r")

    def test_retry_after_is_honored_for_a_get_and_recorded(self):
        for status, body in ((403, SECONDARY), (403, PRIMARY), (429, {"message": "slow down"})):
            with self.subTest(status=status), FakeGitHub(self.throttling((status, {"Retry-After": "7"}, body))) as server:
                self.sleeps.clear()
                client = self.client(server)
                self.assertEqual("o/r", self.get_repo(client).full_name)
                self.assertEqual([7.0], self.sleeps)
                self.assertEqual(2, len(server.log))
                self.assertEqual((1, 7), (reliability.OBSERVER.waits, int(reliability.OBSERVER.budget.spent)))

    def test_exhausted_primary_limit_waits_for_its_reset_when_the_budget_allows(self):
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Limit": "1000", "X-RateLimit-Reset": str(NOW + 30)}
        with FakeGitHub(self.throttling((403, headers, PRIMARY))) as server:
            self.assertEqual("o/r", self.get_repo(self.client(server)).full_name)
        self.assertEqual([31.0], self.sleeps)

    def test_reset_beyond_the_budget_fails_fast_with_diagnostics_and_no_second_request(self):
        from github.GithubException import RateLimitExceededExceedsMaxWait
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Limit": "1000", "X-RateLimit-Used": "1000",
                   "X-RateLimit-Reset": str(NOW + 3600), "X-RateLimit-Resource": "core"}
        with FakeGitHub(self.throttling((403, headers, PRIMARY), (403, headers, PRIMARY))) as server:
            client = self.client(server)
            with self.assertRaises(RateLimitExceededExceedsMaxWait) as caught:
                client.get_repo("o/r")
        self.assertEqual([], self.sleeps)
        self.assertEqual(1, len(server.log))
        self.assertEqual(3601.0, caught.exception.wait)
        entry = reliability.diagnostic(caught.exception, "provider_init")
        self.assertEqual(("rate_limit_primary", 403, "exceeds_budget"),
                         (entry["category"], entry["status"], entry["decision"]))
        self.assertEqual({"limit": 1000, "remaining": 0, "used": 1000, "reset": NOW + 3600, "resource": "core",
                          "reset_in_seconds": 3600}, entry["rate_limit"])
        self.assertEqual("GET repos", reliability.OBSERVER.last["endpoint"])

    def test_missing_rate_headers_wait_a_minute_then_double_within_the_budget(self):
        from github.GithubException import RateLimitExceededExceedsMaxWait
        always = lambda method, path, attempt: (403, {}, SECONDARY)
        with FakeGitHub(always) as server, self.assertRaises(RateLimitExceededExceedsMaxWait) as caught:
            self.client(server).get_repo("o/r")
        # 60 + 120 spent; the third wait (240) would pass the 300 s budget.
        self.assertEqual([60.0, 120.0], self.sleeps)
        self.assertEqual(3, len(server.log))
        self.assertEqual(240.0, caught.exception.wait)
        self.assertEqual("rate_limit_secondary", reliability.diagnostic(caught.exception)["category"])

    def test_retries_stop_at_the_librarys_total_and_surface_the_real_exception(self):
        from github.GithubException import RateLimitExceededException
        always = lambda method, path, attempt: (403, {"Retry-After": "1"}, SECONDARY)
        with FakeGitHub(always) as server, self.assertRaises(RateLimitExceededException) as caught:
            self.client(server).get_repo("o/r")
        self.assertEqual(reliability.RETRY_TOTAL + 1, len(server.log))
        self.assertEqual([1.0] * reliability.RETRY_TOTAL, self.sleeps)
        # The last response, not a urllib3 "too many error responses" wrapper that embeds the URL.
        self.assertEqual("rate_limit_secondary", reliability.diagnostic(caught.exception)["category"])

    def test_a_429_without_headers_also_waits_a_minute(self):
        with FakeGitHub(self.throttling((429, {}, {"message": "slow down"}))) as server:
            self.assertEqual("o/r", self.get_repo(self.client(server)).full_name)
        self.assertEqual([60.0], self.sleeps)

    def test_malformed_headers_are_never_retried_and_are_named(self):
        from github.GithubException import RateLimitExceededException
        # Header values are latin-1 on the wire, so non-ASCII digits are covered in RateHeaderTests.
        hostile = ["Wed, 21 Oct 2026 07:28:00 GMT", "abc", "-5", "1e3", "", "5.0"]
        for value in hostile:
            for headers in ({"Retry-After": value},
                            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": value},
                            {"X-RateLimit-Remaining": value}):
                with self.subTest(headers=headers), FakeGitHub(self.throttling((403, headers, SECONDARY))) as server:
                    self.sleeps.clear()
                    with self.assertRaises(RateLimitExceededException) as caught:
                        self.client(server).get_repo("o/r")
                    self.assertEqual([], self.sleeps)
                    self.assertEqual(1, len(server.log))
                    self.assertEqual("unbounded", reliability.OBSERVER.decision)
                    self.assertIn("malformed", reliability.diagnostic(caught.exception)["rate_limit"])

    def test_other_failures_are_not_retried(self):
        cases = [(403, {}, {"message": "Resource not accessible by integration"}),
                 (404, {}, {"message": "Not Found"}), (500, {}, {}), (502, {"Retry-After": "1"}, {}),
                 (503, {"Retry-After": "1"}, {}), (401, {}, {"message": "Bad credentials"}),
                 (422, {}, {"message": "Validation Failed"})]
        for status, headers, body in cases:
            with self.subTest(status=status), FakeGitHub(lambda m, p, a, s=(status, headers, body): s) as server:
                with self.assertRaises(Exception):
                    self.client(server).get_repo("o/r")
                self.assertEqual(1, len(server.log))
        self.assertEqual([], self.sleeps)

    def test_a_write_is_never_repeated_whatever_the_failure(self):
        from github.GithubException import GithubException
        cases = [(403, {"Retry-After": "1"}, SECONDARY), (403, {}, PRIMARY), (429, {"Retry-After": "1"}, {}),
                 (500, {}, {}), (502, {"Retry-After": "1"}, {}), (503, {}, {}), (403, {"X-RateLimit-Remaining": "0",
                 "X-RateLimit-Reset": str(NOW + 5)}, PRIMARY), (422, {}, {"message": "Validation Failed"})]
        for verb in ("POST", "PATCH", "PUT", "DELETE"):
            for status, headers, body in cases:
                with self.subTest(verb=verb, status=status), FakeGitHub(lambda m, p, a, s=(status, headers, body): s) as server:
                    client = self.client(server)
                    with self.assertRaises(GithubException) as caught:
                        client.requester.requestJsonAndCheck(verb, "/repos/o/r/issues/1/comments", input={"body": "x"})
                    self.assertEqual(status, caught.exception.status)
                    self.assertEqual(1, len(server.log), (verb, status))
        self.assertEqual([], self.sleeps)

    def test_real_publication_calls_are_not_repeated(self):
        from github.GithubException import GithubException
        root = "/repos/o/r"
        pr = {"url": f"{root}/pulls/1", "issue_url": f"{root}/issues/1", "number": 1,
              "head": {"sha": "a" * 40, "ref": "x"}, "base": {"sha": "b" * 40, "ref": "main"}}

        def respond(method, path, attempt):
            if method == "GET":
                return 200, {}, pr if path.endswith("/pulls/1") else {"full_name": "o/r", "id": 1}
            return 403, {"Retry-After": "1"}, SECONDARY

        with FakeGitHub(respond) as server:
            pull = self.client(server).get_repo("o/r").get_pull(1)
            for call in (lambda: pull.create_issue_comment("x"),
                         lambda: pull.create_review(body="x", event="COMMENT")):
                with self.assertRaises(GithubException):
                    call()
            writes = [m for m, _ in server.methods() if m != "GET"]
        self.assertEqual(["POST", "POST"], writes)
        # Only the client's own one-second spacing between the two writes slept.
        self.assertEqual(0, reliability.OBSERVER.waits)
        self.assertEqual(1, len(self.sleeps))
        self.assertLessEqual(self.sleeps[0], 1.0)

    def test_policy_matches_the_pinned_librarys_own_contracts(self):
        from github import Github
        from github.Requester import Requester
        self.assertEqual(sys.modules["github.GithubRetry"].DEFAULT_SECONDARY_RATE_WAIT, reliability.MIN_THROTTLE_WAIT)
        default = inspect.signature(Github.__init__).parameters["retry"].default
        self.assertEqual(default.total, reliability.RETRY_TOTAL)
        for message in ("API rate limit exceeded for 1", "api rate limit exceeded", "You have exceeded a secondary rate limit",
                        "x please retry your request again later.", "x. Please wait a few minutes before you try again.",
                        "Not Found", "", None, "rate limit", "secondary rate limit exceeded"):
            with self.subTest(message=message):
                expected = ("primary" if Requester.isPrimaryRateLimitError(message)
                            else "secondary" if Requester.isSecondaryRateLimitError(message) else None)
                self.assertEqual(expected, reliability.message_kind(message))

    def test_the_stock_retry_it_replaces_would_have_repeated_writes_and_server_errors(self):
        # Why the adapter does not use the provider's api_retries: GithubRetry's defaults
        # allow POST and every 5xx. Documented here against the real class.
        from github import GithubRetry
        stock = GithubRetry(total=3)
        self.assertIn("POST", stock.allowed_methods)
        self.assertTrue(set(range(500, 600)) <= set(stock.status_forcelist))
        ours = reliability.read_retry(reliability.WaitBudget(0))
        self.assertEqual({"GET", "HEAD"}, set(ours.allowed_methods))
        self.assertEqual({403, 429}, set(ours.status_forcelist))
        self.assertFalse(ours.respect_retry_after_header)
        self.assertEqual((10, 0, False), (ours.total, ours.connect, ours.read))


@unittest.skipUnless(_pinned_runtime(), "requires disposable pr-agent==0.47.0 environment")
class PinnedGitHubPacingTests(unittest.TestCase):
    """Real time: the client's own pacing is what spaces mutations."""

    def test_requests_are_serial_and_mutations_are_spaced_by_at_least_a_second(self):
        from github import Auth
        with FakeGitHub(lambda m, p, a: (200, {}, {"full_name": "o/r", "id": 1})) as server:
            reliability.begin(300)
            client = reliability.github_client(Auth.Token("fixture-token"), server.url,
                                               seconds_between_requests=0, seconds_between_writes=1)
            for _ in range(5):
                client.get_repo("o/r")
            for _ in range(2):
                client.requester.requestJsonAndCheck("POST", "/repos/o/r/issues/1/comments", input={"body": "x"})
        self.assertEqual(1, server.max_inflight)
        posts = [t for method, _, t, _ in server.log if method == "POST"]
        self.assertGreaterEqual(posts[1] - posts[0], 0.95)


@unittest.skipUnless(_pinned_runtime(), "requires disposable pr-agent==0.47.0 environment")
class PinnedRuntimeInstallTests(unittest.TestCase):
    def setUp(self):
        self.settings, _ = review.configure()
        self.addCleanup(self.settings.set, "github.base_url", "https://api.github.com")
        self.addCleanup(self.settings.set, "github.user_token", "")

    def test_provider_client_ignores_api_retries_observes_swallowed_errors_and_is_restored(self):
        from github.Requester import Requester
        from pr_agent.git_providers.github_provider import GithubProvider
        original_client, original_error = GithubProvider._get_github_client, Requester.__dict__["createException"].__func__
        levels = {name: logging.getLogger(name).level for name in ("github", "urllib3")}
        with FakeGitHub(lambda m, p, a: (404, {}, {"message": "Not Found"}) if m == "GET" else (502, {}, {})) as server:
            self.settings.set("github.user_token", "fixture-token")
            self.settings.set("github.base_url", server.url)
            self.settings.set("github.api_retries", 5)  # the stock provider would then retry POST and 5xx
            with patch("urllib3.util.retry.time.sleep"), clock():
                reliability.begin(300)
                with reliability.runtime():
                    for name in levels:
                        self.assertGreaterEqual(logging.getLogger(name).getEffectiveLevel(), logging.ERROR)
                    provider = GithubProvider.__new__(GithubProvider)
                    provider.installation_id, provider.base_url = None, server.url
                    client = provider._get_github_client()
                    with self.assertRaises(Exception):
                        client.get_repo("o/r")
                    with self.assertRaises(Exception):
                        client.requester.requestJsonAndCheck("POST", "/repos/o/r/issues/1/comments", input={"body": "x"})
                    seen = reliability.OBSERVER.snapshot()
        self.assertEqual([("GET", "/repos/o/r"), ("POST", "/repos/o/r/issues/1/comments")], server.methods())
        self.assertEqual(2, seen["errors"])
        self.assertEqual(("not_found", "server_error"), (seen["first"]["category"], seen["last"]["category"]))
        self.assertEqual(original_client, GithubProvider._get_github_client)
        self.assertIs(original_error, Requester.__dict__["createException"].__func__)
        self.assertEqual(levels, {name: logging.getLogger(name).level for name in levels})

    def test_the_provider_does_not_repeat_a_throttled_diff_read_on_its_own(self):
        # Upstream wraps get_diff_files in retry_call(tries=GITHUB.RATELIMIT_RETRIES, default 5,
        # sleeping 2-16 s between tries). Each try re-reads the pull request from a GitHub that
        # just refused, outside the wait budget and after the policy decided to stop.
        from pr_agent.git_providers.github_provider import GithubProvider
        from pr_agent.servers.utils import RateLimitExceeded
        calls = []

        def throttled_read(provider):
            calls.append(1)
            raise RateLimitExceeded("Rate limit exceeded for GitHub API.")

        provider = GithubProvider.__new__(GithubProvider)
        with (patch.object(GithubProvider, "_get_diff_files", throttled_read),
              patch("retry.api.time.sleep") as slept, self.assertRaises(RateLimitExceeded)):
            provider.get_diff_files()
        self.assertEqual(1, len(calls))
        slept.assert_not_called()

    def test_throttled_provider_init_stops_before_any_model_call_and_leaves_an_outcome(self):
        from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler  # noqa: F401
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Limit": "1000", "X-RateLimit-Resource": "core",
                   "X-RateLimit-Reset": str(NOW + 3000), "X-GitHub-Request-Id": SECRET}
        body = {"message": "API rate limit exceeded for installation ID 1.", "documentation_url": SECRET}
        completion = AsyncMock(side_effect=AssertionError("a model call was made"))
        with FakeGitHub(lambda m, p, a: (403, headers, body)) as server, tempfile.TemporaryDirectory() as directory:
            env = {"REVIEW_SOURCE_SHA": "c" * 40, "REVIEW_PROFILE_SHA": "d" * 40, "GITHUB_RUN_ID": "123",
                   "GITHUB_REPOSITORY": "example/project", "PR_NUMBER": "1", "REVIEW_HEAD": "a" * 40,
                   "REVIEW_BASE": "b" * 40, "GITHUB_TOKEN": "fixture-github", "ANTHROPIC_API_KEY": "",
                   "ANTHROPIC_API_KEY_2": "", "OPENAI_API_KEY": "fixture-openai",
                   "GITHUB_EVENT_NAME": "issue_comment", "GITHUB_ACTIONS": "true",
                   "REVIEW_OUTCOME_PATH": f"{directory}/outcome/outcome.json",
                   "GITHUB_STEP_SUMMARY": f"{directory}/summary.md"}
            stderr, stdout = io.StringIO(), io.StringIO()
            with (patch.dict(os.environ, env), patch.object(sys, "argv", ["pr_agent_review.py"]),
                  patch("sys.stderr", stderr), patch("sys.stdout", stdout), clock(),
                  patch("pr_agent.algo.ai_handlers.litellm_ai_handler.LiteLLMAIHandler._get_completion", completion),
                  patch("urllib3.util.retry.time.sleep") as slept):
                self.settings.set("github.base_url", server.url)
                code = review.main()
            outcome = json.loads(Path(env["REVIEW_OUTCOME_PATH"]).read_text())
            summary = Path(env["GITHUB_STEP_SUMMARY"]).read_text()
        self.assertEqual(1, code)
        completion.assert_not_awaited()
        slept.assert_not_called()
        self.assertEqual(1, len(server.log), server.methods())
        self.assertNotIn("POST", [m for m, _ in server.methods()])
        self.assertEqual(("failed", "provider_init", "rate_limit_primary", 403),
                         (outcome["result"], outcome["failure"]["phase"], outcome["failure"]["category"],
                          outcome["failure"]["status"]))
        self.assertEqual({"limit": 1000, "remaining": 0, "reset": NOW + 3000, "resource": "core", "reset_in_seconds": 3000},
                         outcome["failure"]["rate_limit"])
        self.assertEqual("exceeds_budget", outcome["failure"]["decision"])
        self.assertEqual("GET repos", outcome["github"]["last"]["endpoint"])
        self.assertEqual((1, 1), (outcome["github"]["errors"], outcome["github"]["rate_limited"]))
        for text in (stderr.getvalue(), stdout.getvalue(), summary, json.dumps(outcome)):
            for hostile in (SECRET, "fixture-github", "fixture-openai", "127.0.0.1", "example/project"):
                self.assertNotIn(hostile, text)
        self.assertIn("Review failed: ", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
