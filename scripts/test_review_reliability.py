# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""GitHub request policy and failure diagnostics, for the GitHub API client."""

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

from scripts import review_engine as review
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
        wait, timeout = reliability.budget_seconds(), config["config"]["ai_timeout"]
        self.assertEqual(timeout, wait)
        # Provider calls per job: review runs its passes plus verification, reply one, publishers none.
        calls = {"review": len(review.DEPTHS["default"][1]) + 1, "reply": 1, "publish": 0, "publish-reply": 0}
        for job, count in calls.items():
            # Worst case: every completion call times out and every wait is spent.
            self.assertLess(wait + count * timeout, workflow["jobs"][job]["timeout-minutes"] * 60, job)



class ArtifactTimeoutTests(unittest.TestCase):
    def test_artifact_downloads_use_their_own_bounded_timeout(self):
        # Two sequential requests per record must stay small next to the review job limit,
        # and never inherit the much larger provider timeout.
        source = (ROOT / "scripts/review_jobs.py").read_text()
        self.assertIn("followups.ARTIFACT_TIMEOUT", source)
        self.assertNotIn('download_record(a, os.environ["GITHUB_TOKEN"], policy["ai_timeout"])', source)
        config = reliability.tomllib.loads((ROOT / ".github/pr-agent.toml").read_text())
        self.assertLess(followups.ARTIFACT_TIMEOUT, config["config"]["ai_timeout"])


class ArtifactRetryTests(unittest.TestCase):
    def test_a_single_stalled_artifact_request_is_retried_once(self):
        calls = []

        def stalls_once(*args, **kwargs):
            calls.append(kwargs.get("timeout"))
            if len(calls) == 1:
                raise TimeoutError("stalled")
            return "ok"

        self.assertEqual("ok", followups._get_once_more(stalls_once, "url", timeout=followups.ARTIFACT_TIMEOUT))
        self.assertEqual([followups.ARTIFACT_TIMEOUT] * 2, calls)

        def always_stalls(*args, **kwargs):
            calls.append(1)
            raise TimeoutError("stalled")

        calls.clear()
        with self.assertRaises(TimeoutError):
            followups._get_once_more(always_stalls, "url")
        self.assertEqual(2, len(calls))


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
                raise review.ReviewAbort(f"Review failed ({SECRET})") from None
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

    def test_outcome_write_failure_warns_and_still_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {"REVIEW_OUTCOME_PATH": directory, "GITHUB_STEP_SUMMARY": f"{directory}/summary.md"}
            stream = io.StringIO()
            result = reliability.report(env=env, stream=stream)
            self.assertEqual("Review outcome could not be recorded\n", stream.getvalue())
            self.assertIn("Agent review outcome", Path(env["GITHUB_STEP_SUMMARY"]).read_text())
            self.assertEqual("succeeded", result["result"])
            broken = NS(write=lambda text: (_ for _ in ()).throw(OSError("private-error")))
            self.assertEqual(result, reliability.report(env=env, stream=broken))
            stream.close()
            self.assertEqual(result, reliability.report(env=env, stream=stream))
            error = RuntimeError("private-error")
            expected = reliability.outcome(error, env)
            for sink in (stream, broken):
                self.assertEqual(expected, reliability.report(error, env=env, stream=sink))

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
