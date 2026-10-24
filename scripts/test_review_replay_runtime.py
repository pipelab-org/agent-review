# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""Replay tests against the pinned PR-Agent installation.

Every response here is a transport fixture or a canned wire response. None of it is
model output, and nothing in this file measures reviewer quality.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import review_replay as replay
from scripts import review_replay_snapshot as snapshots
from scripts.test_review_quality import decision
from scripts.test_review_replay_snapshot import BASE, HEAD, snapshot_dict

PINNED = importlib.util.find_spec("pr_agent")
CLEAN = "review:\n  relevant_tests: 'No'\n  security_concerns: 'No'\n  key_issues_to_review: []\n"
FINDING = """review:
  security_concerns: 'No'
  relevant_tests: 'No'
  key_issues_to_review:
    - relevant_file: src/file.py
      issue_header: Correctness
      issue_content: This replacement breaks the caller; retain the supported value.
      start_line: 1
      end_line: 1
"""
USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
REVIEWER = "c" * 40


def text(body, **extra):
    return dict({"text": body, "usage": USAGE}, **extra)


def publish(**evidence):
    d = decision()
    d.update(title="Correctness")
    d["evidence"].update(start_line=1, end_line=1, quote="new", **evidence)
    return text(json.dumps({"decisions": [d]}))


@unittest.skipUnless(PINNED, "requires disposable pr-agent==0.47.0 environment")
class ReplayTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.directory, ignore_errors=True))

    def snapshot(self, **changes):
        path = self.directory / "snapshot.json"
        path.write_text(json.dumps(snapshot_dict(**changes)), encoding="utf-8")
        return snapshots.load(path)

    def run_replay(self, responses, *, snapshot=None, **options):
        options.setdefault("reviewer_sha", REVIEWER)
        return replay.run(snapshot or self.snapshot(), mode="fixture", fixture=responses, **options)

    def messages(self, responses, **options):
        doc = self.run_replay(responses, record_prompts=True, **options)
        return doc, [c["messages"] for c in doc["calls"]]


class ReplayRunTests(ReplayTestCase):
    def test_runs_the_real_adapter_and_labels_the_result_a_replay(self):
        doc, calls = self.messages([text("source_requests: [src/consumer.py]\n" + CLEAN), text(CLEAN)])
        self.assertEqual("completed", doc["adapter_outcome"])
        self.assertTrue(doc["replay"])
        self.assertFalse(doc["genuine_github_completion"])
        self.assertEqual(2, len(doc["calls"]))
        self.assertEqual([], doc["findings"])
        self.assertTrue(doc["comparable"])
        # The adapter's own prompts, not a replay-specific reviewer.
        self.assertIn(replay.review.discovery.REQUEST_INSTRUCTIONS, calls[0][0]["content"])
        self.assertTrue(calls[1][0]["content"].startswith(replay.review.discovery.SYSTEM))
        bundle = json.loads(calls[1][1]["content"])
        self.assertEqual("existing_guard()\n", bundle["sources"]["src/consumer.py"]["content"])
        self.assertEqual({"src/file.py": "@@ -1 +1 @@\n-old\n+new\n"}, bundle["diff"])
        self.assertEqual("Untrusted description", bundle["description"])
        self.assertEqual(
            {"models": ["anthropic/claude-sonnet-5-5"], "successful_calls": 2, "billing_rejections": 0,
             "reported_tokens": {"completion_tokens": 100, "prompt_tokens": 200, "total_tokens": 300}},
            {k: doc["run"][k] for k in ("models", "successful_calls", "billing_rejections", "reported_tokens")})

    def test_fixture_results_are_never_quality_evidence(self):
        doc = self.run_replay([text("source_requests: []\n" + CLEAN), text(CLEAN)])
        self.assertEqual("fixture", doc["inference"]["mode"])
        self.assertFalse(doc["inference"]["quality_evidence"])
        self.assertIn("not a model", doc["inference"]["note"])

    def test_handled_unmodeled_access_invalidates_an_otherwise_completed_replay(self):
        original = replay.provider_class

        def provider_with_missing_read():
            base = original()

            class Provider(base):
                def get_languages(self):
                    # Upstream may catch unsupported capabilities and keep reviewing.
                    hasattr(self.pr, "unsupported_capability")
                    return super().get_languages()

            return Provider

        with patch.object(replay, "provider_class", provider_with_missing_read):
            doc = self.run_replay([text(CLEAN), text(CLEAN)])
        self.assertEqual("completed", doc["adapter_outcome"])
        self.assertTrue(doc["unmodeled_provider_access"])
        self.assertFalse(doc["comparable"])
        self.assertFalse(doc["inference"]["quality_evidence"])

    def test_handled_reads_of_nested_provider_data_are_recorded_too(self):
        original = replay.provider_class

        def provider_probing_unmodeled_fields():
            base = original()

            class Provider(base):
                def get_languages(self):
                    # Plain data objects, not only the Strict API objects.
                    for probe in (lambda: self.pr.head.repo.owner, lambda: self.pr.base.user,
                                  lambda: self.last_commit_id.committer):
                        try:
                            probe()
                        except AttributeError:
                            pass
                    return super().get_languages()

            return Provider

        with patch.object(replay, "provider_class", provider_probing_unmodeled_fields):
            doc = self.run_replay([text(CLEAN), text(CLEAN)])
        self.assertEqual("completed", doc["adapter_outcome"])
        self.assertEqual(["Obj.committer", "Obj.owner", "Obj.user"],
                         doc["unmodeled_provider_access"])
        self.assertFalse(doc["comparable"])
        self.assertFalse(doc["inference"]["quality_evidence"])

    def test_plain_provider_data_compares_by_fields_like_the_namespaces_it_replaces(self):
        first = replay.Obj(None, x=1, y=2)
        self.assertEqual(first, replay.Obj(replay.Store(), y=2, x=1))
        self.assertNotEqual(first, replay.Obj(None, x=1))
        self.assertIn("'x': 1", repr(first))

    def test_configured_models_effort_budgets_and_single_attempt_are_preserved(self):
        doc = self.run_replay([text("source_requests: []\n" + CLEAN), text(CLEAN)])
        config = doc["reviewer"]["configuration"]
        self.assertEqual(["anthropic/claude-sonnet-5-5", "gpt-6-luna"], config["models"])
        self.assertEqual((90000, 16000, 300, 3, 3),
                         (config["max_model_tokens"], config["max_output_tokens"], config["ai_timeout"],
                          config["max_calls"], config["max_findings"]))
        first = doc["calls"][0]
        self.assertEqual((0, 0, 16000, "anthropic/claude-sonnet-5-5", {"effort": "medium"}),
                         (first["num_retries"], first["max_retries"], first["max_tokens"], first["model"],
                          first["output_config"]))
        self.assertEqual([["anthropic/claude-sonnet-5-5", "ANTHROPIC_API_KEY"],
                          ["anthropic/claude-sonnet-5-5", "ANTHROPIC_API_KEY_2"], ["gpt-6-luna", "OPENAI_API_KEY"]],
                         doc["reviewer"]["credential_route"])
        deep = self.run_replay([text("source_requests: []\n" + CLEAN), text(CLEAN)], depth="deep")
        self.assertEqual("anthropic/claude-opus-5-5", deep["calls"][0]["model"])
        self.assertEqual({"effort": "high"}, deep["calls"][0]["output_config"])
        self.assertEqual("deep", deep["reviewer"]["depth"])

    def test_result_binds_the_executed_files_and_snapshot_identity(self):
        snapshot = self.snapshot()
        doc = self.run_replay([text("source_requests: []\n" + CLEAN), text(CLEAN)], snapshot=snapshot)
        self.assertEqual(snapshot.sha256, doc["snapshot"]["sha256"])
        self.assertEqual((HEAD, BASE), (doc["snapshot"]["head"], doc["snapshot"]["base"]))
        self.assertEqual("2026-01-01T00:00:00Z", doc["snapshot"]["as_of"])
        files = doc["reviewer"]["executed_files"]
        self.assertEqual(set(replay.EXECUTED) | {"profile"}, set(files))
        engine = Path(replay.review.__file__)
        self.assertEqual(hashlib.sha256(engine.read_bytes()).hexdigest(), files["scripts/pr_agent_review.py"])

    def test_candidate_is_challenged_and_published_only_in_memory(self):
        doc, calls = self.messages([text("source_requests: []\n" + FINDING), text(CLEAN), publish()])
        self.assertEqual("completed", doc["adapter_outcome"])
        self.assertEqual(3, len(doc["calls"]))
        self.assertEqual(replay.review.quality.SYSTEM, calls[2][0]["content"])
        self.assertEqual(["src/file.py"], [f["path"] for f in doc["findings"]])
        self.assertEqual(1, len(doc["captured"]["inline_reviews"]))
        self.assertEqual("src/file.py", doc["captured"]["inline_reviews"][0]["comments"][0]["path"])
        self.assertEqual("publish", doc["quality"]["decisions"][0]["verdict"])

    def test_rejected_candidate_publishes_nothing(self):
        d = decision(verdict="unsupported")
        d["reason"] = "The supplied source refutes the claim."
        doc = self.run_replay([text("source_requests: []\n" + FINDING), text(CLEAN), text(json.dumps({"decisions": [d]}))])
        self.assertEqual("completed", doc["adapter_outcome"])
        self.assertEqual([], doc["findings"])
        self.assertEqual([], doc["captured"]["inline_reviews"])
        self.assertEqual("unsupported", doc["quality"]["decisions"][0]["verdict"])

    def test_malformed_challenge_never_completes_or_captures_a_summary(self):
        for answer in ("not JSON", json.dumps({"decisions": []})):
            with self.subTest(answer=answer):
                doc = self.run_replay([text("source_requests: []\n" + FINDING), text(CLEAN), text(answer)])
                self.assertEqual("aborted", doc["adapter_outcome"])
                self.assertEqual("Review incomplete or invalid at finding challenge decisions", doc["failure"])
                self.assertIsNone(doc["captured"]["summary"])
                self.assertEqual([], doc["captured"]["inline_reviews"])
                self.assertFalse(doc["comparable"])

    def test_as_of_discussion_reaches_the_challenge_and_can_retain_a_finding(self):
        thread = {"id": 77, "in_reply_to_id": None, "path": "src/file.py", "body": "Earlier report of this failure."}
        snapshot = self.snapshot(discussion={"as_of": "2026-01-01T00:00:00Z", "threads": [thread]})
        d = decision(verdict="duplicate")
        d.update(prior_comment_id=77, reason="Same scenario as the earlier thread.")
        doc, calls = self.messages([text("source_requests: []\n" + FINDING), text(CLEAN), text(json.dumps({"decisions": [d]}))],
                                   snapshot=snapshot)
        self.assertEqual("completed", doc["adapter_outcome"])
        self.assertEqual([thread], [{k: t[k] for k in thread if k != "in_reply_to_id"} | {"in_reply_to_id": t["parent"]}
                                    for t in json.loads(calls[2][1]["content"])["threads"]])
        self.assertEqual([], doc["captured"]["inline_reviews"])
        self.assertIn("Existing unresolved finding", doc["findings"][0]["title"])

    def test_source_requests_come_only_from_the_snapshot_and_unknown_paths_are_named(self):
        for request, reason in (("src/gone.py", "not found at the reviewed head"),
                                ("../escape", "invalid repository-relative file path")):
            with self.subTest(request=request):
                snapshot = self.snapshot(absent=["src/gone.py"])
                doc = self.run_replay([text(f"source_requests: [{request}]\n" + CLEAN), text(CLEAN)], snapshot=snapshot)
                self.assertEqual("completed", doc["adapter_outcome"])
                self.assertEqual({request: reason}, doc["quality"]["discovery"]["unavailable_requests"])
                self.assertIn("Source limitations", doc["captured"]["summary"])

    def test_missing_source_aborts_names_the_path_and_marks_the_run_not_comparable(self):
        doc = self.run_replay([text("source_requests: [src/not_in_snapshot.py]\n" + CLEAN)])
        self.assertEqual("aborted", doc["adapter_outcome"])
        self.assertEqual("Review incomplete or invalid at source context", doc["failure"])
        self.assertEqual(["src/not_in_snapshot.py"], doc["uncaptured_source_requests"])
        self.assertFalse(doc["comparable"])
        self.assertEqual(1, len(doc["calls"]))

    def test_source_request_count_keeps_the_providers_own_limit(self):
        from pr_agent.git_providers.github_provider import MAX_FILES_ALLOWED_FULL
        sources = {f"src/c{i}.py": "x\n" for i in range(MAX_FILES_ALLOWED_FULL + 1)}
        for count, outcome in ((MAX_FILES_ALLOWED_FULL, "completed"), (MAX_FILES_ALLOWED_FULL + 1, "aborted")):
            with self.subTest(count=count):
                requested = json.dumps([f"src/c{i}.py" for i in range(count)])
                doc = self.run_replay([text(f"source_requests: {requested}\n" + CLEAN), text(CLEAN)],
                                      snapshot=self.snapshot(sources=sources))
                self.assertEqual(outcome, doc["adapter_outcome"])

    def test_failed_inference_never_publishes_or_retries(self):
        for error in ({"timeout": True}, {"status": 429, "body": {"error": {"type": "rate_limit_error", "message": "slow"}}},
                      {"status": 500, "body": {}}):
            with self.subTest(error=error):
                doc = self.run_replay([{"error": error}, text(CLEAN)])
                self.assertEqual("aborted", doc["adapter_outcome"])
                self.assertEqual(1, len(doc["calls"]))
                self.assertEqual(1, doc["unused_fixture_responses"])
                self.assertIsNone(doc["captured"]["summary"])
                self.assertEqual([], doc["captured"]["status_comments"][1:])

    def test_billing_rejection_uses_the_configured_next_slot_only(self):
        credit = {"status": 400, "body": {"error": {"type": "invalid_request_error",
                                                    "message": "Your credit balance is too low to access the API."}}}
        doc = self.run_replay([{"error": credit}, text("source_requests: []\n" + CLEAN), text(CLEAN)])
        self.assertEqual("completed", doc["adapter_outcome"])
        self.assertEqual(["error:APIStatusError", "returned", "returned"], [c["outcome"] for c in doc["calls"]])
        self.assertEqual(1, doc["run"]["billing_rejections"])

    def test_exhausted_fixture_is_a_failure_not_a_pass(self):
        doc = self.run_replay([text("source_requests: []\n" + CLEAN)])
        self.assertEqual("aborted", doc["adapter_outcome"])
        self.assertIsNone(doc["findings"])

    def test_captured_publications_cannot_be_reused_as_completion_evidence(self):
        doc = self.run_replay([text("source_requests: []\n" + FINDING), text(CLEAN), publish()])
        captured = [doc["captured"]["summary"], *doc["captured"]["status_comments"]]
        followups = replay.review.followups
        for body in captured:
            self.assertNotIn("<!-- agent-review-", body)
            self.assertIsNone(replay.review.parse_result(body))
            self.assertNotIn(replay.REPLAY_RUN_ID, body.replace("replay:no-actions-run", ""))
        self.assertNotIn("github.com/example/project/actions/runs", " ".join(captured))
        # The neutralization is what blocks reuse: restored, the same text would parse.
        restored = doc["captured"]["summary"].replace("<!-- replay-neutralized-agent-review-", "<!-- agent-review-")
        self.assertIsNotNone(replay.review.parse_result(restored))
        status = next(b for b in doc["captured"]["status_comments"] if "status:v1 state=success" in b)
        self.assertEqual(0, len(followups.STATUS.findall(status)))
        # The replay envelope is not a completion record or a comment, and its run id is no Actions run.
        self.assertFalse({"key", "summary", "context", "run_id", "roots"} & set(doc))
        self.assertEqual(replay.REPLAY_RUN_ID, str(2**63 - 1))

    def test_gateway_hosts_follow_the_configured_model_chain(self):
        with patch.dict(os.environ, {"REVIEW_SOURCE_SHA": REVIEWER, "REVIEW_PROFILE_SHA": REVIEWER,
                                     "OPENAI_API_KEY": "label", "ANTHROPIC_API_KEY": "label"}):
            settings, litellm = replay.review.configure()
            models, chain, hosts = replay.chain_for(settings, litellm)
            self.assertEqual({"api.anthropic.com", "api.openai.com"}, hosts)
            settings.set("config.model", "gpt-6-luna")
            settings.set("config.fallback_models", [])
            self.assertEqual({"api.openai.com"}, replay.chain_for(settings, litellm)[2])
            settings.set("config.model", "other-vendor/model")
            with self.assertRaises(replay.ReplayError):
                replay.chain_for(settings, litellm)

    def test_unmodeled_provider_capabilities_fail_visibly_not_with_mock_results(self):
        store = replay.Store()
        pull = replay.Pull(store, self.snapshot())
        with self.assertRaises(replay.ReplayUnsupported):
            pull.merge()
        self.assertFalse(hasattr(pull, "labels"))
        self.assertEqual(["Pull.merge", "Pull.labels"], store.unmodeled)
        provider = replay.provider_class()(self.snapshot(), store)
        with self.assertRaises(replay.ReplayUnsupported):
            provider.github_client.get_user()
        with self.assertRaises(replay.Uncaptured):
            provider.github_client.get_repo("other/repo")
        with self.assertRaises(replay.Uncaptured):
            provider.repo_obj.get_pull(99)
        with self.assertRaises(replay.Uncaptured):
            provider.repo_obj.get_contents("src/consumer.py", ref="d" * 40)
        self.assertEqual([], provider.pr.get_issue_comments())
        doc = self.run_replay([text("source_requests: []\n" + CLEAN), text(CLEAN)])
        self.assertEqual([], doc["unmodeled_provider_access"])

    def test_snapshot_file_is_not_modified_and_replays_are_independent(self):
        snapshot = self.snapshot()
        before = (self.directory / "snapshot.json").read_bytes()
        first = self.run_replay([text("source_requests: []\n" + FINDING), text(CLEAN), publish()], snapshot=snapshot)
        second = self.run_replay([text("source_requests: []\n" + CLEAN), text(CLEAN)], snapshot=snapshot)
        self.assertEqual(before, (self.directory / "snapshot.json").read_bytes())
        self.assertEqual(1, len(first["findings"]))
        self.assertEqual([], second["findings"])
        self.assertEqual([], second["captured"]["inline_reviews"])


class NoGitHubTests(ReplayTestCase):
    def test_a_full_replay_never_constructs_a_github_client_or_touches_a_socket(self):
        import github
        from github.Requester import Requester

        def refuse(*args, **kwargs):
            raise AssertionError("GitHub or network access attempted")

        with (patch.object(github.Github, "__init__", refuse),
              patch.object(Requester, "requestJsonAndCheck", refuse),
              patch.object(Requester, "requestBlobAndCheck", refuse),
              patch.object(Requester, "requestMultipartAndCheck", refuse),
              patch.object(socket.socket, "connect", refuse), patch.object(socket.socket, "connect_ex", refuse),
              patch("requests.Session.send", refuse)):
            doc = self.run_replay([text("source_requests: [src/consumer.py]\n" + FINDING), text(CLEAN), publish()])
        self.assertEqual("completed", doc["adapter_outcome"])
        self.assertEqual(1, len(doc["captured"]["inline_reviews"]))
        self.assertEqual([], doc["network_denials"])

    def test_the_guard_denies_github_and_every_other_destination_in_fixture_mode(self):
        import httpx
        import requests
        guard = replay.Egress(())
        with guard.active():
            with self.assertRaises(replay.ReplayError):
                requests.get("https://api.github.com/repos/example/project/pulls/1/reviews")
            with self.assertRaises(replay.ReplayError):
                httpx.get("https://api.github.com/repos/example/project")
            with self.assertRaises(replay.ReplayError):
                socket.getaddrinfo("api.github.com", 443)
            with self.assertRaises(replay.ReplayError):
                httpx.get("https://api.anthropic.com/v1/messages")
            with self.assertRaises(replay.ReplayError):
                socket.create_connection(("127.0.0.1", 9), timeout=1)
            with contextlib.closing(socket.socket(socket.AF_UNIX)) as unix, self.assertRaises(replay.ReplayError):
                unix.connect("/var/run/docker.sock")
        self.assertEqual(6, len(guard.denied))

    def test_requests_is_denied_even_where_no_socket_or_name_lookup_would_happen(self):
        # A transport adapter that never touches a socket isolates the Session.send layer
        # PyGithub uses; the name-lookup layer alone cannot be what denies this call.
        import requests

        class Canned(requests.adapters.BaseAdapter):
            sent = 0

            def send(self, request, **kwargs):
                Canned.sent += 1
                response = requests.Response()
                response.status_code = 200
                return response

            def close(self):
                pass

        session = requests.Session()
        session.mount("https://", Canned())
        guard = replay.Egress(("api.anthropic.com",))
        with guard.active(), self.assertRaises(replay.ReplayError):
            session.get("https://api.github.com/repos/example/project/pulls/1/reviews")
        self.assertEqual(0, Canned.sent)
        self.assertEqual(200, session.get("https://api.github.com/").status_code)  # unguarded control

    def test_real_mode_guard_permits_only_the_configured_gateways_over_https(self):
        import asyncio
        import httpx
        guard = replay.Egress(("api.anthropic.com",))
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))

        async def call(url):
            async with httpx.AsyncClient(transport=transport) as client:
                return await client.get(url)

        with guard.active():
            self.assertEqual(200, asyncio.run(call("https://api.anthropic.com/v1/messages")).status_code)
            for denied in ("https://api.github.com/repos/example/project", "http://api.anthropic.com/v1/messages",
                           "https://api.openai.com/v1/chat/completions", "https://api.anthropic.com.evil.example/"):
                with self.subTest(url=denied), self.assertRaises(replay.ReplayError):
                    asyncio.run(call(denied))
            with self.assertRaises(replay.ReplayError):
                socket.getaddrinfo("api.github.com", 443)
        self.assertEqual(4 + 1, len(guard.denied))

    def test_real_mode_reaches_only_the_gateway_through_the_existing_handler(self):
        # Wire-level canned response; no network. Proves the real route, the guard and
        # the unchanged handler cooperate, and that credentials never appear in the result.
        import httpx
        sent = []
        answers = iter([CLEAN.replace("review:", "source_requests: []\nreview:", 1), CLEAN])

        async def wire(client, request, *args, **kwargs):
            sent.append(str(request.url))
            body = {"id": "msg_replay", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
                    "content": [{"type": "text", "text": next(answers)}], "stop_reason": "end_turn",
                    "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}
            return httpx.Response(200, request=request, json=body)

        credentials = {"ANTHROPIC_API_KEY": "sk-ant-fixture-secret-one", "ANTHROPIC_API_KEY_2": "",
                       "OPENAI_API_KEY": "sk-fixture-secret-openai"}
        with patch.dict(os.environ, credentials), patch.object(httpx.AsyncClient, "send", wire):
            doc = replay.run(self.snapshot(), mode="real", reviewer_sha=REVIEWER, record_prompts=True)
        self.assertEqual("completed", doc["adapter_outcome"])
        self.assertEqual({"https://api.anthropic.com/v1/messages"}, set(sent))
        self.assertEqual(2, len(sent))
        self.assertEqual("real", doc["inference"]["mode"])
        self.assertTrue(doc["inference"]["quality_evidence"])
        self.assertFalse(doc["genuine_github_completion"])
        self.assertEqual([["anthropic/claude-sonnet-5-5", "ANTHROPIC_API_KEY"], ["gpt-6-luna", "OPENAI_API_KEY"]],
                         doc["reviewer"]["credential_route"])
        dumped = json.dumps(doc)
        for secret in credentials.values():
            self.assertTrue(not secret or secret not in dumped)
        self.assertEqual([], doc["network_denials"])

    def test_real_mode_refuses_without_a_credential_or_with_a_fixture(self):
        empty = {name: "" for name in replay.CREDENTIALS}
        with patch.dict(os.environ, empty), self.assertRaisesRegex(replay.ReplayError, "real inference is not attempted"):
            replay.run(self.snapshot(), mode="real", reviewer_sha=REVIEWER)
        with self.assertRaises(replay.ReplayError):
            replay.run(self.snapshot(), mode="real", fixture=[], reviewer_sha=REVIEWER)
        with self.assertRaises(replay.ReplayError):
            replay.run(self.snapshot(), mode="fixture", reviewer_sha=REVIEWER)
        for kwargs in ({"reviewer_sha": "short"}, {"reviewer_sha": REVIEWER, "depth": "extra"},
                       {"reviewer_sha": REVIEWER, "profile_sha": "X" * 40}):
            with self.subTest(kwargs=kwargs), self.assertRaises(replay.ReplayError):
                replay.run(self.snapshot(), mode="fixture", fixture=[], **kwargs)

    def test_ambient_github_credentials_never_reach_a_replay(self):
        seen = []
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghs_ambient_secret", "GH_TOKEN": "ghs_other",
                                     "ORG_MEMBERS_TOKEN": "ghs_members", "GITHUB_EVENT_NAME": "pull_request_review_comment",
                                     "REVIEW_COMPLETION_PATH": str(self.directory / "completion.json")}):
            doc = self.run_replay([text("source_requests: []\n" + CLEAN), text(CLEAN)])
            seen.append(os.environ.get("GITHUB_TOKEN"))
        self.assertEqual("completed", doc["adapter_outcome"])
        self.assertEqual(["ghs_ambient_secret"], seen)  # restored afterwards
        self.assertFalse((self.directory / "completion.json").exists())
        self.assertNotIn("ghs_", json.dumps(doc))


class CliTests(ReplayTestCase):
    def cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = replay.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def files(self, responses, **changes):
        self.snapshot(**changes)
        fixture = self.directory / "fixture.json"
        fixture.write_text(json.dumps({"schema": replay.FIXTURE_SCHEMA, "version": 1, "responses": responses}))
        return ["--snapshot", str(self.directory / "snapshot.json"), "--reviewer-sha", REVIEWER,
                "--inference", "fixture", "--fixture", str(fixture)]

    def test_completed_replay_prints_one_json_document(self):
        code, out, err = self.cli(*self.files([text("source_requests: []\n" + CLEAN), text(CLEAN)]))
        self.assertEqual(0, code)
        doc = json.loads(out)
        self.assertEqual((True, False, "completed"), (doc["replay"], doc["genuine_github_completion"], doc["adapter_outcome"]))
        self.assertNotIn("messages", doc["calls"][0])

    def test_output_file_prompt_recording_and_snapshot_pin(self):
        argv = self.files([text("source_requests: []\n" + CLEAN), text(CLEAN)])
        target = self.directory / "result.json"
        pin = hashlib.sha256((self.directory / "snapshot.json").read_bytes()).hexdigest()
        code, out, _ = self.cli(*argv, "--output", str(target), "--record-prompts", "--snapshot-sha256", pin)
        self.assertEqual((0, ""), (code, out))
        self.assertIn("messages", json.loads(target.read_text())["calls"][0])
        code, out, err = self.cli(*argv, "--snapshot-sha256", "0" * 64)
        self.assertEqual((2, ""), (code, out))
        self.assertIn("expected SHA-256", err)

    def test_aborted_replay_still_reports_and_exits_nonzero(self):
        code, out, _ = self.cli(*self.files([{"error": {"timeout": True}}]))
        self.assertEqual(1, code)
        self.assertEqual("aborted", json.loads(out)["adapter_outcome"])

    def test_completed_but_not_comparable_replay_exits_nonzero_and_still_reports(self):
        original = replay.provider_class

        def provider_with_missing_read():
            class Provider(original()):
                def get_languages(self):
                    hasattr(self.pr, "unsupported_capability")  # handled, so the run completes
                    return super().get_languages()

            return Provider

        with patch.object(replay, "provider_class", provider_with_missing_read):
            code, out, _ = self.cli(*self.files([text("source_requests: []\n" + CLEAN), text(CLEAN)]))
        doc = json.loads(out)
        self.assertEqual((1, "completed", False), (code, doc["adapter_outcome"], doc["comparable"]))

    def test_bad_inputs_exit_two_without_running_or_echoing_content(self):
        base = self.files([text(CLEAN)])
        (self.directory / "bad.json").write_text('{"schema": "private-marker-text"}')

        def replaced(flag, value):
            argv = list(base)
            argv[argv.index(flag) + 1] = value
            return argv

        cases = {
            "malformed snapshot": replaced("--snapshot", str(self.directory / "bad.json")),
            "missing snapshot": replaced("--snapshot", str(self.directory / "none.json")),
            "short reviewer sha": replaced("--reviewer-sha", "abc"),
            "missing fixture": replaced("--fixture", str(self.directory / "none.json")),
            "profile without sha": [*base, "--profile", str(self.directory / "p.toml")],
        }
        for name, argv in cases.items():
            with self.subTest(name=name):
                code, out, err = self.cli(*argv)
                self.assertEqual((2, ""), (code, out))
                self.assertTrue(err.startswith("Replay not run:"))
                self.assertNotIn("private-marker", err)

    def test_malformed_fixtures_and_real_without_credentials_exit_two(self):
        base = self.files([text(CLEAN)])
        fixture = Path(base[base.index("--fixture") + 1])
        for body in ('{"schema": "x"}', json.dumps({"schema": replay.FIXTURE_SCHEMA, "version": 1, "responses": [5]}),
                     json.dumps({"schema": replay.FIXTURE_SCHEMA, "version": 1, "responses": [{"text": 1}]}), "["):
            fixture.write_text(body)
            with self.subTest(body=body[:40]):
                self.assertEqual(2, self.cli(*base)[0])
        empty = {name: "" for name in replay.CREDENTIALS}
        argv = ["--snapshot", base[1], "--reviewer-sha", REVIEWER, "--inference", "real"]
        with patch.dict(os.environ, empty):
            code, out, err = self.cli(*argv)
        self.assertEqual((2, ""), (code, out))
        self.assertIn("real inference is not attempted", err)
        code, _, _ = self.cli(*base[:4], "--inference", "real", "--fixture", base[-1])
        self.assertEqual(2, code)


if __name__ == "__main__":
    unittest.main()
