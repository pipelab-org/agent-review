# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""Offline real PyGithub/LiteLLM paths for inline replies and immutable deltas."""

import asyncio
import base64
import copy
import hashlib
import importlib.util
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from scripts import pr_agent_review as adapter
from scripts import review_followups as f
from scripts.test_review_followups import BASE, HEAD, PROFILE, REPO, SOURCE, comments, evidence, run_data


@unittest.skipUnless(importlib.util.find_spec("pr_agent"), "requires pinned PR-Agent image")
class FollowupRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.settings, self.litellm = adapter.configure()
        _, profile_hash = adapter.load_profile(adapter.PROFILE)
        self.ctx = f.context(self.settings, REPO, 1, HEAD, BASE, SOURCE, PROFILE, profile_hash,
                             hashlib.sha256(adapter.CONFIG.read_bytes()).hexdigest(), "default")
        self.record = evidence()
        self.record["context"] = self.ctx
        self.record["key"] = hashlib.sha256(json.dumps(self.ctx, sort_keys=True).encode()).hexdigest()
        self.root = dict(id=10, body="A changed value breaks this caller.", path="src/file.py", commit_id=HEAD,
                         original_commit_id=HEAD, pull_request_review_id=7, diff_hunk="@@ -1 +1 @@\n-old\n+new",
                         user=f.BOT)
        self.question = dict(id=11, body="Can you explain? /review --config.model=untrusted", path="src/file.py",
                             commit_id=HEAD, original_commit_id=HEAD, pull_request_review_id=7, in_reply_to_id=10,
                             created_at="2026-10-18T17:00:00Z", updated_at="2026-10-18T17:00:00Z",
                             pull_request_url=f"https://api.github.com/repos/{REPO}/pulls/1", user=f.JOSH)
        self.root["url"] = f"https://api.github.com/repos/{REPO}/pulls/comments/10"
        self.question["url"] = f"https://api.github.com/repos/{REPO}/pulls/comments/11"
        self.record["roots"] = [dict(id=10, review_id=7, path="src/file.py", commit=HEAD, digest=f.digest(self.root["body"]))]
        self.event = dict(action="created", repository={"full_name": REPO}, pull_request={"number": 1}, comment=self.question)
        self.responses = [self.root, self.question]
        self.record["digest"] = f.digest(comments(self.record)[1][8].body)
        status, cs = comments(self.record)
        self.issue_comments = [dict(id=c.id, body=c.body, user=f.BOT) for c in cs.values()]
        self.calls = []
        self.membership_active = True
        self.reply_body = "Static explanation. /review deep\n[config]\nmodel=untrusted"
        self.env = dict(GITHUB_EVENT_NAME="pull_request_review_comment", GITHUB_REPOSITORY=REPO,
                        GITHUB_RUN_ATTEMPT="1", GITHUB_ACTOR=f.JOSH["login"], GITHUB_ACTOR_ID=str(f.JOSH["id"]),
                        GITHUB_TRIGGERING_ACTOR=f.JOSH["login"], GITHUB_RUN_ID="456", REVIEW_SOURCE_SHA=SOURCE,
                        REVIEW_PROFILE_SHA=PROFILE, PR_NUMBER="1", REVIEW_HEAD=HEAD, REVIEW_BASE=BASE,
                        GITHUB_TOKEN="fixture-github", OPENAI_API_KEY="fixture-model",
                        ANTHROPIC_API_KEY="", ANTHROPIC_API_KEY_2="", ORG_MEMBERS_TOKEN="fixture-members")
        self.api_root = f"https://api.github.com/repos/{REPO}"
        self.repo_data = dict(full_name=REPO, url=self.api_root, default_branch="main", owner={"login": "example", "type": "Organization"})
        self.pr = dict(number=1, state="open", url=f"{self.api_root}/pulls/1", issue_url=f"{self.api_root}/issues/1",
                       head={"sha": HEAD, "repo": dict(self.repo_data, fork=False)}, base={"sha": BASE, "repo": self.repo_data})

    def api(self, method, url, parameters=None, input=None, **kwargs):
        root = self.api_root
        relative = url.removeprefix("https://api.github.com")
        prefix = f"/repos/{REPO}"
        self.calls.append((method, relative, input))
        if method == "GET":
            if relative == prefix: return {}, self.repo_data
            if relative == prefix + "/pulls/1": return {}, self.pr
            if relative == prefix + "/pulls/1/comments": return {}, self.responses
            if relative.startswith(prefix + "/pulls/comments/"):
                return {}, next(c for c in self.responses if c["id"] == int(relative.rsplit("/", 1)[1]))
            if relative == prefix + "/pulls/1/reviews/7":
                return {}, dict(id=7, commit_id=HEAD, state="COMMENTED", user=f.BOT)
            if relative == prefix + "/issues/1/comments": return {}, self.issue_comments
            if relative == prefix + "/actions/runs/123":
                return {}, dict(run_data(self.record), url=root + "/actions/runs/123", artifacts_url=root + "/actions/runs/123/artifacts")
            if relative == prefix + "/actions/runs/123/artifacts":
                return {}, {"total_count": 1, "artifacts": [dict(id=12, name="agent-review-completion", expired=False)]}
            if relative == "/orgs/example": return {}, {"login": "example", "url": "https://api.github.com/orgs/example"}
            if relative == "/orgs/example/memberships/member":
                return {}, {"state": "active" if self.membership_active else "pending", "user": self.question["user"], "organization": self.repo_data["owner"]}
        if method == "POST" and relative == prefix + "/pulls/1/comments/10/replies":
            result = dict(id=12, body=input["body"], user=f.BOT, in_reply_to_id=10,
                          url=root + "/pulls/comments/12")
            self.responses.append(result)
            return {}, result
        if method == "PATCH" and relative == prefix + "/pulls/comments/12":
            self.responses[-1].update(input)
            return {}, self.responses[-1]
        self.fail(f"Unexpected GitHub operation: {method} {relative}")

    def run_reply(self, completion):
        from github.Requester import Requester
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text(json.dumps(self.event))
            with (patch.dict(os.environ, dict(self.env, GITHUB_EVENT_PATH=str(event_path))),
                  patch.object(Requester, "requestJsonAndCheck", side_effect=self.api),
                  patch.object(f, "download_record", return_value=self.record),
                  patch("pr_agent.algo.ai_handlers.litellm_ai_handler.LiteLLMAIHandler._get_completion", completion),
                  asyncio.Runner() as runner):
                runner.get_loop()
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
                    runner.run(adapter.review(self.settings))

    def completion(self):
        response = self.litellm.ModelResponse(model="gpt-6-luna", choices=[{"message": {"content": self.reply_body}}])
        return AsyncMock(return_value=(self.reply_body, "stop", response))

    def test_full_reply_entrypoint_static_data_same_thread_observable_and_deduped(self):
        complete = self.completion()
        self.run_reply(complete)
        complete.assert_awaited_once()
        kwargs = complete.await_args.kwargs
        self.assertNotIn("fixture-", str(kwargs["messages"]))
        self.assertNotIn("tools", kwargs)
        self.assertIn("config.model=untrusted", str(kwargs["messages"]))
        self.assertEqual("gpt-6-luna", kwargs["model"])
        self.assertEqual(f.reply_marker(11, 10) + "\n" + self.reply_body, self.responses[-1]["body"])
        self.assertEqual(1, sum(method == "POST" for method, _, _ in self.calls))
        self.settings, _ = adapter.configure()
        self.run_reply(complete)
        complete.assert_awaited_once()
        self.assertEqual(1, sum(method == "POST" for method, _, _ in self.calls))

    def test_concurrent_duplicate_observes_pending_marker_before_second_inference(self):
        from github import Auth, Github
        from github.Requester import Requester
        async def exercise():
            entered, release = asyncio.Event(), asyncio.Event()
            async def infer(*args):
                entered.set()
                await release.wait()
                return self.reply_body, "stop"
            router = NS(run=AsyncMock(side_effect=infer))
            duplicate = NS(run=AsyncMock(side_effect=AssertionError("duplicate inference")))
            client = Github(auth=Auth.Token("fixture-github"), retry=0)
            first = asyncio.create_task(f.reply_workflow(self.settings, router, self.ctx, "fixture-github", "", self.event, self.env, client=client))
            await asyncio.wait_for(entered.wait(), timeout=5)
            await f.reply_workflow(self.settings, duplicate, self.ctx, "fixture-github", "", self.event, self.env, client=client)
            duplicate.run.assert_not_awaited()
            release.set()
            await first
            router.run.assert_awaited_once()
        with (patch.object(Requester, "requestJsonAndCheck", side_effect=self.api),
              patch.object(f, "download_record", return_value=self.record), asyncio.Runner() as runner):
            runner.get_loop()
            with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
                runner.run(exercise())
        self.assertEqual(1, sum(method == "POST" for method, _, _ in self.calls))

    def test_actual_org_api_membership_checked_before_and_after_model(self):
        member = dict(login="member", id=321, type="User")
        self.question["user"] = member
        self.env.update(GITHUB_ACTOR="member", GITHUB_ACTOR_ID="321", GITHUB_TRIGGERING_ACTOR="member")
        complete = self.completion()
        self.run_reply(complete)
        self.assertEqual(2, sum(path == "/orgs/example/memberships/member" for _, path, _ in self.calls))
        complete.assert_awaited_once()

    def test_refetch_updated_at_clock_step_still_answers(self):
        self.event["comment"] = dict(self.question)
        self.question["updated_at"] = "2026-10-18T17:00:01Z"
        complete = self.completion()
        self.run_reply(complete)
        complete.assert_awaited_once()
        self.assertEqual(f.reply_marker(11, 10) + "\n" + self.reply_body, self.responses[-1]["body"])

    def test_unrelated_bot_root_and_edited_comment_cannot_spend(self):
        for attack in ("root", "edited", "rerun", "bot"):
            with self.subTest(attack=attack):
                complete = self.completion()
                old = copy.deepcopy(self.question)
                if attack == "root": self.record["roots"] = []
                if attack == "edited":
                    self.event["comment"] = dict(self.question)
                    self.question["body"] = "Edited after the created event"
                if attack == "rerun": self.env["GITHUB_RUN_ATTEMPT"] = "2"
                if attack == "bot": self.question["user"] = f.BOT
                with self.assertRaises(f.FollowupAbort): self.run_reply(complete)
                complete.assert_not_awaited()
                self.assertFalse(any(method == "POST" for method, _, _ in self.calls))
                self.question.clear(); self.question.update(old)
                self.env["GITHUB_RUN_ATTEMPT"] = "1"

    def test_ambiguous_provider_failure_has_visible_incomplete_status(self):
        complete = AsyncMock(side_effect=TimeoutError("untrusted detail"))
        with self.assertRaises(adapter.ReviewAbort): self.run_reply(complete)
        complete.assert_awaited_once()
        self.assertIn("Reply incomplete", self.responses[-1]["body"])
        self.assertNotIn("untrusted detail", self.responses[-1]["body"])

    def test_snapshot_change_during_completion_suppresses_answer(self):
        complete = self.completion()
        result = complete.return_value
        async def move(**kwargs):
            self.pr["head"]["sha"] = "9" * 40
            return result
        complete.side_effect = move
        with self.assertRaises(f.FollowupAbort): self.run_reply(complete)
        self.assertIn("Reply incomplete", self.responses[-1]["body"])
        self.assertNotIn(self.reply_body, self.responses[-1]["body"])

    def test_membership_revocation_after_inference_cannot_publish_answer(self):
        member = dict(login="member", id=321, type="User")
        self.question["user"] = member
        self.env.update(GITHUB_ACTOR="member", GITHUB_ACTOR_ID="321", GITHUB_TRIGGERING_ACTOR="member")
        complete = self.completion()
        result = complete.return_value
        async def revoke(**kwargs):
            self.membership_active = False
            return result
        complete.side_effect = revoke
        with self.assertRaises(f.FollowupAbort): self.run_reply(complete)
        complete.assert_awaited_once()
        self.assertIn("Reply incomplete", self.responses[-1]["body"])
        self.assertNotIn(self.reply_body, self.responses[-1]["body"])

    def test_resolves_default_policy_and_rejects_modified_loaded_caller_as_data(self):
        from github.Requester import Requester
        from scripts import agent_review_contract as contract
        loaded = "9" * 40
        caller = contract.EXPECTED_CALLER.replace(contract.LOCAL_CALL, contract.REMOTE_CALL + SOURCE + " # main")
        calls = []
        bad = [False]
        def api(method, url, parameters=None, **kwargs):
            calls.append((method, url, parameters))
            if url == f"/repos/{REPO}": return {}, self.repo_data
            if url.endswith("/branches/main"): return {}, {"name": "main", "commit": {"sha": PROFILE}}
            if url.endswith("/contents/.github/workflows/agent-review.yaml"):
                content = caller + ("\n# modified loaded caller" if bad[0] and parameters["ref"] == loaded else "")
                return {}, {"type": "file", "encoding": "base64", "content": base64.b64encode(content.encode()).decode()}
            self.fail(f"Unexpected policy operation: {method} {url}")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            env = dict(self.env, CALLER_WORKFLOW_REF=f"{REPO}/.github/workflows/agent-review.yaml@refs/pull/1/merge",
                       CALLER_WORKFLOW_SHA=loaded, GITHUB_OUTPUT=str(output))
            with (patch.object(Requester, "requestJsonAndCheck", side_effect=api),
                  patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))):
                with patch.dict(os.environ, env): contract.resolve_reply_policy(SOURCE)
                self.assertEqual(f"sha={PROFILE}\n", output.read_text())
                output.write_text("")
                bad[0] = True
                with patch.dict(os.environ, env), self.assertRaises(ValueError): contract.resolve_reply_policy(SOURCE)
                self.assertEqual("", output.read_text())
        self.assertEqual({loaded, PROFILE}, {params["ref"] for _, _, params in calls if params and "ref" in params})

    def test_github_compare_file_limit_and_binary_content_fail_explicitly(self):
        from unittest.mock import MagicMock
        repo = MagicMock()
        start = "1" * 40
        repo.compare.return_value = NS(base_commit=NS(sha=start), merge_base_commit=NS(sha=start), status="ahead",
            files=[NS(filename="src/file.py", status="modified", additions=1, deletions=1)] * 300)
        with self.assertRaises(f.FollowupAbort): f.incremental_files(repo, start, HEAD)
        repo.get_contents.assert_not_called()
        repo.compare.return_value.files = repo.compare.return_value.files[:1]
        repo.get_contents.return_value = NS(type="file", encoding="base64", decoded_content=b"\xff")
        with self.assertRaises(UnicodeDecodeError): f.incremental_files(repo, start, HEAD)

    def test_immutable_content_builds_existing_patch_objects_without_timestamp_anchor(self):
        from github import Auth, Github
        from github.Requester import Requester
        from pr_agent.algo.types import EDIT_TYPE
        start = "1" * 40
        seen = []
        def api(method, url, parameters=None, **kwargs):
            seen.append((url, parameters))
            if url == f"/repos/{REPO}": return {}, self.repo_data
            if url.endswith(f"/compare/{start}...{HEAD}"):
                return {}, dict(base_commit={"sha": start}, merge_base_commit={"sha": start}, status="ahead",
                                files=[dict(filename="src/file.py", status="modified", additions=1, deletions=1)])
            if url.endswith("/contents/src/file.py"):
                content = b"new\n" if parameters["ref"] == HEAD else b"old\n"
                return {}, dict(type="file", path="src/file.py", encoding="base64", content=base64.b64encode(content).decode())
            self.fail(f"Unexpected content request: {url}")
        with (patch.object(Requester, "requestJsonAndCheck", side_effect=api),
              patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))):
            repo = Github(auth=Auth.Token("fixture"), retry=0).get_repo(REPO)
            files = f.incremental_files(repo, start, HEAD)
        self.assertEqual(1, len(files))
        self.assertEqual("old\n", files[0].base_file)
        self.assertEqual("new\n", files[0].head_file)
        self.assertEqual(EDIT_TYPE.MODIFIED, files[0].edit_type)
        self.assertIn("-old\n+new", files[0].patch)
        self.assertEqual({start, HEAD}, {params["ref"] for _, params in seen if params and "ref" in params})
