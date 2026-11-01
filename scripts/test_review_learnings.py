# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline learnings producer, prompt, credential and publication boundary contracts."""
import copy
import datetime as dt
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch
from types import SimpleNamespace as NS

import yaml

from scripts import review_engine as e, review_jobs as j, review_learnings as l, review_followups as f
from scripts import test_review_engine as engine_tests, test_review_jobs as job_tests

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
TODAY = dt.date(2026, 10, 31)
OWNERS = 'maintainers = ["maintainer"]\n'
ENTRY = """[[learning]]
id = "example-0001"
status = "active"
kind = "rejected-finding"
text = "This pattern is fine; evidence is checked in the parser."
source = "https://github.com/example/project/pull/1#discussion_r1"
taught_by = "contributor"
approved_by = "maintainer"
added = 2026-01-01
review_by = 2099-01-01
"""


def data(text=ENTRY, owners=OWNERS, sha=SHA):
    return {"sha": sha, "reason": "", "entries": l.parse(owners, text, TODAY)}


def snapshot(root, text=ENTRY, owners=OWNERS, sha=SHA, reason=""):
    root.mkdir(exist_ok=True)
    (root / "state.json").write_text(json.dumps({"sha": sha, "reason": reason}))
    (root / "maintainers.toml").write_text(owners)
    (root / "repository.toml").write_text(text)
    return root


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = dict(engine_tests.POLICY)
        snapshot(self.root)

    def load(self, text=ENTRY, owners=OWNERS):
        snapshot(self.root, text, owners)
        return l.load(self.root, self.policy, TODAY)

    def test_status_guard_and_repository_wide_selection(self):
        for status, count in (("active", 1), ("proposed", 0), ("retired", 0)):
            with self.subTest(status=status):
                text = ENTRY.replace('status = "active"', f'status = "{status}"')
                if status == "retired":
                    text += 'retired_reason = "Superseded with evidence."\n'
                meta, section = l.select(self.load(text), ["src/app.py"])
                self.assertEqual(count, meta["count"])
                self.assertEqual(bool(count), bool(section))
        self.assertEqual(1, l.select(self.load(), [])[0]["count"])

    def test_approval_guard_uses_same_snapshot_maintainers(self):
        for text, owners in ((ENTRY.replace('approved_by = "maintainer"', 'approved_by = "outsider"'), OWNERS),
                             (ENTRY.replace('approved_by = "maintainer"\n', ''), OWNERS),
                             (ENTRY, 'maintainers = ["replacement"]')):
            with self.subTest(text=text, owners=owners):
                loaded = self.load(text, owners)
                self.assertEqual([], loaded["entries"])
                self.assertEqual("invalid data", loaded["reason"])
        self.assertEqual(1, len(self.load(ENTRY.replace('"maintainer"', '"Maintainer"'))["entries"]))

    def test_expiry_guard_inclusive_review_date(self):
        expired = ENTRY.replace("2099-01-01", "2026-10-30")
        self.assertEqual([], self.load(expired)["entries"])
        self.assertEqual(1, len(self.load(ENTRY.replace("2099-01-01", "2026-10-31"))["entries"]))

    def test_path_globs_and_repository_wide_entries(self):
        text = ENTRY + 'paths = ["src/**", "docs/*.md"]\n'
        scoped = self.load(text)
        self.assertEqual(1, l.select(scoped, ["src/nested/app.py"])[0]["count"])
        self.assertEqual(1, l.select(scoped, ["docs/guide.md"])[0]["count"])
        self.assertEqual(0, l.select(scoped, ["src-other/app.py", "SRC/app.py"])[0]["count"])
        self.assertEqual(0, l.select(scoped, [])[0]["count"])
        self.assertEqual(1, l.select(self.load(), ["unrelated.txt"])[0]["count"])

    def test_schema_guard_rejects_invalid_whole_file_without_private_errors(self):
        cases = [
            "PRIVATE_BAD_TOML = [", ENTRY + 'unknown = "private value"\n',
            ENTRY + ENTRY, ENTRY.replace('kind = "rejected-finding"', 'kind = []'),
            ENTRY.replace('status = "active"', 'status = false'),
            ENTRY.replace('source = "https:', 'source = "http:'),
            ENTRY.replace('text = "This pattern is fine; evidence is checked in the parser."', 'text = ""'),
            ENTRY.replace("review_by = 2099-01-01", 'review_by = "2099-01-01"'),
            ENTRY.replace("review_by = 2099-01-01", "review_by = 2025-01-01"),
            ENTRY.replace('taught_by = "contributor"', 'taught_by = "-bad-"'),
            ENTRY + "paths = []\n", ENTRY + "retired_reason = \"unexpected\"\n",
            ENTRY.replace('status = "active"', 'status = "retired"'),
            'learning = 1', 'other = "private"']
        for text in cases:
            with self.subTest(text=text):
                loaded = self.load(text)
                self.assertEqual([], loaded["entries"])
                self.assertEqual("invalid data", loaded["reason"])
                self.assertNotIn("PRIVATE", l.summary(l.select(loaded, [])[0]))
        for owners in ('maintainers = []', 'maintainers = ["-bad"]', OWNERS + 'extra = 1'):
            self.assertEqual("invalid data", self.load(owners=owners)["reason"])

    def test_raw_size_guard_derived_from_prompt_allowance_and_combined_files(self):
        limit = l.allowance(self.policy)
        self.assertEqual(self.policy["max_model_tokens"] - self.policy["max_output_tokens"], limit)
        snapshot(self.root, ENTRY + "#" * limit, OWNERS)
        loaded = l.load(self.root, self.policy, TODAY)
        self.assertEqual([], loaded["entries"])
        self.assertEqual("input too large", loaded["reason"])

    def test_nonregular_and_symlink_files_are_rejected(self):
        (self.root / "repository.toml").unlink()
        target = self.root / "other"; target.write_text(ENTRY)
        (self.root / "repository.toml").symlink_to(target)
        self.assertEqual("invalid data", l.load(self.root, self.policy, TODAY)["reason"])
        (self.root / "repository.toml").unlink()
        os.mkfifo(self.root / "repository.toml")
        self.assertEqual("invalid data", l.load(self.root, self.policy, TODAY)["reason"])

    def test_model_process_environment_has_no_learning_or_github_credentials(self):
        with patch.dict(os.environ, {"LEARNINGS_TOKEN": "fixture", "AGENT_REVIEW_LEARNINGS_APP_KEY": "fixture",
                                     "AGENT_REVIEW_LEARNINGS_APP_ID": "fixture", "GITHUB_TOKEN": "fixture"}):
            environment = e.clean_env()
        for name in ("LEARNINGS_TOKEN", "AGENT_REVIEW_LEARNINGS_APP_KEY",
                     "AGENT_REVIEW_LEARNINGS_APP_ID", "GITHUB_TOKEN"):
            self.assertNotIn(name, environment)

    def test_metadata_guard_accepts_only_bounded_public_fields(self):
        meta, _ = l.select(data(), ["app.py"])
        self.assertEqual(meta, l.validate_metadata(meta, self.policy))
        for change in ({"text": "private"}, {"sha": "malformed"}, {"digest": "bad"},
                       {"count": True}, {"bytes": -1}, {"bytes": l.allowance(self.policy) + 1},
                       {"reason": "exception private"}, {"count": 0}, {"sha": "none"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                l.validate_metadata(dict(meta, **change), self.policy)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths, self.requests = [], []
        self.payload = ENTRY.encode()
        self.error = None
        def request(req, timeout):
            self.requests.append(req)
            self.paths.append(req.full_url)
            self.assertEqual(engine_tests.POLICY["ai_timeout"], timeout)
            self.assertEqual("Bearer private-fixture-token", req.get_header("Authorization"))
            if "/contents/repos/" in req.full_url:
                if self.error:
                    raise self.error
                return io.BytesIO(self.payload)
            if "/contents/" in req.full_url:
                return io.BytesIO(OWNERS.encode())
            if "/commits/" in req.full_url:
                return io.BytesIO(json.dumps({"sha": SHA}).encode())
            return io.BytesIO(b'{"default_branch":"main"}')
        self.opener = NS(open=request)

    def fetch(self, token="private-fixture-token", credentials=True):
        l.fetch(self.root, "example/project", token, credentials, engine_tests.POLICY, opener=self.opener)
        return l.load(self.root, engine_tests.POLICY, TODAY)

    def test_fetch_pins_both_exact_files_to_one_default_branch_commit(self):
        loaded = self.fetch()
        prefix = "https://api.github.com/repos/pipelab-org/agent-review-learnings"
        self.assertEqual([prefix, prefix + "/commits/main",
                          prefix + "/contents/maintainers.toml?ref=" + SHA,
                          prefix + "/contents/repos/example/project.toml?ref=" + SHA], self.paths)
        self.assertEqual(SHA, loaded["sha"])
        self.assertEqual(1, len(loaded["entries"]))
        for name in ("maintainers.toml", "repository.toml", "state.json"):
            self.assertEqual(0o600, (self.root / name).stat().st_mode & 0o777)

    def test_missing_secrets_and_failed_mint_never_fetch_or_block(self):
        for token, credentials, reason in (("", False, "credentials unavailable"),
                                            ("", True, "token unavailable"),
                                            ("private-fixture-token", False, "credentials unavailable")):
            self.paths.clear()
            loaded = self.fetch(token, credentials)
            self.assertEqual([], self.paths)
            self.assertEqual(reason, loaded["reason"])

    def test_404_invalid_toml_and_network_errors_are_fixed_reasons(self):
        for error, expected in ((urllib.error.HTTPError("private", 404, "private", {}, None),
                                 "repository file unavailable"),
                                (OSError("private credential"), "fetch failed")):
            self.error = error
            loaded = self.fetch()
            self.assertEqual(expected, loaded["reason"])
            self.assertNotIn("private", l.summary(l.select(loaded, [])[0]))
        self.error = None
        self.payload = b"PRIVATE_CONTENT = ["
        self.assertEqual("invalid data", self.fetch()["reason"])
        self.assertFalse((self.root / "repository.toml").exists())

    def test_network_size_guard_and_no_redirect_token_forwarding(self):
        self.payload = b"#" * l.allowance(engine_tests.POLICY)
        self.assertEqual("input too large", self.fetch()["reason"])
        self.assertEqual("input too large", json.loads((self.root / "state.json").read_text())["reason"])
        self.assertFalse((self.root / "repository.toml").exists())
        self.assertIsNone(l.NoRedirect().redirect_request(
            None, None, 302, "redirect", {}, "https://example.invalid/private"))

    def test_invalid_snapshot_commit_and_lookup_key_never_admit_data(self):
        for repository in ("../another/project", "example/project/alias"):
            l.fetch(self.root, repository, "private-fixture-token", True,
                    engine_tests.POLICY, opener=self.opener)
            self.assertEqual("invalid data", json.loads((self.root / "state.json").read_text())["reason"])
            self.assertEqual([], self.paths)
        original = self.opener.open
        def bad_commit(request, timeout):
            if "/commits/" in request.full_url:
                return io.BytesIO(b'{"sha":"main"}')
            return original(request, timeout)
        self.opener.open = bad_commit
        self.assertEqual("invalid data", self.fetch()["reason"])
        self.assertFalse(any("/contents/" in p for p in self.paths))

    def test_empty_fetch_output_and_malformed_state_proceed_without_context(self):
        with patch.dict(os.environ, {"REVIEW_LEARNINGS_DIR": ""}):
            self.assertEqual("fetch failed", l.from_env(engine_tests.POLICY)["reason"])
        snapshot(self.root, sha="bad")
        self.assertEqual("invalid data", l.load(self.root, engine_tests.POLICY)["reason"])


class PromptTests(unittest.TestCase):
    def setUp(self):
        fixture = engine_tests.BundleTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture

    def test_both_providers_label_id_kind_text_and_source_as_untrusted_context(self):
        bundle = self.fixture.bundle(learning_data=data())
        for prompt in (e.claude_prompt(bundle, "bundle"), e.claude_prompt(bundle, "callers"), e.codex_prompt(bundle)):
            self.assertIn(l.FRAMING, prompt)
            for key in ("id", "kind", "text", "source"):
                self.assertIn(data()["entries"][0][key], prompt)
            self.assertEqual(1, prompt.count(l.FRAMING))

    def test_exact_sizing_counts_learnings_and_public_rebuild_uses_only_metadata(self):
        fx = self.fixture
        lines = [f"line {i}" for i in range(3000)]
        (fx.root / "app.py").write_text("\n".join(lines) + "\n")
        fx.base = fx.commit()
        lines[1500] = "changed"
        (fx.root / "app.py").write_text("\n".join(lines) + "\n")
        fx.commit()
        policy = dict(engine_tests.POLICY, max_model_tokens=40000, max_output_tokens=1000)
        self.assertIsNotNone(fx.bundle(policy)["files"][0]["content"])
        learning = data(ENTRY.replace("This pattern is fine; evidence is checked in the parser.",
                                     "Check the parser with evidence. " * 300))
        bundle = fx.bundle(policy, learning_data=learning)
        self.assertEqual(1, bundle["learnings"]["count"])
        self.assertIsNone(bundle["files"][0]["content"])
        for prompt in (e.claude_prompt(bundle, "bundle"), e.claude_prompt(bundle, "callers"), e.codex_prompt(bundle)):
            self.assertLessEqual(e.token_bound(prompt + json.dumps(e.OUTPUT_SCHEMA)), l.allowance(policy))
        public = dict(bundle); public.pop("learnings_text")
        self.assertEqual(public, fx.bundle(policy, learning_metadata=bundle["learnings"]))

    def test_optional_context_never_blocks_an_otherwise_fitting_bundle(self):
        fx = self.fixture
        (fx.root / "app.py").write_text("changed\n")
        fx.commit()
        policy = dict(engine_tests.POLICY, max_model_tokens=14000, max_output_tokens=1000)
        fx.bundle(policy)
        learning = data(ENTRY.replace("This pattern is fine; evidence is checked in the parser.", "x" * 11000))
        bundle = fx.bundle(policy, learning_data=learning)
        self.assertEqual("prompt allowance exceeded", bundle["learnings"]["reason"])
        self.assertNotIn("learnings_text", bundle)
        self.assertEqual(bundle, fx.bundle(policy, learning_metadata=bundle["learnings"]))
        huge = data(ENTRY.replace("This pattern is fine; evidence is checked in the parser.", "x" * 14000))
        self.assertEqual("input too large", fx.bundle(policy, learning_data=huge)["learnings"]["reason"])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        fixture = job_tests.JobTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fx = fixture
        self.directory = snapshot(fixture.root / "learnings")
        self.addCleanup(patch.stopall)

    def enable(self):
        patch.dict(os.environ, {"REVIEW_LEARNINGS_DIR": str(self.directory)}).start()

    def test_never_filter_findings_and_no_private_text_in_public_artifact_or_summary(self):
        fx = self.fx
        findings = [engine_tests.finding(path="app.py")]
        fx.build(findings)
        original = json.loads(fx.artifact.read_text())["result"]["findings"]
        self.enable()
        fx.build(findings)
        artifact = json.loads(fx.artifact.read_text())
        self.assertEqual(original, artifact["result"]["findings"])
        self.assertNotIn(data()["entries"][0]["text"], fx.artifact.read_text())
        # Publication rebuilds at the exact reserved size with no private checkout,
        # no installation token and no model credential.
        identity = artifact["plan"]["context"]["learnings"]
        with patch.dict(os.environ, {"REVIEW_LEARNINGS_DIR": "", "REVIEW_LEARNINGS_IDENTITY": json.dumps(identity)}):
            j.publish_artifact(fx.artifact, fx.head_dir)
        summary = next(c.body for c in fx.pr.comments.values() if "<summary>Review details" in c.body)
        self.assertIn("Learnings: 1 loaded from agent-review-learnings@aaaaaaa", summary)
        self.assertNotIn(data()["entries"][0]["text"], summary)

    def test_context_policy_identity_changes_on_revision_and_expiry(self):
        fx = self.fx
        policy, depth = j.settings()
        baseline, _ = j.context(policy, depth)
        self.enable()
        first, _ = j.context(policy, depth)
        self.assertFalse(f.same_policy(baseline, first))
        snapshot(self.directory, sha="b" * 40)
        second, _ = j.context(policy, depth)
        self.assertFalse(f.same_policy(first, second))
        snapshot(self.directory, ENTRY.replace("2099-01-01", "2026-01-02"))
        expired, _ = j.context(policy, depth)
        self.assertFalse(f.same_policy(first, expired))
        self.assertTrue(f.same_policy(first, dict(first, head="e" * 40)))

    def test_reply_prompt_includes_context_but_its_artifact_does_not(self):
        fx = self.fx
        self.enable()
        proof, root = fx.prior_threads()
        plan = fx.reply_plan(proof, root)
        policy, _ = j.settings()
        with patch.object(e, "run_claude", return_value=fx.answer("none")) as run:
            j.infer_reply(policy, plan, fx.head_dir)
        self.assertIn(l.FRAMING, run.call_args.kwargs["prompt"])
        self.assertNotIn(l.FRAMING, plan["prompt"])
        self.assertEqual(1, plan["learnings"]["count"])

    def test_reply_omits_context_when_optional_path_discovery_cannot_fit(self):
        self.enable()
        policy, depth = j.settings()
        ctx, _ = j.context(policy, depth)
        with patch.object(e, "git", side_effect=e.ReviewAbort("Source exceeds allowance")):
            meta, text = j.reply_learnings(ctx, policy, self.fx.head_dir)
        self.assertEqual("input too large", meta["reason"])
        self.assertEqual("", text)

    def test_reply_omits_oversized_context_and_detects_changed_prepared_context(self):
        fx = self.fx
        self.enable()
        proof, root = fx.prior_threads()
        policy, _ = j.settings()
        meta, _ = l.select(data(), ["app.py"])
        with patch.object(j, "reply_learnings", return_value=(meta, "x" * l.allowance(policy))):
            plan = fx.reply_plan(proof, root)
        self.assertEqual("prompt allowance exceeded", plan["learnings"]["reason"])
        snapshot(self.directory)
        changed = copy.deepcopy(plan)
        changed["learnings"] = dict(meta, digest="0" * 64)
        with patch.object(e, "run_claude", return_value=fx.answer("none")) as provider:
            with self.assertRaises(e.ReviewAbort):
                j.infer_reply(policy, changed, fx.head_dir)
        provider.assert_not_called()

    def test_fixed_summary_for_no_learnings_and_identity_tampering(self):
        fx = self.fx
        self.enable()
        snapshot(self.directory, reason="token unavailable", sha="none")
        fx.build()
        j.publish_artifact(fx.artifact, fx.head_dir)
        summary = next(c.body for c in fx.pr.comments.values() if "<summary>Review details" in c.body)
        self.assertIn("Learnings: none loaded (token unavailable)", summary)
        with patch.dict(os.environ, {"REVIEW_LEARNINGS_IDENTITY": '{"sha":"bad","digest":"bad"}'}):
            with self.assertRaises(e.ReviewAbort):
                j.context(*j.settings())

    def test_invalid_or_unavailable_learnings_never_block_the_review_pipeline(self):
        fx = self.fx
        self.enable()
        policy, _ = j.settings()
        cases = (("PRIVATE_BROKEN = [", "", "invalid data"),
                 (ENTRY + "#" * l.allowance(policy), "", "input too large"),
                 (ENTRY, "credentials unavailable", "credentials unavailable"),
                 (ENTRY, "repository file unavailable", "repository file unavailable"),
                 (ENTRY, "fetch failed", "fetch failed"))
        for text, reason, expected in cases:
            with self.subTest(reason=expected):
                # Track only writes performed by this iteration's read-only stages.
                fx.pr.writes.clear()
                snapshot(self.directory, text, reason=reason)
                fx.build([engine_tests.finding(path="app.py")])
                artifact = json.loads(fx.artifact.read_text())
                self.assertEqual(expected, artifact["plan"]["bundle"]["learnings"]["reason"])
                self.assertEqual(1, len(artifact["result"]["findings"]))
                j.publish_artifact(fx.artifact, fx.head_dir)

    def test_artifact_rejects_private_fields_and_another_learning_commit(self):
        fx = self.fx
        self.enable()
        fx.build()
        artifact = json.loads(fx.artifact.read_text())
        for change in ({"text": "private"}, {"sha": "b" * 40}):
            changed = copy.deepcopy(artifact)
            changed["plan"]["bundle"]["learnings"].update(change)
            fx.artifact.write_text(json.dumps(changed))
            with self.assertRaises((e.ReviewAbort, ValueError)):
                j.publish_artifact(fx.artifact, fx.head_dir)
        self.assertEqual([], fx.pr.writes)


class WorkflowTests(unittest.TestCase):
    def workflow(self):
        return yaml.load((ROOT / ".github/workflows/agent-review-reusable.yaml").read_text(), Loader=yaml.BaseLoader)

    def test_token_is_scoped_optional_and_not_in_any_inference_step_environment(self):
        workflow = self.workflow()
        for name in ("AGENT_REVIEW_LEARNINGS_APP_ID", "AGENT_REVIEW_LEARNINGS_APP_KEY"):
            self.assertEqual("false", workflow["on"]["workflow_call"]["secrets"][name]["required"])
        for name in ("review", "reply"):
            job = workflow["jobs"][name]
            steps = job["steps"]
            token = next(s for s in steps if s.get("id") == "learnings-token")
            fetch = next(s for s in steps if s.get("id") == "learnings-fetch")
            prepare = next(s for s in steps if s.get("id") == "plan")
            inference = next(s for s in steps if "review_jobs.py infer " in s.get("run", ""))
            self.assertEqual("true", token["continue-on-error"])
            self.assertEqual("true", fetch["continue-on-error"])
            self.assertEqual("actions/create-github-app-token@64ab729a6294000f5aa4a4bc0b73d1f32bc45510", token["uses"])
            self.assertEqual("pipelab-org", token["with"]["owner"])
            self.assertEqual("agent-review-learnings", token["with"]["repositories"])
            self.assertEqual({"permission-contents": "read", "permission-metadata": "read"},
                             {k: v for k, v in token["with"].items() if k.startswith("permission-")})
            self.assertIn("available == 'true'", token["if"])
            self.assertLess(steps.index(token), steps.index(fetch))
            self.assertLess(steps.index(fetch), steps.index(prepare))
            self.assertLess(steps.index(prepare), steps.index(inference))
            self.assertEqual("${{ steps.learnings-token.outputs.token }}", fetch["env"]["LEARNINGS_TOKEN"])
            for step in steps:
                if step is fetch:
                    continue
                self.assertNotIn("learnings-token.outputs.token", json.dumps(step))
            self.assertNotIn("LEARNINGS_TOKEN", json.dumps(job.get("env", {})))
            for step in (prepare, inference):
                env = json.dumps({**job.get("env", {}), **step.get("env", {})})
                self.assertNotIn("LEARNINGS_TOKEN", env)
                self.assertNotIn("AGENT_REVIEW_LEARNINGS_APP_KEY", env)
        for name, source in (("publish", "review"), ("publish-reply", "reply")):
            job = workflow["jobs"][name]
            env = next(s["env"] for s in job["steps"] if "review_jobs.py publish " in s.get("run", ""))
            self.assertEqual("${{ needs." + source + ".outputs.learnings_identity }}", env["REVIEW_LEARNINGS_IDENTITY"])


if __name__ == "__main__":
    unittest.main()
