# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the read/infer/write boundary with real validation and a fake GitHub API."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import yaml

from scripts import review_engine as e, review_jobs as j, review_followups as f
from scripts.test_review_engine import POLICY, receipt, finding


class Comment:
    def __init__(self, id, body, user, raw=None, **extra):
        self.id, self.body, self.user = id, body, user
        self.raw_data = {"id": id, "body": body, **(raw or {})}
        self.__dict__.update(extra)
        self.html_url = "https://github.com/example/project/pull/1#comment"

    def edit(self, body):
        self.body = body
        self.raw_data["body"] = body


class Pull:
    def __init__(self, repo, head, base):
        self.number, self.state = 1, "open"
        self.head = NS(sha=head, repo=NS(full_name=repo, fork=False))
        self.base = NS(sha=base, repo=NS(full_name=repo))
        self.comments, self.roots, self.reviews, self.writes = {}, [], {}, []

    def get_issue_comments(self): return list(self.comments.values())
    def get_issue_comment(self, id): return self.comments[id]
    def get_review_comments(self): return list(self.roots)
    def get_review_comment(self, id): return next(c for c in self.roots if c.id == id)
    def get_review(self, id): return self.reviews[id]

    def create_issue_comment(self, body):
        self.writes.append(("issue", body))
        comment = Comment(len(self.comments) + 100, body, NS(**f.BOT))
        self.comments[comment.id] = comment
        return comment

    def create_review(self, **kwargs):
        self.writes.append(("review", kwargs))
        id = len(self.reviews) + 200
        review = NS(id=id, user=NS(**f.BOT), commit_id=kwargs["commit"].sha, state="COMMENTED")
        self.reviews[id] = review
        for request in kwargs["comments"]:
            self.roots.append(Comment(len(self.roots) + 300, request["body"], NS(**f.BOT),
                         {"pull_request_review_id": id}, path=request["path"], commit_id=kwargs["commit"].sha))
        return review

    def create_review_comment_reply(self, root, body):
        self.writes.append(("reply", body))
        comment = Comment(len(self.roots) + 400, body, NS(**f.BOT), {"in_reply_to_id": root},
                          path="app.py", commit_id=self.head.sha)
        self.roots.append(comment)
        return comment


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.head_dir = self.root / "head"; self.head_dir.mkdir()
        def git(*args):
            return subprocess.check_output(["git", "-C", str(self.head_dir), *args], text=True).strip()
        git("init", "-q", "-b", "main")
        git("config", "user.email", "fixture@example.invalid")
        git("config", "user.name", "fixture")
        (self.head_dir / "app.py").write_text("old\n")
        (self.head_dir / "helper.py").write_text("helper\n")
        git("add", "."); git("commit", "-qm", "base")
        base = git("rev-parse", "HEAD")
        (self.head_dir / "app.py").write_text("new\n")
        git("add", "."); git("commit", "-qm", "head")
        head = git("rev-parse", "HEAD")
        self.pr = Pull("example/project", head, base)
        self.repo = NS(full_name="example/project", get_pull=lambda _: self.pr,
                       owner=NS(login="example", type="Organization"))
        from github.Commit import Commit
        self.repo.get_commit = lambda sha: Commit(NS(per_page=30, is_not_lazy=False), {}, {"sha": sha}, completed=True)
        self.repo.get_workflow_run = lambda _: (_ for _ in ()).throw(ValueError("no old artifact"))
        self.repo.compare = lambda older, newer: NS(base_commit=NS(sha=older),
                merge_base_commit=NS(sha=base if older == base else older),
                status="identical" if older == newer else "ahead")
        self.profile = self.root / "profile.toml"
        self.profile.write_text('[review]\ninstructions="Review correctness"\n[ignore]\nglob=[]\n')
        self.config = self.root / "config.toml"
        # Explicit fixture values only, never a shipped operator config.
        self.config.write_text('[config]\nmax_model_tokens=90000\nmax_output_tokens=16000\nai_timeout=300\n' +
            "\n".join(f'[review_depth.{d}]\nmodel="{POLICY["model"]}"\nreasoning_effort="medium"\nmax_turns=10\nmax_budget_usd=5.0\n'
                      for d in ("default", "deep", "codex", "codex-deep")))
        self.event = {"action": "created", "repository": {"full_name": "example/project"},
                      "issue": {"number": 1, "pull_request": {}},
                      "comment": {"id": 9, "body": "/review", "user": f.JOSH, "author_association": "MEMBER",
                                  "created_at": "2026-10-28T00:00:00Z", "updated_at": "2026-10-28T00:00:00Z"}}
        self.event["issue"]["pull_request"]["url"] = "https://api.github.com/repos/example/project/pulls/1"
        self.event_path = self.root / "event.json"
        self.event_path.write_text(json.dumps(self.event))
        self.pr.comments[9] = Comment(9, "/review", NS(**f.JOSH), self.event["comment"])
        self.env = {"PR_NUMBER": "1", "GITHUB_REPOSITORY": "example/project", "GITHUB_TOKEN": "fixture",
                    "GITHUB_ACTOR": f.JOSH["login"], "GITHUB_TRIGGERING_ACTOR": f.JOSH["login"],
                    "GITHUB_ACTOR_ID": str(f.JOSH["id"]), "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_ID": "123",
                    "GITHUB_EVENT_NAME": "issue_comment", "GITHUB_EVENT_PATH": str(self.event_path),
                    "REVIEW_HEAD": head, "REVIEW_BASE": base, "REVIEW_SOURCE_SHA": "c" * 40,
                    "REVIEW_PROFILE_SHA": "d" * 40, "REVIEW_PROFILE_PATH": str(self.profile), "REVIEW_DEPTH": "default",
                    "REVIEW_COMPLETION_PATH": str(self.root / "completion.json")}
        self.plan, self.artifact = self.root / "plan.json", self.root / "findings.json"
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, self.env).start()
        patch.object(e, "CONFIG", self.config).start()
        patch.object(j, "github", return_value=NS(get_repo=lambda _: self.repo)).start()

    def build(self, findings=None):
        j.prepare(self.head_dir, self.plan)
        def run(policy, bundle, strategy, head):
            r = receipt(strategy); r["files_opened"] = ["app.py"]
            return {"findings": findings or [], "receipt": r}
        with patch.object(e, "run_claude", side_effect=run) as provider:
            j.infer(self.plan, self.artifact, self.head_dir)
        self.assertEqual(["bundle", "callers"], [c.args[2] for c in provider.call_args_list])
        self.assertEqual([], self.pr.writes, "read-only stages attempted a write")

    def test_summary_names_region_files_no_pass_opened(self):
        self.build([finding(path="app.py")])
        with patch.object(e, "unopened_regions", return_value=["docs/big.md"]) as unopened:
            j.publish_artifact(self.artifact, self.head_dir)
        self.assertEqual(2, len(unopened.call_args.args[1]))
        summary = next(c.body for c in self.pr.comments.values()
                       if "Reviewed base" in c.body)
        self.assertIn('Sent as changed regions only and not opened by any pass: `["docs/big.md"]`.', summary)

    def test_complete_passes_publish_inline_summary_and_verifiable_completion(self):
        self.build([finding(path="app.py")])
        j.publish_artifact(self.artifact, self.head_dir)
        record = json.loads((self.root / "completion.json").read_text())
        self.assertEqual(1, len(self.pr.roots))
        self.assertEqual(1, len(record["findings"]))
        self.assertEqual(2, len(record["receipt"]["passes"]))
        status = next(c for c in self.pr.comments.values() if c.body.startswith("<!-- agent-review-status:v1"))
        self.assertEqual(record, f.parse_record(status, self.pr.comments))
        self.assertEqual(self.env["REVIEW_HEAD"], record["roots"][0]["commit"])

    def test_actual_pygithub_review_method_receives_commit_object_and_serializes_anchors(self):
        from github.PullRequest import PullRequest
        self.build([finding(path="app.py")])
        pull = self.pr
        original = pull.create_review
        class Requester:
            per_page = 30  # PyGithub DEFAULT_PER_PAGE, fixture request shape
            is_not_lazy = False
            is_lazy = False
            def requestJsonAndCheck(self, method, url, input):
                self.sent = input
                # Feed the actual SDK's serialized HTTP payload into the fixture server.
                fixture = dict(input)
                fixture["commit"] = NS(sha=fixture.pop("commit_id"))
                review = original(**fixture)
                return {}, {"id": review.id, "user": f.BOT, "commit_id": review.commit_id,
                            "state": "COMMENTED", "url": url + "/" + str(review.id)}
        requester = Requester()
        pull._requester = requester
        pull.url = "https://api.github.com/repos/example/project/pulls/1"
        pull.create_review = lambda **kwargs: PullRequest.create_review(pull, **kwargs)
        j.publish_artifact(self.artifact, self.head_dir)
        self.assertEqual(self.env["REVIEW_HEAD"], requester.sent["commit_id"])
        self.assertEqual("RIGHT", requester.sent["comments"][0]["side"])
        self.assertTrue((self.root / "completion.json").is_file())

    def test_clean_is_published_only_with_complete_pass_receipts(self):
        self.build()
        j.publish_artifact(self.artifact, self.head_dir)
        record = json.loads((self.root / "completion.json").read_text())
        self.assertEqual([], record["findings"])
        self.assertEqual([], record["roots"])
        self.assertTrue(any("No findings reported" in c.body for c in self.pr.comments.values()))

    def test_summary_only_locations_do_not_get_fabricated_changed_line(self):
        self.build([finding(path="helper.py")])
        j.publish_artifact(self.artifact, self.head_dir)
        self.assertEqual([], self.pr.roots)
        record = json.loads((self.root / "completion.json").read_text())
        self.assertEqual("helper.py", record["findings"][0]["path"])

    def test_unconfirmed_locations_stay_in_summary_without_discarding_the_review(self):
        self.build([finding(path="removed.py"), finding(path="app.py", line=900, end_line=900), finding(path="app.py")])
        j.publish_artifact(self.artifact, self.head_dir)
        record = json.loads((self.root / "completion.json").read_text())
        self.assertEqual(3, len(record["findings"]))
        notes = {(f["path"], f["line"]): f["body"].count(j.UNVERIFIED_LOCATION_NOTE) for f in record["findings"]}
        self.assertEqual({("removed.py", 1): 1, ("app.py", 900): 1, ("app.py", 1): 0}, notes)
        anchored = {(getattr(c, "path", None), getattr(c, "line", None)) for c in self.pr.roots}
        self.assertNotIn(("removed.py", 1), anchored)
        self.assertNotIn(("app.py", 900), anchored)

    def test_unsafe_finding_path_still_aborts(self):
        with self.assertRaisesRegex(e.ReviewAbort, "Invalid finding"):
            self.build([finding(path="../escape.py")])
        self.assertEqual([], self.pr.writes)

    def test_artifact_receipt_context_and_bundle_tampering_never_publish(self):
        self.build()
        good = json.loads(self.artifact.read_text())
        mutations = []
        data = copy.deepcopy(good); data["result"]["receipt"]["passes"] = []; mutations.append(data)
        data = copy.deepcopy(good); data["plan"]["context"]["head"] = "f" * 40; mutations.append(data)
        data = copy.deepcopy(good); data["plan"]["bundle"]["files"] = []; mutations.append(data)
        data = copy.deepcopy(good); data["result"]["findings"] = [{"path": "app.py"}]; mutations.append(data)
        for data in mutations:
            self.artifact.write_text(json.dumps(data))
            with self.subTest(data=data), self.assertRaises((e.ReviewAbort, KeyError)):
                j.publish_artifact(self.artifact, self.head_dir)
            self.assertEqual([], self.pr.writes)

    def test_moved_fork_closed_and_edited_commands_fail_before_inference(self):
        for field, value in (("state", "closed"),):
            setattr(self.pr, field, value)
            with self.assertRaises(e.ReviewAbort): j.prepare(self.head_dir, self.plan)
            setattr(self.pr, field, "open")
        self.pr.head.repo.fork = True
        with self.assertRaises(e.ReviewAbort): j.prepare(self.head_dir, self.plan)
        self.pr.head.repo.fork = False
        self.pr.comments[9].body = "/review deep"
        with self.assertRaises(e.ReviewAbort): j.prepare(self.head_dir, self.plan)
        self.assertEqual([], self.pr.writes)

    def test_missing_artifact_and_truncated_output_fail(self):
        with self.assertRaises(FileNotFoundError): j.publish_artifact(self.artifact, self.head_dir)
        self.artifact.write_text('{"plan":')
        with self.assertRaises(e.ReviewAbort): j.publish_artifact(self.artifact, self.head_dir)
        self.assertEqual([], self.pr.writes)

    def test_ambiguous_or_partial_inline_publication_has_no_success_proof(self):
        self.build([finding(path="app.py")])
        original = self.pr.create_review
        def partial(**kwargs):
            review = original(**kwargs)
            self.pr.roots.clear()
            return review
        self.pr.create_review = partial
        with self.assertRaisesRegex(e.ReviewAbort, "publication verification failed"):
            j.publish_artifact(self.artifact, self.head_dir)
        self.assertFalse((self.root / "completion.json").exists())
        self.assertFalse(any("state=success" in c.body for c in self.pr.comments.values()))

    def test_publication_rechecks_commits_after_inference(self):
        self.build()
        self.pr.head.sha = "e" * 40
        with self.assertRaisesRegex(e.ReviewAbort, "moved"):
            j.publish_artifact(self.artifact, self.head_dir)
        self.assertEqual([], self.pr.writes)

    def test_reply_revalidates_provenance_and_deduplicates_same_trigger(self):
        self.build([finding(path="app.py")])
        j.publish_artifact(self.artifact, self.head_dir)
        proof = json.loads((self.root / "completion.json").read_text())
        root = self.pr.roots[0]
        raw = {"id": 500, "body": "Which state is lost?", "in_reply_to_id": root.id,
               "pull_request_review_id": root.raw_data["pull_request_review_id"],
               "commit_id": self.pr.head.sha, "original_commit_id": self.pr.head.sha,
               "path": root.path, "created_at": "2026-10-28T01:00:00Z", "updated_at": "2026-10-28T01:00:00Z",
               "pull_request_url": "https://api.github.com/repos/example/project/pulls/1"}
        live = Comment(500, raw["body"], NS(**f.JOSH), raw, path=root.path, commit_id=self.pr.head.sha)
        self.pr.roots.append(live)
        event = {"action": "created", "repository": {"full_name": self.repo.full_name},
                 "pull_request": {"number": 1}, "comment": {**raw, "user": f.JOSH}}
        self.event_path.write_text(json.dumps(event))
        os.environ["GITHUB_EVENT_NAME"] = "pull_request_review_comment"
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        with patch.object(f, "verified_records", return_value={123: proof}):
            j.prepare(self.head_dir, self.plan, reply=True)
            plan = json.loads(self.plan.read_text())
            r = receipt("reply"); r["files_opened"] = []
            j.publish_reply(self.repo, self.pr, ctx, policy, plan,
                            {"answer": "The resolved local value is lost during the copy.", "receipt": r})
            count = len(self.pr.writes)
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, None)
            self.assertEqual(count, len(self.pr.writes))
            self.assertTrue(any("The resolved local value" in c.body for c in self.pr.roots))

    def summary(self):
        return next(c.body for c in self.pr.comments.values()
                    if f.RESULT.search(c.body) and not c.body.startswith("<!-- agent-review-status:v1"))

    def test_summary_is_a_table_and_inline_text_is_not_repeated(self):
        self.build([finding(path="app.py", mechanism="Inline mechanism text"),
                    finding(path="helper.py", title="Related helper", mechanism="Helper mechanism text",
                            confidence="medium")])
        j.publish_artifact(self.artifact, self.head_dir)
        summary = self.summary()
        self.assertTrue(summary.startswith("**AI review:** 2 findings (1 high, 1 medium)."))
        self.assertIn("| high | Missing state transfer | `app.py:1` | inline thread |", summary)
        self.assertIn("| medium | Related helper | `helper.py:1` | below |", summary)
        self.assertNotIn("Inline mechanism text", summary)
        self.assertIn("Helper mechanism text", summary.split("<details>")[0])
        self.assertIn("Reviewed base", summary.split("<details>")[1])

    def test_table_cells_cannot_break_the_table(self):
        self.build([finding(path="app.py", title="a | b\nc")])
        j.publish_artifact(self.artifact, self.head_dir)
        self.assertIn("| a \\| b c |", self.summary())

    def test_table_location_and_raw_html_title_keep_the_row_intact(self):
        title = "literal </td></tr></table><details><summary>Hidden</summary>"
        self.build([finding(path="helper|table.py", title=title)])
        j.publish_artifact(self.artifact, self.head_dir)
        row = next(line for line in self.summary().splitlines() if line.startswith("| high |"))
        self.assertIn("| `helper\\|table.py:1` | below |", row)
        self.assertNotIn("<", row)
        self.assertIn("&lt;/td&gt;", row)
        self.assertEqual(r"a\\\|b", j.cell(r"a\|b"))
        self.assertEqual("`` a`b ``", j.code("a`b"))
        self.assertEqual(r"`a\nb`", j.code("a\nb"))

    def test_overlapping_findings_share_one_thread_and_keep_both_records(self):
        self.build([finding(path="app.py", title="First", mechanism="One mechanism"),
                    finding(path="app.py", title="Second", mechanism="Another mechanism", confidence="low")])
        j.publish_artifact(self.artifact, self.head_dir)
        self.assertEqual(1, len(self.pr.roots))
        body = self.pr.roots[0].body
        self.assertTrue(body.startswith("**2 findings on these lines**"))
        self.assertIn("One mechanism", body)
        self.assertIn("Another mechanism", body)
        record = json.loads((self.root / "completion.json").read_text())
        self.assertEqual(2, len(record["findings"]))
        self.assertEqual(1, len(record["roots"]))

    def test_overlap_groups_merge_transitively_and_keep_separate_ranges_apart(self):
        items = [{"path": "a", "line": 1, "end_line": 3}, {"path": "a", "line": 10, "end_line": 12},
                 {"path": "a", "line": 3, "end_line": 10}, {"path": "b", "line": 1, "end_line": 3},
                 {"path": "a", "line": 20, "end_line": 20}]
        self.assertEqual([[0, 1, 2], [3], [4]], j.overlap_groups(items, [0, 1, 2, 3, 4]))
        self.assertEqual([[0], [1]], j.overlap_groups(items, [0, 1]))
        merged = j.inline_comment([dict(items[0], title="x", body="y"), dict(items[2], title="z", body="w")])
        self.assertEqual((1, 10), (merged["start_line"], merged["line"]))

    def test_success_collapses_status_and_earlier_reviewer_comments_only(self):
        calls = []
        self.pr.requester = NS(graphql_named_mutation=lambda name, data, schema: calls.append((name, data)))
        old = Comment(50, "<!-- agent-review-notice:v1 run=1 -->\nReview incomplete", NS(**f.BOT), {"node_id": "OLD"})
        human = Comment(51, "<!-- agent-review-notice:v1 run=1 --> copied", NS(**f.JOSH), {"node_id": "HUMAN"})
        self.pr.comments.update({50: old, 51: human})
        self.build([finding(path="app.py")])
        j.publish_artifact(self.artifact, self.head_dir)
        status = next(c for c in self.pr.comments.values() if c.body.startswith("<!-- agent-review-status:v1"))
        summary = next(c for c in self.pr.comments.values() if f.RESULT.search(c.body) and c is not status)
        hidden = {data["subjectId"] for _, data in calls}
        self.assertEqual({"OLD"}, hidden - {None})
        self.assertEqual({"minimizeComment"}, {name for name, _ in calls})
        self.assertTrue(all(data["classifier"] == "OUTDATED" for _, data in calls))
        status.raw_data["node_id"], summary.raw_data["node_id"] = "STATUS", "SUMMARY"
        later = Comment(status.id + 50, "<!-- agent-review-notice:v1 run=9 -->", NS(**f.BOT), {"node_id": "LATER"})
        self.pr.comments[later.id] = later
        calls.clear()
        j.collapse_earlier(self.pr, status.id, summary.id)
        self.assertEqual({"OLD"}, {data["subjectId"] for _, data in calls})

    def test_collapse_refusal_never_fails_a_published_review(self):
        def refuse(*_):
            raise RuntimeError("forbidden")
        self.pr.requester = NS(graphql_named_mutation=refuse)
        self.pr.comments[50] = Comment(50, "<!-- agent-review-notice:v1 run=1 -->", NS(**f.BOT), {"node_id": "OLD"})
        self.build([finding(path="app.py")])
        j.publish_artifact(self.artifact, self.head_dir)
        self.assertTrue((self.root / "completion.json").is_file())

    def test_failed_completion_upload_keeps_the_notifier_warning_visible(self):
        import io
        hidden = set()
        create = self.pr.create_issue_comment
        def create_with_node(body):
            comment = create(body)
            comment.raw_data["node_id"] = str(comment.id)
            return comment
        self.pr.create_issue_comment = create_with_node
        self.pr.requester = NS(graphql_named_mutation=lambda name, data, schema: hidden.add(data["subjectId"]))
        self.build([finding(path="app.py")])
        j.publish_artifact(self.artifact, self.head_dir)
        status = next(c for c in self.pr.comments.values() if c.body.startswith("<!-- agent-review-status:v1"))

        # Run the shipped notifier as it runs after the completion upload fails.
        workflow = yaml.load((Path(__file__).resolve().parents[1] /
                              ".github/workflows/agent-review-reusable.yaml").read_text(), Loader=yaml.BaseLoader)
        script = workflow["jobs"]["notify-failure"]["steps"][0]["run"].split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        def request(req):
            if req.full_url.endswith("/pulls/1"):
                value = {"state": "open", "head": {"sha": self.pr.head.sha, "repo": {"full_name": self.repo.full_name, "fork": False}},
                         "base": {"sha": self.pr.base.sha, "repo": {"full_name": self.repo.full_name}}}
            elif req.method == "GET":
                value = [{"id": c.id, "body": c.body, "user": vars(c.user)} for c in self.pr.comments.values()]
            else:
                self.assertEqual("PATCH", req.method)
                self.assertTrue(req.full_url.endswith("/" + str(status.id)))
                status.edit(json.loads(req.data)["body"])
                value = {"id": status.id}
            return io.BytesIO(json.dumps(value).encode())
        with patch.dict(os.environ, {"GH_TOKEN": "fixture", "ADMISSION_RESULT": "success"}), \
                patch("urllib.request.urlopen", side_effect=request):
            exec(compile(script, "<trusted-notifier>", "exec"), {})
        self.assertIn("Treat this run as incomplete", status.body)
        self.assertNotIn(str(status.id), hidden)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load((e.ROOT / ".github/workflows/agent-review-reusable.yaml").read_text())

    def test_inference_has_no_write_permissions_and_publish_has_no_provider_secrets(self):
        for name in ("review", "reply"):
            job = self.workflow["jobs"][name]
            self.assertTrue(all(value == "read" for value in job["permissions"].values()))
            infer = next(s for s in job["steps"] if s.get("name") == "Review the admitted diff")
            self.assertNotIn("GITHUB_TOKEN", infer["env"])
            self.assertNotIn("ORG_MEMBERS_TOKEN", infer["env"])
            self.assertNotIn("container", job)
        for name in ("publish", "publish-reply"):
            text = json.dumps(self.workflow["jobs"][name])
            self.assertNotIn("ANTHROPIC_API_KEY", text)
            self.assertNotIn("OPENAI_API_KEY", text)
            self.assertIn("Download this run's inference artifact", text)

    def test_codex_depth_step_never_receives_the_anthropic_key(self):
        # The parent process environment is readable from a same-user child, so the
        # step itself, not only the engine's subprocess environment, must omit it.
        infer = next(s for s in self.workflow["jobs"]["review"]["steps"] if s.get("name") == "Review the admitted diff")
        self.assertEqual("${{ !startsWith(env.REVIEW_DEPTH, 'codex') && secrets.ANTHROPIC_API_KEY || '' }}",
                         infer["env"]["ANTHROPIC_API_KEY"])
        # Every Codex depth the workflow can select is covered by the prefix test above.
        literals = self.workflow["jobs"]["review"]["env"]["REVIEW_DEPTH"].split("'")[1::2]
        depths = dict(zip(literals[0:-1:2], literals[1::2]))
        self.assertEqual({"/review codex deep": "codex-deep", "/review codex": "codex", "/review deep": "deep"}, depths)
        self.assertEqual({d for d in e.DEPTHS if e.DEPTHS[d][0] == "codex"},
                         {d for d in depths.values() if d.startswith("codex")})
        for step in self.workflow["jobs"]["review"]["steps"]:
            if step.get("name") != "Review the admitted diff":
                self.assertNotIn("ANTHROPIC_API_KEY", json.dumps(step.get("env", {})))

    def test_vendor_actions_and_runtime_versions_are_pinned(self):
        for job in self.workflow["jobs"].values():
            for step in job.get("steps", []):
                if "uses" in step:
                    self.assertRegex(step["uses"], r"@[0-9a-f]{40}$")
        # Both jobs that run Codex (review and inline reply) carry their own setup copy.
        setups = [(name, step) for name, job in self.workflow["jobs"].items() for step in job.get("steps", [])
                  if "openai/codex-action@" in step.get("uses", "")]
        self.assertEqual(["reply", "review"], sorted(name for name, _ in setups))
        for name, action in setups:
            with self.subTest(job=name):
                self.assertEqual(":read-only", action["with"]["permission-profile"])
                # Setup keeps sudo so the engine launch can drop it; the launch itself uses drop-sudo.
                self.assertEqual("read-only", action["with"]["safety-strategy"])
                self.assertEqual("0.165.0", action["with"]["codex-version"])
                self.assertIn("output-schema-file", action["with"])
        for each_job in self.workflow["jobs"].values():
            for other in each_job.get("steps", []):
                self.assertNotIn("drop-sudo", json.dumps(other.get("with", {})))
        self.assertIn("2.1.312", json.dumps(self.workflow["jobs"]["review"]))

    def test_root_base_head_subdirectory_and_trusted_policy_binding(self):
        for name in ("review", "reply", "publish", "publish-reply"):
            steps = self.workflow["jobs"][name]["steps"]
            base = next(s for s in steps if s.get("name") == "Check out trusted base at workspace root")
            head = next(s for s in steps if s.get("name") == "Check out immutable head as read-only data")
            self.assertNotIn("path", base["with"])
            self.assertEqual("pr-head", head["with"]["path"])
            self.assertFalse(head["with"]["persist-credentials"])
            self.assertLess(steps.index(base), steps.index(head))
            engine = next(s for s in steps if s.get("name") == "Check out immutable agent-review source")
            self.assertEqual("pipelab-org/agent-review", engine["with"]["repository"])
            self.assertNotIn("pull_request.head", engine["with"]["ref"])

    def test_fork_admission_is_an_explicit_failure_and_no_provider_is_admitted(self):
        script = self.workflow["jobs"]["admit"]["steps"][0]["run"]
        self.assertIn("no secrets or provider work admitted", script)
        self.assertIn("exit 1", script)
        self.assertIn("'/review codex'", script)


if __name__ == "__main__":
    unittest.main()
