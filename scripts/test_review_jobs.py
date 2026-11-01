# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the read/infer/write boundary with real validation and a fake GitHub API."""
import copy
import io
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
        self.raw_data = {"id": id, "body": body, "user": vars(user), **(raw or {})}
        self._rawData = self.raw_data
        self.__dict__.update(extra)
        self.html_url = "https://github.com/example/project/pull/1#comment"

    def edit(self, body):
        self.body = body
        self.raw_data["body"] = body


class Pull:
    def __init__(self, repo, head, base):
        self.number, self.state = 1, "open"
        self.user = NS(login="pr-author")
        self.head = NS(sha=head, repo=NS(full_name=repo, fork=False))
        self.base = NS(sha=base, repo=NS(full_name=repo))
        self.comments, self.roots, self.reviews, self.writes = {}, [], {}, []
        self.node_id, self.threads = "PR_fixture", {}
        self.requester = NS(graphql_query=self.graphql_query, graphql_named_mutation=self.graphql_named_mutation)

    def graphql_query(self, query, variables):
        if "... on PullRequestReviewThread" in query:
            root, thread = next((root, t) for root, t in self.threads.items() if t["id"] == variables["id"])
            ids = [c.id for c in self.roots if c.id == root or c.raw_data.get("in_reply_to_id") == root]
            return {}, {"data": {"node": dict(thread, comments={"nodes": [{"databaseId": i} for i in ids],
                "pageInfo": {"hasNextPage": False, "endCursor": None}})}}
        nodes = [dict(thread, comments={"nodes": [{"databaseId": root}]}) for root, thread in self.threads.items()]
        return {}, {"data": {"node": {"reviewThreads": {"nodes": nodes,
            "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}

    def graphql_named_mutation(self, name, data, schema):
        if name == "resolveReviewThread":
            self.writes.append(("resolve", data["threadId"]))
            thread = next(t for t in self.threads.values() if t["id"] == data["threadId"])
            thread["isResolved"] = True
            return {}, {"thread": {"id": thread["id"], "isResolved": True}}
        return {}, {}

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
            root = Comment(len(self.roots) + 300, request["body"], NS(**f.BOT),
                {"pull_request_review_id": id, "original_commit_id": kwargs["commit"].sha,
                 "commit_id": kwargs["commit"].sha, "line": request["line"]},
                path=request["path"], commit_id=kwargs["commit"].sha)
            self.roots.append(root)
            self.threads[root.id] = {"id": "THREAD_" + str(root.id), "isResolved": False, "isOutdated": False}
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
        # Repository permission per login, as GitHub's legacy permission field reports it.
        self.permissions = {f.JOSH["login"]: "write"}
        self.repo.get_collaborator_permission = lambda login: self.permissions.get(login, "none")
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
        def run(policy, bundle, strategy, head, **kwargs):
            r = receipt(strategy); r["files_opened"] = ["app.py"]
            if strategy == "verify-threads":
                plan = json.loads(self.plan.read_text())
                return {"output": {"threads": [{"thread": t["thread"], "status": "present", "evidence": []}
                    for t in plan["threads"]]}, "receipt": r}
            return {"findings": findings or [], "receipt": r}
        with patch.object(e, "run_claude", side_effect=run) as provider:
            j.infer(self.plan, self.artifact, self.head_dir)
        self.assertEqual(["bundle", "callers"], [c.args[2] for c in provider.call_args_list if c.args[2] != "verify-threads"])
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
                            {"output": {"answer": "The resolved local value is lost during the copy.",
                                "resolve": False, "reason": "none", "evidence": []}, "receipt": r})
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

    def test_table_and_full_text_show_each_findings_line_range(self):
        self.build([finding(path="helper.py", line=1, end_line=3, title="Spanning helper")])
        j.publish_artifact(self.artifact, self.head_dir)
        summary = self.summary().split("<details>")[0]
        self.assertIn("| `helper.py:1-3` | below |", summary)
        self.assertIn("**Spanning helper** (`helper.py:1-3`)", summary)

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

    def advance_head(self, path):
        (self.head_dir / path).write_text("next\n")
        e.git(self.head_dir, "add", "--", path)
        e.git(self.head_dir, "commit", "-qm", "next")
        self.pr.head.sha = e.git(self.head_dir, "rev-parse", "HEAD").strip()
        # GitHub moves commit_id for live comments; rewriting their line instead
        # freezes that commit and sets line=null. original_commit_id never moves.
        for comment in self.pr.roots:
            thread = self.pr.threads.get(comment.raw_data.get("in_reply_to_id", comment.id))
            if thread and not thread["isOutdated"]:
                if comment.path == path:
                    thread["isOutdated"] = True
                    comment.raw_data["line"] = None
                else:
                    comment.commit_id = self.pr.head.sha
                    comment.raw_data["commit_id"] = self.pr.head.sha
        os.environ.update(REVIEW_HEAD=self.pr.head.sha, GITHUB_RUN_ID="124")
        self.pr.writes.clear()

    def test_prior_summary_finding_keeps_full_text_when_it_becomes_anchorable(self):
        item = finding(path="helper.py", mechanism="Only copy of this mechanism")
        self.build([item])
        j.publish_artifact(self.artifact, self.head_dir)
        old = json.loads((self.root / "completion.json").read_text())
        self.assertEqual([], old["roots"])
        self.advance_head("helper.py")
        with patch.object(j, "records", return_value={123: old}):
            self.build([item])
            j.publish_artifact(self.artifact, self.head_dir)
        self.assertEqual([], self.pr.roots, "an identical prior finding opened another thread")
        newest = self.summary_after(old["summary"])
        self.assertIn("| below |", newest)
        self.assertIn("Only copy of this mechanism", newest.split("<details>")[0])

    def test_prior_thread_findings_repeat_full_text_when_github_moves_or_outdates_the_root(self):
        # GitHub moves a live root's commit_id to the new head and outdates a root whose
        # line is rewritten; neither state may leave a finding with no visible text.
        items = [finding(path="app.py", title="First", mechanism="One mechanism"),
                 finding(path="app.py", title="Second", mechanism="Another mechanism")]
        self.build(items)
        j.publish_artifact(self.artifact, self.head_dir)
        record = json.loads((self.root / "completion.json").read_text())
        root = self.pr.roots[0]
        self.assertIs(root, f.owned_root(self.pr, NS(raw_data={"in_reply_to_id": root.id}, path=root.path), {123: record}))
        self.advance_head("app.py")
        root.commit_id = self.pr.head.sha
        root.raw_data["line"] = None
        with patch.object(j, "records", return_value={123: record}):
            self.build(items)
            j.publish_artifact(self.artifact, self.head_dir)
        self.assertEqual(1, len(self.pr.roots), "grouped findings were posted twice")
        newest = self.summary_after(record["summary"]).split("<details>")[0]
        self.assertEqual(2, newest.count("| below |"))
        self.assertIn("One mechanism", newest)
        self.assertIn("Another mechanism", newest)

    def summary_after(self, old_summary):
        return max((c for c in self.pr.comments.values() if c.id != old_summary and f.RESULT.search(c.body)
                    and not c.body.startswith("<!-- agent-review-status:v1")), key=lambda c: c.id).body

    def test_success_collapses_status_and_earlier_reviewer_comments_only(self):
        calls = []
        self.pr.requester = NS(graphql_named_mutation=lambda name, data, schema: calls.append((name, data)))
        self.build([finding(path="app.py")])
        j.publish_artifact(self.artifact, self.head_dir)
        old = json.loads((self.root / "completion.json").read_text())
        status = next(c for c in self.pr.comments.values() if c.body.startswith("<!-- agent-review-status:v1"))
        summary = next(c for c in self.pr.comments.values() if f.RESULT.search(c.body) and c is not status)
        status.raw_data["node_id"], summary.raw_data["node_id"] = "STATUS", "SUMMARY"
        other = Comment(50, "Other workflow quoted agent-review-result:v1", NS(**f.BOT), {"node_id": "OTHER"})
        human = Comment(51, summary.body, NS(**f.JOSH), {"node_id": "HUMAN"})
        pending = Comment(52, "<!-- agent-review-notice:v1 run=9 -->\nPublication pending", NS(**f.BOT), {"node_id": "PENDING"})
        later = Comment(500, status.body, NS(**f.BOT), {"node_id": "LATER"})
        self.pr.comments.update({50: other, 51: human, 52: pending, 500: later})
        j.collapse_earlier(self.pr, 400, 401, {})
        self.assertEqual([], calls, "a parseable record is not immutable workflow proof")
        j.collapse_earlier(self.pr, 400, 401, {123: old})
        self.assertEqual({"STATUS"}, {data["subjectId"] for _, data in calls})
        self.assertEqual({"minimizeComment"}, {name for name, _ in calls})
        self.assertTrue(all(data["classifier"] == "OUTDATED" for _, data in calls))
        self.assertEqual(old, f.parse_record(status, self.pr.comments))
        # The run's own status, at exactly status_id, must stay visible for the failure notifier.
        calls.clear()
        j.collapse_earlier(self.pr, status.id, summary.id, {123: old})
        self.assertEqual(set(), {data["subjectId"] for _, data in calls})

    def test_full_review_omitting_a_finding_keeps_its_earlier_summary_visible(self):
        hidden = set()
        self.pr.requester = NS(graphql_named_mutation=lambda name, data, schema: hidden.add(data["subjectId"]))
        self.build([finding(path="helper.py", mechanism="Only copy of this advice")])
        j.publish_artifact(self.artifact, self.head_dir)
        old = json.loads((self.root / "completion.json").read_text())
        status = next(c for c in self.pr.comments.values() if c.body.startswith("<!-- agent-review-status:v1"))
        status.raw_data["node_id"] = "OLD-STATUS"
        self.pr.comments[old["summary"]].raw_data["node_id"] = "OLD-SUMMARY"
        self.advance_head("app.py")
        full = {"kind": "full", "from": self.pr.base.sha, "to": self.pr.head.sha, "full_head": True, "parent": None}
        with patch.object(j, "records", return_value={123: old}), \
                patch.object(j.followups, "scope_for", return_value=full):
            self.build()
            j.publish_artifact(self.artifact, self.head_dir)
        self.assertEqual({"OLD-STATUS"}, hidden)

    def test_incremental_review_keeps_every_earlier_summary_visible(self):
        hidden = set()
        self.pr.requester = NS(graphql_named_mutation=lambda name, data, schema: hidden.add(data["subjectId"]))
        for findings in ([finding(path="helper.py")], [finding(path="app.py")], []):
            with self.subTest(findings=len(findings)):
                hidden.clear()
                self.pr.comments = {k: v for k, v in self.pr.comments.items() if k == 9}
                self.pr.roots.clear()
                self.pr.writes.clear()
                os.environ.update(REVIEW_HEAD=self.env["REVIEW_HEAD"], GITHUB_RUN_ID="123")
                self.pr.head.sha = self.env["REVIEW_HEAD"]
                e.git(self.head_dir, "checkout", "-q", self.env["REVIEW_HEAD"])
                # A zero-finding summary can hold the only disclosure of a coverage gap.
                with patch.object(e, "unopened_regions", return_value=["docs/big.md"]):
                    self.build(findings)
                    j.publish_artifact(self.artifact, self.head_dir)
                old = json.loads((self.root / "completion.json").read_text())
                status = next(c for c in self.pr.comments.values() if c.body.startswith("<!-- agent-review-status:v1"))
                status.raw_data["node_id"] = "OLD-STATUS"
                self.pr.comments[old["summary"]].raw_data["node_id"] = "OLD-SUMMARY"
                self.advance_head("app.py")
                with patch.object(j, "records", return_value={123: old}):
                    self.build()
                    j.publish_artifact(self.artifact, self.head_dir)
                self.assertEqual({"OLD-STATUS"}, hidden)

    def test_collapse_refusal_never_fails_a_published_review(self):
        import io
        from github import GithubException
        calls = []
        def refuse(*_):
            calls.append(True)
            raise GithubException(403, {"message": "private response detail"})
        self.pr.requester = NS(graphql_named_mutation=refuse, graphql_query=self.pr.graphql_query)
        self.build([finding(path="app.py")])
        j.publish_artifact(self.artifact, self.head_dir)
        old = json.loads((self.root / "completion.json").read_text())
        status = next(c for c in self.pr.comments.values() if c.body.startswith("<!-- agent-review-status:v1"))
        status.raw_data["node_id"] = "OLD-STATUS"
        self.advance_head("app.py")
        output = io.StringIO()
        with patch.object(j, "records", return_value={123: old}), patch("sys.stdout", output):
            self.build([finding(path="app.py")])
            j.publish_artifact(self.artifact, self.head_dir)
        self.assertTrue((self.root / "completion.json").is_file())
        self.assertEqual([True], calls)
        self.assertEqual("Earlier review comments were left visible (GithubException)\n", output.getvalue())

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


    def prior_threads(self, grouped=False, outdated=True):
        items = [finding(path="app.py", title="First")]
        if grouped:
            items.append(finding(path="app.py", title="Second", mechanism="Another cause"))
        self.build(items)
        j.publish_artifact(self.artifact, self.head_dir)
        proof = json.loads((self.root / "completion.json").read_text())
        root = self.pr.roots[0]
        original = root.commit_id
        if outdated:
            # Rewriting the finding's file outdates the thread and freezes its commit.
            self.advance_head("app.py")
            self.assertTrue(self.pr.threads[root.id]["isOutdated"])
            self.assertIsNone(root.raw_data["line"])
        else:
            # A change elsewhere keeps the thread live and moves its commit_id.
            self.advance_head("helper.py")
            self.assertNotEqual(original, root.commit_id)
            self.assertFalse(self.pr.threads[root.id]["isOutdated"])
        self.assertEqual(original, root.raw_data["original_commit_id"])
        return proof, root

    def verified(self, plan, status="fixed"):
        r = receipt("verify-threads"); r["files_opened"] = ["app.py"]
        return {"output": {"threads": [{"thread": t["thread"], "status": status,
            "evidence": [{"finding": i, "path": "app.py", "line": 1, "change": "The value is preserved."}
                         for i in range(len(t["findings"]))] if status == "fixed" else []}
            for t in plan["threads"]]}, "receipt": r}

    def prepare_verification(self, proof):
        with patch.object(j, "records", return_value={123: proof}):
            self.build()
        return json.loads(self.plan.read_text())

    def publish_verified(self, proof, plan, verification):
        artifact = json.loads(self.artifact.read_text())
        artifact.update(plan=plan, verification=verification)
        self.artifact.write_text(json.dumps(artifact))
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_artifact(self.artifact, self.head_dir)

    def test_fixed_grouped_root_gets_one_verified_reply_then_resolution(self):
        proof, root = self.prior_threads(grouped=True)
        plan = self.prepare_verification(proof)
        self.assertEqual(2, len(plan["threads"][0]["findings"]))
        verification = self.verified(plan)
        self.publish_verified(proof, plan, verification)
        self.assertEqual(["reply", "resolve"], [kind for kind, _ in self.pr.writes if kind in ("reply", "resolve")])
        self.assertTrue(self.pr.threads[root.id]["isResolved"])
        answer = next(c for c in self.pr.roots if c.raw_data.get("in_reply_to_id") == root.id)
        self.assertIn("app.py:1", answer.body)
        self.assertIn("The value is preserved", answer.body)

    def test_human_reopened_verified_thread_is_not_resolved_again_on_a_later_head(self):
        proof, root = self.prior_threads()
        plan = self.prepare_verification(proof)
        self.publish_verified(proof, plan, self.verified(plan))
        self.assertTrue(self.pr.threads[root.id]["isResolved"])
        # A maintainer disagrees and reopens the thread; the bot must not fight that.
        self.pr.threads[root.id]["isResolved"] = False
        self.advance_head("later.py")
        later = self.prepare_verification(proof)
        self.assertEqual([], later["threads"])
        self.publish_verified(proof, later, self.verified(plan))
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        self.assertEqual(0, sum(k == "resolve" for k, _ in self.pr.writes))

    def test_publication_rechecks_only_the_target_thread_and_loads_proof_once(self):
        # Each fixed thread is rechecked several times before it resolves. Reloading
        # every completion record and every open thread on each recheck multiplies
        # GitHub requests by open threads times fixed threads.
        proof, root = self.prior_threads()
        plan = self.prepare_verification(proof)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        with patch.object(j, "records", return_value={123: proof}) as loaded, \
                patch.object(j, "collect_threads", side_effect=AssertionError("full scan")):
            j.publish_verification(self.repo, self.pr, ctx, policy, plan, self.verified(plan), self.head_dir)
        self.assertEqual(1, loaded.call_count)
        self.assertTrue(self.pr.threads[root.id]["isResolved"])

    def test_root_original_commit_must_match_the_recorded_commit(self):
        # Isolates the immutable original_commit_id binding: the record, review,
        # path and body still match, so only that check can refuse the root.
        proof, root = self.prior_threads()
        self.assertEqual(1, len(j.collect_threads(self.pr, {123: proof})))
        root.raw_data["original_commit_id"] = self.pr.head.sha
        self.assertEqual([], j.collect_threads(self.pr, {123: proof}))
        with self.assertRaises(f.FollowupAbort):
            f.verify_root(self.pr, root, {123: proof})

    def test_reply_resolved_thread_reopened_by_a_person_is_never_verified_again(self):
        proof, root = self.prior_threads()
        plan = self.reply_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, self.answer("fixed"), self.head_dir)
        self.assertTrue(self.pr.threads[root.id]["isResolved"])
        self.pr.threads[root.id]["isResolved"] = False
        self.advance_head("later.py")
        self.assertEqual([], j.collect_threads(self.pr, {123: proof}))
        # An answer that did not resolve leaves the thread eligible for verification.
        self.pr.roots = [root]
        self.pr.threads[root.id]["isResolved"] = False
        self.advance_head("again.py")
        plan = self.reply_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, self.answer(), self.head_dir)
        self.assertEqual(1, len(j.collect_threads(self.pr, {123: proof})))

    def test_present_unclear_and_model_silence_never_resolve(self):
        proof, root = self.prior_threads()
        plan = self.prepare_verification(proof)
        for status in ("present", "unclear", None):
            self.pr.writes.clear()
            self.publish_verified(proof, plan, self.verified(plan, status) if status else None)
            self.assertFalse(self.pr.threads[root.id]["isResolved"])
            self.assertFalse(any(k in ("reply", "resolve") for k, _ in self.pr.writes))

    def test_failed_resolution_is_never_automatically_retried(self):
        proof, root = self.prior_threads()
        plan = self.prepare_verification(proof)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "resolve_thread", side_effect=RuntimeError("private")):
            j.publish_verification(self.repo, self.pr, ctx, policy, plan, self.verified(plan), self.head_dir)
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_verification(self.repo, self.pr, ctx, policy, plan, self.verified(plan), self.head_dir)
            j.publish_verification(self.repo, self.pr, ctx, policy, plan, self.verified(plan), self.head_dir)
        self.assertEqual(1, sum(k == "reply" for k, _ in self.pr.writes))
        self.assertEqual(0, sum(k == "resolve" for k, _ in self.pr.writes))
        self.assertFalse(self.pr.threads[root.id]["isResolved"])

    def test_unowned_or_changed_roots_are_never_collected_or_resolved(self):
        proof, root = self.prior_threads()
        plan = self.prepare_verification(proof)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        for mutate in (lambda p: p["roots"].clear(),
                       lambda p: p["roots"][0].update(digest="0" * 64),
                       lambda p: p["roots"][0].update(commit=self.pr.head.sha),
                       lambda p: p["roots"][0].update(review_id=999),
                       lambda p: p["roots"][0].update(path="helper.py")):
            bad = copy.deepcopy(proof); mutate(bad)
            self.pr.writes.clear()
            with patch.object(j, "records", return_value={123: bad}):
                j.publish_verification(self.repo, self.pr, ctx, policy, plan, self.verified(plan), self.head_dir)
            self.assertFalse(any(k in ("reply", "resolve") for k, _ in self.pr.writes))
        root.body += "edited"
        self.assertEqual([], j.collect_threads(self.pr, {123: proof}))
        self.assertFalse(self.pr.threads[root.id]["isResolved"])

    def test_reconstruction_requires_exact_unambiguous_group_members(self):
        proof, root = self.prior_threads(grouped=True)
        self.assertEqual(proof["findings"], j.reconstruct_root(root, proof))
        for change in (lambda p: p["findings"].append(copy.deepcopy(p["findings"][0])),
                       lambda p: p["findings"].pop(),
                       lambda p: p["findings"][0].update(body="edited"),
                       lambda p: p["findings"][1].update(line=9, end_line=9)):
            bad = copy.deepcopy(proof); change(bad)
            with self.assertRaises(e.ReviewAbort): j.reconstruct_root(root, bad)
        bad = copy.deepcopy(proof)
        bad["findings"].append(dict(bad["findings"][0], title="Summary only"))
        self.assertEqual(proof["findings"], j.reconstruct_root(root, bad))
        bad["findings"].append(dict(bad["findings"][0], title="2 findings on these lines",
                                   body=root.body.split("\n\n", 1)[1]))
        with self.assertRaises(e.ReviewAbort): j.reconstruct_root(root, bad)
        root.body += "extra"
        with self.assertRaises(e.ReviewAbort): j.reconstruct_root(root, proof)

    def test_resolved_roots_are_skipped_and_outdated_roots_can_be_verified(self):
        proof, root = self.prior_threads()
        self.pr.threads[root.id]["isResolved"] = True
        self.assertEqual([], j.collect_threads(self.pr, {123: proof}))
        self.pr.threads[root.id].update(isResolved=False, isOutdated=True)
        root.raw_data["line"] = None
        self.assertEqual(1, len(j.collect_threads(self.pr, {123: proof})))

    def test_verification_schema_receipt_completeness_and_head_evidence_guards(self):
        with self.assertRaises(e.ReviewAbort): j.validate_evidence_schema({})
        proof, _ = self.prior_threads(grouped=True)
        plan = self.prepare_verification(proof)
        policy, _ = j.settings()
        good = self.verified(plan)
        mutations = [
            lambda v: v.update(extra=True),
            lambda v: v["receipt"].update(strategy="bundle"),
            lambda v: v["output"].update(extra=True),
            lambda v: v["output"].update(threads={}),
            lambda v: v["output"]["threads"].clear(),
            lambda v: v["output"]["threads"].append(copy.deepcopy(v["output"]["threads"][0])),
            lambda v: v["output"]["threads"][0].update(thread="other"),
            lambda v: v["output"]["threads"][0].update(status="silent"),
            lambda v: v["output"]["threads"][0].update(evidence={}),
            lambda v: v["output"]["threads"][0]["evidence"].pop(),
            lambda v: v["receipt"].update(files_opened=[]),
        ]
        for key, value in (("finding", True), ("finding", -1), ("path", "../escape"),
                           ("line", True), ("line", 0), ("line", 999), ("change", ""),
                           ("change", None), ("extra", "value")):
            mutations.append(lambda v, k=key, val=value: v["output"]["threads"][0]["evidence"][0].update({k: val}))
        for mutate in mutations:
            bad = copy.deepcopy(good); mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(e.ReviewAbort):
                j.validate_verification_result(bad, plan["threads"], policy, self.head_dir, self.pr.head.sha)
        bad = copy.deepcopy(good)
        for evidence in bad["output"]["threads"][0]["evidence"]: evidence["path"] = "gone.py"
        bad["receipt"]["files_opened"] = ["gone.py"]
        with self.assertRaises(e.ReviewAbort):
            j.validate_verification_result(bad, plan["threads"], policy, self.head_dir, self.pr.head.sha)
        folder = self.head_dir / "nested"; folder.mkdir()
        (folder / "source.py").write_text("source\n")
        e.git(self.head_dir, "add", "--", "nested")
        e.git(self.head_dir, "commit", "-qm", "nested fixture")
        tree_head = e.git(self.head_dir, "rev-parse", "HEAD").strip()
        bad = copy.deepcopy(good)
        for item in bad["output"]["threads"][0]["evidence"]: item["path"] = "nested"
        bad["receipt"]["files_opened"] = ["nested"]
        with self.assertRaises(e.ReviewAbort):
            j.validate_verification_result(bad, plan["threads"], policy, self.head_dir, tree_head)
        for validator, value in ((j.validate_verification, {"threads": []}),
                                 (j.validate_reply, self.answer()["output"])):
            with patch.object(f, "MAX_DATA_BYTES", len(f.canonical(value).encode()) - 1), self.assertRaises(e.ReviewAbort):
                validator(value)

    def test_oversized_verification_prompt_skips_without_a_provider_call(self):
        import io
        proof, _ = self.prior_threads()
        plan = self.prepare_verification(proof)
        actual_run = e.run_claude
        def run(policy, bundle, strategy, head, **kwargs):
            if strategy == "verify-threads":
                return actual_run(dict(policy, max_model_tokens=policy["max_output_tokens"]), bundle, strategy, head, **kwargs)
            return {"findings": [], "receipt": receipt(strategy)}
        with patch.object(e, "run_claude", side_effect=run), patch.object(e.subprocess, "run") as provider, \
                patch("sys.stdout", new_callable=io.StringIO) as output, \
                patch("sys.stderr", new_callable=io.StringIO) as errors:
            j.infer(self.plan, self.artifact, self.head_dir)
        provider.assert_not_called()
        self.assertEqual(j.FOLLOWUP_NOTE + "\n", output.getvalue())
        # The fixed engine cause reaches the Actions log; stdout keeps the note alone.
        self.assertEqual("Thread verification stopped: Complete prompt exceeds the review depth's max_model_tokens\n",
                         errors.getvalue())
        self.assertIsNone(json.loads(self.artifact.read_text())["verification"])

    def test_graphql_prepare_inference_and_publish_failures_leave_review_successful(self):
        import io
        proof, root = self.prior_threads()
        with patch.object(j, "records", return_value={123: proof}), \
                patch.object(self.pr.requester, "graphql_query", side_effect=RuntimeError("private")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            self.build()
        self.assertEqual(j.FOLLOWUP_NOTE + "\n", output.getvalue())
        self.assertEqual([], json.loads(self.plan.read_text())["threads"])
        plan = self.prepare_verification(proof)
        original = self.pr.requester.graphql_named_mutation
        for failure in ("graphql", "resolve", "reply", "reply_readback", "root_changed", "head_moved"):
            self.pr.roots = [c for c in self.pr.roots if c.raw_data.get("in_reply_to_id") is None]
            self.pr.writes.clear()
            artifact = json.loads(self.artifact.read_text()); artifact["verification"] = self.verified(plan)
            self.artifact.write_text(json.dumps(artifact))
            def refuse(name, data, schema):
                if name == "resolveReviewThread": raise RuntimeError("private provider body")
                return original(name, data, schema)
            target, attr, effect = {
                "graphql": (self.pr.requester, "graphql_query", RuntimeError("private")),
                "resolve": (self.pr.requester, "graphql_named_mutation", refuse),
                "reply": (self.pr, "create_review_comment_reply", RuntimeError("private")),
                "reply_readback": (j, "publish_answer", RuntimeError("private")),
                "root_changed": (j, "thread_item", lambda *a: None),
                "head_moved": (j, "check_pr", RuntimeError("moved")),
            }[failure]
            policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
            # Check publication as an optional post-success action, never un-green it.
            with patch.object(j, "records", return_value={123: proof}), \
                    patch.object(target, attr, side_effect=effect), patch("sys.stdout", new_callable=io.StringIO) as output:
                j.publish_verification(self.repo, self.pr, ctx, policy, plan, self.verified(plan), self.head_dir)
            self.assertEqual(j.FOLLOWUP_NOTE + "\n", output.getvalue())
            self.assertFalse(self.pr.threads[root.id]["isResolved"])
        # Full successful publication still writes completion when optional resolution fails.
        with patch.object(f, "resolve_thread", side_effect=RuntimeError("private")):
            self.publish_verified(proof, plan, self.verified(plan))
        self.assertTrue((self.root / "completion.json").exists())
        self.assertTrue(any("state=success" in c.body for c in self.pr.comments.values()))

    def test_batched_verification_uses_host_provider_at_every_depth_and_skips_failures(self):
        import io
        proof, _ = self.prior_threads()
        plan = self.prepare_verification(proof)
        for depth in e.DEPTHS:
            os.environ["REVIEW_DEPTH"] = depth
            self.pr.comments[9].body = {"default": "/review", "deep": "/review deep",
                "codex": "/review codex", "codex-deep": "/review codex deep"}[depth]
            policy, _ = j.settings(); ctx, _ = j.context(policy, depth)
            plan["context"] = ctx
            self.plan.write_text(json.dumps(plan))
            def run(policy, bundle, strategy, head, *args, **kwargs):
                r = receipt(strategy); r.update(provider=policy["provider"], model=policy["model"], files_opened=["app.py"])
                return {"output": self.verified(plan)["output"], "receipt": r} if strategy == "verify-threads" else {"findings": [], "receipt": r}
            active = "run_claude" if policy["provider"] == "claude" else "run_codex"
            inactive = "run_codex" if active == "run_claude" else "run_claude"
            with patch.object(e, active, side_effect=run) as provider, patch.object(e, inactive) as unused:
                j.infer(self.plan, self.artifact, self.head_dir, "/action", "/home")
            unused.assert_not_called()
            self.assertEqual(1, sum(c.args[2] == "verify-threads" for c in provider.call_args_list))
            self.assertEqual(policy, provider.call_args_list[-1].args[0])
            self.assertIs(j.VERIFICATION_SCHEMA, provider.call_args_list[-1].kwargs["schema"])
            self.assertIsNotNone(json.loads(self.artifact.read_text())["verification"])
            def fail(policy, bundle, strategy, head, *args, **kwargs):
                if strategy == "verify-threads": raise e.ReviewAbort("private model text")
                return run(policy, bundle, strategy, head, *args, **kwargs)
            with patch.object(e, active, side_effect=fail), patch("sys.stdout", new_callable=io.StringIO) as output:
                j.infer(self.plan, self.artifact, self.head_dir, "/action", "/home")
            self.assertEqual(j.FOLLOWUP_NOTE + "\n", output.getvalue())
            self.assertIsNone(json.loads(self.artifact.read_text())["verification"])

    def test_unchanged_scope_never_triggers_verification(self):
        proof, _ = self.prior_threads()
        plan = self.prepare_verification(proof)
        plan["scope"]["kind"] = "unchanged"
        self.plan.write_text(json.dumps(plan))
        with patch.object(e, "run_claude") as claude, patch.object(e, "run_codex") as codex:
            j.infer(self.plan, self.artifact, self.head_dir)
        claude.assert_not_called(); codex.assert_not_called()
        self.assertIsNone(json.loads(self.artifact.read_text())["verification"])

    def test_no_proof_and_unchanged_prepare_do_not_collect_threads(self):
        with patch.object(f, "review_threads") as query:
            self.assertEqual([], j.collect_threads(self.pr, {}))
        query.assert_not_called()
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        scope = {"kind": "unchanged", "from": ctx["head"], "to": ctx["head"], "parent": 123, "full_head": False}
        with patch.object(f, "scope_for", return_value=scope), patch.object(j, "collect_threads") as collect:
            j.prepare(self.head_dir, self.plan)
        collect.assert_not_called()

    def reply_plan(self, proof, root):
        raw = {"id": 500, "body": "Fixed now, please check the code.", "in_reply_to_id": root.id,
            "pull_request_review_id": root.raw_data["pull_request_review_id"],
            "commit_id": root.commit_id, "original_commit_id": root.raw_data["original_commit_id"],
            "path": root.path, "created_at": "2026-10-28T01:00:00Z", "updated_at": "2026-10-28T01:00:00Z",
            "pull_request_url": "https://api.github.com/repos/example/project/pulls/1"}
        self.pr.roots.append(Comment(500, raw["body"], NS(**f.JOSH), raw, path=root.path))
        self.event_path.write_text(json.dumps({"action": "created", "repository": {"full_name": self.repo.full_name},
            "pull_request": {"number": 1}, "comment": dict(raw, user=f.JOSH)}))
        os.environ["GITHUB_EVENT_NAME"] = "pull_request_review_comment"
        with patch.object(j, "records", return_value={123: proof}):
            j.prepare(self.head_dir, self.plan, reply=True)
        return json.loads(self.plan.read_text())

    def answer(self, reason="none"):
        r = receipt("reply"); r["files_opened"] = ["app.py"]
        return {"receipt": r, "output": {"answer": "app.py:1 preserves the local value.",
            "resolve": reason != "none", "reason": reason,
            "evidence": [{"finding": 0, "path": "app.py", "line": 1, "change": "Preserves the value."}] if reason != "none" else []}}

    def test_earlier_head_code_reading_reply_resolves_only_confirmed_fixes(self):
        proof, root = self.prior_threads()
        for reason in ("none", "fixed", "finding_wrong"):
            self.pr.threads[root.id]["isResolved"] = False
            self.pr.roots = [root]
            plan = self.reply_plan(proof, root)
            self.assertIn("read-only file tools", plan["prompt"])
            policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
            with patch.object(j, "records", return_value={123: proof}):
                j.publish_reply(self.repo, self.pr, ctx, policy, plan, self.answer(reason), self.head_dir)
            self.assertEqual(reason == "fixed", self.pr.threads[root.id]["isResolved"])
            posted = [c.body for c in self.pr.roots if c.raw_data.get("in_reply_to_id") == root.id][-1]
            self.assertEqual(reason == "finding_wrong", j.KEPT_OPEN_NOTE in posted)
            self.assertEqual(reason == "fixed", "agent-review-resolved:v1" in posted)

    def test_live_threads_are_never_verified_or_resolved_by_a_reply(self):
        # The code at a live thread's location did not change, so neither the
        # verification stage nor a "fixed" reply may resolve it.
        proof, root = self.prior_threads(outdated=False)
        self.assertEqual([], j.collect_threads(self.pr, {123: proof}))
        plan = self.reply_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, self.answer("fixed"), self.head_dir)
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        posted = [c.body for c in self.pr.roots if c.raw_data.get("in_reply_to_id") == root.id][-1]
        self.assertIn(j.KEPT_OPEN_NOTE, posted)
        self.assertNotIn("agent-review-resolved:v1", posted)

    def test_reply_resolution_requires_disposition_receipt_all_findings_and_head_reads(self):
        for field, value in (("resolve", 1), ("reason", "other"), ("answer", ""), ("reason", "none")):
            bad = self.answer("fixed")["output"]; bad[field] = value
            with self.assertRaises(e.ReviewAbort): j.validate_reply(bad)
        proof, root = self.prior_threads(grouped=True)
        plan = self.reply_plan(proof, root)
        policy, _ = j.settings()
        for mutate in (lambda v: v["output"].update(resolve=1),
                       lambda v: v["output"].update(reason="other"),
                       lambda v: v["output"].update(answer=""),
                       lambda v: v["output"].update(reason="none"),
                       lambda v: v["receipt"].update(files_opened=[]),
                       lambda v: v["output"].update(answer="x" * (f.MAX_DATA_BYTES + 1))):
            bad = self.answer("fixed"); mutate(bad)
            with self.assertRaises(e.ReviewAbort): j.validate_reply_result(bad, plan, policy, self.head_dir)
        with self.assertRaises(e.ReviewAbort): j.validate_reply_result(self.answer("fixed"), plan, policy, self.head_dir)

    def test_reply_provider_reads_files_and_attests_model_without_execution(self):
        from scripts.test_review_engine import transcript
        proof, root = self.prior_threads()
        plan = self.reply_plan(proof, root)
        policy, _ = j.settings()
        events = transcript()
        events[0]["message"]["content"][0]["input"]["file_path"] = "app.py"
        events[-1]["structured_output"] = self.answer("fixed")["output"]
        proc = NS(returncode=0, stdout="\n".join(map(json.dumps, events)))
        actual_run = e.subprocess.run
        def provider(*args, **kwargs):
            return proc if args[0][0] == "claude" else actual_run(*args, **kwargs)
        with patch.object(e.subprocess, "run", side_effect=provider) as run:
            answer = j.infer_reply(policy, plan, self.head_dir)
        self.assertEqual(["app.py"], answer["receipt"]["files_opened"])
        self.assertIn("Read,Grep,Glob", run.call_args_list[0].args[0])
        self.assertIn("dontAsk", run.call_args_list[0].args[0])
        events[0]["message"]["content"][0]["name"] = "Bash"
        proc.stdout = "\n".join(map(json.dumps, events))
        with patch.object(e.subprocess, "run", return_value=proc), self.assertRaises(e.ReviewAbort):
            j.infer_reply(policy, plan, self.head_dir)

    def test_reply_resolution_failure_leaves_verified_answer_visible_and_thread_open(self):
        import io
        proof, root = self.prior_threads()
        plan = self.reply_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        with patch.object(j, "records", return_value={123: proof}), \
                patch.object(f, "resolve_thread", side_effect=RuntimeError("private model text")), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, self.answer("fixed"), self.head_dir)
        self.assertEqual(j.FOLLOWUP_NOTE + "\n", output.getvalue())
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        self.assertTrue(any(self.answer()["output"]["answer"] in c.body for c in self.pr.roots))

    def test_edited_root_reply_and_ambiguous_answer_never_resolve(self):
        proof, root = self.prior_threads()
        plan = self.reply_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        root.body += "edited"
        with patch.object(j, "records", return_value={123: proof}), self.assertRaises(f.FollowupAbort):
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, self.answer("fixed"), self.head_dir)
        root.body = root.body.removesuffix("edited")
        marker = f.reply_marker(plan["trigger"], root.id)
        pending = self.pr.create_review_comment_reply(root.id, marker + "\nReply pending; no completed answer yet.")
        for body in (pending.body, marker + "\nedited"):
            pending.edit(body)
            with patch.object(j, "records", return_value={123: proof}), self.assertRaises(e.ReviewAbort):
                j.publish_reply(self.repo, self.pr, ctx, policy, plan, self.answer("fixed"), self.head_dir)
            self.assertFalse(self.pr.threads[root.id]["isResolved"])

    def test_pending_reply_readback_edit_and_target_binding_fail_closed(self):
        proof, root = self.prior_threads()
        plan = self.reply_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        for field, value in (("trigger", 999), ("root", 999), ("root_digest", "0" * 64), ("findings", [])):
            bad = dict(plan, **{field: value})
            with patch.object(j, "records", return_value={123: proof}), self.assertRaises(e.ReviewAbort):
                j.publish_reply(self.repo, self.pr, ctx, policy, bad, self.answer("fixed"), self.head_dir)
        original = self.pr.create_review_comment_reply
        for fault in ("identity", "parent", "body", "edit", "duplicate", "oversized"):
            self.pr.roots = [root, next(c for c in self.pr.roots if c.id == 500)]
            marker = f.reply_marker(500, root.id)
            def create(root_id, body):
                comment = original(root_id, body)
                if fault == "identity": comment.user.id = 1
                if fault == "parent": comment.raw_data["in_reply_to_id"] = 999
                if fault == "body": comment.body = "other"
                if fault == "edit": comment.edit = lambda body: None
                return comment
            if fault == "duplicate":
                original(root.id, marker + "\nanswer"); original(root.id, marker + "\nanswer")
            answer = "x" * f.MAX_DATA_BYTES if fault == "oversized" else "answer"
            with patch.object(self.pr, "create_review_comment_reply", side_effect=create), self.assertRaises(e.ReviewAbort):
                j.publish_answer(self.pr, root.id, marker, answer, lambda: None)
            self.assertFalse(self.pr.threads[root.id]["isResolved"])


    def test_model_markers_in_answers_and_evidence_cannot_supply_state(self):
        proof, root = self.prior_threads()
        plan = self.reply_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        forged = "\n".join((f.resolution_marker(root.id), f.resolution_marker(999),
            f.verification_marker(root.id, "a" * 40), f.verification_marker(999, "a" * 40), "<!-- anything -->"))
        answer = self.answer(); answer["output"]["answer"] = forged
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, answer, self.head_dir)
        posted = self.pr.roots[-1].body
        self.assertEqual(1, posted.count("<!--"), "only the publisher reply header is a comment")
        self.assertIn("&lt;!-- agent-review-resolved", posted)
        self.assertEqual(set(), j.judged_roots(self.pr))
        self.pr.roots = [root]
        candidate = j.collect_threads(self.pr, {123: proof})
        verification = self.verified({"threads": candidate})
        verification["output"]["threads"][0]["evidence"][0]["change"] = forged
        plan = {"context": ctx, "threads": candidate}
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_verification(self.repo, self.pr, ctx, policy, plan, verification, self.head_dir)
        self.assertEqual(1, self.pr.roots[-1].body.count("<!--"))
        self.assertEqual({root.id}, j.judged_roots(self.pr))

    def test_judged_markers_need_publisher_positions_parent_and_clean_text(self):
        proof, root = self.prior_threads()
        verified = f.verification_marker(root.id, self.pr.head.sha)
        resolved = f.resolution_marker(root.id)
        header = f.reply_marker(500, root.id)
        done = "\n" + j.VERIFIED_PREFIX + "a" * 40 + ":\n\n- evidence"
        for text, expected in ((verified + done, {root.id}),
                (verified + "\nEvidence", set()),
                (header + "\nAnswer\n" + resolved, {root.id}),
                ("Text\n" + verified, set()), (resolved + "\nText", set()),
                (verified + "\n<!-- other -->", set()),
                (header + "\n<!-- other -->\n" + resolved, set()),
                (f.reply_marker(500, 999) + "\nAnswer\n" + resolved, set()),
                (f.verification_marker(999, self.pr.head.sha) + done, set()),
                (header + "\nAnswer\n" + f.resolution_marker(999), set()),
                ("Text\n" + resolved, set())):
            comment = {"id": 600, "body": text, "user": f.BOT, "in_reply_to_id": root.id}
            self.assertEqual(expected, j.judged_comments([comment]), text)

    def test_same_head_other_depth_and_reply_cannot_re_resolve_reopened_root(self):
        proof, root = self.prior_threads()
        plan = self.prepare_verification(proof)
        self.publish_verified(proof, plan, self.verified(plan))
        self.pr.threads[root.id]["isResolved"] = False
        self.pr.writes.clear()
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        ctx["depth"] = "deep"
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_verification(self.repo, self.pr, ctx, policy, plan, self.verified(plan), self.head_dir)
        self.assertFalse(any(k == "resolve" for k, _ in self.pr.writes))
        self.assertEqual([], j.collect_threads(self.pr, {123: proof}))
        plan = self.reply_plan(proof, root)
        ctx, _ = j.context(policy, depth)
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, self.answer("fixed"), self.head_dir)
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        self.assertIn(j.KEPT_OPEN_NOTE, self.pr.roots[-1].body)

    def test_fixed_evidence_requires_each_findings_recorded_path(self):
        proof, root = self.prior_threads(grouped=True)
        plan = self.prepare_verification(proof)
        policy, _ = j.settings()
        verification = self.verified(plan)
        verification["receipt"]["files_opened"].append("helper.py")
        verification["output"]["threads"][0]["evidence"][0]["path"] = "helper.py"
        normalized = j.validate_verification_result(verification, plan["threads"], policy, self.head_dir, self.pr.head.sha)
        self.assertEqual("unclear", normalized["threads"][0]["status"])
        self.publish_verified(proof, plan, verification)
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        self.assertFalse(any(k == "reply" for k, _ in self.pr.writes))
        reply_plan = self.reply_plan(proof, root)
        answer = self.answer("fixed")
        answer["output"]["evidence"] = copy.deepcopy(verification["output"]["threads"][0]["evidence"])
        answer["receipt"]["files_opened"].append("helper.py")
        self.assertFalse(j.validate_reply_result(answer, reply_plan, policy, self.head_dir)["resolve"])
        # Related-source evidence is allowed in addition to the original path.
        verification["output"]["threads"][0]["evidence"].append(
            {"finding": 0, "path": "app.py", "line": 1, "change": "Fix at the original path."})
        normalized = j.validate_verification_result(verification, plan["threads"], policy, self.head_dir, self.pr.head.sha)
        self.assertEqual("fixed", normalized["threads"][0]["status"])

    def resolve_plan(self, proof, root, body="/resolve"):
        self.reply_plan(proof, root)
        live = next(c for c in self.pr.roots if c.id == 500)
        live.edit(body)
        event = json.loads(self.event_path.read_text()); event["comment"]["body"] = body
        self.event_path.write_text(json.dumps(event))
        with patch.object(j, "records", return_value={123: proof}):
            j.prepare(self.head_dir, self.plan, reply=True)
        return json.loads(self.plan.read_text())

    def test_resolve_command_skips_provider_and_resolves_only_its_owned_thread(self):
        proof, root = self.prior_threads(outdated=False)
        # An explicit request may resolve a live or previously judged thread.
        self.pr.create_review_comment_reply(root.id, f.verification_marker(root.id, self.pr.head.sha) + "\nEvidence")
        plan = self.resolve_plan(proof, root, " \n/resolve\t ")
        with patch.object(j, "records", return_value={123: proof}), \
                patch.object(j, "reconstruct_root", side_effect=AssertionError("command does not judge findings")):
            j.prepare(self.head_dir, self.plan, reply=True)
        with patch.object(e, "run_claude") as claude, patch.object(e, "run_codex") as codex:
            j.infer(self.plan, self.artifact, self.head_dir)
        claude.assert_not_called(); codex.assert_not_called()
        self.assertIsNone(json.loads(self.artifact.read_text())["result"])
        resolve = f.resolve_thread
        def observed(pr, thread):
            self.assertFalse(any(f.resolution_marker(root.id) in c.body for c in self.pr.roots))
            return resolve(pr, thread)
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "resolve_thread", side_effect=observed) as call:
            j.publish_artifact(self.artifact, self.head_dir)
            j.publish_artifact(self.artifact, self.head_dir)
        self.assertEqual(1, call.call_count)
        self.assertTrue(self.pr.threads[root.id]["isResolved"])
        self.assertTrue(self.pr.roots[-1].body.endswith(f.resolution_marker(root.id)))
        self.assertIn("confirmed by GitHub", self.pr.roots[-1].body)

    def test_resolve_command_admission_and_exact_text_cannot_be_bypassed(self):
        proof, root = self.prior_threads()
        plan = self.resolve_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        live = next(c for c in self.pr.roots if c.id == 500)
        for failure in ("edited", "unowned", "member"):
            event = json.loads(self.event_path.read_text())
            if failure == "edited": live.edit("edited")
            if failure == "member":
                user = {"id": 123, "login": "member", "type": "User"}
                live.user = NS(**user); event["comment"]["user"] = user
                os.environ.update(GITHUB_ACTOR="member", GITHUB_TRIGGERING_ACTOR="member", GITHUB_ACTOR_ID="123")
                self.event_path.write_text(json.dumps(event))
            client = NS(requester=NS(requestJsonAndCheck=lambda *a: ({}, {"state": "inactive"})))
            with patch.object(j, "records", return_value={} if failure == "unowned" else {123: proof}), \
                    patch.object(j.reliability, "github_client", return_value=client), self.assertRaises(f.FollowupAbort):
                j.publish_reply(self.repo, self.pr, ctx, policy, plan, None, self.head_dir)
            live.edit("/resolve")
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        self.assertFalse(any(k == "resolve" for k, _ in self.pr.writes))

    def test_resolve_command_already_resolved_and_other_text_do_not_resolve(self):
        proof, root = self.prior_threads()
        plan = self.resolve_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        self.pr.threads[root.id]["isResolved"] = True
        self.pr.writes.clear()
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, None, self.head_dir)
        self.assertEqual([], self.pr.writes)
        self.pr.threads[root.id]["isResolved"] = False
        for body in ("/Resolve", "/resolve please", "please /resolve", "/resolve\nplease"):
            self.pr.roots = [root]
            plan = self.resolve_plan(proof, root, body)
            self.assertFalse(plan["resolve_command"])
            with patch.object(j, "infer_reply", return_value=self.answer()) as infer:
                j.infer(self.plan, self.artifact, self.head_dir)
            infer.assert_called_once()
            with patch.object(j, "records", return_value={123: proof}):
                j.publish_artifact(self.artifact, self.head_dir)
            self.assertFalse(self.pr.threads[root.id]["isResolved"])
            with patch.object(j, "records", return_value={123: proof}), self.assertRaises(e.ReviewAbort):
                j.publish_reply(self.repo, self.pr, ctx, policy, dict(plan, resolve_command=True), None, self.head_dir)

    def test_resolution_markers_describe_attempts_not_unobserved_success(self):
        import io
        proof, root = self.prior_threads()
        plan = self.resolve_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        def refuse(*a):
            self.assertNotIn(f.resolution_marker(root.id), self.pr.roots[-1].body)
            raise RuntimeError("private response")
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "resolve_thread", side_effect=refuse), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, None, self.head_dir)
        self.assertEqual(j.FOLLOWUP_NOTE + "\n", output.getvalue())
        self.assertNotIn("confirmed by GitHub", self.pr.roots[-1].body)
        self.assertIn("could not be confirmed", self.pr.roots[-1].body)
        self.assertTrue(self.pr.roots[-1].body.endswith(f.resolution_marker(root.id)))
        # A thread that becomes resolved before the mutation has no attempt marker.
        self.pr.roots = [root]
        plan = self.resolve_plan(proof, root)
        snapshot = f.thread_snapshot
        def resolved_before_attempt(*args):
            state, comments = snapshot(*args); state["isResolved"] = True
            return state, comments
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "thread_snapshot", side_effect=resolved_before_attempt), \
                patch.object(f, "resolve_thread") as resolve:
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, None, self.head_dir)
        resolve.assert_not_called()
        self.assertNotIn("agent-review-resolved", self.pr.roots[-1].body)

    def test_resolve_requires_pr_author_or_write_access_at_admission_and_mutation(self):
        # A current organization member with only read access who did not open the PR.
        proof, root = self.prior_threads()
        self.permissions[f.JOSH["login"]] = "read"
        plan = self.resolve_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "resolve_thread") as resolve:
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, None, self.head_dir)
        resolve.assert_not_called()
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        self.assertIn(j.NOT_AUTHORIZED_NOTE, self.pr.roots[-1].body)
        self.assertNotIn("agent-review-resolved", self.pr.roots[-1].body)
        # A code-confirmed fix reported by that member is answered and kept open too.
        self.pr.roots = [root]
        plan = self.reply_plan(proof, root)
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "resolve_thread") as resolve:
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, self.answer("fixed"), self.head_dir)
        resolve.assert_not_called()
        self.assertIn(j.NOT_AUTHORIZED_NOTE, self.pr.roots[-1].body)
        # A failed permission lookup never authorizes.
        self.pr.roots = [root]
        self.repo.get_collaborator_permission = lambda login: (_ for _ in ()).throw(RuntimeError("lookup failed"))
        plan = self.resolve_plan(proof, root)
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "resolve_thread") as resolve:
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, None, self.head_dir)
        resolve.assert_not_called()
        # The PR author may resolve even with read access.
        self.pr.roots = [root]
        self.pr.user = NS(login=f.JOSH["login"])
        plan = self.resolve_plan(proof, root)
        with patch.object(j, "records", return_value={123: proof}):
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, None, self.head_dir)
        self.assertTrue(self.pr.threads[root.id]["isResolved"])

    def test_authority_revoked_before_the_mutation_never_resolves(self):
        proof, root = self.prior_threads()
        plan = self.resolve_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        answers = iter(["write"])
        self.repo.get_collaborator_permission = lambda login: next(answers, "read")
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "resolve_thread") as resolve, \
                patch("sys.stdout", new_callable=io.StringIO):
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, None, self.head_dir)
        resolve.assert_not_called()
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        self.assertIn(j.COMMAND_NOT_ATTEMPTED_NOTE, self.pr.roots[-1].body)
        self.assertNotIn("awaiting GitHub confirmation", self.pr.roots[-1].body)

    def test_authority_revoked_only_at_the_final_recheck_never_resolves(self):
        # Admission, the first recheck and the snapshot pass; authority is lost
        # only at the check immediately before the mutation.
        proof, root = self.prior_threads()
        plan = self.resolve_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        answers = iter(["write", "write"])
        self.repo.get_collaborator_permission = lambda login: next(answers, "read")
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "resolve_thread") as resolve, \
                patch("sys.stdout", new_callable=io.StringIO):
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, None, self.head_dir)
        resolve.assert_not_called()
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        self.assertIn(j.COMMAND_NOT_ATTEMPTED_NOTE, self.pr.roots[-1].body)

    def test_unrelated_or_deleted_roots_never_cancel_verification_of_owned_threads(self):
        proof, root = self.prior_threads()
        # Another bot's thread whose root fetch would fail must never be fetched.
        self.pr.threads[987654] = {"id": "THREAD_other", "isResolved": False, "isOutdated": True}
        fetch, fetched = self.pr.get_review_comment, []
        def recorded(comment_id):
            fetched.append(comment_id)
            return fetch(comment_id)
        self.pr.get_review_comment = recorded
        self.assertEqual([root.id], [item["root"]["id"] for item in j.collect_threads(self.pr, {123: proof})])
        self.assertNotIn(987654, fetched, "an unowned root was fetched")
        # A deleted owned root leaves only its own thread out.
        from github import GithubException
        self.pr.get_review_comment = lambda comment_id: (_ for _ in ()).throw(GithubException(404, {"message": "Not Found"}))
        self.assertEqual([], j.collect_threads(self.pr, {123: proof}))

    def test_pending_or_incomplete_verification_reply_is_not_a_decision(self):
        proof, root = self.prior_threads()
        marker = f.verification_marker(root.id, "a" * 40)
        for body in (marker + "\nReply pending; no completed answer yet.",
                     marker + "\nReply incomplete; inspect Actions and post a new reply to retry."):
            self.pr.roots = [root]
            self.pr.create_review_comment_reply(root.id, body)
            self.assertEqual(1, len(j.collect_threads(self.pr, {123: proof})), body)
        self.pr.roots = [root]
        self.pr.create_review_comment_reply(root.id, marker + "\n" + j.VERIFIED_PREFIX + "a" * 40 + ":\n\n- evidence")
        self.assertEqual([], j.collect_threads(self.pr, {123: proof}))

    def test_command_flag_and_existing_answer_never_authorize_an_extra_resolve(self):
        proof, root = self.prior_threads()
        plan = self.resolve_plan(proof, root)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        for value in (False, 1):
            with patch.object(j, "records", return_value={123: proof}), self.assertRaises(e.ReviewAbort):
                j.publish_reply(self.repo, self.pr, ctx, policy, dict(plan, resolve_command=value), None, self.head_dir)
        self.assertFalse(self.pr.threads[root.id]["isResolved"])
        self.pr.roots = [root]
        plan = self.reply_plan(proof, root)
        text = self.answer("fixed")["output"]["answer"]
        self.pr.create_review_comment_reply(root.id, f.reply_marker(500, root.id) + "\n" + text)
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "resolve_thread") as resolve:
            j.publish_reply(self.repo, self.pr, ctx, policy, plan, self.answer("fixed"), self.head_dir)
        resolve.assert_not_called()
        self.assertFalse(self.pr.threads[root.id]["isResolved"])

    def test_verification_continues_after_failures_and_emits_one_fixed_note(self):
        import io
        (self.head_dir / "helper.py").write_text("changed\n")
        (self.head_dir / "third.py").write_text("changed\n")
        e.git(self.head_dir, "add", "--", "helper.py", "third.py")
        e.git(self.head_dir, "commit", "-qm", "fixture files")
        self.pr.head.sha = e.git(self.head_dir, "rev-parse", "HEAD").strip()
        os.environ["REVIEW_HEAD"] = self.pr.head.sha
        self.build([finding(path=p) for p in ("app.py", "helper.py", "third.py")])
        j.publish_artifact(self.artifact, self.head_dir)
        proof = json.loads((self.root / "completion.json").read_text())
        self.advance_head("app.py")
        for state in self.pr.threads.values(): state["isOutdated"] = True
        plan = self.prepare_verification(proof)
        self.assertEqual(3, len(plan["threads"]))
        verification = self.verified(plan)
        verification["receipt"]["files_opened"] = ["app.py", "helper.py", "third.py"]
        for item, target in zip(verification["output"]["threads"], plan["threads"]):
            item["evidence"][0]["path"] = target["findings"][0]["path"]
        # One bad evidence location and one failed mutation must not suppress the
        # remaining thread. The evidence failure must be handled per thread too.
        verification["output"]["threads"][0]["evidence"][0]["line"] = 999
        resolve = f.resolve_thread
        attempted = []
        def sometimes(pr, thread):
            attempted.append(thread)
            if thread != plan["threads"][-1]["thread"]: raise RuntimeError("private")
            resolve(pr, thread)
        policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
        with patch.object(j, "records", return_value={123: proof}), patch.object(f, "resolve_thread", side_effect=sometimes), \
                patch("sys.stdout", new_callable=io.StringIO) as output:
            j.publish_verification(self.repo, self.pr, ctx, policy, plan, verification, self.head_dir)
        self.assertEqual([t["thread"] for t in plan["threads"][1:]], attempted)
        self.assertEqual(j.FOLLOWUP_NOTE + "\n", output.getvalue())
        self.assertEqual(1, sum(t["isResolved"] for t in self.pr.threads.values()))

    def test_publication_request_count_with_real_pygithub_comments_ignores_unrelated_replies(self):
        from github.PullRequest import PullRequest
        proof, root = self.prior_threads()
        plan = self.prepare_verification(proof)
        fake = self.pr
        measurements = []
        for unrelated in (0, 30):  # Fixture sizes, not a production request budget.
            fake.roots = [root]
            fake.threads[root.id]["isResolved"] = False
            for index in range(unrelated):
                fake.roots.append(Comment(1000 + index, "Unrelated bot answer", NS(**f.BOT), {"in_reply_to_id": 9999}))
            counts = {"list": 0, "comments": []}
            def payload(comment):
                return dict(comment.raw_data, path=getattr(comment, "path", "app.py"),
                    commit_id=getattr(comment, "commit_id", fake.head.sha),
                    url=f"https://api.github.com/repos/example/project/pulls/comments/{comment.id}")
            class Requester:
                per_page = 100  # GitHub's published REST maximum; all fixture comments fit one page.
                is_not_lazy = True
                is_lazy = False
                graphql_query = staticmethod(fake.graphql_query)
                graphql_named_mutation = staticmethod(fake.graphql_named_mutation)
                def requestJsonAndCheck(self, method, url, parameters=None, headers=None, input=None):
                    if url.endswith("/pulls/1/comments"):
                        counts["list"] += 1
                        return {}, [payload(c) for c in fake.roots]
                    if "/pulls/comments/" in url:
                        ident = int(url.rsplit("/", 1)[-1])
                        counts["comments"].append(ident)
                        comment = fake.get_review_comment(ident)
                        if method == "PATCH": comment.edit(input["body"])
                        return {}, payload(comment)
                    if "/reviews/" in url:
                        review = fake.reviews[int(url.rsplit("/", 1)[-1])]
                        return {}, {"id": review.id, "user": f.BOT, "commit_id": review.commit_id, "state": review.state}
                    raise AssertionError((method, url))
            real = PullRequest(Requester(), {}, {"url": "https://api.github.com/repos/example/project/pulls/1",
                "node_id": fake.node_id, "number": 1, "state": "open",
                "head": {"sha": fake.head.sha, "repo": {"full_name": self.repo.full_name, "fork": False}},
                "base": {"sha": fake.base.sha, "repo": {"full_name": self.repo.full_name}}}, completed=True)
            def create(root_id, body):
                return real.get_review_comment(fake.create_review_comment_reply(root_id, body).id)
            real.create_review_comment_reply = create
            policy, depth = j.settings(); ctx, _ = j.context(policy, depth)
            with patch.object(self.repo, "get_pull", return_value=real), patch.object(j, "records", return_value={123: proof}) as loaded:
                j.publish_verification(self.repo, real, ctx, policy, plan, self.verified(plan), self.head_dir)
            self.assertTrue(fake.threads[root.id]["isResolved"])
            self.assertEqual(1, loaded.call_count)
            self.assertEqual(1, counts["list"])
            self.assertTrue(set(counts["comments"]) <= {root.id, fake.roots[-1].id}, counts)
            measurements.append(len(counts["comments"]))
        self.assertEqual(measurements[0], measurements[1])

    def test_resolution_footer_size_identity_and_readback_fail_closed(self):
        proof, root = self.prior_threads()
        posted = self.pr.create_review_comment_reply(root.id, f.reply_marker(500, root.id) + "\nAnswer")
        text = posted.body
        with patch.object(f, "MAX_DATA_BYTES", len(text.encode())), self.assertRaises(e.ReviewAbort):
            j.resolution_body(text, root.id, False)
        for fault in ("identity", "parent", "changed", "readback"):
            posted.user = NS(**f.BOT); posted.raw_data["in_reply_to_id"] = root.id; posted.edit(text)
            if fault == "identity": posted.user.id = 1
            if fault == "parent": posted.raw_data["in_reply_to_id"] = 999
            if fault == "changed": posted.edit("Edited")
            def no_edit(_): pass
            edit = patch.object(posted, "edit", side_effect=no_edit) if fault == "readback" else patch.object(f, "full_sha", wraps=f.full_sha)
            with edit, self.assertRaises(e.ReviewAbort):
                j.finish_resolution(self.pr, posted, root.id, True, text)


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
