# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed state transitions for workflow-owned threads and coverage chains."""

import base64
import copy
import hashlib
import io
import json
import unittest
import zipfile
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

from scripts import review_followups as f

REPO = "example/project"
HEAD, BASE, SOURCE, PROFILE = [x * 40 for x in "abcd"]


def binding(head=HEAD):
    return dict(repository=REPO, number=1, head=head, base=BASE, reviewer=SOURCE,
                profile_sha=PROFILE, profile_hash="e" * 64, common_config="f" * 64,
                depth="default", models=["anthropic/claude-sonnet-5-5", "gpt-6-luna"])


def evidence(head=HEAD, run=123, parent=None, start=None):
    ctx = binding(head)
    record = dict(version=2, context=ctx, key=hashlib.sha256(json.dumps(ctx, sort_keys=True).encode()).hexdigest(),
                repository=REPO, number=1, head=head, base=BASE, depth="default", summary=8,
                digest=f.digest("summary"), run_id=run, findings=[], roots=[],
                scope=dict(kind="incremental" if parent else "full", to=head,
                           **{"from": start or BASE}, parent=parent, full_head=parent is None))
    payload = {k: v for k, v in record.items() if k != "digest"}
    record["digest"] = f.digest("summary\n<!-- agent-review-result:v1 " + base64.b64encode(f.canonical(payload).encode()).decode() + " -->")
    return record


def comments(record):
    payload = {k: v for k, v in record.items() if k != "digest"}
    manifest = "<!-- agent-review-result:v1 " + base64.b64encode(f.canonical(payload).encode()).decode() + " -->"
    summary = NS(id=8, body="summary\n" + manifest, user=NS(**f.BOT))
    body = (f"<!-- agent-review-status:v1 state=success key={record['key']} summary={record['summary']} digest={record['digest']} -->\n"
            + manifest)
    status = NS(id=9, body=body, user=NS(**f.BOT))
    return status, {8: summary, 9: status}


def run_data(record):
    return dict(id=record["run_id"], run_attempt=1, event="issue_comment", status="completed", conclusion="success",
                path=".github/workflows/agent-review.yaml", head_sha=PROFILE,
                repository={"full_name": REPO}, head_repository={"full_name": REPO}, actor=f.JOSH,
                triggering_actor=f.JOSH, referenced_workflows=[{
                    "path": f"pipelab-org/agent-review/.github/workflows/agent-review-reusable.yaml@{SOURCE}", "sha": SOURCE}])


class RecordTests(unittest.TestCase):
    def test_record_requires_complete_structured_context_and_intact_summary(self):
        good = evidence()
        c, all_comments = comments(good)
        self.assertEqual(good, f.parse_record(c, all_comments))
        mutations = [dict(good, version=True), dict(good, run_id=True), dict(good, number=True),
                     dict(good, summary=True), dict(good, findings=[None]), dict(good, context=dict(good["context"], reviewer="main")),
                     dict(good, digest="0" * 64), dict(good, scope=dict(good["scope"], full_head=False)),
                     dict(good, scope=dict(good["scope"], kind="partial")), dict(good, key="0" * 64),
                     dict(good, roots=[dict(id=1, review_id=2, digest="0" * 64, path="x", commit="bad")])]
        for broken in mutations:
            c, cs = comments(broken)
            with self.subTest(broken=broken):
                self.assertIsNone(f.parse_record(c, cs))
        c, cs = comments(good)
        cs[8].body += "tampered"
        self.assertIsNone(f.parse_record(c, cs))
        c, cs = comments(good)
        c.user.id = 1
        self.assertIsNone(f.parse_record(c, cs))
        c, cs = comments(good)
        c.body += "\n" + c.body.split("\n")[1]
        self.assertIsNone(f.parse_record(c, cs))

    def test_summary_and_status_must_bind_identical_findings(self):
        good = evidence()
        status, cs = comments(good)
        payload = {k: v for k, v in good.items() if k != "digest"}
        payload["findings"] = [{"path": "src/file.py", "line": 1, "end_line": 1,
                                "title": "Changed finding", "body": "Different summary"}]
        forged = "<!-- agent-review-result:v1 " + base64.b64encode(f.canonical(payload).encode()).decode() + " -->"
        status.body = status.body.split("\n")[0] + "\n" + forged
        self.assertIsNone(f.parse_record(status, cs))
        for name in ("number", "summary", "run_id"):
            for value in (True, 1.0, "1", 0, -1, None):
                bad = copy.deepcopy(good)
                bad[name] = value
                c, all_comments = comments(bad)
                with self.subTest(name=name, value=value):
                    self.assertIsNone(f.parse_record(c, all_comments))

    def test_duplicate_json_and_malformed_base64_are_refused(self):
        record = evidence()
        c, cs = comments(record)
        raw = f.canonical(record).replace('"version":2', '"version":2,"version":2')
        c.body = c.body.split("\n")[0] + "\n<!-- agent-review-result:v1 " + base64.b64encode(raw.encode()).decode() + " -->"
        self.assertIsNone(f.parse_record(c, cs))
        c.body = c.body.split("\n")[0] + "\n<!-- agent-review-result:v1 AA= -->"
        self.assertIsNone(f.parse_record(c, cs))

    def test_successful_loaded_workflow_and_source_must_match(self):
        record = evidence()
        good = run_data(record)
        self.assertTrue(f.trusted_run(NS(raw_data=good), record))
        for key, value in (("run_attempt", 2), ("event", "workflow_dispatch"), ("conclusion", "failure"),
                           ("status", "in_progress"), ("path", ".github/workflows/other.yaml"),
                           ("head_sha", "a" * 40), ("actor", {"login": "other"}),
                           ("referenced_workflows", []), ("head_repository", {"full_name": "fork/project"})):
            with self.subTest(key=key):
                self.assertFalse(f.trusted_run(NS(raw_data=dict(good, **{key: value})), record))

    def test_comment_is_insufficient_without_matching_immutable_artifact(self):
        record = evidence()
        c, cs = comments(record)
        repo, pr = MagicMock(), MagicMock()
        pr.get_issue_comments.return_value = list(cs.values())
        artifact = NS(name="agent-review-completion", expired=False)
        run = NS(raw_data=run_data(record), get_artifacts=lambda: [artifact])
        repo.get_workflow_run.return_value = run
        self.assertEqual({123: record}, f.verified_records(repo, pr, binding(), lambda a: record))
        for loader in (lambda a: {}, lambda a: (_ for _ in ()).throw(ValueError("unavailable"))):
            self.assertEqual({}, f.verified_records(repo, pr, binding(), loader))
        artifact.expired = True
        self.assertEqual({}, f.verified_records(repo, pr, binding(), lambda a: record))
        artifact.expired = False
        for key, value in (("reviewer", "0" * 40), ("profile_sha", "0" * 40), ("profile_hash", "0" * 64), ("depth", "deep"),
                           ("models", ["different"]), ("base", "0" * 40), ("common_config", "0" * 64)):
            with self.subTest(key=key):
                self.assertEqual({}, f.verified_records(repo, pr, dict(binding(), **{key: value}), lambda a: record))

    def test_archive_digest_paths_and_authorization_redirect(self):
        # No real network. Check exact archive bytes and token removal on the
        # signed URL, plus malformed ZIP and metadata boundaries.
        try:
            import requests
        except ImportError:
            self.skipTest("requests is exercised in the pinned runtime")
        record = evidence()
        def archive(name="completion.json", content=None):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as bundle:
                bundle.writestr(name, f.canonical(record) if content is None else content)
            return stream.getvalue()
        def artifact(data):
            return NS(archive_download_url="https://api.github.com/repos/example/project/actions/artifacts/9/zip",
                      raw_data={"digest": "sha256:" + hashlib.sha256(data).hexdigest()})
        for name, content, okay in (("completion.json", None, True), ("../completion.json", None, False),
                                    ("completion.json", "x" * (f.MAX_DATA_BYTES + 1), False)):
            data = archive(name, content)
            with patch.object(requests, "get", side_effect=[NS(status_code=302, headers={"Location": "https://storage.example/archive?sig=fixture"}),
                    NS(content=data, raise_for_status=lambda: None)]) as get:
                if okay:
                    self.assertEqual(record, f.download_record(artifact(data), "fixture-token", 300))
                else:
                    with self.assertRaises(f.FollowupAbort):
                        f.download_record(artifact(data), "fixture-token", 300)
                self.assertNotIn("headers", get.call_args_list[1].kwargs)
                self.assertFalse(get.call_args_list[1].kwargs["allow_redirects"])
        data = archive()
        with patch.object(requests, "get", side_effect=[NS(status_code=302, headers={"Location": "https://storage.example/archive"}),
                    NS(content=data + b"tampered", raise_for_status=lambda: None)]), self.assertRaises(f.FollowupAbort):
            f.download_record(artifact(data), "fixture-token", 300)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.repo = MagicMock()
        def compare(old, new):
            # Four commits in one ancestry, plus a divergent branch.
            order = [BASE, "1" * 40, "2" * 40, HEAD]
            if old == new:
                return NS(base_commit=NS(sha=old), merge_base_commit=NS(sha=old), status="identical")
            if old in order and new in order and order.index(old) < order.index(new):
                return NS(base_commit=NS(sha=old), merge_base_commit=NS(sha=old), status="ahead")
            return NS(base_commit=NS(sha=old), merge_base_commit=NS(sha=BASE), status="diverged")
        self.repo.compare.side_effect = compare

    def test_full_incremental_and_unchanged_chain(self):
        first = evidence("1" * 40)
        scope = f.scope_for(self.repo, binding(), {123: first})
        self.assertEqual("incremental", scope["kind"])
        self.assertEqual("1" * 40, scope["from"])
        self.assertFalse(scope["full_head"])
        second = evidence("2" * 40, 124, 123, "1" * 40)
        # Reverse order proves chain validation doesn't depend on API order.
        scope = f.scope_for(self.repo, binding(), {124: second, 123: first})
        self.assertEqual(124, scope["parent"])
        last = evidence(HEAD, 125, 124, "2" * 40)
        self.assertEqual("unchanged", f.scope_for(self.repo, binding(), {125: last, 123: first, 124: second})["kind"])

    def test_failed_missing_cyclic_or_divergent_parent_never_suppresses_review(self):
        cases = [{}, {124: evidence("2" * 40, 124, 123, "1" * 40)},
                 {124: evidence("2" * 40, 124, 124, "2" * 40)}, {123: evidence("9" * 40)}]
        for records in cases:
            with self.subTest(records=records):
                self.assertEqual("full", f.scope_for(self.repo, binding(), records)["kind"])
        first = evidence("1" * 40)
        second = evidence("2" * 40, 124, 123, "9" * 40)
        self.assertEqual(123, f.scope_for(self.repo, binding(), {123: first, 124: second})["parent"])

    def test_new_merge_base_invalidates_previous_full_context(self):
        previous = evidence("1" * 40)
        compare = self.repo.compare.side_effect
        def shifted(old, new):
            result = compare(old, new)
            if old == BASE and new == HEAD:
                result.merge_base_commit.sha = "1" * 40
            return result
        self.repo.compare.side_effect = shifted
        self.assertEqual("full", f.scope_for(self.repo, binding(), {123: previous})["kind"])

    def test_deciding_mutation_ancestry_guard_detects_force_push(self):
        record = evidence("9" * 40)
        self.assertEqual("full", f.scope_for(self.repo, binding(), {123: record})["kind"])
        with patch.object(f, "compare_ancestor", return_value=None):
            self.assertEqual("incremental", f.scope_for(self.repo, binding(), {123: record})["kind"])


class ReplyAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.user = dict(f.JOSH)
        self.data = dict(id=11, body="Why does this fail?", in_reply_to_id=10, pull_request_review_id=7,
                         commit_id=HEAD, original_commit_id=HEAD, path="src/file.py", created_at="2026-10-18T17:00:00Z",
                         updated_at="2026-10-18T17:00:00Z", pull_request_url=f"https://api.github.com/repos/{REPO}/pulls/1")
        self.live = NS(**self.data, user=NS(**self.user), raw_data=copy.deepcopy(self.data))
        self.repo = NS(full_name=REPO, owner=NS(login="example", type="Organization"))
        self.pr = NS(number=1, state="open", head=NS(sha=HEAD, repo=NS(full_name=REPO, fork=False)),
                     base=NS(sha=BASE, repo=NS(full_name=REPO)), get_review_comment=lambda i: self.live)
        self.event = dict(action="created", comment=dict(self.data, user=self.user), repository={"full_name": REPO}, pull_request={"number": 1})
        self.env = dict(GITHUB_EVENT_NAME="pull_request_review_comment", GITHUB_RUN_ATTEMPT="1",
                        GITHUB_ACTOR=self.user["login"], GITHUB_TRIGGERING_ACTOR=self.user["login"],
                        GITHUB_ACTOR_ID=str(self.user["id"]), GITHUB_REPOSITORY=REPO)

    def admit(self, membership=lambda *args: False):
        return f.admit_reply(self.event, self.env, self.repo, self.pr, membership)

    def test_josh_and_actual_current_org_member_only(self):
        self.assertIs(self.live, self.admit())
        self.user.update(login="member", id=123)
        self.live.user = NS(**self.user)
        self.env.update(GITHUB_ACTOR="member", GITHUB_TRIGGERING_ACTOR="member", GITHUB_ACTOR_ID="123")
        membership = MagicMock(return_value=True)
        self.assertIs(self.live, self.admit(membership))
        membership.assert_called_once_with("example", "member")
        with self.assertRaises(f.FollowupAbort):
            self.admit()
        self.repo.owner.type = "User"
        with self.assertRaises(f.FollowupAbort):
            self.admit(membership)

    def test_event_identity_rerun_edit_fork_closed_and_mismatch(self):
        for key, value in (("GITHUB_EVENT_NAME", "issue_comment"), ("GITHUB_RUN_ATTEMPT", "2"),
                           ("GITHUB_ACTOR_ID", "123"), ("GITHUB_TRIGGERING_ACTOR", "other")):
            old = self.env[key]
            self.env[key] = value
            with self.subTest(key=key), self.assertRaises(f.FollowupAbort):
                self.admit()
            self.env[key] = old
        self.event["action"] = "edited"
        with self.assertRaises(f.FollowupAbort): self.admit()
        self.event["action"] = "created"
        for key, value in (("body", "edited"), ("commit_id", "9" * 40), ("in_reply_to_id", None),
                           ("pull_request_url", "https://api.github.com/repos/other/project/pulls/1"),
                           ("created_at", "2026-10-18T17:00:01Z")):
            old = self.event["comment"][key]
            self.event["comment"][key] = value
            with self.subTest(key=key), self.assertRaises(f.FollowupAbort): self.admit()
            self.event["comment"][key] = old
        self.pr.state = "closed"
        with self.assertRaises(f.FollowupAbort): self.admit()
        self.pr.state = "open"
        self.pr.head.repo.fork = True
        with self.assertRaises(f.FollowupAbort): self.admit()
        self.pr.head.repo.fork = False
        self.user["type"] = "Bot"
        self.live.user.type = "Bot"
        with self.assertRaises(f.FollowupAbort): self.admit()

    def test_github_creation_clock_step_still_admits_same_head_reply(self):
        # The created event keeps equal timestamps. The refetch has the one-second
        # updated_at step GitHub stores on unedited replies.
        self.live.raw_data["updated_at"] = "2026-10-18T17:00:01Z"
        self.assertIs(self.live, self.admit())
        self.event["comment"]["updated_at"] = ""
        with self.assertRaises(f.FollowupAbort):
            self.admit()

    def test_earlier_commit_reply_still_requires_exact_event_and_full_commits(self):
        self.pr.head.sha = "9" * 40
        self.assertIs(self.live, self.admit())
        for field in ("commit_id", "original_commit_id"):
            original = self.event["comment"][field]
            for value in (None, "invalid"):
                self.live.raw_data[field] = self.event["comment"][field] = value
                with self.assertRaises(f.FollowupAbort): self.admit()
            self.live.raw_data[field] = self.event["comment"][field] = original

    def test_exact_root_manifest_review_and_digest_required(self):
        record = evidence()
        root = NS(id=10, user=NS(**f.BOT), body="Finding", path="src/file.py", commit_id=HEAD,
                  raw_data={"pull_request_review_id": 7, "original_commit_id": HEAD})
        review = NS(user=NS(**f.BOT), commit_id=HEAD, state="COMMENTED")
        self.pr.get_review_comment = lambda i: root
        self.pr.get_review = lambda i: review
        record["roots"] = [dict(id=10, review_id=7, digest=f.digest(root.body), path=root.path, commit=HEAD)]
        self.assertIs(root, f.owned_root(self.pr, self.live, {123: record}))
        with self.assertRaises(f.FollowupAbort): f.owned_root(self.pr, self.live, {})
        # commit_id follows the live head without changing the original identity.
        root.commit_id = "9" * 40
        self.pr.head.sha = root.commit_id
        self.assertIs(root, f.owned_root(self.pr, self.live, {123: record}))
        root.raw_data["original_commit_id"] = "8" * 40
        with self.assertRaises(f.FollowupAbort): f.owned_root(self.pr, self.live, {123: record})
        root.raw_data["original_commit_id"] = HEAD
        for obj, field, value in ((root, "body", "edited"),
                                  (root.user, "id", 1), (review, "state", "PENDING"),
                                  (root, "path", "other.py")):
            old = getattr(obj, field)
            setattr(obj, field, value)
            with self.subTest(field=field), self.assertRaises(f.FollowupAbort):
                f.owned_root(self.pr, self.live, {123: record})
            setattr(obj, field, old)


class ThreadStateTests(unittest.TestCase):
    def target_page(self, ids=(10,), **updates):
        value = {"id": "THREAD", "isResolved": False, "isOutdated": True,
            "comments": {"nodes": [{"databaseId": i} for i in ids],
                         "pageInfo": {"hasNextPage": False, "endCursor": None}}}
        value.update(updates)
        return {"data": {"node": value}}

    def test_target_snapshot_paginates_only_its_comments(self):
        first, second = self.target_page(), self.target_page((11,))
        first["data"]["node"]["comments"]["pageInfo"] = {"hasNextPage": True, "endCursor": "next"}
        query = MagicMock(side_effect=[({}, first), ({}, second)])
        fetch = MagicMock(side_effect=lambda i: NS(raw_data={"id": i}))
        pr = NS(get_review_comment=fetch, requester=NS(graphql_query=query))
        state, comments = f.thread_snapshot(pr, "THREAD", 10)
        self.assertEqual([{"id": 10}, {"id": 11}], comments)
        self.assertEqual({"id": "THREAD", "isResolved": False, "isOutdated": True}, state)
        self.assertEqual({"id": "THREAD", "cursor": "next"}, query.call_args.args[1])
        self.assertEqual([10, 11], [call.args[0] for call in fetch.call_args_list])

    def test_target_snapshot_rejects_unbound_partial_or_looped_state(self):
        bad = [dict(self.target_page(), errors=[{"message": "private"}]), self.target_page(()),
               self.target_page((99,)), self.target_page((True,)), self.target_page((10, 10))]
        for updates in ({"id": "other"}, {"isResolved": None}, {"isOutdated": 1}):
            bad.append(self.target_page(**updates))
        for page in ({"hasNextPage": None, "endCursor": "x"}, {"hasNextPage": True, "endCursor": ""}):
            data = self.target_page(); data["data"]["node"]["comments"]["pageInfo"] = page; bad.append(data)
        for data in bad:
            pr = NS(requester=NS(graphql_query=MagicMock(return_value=({}, data))))
            with self.subTest(data=data), self.assertRaises(f.FollowupAbort): f.thread_snapshot(pr, "THREAD", 10)
        first, second = self.target_page(), self.target_page((11,))
        for data in (first, second):
            data["data"]["node"]["comments"]["pageInfo"] = {"hasNextPage": True, "endCursor": "same"}
        pr = NS(requester=NS(graphql_query=MagicMock(side_effect=[({}, first), ({}, second)])))
        with self.assertRaises(f.FollowupAbort): f.thread_snapshot(pr, "THREAD", 10)

    def page(self, root=10, **updates):
        thread = dict(id="THREAD", isResolved=False, isOutdated=False, comments={"nodes": [{"databaseId": root}]})
        thread.update(updates)
        return {"data": {"node": {"reviewThreads": {"nodes": [thread],
            "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}

    def test_graphql_pagination_and_resolve_response_are_verified(self):
        first, second = self.page(), self.page(11, id="NEXT")
        first["data"]["node"]["reviewThreads"]["pageInfo"] = {"hasNextPage": True, "endCursor": "cursor"}
        query = MagicMock(side_effect=[({}, first), ({}, second)])
        mutation = MagicMock(return_value=({}, {"thread": {"id": "THREAD", "isResolved": True}}))
        pr = NS(node_id="PR", requester=NS(graphql_query=query, graphql_named_mutation=mutation))
        self.assertEqual({10, 11}, set(f.review_threads(pr)))
        self.assertEqual({"id": "PR", "cursor": "cursor"}, query.call_args.args[1])
        f.resolve_thread(pr, "THREAD")
        self.assertEqual(("resolveReviewThread", {"threadId": "THREAD"}, "thread { id isResolved }"), mutation.call_args.args)
        for response in ({}, {"thread": {"id": "other", "isResolved": True}},
                         {"thread": {"id": "THREAD", "isResolved": False}}):
            mutation.return_value = {}, response
            with self.assertRaises(f.FollowupAbort): f.resolve_thread(pr, "THREAD")

    def test_graphql_partial_malformed_or_repeated_state_fails_closed(self):
        bad = [dict(self.page(), errors=[{"message": "private"}])]
        for updates in ({"id": ""}, {"isResolved": None}, {"isOutdated": 1},
                        {"comments": {"nodes": []}}, {"comments": {"nodes": [{"databaseId": True}]}}):
            bad.append(self.page(**updates))
        for page in ({"hasNextPage": None, "endCursor": "cursor"},
                     {"hasNextPage": True, "endCursor": ""}):
            data = self.page(); data["data"]["node"]["reviewThreads"]["pageInfo"] = page; bad.append(data)
        for data in bad:
            pr = NS(node_id="PR", requester=NS(graphql_query=MagicMock(return_value=({}, data))))
            with self.subTest(data=data), self.assertRaises(f.FollowupAbort): f.review_threads(pr)
        data = self.page()
        data["data"]["node"]["reviewThreads"]["pageInfo"] = {"hasNextPage": True, "endCursor": "cursor"}
        second = self.page(11, id="NEXT")
        second["data"]["node"]["reviewThreads"]["pageInfo"] = {"hasNextPage": True, "endCursor": "cursor"}
        for pages in ([data, data], [data, second]):
            pr = NS(node_id="PR", requester=NS(graphql_query=MagicMock(side_effect=[({}, p) for p in pages])))
            with self.assertRaises(f.FollowupAbort): f.review_threads(pr)


if __name__ == "__main__":
    unittest.main()
