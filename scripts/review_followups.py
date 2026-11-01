# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""Immutable completion evidence, commit scopes, and inline reply admission.

Publisher contracts:
https://docs.github.com/en/rest/actions/workflow-runs#get-a-workflow-run
https://docs.github.com/en/rest/actions/artifacts#download-an-artifact
https://github.com/actions/upload-artifact#v4---whats-new
https://docs.github.com/en/rest/pulls/comments
https://docs.github.com/en/rest/orgs/members#get-organization-membership-for-a-user
https://docs.github.com/en/rest/commits/commits#compare-two-commits

The result/artifact schema below is this repository's v2 completion contract.
It extends v1; old records never admit a reply or an incremental scope.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import zipfile
from urllib.parse import urlparse

if __package__:
    from .agent_review_contract import MAX_DATA_BYTES
    from . import review_reliability as reliability, review_learnings as learnings
else:
    from agent_review_contract import MAX_DATA_BYTES
    import review_reliability as reliability
    import review_learnings as learnings


class FollowupAbort(Exception):
    """Fail closed before suppressing review or publishing an unbound reply."""


# GitHub account identity: GET /users/github-actions%5Bbot%5D (2026-10-18).
BOT = {"login": "github-actions[bot]", "id": 41898282, "type": "Bot"}
JOSH = {"login": "luckyPipewrench", "id": 142104046, "type": "User"}
SHA = r"[0-9a-f]{40}"
DIGEST = r"[0-9a-f]{64}"
STATUS = re.compile(rf"<!-- agent-review-status:v1 state=success key=({DIGEST}) summary=([1-9][0-9]*) digest=({DIGEST}) -->")
RESULT = re.compile(r"<!-- agent-review-result:v1 ([A-Za-z0-9+/=]+) -->")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def identity(user, expected):
    return all(getattr(user, key, None) == value for key, value in expected.items())


def positive(value):
    return type(value) is int and value > 0


def full_sha(value):
    return isinstance(value, str) and re.fullmatch(SHA, value) is not None


def context(settings, repository, number, head, base, source, profile_sha, profile_hash, config_hash, depth):
    return {"repository": repository, "number": int(number), "head": head, "base": base,
            "reviewer": source, "profile_sha": profile_sha, "profile_hash": profile_hash,
            "common_config": config_hash, "depth": depth,
            "models": [settings.config.model, *settings.config.fallback_models]}


def same_policy(old, current, *, ownership=False):
    # Thread ownership ignores optional context. Coverage compares file content
    # and eligible entries; the learnings commit remains provenance only.
    ignored = {"head", "learnings"}
    if (not ownership and ("learnings" in old or "learnings" in current)
            and (not learnings.valid_identity(old.get("learnings"))
                 or not learnings.valid_identity(current.get("learnings"))
                 or old["learnings"]["digest"] != current["learnings"]["digest"])):
        return False
    return (set(old) == set(current)
            and all(old[key] == current[key] for key in current if key not in ignored))


def valid_findings(findings):
    if not isinstance(findings, list):
        return False
    for finding in findings:
        if (not isinstance(finding, dict)
                or set(finding) != {"path", "line", "end_line", "title", "body"}
                or any(not isinstance(finding[k], str) or not finding[k].strip() for k in ("path", "title", "body"))):
            return False
        if finding["path"] == "unknown" and finding["line"] is None and finding["end_line"] is None:
            continue
        if not positive(finding["line"]) or not positive(finding["end_line"]) or finding["end_line"] < finding["line"]:
            return False
    return True


def parse_record(comment, comments):
    """Validate both structured content and its independently published summary."""
    if not identity(comment.user, BOT) or not positive(comment.id) or not isinstance(comment.body, str):
        return None
    status = STATUS.fullmatch(comment.body.split("\n", 1)[0])
    results = RESULT.findall(comment.body)
    if not status or len(results) != 1:
        return None
    try:
        raw = base64.b64decode(results[0], validate=True)
        if len(raw) > MAX_DATA_BYTES:
            return None
        record = strict_json(raw)
        if "digest" in record:
            return None
        record["digest"] = status[3]
        ctx = record["context"]
        summary = comments.get(record["summary"])
        if (record["version"] != 2 or type(record["version"]) is not int
                or type(record["number"]) is not int
                or type(ctx["number"]) is not int or not positive(ctx["number"])
                or not all(full_sha(ctx[k]) for k in ("base", "head", "reviewer", "profile_sha"))
                or not all(isinstance(ctx[k], str) and re.fullmatch(DIGEST, ctx[k]) for k in ("profile_hash", "common_config"))
                or ("learnings" in ctx and not learnings.valid_identity(ctx["learnings"]))
                or ctx["depth"] not in ("default", "deep", "codex", "codex-deep")
                or not isinstance(ctx["models"], list) or not ctx["models"]
                or any(not isinstance(m, str) or not m for m in ctx["models"])
                or not isinstance(ctx["repository"], str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", ctx["repository"])
                or not positive(record["summary"]) or not positive(record["run_id"])
                or record["key"] != hashlib.sha256(json.dumps(ctx, sort_keys=True).encode()).hexdigest()
                or tuple(status.groups()) != (record["key"], str(record["summary"]), record["digest"])
                or summary is None or not identity(summary.user, BOT)
                or not isinstance(summary.body, str) or digest(summary.body) != record["digest"]
                or RESULT.findall(summary.body) != results
                or any(record[k] != ctx[k] for k in ("repository", "number", "base", "head", "depth"))
                or not valid_findings(record["findings"]) or not isinstance(record["roots"], list)):
            return None
        scope = record["scope"]
        if (set(scope) != {"kind", "from", "to", "parent", "full_head"}
                or scope["kind"] not in ("full", "incremental", "unchanged")
                or not full_sha(scope["from"]) or scope["to"] != ctx["head"]
                or type(scope["full_head"]) is not bool
                or (scope["kind"] == "full" and (scope["parent"] is not None or scope["from"] != ctx["base"] or not scope["full_head"]))
                or (scope["kind"] != "full" and (not positive(scope["parent"]) or scope["full_head"]))
                or (scope["kind"] == "unchanged" and scope["from"] != scope["to"])
                or (scope["kind"] == "incremental" and scope["from"] == scope["to"])):
            return None
        seen = set()
        for root in record["roots"]:
            if (set(root) != {"id", "review_id", "digest", "path", "commit"}
                    or not positive(root["id"]) or root["id"] in seen or not positive(root["review_id"])
                    or not isinstance(root["digest"], str) or not re.fullmatch(DIGEST, root["digest"])
                    or not isinstance(root["path"], str) or not root["path"] or root["commit"] != ctx["head"]):
                return None
            seen.add(root["id"])
        return record
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def trusted_run(run, record):
    """A bot comment is not run provenance. Require the immutable run artifact."""
    data = run.raw_data
    ctx = record["context"]
    refs = data.get("referenced_workflows", [])
    return (data.get("id") == record["run_id"] and data.get("run_attempt") == 1
            and data.get("event") == "issue_comment" and data.get("status") == "completed"
            and data.get("conclusion") == "success"
            and data.get("path") == ".github/workflows/agent-review.yaml"
            and data.get("head_sha") == ctx["profile_sha"]
            and data.get("repository", {}).get("full_name") == ctx["repository"]
            and data.get("head_repository", {}).get("full_name") == ctx["repository"]
            and all(data.get("actor", {}).get(k) == v for k, v in JOSH.items())
            and all(data.get("triggering_actor", {}).get(k) == v for k, v in JOSH.items())
            and isinstance(refs, list) and any(
                isinstance(ref, dict) and isinstance(ref.get("path"), str)
                and ref["path"].startswith("pipelab-org/agent-review/.github/workflows/agent-review-reusable.yaml@")
                and ref.get("sha") == ctx["reviewer"] for ref in refs))


# Seconds of socket inactivity allowed per artifact request. GitHub ends any REST
# request after 10 seconds of processing (docs.github.com/en/rest/using-the-rest-api/
# troubleshooting-the-rest-api); twice that leaves transfer headroom for the bounded
# completion archive and keeps a stalled download from consuming the inference budget.
ARTIFACT_TIMEOUT = 20


def download_record(artifact, token, timeout):
    """Follow GitHub's signed archive redirect without forwarding the token.

    Artifacts are zip archives. Only the single bounded regular completion.json
    from the trusted upload is accepted; never extract paths to the filesystem.
    """
    import requests
    endpoint = artifact.archive_download_url
    if not (urlparse(endpoint).scheme == "https" and urlparse(endpoint).netloc == "api.github.com"
            and re.fullmatch(r"/repos/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/actions/artifacts/[1-9][0-9]*/zip", urlparse(endpoint).path)):
        raise FollowupAbort("Invalid artifact archive URL")
    response = requests.get(endpoint, headers={"Authorization": "Bearer " + token,
                            "Accept": "application/vnd.github+json"}, timeout=timeout, allow_redirects=False)
    message = None
    if response.status_code in reliability.THROTTLE_STATUSES:
        try:
            body = response.json()
            message = body.get("message") if isinstance(body, dict) else None
        except ValueError:
            pass
    if response.status_code in reliability.THROTTLE_STATUSES and reliability.classify(
            response.status_code, response.headers, message) in reliability.RATE_CATEGORIES:
        # Retain only the existing classifier's fixed wording, never the response text.
        raise reliability.Throttled(response.status_code, response.headers,
                                    reliability.message_kind(message))
    if response.status_code != 302:
        raise FollowupAbort("Completion artifact unavailable")
    location = response.headers.get("Location", "")
    target = urlparse(location)
    if target.scheme != "https" or not target.hostname or target.username or target.password:
        raise FollowupAbort("Invalid artifact redirect")
    archive = requests.get(location, timeout=timeout, allow_redirects=False)
    archive.raise_for_status()
    # v4 artifacts cannot be changed after upload. Verify the publisher's digest
    # as well as the run binding; expired or old (digestless) artifacts fail closed.
    expected = artifact.raw_data.get("digest", "")
    if expected != "sha256:" + hashlib.sha256(archive.content).hexdigest():
        raise FollowupAbort("Completion artifact digest mismatch")
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        files = bundle.infolist()
        if len(files) != 1 or files[0].filename != "completion.json" or files[0].file_size > MAX_DATA_BYTES:
            raise FollowupAbort("Invalid completion artifact contents")
        return strict_json(bundle.read(files[0]))


def verified_records(repo, pr, current, load_artifact, *, contexts=None, ownership=False):
    try:
        comments = {c.id: c for c in pr.get_issue_comments()}
    except Exception as error:
        # Optional old proof never prevents a fresh full review, but a rate limit is
        # not missing proof: paid inference would follow into a throttled GitHub.
        if reliability.throttled(error):
            raise
        return {}
    records = {}
    for comment in comments.values():
        record = parse_record(comment, comments)
        if record is None or not any(same_policy(record["context"], ctx, ownership=ownership)
                                     for ctx in (contexts or [current])):
            continue
        try:
            run = repo.get_workflow_run(record["run_id"])
            if not trusted_run(run, record):
                continue
            artifacts = [a for a in run.get_artifacts() if a.name == "agent-review-completion" and not a.expired]
            if len(artifacts) != 1 or load_artifact(artifacts[0]) != record:
                continue
            records[record["run_id"]] = record
        except Exception as error:
            # Missing, stale, malformed, or unavailable proof must never suppress
            # needed review. Reply admission also refuses an unverified root.
            if reliability.throttled(error):
                raise
            continue
    return records


def compare_ancestor(repo, older, newer):
    comparison = repo.compare(older, newer)
    if (comparison.base_commit.sha != older or comparison.merge_base_commit.sha != older
            or comparison.status not in ("ahead", "identical")):
        raise FollowupAbort("Commit ancestry could not be verified")
    return comparison


def scope_for(repo, current, records):
    """Verify every link back to a full review; never infer a time-based anchor."""
    valid = {}
    if records:
        try:
            current_compare = repo.compare(current["base"], current["head"])
            if current_compare.base_commit.sha != current["base"] or not full_sha(current_compare.merge_base_commit.sha):
                raise FollowupAbort("Invalid current base relationship")
            merge_base = current_compare.merge_base_commit.sha
        except Exception as error:
            if reliability.throttled(error):
                raise
            records = {}
    pending = dict(records)
    while pending:
        progressed = False
        for run_id, record in list(pending.items()):
            scope = record["scope"]
            parent = valid.get(scope["parent"])
            if scope["kind"] != "full" and parent is None:
                continue
            try:
                if scope["kind"] == "full":
                    # A full review covers a PR merge-base diff. Pin that
                    # relationship even when the current base is ahead of it.
                    comparison = repo.compare(current["base"], record["head"])
                    if comparison.base_commit.sha != current["base"] or comparison.merge_base_commit.sha != merge_base:
                        raise FollowupAbort("Base relationship unavailable")
                else:
                    if scope["from"] != parent["head"]:
                        raise FollowupAbort("Broken completion chain")
                    compare_ancestor(repo, scope["from"], record["head"])
                compare_ancestor(repo, record["head"], current["head"])
                valid[run_id] = record
            except Exception as error:
                if reliability.throttled(error):
                    raise
            del pending[run_id]
            progressed = True
        if not progressed:
            break
    if not valid:
        return {"kind": "full", "from": current["base"], "to": current["head"], "parent": None, "full_head": True}
    # Immutable ancestry, not dates, determines the latest reviewed head. A
    # divergent set is unsafe; incomparable candidates fall back to full review.
    candidates = list(valid.values())
    newest = candidates[0]
    try:
        for candidate in candidates[1:]:
            if newest["head"] == candidate["head"]:
                continue
            try:
                compare_ancestor(repo, newest["head"], candidate["head"])
                newest = candidate
            except Exception as error:
                if reliability.throttled(error):
                    raise
                compare_ancestor(repo, candidate["head"], newest["head"])
    except Exception as error:
        if reliability.throttled(error):
            raise
        return {"kind": "full", "from": current["base"], "to": current["head"], "parent": None, "full_head": True}
    return {"kind": "unchanged" if newest["head"] == current["head"] else "incremental",
            "from": newest["head"], "to": current["head"], "parent": newest["run_id"], "full_head": False}


def admit_reply(event, env, repo, pr, membership):
    """Re-fetch the created comment and bind every event field before inference."""
    comment = event.get("comment", {})
    user = comment.get("user", {})
    number = event.get("pull_request", {}).get("number")
    if not (env.get("GITHUB_EVENT_NAME") == "pull_request_review_comment" and event.get("action") == "created"
            and env.get("GITHUB_RUN_ATTEMPT") == "1" and env.get("GITHUB_ACTOR") == user.get("login")
            and env.get("GITHUB_TRIGGERING_ACTOR") == user.get("login")
            and env.get("GITHUB_ACTOR_ID") == str(user.get("id")) and user.get("type") == "User"
            and positive(user.get("id")) and positive(comment.get("id")) and positive(number)
            and repo.full_name == env.get("GITHUB_REPOSITORY") == event.get("repository", {}).get("full_name")
            and pr.number == number and pr.state == "open"
            and pr.head.repo.full_name == repo.full_name and pr.head.repo.fork is False
            and pr.base.repo.full_name == repo.full_name):
        raise FollowupAbort("Reply event is ineligible")
    live = pr.get_review_comment(comment["id"])
    # Explicit field equality catches edited and stale/mismatched event payloads.
    # GitHub can store updated_at one second after created_at on a reply that
    # was not edited. Public cli/cli pull 14673 comments 4295607645 and
    # 4295608262 are on that pull's head, with commit_id == original_commit_id,
    # and updated_at one second after created_at. An edit is a mismatch of
    # body, creation time, commits, path, or thread ids against the created
    # event. updated_at is not that signal.
    fields = ("id", "body", "in_reply_to_id", "pull_request_review_id", "commit_id", "original_commit_id", "path", "created_at", "pull_request_url")
    updated = comment.get("updated_at")
    if (not identity(live.user, {k: user.get(k) for k in ("id", "login", "type")}) or any(live.raw_data.get(k) != comment.get(k) for k in fields)
            or not isinstance(comment.get("created_at"), str) or not comment["created_at"]
            or not isinstance(updated, str) or not updated
            or not isinstance(live.body, str) or not live.body.strip()
            or not positive(comment.get("in_reply_to_id"))
            or not full_sha(comment.get("commit_id"))
            or not full_sha(comment.get("original_commit_id"))
            or comment.get("pull_request_url") != f"https://api.github.com/repos/{repo.full_name}/pulls/{number}"):
        raise FollowupAbort("Reply event changed or is not an inline reply")
    if not all(user.get(k) == v for k, v in JOSH.items()):
        if (repo.owner.type != "Organization" or not membership(repo.owner.login, user["login"])):
            raise FollowupAbort("Reply author is not a current organization member")
    return live


def owned_root(pr, live, records):
    root_id = live.raw_data["in_reply_to_id"]
    root = pr.get_review_comment(root_id)
    verify_root(pr, root, records, path=live.path)
    return root


def verify_root(pr, root, records, *, path=None):
    """Bind immutable original commit, never GitHub's movable commit_id.

    Source: https://docs.github.com/en/graphql/reference/pulls#object-pullrequestreviewcomment
    originalCommit identifies the original commit; REST calls it original_commit_id.
    """
    for record in records.values():
        for entry in record["roots"]:
            if entry["id"] != root.id:
                continue
            review = pr.get_review(entry["review_id"])
            if (identity(root.user, BOT) and identity(review.user, BOT)
                    and root.raw_data.get("in_reply_to_id") is None
                    and root.raw_data.get("pull_request_review_id") == entry["review_id"]
                    and root.raw_data.get("original_commit_id") == entry["commit"] == record["head"]
                    and root.path == entry["path"] and (path is None or path == root.path)
                    and digest(root.body) == entry["digest"]
                    and review.commit_id == entry["commit"] and review.state == "COMMENTED"):
                return record, entry
    raise FollowupAbort("Inline root has no verified workflow provenance")


def reply_marker(trigger, root):
    return f"<!-- agent-review-reply:v1 trigger={trigger} root={root} -->"


def verification_marker(root, head):
    # Source: docs/agent-review.md thread verification publication contract.
    return f"<!-- agent-review-verified:v1 root={root} head={head} -->"


def resolution_marker(root):
    # Source: docs/agent-review.md inline reply contract. Records a resolution
    # attempt, not a success claim; later automatic checks leave it to a person.
    return f"<!-- agent-review-resolved:v1 root={root} -->"


def model_text(text):
    # Source: docs/agent-review.md publisher marker contract. Model-controlled
    # HTML comments cannot supply publisher state, even inside a code fence.
    return text.replace("<!--", "&lt;!--")


def listed_comments(pr):
    # PyGithub 2.10.0 CompletableGithubObject.raw_data calls _completeIfNeeded,
    # even for populated list entries. Read the received REST fields without I/O.
    # https://github.com/PyGithub/PyGithub/blob/v2.10.0/github/GithubObject.py
    return [comment._rawData for comment in pr.get_review_comments()]


def thread_snapshot(pr, thread_id, root_id):
    """Refresh state and comments of one thread, never the PR-wide comment list.

    Sources: https://docs.github.com/en/graphql/reference/pulls#object-pullrequestreviewthread
    https://docs.github.com/en/graphql/guides/using-pagination-in-the-graphql-api
    first:100 uses the published maximum connection page size.
    """
    query = """query($id: ID!, $cursor: String) {
      node(id: $id) { ... on PullRequestReviewThread {
        id isResolved isOutdated
        comments(first: 100, after: $cursor) {
          nodes { databaseId }
          pageInfo { hasNextPage endCursor }
        }
      } }
    }"""
    cursor, seen, ids = None, set(), []
    while True:
        _, data = pr.requester.graphql_query(query, {"id": thread_id, "cursor": cursor})
        if data.get("errors"):
            raise FollowupAbort("Review thread state unavailable")
        thread = data["data"]["node"]
        if (thread["id"] != thread_id or type(thread["isResolved"]) is not bool
                or type(thread["isOutdated"]) is not bool):
            raise FollowupAbort("Review thread state unavailable")
        for comment in thread["comments"]["nodes"]:
            value = comment["databaseId"]
            if not positive(value) or value in ids:
                raise FollowupAbort("Review thread comments unavailable")
            ids.append(value)
        page = thread["comments"]["pageInfo"]
        if page["hasNextPage"] is False:
            break
        cursor = page["endCursor"]
        if page["hasNextPage"] is not True or not isinstance(cursor, str) or not cursor or cursor in seen:
            raise FollowupAbort("Review thread pagination unavailable")
        seen.add(cursor)
    if not ids or ids[0] != root_id:
        raise FollowupAbort("Review thread root changed")
    # These GETs are confined to the target's own comments. raw_data completes
    # each target once; unrelated list entries never cause completion requests.
    return ({k: thread[k] for k in ("id", "isResolved", "isOutdated")},
            [pr.get_review_comment(value).raw_data for value in ids])


def review_threads(pr):
    """Map GraphQL thread IDs to their first (root) REST comment ID.

    Sources: https://docs.github.com/en/graphql/reference/pulls#object-pullrequestreviewthread
    https://docs.github.com/en/graphql/guides/using-pagination-in-the-graphql-api
    Connections allow at most 100 nodes per page. first:1 selects only the root;
    all replies are read through the existing paginated REST comment API.
    """
    query = """query($id: ID!, $cursor: String) {
      node(id: $id) { ... on PullRequest {
        reviewThreads(first: 100, after: $cursor) {
          nodes { id isResolved isOutdated comments(first: 1) { nodes { databaseId } } }
          pageInfo { hasNextPage endCursor }
        }
      } }
    }"""
    result, cursor, seen = {}, None, set()
    while True:
        _, data = pr.requester.graphql_query(query, {"id": pr.node_id, "cursor": cursor})
        if data.get("errors"):
            raise FollowupAbort("Review thread state unavailable")
        connection = data["data"]["node"]["reviewThreads"]
        for thread in connection["nodes"]:
            roots = thread["comments"]["nodes"]
            if (not isinstance(thread["id"], str) or not thread["id"]
                    or type(thread["isResolved"]) is not bool or type(thread["isOutdated"]) is not bool
                    or len(roots) != 1 or not positive(roots[0]["databaseId"])
                    or roots[0]["databaseId"] in result):
                raise FollowupAbort("Review thread state unavailable")
            result[roots[0]["databaseId"]] = {k: thread[k] for k in ("id", "isResolved", "isOutdated")}
        page = connection["pageInfo"]
        if page["hasNextPage"] is False:
            return result
        cursor = page["endCursor"]
        if page["hasNextPage"] is not True or not isinstance(cursor, str) or not cursor or cursor in seen:
            raise FollowupAbort("Review thread pagination unavailable")
        seen.add(cursor)


def resolve_thread(pr, thread):
    # Source: https://docs.github.com/en/graphql/reference/pulls#mutation-resolvereviewthread
    _, result = pr.requester.graphql_named_mutation(
        "resolveReviewThread", {"threadId": thread}, "thread { id isResolved }")
    if (result.get("thread", {}).get("id") != thread
            or result["thread"].get("isResolved") is not True):
        raise FollowupAbort("Review thread resolution could not be verified")


def write_completion(path, evidence):
    from pathlib import Path
    raw = canonical(evidence)
    if len(raw.encode()) > MAX_DATA_BYTES:
        raise FollowupAbort("Completion evidence exceeds the existing data budget")
    target = Path(path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.write_text(raw, encoding="utf-8")
    target.chmod(0o600)
