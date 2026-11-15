# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Verified inline publication and the stable v2 completion envelope."""
import base64
import hashlib
import json
from enum import Enum

try:
    from . import review_followups as followups, review_reliability as reliability
    from .review_engine import ReviewAbort
except ImportError:
    import review_followups as followups
    import review_reliability as reliability
    from review_engine import ReviewAbort

class _Check(Enum):
    """Fixed diagnostics for the existing GitHub publication checks; never API text."""

    UNKNOWN = "unknown"
    BINDING = "binding"
    PRIOR_ROOT = "prior_root"
    REVIEW_AUTHOR = "review_author"
    REVIEW_ID = "review_id"
    REVIEW_COMMIT = "review_commit"
    COMMENT_COUNT = "comment_count"
    COMMENT_PAIR = "comment_pair"
    COMMENT_DUPLICATE = "comment_duplicate"
    COMMENT_IDENTITY = "comment_identity"
    FILE_AUTHOR = "file_author"
    FILE_ID = "file_id"
    FILE_CREATED_ID = "file_created_id"
    FILE_COMMIT = "file_commit"
    FILE_ORIGINAL_COMMIT = "file_original_commit"
    FILE_PATH = "file_path"
    FILE_BODY = "file_body"
    FILE_REPLY = "file_reply"
    FILE_SUBJECT = "file_subject"
    FILE_REVIEW_ID = "file_review_id"
    FILE_REVIEW_STATE = "file_review_state"


class _PublicationCheck(ReviewAbort):
    def __init__(self, check):
        self.check = check if isinstance(check, _Check) else _Check.UNKNOWN
        super().__init__(f"Inline publication verification failed (check={self.check.value})")


def _require(condition, check):
    if not condition:
        raise _PublicationCheck(check)


class InlinePublisher:
    """Verify publications and make repeated marked requests idempotent per run."""

    def __init__(self, pr, head, check_binding, roots):
        self.pr, self.head = pr, head
        self.check_binding, self.roots = check_binding, roots
        self.create_review = pr.create_review
        self.reviews = {}

    def valid_root(self, comment, review_id):
        return (followups.identity(comment.user, followups.BOT)
                and followups.positive(comment.id) and comment.commit_id == self.head
                and comment.raw_data.get("in_reply_to_id") is None
                and comment.raw_data.get("pull_request_review_id") == review_id)

    def verify_binding(self):
        try:
            self.check_binding()
        except ReviewAbort:
            raise _PublicationCheck(_Check.BINDING) from None

    def __call__(self, *args, **kwargs):
        try:
            with reliability.phase("publication"):
                return self.publish(*args, **kwargs)
        except _PublicationCheck as error:
            raise ReviewAbort(f"Inline publication verification failed (check={error.check.value})") from None
        except Exception as error:
            # Includes HTTP 422: never let upstream retry an ambiguous write.
            raise ReviewAbort(f"Inline publication verification failed ({type(error).__name__})") from None

    def publish(self, *args, **kwargs):

        self.verify_binding()
        requested = kwargs.get("comments", [])
        remaining, reused = [], None
        # Only roots verified by this adapter in this run may suppress a write.
        # A copied marker, another bot workflow, or an older head isn't proof.
        live = {c.id: c for c in self.pr.get_review_comments()} if self.roots else {}
        for root in self.roots:
            comment = live.get(root["id"])
            if (comment is None or not self.valid_root(comment, root["review_id"])
                    or comment.path != root["path"]
                    or followups.digest(comment.body) != root["digest"]):
                raise _PublicationCheck(_Check.PRIOR_ROOT)
        for request in requested:
            found = False
            if request["body"]:
                for root in self.roots:
                    comment = live.get(root["id"])
                    if (comment is not None and self.valid_root(comment, root["review_id"])
                            and comment.path == root["path"] == request["path"]
                            and followups.digest(comment.body) == root["digest"]
                            and comment.body == request["body"]):
                        found, reused = True, self.reviews[root["review_id"]]
                        break
            if not found:
                remaining.append(request)
        if requested and not remaining:
            return reused
        result = self.create_review(*args, **dict(kwargs, comments=remaining))
        _require(followups.identity(result.user, followups.BOT), _Check.REVIEW_AUTHOR)
        _require(followups.positive(result.id), _Check.REVIEW_ID)
        _require(result.commit_id == self.head, _Check.REVIEW_COMMIT)
        comments = [c for c in self.pr.get_review_comments()
                    if c.raw_data.get("pull_request_review_id") == result.id]
        if len(comments) != len(remaining):
            raise _PublicationCheck(_Check.COMMENT_COUNT)
        verified = []
        # GitHub lists review comments by its own ordering, not the order in the
        # create-review request. Match the complete multiset; consume each request
        # exactly once so missing, duplicated or changed comments still fail closed.
        unmatched = list(remaining)
        seen = set()
        for comment in comments:
            request = next((r for r in unmatched
                            if comment.body == r["body"] and comment.path == r["path"]), None)
            _require(request is not None, _Check.COMMENT_PAIR)
            _require(comment.id not in seen, _Check.COMMENT_DUPLICATE)
            if (not self.valid_root(comment, result.id) or comment.body != request["body"]
                    or comment.path != request["path"]):
                raise _PublicationCheck(_Check.COMMENT_IDENTITY)
            unmatched.remove(request)
            seen.add(comment.id)
            verified.append({"id": comment.id, "review_id": result.id, "path": comment.path,
                             "commit": self.head, "digest": followups.digest(comment.body)})
        self.roots.extend(verified)
        self.reviews[result.id] = result
        return result


    def file_thread(self, path, body):
        """Create one file-level review thread and verify every identity it carries.

        GitHub's create-a-review-comment endpoint takes subject_type "file" with no line
        (https://docs.github.com/en/rest/pulls/comments#create-a-review-comment-for-a-pull-request);
        the batch create-a-review comments are not documented to accept it. One write per
        thread, never retried: an ambiguous failure aborts, and the head is rechecked
        before each separate write.
        """
        try:
            with reliability.phase("publication"):
                return self.publish_file(path, body)
        except _PublicationCheck as error:
            raise ReviewAbort(f"Inline publication verification failed (check={error.check.value})") from None
        except Exception as error:
            raise ReviewAbort(f"Inline publication verification failed ({type(error).__name__})") from None

    def publish_file(self, path, body):
        self.verify_binding()
        created = self.pr.create_review_comment(body, self.head, path, subject_type="file")
        comment = self.pr.get_review_comment(created.id)
        raw = comment.raw_data
        review_id = raw.get("pull_request_review_id")
        _require(followups.identity(comment.user, followups.BOT), _Check.FILE_AUTHOR)
        _require(followups.positive(comment.id), _Check.FILE_ID)
        _require(comment.id == created.id, _Check.FILE_CREATED_ID)
        _require(comment.commit_id == self.head, _Check.FILE_COMMIT)
        _require(raw.get("original_commit_id") == self.head, _Check.FILE_ORIGINAL_COMMIT)
        _require(comment.path == path, _Check.FILE_PATH)
        _require(comment.body == body, _Check.FILE_BODY)
        _require(raw.get("in_reply_to_id") is None, _Check.FILE_REPLY)
        # GitHub returns line=1 even for subject_type=file responses. Subject
        # type identifies the thread; the verified body carries its location.
        _require(raw.get("subject_type") == "file", _Check.FILE_SUBJECT)
        _require(followups.positive(review_id), _Check.FILE_REVIEW_ID)
        review = self.pr.get_review(review_id)
        _require(followups.identity(review.user, followups.BOT), _Check.REVIEW_AUTHOR)
        _require(review.commit_id == self.head, _Check.REVIEW_COMMIT)
        _require(review.state == "COMMENTED", _Check.FILE_REVIEW_STATE)
        self.roots.append({"id": comment.id, "review_id": review_id, "path": path, "commit": self.head,
                           "digest": followups.digest(body)})
        return comment


STATUS_MARKER = "agent-review-status:v1"
RESULT_MARKER = "agent-review-result:v1"


def result_manifest(evidence):
    """Encode findings without a digest field. The digest covers the comment that contains this."""
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode()


def result_comment(manifest):
    return f"<!-- {RESULT_MARKER} {manifest} -->"
