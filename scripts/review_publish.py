# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Verified inline publication and the stable v2 completion envelope."""
import base64
import hashlib
import json

try:
    from . import review_followups as followups, review_reliability as reliability
    from .review_engine import ReviewAbort
except ImportError:
    import review_followups as followups
    import review_reliability as reliability
    from review_engine import ReviewAbort

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

    def __call__(self, *args, **kwargs):
        try:
            with reliability.phase("publication"):
                return self.publish(*args, **kwargs)
        except Exception as error:
            # Includes HTTP 422: never let upstream retry an ambiguous write.
            raise ReviewAbort(f"Inline publication verification failed ({type(error).__name__})") from None

    def publish(self, *args, **kwargs):

        self.check_binding()
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
                raise ReviewAbort("Previously verified inline publication changed")
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
        if (not followups.identity(result.user, followups.BOT)
                or not followups.positive(result.id) or result.commit_id != self.head):
            raise ReviewAbort("Inline review publication identity could not be verified")
        comments = [c for c in self.pr.get_review_comments()
                    if c.raw_data.get("pull_request_review_id") == result.id]
        if len(comments) != len(remaining):
            raise ReviewAbort("Inline review publication was partial")
        verified = []
        for comment, request in zip(comments, remaining):
            if (not self.valid_root(comment, result.id) or comment.body != request["body"]
                    or comment.path != request["path"]):
                raise ReviewAbort("Inline comment publication could not be verified")
            verified.append({"id": comment.id, "review_id": result.id, "path": comment.path,
                             "commit": self.head, "digest": followups.digest(comment.body)})
        self.roots.extend(verified)
        self.reviews[result.id] = result
        return result


STATUS_MARKER = "agent-review-status:v1"
RESULT_MARKER = "agent-review-result:v1"


def result_manifest(evidence):
    """Encode findings without a digest field. The digest covers the comment that contains this."""
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    return base64.b64encode(raw).decode()


def result_comment(manifest):
    return f"<!-- {RESULT_MARKER} {manifest} -->"
