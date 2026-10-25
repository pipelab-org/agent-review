# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""Immutable input contract for historical review replay.

Why a new schema: FilePatchInfo is an in-memory PR-Agent dataclass and the v2
completion artifact records a review's output, not its input. Neither can be
loaded to re-run a review offline. This document carries exactly what the
adapter reads from GitHub in one review: the captured base/head identity, the
pull request text, each changed file as a FilePatchInfo, the repository files a
model may request at the head, repository languages, and the discussion as of
capture. Field names for files are FilePatchInfo's own.

The schema is closed: unknown metadata fields are rejected. Exporters must still
exclude later outcomes from free-text fields and keep oracles separate. Identity is the
SHA-256 of the exact file bytes; base/head are full commit SHAs as everywhere
else in this repository. Repository-relative paths use the same validator as
model source requests.

No byte ceiling is imposed on the file. Any number would be invented; what
reaches a model is already bounded by the adapter's existing token-budget
refusal, and the file is the operator's own local input.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from . import review_discovery as discovery
    from . import review_followups as followups
else:
    import review_discovery as discovery
    import review_followups as followups

SCHEMA = "agent-review-replay-snapshot"
VERSION = 1
EDIT_TYPES = ("ADDED", "DELETED", "MODIFIED", "RENAMED")

TOP = {"schema", "version", "repository", "number", "base", "head", "title", "branch",
       "description", "commits", "languages", "files", "sources", "absent", "discussion"}
FILE_REQUIRED = {"filename", "base_file", "head_file", "patch", "edit_type"}
FILE_OPTIONAL = {"old_filename", "num_plus_lines", "num_minus_lines"}
THREAD = {"id", "in_reply_to_id", "path", "body"}


class SnapshotError(Exception):
    """A snapshot is malformed or unreadable. Messages are fixed and carry no snapshot text."""


@dataclass(frozen=True)
class Snapshot:
    data: dict
    sha256: str

    @property
    def repository(self):
        return self.data["repository"]

    @property
    def number(self):
        return self.data["number"]

    @property
    def base(self):
        return self.data["base"]

    @property
    def head(self):
        return self.data["head"]

    def file_patch_infos(self):
        """The changed files as the pinned PR-Agent's own diff objects."""
        from pr_agent.algo.types import EDIT_TYPE, FilePatchInfo
        result = []
        for item in self.data["files"]:
            result.append(FilePatchInfo(
                item["base_file"], item["head_file"], item["patch"], item["filename"],
                edit_type=EDIT_TYPE[item["edit_type"]], old_filename=item.get("old_filename"),
                num_plus_lines=item.get("num_plus_lines", -1),
                num_minus_lines=item.get("num_minus_lines", -1)))
        return result


def read_regular(path):
    """Read one regular file without following a symlink in any path component."""
    path = Path(path)
    absolute = Path(os.path.abspath(path))
    if os.path.realpath(absolute) != str(absolute):
        raise SnapshotError("Snapshot path must not contain a symlink")
    try:
        fd = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise SnapshotError("Snapshot could not be opened") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SnapshotError("Snapshot must be a regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def repo_path(value):
    """The validator model source requests already use; no './' or other normalization."""
    if not isinstance(value, str) or value.startswith("./"):
        return False
    try:
        return discovery.requested_paths({"source_requests": [value]}) == [value]
    except discovery.DiscoveryAbort:
        return False


def _closed(obj, required, optional=frozenset()):
    return isinstance(obj, dict) and required <= set(obj) <= required | optional


def _text(value):
    return isinstance(value, str)


def validate(data):
    """Raise SnapshotError unless data is exactly the v1 contract. Returns data."""
    if not _closed(data, TOP) or data["schema"] != SCHEMA or type(data["version"]) is not int or data["version"] != VERSION:
        raise SnapshotError("Snapshot has an unknown schema or fields")
    if (not isinstance(data["repository"], str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", data["repository"])
            or not followups.positive(data["number"])
            or not followups.full_sha(data["base"]) or not followups.full_sha(data["head"])):
        raise SnapshotError("Snapshot repository or commit identity is invalid")
    if not all(_text(data[k]) for k in ("title", "branch", "description")) or not data["branch"]:
        raise SnapshotError("Snapshot pull request text is invalid")
    if not isinstance(data["commits"], list) or not all(_text(m) for m in data["commits"]):
        raise SnapshotError("Snapshot commit messages are invalid")
    langs = data["languages"]
    if (not isinstance(langs, dict)
            or any(not _text(k) or type(v) is not int or v < 0 for k, v in langs.items())):
        raise SnapshotError("Snapshot languages are invalid")
    if not isinstance(data["files"], list) or not data["files"]:
        raise SnapshotError("Snapshot has no changed files")
    changed, deleted = set(), set()
    for item in data["files"]:
        if not _closed(item, FILE_REQUIRED, FILE_OPTIONAL):
            raise SnapshotError("Snapshot file has an unknown shape")
        name = item["filename"]
        if (not repo_path(name) or name in changed or item["edit_type"] not in EDIT_TYPES
                or not all(_text(item[k]) for k in ("base_file", "head_file", "patch"))):
            raise SnapshotError("Snapshot file is invalid")
        old = item.get("old_filename")
        if (old is not None and not repo_path(old)) or (old is not None and item["edit_type"] != "RENAMED"):
            raise SnapshotError("Snapshot file rename is invalid")
        if item["edit_type"] == "RENAMED" and old is None:
            raise SnapshotError("Snapshot file rename is invalid")
        for key in ("num_plus_lines", "num_minus_lines"):
            if key in item and (type(item[key]) is not int or item[key] < 0):
                raise SnapshotError("Snapshot file line counts are invalid")
        if item["edit_type"] == "ADDED" and item["base_file"] != "":
            raise SnapshotError("Snapshot added file has base content")
        if item["edit_type"] == "DELETED":
            if item["head_file"] != "":
                raise SnapshotError("Snapshot deleted file has head content")
            deleted.add(name)
        changed.add(name)
    sources, absent = data["sources"], data["absent"]
    if not isinstance(sources, dict) or not isinstance(absent, list):
        raise SnapshotError("Snapshot source mapping is invalid")
    for path, text in sources.items():
        if not repo_path(path) or not _text(text) or path in changed or path in absent:
            raise SnapshotError("Snapshot source entry is invalid")
    if (any(not repo_path(p) for p in absent) or len(set(absent)) != len(absent)
            or any(p in changed and p not in deleted for p in absent)):
        raise SnapshotError("Snapshot absent-path list is invalid")
    discussion = data["discussion"]
    if (not _closed(discussion, {"as_of", "threads"}) or not _text(discussion["as_of"])
            or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", discussion["as_of"])
            or not isinstance(discussion["threads"], list)):
        raise SnapshotError("Snapshot discussion is invalid")
    ids = set()
    for thread in discussion["threads"]:
        if (not _closed(thread, THREAD) or not followups.positive(thread["id"]) or thread["id"] in ids
                or not _text(thread["body"]) or not repo_path(thread["path"])
                or not (thread["in_reply_to_id"] is None or followups.positive(thread["in_reply_to_id"]))):
            raise SnapshotError("Snapshot discussion thread is invalid")
        ids.add(thread["id"])
    return data


def load(path, expected_sha256=None):
    """Load and validate a snapshot; optionally pin it to an expected SHA-256 of its bytes."""
    raw = read_regular(path)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise SnapshotError("Snapshot does not match the expected SHA-256")
    try:
        data = followups.strict_json(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise SnapshotError("Snapshot is not valid UTF-8 JSON without duplicate fields") from None
    return Snapshot(validate(data), digest)
