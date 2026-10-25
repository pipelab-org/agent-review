# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts import review_replay_snapshot as snapshots

HEAD, BASE = "a" * 40, "b" * 40


def snapshot_dict(**changes):
    """A neutral synthetic snapshot; no historical content belongs in this repository."""
    data = {
        "schema": snapshots.SCHEMA, "version": snapshots.VERSION,
        "repository": "example/project", "number": 1, "base": BASE, "head": HEAD,
        "title": "Example", "branch": "fix/example", "description": "Untrusted description",
        "commits": ["Example change"], "languages": {"Python": 10},
        "files": [{"filename": "src/file.py", "base_file": "old\n", "head_file": "new\n",
                   "patch": "@@ -1 +1 @@\n-old\n+new\n", "edit_type": "MODIFIED",
                   "num_plus_lines": 1, "num_minus_lines": 1}],
        "sources": {"src/consumer.py": "existing_guard()\n"}, "absent": [],
        "discussion": {"as_of": "2026-01-01T00:00:00Z", "threads": []},
    }
    data.update(changes)
    return data


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.directory, ignore_errors=True))

    def write(self, data, name="snapshot.json", raw=None):
        path = self.directory / name
        path.write_bytes(raw if raw is not None else json.dumps(data).encode())
        return path

    def rejects(self, data, message=None):
        with self.assertRaises(snapshots.SnapshotError) as caught:
            snapshots.validate(copy.deepcopy(data))
        return caught.exception

    def test_valid_snapshot_identity_is_the_sha256_of_its_exact_bytes(self):
        path = self.write(snapshot_dict())
        loaded = snapshots.load(path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), loaded.sha256)
        self.assertEqual((HEAD, BASE, "example/project", 1), (loaded.head, loaded.base, loaded.repository, loaded.number))
        self.assertEqual(loaded.sha256, snapshots.load(path, loaded.sha256).sha256)
        with self.assertRaisesRegex(snapshots.SnapshotError, "expected SHA-256"):
            snapshots.load(path, "0" * 64)
        reformatted = self.write(snapshot_dict(), "other.json", raw=json.dumps(snapshot_dict(), indent=2).encode())
        self.assertNotEqual(loaded.sha256, snapshots.load(reformatted).sha256)

    def test_closed_schema_rejects_unknown_fields_at_every_level(self):
        # An oracle, label or later outcome must not be able to ride along into the model's context.
        for mutate in (lambda d: d.update(expected_findings=[]),
                       lambda d: d["files"][0].update(label="bug"),
                       lambda d: d["discussion"].update(outcome="fixed"),
                       lambda d: d["discussion"]["threads"].append(
                           {"id": 1, "in_reply_to_id": None, "path": "a.py", "body": "x", "resolved": True})):
            data = snapshot_dict()
            mutate(data)
            with self.subTest(data=str(data)[:80]):
                self.rejects(data)
        data = snapshot_dict()
        del data["languages"]
        self.rejects(data)

    def test_identity_and_scalar_fields_fail_closed(self):
        bad = {"schema": "other", "version": True, "repository": "no-slash", "number": 0, "base": "A" * 40,
               "head": "a" * 39, "title": 1, "branch": "", "description": None, "commits": "x",
               "languages": {"Python": -1}}
        for key, value in bad.items():
            with self.subTest(key=key):
                self.rejects(snapshot_dict(**{key: value}))
        for number in (True, -1, 1.0, "1"):
            self.rejects(snapshot_dict(number=number))
        self.rejects(snapshot_dict(version=2))
        self.rejects(snapshot_dict(files=[]))

    def test_repository_relative_paths_use_the_source_request_validator(self):
        for path in ("../escape.py", "/abs.py", "a//b.py", "./a.py", "a/./b.py", "a\\b.py", "dir/", "", "a\x00b.py"):
            with self.subTest(path=path):
                data = snapshot_dict()
                data["sources"] = {path: "x"}
                self.rejects(data)
                data = snapshot_dict(absent=[path])
                self.rejects(data)
                data = snapshot_dict()
                data["files"][0]["filename"] = path
                self.rejects(data)
                data = snapshot_dict()
                data["discussion"]["threads"] = [{"id": 1, "in_reply_to_id": None, "path": path, "body": "x"}]
                self.rejects(data)

    def test_overlaps_duplicates_and_file_shape_are_rejected(self):
        extra = copy.deepcopy(snapshot_dict()["files"][0])
        self.rejects(snapshot_dict(files=[extra, copy.deepcopy(extra)]))
        self.rejects(snapshot_dict(sources={"src/file.py": "x"}))
        self.rejects(snapshot_dict(absent=["src/file.py"]))
        self.rejects(snapshot_dict(absent=["a.py", "a.py"]))
        self.rejects(snapshot_dict(sources={"a.py": "x"}, absent=["a.py"]))
        deleted = dict(extra, edit_type="DELETED", head_file="")
        snapshots.validate(snapshot_dict(files=[deleted], absent=["src/file.py"]))
        self.rejects(snapshot_dict(files=[dict(deleted, head_file="still here")]))
        self.rejects(snapshot_dict(files=[dict(extra, edit_type="ADDED")]))  # base content on an added file
        self.rejects(snapshot_dict(files=[dict(extra, edit_type="UNKNOWN")]))
        self.rejects(snapshot_dict(files=[dict(extra, edit_type="RENAMED")]))  # rename without a source name
        self.rejects(snapshot_dict(files=[dict(extra, old_filename="old.py")]))  # name without a rename
        self.rejects(snapshot_dict(files=[dict(extra, num_plus_lines=-1)]))
        self.rejects(snapshot_dict(files=[{k: v for k, v in extra.items() if k != "patch"}]))
        renamed = dict(extra, edit_type="RENAMED", old_filename="old.py")
        snapshots.validate(snapshot_dict(files=[renamed]))

    def test_discussion_requires_an_as_of_time_and_closed_threads(self):
        for as_of in ("2026-01-01", "2026-01-01T00:00:00+00:00", "yesterday", 5, None):
            data = snapshot_dict()
            data["discussion"]["as_of"] = as_of
            with self.subTest(as_of=as_of):
                self.rejects(data)
        thread = {"id": 7, "in_reply_to_id": None, "path": "src/file.py", "body": "earlier"}
        data = snapshot_dict()
        data["discussion"]["threads"] = [thread, dict(thread, in_reply_to_id=7, id=8)]
        snapshots.validate(data)
        for bad in (dict(thread, id=0), dict(thread, id=True), dict(thread, in_reply_to_id=0),
                    dict(thread, body=None)):
            data = snapshot_dict()
            data["discussion"]["threads"] = [bad]
            with self.subTest(thread=bad):
                self.rejects(data)
        data["discussion"]["threads"] = [thread, dict(thread)]
        self.rejects(data)  # duplicate thread id

    def test_unreadable_inputs_fail_without_echoing_content(self):
        duplicate = b'{"schema": "' + snapshots.SCHEMA.encode() + b'", "schema": "x"}'
        cases = {"duplicate key": duplicate, "not utf-8": b"\xff\xfe", "not json": b"{", "array": b"[]"}
        for name, raw in cases.items():
            with self.subTest(name=name), self.assertRaises(snapshots.SnapshotError):
                snapshots.load(self.write(None, raw=raw))
        with self.assertRaises(snapshots.SnapshotError):
            snapshots.load(self.directory / "missing.json")
        with self.assertRaises(snapshots.SnapshotError):
            snapshots.load(self.directory)  # a directory is not a regular file

    def test_symlinks_anywhere_in_the_path_are_refused(self):
        real = self.write(snapshot_dict())
        link = self.directory / "link.json"
        folder = self.directory / "folder"
        folder.mkdir()
        (folder / "snapshot.json").write_bytes(real.read_bytes())
        try:
            os.symlink(real, link)
            os.symlink(folder, self.directory / "folder-link")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with self.assertRaises(snapshots.SnapshotError):
            snapshots.load(link)
        with self.assertRaises(snapshots.SnapshotError):
            snapshots.load(self.directory / "folder-link" / "snapshot.json")
        snapshots.load(folder / "snapshot.json")

    def test_file_patch_infos_reuse_the_pinned_diff_objects(self):
        try:
            from pr_agent.algo.types import EDIT_TYPE, FilePatchInfo
        except ImportError:
            self.skipTest("requires the pinned PR-Agent")
        renamed = dict(snapshot_dict()["files"][0], edit_type="RENAMED", old_filename="old.py")
        loaded = snapshots.load(self.write(snapshot_dict(files=[renamed])))
        (info,) = loaded.file_patch_infos()
        self.assertIsInstance(info, FilePatchInfo)
        self.assertEqual((EDIT_TYPE.RENAMED, "old.py", "new\n", "old\n", 1, 1),
                         (info.edit_type, info.old_filename, info.head_file, info.base_file,
                          info.num_plus_lines, info.num_minus_lines))
        optional = snapshots.load(self.write(snapshot_dict(), "plain.json"))
        data = snapshot_dict()
        for key in ("num_plus_lines", "num_minus_lines"):
            del data["files"][0][key]
        (plain,) = snapshots.load(self.write(data, "min.json")).file_patch_infos()
        self.assertEqual((-1, -1), (plain.num_plus_lines, plain.num_minus_lines))
        self.assertEqual(1, len(optional.file_patch_infos()))


if __name__ == "__main__":
    unittest.main()
