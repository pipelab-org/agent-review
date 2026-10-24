# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from scripts import review_replay_export as exporter
from scripts.review_replay_snapshot import SnapshotError


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        (self.repo/"code.py").write_text("value = 1\n")
        (self.repo/"consumer.py").write_text("from code import value\n")
        (self.repo/"binary.dat").write_bytes(b"\xff")
        self.git("add", ".")
        self.git("commit", "-qm", "Original fixture")
        self.base = self.git("rev-parse", "HEAD").strip()
        (self.repo/"code.py").write_text("value = 2\n")
        self.git("commit", "-qam", "Updated fixture")
        self.head = self.git("rev-parse", "HEAD").strip()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.DEVNULL).decode()

    def export(self, **kwargs):
        return exporter.export(self.repo, "example/project", 1, self.base, self.head, **kwargs)

    def test_export_uses_objects_not_worktree_and_omits_future_metadata(self):
        (self.repo/"code.py").write_text("future fix and expected answer\n")
        (self.repo/"consumer.py").write_text("future discussion\n")
        (self.repo/"oracle.json").write_text("private expected result")
        data = self.export(absent=["missing.py"])
        self.assertEqual("value = 2\n", data["files"][0]["head_file"])
        self.assertEqual("value = 1\n", data["files"][0]["base_file"])
        self.assertEqual("from code import value\n", data["sources"]["consumer.py"])
        self.assertNotIn("binary.dat", data["sources"])
        self.assertNotIn("oracle.json", data["sources"])
        self.assertEqual([], data["commits"])
        self.assertEqual([], data["discussion"]["threads"])
        self.assertTrue(data["files"][0]["patch"].startswith("@@"))
        self.assertNotIn("future", json.dumps(data))

    def test_external_diff_program_is_never_executed(self):
        self.git("config", "diff.external", "a-command-that-must-not-run")
        self.assertEqual(1, len(self.export()["files"]))

    def test_absence_cannot_hide_existing_source(self):
        for path in ["consumer.py", "../outside"]:
            with self.subTest(path=path), self.assertRaises(SnapshotError):
                self.export(absent=[path])

    def test_mutable_ref_is_rejected(self):
        with self.assertRaises(SnapshotError):
            exporter.export(self.repo, "example/project", 1, self.base, "HEAD")

    def test_binary_changed_file_refuses_comparison(self):
        (self.repo/"binary.dat").write_bytes(b"\xfe")
        self.git("commit", "-qam", "Binary fixture")
        self.head = self.git("rev-parse", "HEAD").strip()
        with self.assertRaises(SnapshotError):
            self.export()

    def test_output_is_never_overwritten(self):
        output = self.repo/"snapshot.json"
        output.write_text("preserved")
        rc = exporter.main(["--repo", str(self.repo), "--repository", "example/project",
                            "--number", "1", "--base", self.base, "--head", self.head,
                            "--output", str(output)])
        self.assertEqual(2, rc)
        self.assertEqual("preserved", output.read_text())


if __name__ == "__main__":
    unittest.main()
