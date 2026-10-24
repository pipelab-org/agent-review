# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
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

    def test_no_repository_program_or_hook_runs_during_export(self):
        marker = self.repo.parent / (self.repo.name + "-ran")
        run = f"touch {marker}"
        self.git("config", "core.fsmonitor", f"{run}; echo")
        self.git("config", "diff.evil.textconv", f"sh -c '{run}; cat' --")
        self.git("config", "diff.evil.command", f"{run}; true")
        (self.repo/".gitattributes").write_text("*.py diff=evil\n")
        hook = self.repo/".git"/"hooks"/"post-checkout"
        hook.write_text(f"#!/bin/sh\n{run}\n")
        hook.chmod(0o755)
        with patch.dict(os.environ, {"GIT_EXTERNAL_DIFF": run}):
            self.assertEqual(1, len(self.export()["files"]))
        self.assertFalse(marker.exists())

    def test_ambient_git_settings_do_not_change_the_patch(self):
        (self.repo/"code.py").write_text("".join(f"line {i}\n" for i in range(12)))
        self.git("commit", "-qam", "Longer fixture")
        self.base = self.git("rev-parse", "HEAD").strip()
        (self.repo/"code.py").write_text("".join(f"line {i}\n" if i != 6 else "\nchanged\n" for i in range(12)))
        self.git("commit", "-qam", "Edit the middle")
        self.head = self.git("rev-parse", "HEAD").strip()
        expected = self.export()["files"][0]["patch"]
        self.assertIn(" line 3\n", expected)
        for key, value in (("color.ui", "always"), ("diff.context", "0"), ("diff.suppressBlankEmpty", "true"),
                           ("diff.algorithm", "patience"), ("diff.interHunkContext", "9"),
                           ("diff.indentHeuristic", "false")):
            self.git("config", key, value)
        self.assertEqual(expected, self.export()["files"][0]["patch"])
        self.assertNotIn("\x1b", expected)
        # An inherited repository selector never redirects the export to another repository.
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        subprocess.check_call(["git", "init", "-q", other.name])
        with patch.dict(os.environ, {"GIT_DIR": str(Path(other.name) / ".git")}):
            self.assertEqual(expected, self.export()["files"][0]["patch"])

    def test_every_repository_diff_setting_that_changes_output_is_pinned(self):
        text = lambda lines: "".join(line + "\n" for line in lines)  # noqa: E731
        before = {
            # Inputs found by search where each setting alone changes git's own output.
            "algorithm.txt": ["a", "if (a) {", "a", "{", "  y;", "{", "", "b", "b", "{", "", "  x;"],
            "indent.txt": ["a", "if (a) {", "if (a) {", "{", "{", "{", "{", "  x;", "{", "a", "{"],
            "blank.txt": ["p", "", "q", "r", "s", "t"],
            "hunks.txt": [f"l{i}" for i in range(20)],
        }
        after = {
            "algorithm.txt": ["a", "if (a) {", "}", "if (a) {", "a", "{", "  y;", "{", "c", "", "{", "",
                              "  x;", "  x;", "b", "b", "{", "", "  x;"],
            "indent.txt": ["a", "if (a) {", "  x;", "", "if (a) {", "", "}", "if (a) {", "{", "{", "{", "{", "{",
                           "  x;", "{", "a", "{"],
            "blank.txt": ["p", "", "q", "R", "s", "t"],
            "hunks.txt": [f"l{i}" if i not in (2, 10) else "changed" for i in range(20)],
        }
        settings = {"algorithm.txt": ("diff.algorithm", "patience"), "indent.txt": ("diff.indentHeuristic", "false"),
                    "blank.txt": ("diff.suppressBlankEmpty", "true"), "hunks.txt": ("diff.interHunkContext", "9")}
        for name, lines in before.items():
            (self.repo/name).write_text(text(lines))
        self.git("add", ".")
        self.git("commit", "-qm", "Diff fixtures")
        self.base = self.git("rev-parse", "HEAD").strip()
        for name, lines in after.items():
            (self.repo/name).write_text(text(lines))
        self.git("commit", "-qam", "Edit the diff fixtures")
        self.head = self.git("rev-parse", "HEAD").strip()
        expected = {f["filename"]: f["patch"] for f in self.export()["files"]}
        for name, (key, value) in settings.items():
            with self.subTest(setting=key):
                self.git("config", key, value)
                # The setting really does change plain git's output for this input ...
                plain = self.git("diff", "--no-color", self.base, self.head, "--", name)
                self.assertNotEqual(expected[name], plain[plain.index("@@"):])
                # ... and cannot change the snapshot.
                self.assertEqual(expected[name], {f["filename"]: f["patch"] for f in self.export()["files"]}[name])
                self.git("config", "--unset", key)

    def test_a_file_name_containing_hunk_markers_cannot_start_the_patch_inside_its_header(self):
        (self.repo/"odd@@name.py").write_text("one\n")
        self.git("add", "odd@@name.py")
        self.git("commit", "-qm", "Add a marker-like name")
        self.head = self.git("rev-parse", "HEAD").strip()
        patch = {f["filename"]: f["patch"] for f in self.export()["files"]}["odd@@name.py"]
        self.assertEqual("@@ -0,0 +1 @@\n+one\n", patch)

    @unittest.skipIf(os.name == "nt", "a backslash cannot appear in a Windows file name")
    def test_unrequestable_unchanged_paths_are_skipped_but_changed_ones_refuse_export(self):
        (self.repo/"back\\slash.txt").write_text("legal on Linux, never a valid source request\n")
        self.git("add", ".")
        self.git("commit", "-qm", "Add a path model requests cannot name")
        self.base = self.git("rev-parse", "HEAD").strip()
        (self.repo/"code.py").write_text("value = 3\n")
        self.git("commit", "-qam", "Change something else")
        self.head = self.git("rev-parse", "HEAD").strip()
        data = self.export()
        self.assertEqual(["code.py"], [f["filename"] for f in data["files"]])
        self.assertNotIn("back\\slash.txt", data["sources"])
        with self.assertRaises(SnapshotError):
            self.export(absent=["back\\slash.txt"])
        (self.repo/"back\\slash.txt").write_text("changed\n")
        self.git("commit", "-qam", "Change the unrequestable path")
        self.head = self.git("rev-parse", "HEAD").strip()
        with self.assertRaises(SnapshotError):
            self.export()

    def test_repository_attributes_cannot_change_captured_hunks(self):
        before = "TAG marker\n" + "".join(f"line {n}\n" for n in range(15))
        (self.repo/"code.py").write_text(before)
        self.git("commit", "-qam", "Long fixture")
        self.base = self.git("rev-parse", "HEAD").strip()
        (self.repo/"code.py").write_text(before.replace("line 10", "CHANGED"))
        self.git("commit", "-qam", "Edit")
        self.head = self.git("rev-parse", "HEAD").strip()
        expected = self.export()["files"][0]["patch"]
        (self.repo/".git"/"info").mkdir(exist_ok=True)
        attributes = self.repo/".git"/"info"/"attributes"
        attributes.write_text("*.py diff=custom\n")
        self.git("config", "diff.custom.xfuncname", "^TAG")
        plain = self.git("diff", self.base, self.head, "--", "code.py")
        self.assertNotEqual(expected, plain[plain.index("@@"):])
        self.assertEqual(expected, self.export()["files"][0]["patch"])
        # A local binary attribute must not suppress a UTF-8 source hunk either.
        attributes.write_text("*.py -diff\n")
        self.assertEqual(expected, self.export()["files"][0]["patch"])

    def test_partial_clone_never_fetches_missing_objects(self):
        self.git("config", "uploadpack.allowFilter", "true")
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / "partial"
            subprocess.check_call(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                                   self.repo.as_uri(), str(clone)])
            def git(*args):
                return subprocess.check_output(["git", "-C", str(clone), *args],
                                               stderr=subprocess.DEVNULL).decode()
            missing = git("rev-list", "--objects", "--all", "--missing=print")
            self.assertIn("?", missing)  # real promisor objects, not a mocked subprocess
            git("config", "protocol.file.allow", "always")
            with patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file", "GIT_NO_LAZY_FETCH": "0"}):
                with self.assertRaises(SnapshotError):
                    exporter.export(clone, "example/project", 1, self.base, self.head)
            self.assertEqual(missing, git("rev-list", "--objects", "--all", "--missing=print"))
            # The origin works: ordinary Git lazily retrieves a missing blob.
            self.assertEqual("value = 2\n", git("show", self.head + ":code.py"))
            self.assertNotEqual(missing, git("rev-list", "--objects", "--all", "--missing=print"))

    def test_paths_are_literal_even_if_they_are_git_pathspecs(self):
        for name in ("*.py", ":(glob)*", "ordinary.py"):
            (self.repo/name).write_text("unique " + name + "\n")
        self.git("add", ".")
        self.git("commit", "-qm", "Add pathspec-shaped names")
        self.head = self.git("rev-parse", "HEAD").strip()
        data = self.export()
        for name in ("*.py", ":(glob)*", "ordinary.py"):
            with self.subTest(path=name):
                item = next(f for f in data["files"] if f["filename"] == name)
                self.assertEqual("@@ -0,0 +1 @@\n+unique " + name + "\n", item["patch"])

    def test_absence_cannot_hide_existing_source(self):
        for path in ["consumer.py", "../outside"]:
            with self.subTest(path=path), self.assertRaises(SnapshotError):
                self.export(absent=[path])

    def test_mutable_ref_is_rejected(self):
        with self.assertRaises(SnapshotError):
            exporter.export(self.repo, "example/project", 1, self.base, "HEAD")

    def test_ignored_submodule_changes_cannot_disappear_from_the_snapshot(self):
        self.git("update-index", "--add", "--cacheinfo", "160000", self.base, "dependency")
        self.git("commit", "-qm", "Add gitlink")
        self.head = self.git("rev-parse", "HEAD").strip()
        self.git("config", "diff.ignoreSubmodules", "all")
        with self.assertRaises(SnapshotError):
            self.export()

    def test_binary_changed_file_refuses_comparison(self):
        (self.repo/"binary.dat").write_bytes(b"\xfe")
        self.git("commit", "-qam", "Binary fixture")
        self.head = self.git("rev-parse", "HEAD").strip()
        with self.assertRaises(SnapshotError):
            self.export()

    def test_utf8_decodable_binary_source_is_still_refused(self):
        (self.repo/"code.py").write_bytes(b"binary\0source\n")
        self.git("commit", "-qam", "Nul-containing fixture")
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
