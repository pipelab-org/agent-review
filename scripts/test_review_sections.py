# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact, bounded sections: every patch byte lands in exactly one segment of one section."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import review_engine as e, review_sections as s, review_jobs as jobs
from scripts.test_review_engine import POLICY

FILE_HEADER = b"diff --git a/f.py b/f.py\nindex 1..2 100644\n--- a/f.py\n+++ b/f.py\n"


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def entry(self, index, data, path="f.py", deleted=False):
        target = self.root / f"{index}.diff"
        target.write_bytes(data)
        return {"path": path, "deleted": deleted, "store": target,
                "patch": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}

    def bundle(self, files, sections):
        listing = [{k: v for k, v in f.items() if k != "store"} for f in files]
        return {"files": listing, "sections": [s.section_entry(i, segs) for i, segs in enumerate(sections)]}

    def store(self, files):
        class Store:
            def patch(self, index):
                return files[index]["store"]
        return Store()

    def check(self, files, capacity):
        planned = s.plan_sections(files, capacity)
        bundle, store = self.bundle(files, planned), self.store(files)
        patches = s.reconstruct(bundle, store)
        for index, file in enumerate(files):
            self.assertEqual(file["store"].read_bytes(), patches[index])
        for section in range(len(planned)):
            self.assertLessEqual(len(s.render_section(bundle, store, section).encode()), capacity)
        return planned, bundle, store


class PackingTests(Fixture):
    def test_whole_hunks_stay_together_and_small_changes_are_one_section(self):
        hunks = b"".join(b"@@ -%d,2 +%d,2 @@\n ctx\n-old %d\n+new %d\n" % (i * 10 + 1, i * 10 + 1, i, i) for i in range(50))
        files = [self.entry(0, FILE_HEADER + hunks)]
        planned, bundle, _ = self.check(files, 100_000)
        self.assertEqual(1, len(planned))
        planned, bundle, store = self.check(files, 700)
        self.assertGreater(len(planned), 3)
        for segments in planned:
            for seg in segments:
                data = files[0]["store"].read_bytes()[seg["start"]:seg["end"]]
                # Every cut after the first segment is at a hunk boundary: no hunk is split while one fits.
                if seg["start"]:
                    self.assertTrue(data.startswith(b"@@"), data[:40])

    def test_many_files_keep_git_order_and_cross_section_boundaries(self):
        files = [self.entry(i, b"diff --git a/f%d b/f%d\n@@ -1 +1 @@\n-a\n+b\n" % (i, i), f"f{i}.py") for i in range(80)]
        planned, bundle, _ = self.check(files, 1200)
        order = [seg["file"] for segments in planned for seg in segments]
        self.assertEqual(sorted(order), order)
        self.assertEqual(set(range(80)), set(order))

    def test_one_huge_hunk_is_split_with_exact_old_and_new_coordinates(self):
        body = b"".join((b"+added %d\n" if i % 3 else b"-removed %d\n" if i % 3 == 0 and i % 2 else b" kept %d\n") % i
                        for i in range(1, 2000))
        data = b"diff --git a/h b/h\n--- a/h\n+++ b/h\n@@ -10,700 +20,1400 @@\n" + body
        files = [self.entry(0, data, "h")]
        planned, bundle, _ = self.check(files, 2500)
        self.assertGreater(len(planned), 10)
        for segments in planned:
            for seg in segments:
                if seg["start"] == 0 or seg["start"] == data.index(b"@@"):
                    continue
                self.assertEqual("@@ -10,700 +20,1400 @@", seg["hunk"])
                before = data[data.index(b"@@"):seg["start"]].split(b"\n")[1:-1]
                old = 10 + sum(1 for line in before if line[:1] in (b"-", b" "))
                new = 20 + sum(1 for line in before if line[:1] in (b"+", b" "))
                self.assertEqual((old, new), (seg["old"], seg["new"]))

    def test_one_huge_line_is_split_at_character_boundaries_and_reassembles(self):
        line = ("雪" * 4000 + "`" * 40 + "é" * 3000 + "\U0001f600" * 500).encode()
        data = b"diff --git a/c b/c\n--- a/c\n+++ b/c\n@@ -1 +1 @@\n-x\n+" + line
        files = [self.entry(0, data, "c")]
        planned, bundle, store = self.check(files, 700)
        self.assertGreater(len(planned), 20)
        for segments in planned:
            for seg in segments:
                files[0]["store"].read_bytes()[seg["start"]:seg["end"]].decode("utf-8")
        starts = [seg["line_start"] for segments in planned for seg in segments if "line_start" in seg]
        self.assertTrue(starts and len(set(starts)) == 1)

    def test_crlf_form_feed_unicode_separators_deletion_and_missing_final_newline(self):
        data = (b"diff --git a/d b/d\n--- a/d\n+++ b/d\n@@ -1,3 +1,2 @@\n-a\r\n\x0c+b\xe2\x80\xa8x\r\n"
                b" tail\n\\ No newline at end of file")
        files = [self.entry(0, data, "d"), self.entry(1, b"diff --git a/gone b/gone\ndeleted file mode 100644\n"
                                                          b"@@ -1 +0,0 @@\n-gone", "gone", True)]
        planned, bundle, store = self.check(files, 450)
        self.assertEqual(data, s.reconstruct(bundle, store)[0])
        text = "".join(s.render_section(bundle, store, i) for i in range(len(planned)))
        self.assertIn('"deleted": true', text)
        self.assertTrue(text.rstrip().endswith("```"))

    def test_rendered_size_is_exactly_the_packers_prediction(self):
        for text in ("plain\n", "no newline", "a ``` b\n````\n", "雪\n" * 7):
            rendered = s.source_block("Label", "diff", text)
            run = max((len(m[0]) for m in __import__("re").finditer(r"`+", text)), default=0)
            self.assertEqual(len(rendered.encode()),
                             s.block_size("Label", "diff", len(text.encode()), run, text.endswith("\n")))

    def test_capacity_too_small_for_one_character_refuses_instead_of_dropping(self):
        data = FILE_HEADER + b"@@ -1 +1 @@\n+" + ("雪" * 100).encode() + b"\n"
        files = [self.entry(0, data)]
        with self.assertRaises(e.InputLimitAbort):
            s.plan_sections(files, 120)

    def test_a_single_section_probe_stops_as_soon_as_a_second_is_needed(self):
        files = [self.entry(0, FILE_HEADER + b"@@ -1 +1 @@\n+" + b"x" * 5000 + b"\n")]
        with self.assertRaises(s.MoreSections):
            s.plan_sections(files, 1000, limit=1)
        self.assertEqual(1, len(s.plan_sections(files, 100_000, limit=1)))


class IntegrityTests(Fixture):
    def prepared(self):
        data = FILE_HEADER + b"".join(b"@@ -%d,1 +%d,1 @@\n-a%d\n+b%d\n" % (i * 4 + 1, i * 4 + 1, i, i) for i in range(30))
        files = [self.entry(0, data)]
        planned, bundle, store = self.check(files, 800)
        return files, bundle, store

    def test_gaps_overlaps_reorders_and_altered_bytes_never_reconstruct(self):
        files, bundle, store = self.prepared()
        import copy
        cases = []
        gap = copy.deepcopy(bundle); gap["sections"][1]["segments"][0]["start"] += 1; cases.append(gap)
        short = copy.deepcopy(bundle); short["sections"][-1]["segments"][-1]["end"] -= 1; cases.append(short)
        dropped = copy.deepcopy(bundle); dropped["sections"][1]["segments"].pop(); cases.append(dropped)
        digest = copy.deepcopy(bundle); digest["files"][0]["patch"]["sha256"] = "0" * 64; cases.append(digest)
        lie = copy.deepcopy(bundle); lie["sections"][0]["segments"][0]["sha256"] = "0" * 64; cases.append(lie)
        for case in cases:
            with self.subTest(case=case["sections"][0]["id"]), self.assertRaises(e.ReviewAbort):
                s.reconstruct(case, store)
        files[0]["store"].write_bytes(files[0]["store"].read_bytes().replace(b"a3", b"zz"))
        with self.assertRaises(e.ReviewAbort):
            s.reconstruct(bundle, store)
        with self.assertRaises(e.ReviewAbort):
            s.render_section(bundle, store, 0)

    def test_section_identity_binds_its_segments(self):
        files, bundle, store = self.prepared()
        first = bundle["sections"][0]
        altered = s.section_entry(0, [dict(first["segments"][0], end=first["segments"][0]["end"] - 1)])
        self.assertNotEqual(first["sha256"], altered["sha256"])

    def test_non_utf8_patch_text_is_refused_not_decoded_loosely(self):
        target = self.root / "bad.diff"
        script = "import sys; sys.stdout.buffer.write(b'@@ -1 +1 @@\\n+\\xff\\xfe')"
        with self.assertRaisesRegex(e.ReviewAbort, "binary"):
            s.stream_to_file(["python3", "-c", script], target, message=e.BINARY_MESSAGE)
        script = "import sys; sys.exit(3)"
        with self.assertRaisesRegex(e.ReviewAbort, "Immutable Git source unavailable"):
            s.stream_to_file(["python3", "-c", script], target, message=e.BINARY_MESSAGE)


class RepositoryPlanTests(unittest.TestCase):
    """build_bundle on real Git history, compared with a separate Git enumeration."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        (self.root / "keep.txt").write_text("one\n")
        (self.root / "gone.txt").write_text("deleted\nfile\n")
        (self.root / "crlf.txt").write_bytes(b"a\r\nb\r\n")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True).strip()

    def head(self):
        self.git("add", ".")
        self.git("commit", "-qm", "head")
        return self.git("rev-parse", "HEAD")

    def check_exact(self, bundle):
        patches = s.reconstruct(bundle, bundle.store)
        for index, file in enumerate(bundle["files"]):
            expected = subprocess.check_output(
                ["git", "-C", str(self.root), "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                 "--unified=3", bundle["from"], bundle["head"], "--", file["path"]])
            self.assertEqual(expected, patches[index], file["path"])
        listed = e.git(self.root, "diff", "--name-only", "--no-renames", "-z", bundle["from"], bundle["head"])
        self.assertEqual(sorted(listed.split("\0")[:-1]), sorted(f["path"] for f in bundle["files"] + [
            {"path": p} for p in bundle["exclusions"]]))

    def test_mixed_change_is_planned_in_sections_every_prompt_fits_and_rebuilds_equal(self):
        (self.root / "keep.txt").write_text("".join(f"line {i} `` 雪\n" for i in range(4000)))
        (self.root / "gone.txt").unlink()
        (self.root / "crlf.txt").write_bytes(b"a\r\nB\r\nno final newline")
        for i in range(30):
            (self.root / f"new{i}.py").write_text(f"value = {i}\n" * 120)
        (self.root / "huge.js").write_text("x = [" + ",".join(str(i) for i in range(30000)) + "]")
        head = self.head()
        small = dict(POLICY, max_model_tokens=40_000, max_output_tokens=16_000)
        bundle = e.build_bundle(self.root, head, self.base, small, instructions="Review correctness.")
        self.assertGreater(len(bundle["sections"]), 3)
        self.check_exact(bundle)
        available = small["max_model_tokens"] - small["max_output_tokens"]
        for section in range(len(bundle["sections"])):
            for provider, strategy in e.PROMPT_KINDS:
                prompt = e.section_prompt(bundle, provider, strategy, section)
                self.assertLessEqual(e.token_bound(prompt) + e.token_bound(json.dumps(e.SECTION_SCHEMA)), available)
        self.assertEqual(bundle, e.build_bundle(self.root, head, self.base, small, instructions="Review correctness."))
        self.assertTrue(any(f["deleted"] for f in bundle["files"]))
        # Section prompts point at the index and the full patches, which are readable from the checkout.
        self.assertIn(".git/agent-review/index.tsv", e.claude_prompt(bundle, "bundle"))
        self.assertTrue(bundle.store.index().read_text().startswith("file\tpath\tstate"))

    def test_changed_instructions_exclusions_and_learning_bytes_change_the_plan_identity(self):
        (self.root / "keep.txt").write_text("changed\n")
        head = self.head()
        first = e.build_bundle(self.root, head, self.base, POLICY, instructions="one")
        second = e.build_bundle(self.root, head, self.base, POLICY, instructions="two")
        self.assertNotEqual(first["root"], second["root"])
        self.assertEqual(first["root"], e.build_bundle(self.root, head, self.base, POLICY, instructions="one")["root"])

    def test_reserved_learning_bytes_shrink_every_section_exactly(self):
        (self.root / "keep.txt").write_text("".join(f"line {i}\n" for i in range(9000)))
        head = self.head()
        small = dict(POLICY, max_model_tokens=40_000, max_output_tokens=16_000)
        plain = e.build_bundle(self.root, head, self.base, small)
        with patch.object(e.learnings, "select", return_value=(
                {"sha": "a" * 40, "digest": "b" * 64, "count": 1, "reason": "", "bytes": 12_000}, "L" * 12_000)):
            rich = e.build_bundle(self.root, head, self.base, small, learning_data=e.learnings.empty())
        self.assertGreater(len(rich["sections"]), len(plain["sections"]))
        available = small["max_model_tokens"] - small["max_output_tokens"]
        for section in range(len(rich["sections"])):
            for provider, strategy in e.PROMPT_KINDS:
                total = (e.token_bound(e.section_prompt(rich, provider, strategy, section))
                         + len(rich["learnings_text"].encode()) + e.token_bound(json.dumps(e.SECTION_SCHEMA)))
                self.assertLessEqual(total, available)
        public = dict(rich)
        public.pop("learnings_text")
        self.assertEqual(public, e.build_bundle(self.root, head, self.base, small, learning_metadata=rich["learnings"]))

    def test_excluded_paths_stay_complete_without_filling_the_prompt(self):
        (self.root / "keep.txt").write_text("changed\n")
        ignored = self.root / "ignored"
        ignored.mkdir()
        for index in range(400):
            (ignored / (f"{index:04d}-" + "a" * 100)).write_text("ignored\n")
        head = self.head()
        small = dict(POLICY, max_model_tokens=40_000, max_output_tokens=16_000)
        bundle = e.build_bundle(self.root, head, self.base, small, exclusions=("ignored/*",))
        self.assertEqual(1, len(bundle["sections"]))
        self.assertEqual(["keep.txt"], [file["path"] for file in bundle["files"]])
        self.assertEqual(400, len(bundle["exclusions"]))
        self.check_exact(bundle)
        raw = (bundle.store.directory / "exclusions.json").read_text()
        self.assertEqual(bundle["exclusions"], json.loads(raw))
        for provider, strategy in e.PROMPT_KINDS:
            prompt = e.section_prompt(bundle, provider, strategy, 0)
            self.assertIn(s.sha256_text(raw), prompt)
            self.assertIn(".git/agent-review/exclusions.json", prompt)
            self.assertNotIn(bundle["exclusions"][-1], prompt)
            self.assertLessEqual(e.token_bound(prompt + json.dumps(e.OUTPUT_SCHEMA)),
                                 small["max_model_tokens"] - small["max_output_tokens"])
        self.assertEqual(bundle, e.build_bundle(self.root, head, self.base, small, exclusions=("ignored/*",)))

    def test_the_store_needs_a_real_git_directory_and_is_rebuilt_each_time(self):
        (self.root / "keep.txt").write_text("changed\n")
        head = self.head()
        bundle = e.build_bundle(self.root, head, self.base, POLICY)
        stale = bundle.store.directory / "patches" / "99.diff"
        stale.write_text("stale")
        e.build_bundle(self.root, head, self.base, POLICY)
        self.assertFalse(stale.exists())
        with tempfile.TemporaryDirectory() as plain:
            with self.assertRaisesRegex(e.ReviewAbort, "own .git directory"):
                s.Store.for_checkout(plain)


class VisibleLineTests(unittest.TestCase):
    def test_form_feed_unicode_separators_and_plus_plus_content_do_not_shift_anchors(self):
        patch_text = "@@ -1,5 +1,6 @@\n ctx\x0cmore\n+++i;\n+two\u2028still two\n ctx\r\n-removed\n+last\n"
        self.assertEqual({1, 2, 3, 4, 5}, jobs.anchor_lines(patch_text))
        # A separator followed by `+` or a space would look like extra added or context lines.
        self.assertEqual({1, 2}, jobs.anchor_lines("@@ -1,2 +1,2 @@\n+one\x0c+two\n ctx\n"))
        self.assertEqual({1, 2}, jobs.anchor_lines("@@ -1,2 +1,2 @@\n+one\u2028 two\n ctx\n"))
        header_first = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n+x\n"
        self.assertEqual({1}, jobs.anchor_lines(header_first))


if __name__ == "__main__":
    unittest.main()
