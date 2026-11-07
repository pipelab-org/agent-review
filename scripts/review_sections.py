# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact patch store and deterministic bounded sections for arbitrarily large reviews.

Every byte of every file's patch is streamed once into a trusted on-disk store and
assigned to exactly one segment of exactly one section; reassembling a file's segments
in order reproduces its patch byte for byte. A section is packaging, never a new
semantic hunk. Capacity comes only from the caller's exact prompt framing; this module
holds no file-count, hunk-size, line-size or change-size threshold.

Sources: Git's unified diff format (https://git-scm.com/docs/diff-format) and
`git diff --name-only -z`; hunk headers are `@@ -a[,b] +c[,d] @@`, and every hunk body
line starts with a space, `+`, `-` or a backslash, so a line beginning with `@@` is
always a hunk header. Lines are split only at b"\\n" (a CR stays inside its line), never
with str.splitlines, which also splits at form feeds and Unicode line separators.
UTF-8 continuation bytes are 0b10xxxxxx (RFC 3629 section 3), the only fragment-cut rule.
Source blocks follow CommonMark 0.31.2 section 4.5 fenced code blocks.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

try:
    from .review_errors import InputLimitAbort, ReviewAbort
except ImportError:
    from review_errors import InputLimitAbort, ReviewAbort

STORE = "agent-review"
# Buffer for streaming reads and writes; not a policy value (matches the existing I/O buffer).
IO_CHUNK = 65536
# A signed 64-bit offset has at most 19 decimal digits; framing is sized with this width
# so the section count, which is unknown while packing, never makes a prompt larger.
DIGITS = len(str(2 ** 63))
HUNK = re.compile(rb"@@ -([0-9]+)(?:,[0-9]+)? \+([0-9]+)(?:,[0-9]+)? @@")
BACKTICKS = re.compile(rb"`+")


def git_command(repo, *args):
    return ["git", "--literal-pathspecs", "-C", str(repo), "-c", "core.hooksPath=/dev/null",
            "-c", "core.fsmonitor=false", *args]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Store:
    """Trusted temporary copy of the exact diff bytes, below the checkout's git directory.

    The git directory is never part of the work tree, so a pull request cannot collide
    with it, and the restricted file tools can read it because it lies inside the
    checkout directory they are confined to.
    """

    def __init__(self, directory):
        self.directory = Path(directory)

    @classmethod
    def for_checkout(cls, repo):
        git_dir = Path(repo).resolve() / ".git"
        if not git_dir.is_dir():
            raise ReviewAbort("Review checkout needs its own .git directory for the section store")
        return cls(git_dir / STORE)

    def reset(self):
        shutil.rmtree(self.directory, ignore_errors=True)
        for name in ("patches", "content", "ledger"):
            (self.directory / name).mkdir(parents=True)
        # The model account may differ from the controller; the data is the public patch.
        for path in (self.directory, *self.directory.iterdir()):
            path.chmod(0o755)

    def patch(self, index):
        return self.directory / "patches" / f"{index}.diff"

    def content(self, index):
        return self.directory / "content" / f"{index}.txt"

    def index(self):
        return self.directory / "index.tsv"

    def ledger(self, strategy):
        return self.directory / "ledger" / f"{strategy}.txt"

    def relative(self, path):
        return f".git/{STORE}/{Path(path).relative_to(self.directory).as_posix()}"


def stream_to_file(command, target, *, message, publish=True):
    """Copy a Git command's stdout to `target`, validating UTF-8 and hashing as it goes.

    Nothing is materialized beyond one buffer. Returns size, digest, whether a NUL byte
    appeared, the longest backtick run and the final byte (exact fence and newline
    accounting needs them). `publish=False` validates without keeping the bytes.
    """
    digest, size, nul, last, run, carry = hashlib.sha256(), 0, False, b"", 0, 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    try:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
            with open(target, "wb") as out:
                while chunk := process.stdout.read(IO_CHUNK):
                    decoder.decode(chunk)
                    nul = nul or b"\x00" in chunk
                    digest.update(chunk)
                    out.write(chunk)
                    size += len(chunk)
                    last = chunk[-1:]
                    tail = 0
                    for match in BACKTICKS.finditer(chunk):
                        length = match.end() - match.start() + (carry if match.start() == 0 else 0)
                        run = max(run, length)
                        tail = length if match.end() == len(chunk) else 0
                    carry = tail
            decoder.decode(b"", final=True)
            code = process.wait()
    except UnicodeDecodeError:
        raise ReviewAbort(message) from None
    if code:
        raise ReviewAbort("Immutable Git source unavailable; check admitted refs and fetch depth")
    if publish:
        os.chmod(target, 0o644)
    return {"bytes": size, "sha256": digest.hexdigest(), "nul": nul, "run": run, "last": last}


def run_git_text(repo, *args):
    """Small bounded Git output (ls-tree, rev-parse); never a diff or blob."""
    result = subprocess.run(git_command(repo, *args), capture_output=True, check=False)
    if result.returncode:
        raise ReviewAbort("Immutable Git source unavailable; check admitted refs and fetch depth")
    try:
        return result.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ReviewAbort("Unsafe Git path; review cannot cover this change") from None


def scan_lines(path, start=0, end=None):
    """Yield (start, end, head, run) per physical line without holding a line in memory.

    `head` is the first two bytes; `run` the longest backtick run in the line. Lines end
    only at b"\\n". `end`, when given, is a line boundary.
    """
    with open(path, "rb") as handle:
        handle.seek(start)
        position = line_start = start
        head, run, carry = b"", 0, 0
        while end is None or position < end:
            piece = handle.readline(IO_CHUNK if end is None else min(IO_CHUNK, end - position))
            if not piece:
                break
            if position == line_start:
                head, run, carry = piece[:2], 0, 0
            last = 0
            for match in BACKTICKS.finditer(piece):
                length = match.end() - match.start() + (carry if match.start() == 0 else 0)
                run = max(run, length)
                last = length if match.end() == len(piece) else 0
            carry = last
            position += len(piece)
            if piece.endswith(b"\n"):
                yield line_start, position, head, run
                line_start, head, run, carry = position, b"", 0, 0
        if position > line_start:
            yield line_start, position, head, run


def units(path):
    """Yield the patch header and each hunk as dicts with exact byte ranges."""
    current = None
    for start, end, head, run in scan_lines(path):
        if head.startswith(b"@@"):
            if current:
                yield current
            match = HUNK.match(read_range(path, start, min(end, start + IO_CHUNK)))
            current = {"start": start, "end": end, "run": run,
                       "hunk": match.group(0).decode("ascii") if match else "@@",
                       "old": int(match.group(1)) if match else 0,
                       "new": int(match.group(2)) if match else 0}
        elif current is None:
            current = {"start": start, "end": end, "run": run, "hunk": None, "old": None, "new": None}
        else:
            current["end"] = end
            current["run"] = max(current["run"], run)
    if current:
        yield current


def read_range(path, start, end):
    with open(path, "rb") as handle:
        handle.seek(start)
        data = handle.read(end - start)
    if len(data) != end - start:
        raise ReviewAbort("Review store is shorter than the plan")
    return data


def run_of(data):
    return max((len(m.group(0)) for m in BACKTICKS.finditer(data)), default=0)


def meta_line(path, deleted):
    return "File metadata: " + json.dumps({"path": path, "deleted": deleted}, ensure_ascii=False) + "\n"


def segment_label(path, seg, total):
    if seg["start"] == 0 and seg["end"] == total:
        return "Complete unified diff"
    text = f"Diff segment of {json.dumps(path, ensure_ascii=False)}: patch bytes {seg['start']}-{seg['end']} of {total}"
    where = ("the patch header" if seg.get("header") else
             f"hunk {seg['hunk']} (old line {seg['old']}, new line {seg['new']})" if seg.get("hunk") else None)
    if seg.get("line_start") is not None:
        return text + f"; carries a long line of {where or 'the patch'} that begins at patch byte {seg['line_start']}"
    if where:
        return text + f"; continues {where}"
    return text + ("; starts at the patch header" if seg["start"] == 0 else "; starts at a hunk boundary")


def block_frame(label, language, n, run, ends_newline):
    """The exact text around a source block of `n` UTF-8 bytes, as source_block renders it."""
    fence = "`" * max(3, 1 + run)
    return f"{label} ({n} UTF-8 bytes):\n{fence}{language}\n", ("" if ends_newline else "\n") + fence + "\n"


def source_block(label, language, text):
    """Render exact source inside a fence longer than any source backtick run."""
    # CommonMark 0.31.2 section 4.5: closing fences need at least as many backticks as
    # the opening fence, so source text must not close its own block.
    run = max((len(m[0]) for m in re.finditer(r"`+", text)), default=0)
    opening, closing = block_frame(label, language, len(text.encode("utf-8")), run, text.endswith("\n"))
    return opening + text + closing


def block_size(label, language, n, run, ends_newline):
    opening, closing = block_frame(label, language, n, run, ends_newline)
    return len(opening.encode("utf-8")) + n + len(closing.encode("utf-8"))


class MoreSections(Exception):
    """Raised by a packer limited to `limit` sections when another would be needed."""


class Packer:
    """Greedy, order-preserving packer; the exact rendered size decides every boundary.

    `sections` holds closed sections; each is a list of ranges (one per file, contiguous
    bytes of that file's patch). `open` is the range still allowed to grow.
    """

    def __init__(self, files, capacity, limit=None):
        self.files, self.capacity, self.limit = files, capacity, limit
        self.sections, self.used, self.open = [[]], 0, None

    def size(self, rng):
        file = self.files[rng["file"]]
        label = segment_label(file["path"], rng["seg"], file["patch"]["bytes"])
        return (len(meta_line(file["path"], file["deleted"]).encode("utf-8"))
                + block_size(label, "diff", rng["end"] - rng["start"], rng["run"], rng["last"] == b"\n"))

    def make(self, file, start, end, run, last, coords):
        return {"file": file, "start": start, "end": end, "run": run, "last": last,
                "seg": {"start": start, "end": end, **coords}}

    def close(self):
        rng, self.open = self.open, None
        if rng is not None:
            self.sections[-1].append(rng)
            self.used += self.size(rng)

    def advance(self):
        self.close()
        if self.limit is not None and len(self.sections) >= self.limit:
            raise MoreSections
        self.sections.append([])
        self.used = 0

    def place(self, file, start, end, run, coords):
        """Append a whole piece at `start`; False if it cannot fit even an empty section."""
        last = read_range(self.files[file]["store"], end - 1, end)
        fresh = self.make(file, start, end, run, last, coords)
        if self.open and self.open["file"] == file and self.open["end"] == start:
            merged = dict(self.open, end=end, run=max(self.open["run"], run), last=last,
                          seg=dict(self.open["seg"], end=end))
            if self.used + self.size(merged) <= self.capacity:
                self.open = merged
                return True
            self.advance()
        else:
            self.close()
        if self.used + self.size(fresh) <= self.capacity:
            self.open = fresh
            return True
        if self.sections[-1]:
            self.advance()
            if self.size(fresh) <= self.capacity:
                self.open = fresh
                return True
        return False

    def add_unit(self, file, unit):
        if not self.place(file, unit["start"], unit["end"], unit["run"], {}):
            self.split(file, unit)

    def split(self, file, unit):
        """Cut a unit larger than an empty section into whole lines, then fragments."""
        header = unit["hunk"] is None
        old, new = unit["old"], unit["new"]
        for start, end, head, run in scan_lines(self.files[file]["store"], unit["start"], unit["end"]):
            if start == unit["start"]:
                coords = {}
            elif header:
                coords = {"header": True}
            else:
                coords = {"hunk": unit["hunk"], "old": old, "new": new}
            if not self.place(file, start, end, run, coords):
                self.fragments(file, start, end, {"header": True} if header else
                               {"hunk": unit["hunk"], "old": old, "new": new})
            tag = head[:1]
            if not header and not head.startswith(b"@@"):
                old += tag in (b"-", b" ")
                new += tag in (b"+", b" ")

    def fragments(self, file, line_start, line_end, where):
        """Cut one line larger than an empty section at UTF-8 character boundaries."""
        position = line_start
        while position < line_end:
            coords = {**where, "line_start": line_start}
            cut, data = self.longest(file, position, line_end, coords)
            self.close()
            self.open = self.make(file, position, cut, run_of(data), data[-1:], coords)
            position = cut
            if position < line_end:
                self.advance()

    def longest(self, file, position, end, coords):
        """Largest character-aligned cut whose block fits an empty section (block size is monotone)."""
        path = self.files[file]["store"]
        low, high = position, min(end, position + self.capacity)
        best = (position, b"")
        while low < high:
            middle = (low + high + 1) // 2
            cut = middle
            while cut > position and cut < end and read_range(path, cut, cut + 1)[0] & 0xC0 == 0x80:
                cut -= 1
            data = read_range(path, position, cut) if cut > position else b""
            if data and self.size(self.make(file, position, cut, run_of(data), data[-1:], coords)) <= self.capacity:
                best, low = (cut, data), middle
            else:
                high = middle - 1
        if best[0] <= position:
            raise InputLimitAbort("A section cannot hold even one character of a long line within "
                                  "the review depth's max_model_tokens")
        return best

    def finish(self):
        self.close()
        return [section for section in self.sections if section] or [[]]


def plan_sections(files, capacity, limit=None):
    """Pack every file's patch, in order, into sections of at most `capacity` rendered bytes.

    `files` items carry path, deleted, patch {bytes, sha256} and the store file. Returns
    sections as lists of segment dicts (file index, byte range, optional coordinates,
    sha256). Raises MoreSections if `limit` sections would be exceeded.
    """
    packer = Packer(files, capacity, limit)
    for index, file in enumerate(files):
        if file["patch"]["bytes"] == 0:
            continue
        for unit in units(file["store"]):
            packer.add_unit(index, unit)
    result = []
    for ranges in packer.finish():
        segments = []
        for rng in ranges:
            file = files[rng["file"]]
            segments.append({"file": rng["file"], **rng["seg"],
                             "sha256": hashlib.sha256(read_range(file["store"], rng["start"], rng["end"])).hexdigest()})
        result.append(segments)
    return result


def section_entry(index, segments):
    return {"id": index, "segments": segments, "sha256": sha256_text(canonical(segments))}


def reconstruct(bundle, store):
    """Reassemble every file's patch from its segments and compare with the recorded digest.

    Raises ReviewAbort on any gap, overlap, reorder or altered byte. Returns the patch
    bytes per file index, so tests can compare them with a separate Git enumeration.
    """
    parts = {}
    for section in bundle["sections"]:
        for seg in section["segments"]:
            parts.setdefault(seg["file"], []).append(seg)
    patches = {}
    for index, file in enumerate(bundle["files"]):
        chunks, position = [], 0
        for seg in sorted(parts.get(index, []), key=lambda s: s["start"]):
            data = read_range(store.patch(index), seg["start"], seg["end"])
            if (seg["start"] != position or seg["end"] <= seg["start"]
                    or hashlib.sha256(data).hexdigest() != seg["sha256"]):
                raise ReviewAbort("Section segments do not reproduce the recorded patch")
            chunks.append(data)
            position = seg["end"]
        whole = b"".join(chunks)
        if (position != file["patch"]["bytes"]
                or hashlib.sha256(whole).hexdigest() != file["patch"]["sha256"]):
            raise ReviewAbort("Section segments do not reproduce the recorded patch")
        patches[index] = whole
    return patches


CONTENT_LABEL = "Full head content"


def content_cost(data):
    """Rendered size of one full-content block, from the streamed facts about that blob."""
    return block_size(CONTENT_LABEL, "text", data["bytes"], data["run"], data["last"] == b"\n")


def render_section(bundle, store, section):
    """The file metadata lines and fenced blocks of one section, with integrity checks.

    A file whose whole patch is in this section may also carry its full head content,
    which only a single-section plan ever does (the existing small-review prompt).
    """
    parts = []
    for seg in bundle["sections"][section]["segments"]:
        file = bundle["files"][seg["file"]]
        data = read_range(store.patch(seg["file"]), seg["start"], seg["end"])
        if hashlib.sha256(data).hexdigest() != seg["sha256"]:
            raise ReviewAbort("Review store differs from the plan; no inference requested")
        parts.append(meta_line(file["path"], file["deleted"]))
        parts.append(source_block(segment_label(file["path"], seg, file["patch"]["bytes"]), "diff",
                                  data.decode("utf-8")))
        if file.get("content") == "full" and seg["end"] == file["patch"]["bytes"]:
            raw = store.content(seg["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != file.get("content_sha256") or len(raw) != file.get("content_bytes"):
                raise ReviewAbort("Review store differs from the plan; no inference requested")
            parts.append(source_block(CONTENT_LABEL, "text", raw.decode("utf-8")))
    return "".join(parts)


def write_index(bundle, store):
    """One line per changed file, readable with the restricted file tools, never in a prompt."""
    membership = {}
    for section in bundle["sections"]:
        for seg in section["segments"]:
            membership.setdefault(seg["file"], []).append(section["id"])
    lines = ["file\tpath\tstate\tpatch_bytes\tsections\n"]
    for index, file in enumerate(bundle["files"]):
        sections = ",".join(str(s) for s in sorted(set(membership.get(index, []))))
        lines.append(f"{index}\t{json.dumps(file['path'], ensure_ascii=False)}\t"
                     f"{'deleted' if file['deleted'] else 'present'}\t{file['patch']['bytes']}\t{sections}\n")
    text = "".join(lines)
    store.index().write_text(text, encoding="utf-8")
    store.index().chmod(0o644)
    return text
