# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a code-only historical snapshot from local Git objects, without network access.

Git plumbing contracts: https://git-scm.com/docs/git-ls-tree,
https://git-scm.com/docs/git-cat-file and https://git-scm.com/docs/git-diff.
No PR discussion, current worktree files, or commit messages enter the snapshot.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

if __package__:
    from . import review_replay_snapshot as snapshots
else:
    import review_replay_snapshot as snapshots

# Variables that select a repository or change how a diff is produced. An inherited
# GIT_DIR would silently export a different repository than --repo.
AMBIENT = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
           "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE",
           "GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS", "GIT_PAGER", "GIT_REPLACE_REF_BASE",
           "GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS", "GIT_LITERAL_PATHSPECS")
# Settings that would otherwise be written into the snapshot (color escapes, blank-line
# context). The diff flags restate git-diff's documented defaults: unified=3,
# inter-hunk-context=0, the default algorithm and the indent heuristic.
GIT = ["git", "--no-replace-objects", "--literal-pathspecs", "-c", "color.ui=false", "-c", "core.fsmonitor=false",
       "-c", "diff.suppressBlankEmpty=false"]
DIFF = ["diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--no-color", "--unified=3",
        "--inter-hunk-context=0", "--indent-heuristic", "--diff-algorithm=default"]
HUNK = re.compile(r"(?m)^@@ -")


def export(repo, repository, number, base, head, absent=()):
    if not all(snapshots.followups.full_sha(s) for s in (base, head)):
        raise snapshots.SnapshotError("Export needs full base and head commit SHAs")
    env = {k: v for k, v in os.environ.items() if k not in AMBIENT and not k.startswith("GIT_CONFIG")}
    # git(1): GIT_ALLOW_PROTOCOL overrides protocol.*.allow, including remote helpers.
    # An empty list refuses every transport, even on Git predating GIT_NO_LAZY_FETCH.
    # Partial-clone missing objects must fail instead of fetching behind this reader.
    env.update(GIT_ALLOW_PROTOCOL="", GIT_NO_LAZY_FETCH="1")

    def git(*args):
        result = subprocess.run([*GIT, "-C", str(repo), *args], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise snapshots.SnapshotError("Required Git objects are unavailable")
        return result.stdout

    for sha in (base, head):
        if git("cat-file", "-t", sha).strip() != b"commit":
            raise snapshots.SnapshotError("Export identity must name a commit")

    # A path the model-request validator rejects can never be requested, so an unchanged
    # one is simply not capturable; one the change touches cannot be replayed.
    unlisted = set()

    def tree(sha):
        result = {}
        for entry in git("ls-tree", "-r", "-z", sha).split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, kind, oid = metadata.split()
            try:
                path = raw_path.decode("utf-8")
            except UnicodeDecodeError:
                path = None
            if path is None or not snapshots.repo_path(path):
                unlisted.add(raw_path)
                continue
            result[path] = (mode, kind, oid.decode("ascii"))
        return result

    old, new = tree(base), tree(head)
    changes = git("diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--name-status", "-z", base, head).split(b"\0")
    if changes[-1] == b"":
        changes.pop()
    if len(changes) % 2 or any(raw in unlisted for raw in changes[1::2]):
        raise snapshots.SnapshotError("Git tree contains an unsupported path")
    pairs = [(changes[i].decode("ascii"), changes[i+1].decode("utf-8")) for i in range(0, len(changes), 2)]
    needed = {v[2] for v in new.values() if v[0] in (b"100644", b"100755")}
    needed.update(old[p][2] for _, p in pairs if p in old and old[p][0] in (b"100644", b"100755"))
    # One batch avoids a process for every source file. Git's size-prefixed output
    # keeps arbitrary blob text from being interpreted as another response header.
    oids = sorted(needed)
    result = subprocess.run([*GIT, "-C", str(repo), "cat-file", "--batch"], env=env,
                            input="".join(oid+"\n" for oid in oids).encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise snapshots.SnapshotError("Git blob collection failed")
    raw, offset, blobs = result.stdout, 0, {}
    for oid in oids:
        end = raw.index(b"\n", offset)
        fields = raw[offset:end].split()
        if len(fields) != 3:
            raise snapshots.SnapshotError("Required Git blobs are unavailable")
        actual, kind, size = fields
        if actual.decode() != oid or kind != b"blob":
            raise snapshots.SnapshotError("Git blob identity mismatch")
        offset = end + 1
        count = int(size)
        content = raw[offset:offset+count]
        if len(content) != count or raw[offset+count:offset+count+1] != b"\n":
            raise snapshots.SnapshotError("Git blob response is incomplete")
        try:
            blobs[oid] = content.decode("utf-8")
        except UnicodeDecodeError:
            blobs[oid] = None
        offset += count + 1

    def content(tree, path):
        if path not in tree:
            return ""
        mode, kind, oid = tree[path]
        if mode not in (b"100644", b"100755") or blobs.get(oid) is None:
            raise snapshots.SnapshotError("Changed file is not regular UTF-8 source")
        return blobs[oid]

    files = []
    for status, path in pairs:
        if status not in ("A", "D", "M"):
            raise snapshots.SnapshotError("Changed file status is unsupported")
        before, after = content(old, path), content(new, path)
        # git-diff(1) --no-index compares captured bytes outside the source repository.
        # Attributes and custom drivers there must not change the historical evidence.
        with tempfile.TemporaryDirectory(prefix="review-replay-diff-") as directory:
            left, right = Path(directory) / "before", Path(directory) / "after"
            left.write_bytes(before.encode("utf-8"))
            right.write_bytes(after.encode("utf-8"))
            diff_env = dict(env, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                            GIT_ATTR_NOSYSTEM="1")
            diff_env.pop("GIT_ATTR_SOURCE", None)
            result = subprocess.run(
                [*GIT, "-C", directory, "-c", "core.attributesFile=" + os.devnull,
                 *DIFF, "--no-index", "--text", "--",
                 os.devnull if status == "A" else str(left),
                 os.devnull if status == "D" else str(right)],
                env=diff_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # --no-index implies --exit-code: 1 means differences, not an error.
            if result.returncode not in (0, 1):
                raise snapshots.SnapshotError("Captured Git diff failed")
            patch = result.stdout.decode("utf-8")
        # Header lines, which can contain "@@" in a file name, never start with it.
        start = HUNK.search(patch)
        if start is None:
            raise snapshots.SnapshotError("Changed file has no replayable text hunk")
        files.append(dict(filename=path, base_file=before, head_file=after,
                          patch=patch[start.start():],
                          edit_type={"A":"ADDED", "D":"DELETED", "M":"MODIFIED"}[status]))
    changed = {f["filename"] for f in files}
    sources = {path:blobs[oid] for path,(mode,kind,oid) in new.items()
               if path not in changed and mode in (b"100644", b"100755") and blobs.get(oid) is not None}
    for path in absent:
        raw = path.encode("utf-8", "surrogateescape")
        if (not snapshots.repo_path(path) or path in new or any(p.startswith(path+"/") for p in new)
                or any(u == raw or u.startswith(raw + b"/") for u in unlisted)):
            raise snapshots.SnapshotError("Declared absent path exists or is invalid")
    timestamp = int(git("show", "-s", "--format=%ct", head).strip())
    data = dict(schema=snapshots.SCHEMA, version=snapshots.VERSION, repository=repository, number=number,
                base=base, head=head, title="Review captured change", branch="replay/snapshot",
                description="", commits=[], languages={}, files=files, sources=sources, absent=list(absent),
                discussion={"as_of":datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "threads":[]})
    return snapshots.validate(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("repo", "repository", "base", "head", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--number", type=int, required=True)
    parser.add_argument("--absent", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        data = export(args.repo, args.repository, args.number, args.base, args.head, args.absent)
        raw = (json.dumps(data, sort_keys=True)+"\n").encode()
        # Never overwrite an earlier input used for a comparison.
        with Path(args.output).open("xb") as stream:
            stream.write(raw)
    except (OSError, ValueError, snapshots.SnapshotError):
        print("Snapshot export failed; check local Git objects, input identities and output path.")
        return 2
    print(json.dumps({"sha256":hashlib.sha256(raw).hexdigest(), "files":len(data["files"]), "sources":len(data["sources"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
