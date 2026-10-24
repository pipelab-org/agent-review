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
import subprocess
from pathlib import Path

if __package__:
    from . import review_replay_snapshot as snapshots
else:
    import review_replay_snapshot as snapshots


def export(repo, repository, number, base, head, absent=()):
    if not all(snapshots.followups.full_sha(s) for s in (base, head)):
        raise snapshots.SnapshotError("Export needs full base and head commit SHAs")

    def git(*args):
        result = subprocess.run(["git", "--no-replace-objects", "-C", str(repo), *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise snapshots.SnapshotError("Required Git objects are unavailable")
        return result.stdout

    for sha in (base, head):
        if git("cat-file", "-t", sha).strip() != b"commit":
            raise snapshots.SnapshotError("Export identity must name a commit")

    def tree(sha):
        result = {}
        for entry in git("ls-tree", "-r", "-z", sha).split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, kind, oid = metadata.split()
            path = raw_path.decode("utf-8")
            if not snapshots.repo_path(path):
                raise snapshots.SnapshotError("Git tree contains an unsupported path")
            result[path] = (mode, kind, oid.decode("ascii"))
        return result

    old, new = tree(base), tree(head)
    changes = git("diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--name-status", "-z", base, head).split(b"\0")
    if changes[-1] == b"":
        changes.pop()
    pairs = [(changes[i].decode("ascii"), changes[i+1].decode("utf-8")) for i in range(0, len(changes), 2)]
    needed = {v[2] for v in new.values() if v[0] in (b"100644", b"100755")}
    needed.update(old[p][2] for _, p in pairs if p in old and old[p][0] in (b"100644", b"100755"))
    # One batch avoids a process for every source file. Git's size-prefixed output
    # keeps arbitrary blob text from being interpreted as another response header.
    oids = sorted(needed)
    result = subprocess.run(["git", "--no-replace-objects", "-C", str(repo), "cat-file", "--batch"],
                            input="".join(oid+"\n" for oid in oids).encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise snapshots.SnapshotError("Git blob collection failed")
    raw, offset, blobs = result.stdout, 0, {}
    for oid in oids:
        end = raw.index(b"\n", offset)
        actual, kind, size = raw[offset:end].split()
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
        patch = git("diff", "--no-ext-diff", "--no-textconv", "--no-renames", base, head, "--", path).decode("utf-8")
        start = patch.find("@@")
        if start < 0:
            raise snapshots.SnapshotError("Changed file has no replayable text hunk")
        files.append(dict(filename=path, base_file=before, head_file=after,
                          patch=patch[start:],
                          edit_type={"A":"ADDED", "D":"DELETED", "M":"MODIFIED"}[status]))
    changed = {f["filename"] for f in files}
    sources = {path:blobs[oid] for path,(mode,kind,oid) in new.items()
               if path not in changed and mode in (b"100644", b"100755") and blobs.get(oid) is not None}
    for path in absent:
        if not snapshots.repo_path(path) or path in new or any(p.startswith(path+"/") for p in new):
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
