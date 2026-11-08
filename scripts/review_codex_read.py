# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Read one session's private evidence under its own UID, without following links.

The controller passes this trusted source to isolated Python through the existing
setpriv hop. No repository code or account environment is imported. Directory
descriptors, ownership, regular-file and byte checks match the session reader's
original openat(2) and stat(2) contract, with getuid(2) verifying the helper's identity.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import sys

DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class Unsafe(Exception):
    """Refuse the entire snapshot without returning partial data."""


def open_dir(name, dir_fd, uid):
    fd = os.open(name, DIR, dir_fd=dir_fd)
    try:
        if os.fstat(fd).st_uid != uid:
            raise Unsafe
    except BaseException:
        os.close(fd)
        raise
    return fd


def read_regular(name, dir_fd, uid, max_bytes):
    fd = os.open(name, FILE, dir_fd=dir_fd)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != uid or info.st_size > max_bytes:
            raise Unsafe
        # Free disk space is a storage ceiling, not an allocation size. Python's
        # standard I/O buffer keeps each read independent of that ceiling.
        chunks, size = [], 0
        while chunk := stream.read(min(io.DEFAULT_BUFFER_SIZE, max_bytes + 1 - size)):
            size += len(chunk)
            if size > max_bytes:
                raise Unsafe
            chunks.append(chunk)
    return b"".join(chunks).decode("utf-8")


def rollouts(home, uid, max_bytes):
    result = []

    def walk(fd, prefix):
        for entry in sorted(os.scandir(fd), key=lambda entry: entry.name):
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = open_dir(entry.name, fd, uid)
                try:
                    walk(child, prefix + entry.name + "/")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                if entry.name.endswith(".jsonl"):
                    result.append([prefix + entry.name, read_regular(entry.name, fd, uid, max_bytes)])
            else:
                raise Unsafe

    top = open_dir(str(home), None, uid)
    try:
        try:
            sessions = open_dir("sessions", top, uid)
        except FileNotFoundError:
            return []
        try:
            walk(sessions, "sessions/")
        finally:
            os.close(sessions)
    finally:
        os.close(top)
    return result


def output(directory, name, uid, max_bytes):
    top = open_dir(str(directory), None, uid)
    try:
        try:
            return read_regular(name, top, uid, max_bytes)
        except FileNotFoundError:
            return None
    finally:
        os.close(top)


def main():
    try:
        parent, uid, mode, bound = sys.argv[1:]
        uid, bound = int(uid), int(bound)
        if uid != os.getuid() or bound <= 0:
            raise Unsafe
        if mode == "rollouts":
            data = rollouts(Path(parent) / "codex-home", uid, bound)
        elif mode == "output":
            data = output(Path(parent) / "output", "last-message.json", uid, bound)
        else:
            raise Unsafe
        print(json.dumps(data, ensure_ascii=True))
        return 0
    except (OSError, UnicodeError, ValueError, Unsafe):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
