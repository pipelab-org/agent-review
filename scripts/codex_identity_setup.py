# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Provision and check the dedicated Codex review account on a disposable hosted runner.

Workflow steps only. It changes accounts, socket ACLs and a directory under /var/lib, so
it refuses to run anywhere but a GitHub-hosted runner; --print-plan only prints the
commands and is what local tests cover. It never grants the account access to the runner's
home or to any ancestor of the checkout: the launcher stages a copy of the head under the
controller-owned sessions directory instead.

Socket restriction deliberately reuses the pinned action's own code. Its hidden
`drop-sudo --root-phase --user U --runner-credentials JSON` mode (src/dropSudo.ts in
openai/codex-action@c05cfd74ad95c626cdf05c6685e62becc3bf7813, option definitions
`--root-phase` and `--runner-credentials`, marked internal) strips sudo entries for U (none
exist for a fresh account) and ACL-denies, then verifies, every root-owned unix socket under
/run for UID U, including world-writable ones. Pointing it at the review account instead of
the runner reuses that reviewed logic without removing the controller's sudo.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import sys

try:
    from . import review_codex_sessions as sessions
except ImportError:
    import review_codex_sessions as sessions

SUDO = "/usr/bin/sudo"
SESSIONS_DIR = "/var/lib/agent-review-sessions"
RUNTIME_DIR = "/var/lib/agent-review-runtime"


def hosted():
    return os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"


def valid_user(user):
    if not isinstance(user, str) or not sessions.NAME.fullmatch(user):
        raise SystemExit("Invalid review account name")
    return user


def valid_directory(path):
    # A directory the controller creates for session trees; never an arbitrary location.
    if (not isinstance(path, str) or not path.startswith("/var/lib/") or ".." in Path(path).parts
            or re.search(r"[\x00-\x1f\x7f]", path)):
        raise SystemExit("Sessions directory must be an absolute path under /var/lib")
    return path


def provision_plan(user, sessions_dir, controller_uid, controller_gid):
    return [[SUDO, "-n", "--", "/usr/sbin/useradd", "--system", "--user-group", "--no-create-home",
             "--shell", "/usr/sbin/nologin", "--", valid_user(user)],
            [SUDO, "-n", "--", "/usr/bin/install", "-d", "-o", str(controller_uid), "-g", str(controller_gid),
             "-m", "0711", "--", valid_directory(sessions_dir)]]


def restrict_plan(identity, action, node):
    main = Path(action).resolve() / "dist/main.js"
    credentials = json.dumps({"userId": identity.uid, "primaryGroupId": identity.gid,
                              "supplementaryGroupIds": []}, separators=(",", ":"))
    return [[SUDO, "-n", "--", "/usr/bin/env", "-u", "NODE_OPTIONS", str(node), str(main), "drop-sudo",
             "--root-phase", "--user", identity.user, "--group", "sudo", "--runner-credentials", credentials]]


def runtime_plan(controller_uid, controller_gid, node):
    """Create an npm prefix closed to other writers and copy the selected Node runtime.

    install(1) sets the exact modes; npm's documented --prefix then puts the CLI
    and its dependencies below this directory, outside the shared runner toolcache.
    """
    if not node or not os.path.isabs(node):
        raise SystemExit("The trusted Node runtime was not found")
    return [[SUDO, "-n", "--", "/usr/bin/install", "-d", "-o", str(controller_uid), "-g", str(controller_gid),
             "-m", "0755", "--", directory] for directory in (RUNTIME_DIR, RUNTIME_DIR + "/bin")] + [
            [SUDO, "-n", "--", "/usr/bin/install", "-m", "0755", "--", os.path.realpath(node),
             RUNTIME_DIR + "/bin/node"]]


def execute(plan):
    for number, argv in enumerate(plan, 1):
        print(f"step {number} of {len(plan)}: {argv[3] if len(argv) > 3 else argv[0]}", flush=True)
        subprocess.run(argv, check=True)


def refuse_unless_hosted():
    if not hosted():
        raise SystemExit("Refusing: this changes accounts and ACLs and runs only on a GitHub-hosted runner")


def check_new_account(user, controller_uid):
    entry = pwd.getpwnam(user)
    groups = os.getgrouplist(user, entry.pw_gid)
    if entry.pw_uid in (0, controller_uid) or entry.pw_gid == 0 or groups != [entry.pw_gid]:
        raise SystemExit("Provisioned review account is not unprivileged")
    return sessions.Identity(user, entry.pw_uid, entry.pw_gid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-plan", action="store_true", help="print the commands and stop")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("runtime")
    one = commands.add_parser("provision")
    one.add_argument("--user", required=True)
    one.add_argument("--sessions-dir", default=SESSIONS_DIR)
    two = commands.add_parser("restrict-sockets")
    two.add_argument("--user", required=True)
    two.add_argument("--action", required=True, type=Path)
    four = commands.add_parser("verify")
    four.add_argument("--user", required=True)
    four.add_argument("--codex", required=True)
    four.add_argument("--trusted-path", required=True)
    args = parser.parse_args(argv)
    controller_uid, controller_gid = os.getuid(), os.getgid()
    if args.command == "provision":
        plan = provision_plan(args.user, args.sessions_dir, controller_uid, controller_gid)
    elif args.command == "runtime":
        plan = runtime_plan(controller_uid, controller_gid, shutil.which("node"))
    else:
        valid_user(args.user)
        if args.print_plan and args.command in ("restrict-sockets", "verify"):
            try:
                identity = sessions.resolve_identity(args.user)
            except Exception:
                identity = sessions.Identity(args.user, 60001, 60001)  # shown shape only; nothing runs
        elif args.command != "provision":
            refuse_unless_hosted()
            identity = sessions.resolve_identity(args.user)
        if args.command == "restrict-sockets":
            plan = restrict_plan(identity, args.action, os.path.realpath(shutil.which("node") or "node"))
        else:
            plan = [["verify", args.user]]
    if args.print_plan:
        print(json.dumps(plan))
        return 0
    refuse_unless_hosted()
    if args.command == "runtime":
        if Path(RUNTIME_DIR).exists() or Path(RUNTIME_DIR).is_symlink():
            raise SystemExit("Refusing: the runtime prefix already exists; it must be created by this job")
        execute(plan)
        print("protected runtime prefix prepared")
    elif args.command == "provision":
        try:
            pwd.getpwnam(args.user)
        except KeyError:
            pass
        else:
            raise SystemExit("Refusing: the review account already exists; it must be created by this job")
        execute(plan)
        identity = check_new_account(args.user, controller_uid)
        print(f"provisioned review account uid={identity.uid} gid={identity.gid}")
    elif args.command == "verify":
        launcher = sessions.CodexSessions(identity, codex=args.codex, trusted_path=args.trusted_path)
        launcher.verify()
        print("review identity verified")
    else:
        execute(plan)
        if args.command == "restrict-sockets":
            sessions.CodexSessions(identity, codex="/unused", trusted_path="/usr/bin").verify_sockets()
            print("review identity cannot write any root-owned socket")
    return 0


if __name__ == "__main__":
    sys.exit(main())
