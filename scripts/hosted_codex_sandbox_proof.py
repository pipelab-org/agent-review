# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Launcher-identity proof for the multi-session Codex launcher. Disposable hosted runners only.

This runs the REAL provisioning, the REAL launcher (sudo, setpriv, process cleanup) and a
probe executable in place of Codex, so the kernel evidence it prints is about the launch
identity and cleanup only. It is NOT proof of Codex's :read-only sandbox, of bubblewrap, or
of any provider behavior. It uses no provider credentials and makes no model request.

It refuses to run unless GitHub reports a hosted runner, and must never be run on a
workstation: it creates accounts, ACLs, a root socket and setuid/capability fixtures.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time

try:
    from . import review_codex_sessions as sessions
except ImportError:
    import review_codex_sessions as sessions

SUDO, ROOT = "/usr/bin/sudo", Path("/var/lib/agent-review-sessions")
CANARY = "/run/agent-review-canary.sock"
STATUS_KEYS = ("Uid", "Gid", "Groups", "NoNewPrivs", "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
ZERO_CAPS = "0000000000000000"

# Executed as the review identity in place of Codex. Prints booleans and ids, never contents.
PROBE = r'''#!/usr/bin/python3
import json, os, socket, subprocess, sys, time
scenario = json.loads(sys.stdin.read())
KEYS = %(keys)r
def snapshot(source="/proc/self/status"):
    values = {}
    for line in open(source):
        name, _, value = line.partition(":")
        if name in KEYS:
            values[name] = value.strip()
    return values
def denied(action):
    try:
        action()
    except OSError:
        return True
    return False
result = {"self": snapshot()}
result["child"] = json.loads(subprocess.run([sys.executable, "-c",
    "import json;K=%%r;print(json.dumps({n:v.strip() for n,_,v in (l.partition(':') for l in open('/proc/self/status')) if n in K}))" %% (KEYS,)],
    capture_output=True, text=True).stdout)
result["sudo_status"] = subprocess.run(["/usr/bin/sudo", "-n", "true"], capture_output=True).returncode
result["setuid_euid"] = subprocess.run([scenario["suid"], "-u"], capture_output=True, text=True).stdout.strip()
result["capability_read_status"] = subprocess.run([scenario["capcat"], "/etc/shadow"], capture_output=True).returncode
def connect():
    sock = socket.socket(socket.AF_UNIX)
    try:
        sock.connect(scenario["canary"])
    finally:
        sock.close()
result["canary_denied"] = denied(connect)
pid = scenario["controller_pid"]
result["sentinel"] = {
    "env_name_absent": scenario["sentinel_env"] not in os.environ,
    "file_denied": denied(lambda: open(scenario["sentinel_file"]).read()),
    "environ_denied": denied(lambda: open(f"/proc/{pid}/environ", "rb").read(1)),
    "fd_list_denied": denied(lambda: os.listdir(f"/proc/{pid}/fd")),
    "mem_denied": denied(lambda: open(f"/proc/{pid}/mem", "rb").read(1)),
}
head = scenario["head"]
result["head"] = {"readable": os.access(head, os.R_OK | os.X_OK),
                  "write_denied": denied(lambda: open(os.path.join(head, "proof-write-probe"), "w")),
                  "create_denied": denied(lambda: os.mkdir(os.path.join(head, "proof-dir")))}
if scenario.get("escape"):
    reader, writer = os.pipe()
    if os.fork() == 0:
        os.setsid()
        if os.fork() == 0:
            os.close(reader)
            for descriptor in (0, 1, 2):
                os.dup2(os.open(os.devnull, os.O_RDWR), descriptor)
            os.write(writer, str(os.getpid()).encode())
            os.close(writer)
            time.sleep(600)
        os._exit(0)
    os.close(writer)
    result["escapee"] = int(os.read(reader, 32).decode())
if scenario.get("see"):
    first = scenario["see"]
    me = os.getuid()
    def owned(name):
        try:
            return any(line.startswith("Uid:") and line.split()[1] == str(me) for line in open(f"/proc/{name}/status"))
        except OSError:
            return False
    mine = [name for name in os.listdir("/proc") if name.isdigit() and owned(name)]
    result["see"] = {"first_tree_gone": not os.path.exists(first["parent"]),
                     "escapee_gone": not os.path.exists(f"/proc/{first['escapee']}"),
                     "identity_processes": len(mine)}
print(json.dumps(result))
''' % {"keys": STATUS_KEYS}


class ProofFailure(Exception):
    pass


def check(condition, name):
    if not condition:
        raise ProofFailure(name)
    print(f"PASS {name}", flush=True)


def require_hosted():
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted":
        raise SystemExit("Refusing: this proof creates accounts, sockets and privileged fixtures and runs only "
                         "on a disposable GitHub-hosted runner")


def as_identity(identity, command, **kwargs):
    return subprocess.run([SUDO, "-n", "--", "/usr/bin/env", "-i", "PATH=" + sessions.PRIVILEGED_PATH,
                           "/usr/bin/setpriv", f"--reuid={identity.uid}", f"--regid={identity.gid}",
                           "--clear-groups", "--no-new-privs", "--bounding-set=-all", "--inh-caps=-all",
                           "--ambient-caps=-all", "--", *command], capture_output=True, text=True, **kwargs)


def canary_start():
    """A root-owned, world-writable unix socket: the shape drop-sudo must deny."""
    source = ("import os, socket, sys, time\n"
              f"path = {CANARY!r}\n"
              "sock = socket.socket(socket.AF_UNIX)\n"
              "sock.bind(path)\n"
              "os.chmod(path, 0o666)\n"
              "sock.listen(8)\n"
              "sys.stdout.write('ready\\n'); sys.stdout.flush()\n"
              "time.sleep(3600)\n")
    # No inherited step streams: a background process holding them can stall the workflow step.
    process = subprocess.Popen([SUDO, "-n", "--", "/usr/bin/python3", "-c", source], stdout=subprocess.PIPE,
                               stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True,
                               start_new_session=True)
    check(process.stdout.readline().strip() == "ready", "canary root socket is listening")


def canary_probe(identity, expect):
    connect = ("import socket,sys\ns=socket.socket(socket.AF_UNIX)\n"
               f"try:\n s.connect({CANARY!r})\nexcept OSError:\n sys.exit(3)\n")
    status = as_identity(identity, ["/usr/bin/python3", "-c", connect]).returncode
    check((status == 0) == (expect == "allowed"), f"canary socket is {expect} to the review identity")


def install_fixtures(root):
    """Controller-owned probe and root-owned privilege fixtures; all harmless copies."""
    probe_dir, fixtures = root / "probe", root / "fixtures"
    # The probe belongs to the controller (it writes it); the privilege fixtures belong to root.
    subprocess.run([SUDO, "-n", "--", "/usr/bin/install", "-d", "-o", str(os.getuid()), "-g", str(os.getgid()),
                    "-m", "0755", "--", str(probe_dir)], check=True)
    subprocess.run([SUDO, "-n", "--", "/usr/bin/install", "-d", "-m", "0755", "--", str(fixtures)], check=True)
    probe = probe_dir / "codex"
    probe.write_text(PROBE)
    probe.chmod(0o755)
    suid, capcat = fixtures / "suid-id", fixtures / "cap-cat"
    # A setuid-root copy of id(1): with no_new_privs the bit must not raise the effective UID.
    subprocess.run([SUDO, "-n", "--", "/usr/bin/install", "-o", "root", "-g", "root", "-m", "4755", "--",
                    "/usr/bin/id", str(suid)], check=True)
    # A copy of cat(1) with CAP_DAC_READ_SEARCH: an empty bounding set must keep it from reading /etc/shadow.
    subprocess.run([SUDO, "-n", "--", "/usr/bin/install", "-o", "root", "-g", "root", "-m", "0755", "--",
                    "/usr/bin/cat", str(capcat)], check=True)
    subprocess.run([SUDO, "-n", "--", "/usr/sbin/setcap", "cap_dac_read_search+ep", str(capcat)], check=True)
    return probe, suid, capcat


def scenario(identity, head, suid, capcat, sentinel, **extra):
    return json.dumps({"suid": str(suid), "capcat": str(capcat), "canary": CANARY, "head": str(head),
                       "controller_pid": os.getpid(), "sentinel_env": sentinel["env"],
                       "sentinel_file": sentinel["file"], **extra})


def run_probe(launcher, head, suid, capcat, sentinel, workspace, **extra):
    with launcher.session(workspace, trusted_home=sentinel["home"], configure=lambda path: None, schema_text="{}",
                          deadline=time.monotonic() + 120, max_bytes=1 << 20) as live:
        process = live.start(scenario(launcher.identity, head, suid, capcat, sentinel, **extra),
                             head=head, model="probe", effort="high")
        out, _ = process.communicate(timeout=120)
        parent = live.parent
    return json.loads(out.strip().splitlines()[-1]), parent


def identity_proof(user, head):
    identity = sessions.resolve_identity(user)
    root = ROOT
    probe, suid, capcat = install_fixtures(root)
    sentinel_dir = Path(tempfile.mkdtemp(prefix="proof-sentinel-"))
    sentinel_dir.chmod(0o700)
    value, name = secrets.token_hex(16), "AGENT_REVIEW_PROOF_SENTINEL_" + secrets.token_hex(4).upper()
    os.environ[name] = value
    (sentinel_dir / "sentinel.txt").write_text(value)
    held = open(sentinel_dir / "sentinel.txt")  # an open descriptor the identity must not reach
    home = Path(tempfile.mkdtemp(prefix="proof-trusted-home-"))
    (home / "config.toml").write_text("# proof fixture\n")
    sentinel = {"env": name, "file": str(sentinel_dir / "sentinel.txt"), "home": home}
    path = f"{probe.parent}:/usr/bin:/bin"
    def launcher():
        # The probe is a Python script, so no Node runtime is required of this launcher instance.
        return sessions.CodexSessions(identity, codex=str(probe), trusted_path=path, sockets_root="/run", runtime=None)
    first = launcher()
    first.verify()
    check(True, "launcher pre-launch verification passes for the provisioned identity")
    result, parent = run_probe(first, head, suid, capcat, sentinel, root, escape=True)
    for field in ("self", "child"):
        who = result[field]
        label = "session process" if field == "self" else "descendant"
        check(who["Uid"].split() == [str(identity.uid)] * 4, f"{label}: real/effective/saved/fs UID are the reserved UID")
        check(who["Gid"].split() == [str(identity.gid)] * 4, f"{label}: real/effective/saved/fs GID are the reserved GID")
        check(who["Groups"] == "", f"{label}: no supplementary groups")
        check(who["NoNewPrivs"] == "1", f"{label}: no_new_privs is set")
        check(all(who[key] == ZERO_CAPS for key in STATUS_KEYS[5:]), f"{label}: inheritable/permitted/effective/bounding/ambient capabilities are empty")
    check(result["sudo_status"] != 0, "sudo -n true fails for the session")
    check(result["setuid_euid"] == str(identity.uid), "a setuid-root fixture does not raise the effective UID")
    check(result["capability_read_status"] != 0, "a file-capability fixture cannot read a root-only file")
    check(result["canary_denied"] is True, "the root canary socket is denied to the session")
    check(all(result["sentinel"].values()), "controller environment, file, descriptors, memory and /proc entries are unreachable")
    check(result["head"] == {"readable": True, "write_denied": True, "create_denied": True},
          "the session reads the shared head checkout and cannot write it")
    escapee = result["escapee"]
    check(sessions.uid_pids("/proc", identity.uid) == [], "cleanup left no process of the review identity")
    check(not Path(f"/proc/{escapee}").exists(), "a setsid escaping descendant was killed by UID-based cleanup")
    check(not parent.exists(), "the session tree was removed")
    second, _ = run_probe(launcher(), head, suid, capcat, sentinel, root,
                          see={"parent": str(parent), "escapee": escapee})
    check(second["see"] == {"first_tree_gone": True, "escapee_gone": True, "identity_processes": 1},
          "a second session sees no first-session tree and no surviving process")
    # A failed cleanup must stop the review: inject the failure at the helper seam only.
    broken = launcher()
    real = broken._run
    def refuse_removal(argv, **kw):
        if "-mindepth" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "")
        return real(argv, **kw)
    broken._run = refuse_removal
    leftover = None
    try:
        with broken.session(root, trusted_home=home, configure=lambda path: None, schema_text="{}",
                            deadline=time.monotonic() + 60, max_bytes=1 << 20) as live:
            leftover = live.parent
    except sessions.SessionCleanupError:
        pass
    check(broken.poisoned and leftover is not None, "a failed cleanup poisons the launcher")
    try:
        with broken.session(root, trusted_home=home, configure=lambda path: None, schema_text="{}",
                            deadline=time.monotonic() + 60, max_bytes=1 << 20):
            launched = True
    except sessions.SessionCleanupError:
        launched = False
    check(launched is False, "no session launches after a failed cleanup")
    subprocess.run([SUDO, "-n", "--", "/usr/bin/rm", "-rf", "--one-file-system", "--", str(leftover)], check=True)
    held.close()
    shutil.rmtree(sentinel_dir)
    print("LIMIT: this is launcher-identity evidence with a probe executable. It is not evidence about "
          "Codex's :read-only profile, bubblewrap, or any provider.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("canary-start")
    probe = commands.add_parser("canary-probe")
    probe.add_argument("--user", required=True)
    probe.add_argument("--expect", choices=("allowed", "denied"), required=True)
    identity = commands.add_parser("identity")
    identity.add_argument("--user", required=True)
    identity.add_argument("--head", required=True, type=Path)
    args = parser.parse_args(argv)
    require_hosted()
    try:
        if args.command == "canary-start":
            canary_start()
        elif args.command == "canary-probe":
            canary_probe(sessions.resolve_identity(args.user), args.expect)
        else:
            identity_proof(args.user, args.head.resolve())
    except ProofFailure as failure:
        print(f"FAIL {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
