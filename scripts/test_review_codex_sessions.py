# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the multi-session Codex launcher.

Nothing here uses sudo, changes an account or exercises the kernel. The helper-command
runner, the Popen seam and the OS lookups are replaced by local fixtures, and the REAL
argv builder is asserted separately. Kernel, sandbox and hosted evidence comes only from
scripts/hosted_codex_sandbox_proof.py on a disposable runner.
"""
import contextlib
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

from scripts import review_codex_sessions as s
from scripts.review_errors import ReviewAbort

ME = os.getuid()
CONTROLLER = 424242  # the fixture's controller UID; real files are reported as owned by it
FAKE_CODEX = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
home = pathlib.Path(os.environ["CODEX_HOME"])
prompt = sys.stdin.read()
seen = sorted(p.name for p in home.iterdir())
sessions = home / "sessions" / "2026"
sessions.mkdir(parents=True, exist_ok=True)
tag = os.environ.get("FIXTURE_TAG", "a")
(sessions / (tag + ".jsonl")).write_text(json.dumps({"home": seen}) + "\n")
pathlib.Path(args[args.index("--output-last-message") + 1]).write_text('{"findings": []}')
print(json.dumps({"prompt_bytes": len(prompt.encode())}))
'''


def owned_by(info, uid):
    values = list(info)
    values[4] = uid
    return os.stat_result(values)


class FakeSystem(s._System):
    """The host as the launcher sees it: a controller that is not the identity."""

    def __init__(self, identity):
        self.identity = identity
        self.groups = [NS(gr_name=identity.user, gr_gid=identity.gid, gr_mem=[])]
        self.passwd = [NS(pw_name=identity.user, pw_uid=identity.uid, pw_gid=identity.gid)]
        self.xattr = None
        self.socket_paths = []

    def controller_uid(self):
        return CONTROLLER

    def getpwnam(self, name):
        for entry in self.passwd:
            if entry.pw_name == name:
                return entry
        raise KeyError(name)

    def getpwall(self):
        return self.passwd

    def getgrall(self):
        return self.groups

    def getgrgid(self, gid):
        return next(g for g in self.groups if g.gr_gid == gid)

    def lstat(self, path):
        info = os.lstat(path)
        return owned_by(info, CONTROLLER) if info.st_uid == ME else info

    def getxattr(self, path, name):
        if self.xattr is not None:
            return self.xattr
        raise OSError(61, "no data")  # ENODATA

    def sockets(self, root):
        return list(self.socket_paths)


class Host:
    """Stands in for sudo/setpriv helpers by running the equivalent local operation."""

    def __init__(self, root):
        self.proc = root / "proc"
        self.proc.mkdir()
        self.calls, self.writable, self.kills_work = [], set(), True
        self.residue, self.find_status, self.ipcrm, self.ipcrm_works = {}, 0, [], True
        self.sudo_listing = "User fixture is not allowed to run sudo on host.\n"

    def add_process(self, pid, uid):
        directory = self.proc / str(pid)
        directory.mkdir()
        (directory / "status").write_text(f"Name:\tx\nState:\tS (sleeping)\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")

    def run(self, argv, **kw):
        self.calls.append(list(argv))
        args = list(argv)
        assert args[:2] == ["/usr/bin/sudo", "-n"], args
        if args[2:4] == ["-l", "-U"]:
            return subprocess.CompletedProcess(args, 0, self.sudo_listing, "")
        assert args[2] == "--", args
        inner = args[3:]
        if inner[0] == "/usr/bin/env":  # a command under the identity's credentials
            command = inner[inner.index("--") + 1:]
            if command[:2] == ["/usr/bin/test", "-w"]:
                return subprocess.CompletedProcess(args, 0 if command[2] in self.writable else 1, "", "")
            return subprocess.run(command, capture_output=True, text=True, check=False)
        if inner[0] == "/usr/bin/install":
            *options, target = inner[1:]
            paths = [p for p in options if p.startswith("/")]
            if "-d" in options:
                Path(target).mkdir(parents=True, mode=0o700, exist_ok=True)
                Path(target).chmod(0o700)
            else:
                Path(target).write_bytes(Path(paths[-1]).read_bytes())
                Path(target).chmod(0o600)
            return subprocess.CompletedProcess(args, 0, "", "")
        if inner[0] == "/usr/bin/pkill":
            if self.kills_work:
                uid = inner[-1]
                for entry in list(self.proc.iterdir()):
                    if f"\t{uid}\t" in (entry / "status").read_text():
                        (entry / "status").unlink()
                        entry.rmdir()
            return subprocess.CompletedProcess(args, 0, "", "")
        if inner[0] == "/usr/bin/find":
            if "-print" in inner:
                return subprocess.CompletedProcess(args, self.find_status, self.residue.get(inner[1], ""), "")
            return subprocess.CompletedProcess(args, 0, "", "")
        if inner[0] == "/usr/bin/ipcrm":
            self.ipcrm.append(inner[1:])
            if self.ipcrm_works:
                for table in (self.proc / "sysvipc").glob("*"):
                    lines = table.read_text().splitlines()
                    table.write_text("\n".join(l for l in lines if inner[2] not in l.split()[:2]) + "\n")
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.identity = s.Identity("fixture", ME, 60001)
        self.system = FakeSystem(self.identity)
        self.host = Host(self.root)
        bins = self.root / "bin"
        bins.mkdir()
        for name in ("codex", "node"):
            (bins / name).write_text("#!/bin/sh\n")
            (bins / name).chmod(0o755)
        self.codex, self.path = str(bins / "codex"), str(bins)
        self.launched = []
        (self.root / "fake_codex.py").write_text(FAKE_CODEX)
        self.sessions = self.make()
        self.work = self.root / "work"
        self.work.mkdir()
        self.trusted = self.root / "trusted-home"
        self.trusted.mkdir()
        (self.trusted / "config.toml").write_text("model_provider = 'fixture'\n")
        self.seen = []

    def make(self, **kwargs):
        def runner(argv, **popen):
            self.launched.append(list(argv))
            codex = argv.index(self.codex)
            # Pairs sit after the second "-i": sudo's -- comes first, then setpriv's.
            hop = argv.index("--", argv.index("--") + 1)
            env = dict(pair.split("=", 1) for pair in argv[argv.index("-i", hop) + 1:codex])
            return subprocess.Popen([sys.executable, str(self.root / "fake_codex.py"), *argv[codex + 1:]],
                                    env=env, stdin=popen["stdin"], stdout=popen["stdout"], stderr=popen["stderr"])
        options = dict(codex=self.codex, trusted_path=self.path, runner=runner, run=self.host.run,
                       system=self.system, proc_root=str(self.host.proc), sockets_root="/nonexistent")
        options.update(kwargs)
        return s.CodexSessions(self.identity, **options)

    def open(self, **kwargs):
        options = dict(trusted_home=self.trusted, configure=self.seen.append, schema_text="{}",
                       deadline=time.monotonic() + 30, max_bytes=65536)
        options.update(kwargs)
        return self.sessions.session(self.work, **options)


class ArgvTests(Base):
    def env(self, **extra):
        return s.environment(self.identity, home="/s/home", codex_home="/s/codex-home", tmp="/s/tmp",
                             trusted_path=self.path, extra=extra)

    def argv(self):
        return s.launch_argv(self.identity, sudo="/usr/bin/sudo", setpriv="/usr/bin/setpriv", codex=self.codex,
                             env=self.env(), head="/h/pr-head", output="/s/output/last-message.json",
                             schema="/s/schema.json", model="gpt-6.1-terra", effort="high")

    def test_exact_argv_shape(self):
        self.assertEqual(
            ["/usr/bin/sudo", "-n", "--", "/usr/bin/env", "-i", "PATH=/usr/sbin:/usr/bin:/sbin:/bin",
             "/usr/bin/setpriv", f"--reuid={ME}", "--regid=60001", "--clear-groups", "--no-new-privs",
             "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", "--", "/usr/bin/env", "-i",
             "HOME=/s/home", "CODEX_HOME=/s/codex-home", "TMPDIR=/s/tmp", f"USER=fixture", "LOGNAME=fixture",
             f"PATH={self.path}", "CODEX_INTERNAL_ORIGINATOR_OVERRIDE=codex_github_action",
             self.codex, "exec", "--skip-git-repo-check", "--cd", "/h/pr-head",
             "--output-last-message", "/s/output/last-message.json", "--output-schema", "/s/schema.json",
             "--model", "gpt-6.1-terra", "--config", 'model_reasoning_effort="high"', "--json",
             "--config", 'default_permissions=":read-only"', "-"], self.argv())

    def test_only_the_read_only_profile_and_no_preserved_environment(self):
        argv = self.argv()
        for banned in ("--sandbox", "-E", "--preserve-env", "danger-full-access", "--dangerously-bypass-approvals-and-sandbox"):
            self.assertNotIn(banned, argv)
        self.assertEqual(1, [a for a in argv if a.startswith("default_permissions=")].__len__())

    def test_environment_is_an_allowlist_and_forbidden_names_are_refused(self):
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN", "LD_PRELOAD", "NODE_OPTIONS",
                     "SUDO_USER", "ACTIONS_RUNTIME_TOKEN", "GITHUB_ENV", "BASH_ENV", "CODEX_HOME", "PATH"):
            with self.subTest(name=name), self.assertRaises(ReviewAbort):
                self.env(**{name: "x"})
        self.assertEqual("/ca.pem", self.env(SSL_CERT_FILE="/ca.pem")["SSL_CERT_FILE"])
        with self.assertRaises(ReviewAbort):
            self.env(LANG="a\nb")
        with self.assertRaises(ReviewAbort):
            s.CodexSessions(self.identity, codex=self.codex, trusted_path=self.path, env_extra={"GITHUB_TOKEN": "x"})

    def test_invalid_paths_models_and_efforts_are_refused(self):
        base = dict(sudo="/usr/bin/sudo", setpriv="/usr/bin/setpriv", codex=self.codex, env=self.env(),
                    head="/h", output="/o", schema="/s", model="m", effort="high")
        for field, value in (("head", "relative"), ("output", "/o\n"), ("model", "m; rm"), ("effort", "High"),
                             ("codex", "codex")):
            with self.subTest(field=field), self.assertRaises(ReviewAbort):
                s.launch_argv(self.identity, **dict(base, **{field: value}))

    def test_session_start_uses_the_real_builder_and_feeds_the_prompt(self):
        with self.open() as session:
            head = self.root / "head"
            head.mkdir()
            process = session.start("review this", head=head, model="gpt-6.1-terra", effort="high")
            out = process.stdout.read()
            process.wait(timeout=20)
        self.assertEqual(b'{"prompt_bytes": 11}', out.strip())
        argv = self.launched[0]
        self.assertEqual(argv[:3], ["/usr/bin/sudo", "-n", "--"])
        self.assertIn("--no-new-privs", argv)
        self.assertEqual(argv[argv.index("--cd") + 1], str(head))


class IdentityTests(Base):
    def test_clean_identity_verifies(self):
        self.sessions.verify()

    def refuse(self, pattern=None):
        with self.assertRaisesRegex(ReviewAbort, pattern or ".") as caught:
            self.sessions.verify()
        self.assertNotIsInstance(caught.exception, s.SessionCleanupError)

    def test_root_controller_and_account_mismatch_are_refused(self):
        for identity in (s.Identity("fixture", 0, 60001), s.Identity("fixture", 60001, 0)):
            with self.subTest(identity=identity):
                self.sessions.identity = identity
                self.system.passwd = [NS(pw_name="fixture", pw_uid=identity.uid, pw_gid=identity.gid)]
                self.refuse("neither root nor the controller")
        self.sessions.identity = s.Identity("fixture", CONTROLLER, 60001)
        self.refuse("neither root nor the controller")
        self.sessions.identity = self.identity
        self.system.passwd = [NS(pw_name="fixture", pw_uid=ME + 1, pw_gid=60001)]
        self.refuse("does not match")

    def test_group_membership_and_shared_primary_group_are_refused(self):
        self.system.groups.append(NS(gr_name="docker", gr_gid=999, gr_mem=["fixture"]))
        self.refuse("no supplementary groups")
        self.system.groups.pop()
        self.system.groups[0].gr_mem = ["someone"]
        self.refuse("private primary group")
        self.system.groups[0].gr_mem = []
        self.system.passwd.append(NS(pw_name="other", pw_uid=ME + 5, pw_gid=60001))
        self.refuse("private primary group")

    def test_sudo_rights_and_live_processes_are_refused(self):
        self.host.sudo_listing = "User fixture may run the following commands on host:\n    (ALL) NOPASSWD: ALL\n"
        self.refuse("sudo rights")
        self.host.sudo_listing = "User fixture is not allowed to run sudo on host.\n"
        self.host.add_process(77, ME)
        self.refuse("job-exclusive")

    def test_untrusted_codex_and_path_are_refused(self):
        bins = Path(self.path)
        os.chmod(self.codex, 0o775)  # group-writable
        self.refuse("writable")
        os.chmod(self.codex, 0o755)
        os.chmod(bins, 0o777)  # writable, not sticky
        self.refuse("writable")
        os.chmod(bins, 0o700)
        extra = self.root / "extra"
        extra.mkdir()
        self.sessions.trusted_path = f"{self.path}:{extra}"
        self.sessions.verify()
        os.chmod(extra, 0o777)  # a PATH directory the identity could write, away from the codex file
        self.refuse("PATH is writable")
        os.chmod(extra, 0o700)
        self.sessions.trusted_path = self.path
        os.chmod(self.codex, 0o4755)
        self.refuse("plain world-executable")
        os.chmod(self.codex, 0o755)
        self.system.xattr = b"\x01\x00"
        self.refuse("file capabilities")
        self.system.xattr = None
        self.sessions.trusted_path = "relative:" + self.path
        self.refuse("PATH is invalid")

    def test_identity_owned_executable_ancestor_is_refused(self):
        original = self.system.lstat
        self.system.lstat = lambda p: owned_by(original(p), ME) if p == self.path else original(p)
        self.refuse("untrusted owner")

    def test_socket_the_identity_can_write_is_refused(self):
        self.system.socket_paths = ["/run/fixture.sock", "/run/other.sock"]
        self.sessions.verify()
        self.host.writable.add("/run/other.sock")
        self.refuse("can write a privileged service socket")

    def test_unverifiable_socket_probe_is_refused(self):
        self.system.socket_paths = ["/run/fixture.sock"]
        real = self.host.run
        self.sessions._run = lambda argv, **kw: (subprocess.CompletedProcess(argv, 2, "", "")
                                                 if "/usr/bin/test" in argv and "-w" in argv else real(argv, **kw))
        self.refuse("could not be verified")


class ReaderTests(Base):
    def session(self):
        parent = self.root / "parent"
        (parent / "codex-home" / "sessions" / "2026").mkdir(parents=True)
        (parent / "output").mkdir()
        return s.Session(self.sessions, parent, 64, time.monotonic() + 5), parent

    def test_regular_rollouts_and_output_are_read(self):
        session, parent = self.session()
        (parent / "codex-home" / "sessions" / "2026" / "a.jsonl").write_text('{"x": 1}\n')
        (parent / "codex-home" / "sessions" / "2026" / "ignored.sqlite").write_text("not read")
        (parent / "output" / s.OUTPUT_NAME).write_text('{"findings": []}')
        self.assertEqual([("sessions/2026/a.jsonl", '{"x": 1}\n')], session.rollout_texts())
        self.assertEqual('{"findings": []}', session.read_output())

    def test_missing_sessions_and_output_are_not_errors(self):
        session, parent = self.session()
        self.assertEqual([], session.rollout_texts())
        self.assertIsNone(session.read_output())
        (parent / "codex-home" / "sessions").rmdir() if False else None
        import shutil
        shutil.rmtree(parent / "codex-home" / "sessions")
        self.assertEqual([], session.rollout_texts())

    def test_symlinked_file_directory_and_special_files_are_refused(self):
        secret = self.root / "secret.jsonl"
        secret.write_text('{"secret": true}\n')
        for build in (
                lambda d: (d / "link.jsonl").symlink_to(secret),
                lambda d: (d / "dir").symlink_to(self.root),
                lambda d: os.mkfifo(d / "pipe.jsonl")):
            session, parent = self.session()
            directory = parent / "codex-home" / "sessions" / "2026"
            build(directory)
            with self.subTest(), self.assertRaisesRegex(ReviewAbort, "unsafe or unreadable"):
                session.rollout_texts()
            import shutil
            shutil.rmtree(parent)

    def test_symlinked_sessions_directory_and_output_are_refused(self):
        session, parent = self.session()
        import shutil
        shutil.rmtree(parent / "codex-home" / "sessions")
        # A valid-looking tree elsewhere: following the link would return it as a genuine rollout.
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "a.jsonl").write_text('{"forged": true}\n')
        (parent / "codex-home" / "sessions").symlink_to(outside)
        with self.assertRaises(ReviewAbort):
            session.rollout_texts()
        (parent / "output" / s.OUTPUT_NAME).symlink_to(self.root / "trusted-home" / "config.toml")
        with self.assertRaises(ReviewAbort):
            session.read_output()

    def test_oversize_and_wrong_owner_are_refused(self):
        session, parent = self.session()
        (parent / "codex-home" / "sessions" / "2026" / "big.jsonl").write_text("x" * 65)
        with self.assertRaises(ReviewAbort):
            session.rollout_texts()
        (parent / "codex-home" / "sessions" / "2026" / "big.jsonl").write_text("x\n")
        self.assertEqual(1, len(session.rollout_texts()))
        self.sessions.identity = s.Identity("fixture", ME + 1, 60001)
        with self.assertRaises(ReviewAbort):
            session.rollout_texts()
        with self.assertRaises(ReviewAbort):
            session.read_output()


class CleanupResidueTests(Base):
    """What a session may leave behind outside its own tree: IPC objects and owned files."""

    def ipc(self, *rows):
        table = self.host.proc / "sysvipc"
        table.mkdir(exist_ok=True)
        (table / "shm").write_text("key shmid perms size cpid lpid nattch uid gid cuid cgid\n" + "".join(rows))

    def test_sysv_ipc_objects_owned_or_created_by_the_identity_are_removed_and_others_kept(self):
        self.ipc(f"0 77 600 4096 1 1 0 {ME} 9 9 9\n", f"0 78 600 4096 1 1 0 9 9 {ME} 9\n", "0 79 600 4096 1 1 0 9 9 9 9\n")
        with self.open():
            pass
        self.assertEqual([["-m", "77"], ["-m", "78"]], self.host.ipcrm)
        self.assertIn("79", (self.host.proc / "sysvipc" / "shm").read_text())

    def test_an_ipc_object_that_cannot_be_removed_blocks_the_next_launch(self):
        self.ipc(f"0 77 600 4096 1 1 0 {ME} 9 9 9\n")
        self.host.ipcrm_works = False
        with self.assertRaisesRegex(s.SessionCleanupError, "IPC objects could not be removed"), self.open():
            pass
        self.assertTrue(self.sessions.poisoned)

    def test_an_unreadable_or_malformed_ipc_table_fails_closed(self):
        table = self.host.proc / "sysvipc"
        table.mkdir()
        (table / "shm").write_text("key shmid perms\n0 1 600\n")
        with self.assertRaisesRegex(s.SessionCleanupError, "unreadable"), self.open():
            pass
        self.sessions.poisoned = False
        (table / "shm").write_text("key shmid perms size cpid lpid nattch uid gid cuid cgid\n0 1\n")
        with self.assertRaisesRegex(s.SessionCleanupError, "unreadable"), self.open():
            pass

    def test_files_owned_by_the_identity_anywhere_on_a_scanned_root_block_the_next_launch(self):
        self.host.residue = {"/": "/var/lib/stray-file\n"}
        with self.assertRaisesRegex(s.SessionCleanupError, "left files owned by its account"), self.open():
            pass
        self.assertTrue(self.sessions.poisoned)
        self.sessions.poisoned = False
        self.host.residue = {"/dev/shm": "/dev/shm/leftover\n"}
        with self.assertRaisesRegex(s.SessionCleanupError, "left files owned by its account"), self.open():
            pass
        self.sessions.poisoned = False
        self.host.residue, self.host.find_status = {}, 2
        with self.assertRaisesRegex(s.SessionCleanupError, "left files owned by its account"), self.open():
            pass

    def test_every_documented_scratch_location_is_cleaned(self):
        self.assertEqual({"/tmp", "/var/tmp", "/dev/shm", "/dev/mqueue", "/run/lock", "/var/crash"}, set(s.SCRATCH_DIRS))
        self.assertIn("/", s.RESIDUE_ROOTS)
        with self.open():
            pass
        cleaned = {c[c.index("/usr/bin/find") + 1] for c in self.host.calls if "-delete" in c and "-user" in c}
        self.assertEqual({d for d in s.SCRATCH_DIRS if os.path.isdir(d)}, cleaned)


class StageHeadTests(Base):
    def source(self):
        tree = self.root / "checkout"
        (tree / "pkg").mkdir(parents=True)
        (tree / ".git" / "objects").mkdir(parents=True)
        (tree / ".git" / "objects" / "history").write_text("secret history")
        (tree / ".git" / "agent-review" / "patches").mkdir(parents=True)
        (tree / ".git" / "agent-review" / "patches" / "0.diff").write_text("patch")
        (tree / "pkg" / "a.py").write_text("print(1)\n")
        (tree / "pkg" / "a.py").chmod(0o600)
        (tree / "run.sh").write_text("#!/bin/sh\n")
        (tree / "run.sh").chmod(0o777)
        (tree / "link").symlink_to("/etc/hostname")
        return tree

    def test_the_head_is_copied_without_history_and_stays_read_only_for_others(self):
        tree = self.source()
        staged = self.sessions.stage_head(tree, self.work)
        self.assertEqual(self.work / "head", staged)
        self.assertEqual("print(1)\n", (staged / "pkg" / "a.py").read_text())
        self.assertFalse((staged / ".git" / "objects").exists())
        self.assertEqual("patch", (staged / ".git" / "agent-review" / "patches" / "0.diff").read_text())
        self.assertTrue((staged / "link").is_symlink())
        for path in (staged / "pkg" / "a.py", staged / "run.sh", staged / "pkg", staged):
            mode = path.stat().st_mode
            self.assertTrue(mode & stat.S_IROTH, path)
            self.assertFalse(mode & 0o022, path)
        self.assertTrue((staged / "run.sh").stat().st_mode & stat.S_IXOTH)
        # The original checkout is untouched.
        self.assertTrue((tree / ".git" / "objects" / "history").exists())

    def test_staging_again_replaces_the_copy_and_refuses_a_hostile_destination(self):
        tree = self.source()
        self.sessions.stage_head(tree, self.work)
        (tree / "pkg" / "a.py").write_text("print(2)\n")
        staged = self.sessions.stage_head(tree, self.work)
        self.assertEqual("print(2)\n", (staged / "pkg" / "a.py").read_text())
        import shutil as _shutil
        _shutil.rmtree(staged)
        staged.symlink_to(self.root)
        with self.assertRaisesRegex(ReviewAbort, "staged head location is invalid"):
            self.sessions.stage_head(tree, self.work)
        staged.unlink()
        with self.assertRaisesRegex(ReviewAbort, "workspace is invalid"):
            self.sessions.stage_head(tree, self.root / "missing")

    def test_a_copy_the_identity_cannot_read_is_refused(self):
        real = self.host.run
        def deny(argv, **kw):
            if "/usr/bin/test" in argv and "-r" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "")
            return real(argv, **kw)
        self.sessions._run = deny
        with self.assertRaisesRegex(ReviewAbort, "cannot read the staged head"):
            self.sessions.stage_head(self.source(), self.work)


class LifecycleTests(Base):
    def run_one(self, tag="a", head=None, **kwargs):
        head = head or self.root / "head"
        head.mkdir(exist_ok=True)
        with self.open(**kwargs) as session:
            process = session.start("prompt", head=head, model="gpt-6.1-terra", effort="high")
            process.stdout.read()
            process.wait(timeout=20)
            return session.rollout_texts(), session.read_output(), session.parent

    def test_sessions_are_fresh_and_fully_removed(self):
        first, output, parent = self.run_one()
        self.assertEqual('{"findings": []}', output)
        self.assertEqual("sessions/2026/a.jsonl", first[0][0])
        self.assertFalse(parent.exists())
        self.assertEqual([], list(self.work.iterdir()))
        second, _, _ = self.run_one()
        # The fake lists the home it sees: only trusted config, never the first rollout.
        self.assertEqual(1, len(second))
        self.assertIn('"home": ["config.toml"]', second[0][1])
        self.assertEqual([], list(self.work.iterdir()))

    def test_configure_hook_runs_on_the_staged_copy_before_the_identity_owns_it(self):
        def configure(path):
            (path / "config.toml").write_text("model_provider = 'edited'\n")
            self.seen.append(path)
        parent_holder = []
        with self.open(configure=configure) as session:
            parent_holder.append(session.parent)
            self.assertEqual("model_provider = 'edited'\n", (session.home / "config.toml").read_text())
        self.assertEqual("model_provider = 'fixture'\n", (self.trusted / "config.toml").read_text())
        self.assertNotIn("stage", [p.name for p in self.work.iterdir()])

    def test_trusted_home_with_a_symlink_or_special_file_is_refused(self):
        (self.trusted / "link").symlink_to(self.root)
        with self.assertRaisesRegex(ReviewAbort, "unsupported entry"), self.open():
            pass
        self.assertEqual([], list(self.work.iterdir()))
        (self.trusted / "link").unlink()
        os.mkfifo(self.trusted / "pipe")
        with self.assertRaisesRegex(ReviewAbort, "unsupported entry"), self.open():
            pass

    def test_failure_inside_the_session_still_cleans(self):
        with self.assertRaisesRegex(RuntimeError, "boom"), self.open() as session:
            (session.parent / "tmp" / "scratch").write_text("x")
            raise RuntimeError("boom")
        self.assertEqual([], list(self.work.iterdir()))

    def test_surviving_identity_process_blocks_the_next_launch(self):
        self.host.kills_work = False
        with self.assertRaisesRegex(s.SessionCleanupError, "could not be stopped"):
            with self.open(deadline=time.monotonic() - 1) as session:
                self.host.add_process(900, ME)
        self.assertTrue(self.sessions.poisoned)
        self.launched.clear()
        with self.assertRaisesRegex(s.SessionCleanupError, "not cleaned up"), self.open():
            pass
        self.assertEqual([], self.launched)

    def test_detached_descendant_is_found_by_uid_not_process_group(self):
        killed = []
        real = self.host.run
        def run(argv, **kw):
            if "/usr/bin/pkill" in argv:
                killed.append(argv[-2:])
            return real(argv, **kw)
        self.sessions._run = run
        with self.open() as session:
            # A setsid/double-fork child: its process group is not the session's. Only the UID names it.
            self.host.add_process(4242, ME)
            self.assertEqual([4242], s.uid_pids(str(self.host.proc), ME))
        self.assertIn(["-U", str(ME)], killed)
        self.assertEqual([], s.uid_pids(str(self.host.proc), ME))

    def test_zombies_and_other_users_do_not_count_and_threads_do(self):
        self.host.add_process(10, ME + 9)
        zombie = self.host.proc / "11"
        zombie.mkdir()
        (zombie / "status").write_text(f"Name:x\nState:\tZ (zombie)\nUid:\t{ME}\t{ME}\t{ME}\t{ME}\n")
        threaded = self.host.proc / "12"
        (threaded / "task" / "13").mkdir(parents=True)
        (threaded / "status").write_text(f"State:\tS (sleeping)\nUid:\t1\t1\t1\t1\n")
        (threaded / "task" / "13" / "status").write_text(f"State:\tS (sleeping)\nUid:\t1\t{ME}\t1\t1\n")
        self.assertEqual([12], s.uid_pids(str(self.host.proc), ME))
        (threaded / "status").write_text("garbage")
        with self.assertRaisesRegex(ReviewAbort, "Process table"):
            s.uid_pids(str(self.host.proc), ME)

    def test_unreadable_session_files_cannot_hide_a_cleanup_failure(self):
        real = self.host.run
        def run(argv, **kw):
            if "/usr/bin/find" in argv and "-mindepth" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "")
            return real(argv, **kw)
        self.sessions._run = run
        with self.assertRaisesRegex(s.SessionCleanupError, "could not be removed"), self.open():
            pass
        self.assertTrue(self.sessions.poisoned)

    def test_scratch_cleanup_failure_blocks_the_next_session(self):
        real = self.host.run
        def run(argv, **kw):
            if "/usr/bin/find" in argv and "-user" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "")
            return real(argv, **kw)
        self.sessions._run = run
        with self.assertRaisesRegex(s.SessionCleanupError, "scratch"), self.open():
            pass
        with self.assertRaises(s.SessionCleanupError), self.open():
            pass

    def test_identity_must_read_the_head_it_is_given(self):
        real = self.host.run
        def run(argv, **kw):
            if "/usr/bin/test" in argv and "-r" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "")
            return real(argv, **kw)
        self.sessions._run = run
        with self.assertRaisesRegex(ReviewAbort, "cannot read the head"), self.open() as session:
            session.start("p", head=self.root, model="m", effort="high")
        self.assertEqual([], self.launched)

    def test_pre_launch_verify_runs_for_every_session(self):
        self.host.add_process(5, ME)
        with self.assertRaisesRegex(ReviewAbort, "job-exclusive"), self.open():
            pass
        self.assertEqual([], list(self.work.iterdir()))


class SetupTests(unittest.TestCase):
    """The provisioning commands only print their plan here; nothing is executed."""

    def plan(self, *argv, hosted=False):
        import contextlib
        import io
        from unittest.mock import patch
        from scripts import codex_identity_setup as setup
        out = io.StringIO()
        env = {"GITHUB_ACTIONS": "true", "RUNNER_ENVIRONMENT": "github-hosted"} if hosted else {}
        with patch.dict(os.environ, env), patch.object(subprocess, "run", side_effect=AssertionError("executed")), \
                contextlib.redirect_stdout(out):
            code = setup.main(["--print-plan", *argv])
        self.assertEqual(0, code)
        import json
        return json.loads(out.getvalue())

    def test_provision_plan_creates_a_system_account_without_home_or_shell(self):
        plan = self.plan("provision", "--user", "agent-review-model")
        self.assertEqual(["/usr/sbin/useradd", "--system", "--user-group", "--no-create-home", "--shell",
                          "/usr/sbin/nologin", "--", "agent-review-model"], plan[0][3:])
        self.assertEqual(["/usr/bin/install", "-d", "-o", str(os.getuid()), "-g", str(os.getgid()), "-m", "0711",
                          "--", "/var/lib/agent-review-sessions"], plan[1][3:])

    def test_socket_restriction_reuses_the_pinned_action_root_phase_for_the_identity(self):
        plan = self.plan("restrict-sockets", "--user", "agent-review-model", "--action", "/tmp/act")
        command = plan[0]
        self.assertEqual(["/usr/bin/sudo", "-n", "--", "/usr/bin/env", "-u", "NODE_OPTIONS"], command[:6])
        self.assertEqual(["/tmp/act/dist/main.js", "drop-sudo", "--root-phase", "--user", "agent-review-model",
                          "--group", "sudo", "--runner-credentials"], command[7:15])
        import json
        credentials = json.loads(command[15])
        self.assertEqual([], credentials["supplementaryGroupIds"])
        self.assertNotEqual(os.getuid(), credentials["userId"])

    def test_no_ancestor_acl_grant_exists_any_more(self):
        from scripts import codex_identity_setup as setup
        self.assertFalse(hasattr(setup, "share_plan"))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            setup.main(["--print-plan", "share", "--user", "agent-review-model", "--path", "/tmp"])

    def test_invalid_names_directories_and_unhosted_execution_are_refused(self):
        from unittest.mock import patch
        from scripts import codex_identity_setup as setup
        for bad in ("Root", "a b", "", "x;y"):
            with self.subTest(bad), self.assertRaises(SystemExit):
                setup.provision_plan(bad, "/var/lib/x", 1, 1)
        for bad in ("/etc/x", "/var/lib/../etc", "relative", "/var/lib/a\nb"):
            with self.subTest(bad), self.assertRaises(SystemExit):
                setup.provision_plan("ok", bad, 1, 1)
        from scripts import hosted_codex_sandbox_proof as proof
        # This negative state must remain unhosted when the suite itself runs in Actions.
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(setup, "execute", side_effect=AssertionError("privileged execution")):
            with self.assertRaises(SystemExit) as caught:
                setup.main(["provision", "--user", "agent-review-model"])
            self.assertIn("GitHub-hosted", str(caught.exception))
            with self.assertRaises(SystemExit) as caught:
                proof.main(["identity", "--user", "x", "--head", "/tmp"])
            self.assertIn("disposable GitHub-hosted", str(caught.exception))

    def test_the_probe_source_is_valid_python_and_prints_no_secret_values(self):
        from scripts import hosted_codex_sandbox_proof as proof
        compile(proof.PROBE, "probe", "exec")
        self.assertNotIn("sentinel_value", proof.PROBE)

    def test_probe_reports_exec_permission_denial_and_preserves_command_status(self):
        import json
        from unittest.mock import patch
        from scripts import hosted_codex_sandbox_proof as proof
        with tempfile.TemporaryDirectory() as temp:
            scenario = dict(suid="/fixture/id", capcat="/fixture/cat", canary="/fixture/socket",
                            controller_pid=os.getpid(), sentinel_env="FIXTURE_ABSENT_ENV",
                            sentinel_file=temp + "/absent", head=temp, runner_home=temp, source=temp)
            for status in ("exec-denied", 0, 1):
                def command(argv, **kwargs):
                    if argv[0] == scenario["capcat"] and status == "exec-denied":
                        raise PermissionError("fixture capability execution denied")
                    return NS(returncode=status if argv[0] == scenario["capcat"] else 1, stdout="{}")
                out = io.StringIO()
                with self.subTest(status=status), patch("subprocess.run", side_effect=command), \
                        patch("socket.socket") as sock, patch("sys.stdin", io.StringIO(json.dumps(scenario))), \
                        contextlib.redirect_stdout(out):
                    sock.return_value.connect.side_effect = PermissionError("fixture socket denied")
                    exec(compile(proof.PROBE, "probe", "exec"), {"__name__": "__main__"})
                self.assertEqual(status, json.loads(out.getvalue())["capability_read_status"])


if __name__ == "__main__":
    unittest.main()
