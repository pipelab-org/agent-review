# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Many native Codex sessions in one job, each under a dedicated unprivileged identity.

The pinned action's drop-sudo removes the runner's sudo for the rest of the job, so
it cannot launch a second session. This launcher keeps sudo for the trusted
controller and starts every session as `codex exec` under an account that has no
sudo, no groups and no capabilities, then proves the account is gone from the
process table before the next session can start.

Sources (nothing here is a guessed limit):
- util-linux setpriv(1): --reuid/--regid/--clear-groups/--no-new-privs/--bounding-set,
  https://github.com/util-linux/util-linux/blob/master/sys-utils/setpriv.1.adoc
- sudo(8): -n never prompts; env_reset drops the caller's environment.
- proc(5): per-process and per-task `Uid:` (real, effective, saved, filesystem) and `State:`.
- openat(2): O_NOFOLLOW/O_DIRECTORY/O_NONBLOCK, the no-follow reads below.
- capabilities(7): file capabilities live in the security.capability xattr.
- Codex permission profiles, https://developers.openai.com/codex/permissions , and the
  pinned runner's argv shape, openai/codex-action@c05cfd74ad95c626cdf05c6685e62becc3bf7813
  src/runCodexExec.ts and src/dropSudo.ts (privileged service socket restriction).
"""
from __future__ import annotations

import contextlib
import errno
import grp
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from typing import Callable, Iterator, Mapping, NamedTuple

try:
    from .review_errors import ReviewAbort
except ImportError:
    from review_errors import ReviewAbort


class SessionCleanupError(ReviewAbort):
    """Cleanup could not be confirmed; the caller must stop and launch nothing else."""


class Identity(NamedTuple):
    user: str
    uid: int
    gid: int


NAME = re.compile(r"[a-z_][a-z0-9_-]{0,31}")
# Same shape the pinned action's sudo hop uses before it hands over to setpriv.
PRIVILEGED_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
# The transport and locale names the engine's clean_env already forwards, plus nothing else.
ENV_ALLOWED = frozenset((
    "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy"))
SCAN_INTERVAL = 0.05  # seconds between process-table scans; not a policy bound
LEAVES = ("home", "codex-home", "tmp", "output")
OUTPUT_NAME = "last-message.json"
# Identity-owned scratch locations outside the session tree (sticky world-writable dirs).
SCRATCH_DIRS = ("/tmp", "/var/tmp", "/dev/shm")
_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class _System:
    """Operating-system lookups behind one seam so tests can model other hosts."""

    controller_uid = staticmethod(os.getuid)
    getpwnam = staticmethod(pwd.getpwnam)
    getpwall = staticmethod(pwd.getpwall)
    getgrall = staticmethod(grp.getgrall)
    getgrgid = staticmethod(grp.getgrgid)
    lstat = staticmethod(os.lstat)
    realpath = staticmethod(os.path.realpath)

    @staticmethod
    def getxattr(path, name):
        return os.getxattr(path, name, follow_symlinks=False)

    @staticmethod
    def sockets(root):
        return root_sockets(root)


def root_sockets(root):
    """Root-owned unix sockets below `root`, found without following symlinks.

    Mirrors the pinned action's findRootServiceSockets: O_NOFOLLOW directory
    descriptors, lstat on every entry, only uid 0 sockets. A directory the
    controller cannot list is one the identity (no groups) cannot list either.
    """
    found = []

    def walk(fd, shown):
        try:
            entries = list(os.scandir(fd))
        except OSError:
            return
        for entry in sorted(entries, key=lambda e: e.name):
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            path = shown + "/" + entry.name
            if stat.S_ISDIR(info.st_mode):
                try:
                    child = os.open(entry.name, _DIR, dir_fd=fd)
                except OSError:
                    continue
                try:
                    walk(child, path)
                finally:
                    os.close(child)
            elif stat.S_ISSOCK(info.st_mode) and info.st_uid == 0:
                found.append(path)

    try:
        top = os.open(root, _DIR)
    except OSError as error:
        if error.errno == errno.ENOENT:
            return []
        raise
    try:
        walk(top, str(root).rstrip("/"))
    finally:
        os.close(top)
    return found


def resolve_identity(user):
    if not isinstance(user, str) or not NAME.fullmatch(user):
        raise ReviewAbort("Codex review identity name is invalid")
    try:
        entry = pwd.getpwnam(user)
    except KeyError:
        raise ReviewAbort("Codex review identity is not provisioned") from None
    return Identity(user, entry.pw_uid, entry.pw_gid)


def uid_pids(proc_root, uid):
    """Thread-group IDs with any real/effective/saved/filesystem UID equal to `uid`.

    Tasks are scanned too, since credentials are per thread. A zombie holds no
    resources and cannot read anything, so it does not count. A process that
    vanishes mid-scan is gone; an unreadable process table is an error.
    """
    found = set()
    try:
        names = [e.name for e in os.scandir(proc_root) if e.name.isdigit()]
    except OSError:
        raise ReviewAbort("Process table is unavailable; Codex isolation cannot be confirmed") from None
    for name in names:
        statuses = [f"{proc_root}/{name}/status"]
        try:
            statuses += [f"{proc_root}/{name}/task/{t.name}/status"
                         for t in os.scandir(f"{proc_root}/{name}/task") if t.name.isdigit()]
        except OSError:
            pass
        for path in statuses:
            try:
                text = Path(path).read_text(errors="replace")
            except OSError as error:
                if error.errno in (errno.ENOENT, errno.ESRCH):
                    continue
                raise ReviewAbort("Process table is unavailable; Codex isolation cannot be confirmed") from None
            state = re.search(r"(?m)^State:\s*(\S)", text)
            ids = re.search(r"(?m)^Uid:\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*$", text)
            if not state or not ids:
                raise ReviewAbort("Process table is unavailable; Codex isolation cannot be confirmed")
            if state[1] != "Z" and str(uid) in ids.groups():
                found.add(int(name))
    return sorted(found)


def _ancestors(path):
    path = Path(path)
    return [path, *path.parents]


def trusted_executable(path, controller_uid, identity, system):
    """Refuse an executable the identity could replace or that carries extra privilege.

    Every component of the path, before and after symlink resolution, must belong
    to root or the controller and be closed to group/other writers. The file must
    not be setuid/setgid and must not carry a file capability.
    """
    if not isinstance(path, str) or not os.path.isabs(path) or "\x00" in path:
        raise ReviewAbort("Codex executable path is invalid")
    real = system.realpath(path)
    for item in dict.fromkeys(_ancestors(path) + _ancestors(real)):
        if str(item) == "/":
            continue
        try:
            info = system.lstat(str(item))
        except OSError:
            raise ReviewAbort("Codex executable path is unreadable") from None
        if not closed_to_identity(info, controller_uid):
            raise ReviewAbort("Codex executable path is writable by the review identity or an untrusted owner")
    info = system.lstat(real)
    if not stat.S_ISREG(info.st_mode) or info.st_mode & (stat.S_ISUID | stat.S_ISGID) or not info.st_mode & 0o001:
        raise ReviewAbort("Codex executable is not a plain world-executable file")
    try:
        system.getxattr(real, "security.capability")
    except OSError as error:
        if error.errno not in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise ReviewAbort("Codex executable capabilities could not be checked") from None
    else:
        raise ReviewAbort("Codex executable carries file capabilities")
    return real


def environment(identity, *, home, codex_home, tmp, trusted_path, extra=(), lang=None):
    """The complete session environment; nothing is inherited from the controller."""
    env = {"HOME": str(home), "CODEX_HOME": str(codex_home), "TMPDIR": str(tmp),
           "USER": identity.user, "LOGNAME": identity.user, "PATH": trusted_path,
           # Set by the pinned runner for the same originator.
           "CODEX_INTERNAL_ORIGINATOR_OVERRIDE": "codex_github_action"}
    if lang:
        env["LANG"] = lang
    for name, value in dict(extra).items():
        if name not in ENV_ALLOWED or name in env:
            raise ReviewAbort("Codex session environment name is not allowed")
        env[name] = value
    for name, value in env.items():
        if (not isinstance(value, str) or "=" in name or re.search(r"[\x00-\x1f\x7f]", value)
                or (name in ("HOME", "CODEX_HOME", "TMPDIR", "PATH") and not os.path.isabs(value.split(":")[0]))):
            raise ReviewAbort("Codex session environment value is invalid")
    return env


def launch_argv(identity, *, sudo, setpriv, codex, env, head, output, schema, model, effort):
    """argv for one session. No shell; every value comes from validated trusted setup.

    The prompt travels on stdin ("-"). Only the :read-only profile is selected; no
    legacy sandbox flag is added beside it.
    """
    for value in (head, output, schema, codex):
        if not isinstance(value, str) or not os.path.isabs(value) or re.search(r"[\x00-\x1f\x7f]", value):
            raise ReviewAbort("Codex session path is invalid")
    if (not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", model)
            or not isinstance(effort, str) or not re.fullmatch(r"[a-z]+", effort)):
        raise ReviewAbort("Codex model or effort is invalid")
    return [sudo, "-n", "--", "/usr/bin/env", "-i", "PATH=" + PRIVILEGED_PATH,
            setpriv, f"--reuid={identity.uid}", f"--regid={identity.gid}", "--clear-groups",
            "--no-new-privs", "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all",
            "--", "/usr/bin/env", "-i", *[f"{k}={v}" for k, v in env.items()],
            codex, "exec", "--skip-git-repo-check", "--cd", head, "--output-last-message", output,
            "--output-schema", schema, "--model", model,
            "--config", f'model_reasoning_effort="{effort}"', "--json",
            "--config", 'default_permissions=":read-only"', "-"]


def _open_dir(name, dir_fd, uid):
    fd = os.open(name, _DIR, dir_fd=dir_fd)
    try:
        if os.fstat(fd).st_uid != uid:
            raise ReviewAbort("Codex session files are unsafe or unreadable")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_regular(name, dir_fd, uid, max_bytes):
    fd = os.open(name, _FILE, dir_fd=dir_fd)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != uid or info.st_size > max_bytes:
            raise ReviewAbort("Codex session files are unsafe or unreadable")
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ReviewAbort("Codex session files are unsafe or unreadable")
    try:
        return data.decode("utf-8")
    except UnicodeError:
        raise ReviewAbort("Codex session files are unsafe or unreadable") from None


class Session:
    """One fresh session tree. Created and torn down only by CodexSessions.session()."""

    def __init__(self, owner, parent, max_bytes, deadline):
        self.owner, self.parent, self.max_bytes, self.deadline = owner, parent, max_bytes, deadline
        self.home = parent / "codex-home"
        self.output = parent / "output" / OUTPUT_NAME
        self.schema = parent / "schema.json"
        self.process = None
        self._feeder = None

    def start(self, prompt, *, head, model, effort):
        if self.process is not None:
            raise ReviewAbort("Codex session was already started")
        head = Path(head)
        # The identity must be able to read the head it is asked to review.
        if self.owner._as_identity(["/usr/bin/test", "-d", str(head), "-a", "-r", str(head), "-a", "-x", str(head)]):
            raise ReviewAbort("The Codex review identity cannot read the head checkout")
        env = environment(self.owner.identity, home=self.parent / "home", codex_home=self.home,
                          tmp=self.parent / "tmp", trusted_path=self.owner.trusted_path,
                          extra=self.owner.env_extra, lang=self.owner.lang)
        argv = launch_argv(self.owner.identity, sudo=self.owner.sudo, setpriv=self.owner.setpriv,
                           codex=self.owner.codex, env=env, head=str(head), output=str(self.output),
                           schema=str(self.schema), model=model, effort=effort)
        self.process = self.owner.runner(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         close_fds=True, cwd="/", env={"PATH": PRIVILEGED_PATH})
        data = prompt.encode("utf-8")

        def feed():
            try:
                self.process.stdin.write(data)
            except OSError:
                pass
            finally:
                with contextlib.suppress(OSError):
                    self.process.stdin.close()

        self._feeder = threading.Thread(target=feed, daemon=True)
        self._feeder.start()
        return self.process

    def rollout_texts(self):
        """(relative name, text) for every sessions/**/*.jsonl, read without following links."""
        uid = self.owner.identity.uid
        result = []

        def walk(fd, prefix):
            for entry in sorted(os.scandir(fd), key=lambda e: e.name):
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child = _open_dir(entry.name, fd, uid)
                    try:
                        walk(child, prefix + entry.name + "/")
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    if entry.name.endswith(".jsonl"):
                        result.append((prefix + entry.name, _read_regular(entry.name, fd, uid, self.max_bytes)))
                else:
                    # Symlinks, FIFOs and sockets never occur in a genuine rollout tree.
                    raise ReviewAbort("Codex session files are unsafe or unreadable")

        try:
            top = os.open(str(self.home), _DIR)
        except OSError:
            raise ReviewAbort("Codex session files are unsafe or unreadable") from None
        try:
            if os.fstat(top).st_uid != uid:
                raise ReviewAbort("Codex session files are unsafe or unreadable")
            try:
                sessions = _open_dir("sessions", top, uid)
            except FileNotFoundError:
                return []
            try:
                walk(sessions, "sessions/")
            finally:
                os.close(sessions)
        except OSError:
            raise ReviewAbort("Codex session files are unsafe or unreadable") from None
        finally:
            os.close(top)
        return result

    def read_output(self):
        uid = self.owner.identity.uid
        try:
            directory = os.open(str(self.output.parent), _DIR)
        except OSError:
            raise ReviewAbort("Codex session files are unsafe or unreadable") from None
        try:
            if os.fstat(directory).st_uid != uid:
                raise ReviewAbort("Codex session files are unsafe or unreadable")
            try:
                return _read_regular(OUTPUT_NAME, directory, uid, self.max_bytes)
            except FileNotFoundError:
                return None
            except OSError:
                raise ReviewAbort("Codex session files are unsafe or unreadable") from None
        finally:
            os.close(directory)


class CodexSessions:
    def __init__(self, identity, *, codex, trusted_path, runner=None, sudo="/usr/bin/sudo",
                 setpriv="/usr/bin/setpriv", env_extra=(), proc_root="/proc", sockets_root="/run",
                 run=None, system=None, lang=None, runtime="node"):
        self.identity, self.codex, self.trusted_path = identity, codex, trusted_path
        # The interpreter the Codex launcher script needs on the trusted PATH; None for a self-contained binary.
        self.runtime = runtime
        self.runner = runner or subprocess.Popen
        self.sudo, self.setpriv, self.env_extra = sudo, setpriv, dict(env_extra)
        self.proc_root, self.sockets_root = proc_root, sockets_root
        # Test seams: the helper-command runner and the OS lookups. Production uses both defaults.
        self._run = run or (lambda argv, **kw: subprocess.run(argv, capture_output=True, text=True, check=False, **kw))
        self.system = system or _System()
        self.lang = lang
        self.poisoned = False
        environment(identity, home="/h", codex_home="/c", tmp="/t", trusted_path=trusted_path,
                    extra=self.env_extra, lang=lang)

    # -- privileged helpers ------------------------------------------------
    def _sudo(self, *args):
        return self._run([self.sudo, "-n", "--", *args])

    def _as_identity(self, command):
        """Run `command` with the session's exact credentials; return its exit status."""
        argv = [self.sudo, "-n", "--", "/usr/bin/env", "-i", "PATH=" + PRIVILEGED_PATH, self.setpriv,
                f"--reuid={self.identity.uid}", f"--regid={self.identity.gid}", "--clear-groups",
                "--no-new-privs", "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all",
                "--", *command]
        return self._run(argv).returncode

    # -- checks ------------------------------------------------------------
    def verify(self):
        """Refuse unless the identity and the launcher are in the exact trusted shape."""
        who, system = self.identity, self.system
        controller = system.controller_uid()
        if who.uid == 0 or who.gid == 0 or who.uid == controller:
            raise ReviewAbort("Codex review identity must be neither root nor the controller")
        try:
            entry = system.getpwnam(who.user)
        except KeyError:
            raise ReviewAbort("Codex review identity is not provisioned") from None
        if entry.pw_uid != who.uid or entry.pw_gid != who.gid:
            raise ReviewAbort("Codex review identity does not match its account")
        # Private primary group: nobody else lists it as a primary or a member.
        group = system.getgrgid(who.gid)
        if group.gr_mem or any(p.pw_gid == who.gid and p.pw_name != who.user for p in system.getpwall()):
            raise ReviewAbort("Codex review identity must have a private primary group")
        if any(who.user in g.gr_mem for g in system.getgrall()):
            raise ReviewAbort("Codex review identity must have no supplementary groups")
        # sudo(8) -l -U reports another user's rights; "not allowed" is the only clean answer.
        listing = self._run([self.sudo, "-n", "-l", "-U", who.user])
        if "is not allowed to run sudo" not in (getattr(listing, "stdout", "") or ""):
            raise ReviewAbort("Codex review identity might hold sudo rights")
        if uid_pids(self.proc_root, who.uid):
            raise ReviewAbort("Codex review identity already has processes; it must be job-exclusive")
        trusted_executable(self.codex, controller, who, system)
        directories = self.trusted_path.split(":")
        for directory in directories:
            if not directory or not os.path.isabs(directory):
                raise ReviewAbort("Codex trusted PATH is invalid")
            trusted_executable_dir(directory, controller, who, system)
        if self.runtime and not shutil.which(self.runtime, path=self.trusted_path):
            raise ReviewAbort("Codex runtime is not reachable on the trusted PATH")
        self.verify_sockets()

    def verify_sockets(self):
        """The identity may write no root-owned unix socket (the pinned action's rule)."""
        for socket_path in self.system.sockets(self.sockets_root):
            # Same technique as the pinned action: test -w under the identity's exact credentials.
            status = self._as_identity(["/usr/bin/test", "-w", socket_path])
            if status == 0:
                raise ReviewAbort("The Codex review identity can write a privileged service socket")
            if status != 1:
                raise ReviewAbort("Access to a privileged service socket could not be verified")

    # -- sessions ----------------------------------------------------------
    @contextlib.contextmanager
    def session(self, workspace, *, trusted_home, configure, schema_text, deadline, max_bytes) -> Iterator[Session]:
        if self.poisoned:
            raise SessionCleanupError("A previous Codex session was not cleaned up; no further session may start")
        self.verify()
        workspace = Path(workspace)
        if workspace.is_symlink() or not workspace.is_dir():
            raise ReviewAbort("Codex session workspace is invalid")
        if self._as_identity(["/usr/bin/test", "-x", str(workspace)]):
            raise ReviewAbort("The Codex review identity cannot traverse the session workspace")
        parent = Path(tempfile.mkdtemp(prefix="codex-session-", dir=workspace))
        live = Session(self, parent, max_bytes, deadline)
        try:
            self._build(live, trusted_home, configure, schema_text)
            yield live
        finally:
            self._finish(live)

    def _build(self, live, trusted_home, configure, schema_text):
        parent, who = live.parent, self.identity
        parent.chmod(0o711)  # identity may traverse by name, never list
        stage = parent / "stage"
        stage.mkdir(mode=0o700)
        staged_home = stage / "codex-home"
        _copy_trusted(Path(trusted_home), staged_home)
        configure(staged_home)
        live.schema.write_text(schema_text, encoding="utf-8")
        live.schema.chmod(0o444)
        for leaf in LEAVES:
            self._need(self._sudo("/usr/bin/install", "-d", "-o", str(who.uid), "-g", str(who.gid),
                                  "-m", "0700", "--", str(parent / leaf)))
        self._install_tree(staged_home, live.home)
        shutil.rmtree(stage)

    def _install_tree(self, source, target):
        who = self.identity
        for current, directories, files in os.walk(source):
            relative = Path(current).relative_to(source)
            for name in sorted(directories):
                self._need(self._sudo("/usr/bin/install", "-d", "-o", str(who.uid), "-g", str(who.gid),
                                      "-m", "0700", "--", str(target / relative / name)))
            for name in sorted(files):
                self._need(self._sudo("/usr/bin/install", "-o", str(who.uid), "-g", str(who.gid),
                                      "-m", "0600", "--", str(Path(current) / name), str(target / relative / name)))

    @staticmethod
    def _need(result):
        if result.returncode:
            raise ReviewAbort("Codex session setup failed")

    def _finish(self, live):
        """Stop every identity process, confirm none remain, then remove all session state."""
        process = live.process
        try:
            self._reap(live.deadline)
            if process is not None and process.poll() is None:
                with contextlib.suppress(OSError):
                    process.kill()
                process.wait()
            if process is not None:
                for stream in (process.stdout, process.stdin):
                    with contextlib.suppress(OSError, ValueError, AttributeError):
                        stream.close()
            if live._feeder is not None:
                live._feeder.join()
            self._remove(live)
        except SessionCleanupError:
            self.poisoned = True
            raise
        except BaseException:
            self.poisoned = True
            raise SessionCleanupError("Codex session cleanup could not be confirmed") from None

    def _reap(self, deadline):
        empty, attempts = 0, 0
        while True:
            pids = uid_pids(self.proc_root, self.identity.uid)
            if not pids:
                empty += 1
                if empty >= 2:
                    return
            else:
                empty = 0
                if attempts and time.monotonic() > deadline:
                    raise SessionCleanupError("Codex session processes could not be stopped; no further session may start")
                attempts += 1
                # By UID, never by PID: a PID can be reused between the scan and the kill.
                for flag in ("-U", "-u"):
                    self._sudo("/usr/bin/pkill", "-KILL", flag, str(self.identity.uid))
            time.sleep(SCAN_INTERVAL)

    def _remove(self, live):
        for leaf in LEAVES:
            path = live.parent / leaf
            if not path.exists() and not path.is_symlink():
                continue
            # Contents go as the identity (least privilege); the empty leaf belongs to a directory
            # the controller owns, so the controller removes it.
            if self._as_identity(["/usr/bin/find", str(path), "-xdev", "-mindepth", "1", "-delete"]):
                raise SessionCleanupError("Codex session files could not be removed; no further session may start")
            os.rmdir(path)
        shutil.rmtree(live.parent)
        for directory in SCRATCH_DIRS:
            if os.path.isdir(directory):
                self._need_cleanup(self._sudo("/usr/bin/find", directory, "-xdev", "-user", str(self.identity.uid), "-delete"))
        if os.path.lexists(live.parent) or uid_pids(self.proc_root, self.identity.uid):
            raise SessionCleanupError("Codex session cleanup could not be confirmed; no further session may start")

    @staticmethod
    def _need_cleanup(result):
        if result.returncode:
            raise SessionCleanupError("Codex session scratch files could not be removed; no further session may start")


def trusted_executable_dir(path, controller_uid, identity, system):
    """A PATH directory and every ancestor must be closed to the identity and untrusted owners."""
    for item in dict.fromkeys(_ancestors(path)):
        if str(item) == "/":
            continue
        try:
            info = system.lstat(str(item))
        except OSError:
            raise ReviewAbort("Codex trusted PATH is unreadable") from None
        if not closed_to_identity(info, controller_uid):
            raise ReviewAbort("Codex trusted PATH is writable by the review identity or an untrusted owner")


def closed_to_identity(info, controller_uid):
    """Owned by root or the controller and closed to group/other writers.

    The identity is neither root nor the controller (verify refuses otherwise), so an
    identity-owned entry fails the owner test.

    A sticky directory (like /tmp) stays acceptable: with the sticky bit set the
    identity cannot rename or remove an entry it does not own. Symlink mode
    bits carry no access meaning on Linux (symlink(7)).
    """
    if info.st_uid not in (0, controller_uid):
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    return not (info.st_mode & 0o022 and not (stat.S_ISDIR(info.st_mode) and info.st_mode & stat.S_ISVTX))


def _copy_trusted(source, target):
    """Copy the trusted Codex home: regular files and directories only."""
    info = os.lstat(source)
    if not stat.S_ISDIR(info.st_mode):
        raise ReviewAbort("Trusted Codex home is not a directory")
    target.mkdir(mode=0o700)
    for entry in sorted(os.scandir(source), key=lambda e: e.name):
        mode = entry.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            _copy_trusted(Path(entry.path), target / entry.name)
        elif stat.S_ISREG(mode):
            shutil.copyfile(entry.path, target / entry.name, follow_symlinks=False)
        else:
            raise ReviewAbort("Trusted Codex home contains an unsupported entry")
