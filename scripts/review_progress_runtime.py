# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Credential-isolated, runner-managed live progress uploader.

The Node action execs this bootstrap. Before inference can start, bootstrap
sends the artifact credential over stdin to a third UID, waits for its ACK,
then execs the same PID into a credential-free supervisor. Only that new
process marks startup safe. Unsetting environ alone does not clear proc(5)'s
initial environment. The model launch paths and their credentials do not change.

GitHub background/cancel owns lifetime. The supervisor holds a pipe writer;
cancel/exit closes it and the worker exits on EOF, killing an in-flight upload.
https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstepsbackground
https://man7.org/linux/man-pages/man5/proc_pid_environ.5.html
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import select
import shutil
import signal
import subprocess
import sys
import threading
import time

RUNTIME = Path('/var/lib/agent-review-progress-runtime')
DATA = Path('/var/lib/agent-review-progress')
PID = DATA / '.runtime-pid'
USER = 'agent-review-upload'
SCRIPT = 'review_progress_runtime.py'
STATE = DATA / '.runtime-state'
TRANSPORT = {'PATH', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy',
             'all_proxy', 'no_proxy', 'SSL_CERT_FILE', 'NODE_EXTRA_CA_CERTS'}
BINDING = {'GITHUB_REPOSITORY', 'PR_NUMBER', 'REVIEW_HEAD', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT'}
RUNTIME_CREDENTIAL = {'ACTIONS_RUNTIME_TOKEN', 'ACTIONS_RESULTS_URL'}


def clean_env(env=None):
    env = os.environ if env is None else env
    return {k: v for k, v in env.items() if k in TRANSPORT | BINDING}


def status(state, worker=None):
    target = STATE.with_suffix('.partial')
    target.write_text(json.dumps(dict(state=state, pid=os.getpid(), worker=worker)))
    target.replace(STATE)


def prepare(source, uploader, node, uid, gid):
    """Host-only provisioning; no credentials or repository-head code involved."""
    if os.geteuid() != 0 or uid == 0 or gid == 0:
        raise ValueError('Progress runtime needs the hosted controller identity')
    if RUNTIME.exists() or DATA.exists():
        raise ValueError('Progress runtime already exists')
    try:
        pwd.getpwnam(USER)
    except KeyError:
        pass
    else:
        raise ValueError('Progress identity already exists')
    subprocess.run(['/usr/sbin/useradd', '--system', '--user-group', '--no-create-home', '--shell',
                    '/usr/sbin/nologin', USER], check=True)
    identity = pwd.getpwnam(USER)
    model = None
    try:
        model = pwd.getpwnam('agent-review-model').pw_uid
    except KeyError:
        pass
    if identity.pw_uid in (0, uid, model) or identity.pw_gid == 0 or os.getgrouplist(USER, identity.pw_gid) != [identity.pw_gid]:
        raise ValueError('Progress identity must be separate and unprivileged')
    RUNTIME.mkdir(mode=0o755)
    for name in (SCRIPT, 'review_progress.py', 'review_sections.py', 'review_errors.py', 'review_reliability.py'):
        shutil.copyfile(Path(source) / name, RUNTIME / name)
        (RUNTIME / name).chmod(0o644)
    shutil.copytree(Path(uploader) / 'dist', RUNTIME / 'dist')
    shutil.copyfile(Path(node).resolve(), RUNTIME / 'node')
    (RUNTIME / 'node').chmod(0o755)
    private = RUNTIME / 'uploads'
    private.mkdir(mode=0o700)
    os.chown(private, identity.pw_uid, identity.pw_gid)
    DATA.mkdir(mode=0o755)
    status('pending')
    os.chown(DATA, uid, gid)
    os.chown(STATE, uid, gid)


def supervisor(fd, worker, state):
    """This exec's initial environment contains neither credential nor provider key."""
    if any(name in os.environ for name in RUNTIME_CREDENTIAL | {'GITHUB_TOKEN', 'GH_TOKEN', 'ANTHROPIC_API_KEY'}):
        raise ValueError('Supervisor credential refusal')
    status(state, worker)
    if fd < 0:
        return 0
    def stop(_signum, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while True:
            # The uploader is a direct child retained across exec. Reap it if it
            # stops early, so an unavailable upload cannot hold the background step.
            found, _ = os.waitpid(worker, os.WNOHANG)
            if found:
                return 0
            time.sleep(0.5)
    finally:
        os.close(fd)
        try:
            os.waitpid(worker, 0)
        except ChildProcessError:
            pass


def record_pid():
    """Same pid across the action's exec. Await uses it only after this process is gone."""
    PID.write_text(str(os.getpid()))


def bootstrap():
    """Never signal readiness until exec removed the initial secret environment."""
    record_pid()
    child, fd = None, -1
    try:
        credentials = {k: os.environ[k] for k in RUNTIME_CREDENTIAL}
        identity = pwd.getpwnam(USER)
        if identity.pw_uid in (0, os.getuid()) or identity.pw_gid == 0:
            raise ValueError('Uploader identity refusal')
        command = ['/usr/bin/sudo', '-n', '--', '/usr/bin/setpriv', '--reuid', str(identity.pw_uid),
                   '--regid', str(identity.pw_gid), '--clear-groups', '--no-new-privs',
                   '--bounding-set=-all', '--inh-caps=-all', '--ambient-caps=-all',
                   '/usr/bin/python3', str(RUNTIME / SCRIPT), 'worker']
        child = subprocess.Popen(command, env=clean_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL)
        child.stdin.write((json.dumps({**clean_env(), **credentials}) + '\n').encode())
        child.stdin.flush()
        # The worker consumes the secret before inference is allowed to read any
        # same-UID pipe descriptors. Its stdout carries only this fixed ACK.
        if not select.select([child.stdout], [], [], 30)[0] or child.stdout.readline() != b'ready\n':
            raise ValueError('Uploader startup unavailable')
        child.stdout.close()
        fd = child.stdin.fileno()
        os.set_inheritable(fd, True)
        state = 'ready'
    except Exception:
        if child is not None and child.stdin:
            child.stdin.close()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # The pipe is closed. A still-running child is the other uid, not
                # this process. Exec below drops the runner-visible credential.
                pass
        state = 'unavailable'
        fd = -1
    # os.execve, not an environment clear: proc retains the old initial block
    # until this replaces the process. No secret is in argv or an output file.
    os.execve('/usr/bin/python3', ['/usr/bin/python3', str(RUNTIME / SCRIPT), 'supervisor',
                                 '--fd', str(fd), '--worker', str(child.pid if child else -1),
                                 '--state', state], clean_env())


def _startup_exited():
    """True only when the recorded action pid is gone or is no longer that action.

    A missing pid file is not enough: the background step may not have started,
    and treating that as finished would admit inference while the credentialed
    process is still about to start.
    """
    try:
        pid = int(PID.read_text())
    except (OSError, ValueError):
        return False
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return True
    return b'review_progress_runtime.py' not in command and b'review-progress' not in command


def await_ready():
    """Inference startup barrier. An unknown live startup never authorizes a model."""
    while True:
        try:
            value = json.loads(STATE.read_text())
            if value.get('state') in ('ready', 'unavailable'):
                # The sentinel is written only by the clean supervisor, which
                # has already exec'd; failure is safe too, without an uploader.
                return 0
        except (OSError, ValueError):
            pass
        if _startup_exited():
            return 0
        time.sleep(0.1)


def worker():
    """Third-UID upload worker. EOF stops the loop and any active Node upload."""
    import review_progress as progress
    if os.getuid() != pwd.getpwnam(USER).pw_uid or os.getuid() == 0:
        raise ValueError('Uploader must not run as controller')
    config = json.loads(sys.stdin.buffer.readline(progress.MAX_RECORD_BYTES + 1))
    if not isinstance(config, dict) or set(config) - TRANSPORT - BINDING - RUNTIME_CREDENTIAL:
        raise ValueError('Invalid progress runtime configuration')
    if not all(isinstance(config.get(k), str) and config[k] for k in RUNTIME_CREDENTIAL):
        raise ValueError('Missing progress runtime credential')
    stop = threading.Event()
    def control():
        while os.read(sys.stdin.fileno(), 1):
            pass
        stop.set()
    threading.Thread(target=control, daemon=True).start()
    print('ready', flush=True)
    def upload(command, env, **_kwargs):
        # Only this UID holds the runtime credential. Output and action summary
        # files are private to it; no toolkit command reaches the runner log.
        upload_env = {**config, **{k: v for k, v in env.items() if k.startswith('INPUT_')},
                      'TMPDIR': str(RUNTIME / 'uploads'),
                      'GITHUB_OUTPUT': str(RUNTIME / 'uploads' / 'output'),
                      'GITHUB_STEP_SUMMARY': str(RUNTIME / 'uploads' / 'summary')}
        command[0] = str(RUNTIME / 'node')
        child = subprocess.Popen(command, env=upload_env, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
        while child.poll() is None:
            if stop.wait(0.1):
                os.killpg(child.pid, signal.SIGKILL)
                break
        return subprocess.CompletedProcess(command, child.wait())
    return progress.serve_uploads(DATA, RUNTIME, env=config, run=upload, stop=stop.is_set)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'bootstrap', 'supervisor', 'await-ready', 'worker'))
    parser.add_argument('--source')
    parser.add_argument('--uploader')
    parser.add_argument('--node')
    parser.add_argument('--uid', type=int)
    parser.add_argument('--gid', type=int)
    parser.add_argument('--fd', type=int, default=-1)
    parser.add_argument('--worker', type=int, default=-1)
    parser.add_argument('--state', choices=('ready', 'unavailable'), default='unavailable')
    args = parser.parse_args()
    if args.mode == 'prepare':
        if os.environ.get('GITHUB_ACTIONS') != 'true' or os.environ.get('RUNNER_ENVIRONMENT') != 'github-hosted':
            raise SystemExit('Progress provisioning runs only on a GitHub-hosted runner')
        prepare(args.source, args.uploader, args.node, args.uid, args.gid)
    elif args.mode == 'bootstrap':
        bootstrap()
    elif args.mode == 'supervisor':
        return supervisor(args.fd, args.worker, args.state)
    elif args.mode == 'await-ready':
        return await_ready()
    else:
        worker()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        raise SystemExit('Progress runtime unavailable') from None
