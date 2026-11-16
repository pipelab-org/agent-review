# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Real proc/pipe checks with synthetic credentials and an existing nobody UID."""
import json
import os
from pathlib import Path
import pwd
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import review_progress as progress
from scripts import review_progress_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def test_cross_uid_worker_clean_exec_and_cancel_with_real_processes(self):
        if os.name != 'posix' or not Path('/proc/self/environ').exists() or not shutil.which('sudo'):
            self.skipTest('requires Linux proc and sudo (hosted runtime platform)')
        if subprocess.run(['sudo', '-n', '/usr/bin/true'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            self.skipTest('requires passwordless sudo, as provided by the hosted runner')
        node = shutil.which('node')
        self.assertIsNotNone(node)
        self.assertNotEqual(os.getuid(), pwd.getpwnam('nobody').pw_uid)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o755)
            data = root / 'data'
            data.mkdir(mode=0o755)
            upload = root / 'uploads'
            upload.mkdir(mode=0o700)
            # This test never provisions a user or changes the protected prefix.
            # Only its own disposable directory is assigned to the existing nobody.
            subprocess.run(['sudo', '-n', 'chown', 'nobody', '--', str(upload)], check=True)
            script = Path(runtime.__file__).read_text().replace(str(runtime.RUNTIME), str(root)).replace(
                str(runtime.DATA), str(data)).replace("USER = 'agent-review-upload'", "USER = 'nobody'")
            (root / runtime.SCRIPT).write_text(script)
            for name in ('review_progress.py', 'review_sections.py', 'review_errors.py', 'review_reliability.py'):
                shutil.copyfile(Path(progress.__file__).parent / name, root / name)
            (root / 'dist/upload').mkdir(parents=True)
            shutil.copyfile(node, root / 'node')
            (root / 'node').chmod(0o755)
            ack = upload / 'ack'
            # A live Node process receives toolkit inputs and a synthetic token.
            # It reports booleans/PID only, never the credential's value.
            (root / 'dist/upload/index.js').write_text("const fs=require('fs');" +
                "fs.writeFileSync(" + json.dumps(str(ack)) + ", JSON.stringify({pid:process.pid,uid:process.getuid()," +
                "credential:process.env.ACTIONS_RUNTIME_TOKEN==='synthetic-only'," +
                "inputs:process.env.INPUT_NAME==='agent-review-progress-99-1-1' && !!process.env.INPUT_PATH}));" +
                "setInterval(()=>{},1000);")
            env = {**runtime.clean_env(), 'ACTIONS_RUNTIME_TOKEN': 'synthetic-only',
                   'ACTIONS_RESULTS_URL': 'https://example.invalid', 'GITHUB_REPOSITORY': 'example/project',
                   'PR_NUMBER': '7', 'REVIEW_HEAD': 'a' * 40, 'GITHUB_RUN_ID': '99', 'GITHUB_RUN_ATTEMPT': '1'}
            child = subprocess.Popen(['/usr/bin/python3', str(root / runtime.SCRIPT), 'bootstrap'], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            state = data / runtime.STATE.name
            node_pid = None
            try:
                deadline = time.monotonic() + 30
                while not state.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(state.exists(), child.stderr.read() if child.poll() is not None else 'startup timeout')
                details = json.loads(state.read_text())
                self.assertEqual('ready', details['state'])
                self.assertEqual(child.pid, details['pid'], 'exec must retain the runner-tracked PID')
                self.assertNotIn(b'ACTIONS_RUNTIME_TOKEN=', Path(f'/proc/{child.pid}/environ').read_bytes())
                self.assertTrue(progress.write_progress(1, kind='section', state='started', pass_index=1,
                                                       passes=1, section=1, sections_count=1,
                                                       env={**env, 'REVIEW_PROGRESS_DIR': str(data)}))
                metadata = None
                while time.monotonic() < deadline:
                    read = subprocess.run(['sudo', '-n', 'cat', '--', str(ack)], capture_output=True)
                    if read.returncode == 0:
                        metadata = json.loads(read.stdout)
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(metadata, 'isolated uploader did not start')
                node_pid = metadata['pid']
                self.assertEqual(pwd.getpwnam('nobody').pw_uid, metadata['uid'])
                self.assertTrue(metadata['credential'])
                self.assertTrue(metadata['inputs'])
                with self.assertRaises(PermissionError):
                    Path(f'/proc/{node_pid}/environ').read_bytes()
                child.send_signal(signal.SIGTERM)
                self.assertEqual(0, child.wait(timeout=5))
                while Path(f'/proc/{node_pid}').exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(Path(f'/proc/{node_pid}').exists(), 'pipe EOF must stop an in-flight upload')
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)
                child.stdout.close()
                child.stderr.close()
                if node_pid and Path(f'/proc/{node_pid}').exists():
                    subprocess.run(['sudo', '-n', 'kill', '-KILL', '--', str(node_pid)], check=False)
                subprocess.run(['sudo', '-n', 'chown', '-R', f'{os.getuid()}:{os.getgid()}', '--', str(upload)], check=True)

    def test_clean_environment_drops_all_runtime_and_provider_credentials(self):
        names = {'ACTIONS_RUNTIME_TOKEN', 'ACTIONS_RESULTS_URL', 'GITHUB_TOKEN', 'GH_TOKEN',
                 'ANTHROPIC_API_KEY', 'ANTHROPIC_API_KEY_2', 'OPENAI_API_KEY', 'NODE_OPTIONS'}
        cleaned = runtime.clean_env({**{name: 'synthetic' for name in names}, 'HTTPS_PROXY': 'proxy', 'PATH': '/usr/bin'})
        self.assertEqual({'HTTPS_PROXY': 'proxy', 'PATH': '/usr/bin'}, cleaned)


class StartupBarrierTests(unittest.TestCase):
    def test_recorded_startup_death_releases_pending_barrier(self):
        repo = Path(__file__).resolve().parents[1]
        script = r'''
import json, sys
from pathlib import Path
from scripts import review_progress_runtime as runtime
root = Path(sys.argv[1])
runtime.DATA = root
runtime.PID = root / ".runtime-pid"
runtime.STATE = root / ".runtime-state"
runtime.PID.write_text("999999")
runtime.STATE.write_text(json.dumps({"state": "pending", "pid": None, "worker": None}))
runtime.await_ready()
print("returned")
'''
        with tempfile.TemporaryDirectory() as directory:
            dead = subprocess.run([sys.executable, "-c", script, directory], cwd=repo,
                                  capture_output=True, text=True, timeout=2)
        self.assertEqual(0, dead.returncode, dead.stderr)
        self.assertIn("returned", dead.stdout)
        self.assertFalse(Path("/proc/999999").exists())

    def test_unfamiliar_live_credential_holder_and_unknown_liveness_do_not_admit(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime, 'PID', Path(directory) / 'pid'):
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                     env={**runtime.clean_env(), 'ACTIONS_RUNTIME_TOKEN': 'synthetic-only'})
            try:
                runtime.PID.write_text(str(child.pid))
                self.assertIn(b'ACTIONS_RUNTIME_TOKEN=synthetic-only', Path(f'/proc/{child.pid}/environ').read_bytes())
                self.assertFalse(runtime._startup_exited())
                for error in (PermissionError('synthetic denial'), OSError('synthetic unknown'), OverflowError()):
                    with self.subTest(error=type(error).__name__), patch.object(runtime.os, 'kill', side_effect=error):
                        self.assertFalse(runtime._startup_exited())
            finally:
                child.terminate()
                child.wait(timeout=5)
            self.assertTrue(runtime._startup_exited())

    def test_missing_pid_and_live_action_fail_closed_by_the_startup_deadline(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime, 'PID', Path(directory) / 'pid'), \
                patch.object(runtime, 'STATE', Path(directory) / 'state'):
            runtime.STATE.write_text(json.dumps({"state": "pending"}))
            for mode in ('missing', 'live', 'unreadable'):
                with self.subTest(mode=mode):
                    if mode == 'missing':
                        runtime.PID.unlink(missing_ok=True)
                    else:
                        runtime.PID.write_text(str(os.getpid() if mode == 'live' else os.getpid() + 1))
                    with patch.object(runtime.os, 'kill', side_effect=PermissionError('denied')):
                        with self.assertRaisesRegex(TimeoutError, 'Progress startup could not be confirmed'):
                            runtime.await_ready(timeout=0)
            for state in ('ready', 'unavailable'):
                runtime.STATE.write_text(json.dumps({"state": state}))
                self.assertEqual(0, runtime.await_ready(timeout=0))


if __name__ == '__main__':
    unittest.main()
