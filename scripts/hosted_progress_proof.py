# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Hosted artifact round trip and cancel proof, without a model or PR write."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request

try:
    from . import review_progress as progress
    from . import review_progress_runtime as runtime
except ImportError:
    import review_progress as progress
    import review_progress_runtime as runtime


def probe():
    runtime.await_ready()
    state = json.loads(runtime.STATE.read_text())
    if state['state'] != 'ready':
        raise SystemExit('Hosted uploader startup failed')
    initial = Path(f"/proc/{state['pid']}/environ").read_bytes()
    if any(name.encode() + b'=' in initial for name in runtime.RUNTIME_CREDENTIAL):
        raise SystemExit('Supervisor retained a runtime credential')
    if not progress.write_progress(1, kind='section', state='processed', pass_index=1, passes=1,
                                   section=1, sections_count=1):
        raise SystemExit('Synthetic counter was not written')
    binding = progress.binding_from_env()
    name = progress.artifact_name(binding['run_id'], '1', 1)
    prefix = f"https://api.github.com/repos/{binding['repository']}"
    opener = progress.RateOpener(urllib.request.build_opener(progress.NoRedirect), time.sleep)
    token = os.environ['GH_TOKEN']
    # Existing REST request ceiling (30 s) bounds fixture polling, twice so one
    # upload request can finish. This is a CI fixture deadline, not review policy.
    deadline = time.monotonic() + 2 * 30
    while time.monotonic() < deadline:
        listed = progress._collect(opener, token,
            f"{prefix}/actions/runs/{binding['run_id']}/artifacts?per_page=100", 'artifacts')
        for artifact in listed:
            if artifact.get('name') != name:
                continue
            blob = progress._download(token, f"{prefix}/actions/artifacts/{artifact['id']}/zip", opener=opener)
            record = progress.validate_record(progress._record_from_zip(blob), binding)
            if record is None or record['seq'] != 1:
                raise SystemExit('Hosted artifact counter mismatch')
            print('Hosted progress artifact round trip verified; no provider or PR write')
            print('Hosted ptrace_scope:', Path('/proc/sys/kernel/yama/ptrace_scope').read_text().strip())
            return
        time.sleep(progress.POLL_SECONDS)
    raise SystemExit('Hosted progress artifact did not arrive')


def stopped():
    state = json.loads(runtime.STATE.read_text())
    for pid in (state['pid'], state['worker']):
        if type(pid) is int and pid > 0 and Path(f'/proc/{pid}').exists():
            raise SystemExit('Progress process survived runner cancellation')
    print('Runner cancel stopped the supervisor and uploader')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('probe', 'stopped'))
    args = parser.parse_args()
    (probe if args.mode == 'probe' else stopped)()
