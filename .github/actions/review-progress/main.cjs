// Copyright 2026 Pipelock contributors
// SPDX-License-Identifier: Apache-2.0
// Node action handlers receive the artifact runtime credential; shell steps do not.
// exec keeps the runner-tracked PID. The Python bootstrap re-execs once more
// without credentials before signaling the inference startup barrier.
// https://nodejs.org/docs/latest-v24.x/api/process.html#processexecvefile-args-env
const fs = require('fs');
try {
  fs.writeFileSync('/var/lib/agent-review-progress/.runtime-pid', String(process.pid));
} catch (error) {
  // Bootstrap records the same pid when this directory is writable. A failed
  // write must still reach exec; leaving this process alive keeps the credential.
}
process.execve('/usr/bin/python3', ['/usr/bin/python3',
  '/var/lib/agent-review-progress-runtime/review_progress_runtime.py', 'bootstrap'], process.env);
