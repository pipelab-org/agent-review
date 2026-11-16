# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Live review counters for the existing per-run notice.

Inference writes fixed diagnostic files only. A separate observer, holding the
notice's write permission, applies them. Job logs are not a live channel: the
log download returns 404 while the job is still running
(https://github.com/orgs/community/discussions/75518). Actions artifacts v4 are
listed and downloaded as soon as they are uploaded, including before the run
completes (https://github.blog/changelog/2023-12-14-github-actions-artifacts-v4-is-now-generally-available/).

List: GET /repos/{owner}/{repo}/actions/runs/{run_id}/artifacts with Actions
read (https://docs.github.com/en/rest/actions/artifacts#list-workflow-run-artifacts).
The zip redirect expires in one minute and the Authorization header is not
forwarded to that host. Each job can create at most 500 artifacts
(https://github.com/actions/upload-artifact/blob/ea165f8d65b6e75b540449e92b4886f43607fa02/README.md).
This job already uploads findings and the outcome record, so progress uploads
stop at 498. Later counters remain in the step log and the job summary.

GITHUB_TOKEN normally allows 1,000 REST requests an hour per repository
(https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api).
The observer waits 15 seconds between polls. Writes keep the existing
one-second pause. Counters use the signed 64-bit digit width already used for
section notes. Nothing here certifies coverage.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import zipfile

try:
    from . import review_sections as sections
except ImportError:
    import review_sections as sections


SCHEMA = "agent-review-progress:v1"
NOTICE = "<!-- agent-review-notice:v1 run={run} -->"
TERMINAL = "<!-- agent-review-notice-terminal:v1 run={run} state={state} -->"
# 500 artifact creations per job, minus the findings and outcome uploads.
MAX_UPLOADS = 498
# Poll cadence is diagnostic only; each request is also paced from rate headers.
# Without headers, 3600/1000 seconds follows GitHub's normal GITHUB_TOKEN limit.
POLL_SECONDS = 15
# Existing comment byte ceiling. A progress record is a counter, not a document.
MAX_RECORD_BYTES = 65536
KINDS = {"section", "consolidation", "threads"}
STATES = {"started", "processed"}
TERMINAL_MARKS = (
    "<!-- agent-review-status:v1",
    "Treat this run as incomplete",
    "Review run cancelled",
    "<!-- agent-review-notice-terminal:v1",
)
NAME = re.compile(r"agent-review-progress-([1-9][0-9]*)-([1-9][0-9]*)-([1-9][0-9]*)$")


def bounded(value):
    """A positive integer that fits the existing section-note digit width. Booleans are not integers here."""
    return type(value) is int and 0 < value < 10 ** sections.DIGITS


def binding_from_env(env=None):
    env = os.environ if env is None else env
    repository, number, head = env.get("GITHUB_REPOSITORY", ""), env.get("PR_NUMBER", ""), env.get("REVIEW_HEAD", "")
    run, attempt = env.get("GITHUB_RUN_ID", ""), env.get("GITHUB_RUN_ATTEMPT", "")
    if not (re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
            and re.fullmatch(r"[1-9][0-9]*", str(number))
            and re.fullmatch(r"[0-9a-f]{40}", head)
            and re.fullmatch(r"[1-9][0-9]*", run) and attempt == "1"):
        return None
    return {"repository": repository, "number": int(number), "head": head,
            "run_id": run, "run_attempt": "1"}


def _counters(record):
    kind, state = record.get("kind"), record.get("state")
    if not isinstance(kind, str) or not isinstance(state, str) or kind not in KINDS or state not in STATES or record.get("publication") != "pending":
        return False
    if kind == "threads":
        return set(record) == {"schema", "repository", "number", "head", "run_id", "run_attempt",
                               "seq", "kind", "state", "publication"}
    if not bounded(record.get("pass")) or not bounded(record.get("passes")) or record["pass"] > record["passes"]:
        return False
    if kind == "consolidation":
        return (bounded(record.get("sections")) and record["sections"] > 1
                and set(record) == {"schema", "repository", "number", "head", "run_id", "run_attempt",
                                    "seq", "pass", "passes", "kind", "sections", "state", "publication"})
    return (bounded(record.get("section")) and bounded(record.get("sections"))
            and record["section"] <= record["sections"]
            and set(record) == {"schema", "repository", "number", "head", "run_id", "run_attempt",
                                "seq", "pass", "passes", "kind", "section", "sections", "state", "publication"})


def validate_record(payload, binding):
    """Return the record when it is bound to this run, or None. Never raises."""
    try:
        record = json.loads(payload) if isinstance(payload, str) else payload
    except (ValueError, UnicodeError, TypeError):
        return None
    if not isinstance(record, dict) or binding is None or record.get("schema") != SCHEMA:
        return None
    if type(record.get("number")) is not int:
        return None
    if any(record.get(key) != binding.get(key) for key in ("repository", "number", "head", "run_id", "run_attempt")):
        return None
    if not bounded(record.get("seq")) or not _counters(record):
        return None
    return record


def is_terminal(body):
    return isinstance(body, str) and any(mark in body for mark in TERMINAL_MARKS)


def notice_text(record):
    """Fixed counter text. Publication pending is explicit. Coverage is not claimed."""
    run, repo = record["run_id"], record["repository"]
    if record["kind"] == "threads":
        stage = "Checking prior threads"
    elif record["kind"] == "consolidation":
        stage = f"Pass {record['pass']}/{record['passes']}, cross-section consolidation: {record['state']}"
    else:
        stage = (f"Pass {record['pass']}/{record['passes']}, "
                 f"section {record['section']}/{record['sections']}: {record['state']}")
    return (f"{NOTICE.format(run=run)}\n"
            f"Review in progress. {stage}. Publication pending. "
            "This comment does not establish completed coverage.\n\n"
            f"[Actions run](https://github.com/{repo}/actions/runs/{run}).")


def decide(body, record, applied_seq):
    """Patch the open notice, or skip. Progress never creates or replaces a terminal notice."""
    if record is None or is_terminal(body) or not isinstance(applied_seq, int):
        return "skip"
    if record["seq"] <= applied_seq:
        return "skip"
    marker = NOTICE.format(run=record["run_id"])
    if not isinstance(body, str) or not body.startswith(marker + "\n"):
        return "skip"
    return "patch"


def write_progress(seq, *, kind, state, pass_index=None, passes=None, section=None, sections_count=None, env=None):
    """Best-effort diagnostic file. A missing directory or a failed write does not affect inference."""
    try:
        binding = binding_from_env(env)
        directory = (env or os.environ).get("REVIEW_PROGRESS_DIR")
        if binding is None or not directory:
            return False
        record = {**binding, "schema": SCHEMA, "seq": seq, "kind": kind, "state": state, "publication": "pending"}
        if kind == "section":
            record.update({"pass": pass_index, "passes": passes, "section": section, "sections": sections_count})
        elif kind == "consolidation":
            record.update({"pass": pass_index, "passes": passes, "sections": sections_count})
        if validate_record(record, binding) is None:
            return False
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        target = path / f"{seq}.json"
        temporary = path / f".{seq}.json.partial"
        temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        temporary.chmod(0o644)
        os.replace(temporary, target)
        return True
    except (OSError, ValueError, TypeError):
        return False


def artifact_name(run, attempt, seq):
    return f"agent-review-progress-{run}-{attempt}-{seq}"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ProgressUnavailable(RuntimeError):
    """Diagnostics yield rather than delaying a terminal writer."""


class RateOpener:
    """Use GitHub's published remaining/reset and Retry-After headers.

    No response bodies or credentials are logged. The observer is best effort;
    Rate exhaustion or an overlong wait disables diagnostics without holding publication.
    """
    def __init__(self, opener, sleep, clock=time.time):
        self.opener, self.sleep, self.clock = opener, sleep, clock
        self.remaining, self.reset, self.last, self.retry = None, None, None, 0
        self.deadline = None
        self.before_request, self.checking = None, False

    def checkpoint(self):
        if self.before_request is not None and not self.checking:
            self.checking = True
            try:
                self.before_request()
            finally:
                self.checking = False

    def begin_request(self):
        # Keep the existing bound per request, including pacing and redirects.
        self.deadline = time.monotonic() + POLL_SECONDS

    def read_timeout(self, maximum=30):
        left = POLL_SECONDS if self.deadline is None else self.deadline - time.monotonic()
        if left <= 0:
            raise ProgressUnavailable('Progress poll deadline reached')
        return min(maximum, left)

    def _headers(self, headers):
        headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        def number(key):
            raw = headers.get(key, '')
            return int(raw) if re.fullmatch(r'[0-9]{1,12}', raw) else None
        remaining, reset = number('x-ratelimit-remaining'), number('x-ratelimit-reset')
        if remaining is not None and reset is not None:
            self.remaining, self.reset = remaining, reset
        retry = number('retry-after')
        if retry is not None:
            self.retry = self.clock() + retry

    def open(self, request, timeout=30):
        self.checkpoint()
        self.begin_request()
        deadline = self.deadline
        while True:
            now = self.clock()
            spacing = 3600 / 1000
            if self.remaining is not None and self.reset is not None and self.reset > now:
                spacing = (self.reset - now + 1) / max(self.remaining, 1)
            wait = max(self.retry - now, 0 if self.last is None else self.last + spacing - now, 0)
            left = deadline - time.monotonic()
            if self.remaining == 0 or wait >= left or left <= 0:
                raise ProgressUnavailable('Progress yields to terminal publication')
            if wait > 0:
                last = self.last
                self.sleep(wait)
                self.checkpoint()
                self.deadline = deadline
                # A job check used the same token: pace again after that read.
                if self.last != last:
                    continue
            break
        left = deadline - time.monotonic()
        if left <= 0:
            raise ProgressUnavailable('Progress request deadline reached')
        self.last = self.clock()
        try:
            response = self.opener.open(request, timeout=min(timeout, left))
        except urllib.error.HTTPError as error:
            self._headers(error.headers)
            if error.code in (403, 429):
                raise ProgressUnavailable('Progress API unavailable') from None
            raise
        self._headers(getattr(response, 'headers', {}))
        return response


def _collect(opener, token, url, key=None):
    """Read all REST pages; a run can contain artifacts from more than one job."""
    rows, page = [], 1
    while True:
        payload = _api(opener, token, 'GET', url + f'&page={page}')
        batch = payload.get(key) if key and isinstance(payload, dict) else payload
        if not isinstance(batch, list):
            raise ValueError('Invalid diagnostic list')
        rows.extend(batch)
        if len(batch) < 100:
            return rows
        page += 1


def _api(opener, token, method, url):
    request = urllib.request.Request(url, method=method, headers={
        "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"})
    with opener.open(request, timeout=30) as response:
        return json.load(response)


def _download(token, url, *, opener=None):
    """Download an artifact zip. Drop Authorization before following the one-minute redirect."""
    opener = opener or urllib.request.build_opener(NoRedirect)
    request = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"})
    try:
        with opener.open(request, timeout=30) as response:
            return response.read(MAX_RECORD_BYTES + 1)
    except urllib.error.HTTPError as error:
        location = error.headers.get("Location") if error.code in (301, 302, 303, 307, 308) else None
        if not location or not location.startswith("https://"):
            raise
        timeout = opener.read_timeout() if isinstance(opener, RateOpener) else 30
        with urllib.request.build_opener(NoRedirect).open(urllib.request.Request(location), timeout=timeout) as response:
            return response.read(MAX_RECORD_BYTES + 1)


def _record_from_zip(blob):
    if len(blob) > MAX_RECORD_BYTES:
        return None
    try:
        import io
        archive = zipfile.ZipFile(io.BytesIO(blob))
        entries = archive.infolist()
        if (len(entries) != 1 or entries[0].filename != "progress.json"
                or entries[0].file_size > MAX_RECORD_BYTES or entries[0].flag_bits & 1):
            return None
        payload = archive.read("progress.json")
    except (zipfile.BadZipFile, ValueError, RuntimeError, KeyError):
        return None
    if len(payload) > MAX_RECORD_BYTES:
        return None
    try:
        return payload.decode("utf-8")
    except UnicodeError:
        return None


def upload_command(action_dir, name, path):
    """Invoke the pinned upload-artifact action. The public REST API has no upload method."""
    return ["node", str(Path(action_dir) / "dist" / "upload" / "index.js")], {
        "INPUT_NAME": name, "INPUT_PATH": str(path), "INPUT_IF-NO-FILES-FOUND": "error",
        "INPUT_OVERWRITE": "false", "INPUT_COMPRESSION-LEVEL": "6", "INPUT_INCLUDE-HIDDEN-FILES": "false"}


def serve_uploads(directory, action_dir, *, limit=MAX_UPLOADS, env=None, run=None, stop=None):
    """Upload each new record once. Failures and a missing action are ignored."""
    binding = binding_from_env(env)
    if binding is None or not Path(action_dir, "dist", "upload", "index.js").is_file():
        return 0
    sent, seen, highest = 0, set(), 0
    runner = run or __import__("subprocess").run
    while sent < limit and (stop is None or not stop()):
        try:
            files = sorted((item for item in Path(directory).glob("*.json") if re.fullmatch(r"[1-9][0-9]*", item.stem)), key=lambda item: int(item.stem), reverse=True)
        except (OSError, ValueError):
            files = []
        for item in files:
            if item.name in seen or sent >= limit or int(item.stem) <= highest:
                continue
            seen.add(item.name)
            try:
                import stat
                descriptor = os.open(item, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, 'rb') as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        continue
                    raw = stream.read(MAX_RECORD_BYTES + 1)
                if len(raw) > MAX_RECORD_BYTES:
                    continue
                record = validate_record(raw.decode('utf-8'), binding)
                if record is None:
                    continue
                import tempfile
                with tempfile.TemporaryDirectory() as temporary:
                    target = Path(temporary) / "progress.json"
                    target.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
                    command, inputs = upload_command(action_dir, artifact_name(
                        binding["run_id"], binding["run_attempt"], record["seq"]), temporary)
                    runner(command, env={**(env if env is not None else os.environ), **inputs}, check=False,
                           stdout=__import__("subprocess").DEVNULL, stderr=__import__("subprocess").DEVNULL)
                sent += 1
                highest = record["seq"]
                break  # Older pending counters are superseded by this newest record.
            except (OSError, ValueError, TypeError, UnicodeError):
                try:
                    print("Progress artifact upload unavailable", flush=True)
                except OSError:
                    pass
        if stop is not None and stop():
            break
        time.sleep(0.5)
    return sent


def _job_done(jobs, name='Review admitted diff'):
    # Reusable job names carry caller prefixes. A skipped acknowledge invocation
    # must never be mistaken for the actual review. Observed hosted producer:
    # `acknowledge / Review admitted diff`, `review / Review admitted diff`.
    matched = [job for job in jobs if isinstance(job, dict) and isinstance(job.get('name'), str)
               and (job['name'] == 'review / ' + name or job['name'].endswith(' / review / ' + name))]
    matched = [job for job in matched if job.get('conclusion') != 'skipped']
    return bool(matched) and all(job.get('status') == 'completed' for job in matched)


def observe(env=None, *, opener=None, sleep=None, iterations=None):
    """Apply only the newest unseen counter to the existing open notice."""
    try:
        env = os.environ if env is None else env
        binding = binding_from_env(env)
        token = env.get('GH_TOKEN') or env.get('GITHUB_TOKEN')
        if binding is None or not token:
            return 0
        sleep = time.sleep if sleep is None else sleep
        opener = RateOpener(opener or urllib.request.build_opener(NoRedirect), sleep)
        repo, run = binding['repository'], binding['run_id']
        prefix = f'https://api.github.com/repos/{repo}'
        applied, quiet, steps, notice_id, rejected = 0, False, 0, None, set()
        next_job_check = [0.0]
        def check_active(*, force=False):
            if not force and time.monotonic() < next_job_check[0]:
                return
            # Recursion-safe: checkpoint reads use this same paced opener.
            checking = opener.checking
            opener.checking = True
            try:
                current = _collect(opener, token, f'{prefix}/actions/runs/{run}/jobs?per_page=100', 'jobs')
            finally:
                opener.checking = checking
            next_job_check[0] = time.monotonic() + POLL_SECONDS
            if _job_done(current):
                raise ProgressUnavailable('Review job finished')
        while iterations is None or steps < iterations:
            steps += 1
            try:
                opener.before_request = None
                jobs = _collect(opener, token, f'{prefix}/actions/runs/{run}/jobs?per_page=100', 'jobs')
                if _job_done(jobs):
                    break
                next_job_check[0] = time.monotonic() + POLL_SECONDS
                opener.before_request = check_active
                if notice_id is None:
                    comments = _collect(opener, token, f"{prefix}/issues/{binding['number']}/comments?per_page=100")
                    matches = [item for item in comments if _notice(item, run)]
                    notice = matches[0] if len(matches) == 1 else None
                    if notice is not None:
                        notice_id = notice['id']
                else:
                    notice = None
                if notice is not None and is_terminal(notice.get('body')):
                    break
                listed = _collect(opener, token, f'{prefix}/actions/runs/{run}/artifacts?per_page=100', 'artifacts')
                candidates = []
                for artifact in listed:
                    if not isinstance(artifact, dict):
                        continue
                    match = NAME.fullmatch(artifact.get('name') or '')
                    identity = artifact.get('id')
                    if (not match or match[1] != run or match[2] != '1' or int(match[3]) <= applied
                            or type(identity) is not int or identity <= 0 or identity in rejected
                            or artifact.get('expired') or not bounded(artifact.get('size_in_bytes'))
                            or artifact['size_in_bytes'] > MAX_RECORD_BYTES):
                        continue
                    candidates.append((int(match[3]), identity))
                if candidates and notice is None and notice_id is not None:
                    notice = _api(opener, token, 'GET', f'{prefix}/issues/comments/{notice_id}')
                    if not _notice(notice, run) or is_terminal(notice.get('body')):
                        break
                for seq, identity in sorted(candidates, reverse=True):
                    try:
                        blob = _download(token, f'{prefix}/actions/artifacts/{identity}/zip', opener=opener)
                        payload = _record_from_zip(blob)
                        record = validate_record(payload, binding) if payload else None
                        if record is None or record['seq'] != seq:
                            rejected.add(identity)
                            continue
                        if notice is not None and decide(notice.get('body'), record, applied) == 'patch':
                            check_active(force=True)
                            fresh = _api(opener, token, 'GET', f"{prefix}/issues/comments/{notice['id']}")
                            if not _notice(fresh, run) or is_terminal(fresh.get('body')):
                                raise ProgressUnavailable('Progress notice is terminal or changed')
                            body = notice_text(record)
                            updated = _patch(opener, token, f"{prefix}/issues/comments/{notice['id']}", body)
                            if isinstance(updated, dict) and updated.get('body') == body:
                                applied = seq
                        break
                    except (urllib.error.URLError, ValueError, KeyError, TypeError):
                        continue
                review_done = _job_done(jobs)
                absent = not any(isinstance(job, dict) and isinstance(job.get('name'), str)
                                 and (job['name'] == 'review / Review admitted diff'
                                      or job['name'].endswith(' / review / Review admitted diff'))
                                 and job.get('conclusion') != 'skipped' for job in jobs)
                if (review_done or absent) and quiet:
                    break
                quiet = review_done or absent
            except ProgressUnavailable:
                break
            except (urllib.error.URLError, ValueError, KeyError, TypeError):
                pass
            if iterations is None or steps < iterations:
                sleep(POLL_SECONDS)
    except Exception:
        try:
            print('Review progress notice unavailable; inference continues', flush=True)
        except OSError:
            pass
    return 0


def _notice(item, run):
    if not isinstance(item, dict) or type(item.get('id')) is not int or item['id'] <= 0:
        return False
    author, body = item.get('user'), item.get('body')
    return (isinstance(author, dict) and author.get('login') == 'github-actions[bot]'
            and author.get('id') == 41898282 and author.get('type') == 'Bot'
            and isinstance(body, str) and body.startswith(NOTICE.format(run=run) + '\n'))


def _patch(opener, token, url, body):
    request = urllib.request.Request(url, method="PATCH", data=json.dumps({"body": body}).encode(), headers={
        "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
        "Content-Type": "application/json"})
    with opener.open(request, timeout=30) as response:
        return json.load(response)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("upload",))
    parser.add_argument("--dir", dest="directory", required=True)
    parser.add_argument("--action", required=True)
    args = parser.parse_args()
    if args.mode == "upload":
        serve_uploads(args.directory, args.action, stop=lambda: not os.path.exists(f"/proc/{os.getppid()}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
