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

An authenticated GITHUB_TOKEN allows 5,000 REST requests an hour
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
# 15s * 70 minutes is about 280 requests, inside the 5,000/hour primary limit.
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
    if kind not in KINDS or state not in STATES or record.get("publication") != "pending":
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
    if any(record.get(key) != binding[key] for key in ("repository", "number", "head", "run_id", "run_attempt")):
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
        os.replace(temporary, target)
        return True
    except (OSError, ValueError, TypeError):
        return False


def artifact_name(run, attempt, seq):
    return f"agent-review-progress-{run}-{attempt}-{seq}"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _collect(opener, token, url, key):
    """Follow artifact pages. per_page 100 is the REST maximum, and a job can hold 500 artifacts."""
    rows = []
    for page in range(1, 6):
        payload = _api(opener, token, "GET", url + f"&page={page}")
        batch = payload.get(key) or [] if isinstance(payload, dict) else []
        rows.extend(batch)
        if len(batch) < 100:
            break
    return rows


def _api(opener, token, method, url):
    request = urllib.request.Request(url, method=method, headers={
        "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"})
    with opener.open(request, timeout=30) as response:
        return json.load(response)


def _download(token, url):
    """Download an artifact zip. Drop Authorization before following the one-minute redirect."""
    opener = urllib.request.build_opener(NoRedirect)
    request = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"})
    try:
        with opener.open(request, timeout=30) as response:
            return response.read(MAX_RECORD_BYTES + 1)
    except urllib.error.HTTPError as error:
        location = error.headers.get("Location") if error.code in (301, 302, 303, 307, 308) else None
        if not location or not location.startswith("https://"):
            raise
        with urllib.request.urlopen(urllib.request.Request(location), timeout=30) as response:
            return response.read(MAX_RECORD_BYTES + 1)


def _record_from_zip(blob):
    if len(blob) > MAX_RECORD_BYTES:
        return None
    try:
        import io
        archive = zipfile.ZipFile(io.BytesIO(blob))
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if names != ["progress.json"]:
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
    return ["node", str(Path(action_dir) / "dist" / "index.js")], {
        "INPUT_name": name, "INPUT_path": str(path), "INPUT_if-no-files-found": "error",
        "INPUT_overwrite": "false", "INPUT_compression-level": "6", "INPUT_include-hidden-files": "false"}


def serve_uploads(directory, action_dir, *, limit=MAX_UPLOADS, env=None, run=None, stop=None):
    """Upload each new record once. Failures and a missing action are ignored."""
    binding = binding_from_env(env)
    if binding is None or not Path(action_dir, "dist", "index.js").is_file():
        return 0
    sent, seen = 0, set()
    runner = run or __import__("subprocess").run
    while sent < limit and (stop is None or not stop()):
        try:
            files = sorted(Path(directory).glob("*.json"), key=lambda item: int(item.stem))
        except (OSError, ValueError):
            files = []
        for item in files:
            if item.name in seen or sent >= limit:
                continue
            seen.add(item.name)
            try:
                record = validate_record(item.read_text(encoding="utf-8"), binding)
                if record is None:
                    continue
                import tempfile
                with tempfile.TemporaryDirectory() as temporary:
                    target = Path(temporary) / "progress.json"
                    target.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
                    command, inputs = upload_command(action_dir, artifact_name(
                        binding["run_id"], binding["run_attempt"], record["seq"]), temporary)
                    runner(command, env={**os.environ, **inputs}, check=False,
                           stdout=__import__("subprocess").DEVNULL, stderr=__import__("subprocess").DEVNULL)
                sent += 1
            except (OSError, ValueError, TypeError):
                try:
                    print("Progress artifact upload unavailable", flush=True)
                except OSError:
                    pass
        if stop is not None and stop():
            break
        time.sleep(0.5)
    return sent


def _job_done(jobs, name):
    matched = [job for job in jobs if job.get("name") == name]
    if not matched:
        return False
    return all(job.get("status") == "completed" for job in matched)


def observe(env=None, *, opener=None, sleep=None, iterations=None):
    """Patch this run's open notice from progress artifacts, then stop. Always best effort."""
    try:
        env = os.environ if env is None else env
        binding = binding_from_env(env)
        token = env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")
        if binding is None or not token:
            return 0
        opener = opener or urllib.request.build_opener(NoRedirect)
        sleep = time.sleep if sleep is None else sleep
        repo, run = binding["repository"], binding["run_id"]
        prefix = f"https://api.github.com/repos/{repo}"
        applied, quiet = 0, False
        steps = 0
        while iterations is None or steps < iterations:
            steps += 1
            try:
                listed = _collect(opener, token, f"{prefix}/actions/runs/{run}/artifacts?per_page=100", "artifacts")
                jobs = _api(opener, token, "GET", f"{prefix}/actions/runs/{run}/jobs?per_page=100")
                comments = _api(opener, token, "GET",
                                f"{prefix}/issues/{binding['number']}/comments?per_page=100")
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                if iterations is not None:
                    break
                sleep(POLL_SECONDS)
                continue
            notice = None
            for comment in comments if isinstance(comments, list) else []:
                author = comment.get("user") or {}
                body = comment.get("body") or ""
                if (author.get("login") == "github-actions[bot]" and author.get("id") == 41898282
                        and author.get("type") == "Bot"
                        and body.startswith(NOTICE.format(run=run) + "\n")):
                    notice = comment
            records = []
            for artifact in listed:
                match = NAME.fullmatch(artifact.get("name") or "")
                if (not match or match.group(1) != run or match.group(2) != "1"
                        or artifact.get("expired") or not bounded(artifact.get("size_in_bytes") or 0)
                        or artifact["size_in_bytes"] > MAX_RECORD_BYTES):
                    continue
                try:
                    blob = _download(token, artifact["archive_download_url"])
                    payload = _record_from_zip(blob)
                except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, TypeError):
                    continue
                record = validate_record(payload, binding) if payload else None
                if record is not None and record["seq"] == int(match.group(3)):
                    records.append(record)
            records.sort(key=lambda item: item["seq"])
            for record in records:
                if notice is None or decide(notice.get("body"), record, applied) != "patch":
                    continue
                body = notice_text(record)
                try:
                    time.sleep(1)
                    updated = _patch(opener, token, f"{prefix}/issues/comments/{notice['id']}", body)
                except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TypeError):
                    continue
                if isinstance(updated, dict) and updated.get("body") == body:
                    notice, applied = updated, record["seq"]
            review_done = _job_done(jobs.get("jobs") or [] if isinstance(jobs, dict) else [], "Review admitted diff")
            if review_done and quiet:
                break
            quiet = review_done
            if iterations is not None:
                break
            sleep(POLL_SECONDS)
    except Exception:
        try:
            print("Review progress notice unavailable; inference continues", flush=True)
        except OSError:
            pass
    return 0


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
