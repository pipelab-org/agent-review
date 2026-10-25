# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""GitHub request policy and allowlisted failure diagnostics for the review engine.

Publisher contracts:
https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api
https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api
PyGithub 2.10.0 (the pinned image): github.GithubRetry, github.Requester.createException
and github.GithubException.RateLimitExceededExceedsMaxWait.

Policy, from the first document: serial requests; at least one second between
mutative requests; a Retry-After header, else the x-ratelimit-reset time when
x-ratelimit-remaining is 0, else at least one minute, lengthening exponentially
for persistent failures and then raising an error. Only idempotent GET and HEAD
requests are ever retried. A write that was rate limited, or that failed in any
other way, is never repeated here: the caller cannot tell whether it persisted.

Diagnostics are built only from structured fields (status code, a fixed set of
rate-limit headers, the exception class) and fixed vocabularies. Server and model
messages, request URLs, response bodies and credentials never reach an outcome.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import sys
import time
import tomllib
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / ".github" / "pr-agent.toml"

# "Wait for at least one minute before retrying" when neither Retry-After nor an
# exhausted primary limit says how long. PyGithub's DEFAULT_SECONDARY_RATE_WAIT agrees.
MIN_THROTTLE_WAIT = 60
THROTTLE_STATUSES = (403, 429)
# Same-attempt ceiling is PyGithub's own default (Github.__init__ retry=GithubRetry(total=10)).
RETRY_TOTAL = 10

# Wording PyGithub's Requester.isPrimaryRateLimitError / isSecondaryRateLimitError accept.
KIND_WORDING = {"primary": "API rate limit exceeded",
                "secondary": "You have exceeded a secondary rate limit"}

SCHEMA = "agent-review-outcome:v1"
# Where a failure happened. The adapter advances through these in order; the host-only
# failure notifier writes the same schema with phase "notification".
PHASES = ("configuration", "provider_init", "context", "review", "inference",
          "publication", "completion", "notification")
CATEGORIES = ("rate_limit_primary", "rate_limit_secondary", "rate_limit_unknown",
              "authentication", "forbidden", "not_found", "validation", "client_error",
              "server_error", "network", "review_abort", "interrupted", "unexpected")
RATE_CATEGORIES = frozenset(c for c in CATEGORIES if c.startswith("rate_limit"))
DECISIONS = ("wait", "exceeds_budget", "unbounded")
EVENTS = ("issue_comment", "pull_request_review_comment")
# Names PyGithub's retry and Requester see; anything else in a URL is dropped.
RESOURCES = frozenset({"pulls", "files", "commits", "comments", "reviews", "issues", "contents",
                       "compare", "actions", "runs", "artifacts", "zip", "languages", "git",
                       "statuses", "branches", "memberships"})
ROOTS = frozenset({"repos", "orgs", "user", "users", "rate_limit"})
HEADERS = ("retry-after", "x-ratelimit-limit", "x-ratelimit-remaining",
           "x-ratelimit-used", "x-ratelimit-reset", "x-ratelimit-resource")

_now = time.time


class Unbounded(Exception):
    """The server did not say, in a form we can trust, how long to wait."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


class Throttled(Exception):
    """A GitHub rate-limit response with allowlisted headers and fixed message wording."""

    def __init__(self, status, headers, kind=None):
        super().__init__(status)
        self.status = status
        self.headers = header_map(headers)
        self.message = KIND_WORDING.get(kind)


def header_map(headers):
    """Lowercase allowlisted headers from a dict, requests, urllib3 or http.client mapping."""
    result = {}
    try:
        for key, value in headers.items():
            name = str(key).lower()
            if name in HEADERS and isinstance(value, str):
                result[name] = value
    except (AttributeError, TypeError):
        return {}
    return result


def _int(value):
    # ASCII digits only: int() also accepts signs, underscores, spaces and other
    # scripts' digits, none of which GitHub documents.
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,12}", value):
        return int(value)
    return None


def rate_headers(headers):
    """Validated rate-limit facts. A present but malformed header is named, never echoed."""
    raw, rate, malformed = header_map(headers), {}, []
    for name, key in (("x-ratelimit-limit", "limit"), ("x-ratelimit-remaining", "remaining"),
                      ("x-ratelimit-used", "used"), ("x-ratelimit-reset", "reset"),
                      ("retry-after", "retry_after_seconds")):
        if name in raw:
            value = _int(raw[name])
            if value is None:
                malformed.append(name)
            else:
                rate[key] = value
    if "x-ratelimit-resource" in raw:
        if re.fullmatch(r"[a-z][a-z0-9_]{0,31}", raw["x-ratelimit-resource"]):
            rate["resource"] = raw["x-ratelimit-resource"]
        else:
            malformed.append("x-ratelimit-resource")
    if "reset" in rate:
        rate["reset_in_seconds"] = max(rate["reset"] - int(_now()), 0)
    if malformed:
        rate["malformed"] = sorted(malformed)
    return rate


def message_kind(message):
    """Mirror PyGithub 2.10.0 Requester.isPrimaryRateLimitError / isSecondaryRateLimitError."""
    if not isinstance(message, str):
        return None
    text = message.lower()
    if text.startswith("api rate limit exceeded"):
        return "primary"
    if (text.startswith("you have exceeded a secondary rate limit")
            or text.endswith("please retry your request again later.")
            or text.endswith("please wait a few minutes before you try again.")):
        return "secondary"
    return None


def classify(status, headers, message=None):
    """A fixed category. The message is inspected for rate-limit wording and discarded."""
    rate = rate_headers(headers)
    if status in THROTTLE_STATUSES:
        kind = message_kind(message)
        # Documented: a primary limit shows x-ratelimit-remaining 0.
        if rate.get("remaining") == 0 or kind == "primary":
            return "rate_limit_primary"
        if kind == "secondary":
            return "rate_limit_secondary"
        if status == 429 or "retry_after_seconds" in rate or "retry-after" in rate.get("malformed", ()):
            return "rate_limit_unknown"
    if status == 401:
        return "authentication"
    if status == 403:
        return "forbidden"
    if status == 404:
        return "not_found"
    if status == 422:
        return "validation"
    if isinstance(status, int) and 500 <= status <= 599:
        return "server_error"
    return "client_error"


def endpoint_kind(url):
    """A label from a request path's structure, never from its content.

    Repository names, numbers, refs and file paths are dropped; only GitHub's own
    resource words survive, so a throttled request is attributable without
    copying anything a pull request controls.
    """
    try:
        parts = str(url).split("?", 1)[0].strip("/").split("/")
    except (TypeError, ValueError):
        return None
    if not parts or parts[0] not in ROOTS:
        return "other"
    words = [part for part in parts[1:] if part in RESOURCES]
    return "/".join([parts[0], *words[:3]])


def plan_wait(headers, attempt, now):
    """Seconds to wait before repeating a throttled GET, and where the number came from.

    Follows the best-practices page: Retry-After; else the reset time when the
    primary limit is exhausted; else one minute, doubling per consecutive
    failure. A header that is present but unusable raises Unbounded, because
    guessing a shorter wait than the server asked for risks the integration ban
    the page warns about.
    """
    raw = header_map(headers)
    if "retry-after" in raw:
        seconds = _int(raw["retry-after"])
        if seconds is None:
            raise Unbounded("retry-after-malformed")
        return float(seconds), "retry-after"
    if "x-ratelimit-remaining" in raw:
        remaining = _int(raw["x-ratelimit-remaining"])
        if remaining is None:
            raise Unbounded("remaining-malformed")
        if remaining == 0:
            reset = _int(raw.get("x-ratelimit-reset"))
            if reset is None:
                raise Unbounded("reset-missing-or-malformed")
            # PyGithub adds a second: the reset's position within its second is unknown.
            return max(reset - now, 0) + 1.0, "ratelimit-reset"
    return float(MIN_THROTTLE_WAIT * 2 ** (attempt - 1)), "default-backoff"


class WaitBudget:
    """Total seconds this run may spend sleeping on GitHub throttles."""

    def __init__(self, seconds):
        self.seconds, self.spent = float(seconds), 0.0

    def reserve(self, seconds):
        if self.spent + seconds > self.seconds:
            return False
        self.spent += seconds
        return True


def budget_seconds(config=CONFIG):
    """One provider-call timeout. The review job's 25-minute limit minus its three
    completion-call timeouts leaves twice this; the rest is job setup and publication."""
    return tomllib.loads(Path(config).read_text(encoding="utf-8"))["config"]["ai_timeout"]


class Observer:
    """Fixed-size record of what GitHub said during one run."""

    def __init__(self):
        self.reset()

    def reset(self, budget=None):
        self.phase = "configuration"
        self.errors, self.rate_limited, self.first, self.last = 0, 0, None, None
        self.throttled_reads, self.waits, self.waited = 0, 0, 0.0
        self.decision = None
        self.budget = budget if budget is not None else WaitBudget(0)

    def error(self, status, headers, message=None, endpoint=None, decision=None):
        entry = {"phase": self.phase, "category": classify(status, headers, message),
                 "status": status if isinstance(status, int) and 100 <= status <= 599 else None}
        rate = rate_headers(headers)
        if rate:
            entry["rate_limit"] = rate
        if endpoint:
            entry["endpoint"] = endpoint
        if decision in DECISIONS:
            entry["decision"] = decision
        self.errors += 1
        if entry["category"] in RATE_CATEGORIES:
            self.rate_limited += 1
        self.first = self.first or entry
        self.last = entry

    def throttle(self, method, url, status, headers, decision, wait=None, message=None):
        """A throttled GET seen by the retry policy; the only place a URL is classified."""
        self.throttled_reads += 1
        if decision == "wait":
            self.waits += 1
            self.waited += wait
        else:
            self.decision = decision
            self.error(status, headers, message, endpoint=f"{str(method).upper()} {endpoint_kind(url)}",
                       decision=decision)

    def snapshot(self):
        # first/last are the first and most recent GitHub error responses seen, including
        # ones the upstream provider handled itself (an expected 404, for example).
        return {"errors": self.errors, "rate_limited": self.rate_limited, "first": self.first, "last": self.last,
                "throttled_reads": self.throttled_reads,
                "waits": {"count": self.waits, "seconds": int(self.waited)},
                "wait_budget_seconds": int(self.budget.seconds)}


OBSERVER = Observer()


def begin(seconds=None):
    """Start a run: clear the record and set the wait budget (default from configuration)."""
    OBSERVER.reset(WaitBudget(budget_seconds() if seconds is None else seconds))


def advance(name):
    """Record that the run entered a phase (until the next advance)."""
    OBSERVER.phase = name


PHASE_TAG = "_review_failed_phase"


def _tag(error, name):
    # The phase belongs to the exception, not to the run: a failure the adapter
    # recovers from (a billing rejection that falls back to the next credential) must
    # not mislabel a later, unrelated failure. The innermost phase crossed wins.
    try:
        if PHASE_TAG not in error.__dict__:
            setattr(error, PHASE_TAG, name)
    except (AttributeError, TypeError):
        pass


def failing_phase(error):
    """The phase recorded on the newest exception in the chain that crossed one, else None."""
    for item in _chain(error):
        name = getattr(item, "__dict__", {}).get(PHASE_TAG)
        if name in PHASES:
            return name
    return None


@contextlib.contextmanager
def phase(name):
    """A nested phase. The innermost phase that an exception crosses is the failing one."""
    previous, OBSERVER.phase = OBSERVER.phase, name
    try:
        yield
    except BaseException as error:
        _tag(error, name)
        raise
    finally:
        OBSERVER.phase = previous


def _chain(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def _github_shape(error):
    status = getattr(error, "status", getattr(error, "code", None))
    if type(status) is not int or not 100 <= status <= 599:
        return None
    headers = getattr(error, "headers", None)
    if not hasattr(headers, "items"):
        headers = {}
    data = getattr(error, "data", None)
    message = data.get("message") if isinstance(data, dict) else getattr(error, "message", None)
    return status, headers, message


def throttled(error):
    """True when anything in the exception chain is a GitHub rate-limit response."""
    if failing_phase(error) == "inference":
        return False  # a model provider's throttle is never a GitHub one
    for item in _chain(error):
        shape = _github_shape(item)
        if shape and classify(*shape) in RATE_CATEGORIES:
            return True
    return False


def _name(error):
    name = type(error).__name__
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) else "Exception"


def diagnostic(error, failed_phase=None):
    """The allowlisted description of one failure."""
    phase_name = failed_phase if failed_phase in PHASES else failing_phase(error) or OBSERVER.phase
    entry = {"phase": phase_name if phase_name in PHASES else "configuration"}
    found = None
    # Only the model call runs in the inference phase, so a numeric status on anything
    # it raised belongs to the model provider (litellm wraps aiohttp errors that carry
    # .status and .headers), never to GitHub.
    for item in _chain(error) if entry["phase"] != "inference" else ():
        shape = _github_shape(item)
        if shape:
            found = (item, shape)
            break
    if found:
        item, (status, headers, message) = found
        entry.update(category=classify(status, headers, message), status=status,
                     exception=_name(item))
        rate = rate_headers(headers)
        if rate:
            entry["rate_limit"] = rate
        if entry["category"] in RATE_CATEGORIES and OBSERVER.decision in DECISIONS:
            entry["decision"] = OBSERVER.decision
        return entry
    if isinstance(error, KeyboardInterrupt):
        entry.update(category="interrupted", exception=_name(error))
    elif any(isinstance(i, OSError) or type(i).__module__.split(".")[0] in ("requests", "urllib3")
             for i in _chain(error)):
        entry.update(category="network", exception=_name(error))
    elif type(error).__name__.endswith("Abort"):
        # Exception names do not establish that their messages are safe to publish.
        entry.update(category="review_abort", exception=_name(error))
    else:
        entry.update(category="unexpected", exception=_name(error))
    if entry["phase"] == "inference":
        # A model provider's HTTP status is a number, not its message. Kept apart from
        # GitHub's `status` so a provider 429 is never read as a GitHub rate limit.
        for item in _chain(error):
            for attribute in ("status_code", "status"):
                status = getattr(item, attribute, None)
                if type(status) is int and 100 <= status <= 599:
                    entry["provider_status"] = status
                    break
            if "provider_status" in entry:
                break
    return entry


def outcome(error=None, env=None):
    env = os.environ if env is None else env
    result = {"schema": SCHEMA, "result": "failed" if error is not None else "succeeded",
              "github": OBSERVER.snapshot()}
    if error is not None:
        result["failure"] = diagnostic(error)
    run_id = env.get("GITHUB_RUN_ID", "")
    if re.fullmatch(r"[1-9][0-9]{0,18}", run_id):
        result["run_id"] = int(run_id)
    if env.get("GITHUB_EVENT_NAME") in EVENTS:
        result["event"] = env["GITHUB_EVENT_NAME"]
    return result


def _cell(value):
    return re.sub(r"[^A-Za-z0-9_./:, -]", "_", str(value))[:120]


def summary(result):
    """Markdown for the Actions step summary: fixed labels, sanitized values."""
    failure = result.get("failure", {})
    rate = failure.get("rate_limit", {})
    rows = [("Result", result["result"])]
    for label, value in (("Phase", failure.get("phase")), ("Category", failure.get("category")),
                         ("HTTP status", failure.get("status")),
                         ("Exception class", failure.get("exception")),
                         ("Rate-limit resource", rate.get("resource")),
                         ("Rate-limit remaining", rate.get("remaining")),
                         ("Rate-limit limit", rate.get("limit")),
                         ("Rate-limit reset (UTC epoch)", rate.get("reset")),
                         ("Seconds until reset at capture", rate.get("reset_in_seconds")),
                         ("Retry-After seconds", rate.get("retry_after_seconds")),
                         ("Malformed rate headers", ", ".join(rate["malformed"]) if "malformed" in rate else None),
                         ("Retry decision", failure.get("decision")),
                         ("Model provider HTTP status", failure.get("provider_status")),
                         ("Detail", failure.get("detail"))):
        if value is not None:
            rows.append((label, value))
    github = result["github"]
    rows.append(("GitHub throttle waits", f"{github['waits']['count']} waits, {github['waits']['seconds']} s "
                                          f"of {github['wait_budget_seconds']} s budget"))
    rows.append(("GitHub error responses seen", github["errors"]))
    return ("### Agent review outcome\n\n| Field | Value |\n| --- | --- |\n"
            + "".join(f"| {label} | {_cell(value)} |\n" for label, value in rows))


def report(error=None, env=None, stream=None):
    """Write the outcome file and step summary, and print one diagnostics line.

    Diagnostics must never change how the run ends: every sink failure is ignored.
    """
    env = os.environ if env is None else env
    result = outcome(error, env)
    try:
        line = json.dumps(result, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return result
    path = env.get("REVIEW_OUTCOME_PATH")
    if path:
        try:
            target = Path(path)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target.write_text(line, encoding="utf-8")
            target.chmod(0o600)
        except OSError:
            try:
                print("Review outcome could not be recorded", file=stream or sys.stderr)
            except OSError:
                pass  # A diagnostic sink must not change the review result.
    sink = env.get("GITHUB_STEP_SUMMARY")
    if sink:
        try:
            with open(sink, "a", encoding="utf-8") as handle:
                handle.write(summary(result) + "\n")
        except OSError:
            pass
    if error is not None:
        print(f"Review diagnostics: {line}", file=stream or sys.stderr)
    return result


@functools.cache
def retry_class():
    """PyGithub's GithubRetry, restricted to what GitHub's guidance makes safe."""
    from github import GithubRetry
    from github.GithubException import RateLimitExceededException, RateLimitExceededExceedsMaxWait

    class BoundedReadRetry(GithubRetry):
        def __init__(self, budget=None, **kwargs):
            self.budget = budget
            super().__init__(**kwargs)

        def new(self, **kw):
            kw["budget"] = self.budget
            return super().new(**kw)

        def get_content(self, resp, url):
            # GithubRetry reads a 403 body once to decide whether it is a rate limit.
            # Keep only the primary/secondary kind it implies, never the text.
            content = GithubRetry.get_content(resp, url)
            try:
                self.kind = message_kind(json.loads(content).get("message"))
            except (ValueError, AttributeError):
                self.kind = None
            return content

        def increment(self, method=None, url=None, response=None, error=None, _pool=None, _stacktrace=None):
            # GithubRetry decides whether a 403 is a rate limit (and raises the plain
            # error when it is not); urllib3 retries a listed 429. Reaching here with
            # a throttle status therefore means the library judged it retryable.
            retry = super().increment(method, url, response, error, _pool, _stacktrace)
            if response is None or response.status not in THROTTLE_STATUSES:
                return retry
            headers = header_map(response.headers)
            # Our own fixed wording, recognized by classify(), stands in for the body.
            kind = getattr(self, "kind", None)
            data = {"message": KIND_WORDING[kind]} if kind in KIND_WORDING else None
            try:
                seconds, _ = plan_wait(headers, len(self.history) + 1, _now())
            except Unbounded:
                OBSERVER.throttle(method, url, response.status, headers, "unbounded", message=data and data["message"])
                raise RateLimitExceededException(response.status, data, headers) from None
            if self.budget is None or not self.budget.reserve(seconds):
                OBSERVER.throttle(method, url, response.status, headers, "exceeds_budget",
                                  message=data and data["message"])
                raise RateLimitExceededExceedsMaxWait(response.status, data, headers, wait=seconds) from None
            OBSERVER.throttle(method, url, response.status, headers, "wait", seconds)
            # urllib3 sleeps get_backoff_time() once Retry-After handling is off.
            retry.get_backoff_time = lambda: seconds
            return retry

    return BoundedReadRetry


def read_retry(budget):
    """GET/HEAD-only retry on rate limits, bounded by a shared wait budget.

    Writes are not retried: a POST or PATCH that failed may still have persisted.
    5xx and network errors are not retried either (no documented wait to honor).
    respect_retry_after_header is off so urllib3 never sleeps an unbounded
    Retry-After (its default cap is six hours) outside the budget.
    """
    return retry_class()(budget=budget, total=RETRY_TOTAL, connect=0, read=False,
                         allowed_methods=frozenset({"GET", "HEAD"}), status_forcelist=[429],
                         respect_retry_after_header=False, raise_on_status=False)


def github_client(auth, base_url, *, seconds_between_requests, seconds_between_writes):
    """A PyGithub client with the review engine's request policy."""
    from github import Github
    return Github(auth=auth, base_url=base_url, retry=read_retry(OBSERVER.budget),
                  seconds_between_requests=seconds_between_requests,
                  seconds_between_writes=seconds_between_writes)


@contextlib.contextmanager
def runtime():
    """Install the policy for the pinned provider and observe GitHub errors for one run."""
    from github.Requester import Requester
    from pr_agent.config_loader import get_settings
    from pr_agent.git_providers.github_provider import GithubProvider

    original_client = GithubProvider._get_github_client
    original_error = Requester.createException

    def policy_client(provider):
        # The provider validates its auth settings and stores provider.auth. Its own
        # client enables retries for every method, so build ours from that auth.
        stock = original_client(provider)
        stock.close()
        github = get_settings().github
        return github_client(provider.auth, provider.base_url,
                             seconds_between_requests=github.get("seconds_between_requests", 0),
                             seconds_between_writes=github.get("seconds_between_writes", 1))

    def observed(cls, status, headers, output):
        error = original_error(status, headers, output)
        OBSERVER.error(status, headers, output.get("message") if isinstance(output, dict) else None)
        return error

    # Debug logging from these libraries includes request paths and bodies.
    loggers = [(logging.getLogger(name), logging.getLogger(name).level) for name in ("github", "urllib3")]
    for logger, _ in loggers:
        logger.setLevel(logging.ERROR)
    GithubProvider._get_github_client = policy_client
    Requester.createException = classmethod(observed)
    try:
        yield
    finally:
        GithubProvider._get_github_client = original_client
        Requester.createException = classmethod(original_error.__func__)
        for logger, level in loggers:
            logger.setLevel(level)
