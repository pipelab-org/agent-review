# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0

"""Nonpublishing historical replay of the real review adapter.

Runs scripts/pr_agent_review.py unmodified: its prompts, model selection, budgets,
billing-only fallback, discovery, challenge and validators. An explicit large-context
replay diagnostic changes only the input context cap, and labels that policy in the
result. The pull request provider differs. PR-Agent's own provider factory (the same seam the runtime tests
use) returns a subclass of the pinned GithubProvider built without its constructor,
so no GitHub client exists; every GitHub read comes from an immutable snapshot and
every write is captured in memory. Results identify themselves as replays and are
never genuine GitHub completion evidence.

Network: none in fixture mode. In real mode only the configured model gateways,
through the existing handler, are reachable. Snapshot code is never executed.
Source: gateway hosts are the vendors' documented API hosts
(https://docs.anthropic.com/en/api/overview, https://platform.openai.com/docs/api-reference).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import hashlib
import io
import ipaddress
import json
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

if __package__:
    from . import pr_agent_review as review
    from . import review_replay_snapshot as snapshots
else:
    import pr_agent_review as review
    import review_replay_snapshot as snapshots

followups = review.followups

RESULT_SCHEMA = "agent-review-replay-result"
FIXTURE_SCHEMA = "agent-review-replay-transport-fixture"
RESULT_VERSION = 1
# The adapter requires a numeric Actions run id. This is the largest signed 64-bit
# integer, which is not a run id; a replay has no run, and the id never leaves memory
# except inside neutralized output.
REPLAY_RUN_ID = str(2**63 - 1)
GATEWAY_HOSTS = {"anthropic": "api.anthropic.com", "openai": "api.openai.com"}
CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_2", "OPENAI_API_KEY")
# Published common context capacity for the diagnostic's two deep-review routes:
# https://platform.claude.com/docs/en/about-claude/models/overview (Opus 5.5: 1M)
# https://developers.openai.com/api/docs/models/gpt-6-luna (1,050,000)
# This replay-only preset is separate from the configured operator budget.
LARGE_CONTEXT_TOKENS = 1_000_000
LARGE_CONTEXT_MODELS = ["anthropic/claude-opus-5-5", "gpt-6-luna"]
EXECUTED = ("scripts/pr_agent_review.py", "scripts/review_discovery.py", "scripts/review_quality.py",
            "scripts/review_followups.py", "scripts/agent_review_contract.py", ".github/pr-agent.toml",
            "scripts/review_reliability.py", "scripts/review_replay.py", "scripts/review_replay_snapshot.py")


class ReplayError(Exception):
    """Operator-facing failure with a fixed message that carries no snapshot or model text."""


class ReplayUnsupported(AttributeError):
    """The replay does not model this provider capability. AttributeError keeps hasattr() semantics."""


class Uncaptured(Exception):
    """A requested file is not in the snapshot; that says nothing about the repository."""


class NotFound(Exception):
    status = 404


class Store:
    """Everything the adapter tried to publish or read, kept in memory and never sent."""

    def __init__(self):
        self.next_id = 1_000_000_000
        self.issue_comments, self.review_comments = [], []
        self.reviews, self.uncaptured, self.unmodeled = [], [], []

    def new_id(self):
        self.next_id += 1
        return self.next_id


class Strict:
    """No MagicMock leniency: an unmodeled attribute is recorded and raises."""

    _store = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if self._store is not None:
            self._store.unmodeled.append(f"{type(self).__name__}.{name}")
        raise ReplayUnsupported(f"{type(self).__name__}.{name}")


class Obj(Strict):
    """Plain data the provider hands out. Reading a field it lacks is an unmodeled access,
    so upstream code that catches the AttributeError still invalidates comparison."""

    def __init__(self, store, **fields):
        self._store = store
        self.__dict__.update(fields)

    # SimpleNamespace semantics, which this replaces: compared and shown by its fields.
    def _fields(self):
        return {k: v for k, v in self.__dict__.items() if k != "_store"}

    def __eq__(self, other):
        return isinstance(other, Obj) and self._fields() == other._fields()

    __hash__ = None

    def __repr__(self):
        return f"Obj({self._fields()!r})"


class Comment(Strict):
    def __init__(self, store, body, *, id=None, user=None, **fields):
        self._store = store
        self.id, self.body = (store.new_id() if id is None else id), body
        self.user = user if user is not None else Obj(store, **followups.BOT)
        self.html_url = f"replay:comment/{self.id}"
        self.raw_data = {}
        self.history = [body]
        self.__dict__.update(fields)

    def edit(self, body):
        self.body = body
        self.history.append(body)


class Repo(Strict):
    def __init__(self, store, snapshot, pull):
        self._store, self.snapshot, self._pull = store, snapshot, pull
        self.full_name = snapshot.repository
        self.default_branch = "main"
        data = snapshot.data
        self._changed = {f["filename"]: f for f in data["files"]}

    def get_pull(self, number):
        if number != self.snapshot.number:
            raise Uncaptured("pull request is not the snapshot")
        return self._pull

    def get_languages(self):
        return dict(self.snapshot.data["languages"])

    def get_contents(self, path, ref=None):
        data = self.snapshot.data
        if ref != self.snapshot.head:
            # Only the admitted head is replayable. Anything else is not captured.
            self._store.uncaptured.append(f"{path}@ref-not-head")
            raise Uncaptured("only the captured head is replayable")
        changed = self._changed.get(path)
        if changed is not None and changed["edit_type"] != "DELETED":
            text = changed["head_file"]
        elif path in data["sources"]:
            text = data["sources"][path]
        elif path in data["absent"] or changed is not None:
            raise NotFound(path)
        else:
            self._store.uncaptured.append(path)
            raise Uncaptured("path is not in the snapshot")
        return Obj(self._store, type="file", encoding="base64", raw_data={}, decoded_content=text.encode("utf-8"))


class Pull(Strict):
    def __init__(self, store, snapshot):
        data = snapshot.data
        self._store, self.snapshot = store, snapshot
        self.number, self.title, self.body = snapshot.number, data["title"], data["description"]
        self.state = "open"  # replay presents the snapshot as an open same-repository PR
        self.html_url = f"replay:pull/{snapshot.number}"
        self.head = Obj(store, sha=snapshot.head, ref=data["branch"],
                        repo=Obj(store, full_name=snapshot.repository, fork=False))
        self.base = Obj(store, sha=snapshot.base, ref="base", repo=Obj(store, full_name=snapshot.repository))
        self.comments_list = []

    def get_commits(self):
        return [Obj(self._store, commit=Obj(self._store, message=m)) for m in self.snapshot.data["commits"]]

    def get_issue_comments(self):
        # Prior completion records would let the adapter short-circuit to reuse. A replay
        # has no verified artifacts, so none are presented; discussion is review threads.
        return []

    def get_comments(self):
        return self.get_review_comments()

    def get_review_comments(self):
        threads = [Comment(self._store, t["body"], id=t["id"], user=Obj(self._store, login="snapshot-author", id=0, type="User"),
                           path=t["path"], commit_id=self.snapshot.head, raw_data={
                               "in_reply_to_id": t["in_reply_to_id"], "pull_request_review_id": None})
                   for t in self.snapshot.data["discussion"]["threads"]]
        return threads + list(self._store.review_comments)

    def create_issue_comment(self, body):
        comment = Comment(self._store, body)
        self._store.issue_comments.append(comment)
        return comment

    def create_review(self, *args, **kwargs):
        review_id = self._store.new_id()
        for item in kwargs.get("comments", []):
            self._store.review_comments.append(Comment(
                self._store, item["body"], path=item["path"], commit_id=self.snapshot.head,
                raw_data={"pull_request_review_id": review_id, "in_reply_to_id": None}))
        self._store.reviews.append({"review_id": review_id, "comments": [
            {k: v for k, v in item.items() if k in ("path", "body", "line", "side", "start_line", "start_side")}
            for item in kwargs.get("comments", [])]})
        return Obj(self._store, id=review_id, user=Obj(self._store, **followups.BOT), commit_id=self.snapshot.head)


class Client(Strict):
    def __init__(self, repo):
        self._repo = repo

    def get_repo(self, name):
        if name != self._repo.full_name:
            raise Uncaptured("repository is not the snapshot")
        return self._repo


def provider_class():
    from pr_agent.git_providers.github_provider import GithubProvider

    class ReplayProvider(GithubProvider):
        """Real upstream publication and description logic over in-memory objects."""

        def __init__(self, snapshot, store):  # deliberately skips GithubProvider.__init__
            self.snapshot, self.store = snapshot, store
            self.pr = Pull(store, snapshot)
            self.repo_obj = Repo(store, snapshot, self.pr)
            self.github_client = Client(self.repo_obj)
            self.repo, self.pr_num = snapshot.repository, snapshot.number
            self.base_url_html = "replay:"
            self.issue_main = None
            self.max_comment_chars = 65000
            self.last_commit_id = Obj(store, sha=snapshot.head)
            self.incremental = Obj(store, is_incremental=False)
            self.unreviewed_files_map = {}
            self.diff_files = None
            self.git_files = snapshot.file_patch_infos()
            self._inline_comment_store = None
            self._verify_code_comments = lambda comments: (comments, [])

        def _get_repo(self):
            return self.repo_obj

        def get_diff_files(self):
            if self.diff_files is None:
                self.diff_files = self.snapshot.file_patch_infos()
            return self.diff_files

        def get_files(self):
            return self.get_diff_files()

        def get_num_of_files(self):
            return len(self.get_diff_files())

        def get_languages(self):
            return self.repo_obj.get_languages()

        def get_pr_branch(self):
            return self.pr.head.ref

        def get_commit_messages(self):
            return "\n".join(f"{i + 1}. {m}" for i, m in enumerate(self.snapshot.data["commits"]))

        def get_pr_url(self):
            return self.pr.html_url

        def get_issue_comments(self):
            return self.pr.get_issue_comments()

    return ReplayProvider


class Egress:
    """Defense in depth behind the provider. Fixture mode permits no network at all."""

    def __init__(self, hosts):
        self.hosts, self.denied = frozenset(hosts), []

    def deny(self, what):
        self.denied.append(what)
        raise ReplayError("Network access outside the model gateway is denied")

    def proxies(self):
        found = set()
        for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            value = os.environ.get(name)
            if value and self.hosts:
                parsed = urlparse(value if "://" in value else "http://" + value)
                if parsed.hostname and parsed.scheme in ("http", "https"):
                    found.add((parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)))
        return found

    @contextlib.contextmanager
    def active(self):
        import httpx
        import requests
        guard = self
        proxies = guard.proxies()
        # Permission belongs to one checked HTTPX transport request, not the process.
        # Async DNS workers receive an explicit copy of that context below.
        active_request = contextvars.ContextVar("replay_gateway_request", default=None)
        getaddrinfo, connect, connect_ex = socket.getaddrinfo, socket.socket.connect, socket.socket.connect_ex
        sendto, sendmsg = socket.socket.sendto, socket.socket.sendmsg
        client_send, async_send = httpx.Client.send, httpx.AsyncClient.send
        single_send = httpx.Client._send_single_request
        async_single_send = httpx.AsyncClient._send_single_request

        def resolve(host, port, *args, **kwargs):
            state = active_request.get()
            host = host.decode("ascii") if isinstance(host, bytes) else host
            if state is None or not state["active"] or (host, int(port)) not in state["destinations"]:
                guard.deny("resolve outside gateway request")
            result = getaddrinfo(host, port, *args, **kwargs)
            state["resolved"].update((item[4][0], item[4][1]) for item in result)
            return result

        async def async_resolve(loop, host, port, *args, **kwargs):
            # BaseEventLoop.getaddrinfo uses run_in_executor, which doesn't copy
            # contextvars. Carry this request's permission into that DNS worker only.
            context = contextvars.copy_context()
            return await loop.run_in_executor(
                None, context.run, lambda: resolve(host, port, *args, **kwargs))

        def check_address(address):
            state = active_request.get()
            if state is None or not state["active"]:
                guard.deny("connect outside gateway request")
            if not isinstance(address, tuple) or len(address) < 2:
                guard.deny("non-IP socket")
            try:
                ip = str(ipaddress.ip_address(address[0]))
            except ValueError:
                guard.deny("connect by name")
            # Numeric proxy addresses can skip getaddrinfo in asyncio/AnyIO.
            numeric = {(host, port) for host, port in state["destinations"]
                       if host == ip}
            if (ip, address[1]) not in state["resolved"] | numeric:
                guard.deny("connect to unresolved address or port")

        def guarded_connect(sock, address):
            check_address(address)
            return connect(sock, address)

        def guarded_connect_ex(sock, address):
            check_address(address)
            return connect_ex(sock, address)

        def guarded_sendto(sock, data, *args):
            # socket(7): an unconnected datagram socket names its peer on each send, so
            # connect() alone never sees it. sendto(data, address) or (data, flags, address).
            check_address(args[-1])
            return sendto(sock, data, *args)

        def guarded_sendmsg(sock, buffers, *args, **kwargs):
            address = kwargs.get("address", args[2] if len(args) > 2 else None)
            if address is not None:
                check_address(address)
            return sendmsg(sock, buffers, *args, **kwargs)

        def check_request(request):
            if (request.url.scheme != "https" or request.url.host not in guard.hosts
                    or request.url.port not in (None, 443)):
                guard.deny(f"http {request.url.host}")

        def sync_send(client, request, *args, **kwargs):
            check_request(request)
            return client_send(client, request, *args, **kwargs)

        async def async_guarded(client, request, *args, **kwargs):
            check_request(request)
            return await async_send(client, request, *args, **kwargs)

        @contextlib.contextmanager
        def transport_request(request):
            # HTTPX's pinned single-request seam runs for each redirect and auth resend.
            check_request(request)
            state = {"destinations": proxies | {(request.url.host, 443)},
                     "resolved": set(), "active": True}
            token = active_request.set(state)
            try:
                yield
            finally:
                # Context copies held by child tasks or DNS workers expire too.
                state["active"] = False
                active_request.reset(token)

        def sync_single(client, request):
            with transport_request(request):
                return single_send(client, request)

        async def async_single(client, request):
            with transport_request(request):
                return await async_single_send(client, request)

        def no_legacy_dns(*args, **kwargs):
            # Python socket docs: gethostbyname[_ex], gethostbyaddr and getnameinfo
            # perform their own resolution, without calling Python's getaddrinfo.
            # Only the checked HTTPX transport's getaddrinfo route is supported.
            # https://docs.python.org/3/library/socket.html#socket.gethostbyname
            guard.deny("unsupported name lookup")

        def no_requests(*args, **kwargs):
            guard.deny("requests")

        with (patch.object(socket, "getaddrinfo", resolve),
              patch.object(socket, "gethostbyname", no_legacy_dns),
              patch.object(socket, "gethostbyname_ex", no_legacy_dns),
              patch.object(socket, "gethostbyaddr", no_legacy_dns),
              patch.object(socket, "getnameinfo", no_legacy_dns),
              patch.object(asyncio.BaseEventLoop, "getaddrinfo", async_resolve),
              patch.object(socket.socket, "connect", guarded_connect),
              patch.object(socket.socket, "connect_ex", guarded_connect_ex),
              patch.object(socket.socket, "sendto", guarded_sendto),
              patch.object(socket.socket, "sendmsg", guarded_sendmsg),
              patch.object(httpx.Client, "send", sync_send),
              patch.object(httpx.AsyncClient, "send", async_guarded),
              patch.object(httpx.Client, "_send_single_request", sync_single),
              patch.object(httpx.AsyncClient, "_send_single_request", async_single),
              patch.object(requests.Session, "send", no_requests)):
            yield self


def load_fixture(path):
    try:
        data = followups.strict_json(snapshots.read_regular(path).decode("utf-8"))
    except (snapshots.SnapshotError, UnicodeDecodeError, ValueError):
        raise ReplayError("Transport fixture could not be read") from None
    if (not isinstance(data, dict) or set(data) != {"schema", "version", "responses"}
            or data["schema"] != FIXTURE_SCHEMA or data["version"] != 1
            or not isinstance(data["responses"], list)):
        raise ReplayError("Transport fixture has an unknown shape")
    for item in data["responses"]:
        text = isinstance(item, dict) and set(item) <= {"text", "usage", "finish_reason"} and "text" in item
        error = isinstance(item, dict) and set(item) == {"error"}
        if text and "finish_reason" in item and not isinstance(item["finish_reason"], (str, type(None))):
            raise ReplayError("Transport fixture finish reason is invalid")
        if not (text and isinstance(item["text"], str)) and not (
                error and isinstance(item["error"], dict)):
            raise ReplayError("Transport fixture response is invalid")
    return data["responses"]


def fixture_error(spec):
    import httpx
    import openai
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if spec.get("timeout") is True:
        return openai.APITimeoutError(request=request)
    status = spec.get("status")
    if type(status) is not int:
        raise ReplayError("Transport fixture error needs a status or timeout")
    response = httpx.Response(status, request=request, json=spec.get("body", {}))
    return openai.APIStatusError("fixture rejection", response=response, body=spec.get("body"))


def executed_files(root, profile):
    result = {}
    for name in (*EXECUTED, None):
        path = Path(profile) if name is None else root / name
        result["profile" if name is None else name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def neutralize(text, repository):
    """Make captured publications unusable as completion or reuse evidence."""
    text = text.replace("<!-- agent-review-", "<!-- replay-neutralized-agent-review-")
    text = text.replace(f"run={REPLAY_RUN_ID}", "run=replay")
    return text.replace(f"https://github.com/{repository}/actions/runs/{REPLAY_RUN_ID}", "replay:no-actions-run")


def chain_for(settings, litellm):
    models = [settings.config.model, *settings.config.fallback_models]
    credentials = {name: os.environ.get(name, "") for name in CREDENTIALS}
    try:
        chain = review.credential_chain(models, credentials)
    except ValueError:
        raise ReplayError("Configured models are outside the adapter contract") from None
    providers = {litellm.get_llm_provider(m)[1] for m in models}
    return models, chain, {GATEWAY_HOSTS[p] for p in providers if p in GATEWAY_HOSTS}


def settings_record(settings, models):
    return {"models": models, "reasoning_effort": settings.config.get("reasoning_effort"),
            "max_model_tokens": settings.config.max_model_tokens,
            "custom_model_max_tokens": settings.config.custom_model_max_tokens,
            "max_output_tokens": settings.config.max_output_tokens,
            "ai_timeout": settings.config.ai_timeout,
            "max_calls": settings.pr_reviewer.max_number_of_calls,
            "max_findings": settings.pr_reviewer.num_max_findings}


def failure_trace(error, root):
    """Local code locations only: exception messages and locals can contain secrets."""
    chain, seen = [], set()
    scripts = (root / "scripts").resolve()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        frames = []
        trace = error.__traceback__
        while trace is not None:
            filename = Path(trace.tb_frame.f_code.co_filename).resolve()
            if filename.parent == scripts:
                frames.append({"file": "scripts/" + filename.name, "line": trace.tb_lineno,
                               "function": trace.tb_frame.f_code.co_name})
            trace = trace.tb_next
        chain.append({"type": type(error).__name__, "frames": frames})
        error = error.__cause__ if error.__cause__ is not None else (
            None if error.__suppress_context__ else error.__context__)
    return chain


def run(snapshot, *, mode, fixture=None, profile=None, reviewer_sha, profile_sha=None, depth="default",
        record_prompts=None, record_responses=False, large_context_diagnostic=False, root=None):
    """Replay one snapshot. Returns the result document; raises ReplayError before any run."""
    root = Path(root) if root else Path(review.__file__).resolve().parents[1]
    profile = Path(profile) if profile else review.PROFILE
    profile_sha = profile_sha or reviewer_sha
    if mode not in ("fixture", "real") or (mode == "fixture") != (fixture is not None):
        raise ReplayError("Choose fixture mode with a transport fixture, or real mode without one")
    if depth not in ("default", "deep") or not all(followups.full_sha(s) for s in (reviewer_sha, profile_sha)):
        raise ReplayError("Depth or reviewer identity is invalid")
    if type(large_context_diagnostic) is not bool or (large_context_diagnostic and depth != "deep"):
        raise ReplayError("Large-context diagnostic requires deep review and an explicit boolean option")
    responses = list(fixture) if mode == "fixture" else None
    env = {"REVIEW_SOURCE_SHA": reviewer_sha, "REVIEW_PROFILE_SHA": profile_sha,
           "REVIEW_PROFILE_PATH": str(profile), "REVIEW_DEPTH": depth, "REVIEW_HEAD": snapshot.head,
           "REVIEW_BASE": snapshot.base, "GITHUB_REPOSITORY": snapshot.repository,
           "PR_NUMBER": str(snapshot.number), "GITHUB_RUN_ID": REPLAY_RUN_ID,
           "GITHUB_TOKEN": "replay-has-no-github-credential"}
    if mode == "fixture":
        # Slots must be populated for the adapter to admit its configured chain. These are
        # labels, not credentials, and no request is ever made with them.
        env.update({name: f"replay-fixture-label-{name.lower()}" for name in CREDENTIALS})
    store, calls, out = Store(), [], io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env))
        for name in ("GH_TOKEN", "ORG_MEMBERS_TOKEN", "GITHUB_EVENT_NAME", "GITHUB_EVENT_PATH",
                     "REVIEW_COMPLETION_PATH", "GITHUB_ACTIONS"):
            os.environ.pop(name, None)
        try:
            settings, litellm = review.configure()
        except (ValueError, OSError, review.ReviewAbort):
            raise ReplayError("Profile or review configuration is invalid") from None
        models, chain, hosts = chain_for(settings, litellm)
        if large_context_diagnostic:
            if models != LARGE_CONTEXT_MODELS:
                raise ReplayError("Large-context diagnostic requires the documented deep-review routes")
            # PR-Agent uses custom_model_max_tokens for models newer than its metadata;
            # max_model_tokens is the additional operator cap. Both must fit the request.
            settings.set("config.max_model_tokens", LARGE_CONTEXT_TOKENS)
            settings.set("config.custom_model_max_tokens", LARGE_CONTEXT_TOKENS)
        if mode == "real" and not chain:
            raise ReplayError("No configured model credential is available; real inference is not attempted")
        egress = Egress(hosts if mode == "real" else ())
        from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
        real_completion = LiteLLMAIHandler._get_completion

        async def completion(handler, **kwargs):
            messages = kwargs.get("messages", [])
            # Request parameters only, never the credential. Absent keys stay absent.
            entry = {"index": len(calls) + 1, **{k: kwargs[k] for k in (
                         "model", "reasoning_effort", "thinking", "output_config", "max_tokens",
                         "num_retries", "max_retries", "timeout") if k in kwargs},
                     "system_sha256": hashlib.sha256(str(messages[0]["content"]).encode()).hexdigest(),
                     "user_sha256": hashlib.sha256(str(messages[1]["content"]).encode()).hexdigest(),
                     "outcome": None}
            if record_prompts is not None:
                entry["messages"] = messages
            calls.append(entry)
            try:
                if mode == "real":
                    result = await real_completion(handler, **kwargs)
                else:
                    if not responses:
                        raise ReplayError("Transport fixture is exhausted")
                    item = responses.pop(0)
                    if "error" in item:
                        raise fixture_error(item["error"])
                    response = litellm.ModelResponse(
                        model=kwargs["model"], choices=[{"message": {"content": item["text"]}}],
                        usage=item.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}))
                    result = item["text"], item.get("finish_reason", "stop"), response
            except BaseException as error:
                entry["outcome"] = f"error:{type(error).__name__}"
                raise
            entry["outcome"] = "returned"
            if record_responses and isinstance(result[0], str):
                entry["response"] = {"text": result[0]}
                if isinstance(result[1], (str, type(None))):
                    entry["response"]["finish_reason"] = result[1]
            return result

        provider = provider_class()(snapshot, store)
        stack.enter_context(patch("pr_agent.tools.pr_reviewer.get_git_provider_with_context",
                                  return_value=provider))
        stack.enter_context(patch.object(LiteLLMAIHandler, "_get_completion", completion))
        stack.enter_context(egress.active())
        failure = None
        diagnostic = []
        try:
            review.reliability.begin()
            with contextlib.redirect_stdout(out), review.reliability.runtime():
                asyncio.run(review.review(settings))
        except (review.ReviewAbort, followups.FollowupAbort) as error:
            failure = str(error)
            if record_responses:
                diagnostic = failure_trace(error, root)
        except Exception as error:
            failure = f"{type(error).__name__}"
            if record_responses:
                diagnostic = failure_trace(error, root)
    document = result_document(snapshot, mode, settings, models, chain, profile, root, reviewer_sha, profile_sha,
                           depth, calls, store, out.getvalue(), failure, egress, responses,
                           record_prompts is not None)
    document["reviewer"]["context_policy"] = ("large-context-diagnostic" if large_context_diagnostic else "configured")
    if record_responses:
        document["diagnostics"] = {"failure_chain": diagnostic}
        # Do not invent provider errors or retry behavior for a partially captured run.
        document["transport_fixture"] = (
            {"schema": FIXTURE_SCHEMA, "version": 1,
             "responses": [dict(call["response"]) for call in calls]}
            if calls and all(call["outcome"] == "returned" and "response" in call and "finish_reason" in call["response"] for call in calls)
            else None)
    return document


def result_document(snapshot, mode, settings, models, chain, profile, root, reviewer_sha, profile_sha,
                    depth, calls, store, stdout, failure, egress, responses, with_prompts):
    summaries = [c for c in store.issue_comments if "agent-review-result:v1" in c.body]
    evidence = None
    if failure is None and summaries:
        parsed = review.parse_result(summaries[-1].body)
        evidence = parsed[1] if parsed else None
    run_line = None
    for line in reversed(stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and "successful_calls" in candidate:
            run_line = candidate
            break
    completed = failure is None and evidence is not None
    repo = snapshot.repository
    doc = {
        "schema": RESULT_SCHEMA, "version": RESULT_VERSION, "replay": True,
        "genuine_github_completion": False,
        "publication": "none; every publication was captured in memory and never sent to GitHub",
        "inference": {
            "mode": mode, "quality_evidence": mode == "real" and completed,
            "note": ("Responses came from a transport fixture, not a model; this exercises the adapter only."
                     if mode == "fixture" else
                     "Responses came from the configured model gateway through the existing handler.")},
        "snapshot": {"sha256": snapshot.sha256, "repository": repo, "number": snapshot.number,
                     "base": snapshot.base, "head": snapshot.head,
                     "as_of": snapshot.data["discussion"]["as_of"]},
        "reviewer": {"asserted_source_sha": reviewer_sha, "asserted_profile_sha": profile_sha,
                     "depth": depth, "executed_files": executed_files(root, profile),
                     "credential_route": [list(step) for step in chain],
                     "configuration": settings_record(settings, models)},
        "adapter_outcome": "completed" if completed else "aborted",
        "failure": None if completed else (failure or "Adapter finished without a completion record"),
        "calls": calls, "run": run_line,
        "findings": evidence["findings"] if evidence else None,
        "quality": evidence["quality"] if evidence else None,
        "scope": evidence["scope"] if evidence else None,
        "captured": {
            "summary": neutralize(summaries[-1].body, repo) if summaries else None,
            "status_comments": [neutralize(c.body, repo) for c in store.issue_comments
                                if c is not (summaries[-1] if summaries else None)],
            "inline_reviews": store.reviews},
        "uncaptured_source_requests": sorted(set(store.uncaptured)),
        "unmodeled_provider_access": sorted(set(store.unmodeled)),
        "network_denials": egress.denied,
        "unused_fixture_responses": None if responses is None else len(responses),
    }
    if not with_prompts:
        for call in doc["calls"]:
            call.pop("messages", None)
    doc["comparable"] = completed and not (
        doc["uncaptured_source_requests"] or doc["unmodeled_provider_access"] or doc["network_denials"])
    doc["inference"]["quality_evidence"] = mode == "real" and doc["comparable"]
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--snapshot-sha256")
    parser.add_argument("--reviewer-sha", required=True, help="full commit SHA of the reviewer revision under test")
    parser.add_argument("--profile")
    parser.add_argument("--profile-sha")
    parser.add_argument("--depth", default="default", choices=("default", "deep"))
    parser.add_argument("--large-context-diagnostic", action="store_true",
                        help="deep replay only: use the documented 1M-token context cap; keep output and call limits")
    parser.add_argument("--inference", required=True, choices=("fixture", "real"))
    parser.add_argument("--fixture")
    parser.add_argument("--output")
    parser.add_argument("--record-prompts", action="store_true",
                        help="include the exact model messages in the result (contains snapshot text)")
    parser.add_argument("--record-responses", action="store_true",
                        help="include model output, local failure locations and an offline fixture (may contain source text)")
    args = parser.parse_args(argv)
    try:
        snapshot = snapshots.load(args.snapshot, args.snapshot_sha256)
        if args.profile and not args.profile_sha:
            raise ReplayError("A custom profile needs --profile-sha")
        fixture = load_fixture(args.fixture) if args.fixture else None
        # Third-party libraries print diagnostics to stdout on provider errors. Keep stdout
        # for the single result document.
        with contextlib.redirect_stdout(sys.stderr):
            document = run(snapshot, mode=args.inference, fixture=fixture, profile=args.profile,
                           reviewer_sha=args.reviewer_sha, profile_sha=args.profile_sha, depth=args.depth,
                           record_prompts=True if args.record_prompts else None, record_responses=args.record_responses,
                           large_context_diagnostic=args.large_context_diagnostic)
    except (snapshots.SnapshotError, ReplayError) as error:
        print(f"Replay not run: {error}", file=sys.stderr)
        return 2
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    # A completed run that read something uncaptured, modeled nothing for it or was denied
    # a request is not a result to compare, so it must not look like success to a script.
    return 0 if document["adapter_outcome"] == "completed" and document["comparable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
