#!/usr/bin/env python3
# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline Responses fixture for the real pinned CLI; never forwards requests.

Sources: developers.openai.com/codex/{permissions,config-reference},
api/docs/guides/{function-calling,streaming-responses}, and the exact wire variants
in openai/codex rust-v0.165.0 codex-rs/protocol/src/models.rs. Code-mode syntax is
checked against the tools advertised by the running CLI before any call is sent.
Synthetic API usage tests receipt plumbing only; it is not a tokenizer or billing proof.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import tempfile
import threading

try:
    from . import review_engine as engine
    from . import review_codex_read as evidence
except ImportError:
    import review_engine as engine
    import review_codex_read as evidence

PINNED_VERSION = "codex-cli 0.165.0"
PROVIDER = "codex-action-responses-proxy"
# Fixture content, not a new patch-store format. Paths follow review_sections.Store.
PATCH = "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def answer():\n-    return 41\n+    return 42\n"
FILES = {
    "app.py": "def answer():\n    return 42\n",
    ".git/agent-review/index.tsv": ('file\tpath\tstate\tpatch_bytes\tsections\n'
        f'0\t"app.py"\tpresent\t{len(PATCH.encode())}\t0\n'),
    ".git/agent-review/patches/0.diff": PATCH,
}


class FixtureFailure(Exception):
    """Only fixed diagnostic strings leave this fixture."""


def runtime_path(codex):
    """Keep only the selected CLI, its Node runtime and the system tools on PATH."""
    node = shutil.which("node")
    return ":".join(dict.fromkeys([str(codex.parent),
        *([str(Path(node).parent)] if node else []), "/usr/bin", "/bin"]))


def require_version(codex, path=None):
    # Even --version receives no account environment or persisted home.
    with tempfile.TemporaryDirectory(prefix="native-version-") as directory:
        env = {"HOME": directory, "CODEX_HOME": directory,
               "PATH": runtime_path(codex) if path is None else path}
        result = subprocess.run([str(codex), "--version"], env=env, cwd=directory,
                                capture_output=True, text=True, timeout=10, check=False)
    if result.returncode or result.stdout.strip() != PINNED_VERSION:
        raise FixtureFailure("The native fixture requires the exact pinned Codex CLI version")
    return PINNED_VERSION


def create_head(head):
    """Create fixture-only source; callers may stage it with the real launcher."""
    head = Path(head)
    head.mkdir(parents=True)
    for name, value in FILES.items():
        target = head / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    return head


def _command(cmd, head, settings):
    # The CLI publishes these names and parameters in its exec description.
    args = {"cmd": cmd, "workdir": str(head), "login": False,
            "max_output_tokens": settings["max_output_tokens"]}
    return "text(await tools.exec_command(" + json.dumps(args) + "));"


def _probe(head, config, sentinel, endpoint, unix_socket, previous, controller_fd):
    # Never print fixture sentinel bytes, controller state or the environment.
    source = '''import json, os, socket
from pathlib import Path
p = json.loads(%r)
def denied(action):
    try:
        result = action()
        if hasattr(result, "close"): result.close()
    except OSError:
        return True
    return False
def tcp():
    return socket.create_connection(("127.0.0.1", p["port"]), timeout=1)
def unix():
    s = socket.socket(socket.AF_UNIX)
    try: s.settimeout(1); s.connect(p["socket"])
    finally: s.close()
r = {
    "source_write_denied": denied(lambda: open(p["source"], "a")),
    "config_write_denied": denied(lambda: open(p["config"], "a")),
    "tcp_denied": denied(tcp), "unix_denied": denied(unix),
    "parent_file_denied": denied(lambda: open(p["sentinel"], "rb")),
    "parent_environ_denied": denied(lambda: open("/proc/%%s/environ" %% p["pid"], "rb")),
    "parent_mem_denied": denied(lambda: open("/proc/%%s/mem" %% p["pid"], "rb")),
    "parent_fd_list_denied": denied(lambda: os.listdir("/proc/%%s/fd" %% p["pid"])),
    "parent_fixture_fd_denied": denied(lambda: open("/proc/%%s/fd/%%s" %% (p["pid"], p["fd"]), "rb")),
    "credentials_absent": all(k not in os.environ for k in
        ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN", "GH_TOKEN", "LD_PRELOAD")),
    "home": os.environ.get("HOME"), "codex_home": os.environ.get("CODEX_HOME"),
    "tmp": os.environ.get("TMPDIR"),
    "previous_tree_gone": p["previous"] is None or not Path(p["previous"]).exists(),
}
print(json.dumps(r))
''' % json.dumps({"source": str(Path(head) / "app.py"), "config": str(config),
                  "sentinel": str(sentinel), "port": endpoint, "socket": str(unix_socket),
                  "previous": str(previous) if previous else None, "pid": os.getpid(), "fd": controller_fd})
    return "python3 -c " + shlex.quote(source)


def _native_result(item):
    """Only accept the real text(r) result body, preserving its exit status."""
    body = item.get("output")
    if not isinstance(body, list) or len(body) != 2:
        raise FixtureFailure("Native code-mode output envelope changed")
    block = body[1]
    if not isinstance(block, dict) or block.get("type") != "input_text":
        raise FixtureFailure("Native code-mode output envelope changed")
    try:
        result = engine.strict_json(block.get("text"))
    except engine.ReviewAbort:
        raise FixtureFailure("Native command returned no structured result") from None
    if not isinstance(result, dict) or result.get("exit_code") != 0 or not isinstance(result.get("output"), str):
        raise FixtureFailure("Native command did not complete successfully under read-only permissions")
    return result["output"]


class NativeCodexFixture:
    """One bounded loopback server and trusted home, serial sessions only.

    Enter with a fresh directory. Before launching each real session call begin_session
    with its actual home/config paths. Launch through the caller's session callback or
    engine.run_codex(..., sessions=launcher), then call assert_complete. No fake binary,
    rollout writer, receipt writer, provider key, URL argument or network forwarder exists.
    """

    def __init__(self, directory, settings):
        self.root, self.settings = Path(directory), dict(settings)
        self.trusted_home = self.root / "trusted-home"
        self._active = None
        self._lock = threading.Lock()
        self.errors = []
        self.completed = []
        self._connections = set()
        self._sentinel_handle = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.trusted_home.mkdir()
        owner = self

        class API(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(owner.settings["ai_timeout"])
                owner._connections.add(self.connection)

            def finish(self):
                try:
                    super().finish()
                finally:
                    owner._connections.discard(self.connection)

            def reject(self):
                self.send_response(400)
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.close_connection = True

            def do_GET(self):
                owner.errors.append("Unexpected fixture method or path")
                self.reject()

            do_CONNECT = do_GET
            do_PUT = do_GET
            do_DELETE = do_GET

            def do_POST(self):
                try:
                    # Same conservative byte-bound as the engine's request-input allowance.
                    size = int(self.headers.get("Content-Length", "0"))
                    limit = owner.settings["max_model_tokens"] - owner.settings["max_output_tokens"]
                    if (self.path != "/v1/responses" or self.headers.get("Transfer-Encoding")
                            or self.headers.get("Content-Encoding") or not 0 < size <= limit
                            or self.headers.get("Authorization") != "Bearer fixture-only"):
                        raise FixtureFailure("Unexpected fixture method, path, size or authentication")
                    raw = self.rfile.read(size)
                    if len(raw) != size:
                        raise FixtureFailure("Truncated fixture request")
                    req = engine.strict_json(raw)
                    with owner._lock:
                        item, number = owner._advance(req)
                except (FixtureFailure, engine.ReviewAbort, ValueError, RecursionError, OSError, TypeError, AttributeError):
                    owner.errors.append("Fixture refused an invalid request or native result")
                    self.reject()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                # Published Responses SSE lifecycle and protocol ResponseItem variants.
                response = {"id": f"resp_fixture_{number}", "object": "response", "status": "completed",
                            "output": [item], "usage": {"input_tokens": 100, "output_tokens": 20,
                                "total_tokens": 120, "input_tokens_details": {"cached_tokens": 0}}}
                events = [("response.created", {"response": dict(response, status="in_progress", output=[])}),
                          ("response.output_item.added", {"output_index": 0, "item": item}),
                          ("response.output_item.done", {"output_index": 0, "item": item}),
                          ("response.completed", {"response": response})]
                try:
                    for sequence, (kind, data) in enumerate(events):
                        event = {"type": kind, "sequence_number": sequence, **data}
                        self.wfile.write(("event: " + kind + "\ndata: " + json.dumps(event) + "\n\n").encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    owner.errors.append("Native client closed the fixture stream")
                self.close_connection = True

        # Internal test listener only, never shown to a user, never forwards any request.
        self.api = http.server.ThreadingHTTPServer(("127.0.0.1", 0), API)
        self.api.daemon_threads = True
        self.thread = threading.Thread(target=self.api.serve_forever, daemon=True)
        self.thread.start()
        self.unix_path = self.root / "network-canary.sock"
        self.unix = socket.socket(socket.AF_UNIX)
        self.unix.bind(str(self.unix_path))
        self.unix_path.chmod(0o666)
        self.unix.listen()
        config = ('default_permissions = ":read-only"\napproval_policy = "never"\n'
                  'project_doc_max_bytes = 0\nproject_doc_fallback_filenames = []\n'
                  'web_search = "disabled"\ncheck_for_update_on_startup = false\n'
                  f'model_provider = "{PROVIDER}"\n'
                  f'[model_providers.{PROVIDER}]\nname = "Offline fixture"\n'
                  f'base_url = "http://127.0.0.1:{self.api.server_port}/v1"\n'
                  'wire_api = "responses"\nrequires_openai_auth = false\n'
                  'experimental_bearer_token = "fixture-only"\nsupports_websockets = false\n')
        (self.trusted_home / "config.toml").write_text(config)
        return self

    def __exit__(self, *_):
        self.api.shutdown()
        for connection in tuple(self._connections):
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        self.api.server_close()
        self.thread.join(timeout=self.settings["ai_timeout"])
        self.unix.close()
        self.unix_path.unlink(missing_ok=True)
        if self._sentinel_handle is not None:
            self._sentinel_handle.close()

    def begin_session(self, *, head, home, codex_home, tmp, sentinel_file, previous_parent=None,
                      require_parent_denial=True):
        if self._active is not None:
            raise FixtureFailure("Previous native fixture session was not verified")
        head = Path(head)
        if not Path(sentinel_file).is_file():
            raise FixtureFailure("Native parent fixture sentinel is missing")
        if previous_parent and os.path.lexists(previous_parent):
            raise FixtureFailure("Previous native session tree still exists")
        for name, value in FILES.items():
            if (head / name).read_text() != value:
                raise FixtureFailure("Native fixture head content changed")
        self._sentinel_handle = open(sentinel_file, "rb")
        commands = ["cat -- " + shlex.quote(str(head / name)) for name in FILES]
        commands.append(_probe(head, Path(codex_home) / "config.toml", sentinel_file,
                               self.api.server_port, self.unix_path, previous_parent, self._sentinel_handle.fileno()))
        self._active = {"step": 0, "commands": commands, "head": head,
                        "paths": {"home": str(home), "codex_home": str(codex_home), "tmp": str(tmp)},
                        "require_parent_denial": require_parent_denial, "probe": None, "final": False}
        return "Run the offline native sandbox fixture and return the requested findings object."

    def _advance(self, request):
        active = self._active
        if (active is None or active["final"] or not isinstance(request, dict)
                or request.get("model") != self.settings["model"] or request.get("stream") is not True
                or request.get("store") is not False or not isinstance(request.get("input"), list)):
            raise FixtureFailure("Unexpected fixture request schema or session state")
        # Discover the pinned CLI's own code-mode schema; refuse a different tool surface.
        namespaces = [t for i in request["input"] if i.get("type") == "additional_tools"
                      for t in i.get("tools", [])]
        execs = [t for n in namespaces if n.get("type") == "namespace" and n.get("name") == "functions"
                 for t in n.get("tools", []) if t.get("name") == "exec" and t.get("type") == "custom"]
        if (len(execs) != 1 or 'tools.exec_command(' not in execs[0].get("description", "")
                or 'text(value:' not in execs[0].get("description", "")):
            raise FixtureFailure("Native code-mode tool schema changed")
        step = active["step"]
        if step:
            matches = [i for i in request["input"] if i.get("type") == "custom_tool_call_output"
                       and i.get("call_id") == f"call_fixture_{step}"]
            if len(matches) != 1:
                raise FixtureFailure("Native tool result missing or duplicated")
            output = _native_result(matches[0])
            if step <= len(FILES):
                if output != list(FILES.values())[step - 1]:
                    raise FixtureFailure("Native read content was not delivered")
            else:
                probe = engine.strict_json(output)
                required = ("source_write_denied", "config_write_denied", "tcp_denied", "unix_denied",
                            "credentials_absent", "previous_tree_gone")
                if not isinstance(probe, dict) or any(probe.get(k) is not True for k in required):
                    raise FixtureFailure("Native sandbox probe failed")
                parent_checks = ("parent_file_denied", "parent_environ_denied", "parent_mem_denied",
                                 "parent_fd_list_denied", "parent_fixture_fd_denied")
                if active["require_parent_denial"] and any(probe.get(k) is not True for k in parent_checks):
                    raise FixtureFailure("Native parent sentinel isolation failed")
                if any(probe.get(k) != v for k, v in active["paths"].items()):
                    raise FixtureFailure("Native session paths differ from staged paths")
                active["probe"] = probe
        active["step"] += 1
        number = active["step"]
        if step < len(active["commands"]):
            return {"type": "custom_tool_call", "id": f"ctc_fixture_{number}", "status": "completed",
                    "call_id": f"call_fixture_{number}", "name": "exec", "namespace": "functions",
                    "input": _command(active["commands"][step], active["head"], self.settings)}, number
        active["final"] = True
        return {"type": "message", "id": f"msg_fixture_{number}", "status": "completed", "role": "assistant",
                "content": [{"type": "output_text", "text": '{"findings":[]}', "annotations": []}]}, number

    def assert_complete(self, *, output, rollout_texts):
        active = self._active
        if self.errors or not active or not active["final"] or active["probe"] is None:
            raise FixtureFailure("Native fixture did not complete all real reads and sandbox probes")
        if engine.strict_json(output) != {"findings": []}:
            raise FixtureFailure("Native structured findings output is invalid")
        turns, usage, opened, cost = engine.codex_progress(None, active["head"], strict=True,
            model=self.settings["model"], request_limit=self.settings["max_model_tokens"] - self.settings["max_output_tokens"],
            texts=rollout_texts)
        if opened != ["app.py"] or turns != len(active["commands"]) + 1 or cost is None:
            raise FixtureFailure(f"Native transcript receipt mismatch: turns={turns}, opened_count={len(opened)}, cost_known={cost is not None}")
        for name, value in FILES.items():
            if (active["head"] / name).read_text() != value:
                raise FixtureFailure("Native fixture source was modified")
        receipt = {"provider": "codex", "model": self.settings["model"], "strategy": "callers",
                   "status": "complete", "turns": turns, "files_opened": opened, "usage": usage, "cost_usd": cost}
        engine.validate_receipt(receipt, self.settings, {"files": [{"path": "app.py", "deleted": False}]}, "callers")
        result = {"requests": active["step"], "receipt": receipt, "probe": active["probe"]}
        if any(old["probe"]["codex_home"] == result["probe"]["codex_home"]
               or old["probe"]["home"] == result["probe"]["home"] for old in self.completed):
            raise FixtureFailure("Native sessions reused a writable home")
        self.completed.append(result)
        self._sentinel_handle.close()
        self._sentinel_handle = None
        self._active = None
        return result


def local_proof(codex):
    """Current-UID evidence only. No account, privilege or host configuration changes."""
    path = runtime_path(codex)
    require_version(codex, path)
    settings = engine.policy("codex-deep")
    with tempfile.TemporaryDirectory(prefix="native-codex-fixture-") as directory:
        root = Path(directory)
        head = create_head(root / "head")
        sentinel = root / "sentinel"
        sentinel.write_text("fixture-only-parent-sentinel")
        sentinel.chmod(0o600)
        previous = None
        with NativeCodexFixture(root / "api", settings) as fixture:
            for _ in range(2):
                with tempfile.TemporaryDirectory(prefix="session-", dir=root) as live:
                    parent = Path(live)
                    home, chome, tmp = (parent / n for n in ("home", "codex-home", "tmp"))
                    home.mkdir(); tmp.mkdir()
                    shutil.copytree(fixture.trusted_home, chome)
                    engine.configure_codex(chome, settings)
                    schema, output = parent / "schema.json", parent / "output.json"
                    schema.write_text(json.dumps(engine.OUTPUT_SCHEMA))
                    prompt = fixture.begin_session(head=head, home=home, codex_home=chome, tmp=tmp,
                        sentinel_file=sentinel, previous_parent=previous, require_parent_denial=False)
                    env = {"PATH": path, "HOME": str(home), "CODEX_HOME": str(chome), "TMPDIR": str(tmp)}
                    command = [str(codex), "exec", "--skip-git-repo-check", "--cd", str(head),
                        "--output-schema", str(schema), "--output-last-message", str(output),
                        "--model", settings["model"], "--config", 'default_permissions=":read-only"', "--json", "-"]
                    process = subprocess.run(command, input=prompt, env=env, cwd="/", capture_output=True,
                                             text=True, timeout=settings["ai_timeout"], check=False)
                    if process.returncode:
                        raise FixtureFailure("Native CLI failed; no sandbox or receipt pass is claimed")
                    room = shutil.disk_usage(root).free
                    result = fixture.assert_complete(
                        output=evidence.output(parent, output.name, os.getuid(), room),
                        rollout_texts=evidence.rollouts(chome, os.getuid(), room))
                    print(json.dumps({"version": PINNED_VERSION, "scope": "current-uid-only", **result}, sort_keys=True))
                previous = parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", type=Path, required=True)
    args = parser.parse_args()
    try:
        local_proof(args.codex.resolve())
    except (FixtureFailure, engine.ReviewAbort, OSError, subprocess.TimeoutExpired) as error:
        # No raw native output, requests, environment, source or provider diagnostics.
        print("FAIL native offline Codex fixture: " + (str(error) if isinstance(error, FixtureFailure) else type(error).__name__))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
