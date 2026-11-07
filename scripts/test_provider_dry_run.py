# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Native Claude CLI dry harness. Only a local fixture API, no paid inference."""
import argparse
import http.server
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from unittest.mock import patch

try:
    from . import review_engine as e
except ImportError:
    import review_engine as e

# TEST FIXTURE LIMITS ONLY, not production policy or recommended operator values.
FIXTURE = {"model": "anthropic/claude-sonnet-5-5", "reasoning_effort": "medium",
           "max_turns": 10, "max_budget_usd": 5.0, "max_model_tokens": 90000,
           "max_output_tokens": 16000, "ai_timeout": 30, "provider": "claude", "passes": ["bundle"]}


def scenario(cli, mode):
    requests = []
    keys = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        head = root / "head"; head.mkdir()
        (head / "app.py").write_text("fixture source\n")
        outside = root / "outside.txt"; outside.write_text("outside-fixture-content")
        (head / "escape.txt").symlink_to(outside)
        sentinel = root / "must-not-exist"
        # Restricted mode must ignore these PR-controlled startup hooks and MCP settings.
        (head / ".claude").mkdir()
        (head / ".claude/settings.json").write_text(json.dumps({"hooks": {"SessionStart": [
            {"hooks": [{"type": "command", "command": "touch " + str(sentinel)}]}]}}))
        (head / ".mcp.json").write_text(json.dumps({"mcpServers": {
            "fixture": {"command": "touch", "args": [str(sentinel)]}}}))
        steps = [("Read", {"file_path": str(head / "app.py")}),
                 ("StructuredOutput", {"findings": []})]
        index_text = "file\tpath\tstate\tpatch_bytes\tsections\n0\t\"app.py\"\tpresent\t12\t0,1\n"
        if mode == "store":
            # The exact patch store lives under the checkout's git directory; restricted tools must read it.
            store = head / ".git" / "agent-review"
            store.mkdir(parents=True)
            (store / "index.tsv").write_text(index_text)
            steps = [("Read", {"file_path": str(store / "index.tsv")}), ("StructuredOutput", {"findings": []})]
        if mode == "schema":
            steps = [("StructuredOutput", {"invalid": True})]
        if mode == "tools":
            steps = [("Bash", {"command": "touch " + str(sentinel)}),
                     ("mcp__fixture__write", {}),
                     ("Read", {"file_path": str(outside)}),
                     ("Read", {"file_path": str(head / "escape.txt")}), *steps]

        class API(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(req)
                keys.append(self.headers.get("x-api-key"))
                if (mode == "billing-both" or mode == "billing" and keys[-1] == "fixture-primary"
                        or mode == "billing-after-output" and len(requests) > 1):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"type": "error", "error": {
                        "type": "invalid_request_error", "message":
                        "Your credit balance is too low to access the Anthropic API."}}).encode())
                    return
                if mode == "api":
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"type":"error","error":{"type":"api_error","message":"fixture error"}}).encode())
                    return
                index = len(requests) - (2 if mode == "billing" else 1)
                name, args = steps[min(index, len(steps)-1)]
                content = {"type": "tool_use", "id": f"fixture-{len(requests)}", "name": name, "input": args}
                message = {"id": "msg_fixture", "type": "message", "role": "assistant",
                           "model": "claude-sonnet-5-5", "content": [content], "stop_reason": "tool_use",
                           "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 20}}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if req.get("stream") else "application/json")
                self.end_headers()
                if not req.get("stream"):
                    self.wfile.write(json.dumps(message).encode())
                    return
                events = [
                  ("message_start", {"type": "message_start", "message": dict(message, content=[])}),
                  ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": dict(content, input={})}),
                  ("content_block_delta", {"type": "content_block_delta", "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)}}),
                  ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                  ("message_delta", {"type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 20}}),
                  ("message_stop", {"type": "message_stop"})]
                try:
                    for kind, data in events:
                        self.wfile.write(("event: " + kind + "\ndata: " + json.dumps(data) + "\n\n").encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        # Internal fixture server, never opened by a user; closed within this test.
        api = http.server.ThreadingHTTPServer(("127.0.0.1", 0), API)
        thread = threading.Thread(target=api.serve_forever, daemon=True); thread.start()
        try:
            policy = dict(FIXTURE)
            if mode == "turn":
                policy["max_turns"] = 1
            if mode == "budget":
                policy["max_budget_usd"] = 0.0000001
            command = e.claude_args(policy)
            command[0] = cli
            env = e.clean_env(policy)
            env.update(HOME=str(root / "home"), ANTHROPIC_API_KEY="fixture-only",
                       ANTHROPIC_BASE_URL=f"http://127.0.0.1:{api.server_port}",
                       CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1")
            if mode.startswith("billing"):
                # Exercise the actual engine's fallback with the pinned native CLI.
                # Only fake keys and this in-process fixture endpoint are supplied.
                actual_run = subprocess.run
                attempts = []
                def launch(argv, **kwargs):
                    argv = list(argv); argv[0] = cli
                    fixture_env = dict(env, ANTHROPIC_API_KEY=kwargs["env"]["ANTHROPIC_API_KEY"])
                    assert "ANTHROPIC_API_KEY_2" not in kwargs["env"]
                    attempt = actual_run(argv, **dict(kwargs, env=fixture_env))
                    attempts.append(attempt)
                    return attempt
                with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fixture-primary",
                                             "ANTHROPIC_API_KEY_2": "fixture-secondary"}), \
                        patch.object(e.subprocess, "run", side_effect=launch):
                    if mode == "billing":
                        result = e.run_claude(policy, {"files": [{"path": "app.py", "deleted": False}]},
                                              "bundle", head, prompt="Read app.py and return findings JSON.")
                        assert result["receipt"]["files_opened"] == ["app.py"]
                        assert keys == ["fixture-primary", "fixture-secondary", "fixture-secondary"], keys
                    else:
                        try:
                            e.run_claude(policy, {"files": []}, "bundle", head,
                                         prompt="Read app.py and return findings JSON.")
                        except e.ReviewAbort:
                            pass
                        else:
                            raise AssertionError("billing fixture produced a clean result")
                        expected = ["fixture-primary", "fixture-secondary"] if mode == "billing-both" else ["fixture-primary"] * 2
                        assert keys == expected, keys
                assert len(attempts) == (1 if mode == "billing-after-output" else 2)
                assert not sentinel.exists(), "PR-controlled command or hook executed"
                print(f"PASS native Claude {mode}: bounded billing fallback and credential isolation")
                return
            proc = subprocess.run(command, input="Read app.py and return findings JSON.",
                                  cwd=head, env=env, text=True, capture_output=True, timeout=policy["ai_timeout"])
            events = [e.strict_json(line) for line in proc.stdout.splitlines() if line.strip()]
            assert requests, "native CLI never called the fixture API"
            inventory = {t["name"] for r in requests for t in r.get("tools", [])}
            # Schema/deferred placeholders are CLI infrastructure, not executable tools.
            assert inventory <= {"Read", "Grep", "Glob", "StructuredOutput", "DeferredToolPlaceholder"}, inventory
            assert all(r["max_tokens"] == policy["max_output_tokens"] for r in requests)
            init = next(v for v in events if v.get("subtype") == "init")
            assert init["mcp_servers"] == [], "PR-controlled MCP server started"
            assert not sentinel.exists(), "PR-controlled command or hook executed"
            if mode == "normal":
                parsed = e.claude_result(events, policy, {"files": [{"path": "app.py", "deleted": False}]},
                                        "bundle", head)
                assert parsed["receipt"]["files_opened"] == ["app.py"]
            elif mode == "store":
                parsed = e.claude_result(events, policy, {"files": [{"path": "app.py", "deleted": False}]},
                                        "bundle", head)
                # The tool result carrying the index went back to the model, and the git-directory
                # read is not a changed-file receipt.
                # Read prefixes line numbers, so look for the index's own words in the returned content.
                returned = json.dumps([m for r in requests for m in r.get("messages", [])])
                assert "patch_bytes" in returned and "present" in returned, "index not delivered"
                assert parsed["receipt"]["files_opened"] == []
            else:
                try:
                    e.claude_result(events, policy, {"files": [{"path": "app.py", "deleted": False}]}, "bundle", head)
                except e.ReviewAbort:
                    pass
                else:
                    raise AssertionError("negative fixture produced a clean result")
                if mode == "tools":
                    for req in requests:
                        for msg in req.get("messages", []):
                            assert "outside-fixture-content" not in json.dumps(msg), "out-of-scope read succeeded"
                elif mode in ("turn", "budget"):
                    final = next(v for v in events if v.get("type") == "result")
                    assert final["subtype"] == ("error_max_turns" if mode == "turn" else "error_max_budget_usd"), final["subtype"]
                elif mode in ("api", "schema"):
                    assert len(requests) == 1, "failed request or invalid schema was retried"
                if mode == "api":
                    cause = e.claude_failure_cause(e.lenient_events(proc.stdout))
                    assert cause and "HTTP 503" in cause and "fixture error" not in cause, cause
            print(f"PASS native Claude {mode}: restricted tools, schema, scoped reads and receipt controls")
        finally:
            api.shutdown(); api.server_close(); thread.join()


def codex_scenario(action):
    """Run the pinned action with a fake executable, without credentials or network."""
    import sys
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        home, head, tools = root / "home", root / "head", root / "bin"
        for directory in (home, head, tools):
            directory.mkdir()
        (home / "config.toml").write_text('model_provider="codex-action-responses-proxy"\n'
            '[model_providers.codex-action-responses-proxy]\n'
            'base_url="http://127.0.0.1:12345/v1"\nwire_api="responses"\n')
        fake = tools / "codex"
        fake.write_text("#!" + sys.executable + "\n" + '''import json,os,pathlib,sys
args=sys.argv[1:]
assert ':read-only' in ' '.join(args)
assert not any(name in os.environ for name in ('ANTHROPIC_API_KEY','ANTHROPIC_API_KEY_2','OPENAI_API_KEY','GITHUB_TOKEN'))
status=pathlib.Path('/proc/self/status').read_text().splitlines()
assert 'NoNewPrivs:\\t1' in status
assert 'CapEff:\\t0000000000000000' in status
assert 'CapBnd:\\t0000000000000000' in status
assert os.getgroups()==[]
home=pathlib.Path(os.environ['CODEX_HOME'])
sessions=home/'sessions';sessions.mkdir(exist_ok=True)
events=[{'type':'turn_context','payload':{'model':'gpt-6.1-terra'}},
{'type':'event_msg','payload':{'type':'token_count','info':{
'last_token_usage':{'input_tokens':100,'output_tokens':20},
'total_token_usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120}}}}]
(sessions/'native-fixture.jsonl').write_text('\\n'.join(map(json.dumps,events)))
pathlib.Path(args[args.index('--output-last-message')+1]).write_text('{"findings":[]}')
''')
        fake.chmod(0o755)
        with patch.dict(os.environ, {"PATH": str(tools) + os.pathsep + os.environ["PATH"]}):
            result = e.run_codex(e.policy("codex"), {"files": []}, "callers", head, action, home, root / "work",
                                 prompt="Inspect this fixture and return no findings.")
        assert result["findings"] == []
        assert result["receipt"]["turns"] == 1
        assert not list((root / "work").iterdir()), "disposable Codex pass was retained"
        print("PASS native Codex launcher: read-only arguments, preserved home, removed groups and capabilities")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", help="Path to the exact pinned native CLI")
    parser.add_argument("--codex-action", type=Path, help="Checkout of the pinned Codex action (fresh disposable Linux runner only)")
    args = parser.parse_args()
    if not args.claude and not args.codex_action:
        parser.error("select --claude or --codex-action")
    if args.claude:
        cli = str(Path(args.claude).resolve())
        for mode in ("normal", "store", "tools", "turn", "budget", "api", "schema", "billing", "billing-both", "billing-after-output"):
            scenario(cli, mode)
    if args.codex_action:
        codex_scenario(args.codex_action.resolve())


if __name__ == "__main__":
    main()
