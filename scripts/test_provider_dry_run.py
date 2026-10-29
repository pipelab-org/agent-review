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
                if mode == "api":
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"type":"error","error":{"type":"api_error","message":"fixture error"}}).encode())
                    return
                name, args = steps[min(len(requests)-1, len(steps)-1)]
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
            print(f"PASS native Claude {mode}: restricted tools, schema, scoped reads and receipt controls")
        finally:
            api.shutdown(); api.server_close(); thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude", required=True, help="Path to the exact pinned native CLI")
    cli = str(Path(parser.parse_args().claude).resolve())
    for mode in ("normal", "tools", "turn", "budget", "api", "schema"):
        scenario(cli, mode)


if __name__ == "__main__":
    main()
