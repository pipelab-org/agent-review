# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Focused protocol refusals; the optional native proof runs the real CLI separately."""
import http.client
import json
from pathlib import Path
import socket
import tempfile
import tomllib
import unittest
from unittest.mock import patch

try:
    from . import native_codex_fixture as fixture
except ImportError:
    import native_codex_fixture as fixture


class NativeFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="native-fixture-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = fixture.engine.policy("codex-deep")
        self.api = fixture.NativeCodexFixture(self.root / "api", self.settings)
        self.api.__enter__()
        self.addCleanup(self.api.__exit__, None, None, None)
        self.head = fixture.create_head(self.root / "head")
        self.sentinel = self.root / "sentinel"
        self.sentinel.write_text("fixture-only")
        self.paths = dict(head=self.head, home=self.root / "home", codex_home=self.root / "codex-home",
                          tmp=self.root / "tmp", sentinel_file=self.sentinel)
        self.api.begin_session(**self.paths)

    def request(self, body, path="/v1/responses", method="POST", **extra):
        conn = http.client.HTTPConnection("127.0.0.1", self.api.api.server_port, timeout=2)
        try:
            headers = {"Authorization": "Bearer fixture-only", "Content-Type": "application/json", **extra}
            conn.request(method, path, body=body, headers=headers)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def initial(self):
        # Published namespace shape and the actual pinned CLI's exec metadata contract.
        return {"model": self.settings["model"], "stream": True, "store": False,
                "input": [{"type": "additional_tools", "tools": [{"type": "namespace", "name": "functions",
                    "tools": [{"name": "exec", "type": "custom", "description": "tools.exec_command( text(value:"}]}]}]}

    def test_only_local_responses_path_is_accepted_and_no_forwarder_exists(self):
        config = tomllib.loads((self.api.trusted_home / "config.toml").read_text())
        provider = config["model_providers"][fixture.PROVIDER]
        self.assertEqual(provider["base_url"], f"http://127.0.0.1:{self.api.api.server_port}/v1")
        self.assertFalse(provider["requires_openai_auth"])
        self.assertEqual(provider["experimental_bearer_token"], "fixture-only")
        self.assertFalse(provider["supports_websockets"])
        # A known positive calibrates the rejection tests: this actually serves an SSE tool call.
        status, body = self.request(json.dumps(self.initial()))
        self.assertEqual(status, 200)
        self.assertIn(b'"type": "custom_tool_call"', body)
        for path in ("https://api.openai.com/v1/responses", "/v1/responses?target=external", "/v1/models"):
            with self.subTest(path=path):
                self.assertEqual(self.request(b"{}", path=path)[0], 400)
        self.assertEqual(self.request(b"", path="api.openai.com:443", method="CONNECT")[0], 400)

    def test_malformed_oversized_and_unauthenticated_requests_fail_closed(self):
        for data in (b"not-json", b"[]", b'{"input":[null]}', b'{"model":1,"model":2}',
                     b"x" * self.settings["max_model_tokens"]):
            with self.subTest(size=len(data)):
                self.assertEqual(self.request(data)[0], 400)
        self.assertEqual(self.request(b"{}", Authorization="Bearer not-the-fixture")[0], 400)
        self.assertEqual(self.api._active["step"], 0)

    def test_missing_native_delivery_and_failed_command_do_not_advance(self):
        request = self.initial()
        self.api._advance(request)
        with self.assertRaises(fixture.FixtureFailure):
            self.api._advance(request)
        bad = {"type": "custom_tool_call_output", "call_id": "call_fixture_1", "output": [
            {"type": "input_text", "text": "runtime header"},
            {"type": "input_text", "text": json.dumps({"exit_code": 1, "output": fixture.FILES['app.py']})}]}
        request["input"].append(bad)
        with self.assertRaises(fixture.FixtureFailure):
            self.api._advance(request)
        self.assertEqual(self.api._active["step"], 1)

    def test_failed_sandbox_probe_cannot_produce_final_findings(self):
        request = self.initial()
        active = self.api._active
        active["step"] = len(active["commands"])
        request["input"].append({"type": "custom_tool_call_output", "call_id": "call_fixture_4", "output": [
            {"type": "input_text", "text": "runtime header"}, {"type": "input_text", "text": json.dumps({
                "exit_code": 0, "output": json.dumps({"source_write_denied": False})})}]})
        with self.assertRaises(fixture.FixtureFailure):
            self.api._advance(request)
        self.assertFalse(active["final"])

    def test_missing_sentinel_and_surviving_tree_refuse_session(self):
        self.api._active = None
        with self.assertRaises(fixture.FixtureFailure):
            self.api.begin_session(**dict(self.paths, sentinel_file=self.root / "absent"))
        with self.assertRaises(fixture.FixtureFailure):
            self.api.begin_session(**self.paths, previous_parent=self.root)

    def test_version_check_has_no_inherited_home_or_credentials(self):
        from subprocess import CompletedProcess
        with patch.object(fixture.subprocess, "run", return_value=CompletedProcess([], 0, "codex-cli 0.167.0", "")) as run:
            with self.assertRaises(fixture.FixtureFailure):
                fixture.require_version(Path("/fixture/codex"))
        env = run.call_args.kwargs["env"]
        self.assertEqual(set(env), {"HOME", "CODEX_HOME", "PATH"})
        self.assertEqual(env["HOME"], env["CODEX_HOME"])
        self.assertNotEqual(env["HOME"], str(Path.home()))

    def test_canary_positive_control_is_connectable_without_native_sandbox(self):
        # The sandbox denial test must not pass merely because no service is listening.
        with socket.create_connection(("127.0.0.1", self.api.api.server_port), timeout=2):
            pass
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(self.api.unix_path))
