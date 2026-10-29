# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline provider, Git coverage, malformed response and receipt regressions."""
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import review_engine as e, review_jobs as jobs

# TEST FIXTURE LIMITS ONLY. These numbers are not production operator policy.
POLICY = {"model": "anthropic/claude-sonnet-5-5", "reasoning_effort": "medium",
          "max_turns": 10, "max_budget_usd": 5.0, "max_model_tokens": 90000,
          "max_output_tokens": 16000, "ai_timeout": 300,
          "provider": "claude", "passes": ["bundle", "callers"]}
BUNDLE = {"files": [{"path": "src/app.py", "deleted": False}], "instructions": "Review correctness."}


def receipt(strategy="bundle"):
    return {"provider": "claude", "model": POLICY["model"], "strategy": strategy,
            "status": "complete", "turns": 3, "files_opened": ["src/app.py"],
            "usage": {"input_tokens": 100, "output_tokens": 20}, "cost_usd": 0.01}


def finding(**kwargs):
    return {"path": "src/app.py", "line": 1, "end_line": 1, "title": "Missing state transfer",
            "mechanism": "Resolved state is not copied", "trigger": "A reload after resolving a local value",
            "consequence": "The existing value is lost", "repair": "Preserve the resolved value",
            "confidence": "high", **kwargs}


def result():
    return {"findings": [], "receipt": {"passes": [receipt(), receipt("callers")]}}


def transcript():
    return [{"type": "assistant", "message": {"model": "claude-sonnet-5-5", "content": [
                {"type": "tool_use", "id": "read-1", "name": "Read", "input": {"file_path": "src/app.py"}}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "read-1", "is_error": False, "content": "source"}]}},
            {"type": "result", "subtype": "success", "is_error": False, "num_turns": 3,
             "structured_output": {"findings": []}, "usage": {"input_tokens": 100, "output_tokens": 20},
             "total_cost_usd": 0.01, "modelUsage": {"claude-sonnet-5-5": {"inputTokens": 100}}}]


class PolicyTests(unittest.TestCase):
    def data(self):
        return {"config": {k: POLICY[k] for k in ("max_model_tokens", "max_output_tokens", "ai_timeout")},
                "review_depth": {d: {**{k: POLICY[k] for k in ("model", "reasoning_effort", "max_turns", "max_budget_usd")}}
                                 for d in e.DEPTHS}}

    def test_missing_and_invalid_caps_name_exact_key_for_every_depth(self):
        for depth in e.DEPTHS:
            for key in ("max_turns", "max_budget_usd"):
                for bad in (None, 0, -1, True, "10", float("nan"), float("inf")):
                    data = self.data()
                    if bad is None:
                        data["review_depth"][depth].pop(key)
                    else:
                        data["review_depth"][depth][key] = bad
                    with self.subTest(depth=depth, key=key, value=bad), self.assertRaisesRegex(
                            e.ReviewAbort, "review_depth." + depth + "." + key):
                        e.policy(depth, data)
        data = self.data()
        data["review_depth"]["default"]["max_budget_usd"] = 10**1000
        with self.assertRaisesRegex(e.ReviewAbort, "review_depth.default.max_budget_usd"):
            e.policy("default", data)
        data = self.data()
        data["review_depth"]["default"]["max_turns"] = 1.5
        with self.assertRaisesRegex(e.ReviewAbort, "max_turns"):
            e.policy("default", data)

    def test_shipped_configuration_sets_approved_limits_and_code_has_no_defaults(self):
        import tomllib
        shipped = tomllib.loads(e.CONFIG.read_text())
        approved = {"default": (40, 2.0), "deep": (40, 4.0), "codex": (40, 2.0), "codex-deep": (40, 10.0)}
        for depth in e.DEPTHS:
            table = shipped["review_depth"][depth]
            self.assertEqual(approved[depth], (table["max_turns"], table["max_budget_usd"]))
            resolved = e.policy(depth)
            self.assertEqual(approved[depth], (resolved["max_turns"], resolved["max_budget_usd"]))
            stripped = self.data()
            del stripped["review_depth"][depth]["max_turns"]
            with self.assertRaisesRegex(e.ReviewAbort, "review_depth." + depth + ".max_turns"):
                e.policy(depth, stripped)
            self.assertEqual(e.DEPTHS[depth][1], tuple(e.policy(depth, self.data())["passes"]))
        self.assertEqual("codex", e.policy("codex", self.data())["provider"])
        self.assertEqual("codex", e.policy("codex-deep", self.data())["provider"])
        # Every shipped Codex model has a sourced price, so no shipped depth fails closed on cost.
        for depth in ("codex", "codex-deep"):
            self.assertIn(shipped["review_depth"][depth]["model"], e.CODEX_PRICES)

    def test_command_routing_is_exact(self):
        for command, depth in (("/review", "default"), ("/review deep", "deep"), ("/review codex", "codex"),
                               ("/review codex deep", "codex-deep")):
            self.assertEqual(depth, jobs.command_depth(command))
        for bad in ("/review ", "/Review", "/review codex extra", "/review codex deep ", "/review deep codex",
                    "/review\n", "", None):
            with self.assertRaises(e.ReviewAbort):
                jobs.command_depth(bad)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        (self.root / "app.py").write_text("old\n")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True).strip()

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "fixture head")
        return self.git("rev-parse", "HEAD")

    def bundle(self, settings=POLICY, **kwargs):
        return e.build_bundle(self.root, self.git("rev-parse", "HEAD"), self.base, settings, **kwargs)

    def test_merge_base_excludes_base_only_changes_and_includes_full_head_content(self):
        self.git("switch", "-qc", "head")
        (self.root / "app.py").write_text("changed\n")
        head = self.commit()
        self.git("switch", "-q", "main")
        (self.root / "only-base.py").write_text("not in head\n")
        base = self.commit()
        self.git("switch", "-q", "head")
        bundle = e.build_bundle(self.root, head, base, POLICY)
        self.assertEqual(self.base, bundle["merge_base"])
        self.assertEqual(["app.py"], [f["path"] for f in bundle["files"]])
        self.assertEqual("changed\n", bundle["files"][0]["content"])

    def test_large_file_keeps_all_changed_regions_with_git_context(self):
        lines = [f"line {i}" for i in range(5000)]
        (self.root / "app.py").write_text("\n".join(lines) + "\n")
        self.base = self.commit()
        lines[40], lines[4500] = "new first", "new last"
        (self.root / "app.py").write_text("\n".join(lines) + "\n")
        self.commit()
        limited = dict(POLICY, max_model_tokens=8000, max_output_tokens=1000)  # fixture operator budget
        file = self.bundle(limited)["files"][0]
        self.assertIsNone(file["content"])
        self.assertEqual(2, len(file["regions"]))
        self.assertIn("new first", file["regions"][0]["content"])
        self.assertIn("new last", file["regions"][1]["content"])

    def test_diff_cannot_fit_is_incomplete_not_clean(self):
        (self.root / "app.py").write_text("new content " * 1000)
        self.commit()
        with self.assertRaisesRegex(e.ReviewAbort, "incomplete"):
            self.bundle(dict(POLICY, max_model_tokens=1500, max_output_tokens=1000))

    def test_exclusions_and_deleted_file_are_explicit(self):
        (self.root / "app.py").unlink()
        (self.root / "generated.bin").write_bytes(b"\x00binary")
        self.commit()
        bundle = self.bundle(exclusions=["*.bin"])
        self.assertEqual(["generated.bin"], bundle["exclusions"])
        self.assertTrue(bundle["files"][0]["deleted"])

    def test_changed_binary_and_symlink_are_not_silent_omissions(self):
        (self.root / "app.py").write_bytes(b"\x00binary")
        self.commit()
        with self.assertRaisesRegex(e.ReviewAbort, "binary"):
            self.bundle()
        (self.root / "app.py").unlink()
        (self.root / "app.py").symlink_to("/etc/passwd")
        self.commit()
        with self.assertRaisesRegex(e.ReviewAbort, "symlinks"):
            self.bundle()

    def test_snapshot_mismatch_and_unsafe_paths_fail_closed(self):
        with self.assertRaisesRegex(e.ReviewAbort, "differs"):
            e.build_bundle(self.root, "f" * 40, self.base, POLICY)
        for path in ("../key", "/etc/passwd", "a/../../b", ".git/config", "a\\b", "-option", "a//b"):
            self.assertFalse(e.safe_path(path), path)
        self.assertTrue(e.safe_path("src/app.py"))


class ResultTests(unittest.TestCase):
    def test_invalid_finding_schema_is_rejected(self):
        for item in ({"path": "src/app.py"}, finding(line=True), finding(end_line=0),
                     finding(confidence="certain"), finding(path="../key"), finding(extra="unknown")):
            with self.subTest(item=item), self.assertRaises(e.ReviewAbort):
                e.validate_result({**result(), "findings": [item]}, POLICY, BUNDLE)

    def test_receipt_check_rejects_unopened_changed_file(self):
        data = result()
        data["receipt"]["passes"][0]["files_opened"] = []
        with self.assertRaisesRegex(e.ReviewAbort, "opened"):
            e.validate_result(data, POLICY, BUNDLE)

    def test_missing_invalid_limit_and_failed_receipts_never_mean_clean(self):
        for bad in (None, {}, {"passes": []}, {"passes": [receipt()]}):
            with self.subTest(bad=bad), self.assertRaises(e.ReviewAbort):
                e.validate_result({"findings": [], "receipt": bad}, POLICY, BUNDLE)
        for key, value in (("status", "budget_stop"), ("turns", POLICY["max_turns"] + 1), ("turns", True),
                           ("cost_usd", POLICY["max_budget_usd"] + 1), ("cost_usd", None), ("cost_usd", float("nan")),
                           ("usage", {}), ("usage", {"input_tokens": True, "output_tokens": 2}),
                           ("provider", "other"), ("model", "wrong"), ("strategy", "callers")):
            data = result()
            data["receipt"]["passes"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(e.ReviewAbort):
                e.validate_result(data, POLICY, BUNDLE)

    def test_json_envelope_and_offline_replay_regressions(self):
        fixtures = Path(__file__).parent / "fixtures/agentic"
        for bad in ((fixtures / "fenced-truncated.txt").read_text(), '{} {}', '{"findings":[],"findings":[]}',
                    '{"findings":', '{"cost":NaN}', '```json\n{"findings":[]}\n```'):
            with self.subTest(raw=bad), self.assertRaises(e.ReviewAbort):
                e.strict_json(bad)
        with self.assertRaisesRegex(e.ReviewAbort, "Missing expected"):
            e.validate_result(e.strict_json((fixtures / "empty-receipt.json").read_text()), POLICY, BUNDLE)

    def test_merge_by_location_and_mechanism_preserves_distinct_defects(self):
        first = {"findings": [finding(confidence="low")], "receipt": receipt()}
        second = {"findings": [finding(mechanism="  RESOLVED state  is not copied "),
                               finding(mechanism="Cancellation interrupts the transfer")], "receipt": receipt("callers")}
        merged = e.merge_results([first, second])
        self.assertEqual(2, len(merged["findings"]))
        self.assertTrue(all(f["confidence"] == "high" for f in merged["findings"]))
        e.validate_result(merged, POLICY, BUNDLE)


class ClaudeTests(unittest.TestCase):
    def test_native_read_receipt_and_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            parsed = e.claude_result(transcript(), POLICY, BUNDLE, "bundle", Path(tmp))
        self.assertEqual([], parsed["findings"])
        self.assertEqual(["src/app.py"], parsed["receipt"]["files_opened"])

    def test_reading_unsafe_paths_inside_head_does_not_discard_the_pass(self):
        events = transcript()
        events[1:1] = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "git-1", "name": "Read", "input": {"file_path": ".git/HEAD"}}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "git-1", "is_error": False, "content": "ref"}]}}]
        with tempfile.TemporaryDirectory() as tmp:
            parsed = e.claude_result(events, POLICY, BUNDLE, "bundle", Path(tmp))
        self.assertEqual(["src/app.py"], parsed["receipt"]["files_opened"])
        events = [e for e in events if e.get("type") != "user" or e["message"]["content"][0]["tool_use_id"] != "read-1"]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(e.ReviewAbort, "never opened"):
            e.claude_result(events, POLICY, BUNDLE, "bundle", Path(tmp))

    def test_provider_stops_invalid_schema_model_drift_and_tools_fail_closed(self):
        for subtype in ("error_max_turns", "error_max_budget_usd", "error_during_execution", "error_max_structured_output_retries"):
            events = transcript()
            events[-1]["subtype"] = subtype
            with self.subTest(subtype=subtype), self.assertRaises(e.ReviewAbort):
                e.claude_result(events, POLICY, BUNDLE, "bundle", Path("/head"))
        variants = []
        events = transcript(); events[-1].pop("structured_output"); variants.append(events)
        events = transcript(); events[-1]["modelUsage"] = {"other-model": {}}; variants.append(events)
        events = transcript(); events[0]["message"]["content"][0]["name"] = "Bash"; variants.append(events)
        events = transcript(); events[0]["message"]["content"][0]["name"] = "mcp__github__write"; variants.append(events)
        events = transcript(); events[1]["message"]["content"][0]["is_error"] = True; variants.append(events)
        variants.extend([transcript()[:-1], transcript() + [transcript()[-1]], transcript() + [{"type": "extra"}]])
        for events in variants:
            with self.subTest(events=events), self.assertRaises(e.ReviewAbort):
                e.claude_result(events, POLICY, BUNDLE, "bundle", Path("/head"))

    def test_dry_cli_forwards_restrictions_budget_schema_and_no_github_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args_file = root / "args.json"
            cli = root / "claude"
            cli.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                           f"open({str(args_file)!r},'w').write(json.dumps({{'args':sys.argv[1:],'cwd':os.getcwd(),'env':list(os.environ)}}))\n"
                           "sys.stdin.read()\n" +
                           "\n".join("print(" + json.dumps(event).__repr__() + ")" for event in transcript()))
            cli.chmod(0o700)
            with patch.dict(os.environ, {"PATH": str(root) + os.pathsep + os.environ["PATH"],
                                         "GITHUB_TOKEN": "fixture-not-forwarded", "ORG_MEMBERS_TOKEN": "fixture-not-forwarded"}):
                actual = e.run_claude(POLICY, BUNDLE, "bundle", root)
            observed = json.loads(args_file.read_text())
            args = observed["args"]
            self.assertEqual("Read,Grep,Glob", args[args.index("--tools") + 1])
            self.assertEqual('{"mcpServers":{}}', args[args.index("--mcp-config") + 1])
            for required in ("--restricted", "--strict-mcp-config", "--disable-slash-commands", "--json-schema"):
                self.assertIn(required, args)
            self.assertIn("mcp__*", args[args.index("--disallowedTools") + 1])
            self.assertEqual(str(POLICY["max_turns"]), args[args.index("--max-turns") + 1])
            self.assertEqual(str(POLICY["max_budget_usd"]), args[args.index("--max-budget-usd") + 1])
            self.assertNotIn("GITHUB_TOKEN", observed["env"])
            self.assertNotIn("ORG_MEMBERS_TOKEN", observed["env"])
            self.assertEqual(str(root), observed["cwd"])
            self.assertEqual([], actual["findings"])

    def test_nonzero_exit_and_timeout_are_fixed_errors_without_retry(self):
        from types import SimpleNamespace as NS
        for value in (NS(returncode=1, stdout="", stderr="private provider body"),
                      subprocess.TimeoutExpired(["claude"], 1)):
            with patch.object(e.subprocess, "run", side_effect=value if isinstance(value, Exception) else None,
                              return_value=value) as run, self.assertRaises(e.ReviewAbort):
                e.run_claude(POLICY, BUNDLE, "bundle", Path("/head"))
            self.assertEqual(1, run.call_count)


class CodexTests(unittest.TestCase):
    def test_native_rollout_usage_read_success_and_per_pass_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            sessions = home / "sessions"; sessions.mkdir()
            events = [
              {"type": "turn_context", "payload": {"model": "gpt-6-luna"}},
              {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "call_id": "r",
               "arguments": json.dumps({"cmd": "cat -- src/app.py", "workdir": "/head"})}},
              {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "r",
               "output": "Process exited with code 0\nsource"}},
              {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": {
                  "input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}}}]
            (sessions / "rollout.jsonl").write_text("\n".join(json.dumps(v) for v in events))
            turns, usage, opened, cost = e.codex_progress(home, Path("/head"), model="gpt-6-luna")
            self.assertEqual(1, turns)
            self.assertEqual(["src/app.py"], opened)
            self.assertAlmostEqual((100 * 0.10 + 20 * 0.50) / 1_000_000, cost)
            # Prices from https://developers.openai.com/api/docs/pricing , per 1M tokens.
            for model, rate_in, rate_out in (("gpt-6.1-sol", 2.00, 10.00), ("gpt-6-astra", 10.00, 50.00),
                                             ("gpt-6.1-terra", 1.00, 5.00)):
                self.assertAlmostEqual((100 * rate_in + 20 * rate_out) / 1_000_000,
                                       e.codex_progress(home, Path("/head"), model=model)[3])
            # A model without a sourced price never yields a cost, so the receipt is refused.
            self.assertIsNone(e.codex_progress(home, Path("/head"), model="unpriced")[3])
            self.assertEqual(opened, e.codex_progress(home, Path("/head"), strict=True, model="gpt-6-luna")[2])
            with self.assertRaisesRegex(e.ReviewAbort, "configured model"):
                e.codex_progress(home, Path("/head"), strict=True, model="other")
            # Repeated token totals do not count as new inference.
            with (sessions / "rollout.jsonl").open("a") as f: f.write("\n"+json.dumps(events[-1]))
            self.assertEqual(1, e.codex_progress(home, Path("/head"))[0])

    def test_option_words_in_a_cat_read_are_not_receipt_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "sessions").mkdir()
            events = [
              {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "call_id": "r",
               "arguments": json.dumps({"cmd": "cat -n src/app.py", "workdir": "/head"})}},
              {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "r",
               "output": "Process exited with code 0\n1 source"}}]
            (home / "sessions/rollout.jsonl").write_text("\n".join(json.dumps(v) for v in events))
            self.assertEqual(["src/app.py"], e.codex_progress(home, Path("/head"))[2])

    def test_truncated_or_missing_native_rollout_is_not_a_complete_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with self.assertRaises(e.ReviewAbort):
                e.codex_progress(home, Path("/head"), strict=True, model="gpt-6-luna")
            (home / "sessions").mkdir()
            (home / "sessions/rollout.jsonl").write_text('{"type":')
            with self.assertRaisesRegex(e.ReviewAbort, "truncated"):
                e.codex_progress(home, Path("/head"), strict=True, model="gpt-6-luna")

    def test_action_home_disables_request_and_stream_retries_without_pr_configuration(self):
        import tomllib
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            config = home / "config.toml"
            config.write_text('model_provider="codex-action-responses-proxy"\n'
                              'default_permissions=":read-only"\n'
                              '[model_providers.codex-action-responses-proxy]\n'
                              'name="Codex Action Responses Proxy"\n'
                              'base_url="http://127.0.0.1:12345/v1"\nwire_api="responses"\n')
            e.configure_codex(home, POLICY)
            data = tomllib.loads(config.read_text())
            self.assertEqual(POLICY["max_model_tokens"], data["model_context_window"])
            provider = data["model_providers"]["codex-action-responses-proxy"]
            self.assertEqual(0, provider["request_max_retries"])
            self.assertEqual(0, provider["stream_max_retries"])
            config.write_text('model_provider="untrusted"')
            with self.assertRaisesRegex(e.ReviewAbort, "no inference"):
                e.configure_codex(home, POLICY)

    def test_codex_runner_never_inherits_the_anthropic_credential(self):
        seen = {}
        def capture(command, **kwargs):
            seen.update(kwargs["env"])
            raise RuntimeError("stop before inference")
        with tempfile.TemporaryDirectory() as tmp,                 patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fixture-not-forwarded"}),                 patch.object(e, "configure_codex"), patch.object(e.subprocess, "Popen", side_effect=capture),                 self.assertRaises(RuntimeError):
            e.run_codex(dict(POLICY, model="gpt-6-luna", provider="codex"), {"files": []}, "callers",
                        Path(tmp), Path(tmp), Path(tmp), Path(tmp))
        self.assertIn("CODEX_HOME", seen)
        self.assertNotIn("ANTHROPIC_API_KEY", seen)

    def test_action_runner_enforces_read_only_and_schema(self):
        command = e.codex_command(dict(POLICY, model="gpt-6-luna"), Path("/head"), Path("/trusted-home"),
                                  Path("/schema.json"), Path("/output.json"), Path("/pinned-action"))
        self.assertEqual(":read-only", command[command.index("--permission-profile")+1])
        self.assertEqual("/schema.json", command[command.index("--output-schema-file")+1])
        self.assertEqual("drop-sudo", command[command.index("--safety-strategy")+1])
        # The workspace root holds the trusted base checkout; reads must resolve in the PR head.
        self.assertEqual("/head", command[command.index("--cd")+1])
        self.assertNotIn("--full-auto", command)
        with self.assertRaisesRegex(e.ReviewAbort, "no sourced price"):
            e.codex_command(dict(POLICY, model="unknown"), Path("/head"), Path("/home"),
                            Path("/schema"), Path("/out"), Path("/action"))
        for model in ("gpt-6.1-sol", "gpt-6-astra"):
            command = e.codex_command(dict(POLICY, model=model), Path("/head"), Path("/home"),
                                      Path("/schema"), Path("/out"), Path("/action"))
            self.assertEqual(model, command[command.index("--model")+1])
        # Above 272K input tokens OpenAI bills long-context rates the estimator does not model.
        self.assertIsNotNone(e.codex_command(dict(POLICY, model="gpt-6.1-sol", max_model_tokens=272_000),
                                             Path("/head"), Path("/home"), Path("/schema"), Path("/out"), Path("/action")))
        with self.assertRaisesRegex(e.ReviewAbort, "short-context"):
            e.codex_command(dict(POLICY, model="gpt-6.1-sol", max_model_tokens=272_001), Path("/head"),
                            Path("/home"), Path("/schema"), Path("/out"), Path("/action"))


class VacuityTests(unittest.TestCase):
    def test_schema_and_receipt_negative_tests_fail_when_each_guard_is_neutralized(self):
        cases = (("validate_schema", "test_invalid_finding_schema_is_rejected"),
                 ("validate_receipt", "test_receipt_check_rejects_unopened_changed_file"))
        for guard, method in cases:
            with patch.object(e, guard, return_value=[]):
                suite = unittest.TestSuite([ResultTests(method)])
                result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
            self.assertFalse(result.wasSuccessful(), guard + " mutant was not detected")
            self.assertTrue(result.failures, guard)
            print("Vacuity: neutralizing " + guard + " caused the negative test to fail")


if __name__ == "__main__":
    unittest.main()
