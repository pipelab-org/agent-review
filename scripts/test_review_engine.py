# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline provider, Git coverage, malformed response and receipt regressions."""
import copy
import io
import json
import math
import os
from pathlib import Path
import re
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
        approved = {"default": (40, 5.0), "deep": (40, 9.0), "codex": (40, 2.0), "codex-deep": (40, 2.0)}
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

    def test_depth_allowance_overrides_shared_input_allowance(self):
        # Shipped: Claude depths take the larger allowance, Codex depths keep the shared 272000.
        shipped = {depth: e.policy(depth)["max_model_tokens"] for depth in e.DEPTHS}
        self.assertEqual({"default": 516_000, "deep": 516_000, "codex": 272_000, "codex-deep": 272_000}, shipped)
        data = self.data()
        data["config"]["max_model_tokens"] = 272_000
        data["review_depth"]["default"]["max_model_tokens"] = 516_000
        self.assertEqual(516_000, e.policy("default", data)["max_model_tokens"])
        self.assertEqual(272_000, e.policy("deep", data)["max_model_tokens"])
        for bad in (0, -1, True, "516000", 1.5, None):
            data["review_depth"]["default"]["max_model_tokens"] = bad
            with self.subTest(value=bad), self.assertRaisesRegex(
                    e.ReviewAbort, "review_depth.default.max_model_tokens"):
                e.policy("default", data)
        data["review_depth"]["default"]["max_model_tokens"] = e.CLAUDE_CONTEXT_TOKENS
        self.assertEqual(e.CLAUDE_CONTEXT_TOKENS, e.policy("default", data)["max_model_tokens"])
        data["review_depth"]["default"]["max_model_tokens"] = e.CLAUDE_CONTEXT_TOKENS + 1
        with self.assertRaisesRegex(e.ReviewAbort, "Claude context window"):
            e.policy("default", data)
        data["review_depth"]["default"]["max_model_tokens"] = data["config"]["max_output_tokens"]
        with self.assertRaisesRegex(e.ReviewAbort, "max_output_tokens below"):
            e.policy("default", data)
        # A raised Codex depth still fails before inference instead of billing long-context rates.
        data = self.data()
        data["review_depth"]["codex"]["max_model_tokens"] = 516_000
        data["review_depth"]["codex"]["model"] = "gpt-6.1-terra"
        settings = e.policy("codex", data)
        with self.assertRaisesRegex(e.ReviewAbort, "short-context"):
            e.codex_command(settings, Path("/head"), Path("/home"), Path("/schema"), Path("/out"), Path("/action"))

    def test_documented_allowances_match_shipped_policy(self):
        expected = {depth: e.policy(depth)["max_budget_usd"] for depth in e.DEPTHS}
        for path in ("README.md", "docs/agent-review.md"):
            with self.subTest(path=path):
                text = (e.ROOT / path).read_text()
                entries = re.findall(
                    r"\bUSD (\d+(?:\.\d+)?) per (default|deep|codex|codex-deep) pass\b", text)
                self.assertEqual(len(expected), len(entries), "Document each depth's allowance once")
                self.assertEqual(expected, {depth: float(amount) for amount, depth in entries})

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

    def test_operator_instructions_count_toward_the_bundle_allowance(self):
        lines = [f"line {i}" for i in range(3000)]
        (self.root / "app.py").write_text("\n".join(lines) + "\n")
        self.base = self.commit()
        lines[1500] = "changed"
        (self.root / "app.py").write_text("\n".join(lines) + "\n")
        self.commit()
        limited = dict(POLICY, max_model_tokens=40000, max_output_tokens=1000)  # fixture operator budget
        self.assertIsNotNone(self.bundle(limited)["files"][0]["content"])
        instructions = "Check every guard. " * 500
        bundle = self.bundle(limited, instructions=instructions)
        # Instructions that no longer leave room for the whole file demote it to regions,
        # and every prompt the passes send then fits instead of failing at launch.
        self.assertIsNone(bundle["files"][0]["content"])
        available = limited["max_model_tokens"] - limited["max_output_tokens"]
        for prompt in (e.claude_prompt(bundle, "bundle"), e.claude_prompt(bundle, "callers"), e.codex_prompt(bundle)):
            self.assertLessEqual(e.token_bound(prompt + json.dumps(e.OUTPUT_SCHEMA)), available)

    def test_each_pass_gets_its_own_focus_and_codex_gets_both(self):
        (self.root / "app.py").write_text("changed\n")
        self.commit()
        bundle = self.bundle()
        first, second, codex = e.claude_prompt(bundle, "bundle"), e.claude_prompt(bundle, "callers"), e.codex_prompt(bundle)
        self.assertIn(e.BOUNDARY_FOCUS, first)
        self.assertNotIn(e.LIFECYCLE_FOCUS, first)
        self.assertIn(e.LIFECYCLE_FOCUS, second)
        self.assertNotIn(e.BOUNDARY_FOCUS, second)
        self.assertIn(e.BOUNDARY_FOCUS, codex)
        self.assertIn(e.LIFECYCLE_FOCUS, codex)

    def test_prompts_state_the_schema_findings_cap(self):
        (self.root / "app.py").write_text("changed\n")
        self.commit()
        bundle = self.bundle()
        cap = f"Return at most {e.MAX_FINDINGS} findings"
        self.assertEqual(e.MAX_FINDINGS, e.OUTPUT_SCHEMA["properties"]["findings"]["maxItems"])
        for prompt in (e.claude_prompt(bundle, "bundle"), e.claude_prompt(bundle, "callers"), e.codex_prompt(bundle)):
            self.assertIn(cap, prompt)

    def test_prompts_carry_operator_instructions_once(self):
        (self.root / "app.py").write_text("changed\n")
        self.commit()
        bundle = self.bundle(instructions="UNIQUE-OPERATOR-CRITERIA")
        for prompt in (e.claude_prompt(bundle, "bundle"), e.claude_prompt(bundle, "callers"), e.codex_prompt(bundle)):
            self.assertEqual(1, prompt.count("UNIQUE-OPERATOR-CRITERIA"))
            self.assertIn("Operator review criteria:", prompt)

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

    def test_receipt_check_rejects_unsafe_opened_path(self):
        data = result()
        data["receipt"]["passes"][0]["files_opened"] = ["../key"]
        with self.assertRaisesRegex(e.ReviewAbort, "receipt is invalid"):
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
        with tempfile.TemporaryDirectory() as tmp:
            parsed = e.claude_result(events, POLICY, BUNDLE, "bundle", Path(tmp))
        # A failed or missing read is not counted, and the summary reports the file.
        self.assertEqual([], parsed["receipt"]["files_opened"])
        self.assertEqual(["src/app.py"], e.unopened_regions(BUNDLE, [parsed["receipt"]]))

    def test_text_message_events_are_tolerated_without_crediting_reads(self):
        events = transcript()
        # Claude Code stream-json can emit an event whose message is plain text.
        events[1:1] = [{"type": "system", "message": "context compacted"}]
        with tempfile.TemporaryDirectory() as tmp:
            parsed = e.claude_result(events, POLICY, BUNDLE, "bundle", Path(tmp))
        self.assertEqual(["src/app.py"], parsed["receipt"]["files_opened"])

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
        failed_read = transcript(); failed_read[1]["message"]["content"][0]["is_error"] = True
        with tempfile.TemporaryDirectory() as tmp:
            parsed = e.claude_result(failed_read, POLICY, BUNDLE, "bundle", Path(tmp))
        self.assertEqual([], parsed["receipt"]["files_opened"])
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


class CoverageTests(unittest.TestCase):
    def test_unopened_region_files_are_reported_not_refused(self):
        bundle = {"files": [{"path": "src/full.py", "deleted": False, "content": "x = 1\n"},
                            {"path": "src/big.py", "deleted": False, "content": None},
                            {"path": "src/gone.py", "deleted": True, "content": None}]}
        r = receipt("bundle")
        r["files_opened"] = []
        # Every changed line is in the bundle, so the pass stands.
        self.assertEqual(r, e.validate_receipt(r, POLICY, bundle, "bundle"))
        # Only region-only, undeleted files that no pass opened are reported.
        self.assertEqual(["src/big.py"], e.unopened_regions(bundle, [r]))
        other = dict(r, files_opened=["src/big.py"])
        self.assertEqual([], e.unopened_regions(bundle, [r, other]))
        self.assertEqual([], e.unopened_regions({"files": [bundle["files"][0]]}, [r]))


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
               "output": "Process exited with code 0\nOutput:\nsource"}},
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

    def test_zsh_wrapper_never_supplies_a_read(self):
        for shell in ("zsh", "/bin/zsh"):
            self.assertIsNone(e.codex_plain_read([shell, "-lc", "cat -- src/app.py"]))
        self.assertEqual("src/app.py", e.codex_plain_read(["/bin/bash", "-lc", "cat -- src/app.py"]))

    def test_codex_error_cause_uses_codex_classification_never_provider_text(self):
        secret = "provider body req_fixture-only"
        self.assertEqual("OpenAI rate limit; request the review again after it resets "
                         "(codex_error_info=rate_limit_exceeded); no paid retry",
                         e.codex_error_cause({"message": secret, "codex_error_info": "rate_limit_exceeded"}))
        disconnected = e.codex_error_cause({"message": secret, "codex_error_info": {
            "response_stream_disconnected": {"http_status_code": 502}}})
        self.assertIn("codex_error_info=response_stream_disconnected, HTTP 502", disconnected)
        # Status is a published integer range; anything else is dropped, not echoed.
        for code in (None, 99, 600, "502", True):
            cause = e.codex_error_cause({"codex_error_info": {"http_connection_failed": {"http_status_code": code}}})
            self.assertTrue(cause.startswith("OpenAI connection failed"))
            self.assertNotIn("HTTP", cause)
        self.assertIn("unclassified Codex error", e.codex_error_cause({"codex_error_info": "other"}))
        for value in ({"codex_error_info": "not_a_variant"}, {"codex_error_info": {"a": {}, "b": {}}},
                      {"codex_error_info": ["rate_limit_exceeded"]}, {"message": secret}, None, "other"):
            self.assertIsNone(e.codex_error_cause(value))
        for cause in map(e.codex_error_cause, ({"codex_error_info": k} for k in e.CODEX_ERROR_CAUSES)):
            self.assertNotIn(secret, cause)

    def test_codex_failure_cause_reads_the_last_recorded_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            self.assertIsNone(e.codex_failure_cause(home))
            (home / "sessions/2026/10/30").mkdir(parents=True)
            events = [
                {"type": "event_msg", "payload": {"type": "error", "message": "x", "codex_error_info": "server_overloaded"}},
                {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": None,
                 "error": {"message": "stream disconnected before completion", "codex_error_info": "unauthorized"}}},
            ]
            (home / "sessions/2026/10/30/rollout.jsonl").write_text(
                "\n".join(json.dumps(v) for v in events) + "\n{truncated")
            cause = e.codex_failure_cause(home)
            # First classified error, matching codex_progress, which aborts on the first error event.
            self.assertTrue(cause.startswith("OpenAI server overloaded"))
            self.assertNotIn("stream disconnected", cause)
            # A turn that completed without an error object leaves no cause.
            (home / "sessions/2026/10/30/rollout.jsonl").write_text(json.dumps(
                {"type": "event_msg", "payload": {"type": "task_complete", "last_agent_message": "done"}}))
            self.assertIsNone(e.codex_failure_cause(home))

    def test_codex_failure_cause_never_raises_on_bad_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "sessions").mkdir()
            classified = json.dumps({"type": "event_msg", "payload": {"type": "error", "codex_error_info": "unauthorized"}})
            (home / "sessions/a.jsonl").write_text('null\n[]\n5\n"text"\n' + "[" * 100000 + "\n")
            (home / "sessions/b.jsonl").write_bytes(b"\xff\xfe not utf-8\n")
            self.assertIsNone(e.codex_failure_cause(home))
            (home / "sessions/c.jsonl").write_text(classified)
            self.assertTrue(e.codex_failure_cause(home).startswith("OpenAI rejected the API key"))
            (home / "sessions/d.jsonl").mkdir()  # an unreadable entry matching the glob
            self.assertTrue(e.codex_failure_cause(home).startswith("OpenAI rejected the API key"))
            # A rollout from an earlier pass in a shared home is not this pass's cause.
            self.assertIsNone(e.codex_failure_cause(home, frozenset({home / "sessions/c.jsonl"})))

    def test_codex_progress_names_the_classified_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "sessions").mkdir()
            rollout = home / "sessions/rollout.jsonl"
            rollout.write_text(json.dumps({"type": "event_msg", "payload": {
                "type": "error", "message": "private", "codex_error_info": "rate_limit_exceeded"}}))
            with self.assertRaisesRegex(e.ReviewAbort, r"OpenAI rate limit.*codex_error_info=rate_limit_exceeded"):
                e.codex_progress(home, Path("/head"))
            rollout.write_text(json.dumps({"type": "event_msg", "payload": {"type": "error", "message": "private"}}))
            with self.assertRaisesRegex(e.ReviewAbort, "^Codex reported an API or runtime error; pass incomplete$"):
                e.codex_progress(home, Path("/head"))

    def test_failed_codex_exit_reports_the_recorded_cause(self):
        settings = dict(POLICY, model="gpt-6.1-terra", provider="codex")
        with tempfile.TemporaryDirectory() as tmp:
            home, workspace = Path(tmp) / "home", Path(tmp) / "work"
            (home / "sessions").mkdir(parents=True); workspace.mkdir()
            python = os.environ.get("PYTHON", "python3")
            failing = [python, "-c", "import sys; sys.exit(1)"]
            # Like Codex, this process writes its own rollout during the pass, then fails.
            recorded = json.dumps({"type": "event_msg", "payload": {"type": "task_complete", "error": {
                "message": "private", "codex_error_info": {"response_stream_disconnected": {"http_status_code": None}}}}})
            writes_then_fails = [python, "-c", "import pathlib, sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); "
                                 "sys.exit(1)", str(home / "sessions/rollout.jsonl"), recorded]
            # A rollout left by an earlier pass in the shared home.
            (home / "sessions/earlier.jsonl").write_text(json.dumps({"type": "event_msg", "payload": {
                "type": "task_complete", "error": {"codex_error_info": "unauthorized"}}}))
            with patch.object(e, "configure_codex"):
                with patch.object(e, "codex_command", return_value=failing), \
                        self.assertRaisesRegex(e.ReviewAbort, "^Codex action failed; inspect provider account"):
                    e.run_codex(settings, {"files": []}, "callers", Path(tmp), "/action", home, workspace)
                with patch.object(e, "codex_command", return_value=writes_then_fails), \
                        self.assertRaisesRegex(e.ReviewAbort,
                                               r"^OpenAI response stream disconnected; request the review again "
                                               r"\(codex_error_info=response_stream_disconnected\); no paid retry$"):
                    e.run_codex(settings, {"files": []}, "callers", Path(tmp), "/action", home, workspace)

    def test_single_request_above_the_allowance_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "sessions").mkdir()
            event = {"type": "event_msg", "payload": {"type": "token_count", "info": {
                "total_token_usage": {"input_tokens": 300_000, "output_tokens": 10, "total_tokens": 300_010},
                "last_token_usage": {"input_tokens": 300_000, "output_tokens": 10, "total_tokens": 300_010}}}}
            (home / "sessions/rollout.jsonl").write_text(json.dumps(event))
            with self.assertRaisesRegex(e.ReviewAbort, "exceeded the review depth's max_model_tokens"):
                e.codex_progress(home, Path("/head"), model="gpt-6.1-terra", request_limit=256_000)
            # At the allowance it is accepted; without a limit (old callers) it is unchanged.
            self.assertEqual(1, e.codex_progress(home, Path("/head"), model="gpt-6.1-terra",
                                                 request_limit=300_000)[0])
            self.assertEqual(1, e.codex_progress(home, Path("/head"), model="gpt-6.1-terra")[0])

    def test_each_completed_response_requires_fresh_usage(self):
        # Native 0.165.0 retains prior counters when response.completed omits usage.
        # session/turn.rs emits token_count after response output and pending tools.
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "sessions").mkdir()
            rollout = home / "sessions/rollout.jsonl"
            response = {"type": "response_item", "payload": {
                "type": "message", "role": "assistant", "content": []}}
            usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
            count = {"type": "event_msg", "payload": {"type": "token_count", "info": {
                "last_token_usage": usage, "total_token_usage": usage}}}
            for events in ([response, {"type": "event_msg", "payload": {
                                "type": "token_count", "info": None}}],
                           [response, count, response, count]):
                rollout.write_text("\n".join(map(json.dumps, events)))
                with self.assertRaisesRegex(e.ReviewAbort, "fresh token usage"):
                    e.codex_progress(home, Path("/head"), request_limit=256_000)
            context = {"type": "turn_context", "payload": {"model": "gpt-6.1-terra"}}
            rollout.write_text("\n".join(map(json.dumps, [context, response])))
            with self.assertRaisesRegex(e.ReviewAbort, "fresh token usage"):
                e.codex_progress(home, Path("/head"), strict=True,
                                 model="gpt-6.1-terra", request_limit=256_000)
            for kind in ("function_call", "custom_tool_call", "reasoning", "local_shell_call",
                         "tool_search_call", "web_search_call", "image_generation_call", "compaction"):
                item = {"type": "response_item", "payload": {"type": kind}}
                rollout.write_text("\n".join(map(json.dumps, [response, count, item, count])))
                with self.assertRaisesRegex(e.ReviewAbort, "fresh token usage", msg=kind):
                    e.codex_progress(home, Path("/head"), request_limit=256_000)
            # Rate-limit updates and compaction may repeat counters without new output.
            events = [response, count, count, {"type": "compacted", "payload": {}}, count]
            rollout.write_text("\n".join(map(json.dumps, events)))
            self.assertEqual(1, e.codex_progress(home, Path("/head"), request_limit=256_000)[0])
            # A second response with fresh usage, including multiple output items, is valid.
            events = [response, count, response, {"type": "response_item", "payload": {
                "type": "custom_tool_call", "call_id": "next", "name": "exec", "input": ""}},
                {"type": "event_msg", "payload": {"type": "token_count", "info": {
                    "last_token_usage": usage, "total_token_usage": {
                        "input_tokens": 200, "output_tokens": 40, "total_tokens": 240}}}}]
            rollout.write_text("\n".join(map(json.dumps, events)))
            self.assertEqual(2, e.codex_progress(home, Path("/head"), request_limit=256_000)[0])

    def test_cached_input_bills_at_the_cached_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "sessions").mkdir()
            usage = {"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 10, "total_tokens": 1010}
            (home / "sessions/rollout.jsonl").write_text(json.dumps(
                {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": usage}}}))
            # gpt-6.1-terra per 1M: input 1.00, cached input 0.05, output 5.00 (OpenAI pricing page).
            self.assertAlmostEqual((200 * 1.00 + 800 * 0.05 + 10 * 5.00) / 1_000_000,
                                   e.codex_progress(home, Path("/head"), model="gpt-6.1-terra")[3])
            # A cached count above the input total cannot make the input charge negative.
            usage["cached_input_tokens"] = 5000
            (home / "sessions/rollout.jsonl").write_text(json.dumps(
                {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": usage}}}))
            self.assertAlmostEqual((1000 * 0.05 + 10 * 5.00) / 1_000_000,
                                   e.codex_progress(home, Path("/head"), model="gpt-6.1-terra")[3])

    def test_code_mode_command_items_supply_reads_only_when_successful(self):
        def item(path, **over):
            data = {"type": "CommandExecution", "command": ["/bin/bash", "-lc", "sed -n '1,9p' " + path],
                    "cwd": "file:///head", "status": "completed", "exit_code": 0,
                    "parsed_cmd": [{"type": "read", "cmd": "sed -n '1,9p' " + path, "name": path, "path": path}],
                    "aggregated_output": "SOURCE_CONTENT"}
            data.update(over)
            return {"type": "event_msg", "payload": {"type": "item_completed", "item": data}}
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "sessions").mkdir()
            events = [{"type": "response_item", "payload": {"type": "custom_tool_call_output",
                       "output": [{"type": "input_text", "text": "Script completed\nOutput:\n"},
                                  {"type": "input_text", "text": "SOURCE_CONTENT"}]}},
                      item("src/ok.py"), item("src/failed.py", exit_code=1),
                      item("src/declined.py", status="declined"),
                      item("src/elsewhere.py", cwd="file:///other"),
                      item("../outside.py"),
                      item("src/search.py", parsed_cmd=[{"type": "search", "cmd": "rg x", "path": "src/search.py"}]),
                      item("src/nested.py", cwd="file:///head/src", command=["/bin/bash", "-lc", "cat nested.py"], parsed_cmd=[
                          {"type": "read", "cmd": "cat nested.py", "name": "nested.py", "path": "nested.py"}]),
                      {"type": "response_item", "payload": {"type": "item_completed", "item": item("src/x.py")}}]
            (home / "sessions/rollout.jsonl").write_text("\n".join(json.dumps(v) for v in events))
            self.assertEqual(["src/nested.py", "src/ok.py"], e.codex_progress(home, Path("/head"))[2])

    def test_expanded_or_reinterpreted_operands_are_not_literal_reads(self):
        # Bash expands these words before cat runs, so the literal operand is not
        # necessarily the file that was read. Quoted forms are literal and stay admitted.
        for script in ("cat -- pages/[id].tsx", "cat -- a?", "cat -- {x,y}", "cat -- ~",
                       "cat -- #a", "cat -- ;", "cat -- *", "sed -n 1,5p [ab]",
                       "cat -- 'a", "cat -- a'b"):
            with self.subTest(script=script):
                self.assertIsNone(e.codex_plain_read(["/bin/bash", "-lc", script]))
        for script, operand in (("cat -- 'pages/[id].tsx'", "pages/[id].tsx"),
                                ('cat -- "~"', "~"), ("cat -- a\\*", "a*"),
                                ("sed -n '1,9p' 'x{y}'", "x{y}"), ("cat -- a#b", "a#b")):
            with self.subTest(script=script):
                self.assertEqual(operand, e.codex_plain_read(["/bin/bash", "-lc", script]))

    def test_assignment_word_tilde_expansion_cannot_credit_the_literal_file(self):
        # Bash Tilde Expansion also applies after = and : in assignment words
        # passed as command arguments, not just at the beginning of a word.
        with tempfile.TemporaryDirectory() as tmp:
            head = Path(tmp)
            home = head / "home"
            (home / "sessions").mkdir(parents=True)
            for operand in ("x=~/y", "x=a:~/y"):
                expanded = operand.replace("~", "/tmp/target")
                for name, content in ((operand, "UNREAD_LITERAL"), (expanded, "ACTUAL_OTHER_FILE")):
                    path = head / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content)
                script = "cat -- " + operand
                actual = subprocess.run(["/bin/bash", "-c", script], cwd=head,
                                        env={"PATH": os.environ["PATH"], "HOME": "/tmp/target"},
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(0, actual.returncode, actual.stderr)
                self.assertEqual("ACTUAL_OTHER_FILE", actual.stdout)
                item = {"type": "CommandExecution", "command": ["/bin/bash", "-c", script],
                        "cwd": head.as_uri(), "status": "completed", "exit_code": 0,
                        "parsed_cmd": [{"type": "read", "path": operand}]}
                code_mode = [{"type": "event_msg", "payload": {"type": "item_completed", "item": item}}]
                legacy = [
                    {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                     "call_id": "r", "arguments": json.dumps({"cmd": script, "workdir": str(head)})}},
                    {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "r",
                     "output": "Process exited with code 0\nOutput:\n" + actual.stdout}}]
                for events in (code_mode, legacy):
                    with self.subTest(operand=operand, surface=events[0]["type"]):
                        (home / "sessions/rollout.jsonl").write_text("\n".join(map(json.dumps, events)))
                        self.assertEqual([], e.codex_progress(home, head)[2])
                self.assertIsNone(e.codex_plain_read(["/bin/bash", "-lc", script]))
                self.assertEqual(operand, e.codex_plain_read(["/bin/bash", "-lc", "cat -- '" + operand + "'"]))

    def test_code_mode_skipped_and_hidden_reads_do_not_supply_coverage(self):
        # Codex 0.165.0 native fixture: only ALREADY is printed, but both
        # reads occur in parsed_cmd. The second command never executes.
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "sessions").mkdir()
            item = {"type": "CommandExecution", "cwd": "file:///head", "status": "completed",
                    "exit_code": 0, "aggregated_output": "SOURCE_CONTENT", "parsed_cmd": [
                        {"type": "read", "cmd": "cat -- already.py", "path": "already.py"},
                        {"type": "read", "cmd": "cat -- unread.py", "path": "unread.py"}]}
            for command in ("cat -- already.py || cat -- unread.py",
                            "cat -- unread.py > /dev/null",
                            "cat -- already.py; cat -- unread.py",
                            "cat -- unread.py | head -n 0",
                            "cat -- already.py\ncat -- unread.py",
                            "cat --help unread.py"):
                with self.subTest(command=command):
                    item["command"] = ["/bin/bash", "-lc", command]
                    event = {"type": "event_msg", "payload": {"type": "item_completed", "item": item}}
                    (home / "sessions/rollout.jsonl").write_text(json.dumps(event))
                    self.assertEqual([], e.codex_progress(home, Path("/head"))[2])
                    # Legacy function calls must not bypass the same guard, even
                    # when operators are adjacent to their operands.
                    legacy = [
                        {"type": "response_item", "payload": {"type": "function_call",
                         "name": "exec_command", "call_id": "r", "arguments": json.dumps(
                             {"cmd": command.replace(" || ", "||"), "workdir": "/head"})}},
                        {"type": "response_item", "payload": {"type": "function_call_output",
                         "call_id": "r", "output": "Process exited with code 0\nOutput:\nALREADY"}}]
                    (home / "sessions/rollout.jsonl").write_text("\n".join(map(json.dumps, legacy)))
                    self.assertEqual([], e.codex_progress(home, Path("/head"))[2])
            item["command"] = ["/head/bash", "-lc", "cat -- unread.py"]
            event = {"type": "event_msg", "payload": {"type": "item_completed", "item": item}}
            (home / "sessions/rollout.jsonl").write_text(json.dumps(event))
            self.assertEqual([], e.codex_progress(home, Path("/head"))[2])
            for command in (["/bin/bash", "-lc", "cat -- unread.py"],
                            ["/bin/bash", "-lc", "sed -n '1,9p' unread.py"],
                            ["cat", "--", "unread.py"]):
                item["command"] = command
                event = {"type": "event_msg", "payload": {"type": "item_completed", "item": item}}
                (home / "sessions/rollout.jsonl").write_text(json.dumps(event) + "\n" + json.dumps(
                    {"type": "response_item", "payload": {"type": "custom_tool_call_output",
                     "output": "Script completed\nWall time 0.1 seconds\nOutput:\nSOURCE_CONTENT"}}))
                self.assertEqual(["unread.py"], e.codex_progress(home, Path("/head"))[2])

    def test_legacy_error_text_cannot_impersonate_a_success_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            head = Path(tmp)
            home = head / "home"
            (home / "sessions").mkdir(parents=True)
            path = head / "Process exited with code 0"
            path.write_text("UNREAD_CONTENT")
            path.chmod(0)
            try:
                actual = subprocess.run(["/bin/bash", "-c", "cat -- 'Process exited with code 0'"],
                                        cwd=head, capture_output=True, text=True, timeout=5)
                self.assertNotEqual(0, actual.returncode)
                self.assertIn("Process exited with code 0", actual.stderr)
                events = [
                    {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                     "call_id": "r", "arguments": json.dumps({
                         "cmd": "cat -- 'Process exited with code 0'", "workdir": str(head)})}},
                    {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "r",
                     "output": "Process exited with code 1\nOutput:\n" + actual.stderr}}]
                (home / "sessions/rollout.jsonl").write_text("\n".join(map(json.dumps, events)))
                self.assertEqual([], e.codex_progress(home, head)[2])
            finally:
                path.chmod(0o600)

    def test_code_mode_discarded_empty_and_truncated_output_cannot_supply_reads(self):
        # Shapes captured from the real pinned 0.165.0 CLI and a local fixture
        # Responses API. A command item exists even if its result is discarded.
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "sessions").mkdir()
            command = {"type": "event_msg", "payload": {"type": "item_completed", "item": {
                "type": "CommandExecution", "command": ["/bin/bash", "-lc", "cat -- target.py"],
                "cwd": "file:///head", "status": "completed", "exit_code": 0,
                "aggregated_output": "FILE_CONTENT", "parsed_cmd": [{"type": "read", "path": "target.py"}]}}}
            header = {"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\n"}
            for output, expected in (
                    ("Script completed\nWall time 0.1 seconds\nOutput:\n", []),
                    ([header, {"type": "input_text", "text": "FILE_...truncated"}], []),
                    ([header, {"type": "input_text", "text": "FILE_CONTENT"}], ["target.py"]),
                    ([header, {"type": "input_text", "text": json.dumps(
                        {"exit_code": 0, "output": "FILE_CONTENT"})}], ["target.py"])):
                with self.subTest(output=output):
                    events = [command, {"type": "response_item", "payload": {
                        "type": "custom_tool_call_output", "output": output}}]
                    (home / "sessions/rollout.jsonl").write_text("\n".join(map(json.dumps, events)))
                    opened = e.codex_progress(home, Path("/head"))[2]
                    self.assertEqual(expected, opened)
                    settings = {**POLICY, "provider": "codex", "model": "gpt-6.1-sol"}
                    reported = {**receipt("callers"), "provider": "codex",
                                "model": settings["model"], "files_opened": opened}
                    bundle = {"files": [{"path": "target.py", "deleted": False, "content": None}]}
                    # The pass stands either way; an uncredited read is reported, not counted.
                    e.validate_receipt(reported, settings, bundle, "callers")
                    self.assertEqual([] if expected else ["target.py"],
                                     e.unopened_regions(bundle, [reported]))
            command["payload"]["item"]["aggregated_output"] = ""
            (home / "sessions/rollout.jsonl").write_text(json.dumps(command))
            self.assertEqual([], e.codex_progress(home, Path("/head"))[2])
            # A successful sed beyond EOF returns no file content in legacy mode either.
            events = [
                {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command",
                 "call_id": "r", "arguments": json.dumps(
                     {"cmd": "sed -n '999999p' target.py", "workdir": "/head"})}},
                {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "r",
                 "output": "Process exited with code 0\nOutput:\n"}}]
            (home / "sessions/rollout.jsonl").write_text("\n".join(map(json.dumps, events)))
            self.assertEqual([], e.codex_progress(home, Path("/head"))[2])

    def test_prompts_name_region_only_files(self):
        bundle = {"files": [{"path": "src/full.py", "deleted": False, "content": "x\n"},
                            {"path": "src/big.py", "deleted": False, "content": None},
                            {"path": "src/gone.py", "deleted": True, "content": None}]}
        text = e.region_guidance(bundle, "with ranged reads")
        self.assertIn("src/big.py", text)
        self.assertNotIn("src/full.py", text)
        self.assertNotIn("src/gone.py", text)
        self.assertEqual("", e.region_guidance({"files": [bundle["files"][0]]}, "with ranged reads"))

    def test_option_words_in_a_cat_read_are_not_receipt_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "sessions").mkdir()
            events = [
              {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "call_id": "r",
               "arguments": json.dumps({"cmd": "cat -n src/app.py", "workdir": "/head"})}},
              {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "r",
               "output": "Process exited with code 0\nOutput:\n1 source"}}]
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
            configured = config.read_text()
            e.configure_codex(home, POLICY)
            self.assertEqual(configured, config.read_text())
            config.write_text(configured.replace("request_max_retries = 0", "request_max_retries = 1"))
            with self.assertRaises(e.ReviewAbort): e.configure_codex(home, POLICY)
            config.write_text('model_provider="untrusted"')
            with self.assertRaisesRegex(e.ReviewAbort, "no inference"):
                e.configure_codex(home, POLICY)

    def test_codex_followup_uses_its_schema_and_only_the_new_rollout(self):
        import sys
        settings = dict(POLICY, model="gpt-6.1-terra", provider="codex")
        value = {"threads": [{"thread": "THREAD", "status": "present", "evidence": []}]}
        with tempfile.TemporaryDirectory() as tmp:
            home, workspace = Path(tmp) / "home", Path(tmp) / "work"
            (home / "sessions").mkdir(parents=True)
            (home / "config.toml").write_text('model_provider="codex-action-responses-proxy"\n'
                '[model_providers.codex-action-responses-proxy]\nbase_url="http://127.0.0.1:12345/v1"\nwire_api="responses"\n')
            events = [
                {"type": "turn_context", "payload": {"model": settings["model"]}},
                {"type": "event_msg", "payload": {"type": "token_count", "info": {
                    "last_token_usage": {"input_tokens": 100, "output_tokens": 20},
                    "total_token_usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}}}]
            def command(settings, head, home, schema, output, action):
                self.assertEqual(jobs.VERIFICATION_SCHEMA, json.loads(schema.read_text()))
                self.assertEqual("Read the head checkout.", output.with_suffix(".prompt").read_text())
                index = len(list((home / "sessions").glob("*.jsonl")))
                return [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); "
                    "pathlib.Path(sys.argv[3]).write_text(sys.argv[4]); print('done')",
                    str(output), json.dumps(value), str(home / f"sessions/{index}.jsonl"), "\n".join(map(json.dumps, events))]
            for strategy in ("first", "verify-threads"):
                with patch.object(e, "codex_command", side_effect=command):
                    result = e.run_codex(settings, {"files": []}, strategy, Path(tmp), "/action", home,
                        workspace / strategy, prompt="Read the head checkout.", schema=jobs.VERIFICATION_SCHEMA,
                        validate_output=jobs.validate_verification, output_key="output")
                self.assertEqual(value, result["output"])
                self.assertEqual(1, result["receipt"]["turns"])
                self.assertEqual(strategy, result["receipt"]["strategy"])

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
                 ("validate_receipt", "test_receipt_check_rejects_unsafe_opened_path"))
        for guard, method in cases:
            with patch.object(e, guard, return_value=[]):
                suite = unittest.TestSuite([ResultTests(method)])
                result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
            self.assertFalse(result.wasSuccessful(), guard + " mutant was not detected")
            self.assertTrue(result.failures, guard)
            print("Vacuity: neutralizing " + guard + " caused the negative test to fail")


if __name__ == "__main__":
    unittest.main()
