# Copyright 2026 Pipelock contributors
# SPDX-License-Identifier: Apache-2.0
"""Logical passes over bounded sections: cumulative allowance, ledger, receipts and tamper checks."""
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import review_engine as e, review_sections as s
from scripts.test_review_engine import POLICY, finding, receipt

SMALL = dict(POLICY, max_model_tokens=40_000, max_output_tokens=16_000)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Repo(unittest.TestCase):
    """A real repository whose change needs several sections under the small fixture allowance."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        (self.root / "writer.py").write_text("def write(state):\n    return state\n")
        (self.root / "reader.py").write_text("def read(state):\n    return state\n")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD")
        (self.root / "writer.py").write_text("def write(state):\n    return None  # drops state\n" + "# note\n" * 4000)
        (self.root / "reader.py").write_text("def read(state):\n    return state['key']  # assumes state\n" + "# note\n" * 4000)
        self.git("add", ".")
        self.git("commit", "-qm", "head")
        self.head = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True).strip()

    def bundle(self, settings=SMALL, **kwargs):
        return e.build_bundle(self.root, self.head, self.base, settings, instructions="Review correctness.", **kwargs)


def section_receipt(strategy, turns=2, cost=0.5):
    r = receipt(strategy)
    r.update(turns=turns, cost_usd=cost, files_opened=["writer.py"])
    return r


class AllowanceTests(unittest.TestCase):
    def test_remaining_native_allowances_shrink_and_totals_are_verified(self):
        clock = Clock()
        allowance = e.Allowance(POLICY, clock)
        first = allowance.remaining()
        self.assertEqual((10, 5.0, 300), (first["max_turns"], first["max_budget_usd"], first["ai_timeout"]))
        allowance.charge(section_receipt("bundle", turns=4, cost=1.25))
        clock.now += 60
        second = allowance.remaining()
        self.assertEqual((6, 3.75, 240), (second["max_turns"], second["max_budget_usd"], second["ai_timeout"]))
        with self.assertRaisesRegex(e.ReviewAbort, "cumulative review_depth allowance"):
            allowance.charge(section_receipt("bundle", turns=7, cost=0.1))

    def test_each_exhausted_allowance_stops_before_another_launch(self):
        for turns, cost, wait, message in ((10, 0.1, 0, "max_turns across its sections"),
                                           (1, 5.0, 0, "max_budget_usd across its sections"),
                                           (1, 0.1, 300, "ai_timeout across the pass's sections")):
            clock = Clock()
            allowance = e.Allowance(POLICY, clock)
            allowance.remaining()
            allowance.charge(section_receipt("bundle", turns=turns, cost=cost))
            clock.now += wait
            with self.subTest(message=message), self.assertRaisesRegex(e.ReviewAbort, message):
                allowance.remaining()

    def test_the_deadline_starts_at_the_first_launch_and_never_resets(self):
        clock = Clock()
        allowance = e.Allowance(POLICY, clock)
        clock.now += 1000                       # setup time before the pass is not charged
        self.assertEqual(300, allowance.remaining()["ai_timeout"])
        clock.now += 100
        allowance.charge(section_receipt("bundle", turns=1, cost=0.1))
        self.assertEqual(200, allowance.remaining()["ai_timeout"])


class PassTests(Repo):
    def invoke_factory(self, outputs=None, record=None):
        outputs = outputs or {}
        calls = record if record is not None else []

        def invoke(current, prompt, schema, validator, key):
            calls.append({"settings": current, "prompt": prompt, "schema": schema, "key": key})
            number = len(calls) - 1
            output = outputs.get(number, {"findings": [], "questions": []})
            if key == "output":
                return {"output": output, "receipt": section_receipt("bundle")}
            return {"findings": output["findings"], "receipt": section_receipt("bundle")}
        return invoke

    def test_every_section_then_a_cross_section_stage_under_one_cumulative_allowance(self):
        bundle = self.bundle()
        count = len(bundle["sections"])
        self.assertGreater(count, 1)
        calls = []
        policy = dict(SMALL, max_turns=100, max_budget_usd=50.0)
        records = e.run_pass(policy, bundle, "bundle", self.root, self.invoke_factory(record=calls))
        self.assertEqual(count + 1, len(calls))
        self.assertEqual(["section"] * count + ["consolidation"], [r["kind"] for r in records])
        self.assertEqual(list(range(count)) + [None], [r["section"] for r in records])
        # Remaining native allowances shrink by exactly what earlier results used.
        self.assertEqual([100 - 2 * i for i in range(count + 1)], [c["settings"]["max_turns"] for c in calls])
        self.assertEqual([50.0 - 0.5 * i for i in range(count + 1)], [c["settings"]["max_budget_usd"] for c in calls])
        self.assertEqual({"section"}, {r["schema"] for r in records[:-1]})
        self.assertEqual(["output"] * count + ["findings"], [c["key"] for c in calls])
        self.assertIs(e.SECTION_SCHEMA, calls[0]["schema"])
        self.assertIs(e.OUTPUT_SCHEMA, calls[-1]["schema"])
        # Each section's prompt carries only that section's blocks.
        for index, call in enumerate(calls[:count]):
            segments = bundle["sections"][index]["segments"]
            self.assertIn(f"section {index + 1} of {count}", call["prompt"])
            self.assertEqual(len(segments), call["prompt"].count("File metadata: "))
        validated = e.combine_results(policy, [("bundle", records)])
        policy_one = dict(policy, passes=["bundle"])
        e.validate_result(validated, policy_one, bundle, prompts=True)

    def test_both_default_strategies_cover_all_sections_independently(self):
        bundle = self.bundle()
        policy = dict(SMALL, max_turns=100, max_budget_usd=50.0)
        seen = {}
        for strategy in policy["passes"]:
            calls = []
            seen[strategy] = e.run_pass(policy, bundle, strategy, self.root, self.invoke_factory(record=calls))
            self.assertEqual(len(bundle["sections"]) + 1, len(calls))
        self.assertEqual({r["section"] for r in seen["bundle"]}, {r["section"] for r in seen["callers"]})
        self.assertNotEqual(seen["bundle"][0]["prompt_sha256"], seen["callers"][0]["prompt_sha256"])
        # Each pass keeps its own ledger and its own allowance.
        self.assertEqual(sorted(bundle.store.ledger(s).name for s in policy["passes"]),
                         sorted(p.name for p in (bundle.store.directory / "ledger").iterdir()))

    def test_the_ledger_and_index_are_read_not_merely_present(self):
        bundle = self.bundle()
        count = len(bundle["sections"])
        policy = dict(SMALL, max_turns=100, max_budget_usd=50.0)
        facts = {}
        question = {"path": "reader.py", "question": "Does the reader tolerate a writer that drops state?"}
        producer = finding(path="writer.py", title="Writer drops the state it was given")

        def invoke(current, prompt, schema, validator, key):
            number = len(facts)
            facts[number] = prompt
            if key == "findings":                       # the cross-section stage
                ledger = (bundle.store.ledger("bundle")).read_text()
                self.assertIn(producer["title"], ledger)
                self.assertIn(question["question"], prompt)
                return {"findings": [finding(path="reader.py", title="Reader assumes state that the writer drops")],
                        "receipt": section_receipt("bundle")}
            # A section reads the ledger and index files the prompt names, as the file tools would.
            ledger_path = re.search(r"([\w./-]*ledger/bundle\.txt)", prompt).group(1)
            index_path = re.search(r"([\w./-]*index\.tsv)", prompt).group(1)
            self.assertTrue((self.root / ledger_path).is_file() and (self.root / index_path).is_file())
            earlier = (self.root / ledger_path).read_text()
            if number == 0:
                self.assertEqual("", earlier)
                return {"output": {"findings": [producer], "questions": [question]}, "receipt": section_receipt("bundle")}
            self.assertIn(producer["title"], earlier)
            self.assertIn(question["question"], earlier)
            return {"output": {"findings": [], "questions": []}, "receipt": section_receipt("bundle")}

        records = e.run_pass(policy, bundle, "bundle", self.root, invoke)
        self.assertEqual(count + 1, len(facts))
        self.assertEqual(1, records[-1]["questions_shown"] + records[-1]["questions_omitted"])
        self.assertEqual(0, records[-1]["questions_omitted"])
        self.assertEqual(2, records[1]["ledger"]["entries"])
        expected = e.Ledger()
        expected.add(0, records[0]["output"])
        self.assertEqual(expected.meta(), records[1]["ledger"])

    def test_questions_that_do_not_fit_the_final_stage_are_counted_not_dropped(self):
        bundle = self.bundle()
        policy = dict(SMALL, max_turns=1000, max_budget_usd=500.0)
        many = [{"path": "reader.py", "question": f"Question {i}. " + "detail " * 400} for i in range(e.MAX_FINDINGS)]
        calls = []
        records = e.run_pass(policy, bundle, "bundle", self.root, self.invoke_factory(
            {i: {"findings": [], "questions": many} for i in range(len(bundle["sections"]))}, calls))
        final = records[-1]
        total = len(bundle["sections"]) * len(many)
        self.assertEqual(total, final["questions_shown"] + final["questions_omitted"])
        self.assertGreater(final["questions_omitted"], 0)
        available = policy["max_model_tokens"] - policy["max_output_tokens"]
        self.assertLessEqual(e.token_bound(calls[-1]["prompt"]) + e.token_bound(json.dumps(e.OUTPUT_SCHEMA)), available)
        self.assertIn(f"{final['questions_shown']} of {total} shown", calls[-1]["prompt"])
        e.validate_result(e.combine_results(policy, [("bundle", records)]), dict(policy, passes=["bundle"]), bundle, prompts=True)

    def test_a_stop_in_the_middle_leaves_the_pass_incomplete_with_counters_only(self):
        bundle = self.bundle()
        policy = dict(SMALL, max_turns=100, max_budget_usd=50.0)
        for failure in ("Provider timed out at config.ai_timeout; no paid retry", "Claude stopped at a blocking limit",
                        "Invalid or truncated JSON; output must be one complete unfenced JSON object"):
            calls = []

            def invoke(current, prompt, schema, validator, key, failure=failure):
                calls.append(1)
                if len(calls) == 2:
                    raise e.ReviewAbort(failure)
                return {"output": {"findings": [], "questions": []}, "receipt": section_receipt("bundle")}

            with self.subTest(failure=failure), self.assertRaises(e.IncompleteReviewAbort) as stopped:
                e.run_pass(policy, bundle, "bundle", self.root, invoke)
            self.assertEqual(2, len(calls), "no section ran after the failed one")
            self.assertEqual({"strategy": "bundle", "completed": 1, "total": len(bundle["sections"]) + 1},
                             stopped.exception.progress)
            self.assertIn(failure, str(stopped.exception))
            self.assertEqual("sections_incomplete", __import__("scripts.review_reliability", fromlist=["x"]).reason(stopped.exception))

    def test_each_resource_exhausted_between_sections_is_incomplete(self):
        bundle = self.bundle()
        for label, turns, cost, tick in (("turns", 2, 0.1, 0), ("budget", 1, 0.5, 0), ("time", 1, 0.1, 400)):
            clock = Clock()
            policy = dict(SMALL, max_turns=2 if label == "turns" else 100, max_budget_usd=0.5 if label == "budget" else 50.0,
                          ai_timeout=300)
            calls = []

            def invoke(current, prompt, schema, validator, key, turns=turns, cost=cost, tick=tick):
                calls.append(1)
                clock.now += tick
                r = section_receipt("bundle", turns=turns, cost=cost)
                return {"output": {"findings": [], "questions": []}, "receipt": r}

            with self.subTest(label=label), self.assertRaises(e.IncompleteReviewAbort) as stopped:
                e.run_pass(policy, bundle, "bundle", self.root, invoke, clock=clock)
            self.assertEqual(1, len(calls))
            self.assertEqual(1, stopped.exception.progress["completed"])

    def test_a_single_section_change_keeps_the_small_review_call_exactly(self):
        (self.root / "writer.py").write_text("def write(state):\n    return None\n")
        (self.root / "reader.py").write_text("def read(state):\n    return state\n")
        self.git("add", ".")
        self.git("commit", "-qm", "small")
        self.head = self.git("rev-parse", "HEAD")
        bundle = self.bundle(POLICY)
        self.assertEqual(1, len(bundle["sections"]))
        calls = []
        records = e.run_pass(POLICY, bundle, "bundle", self.root, self.invoke_factory(record=calls))
        self.assertEqual(1, len(calls))
        self.assertEqual("findings", calls[0]["key"])
        self.assertIs(e.OUTPUT_SCHEMA, calls[0]["schema"])
        self.assertEqual({k: POLICY[k] for k in ("max_turns", "max_budget_usd", "ai_timeout")},
                         {k: calls[0]["settings"][k] for k in ("max_turns", "max_budget_usd", "ai_timeout")})
        self.assertEqual(e.claude_prompt(bundle, "bundle"), calls[0]["prompt"])
        self.assertIsNone(records[0]["ledger"])


class ValidationTests(Repo):
    def passes(self):
        bundle = self.bundle()
        policy = dict(SMALL, max_turns=100, max_budget_usd=50.0, passes=["bundle"])
        calls = []

        def invoke(current, prompt, schema, validator, key):
            calls.append(1)
            findings = [finding(path="writer.py", title=f"Candidate {len(calls)}")] if len(calls) < 3 else []
            if key == "output":
                return {"output": {"findings": findings, "questions": []}, "receipt": section_receipt("bundle")}
            return {"findings": findings, "receipt": section_receipt("bundle")}

        records = e.run_pass(policy, bundle, "bundle", self.root, invoke)
        return bundle, policy, e.combine_results(policy, [("bundle", records)])

    def fix(self, result, policy, mutate):
        data = copy.deepcopy(result)
        mutate(data)
        data["receipt"]["passes"] = [e.pass_receipt(policy, "bundle", data["invocations"]["bundle"])]
        data["findings"] = e.merge_findings(r["output"]["findings"] for r in data["invocations"]["bundle"])
        return data

    def test_the_publisher_recomputes_every_prompt_from_immutable_inputs(self):
        bundle, policy, result = self.passes()
        e.validate_result(result, policy, bundle, prompts=True)
        mutations = {
            "section hash": lambda d: d["invocations"]["bundle"][0].update(section_sha256="0" * 64),
            "manifest root": lambda d: d["invocations"]["bundle"][1].update(manifest_root="0" * 64),
            "prompt digest": lambda d: d["invocations"]["bundle"][0].update(prompt_sha256="0" * 64),
            "prompt size": lambda d: d["invocations"]["bundle"][0].update(prompt_bytes=1),
            "ledger digest": lambda d: d["invocations"]["bundle"][2]["ledger"].update(sha256="0" * 64),
            "ledger count": lambda d: d["invocations"]["bundle"][2]["ledger"].update(entries=99),
            "learning bytes": lambda d: d["invocations"]["bundle"][0].update(learnings_bytes=7),
            "duplicate section": lambda d: d["invocations"]["bundle"].insert(1, copy.deepcopy(d["invocations"]["bundle"][0])),
            "missing section": lambda d: d["invocations"]["bundle"].pop(1),
            "missing consolidation": lambda d: d["invocations"]["bundle"].pop(),
            "reordered": lambda d: d["invocations"]["bundle"].insert(0, d["invocations"]["bundle"].pop(1)),
            "foreign schema": lambda d: d["invocations"]["bundle"][0].update(schema="output"),
            "forged question count": lambda d: d["invocations"]["bundle"][-1].update(questions_omitted=5),
            "earlier output altered after the fact": lambda d: d["invocations"]["bundle"][0]["output"].update(
                findings=[finding(path="writer.py", title="Rewritten later")]),
        }
        for name, mutate in mutations.items():
            data = self.fix(result, policy, mutate)
            with self.subTest(name), self.assertRaises(e.ReviewAbort):
                e.validate_result(data, policy, bundle, prompts=True)

    def test_model_authored_coverage_claims_and_stale_plans_never_validate(self):
        bundle, policy, result = self.passes()
        data = copy.deepcopy(result)
        data["receipt"]["passes"][0]["covered"] = True
        with self.assertRaises(e.ReviewAbort):
            e.validate_result(data, policy, bundle, prompts=True)
        data = copy.deepcopy(result)
        data["invocations"]["bundle"][0]["covered_files"] = ["writer.py"]
        with self.assertRaises(e.ReviewAbort):
            e.validate_result(data, policy, bundle, prompts=True)
        stale = copy.deepcopy(bundle)
        stale["root"] = "9" * 64
        with self.assertRaises(e.ReviewAbort):
            e.validate_result(result, policy, stale, prompts=True)
        smaller = e.build_bundle(self.root, self.head, self.base, dict(SMALL, max_model_tokens=60_000))
        with self.assertRaises(e.ReviewAbort):
            e.validate_result(result, policy, smaller, prompts=True)

    def test_cumulative_totals_must_stay_inside_the_pass_allowance(self):
        bundle, policy, result = self.passes()
        spent = sum(r["receipt"]["cost_usd"] for r in result["invocations"]["bundle"])
        tight = dict(policy, max_budget_usd=spent - 0.01)
        with self.assertRaises(e.ReviewAbort):
            e.validate_result(result, tight, bundle, prompts=True)
        turns = sum(r["receipt"]["turns"] for r in result["invocations"]["bundle"])
        with self.assertRaises(e.ReviewAbort):
            e.validate_result(result, dict(policy, max_turns=turns - 1), bundle, prompts=True)

    def test_more_than_twelve_aggregate_candidates_stay_visible(self):
        bundle = self.bundle()
        policy = dict(SMALL, max_turns=100, max_budget_usd=50.0, passes=["bundle"])
        count = len(bundle["sections"]) + 1

        def invoke(current, prompt, schema, validator, key):
            number = len(seen)
            seen.append(1)
            items = [finding(path="writer.py", line=number * 10 + i + 1, end_line=number * 10 + i + 1,
                             title=f"Candidate {number}.{i}") for i in range(e.MAX_FINDINGS)]
            if key == "output":
                return {"output": {"findings": items, "questions": []}, "receipt": section_receipt("bundle")}
            return {"findings": items, "receipt": section_receipt("bundle")}

        seen = []
        records = e.run_pass(policy, bundle, "bundle", self.root, invoke)
        result = e.combine_results(policy, [("bundle", records)])
        self.assertEqual(count * e.MAX_FINDINGS, len(result["findings"]))
        self.assertGreater(len(result["findings"]), 12)
        e.validate_result(result, policy, bundle, prompts=True)
        with self.assertRaises(e.ReviewAbort):
            e.validate_result(dict(result, findings=result["findings"][:-1]), policy, bundle, prompts=True)


class CredentialTests(Repo):
    def test_one_credential_choice_spans_every_section_and_the_fallback_happens_once(self):
        bundle = self.bundle()
        policy = dict(SMALL, max_turns=100, max_budget_usd=50.0)
        keys, calls = [], []
        billing = [{"type": "system", "subtype": "init"},
                   {"type": "assistant", "error": "billing_error", "is_api_error_message": True, "api_error_status": 400,
                    "message": {"content": [{"type": "text", "text": "x"}],
                                "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
                                          "cache_read_input_tokens": 0}}},
                   {"type": "result", "subtype": "success", "is_error": True, "terminal_reason": "api_error",
                    "api_error_status": 400, "num_turns": 1, "modelUsage": {}, "total_cost_usd": 0,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 0}}]
        good = [{"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "total_cost_usd": 0.1,
                 "usage": {"input_tokens": 10, "output_tokens": 5}, "modelUsage": {"claude-sonnet-5-5": {}},
                 "structured_output": {"findings": [], "questions": []}}]

        def run(command, **kwargs):
            key = kwargs["env"].get("ANTHROPIC_API_KEY")
            keys.append(key)
            calls.append(command)
            schema = json.loads(command[command.index("--json-schema") + 1])
            structured = {"findings": [], "questions": []} if "questions" in schema["properties"] else {"findings": []}
            transcript = billing if key == "primary" and len(keys) == 1 else [dict(good[0], structured_output=structured)]
            out = "\n".join(json.dumps(event) for event in transcript)
            return subprocess.CompletedProcess(command, 1 if transcript is billing else 0, out, "")

        credentials = None
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "primary", "ANTHROPIC_API_KEY_2": "secondary"}), \
                patch.object(e.subprocess, "run", side_effect=run):
            credentials = e.ClaudeCredentials()

            def invoke(current, prompt, schema, validator, key):
                return e.run_claude(current, bundle, "bundle", self.root, prompt=prompt, schema=schema,
                                    validate_output=validator, output_key=key, credentials=credentials)

            records = e.run_pass(policy, bundle, "bundle", self.root, invoke)
        count = len(bundle["sections"]) + 1
        self.assertEqual(count + 1, len(calls), "one zero-work refusal, then exactly one launch per invocation")
        self.assertEqual(["primary"] + ["secondary"] * count, keys)
        self.assertEqual(count, len(records))
        self.assertTrue(credentials.started and credentials.using_secondary)

    def test_a_billing_refusal_in_a_later_section_never_switches_accounts(self):
        bundle = self.bundle()
        policy = dict(SMALL, max_turns=100, max_budget_usd=50.0)
        keys = []
        zero = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        billing = [{"type": "system", "subtype": "init"},
                   {"type": "assistant", "error": "billing_error", "is_api_error_message": True, "api_error_status": 400,
                    "message": {"content": [{"type": "text", "text": "x"}], "usage": zero}},
                   {"type": "result", "subtype": "success", "is_error": True, "terminal_reason": "api_error",
                    "api_error_status": 400, "num_turns": 1, "modelUsage": {}, "total_cost_usd": 0, "usage": zero}]
        good = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "total_cost_usd": 0.1,
                "usage": {"input_tokens": 10, "output_tokens": 5}, "modelUsage": {"claude-sonnet-5-5": {}},
                "structured_output": {"findings": [], "questions": []}}

        def run(command, **kwargs):
            keys.append(kwargs["env"].get("ANTHROPIC_API_KEY"))
            transcript = [good] if len(keys) == 1 else billing
            return subprocess.CompletedProcess(command, 0 if len(keys) == 1 else 1,
                                               "\n".join(json.dumps(event) for event in transcript), "")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "primary", "ANTHROPIC_API_KEY_2": "secondary"}), \
                patch.object(e.subprocess, "run", side_effect=run):
            credentials = e.ClaudeCredentials()

            def invoke(current, prompt, schema, validator, key):
                return e.run_claude(current, bundle, "bundle", self.root, prompt=prompt, schema=schema,
                                    validate_output=validator, output_key=key, credentials=credentials)

            with self.assertRaises(e.IncompleteReviewAbort) as stopped:
                e.run_pass(policy, bundle, "bundle", self.root, invoke)
        self.assertEqual(["primary", "primary"], keys, "the secondary key is never tried after a completed invocation")
        self.assertEqual(1, stopped.exception.progress["completed"])
        self.assertFalse(credentials.using_secondary)


if __name__ == "__main__":
    unittest.main()
