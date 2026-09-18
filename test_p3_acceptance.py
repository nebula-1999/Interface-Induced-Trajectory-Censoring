"""CPU-only regressions. Run: python3 -m unittest test_p3_acceptance -v"""
import contextlib
from collections import UserDict
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chat_policy import ChatPolicy, NativeRolloutPolicy
from code_eval_hook import CodeEvalHook, _base_url
from eval_artifacts import StepArtifact, bind_manifest
from eval_decompose import Task, Turn
from p3.launch_arm import main as launch_main, plan, verify_formal_resume
from p3.check_smoke import check
from p3.record_launch import effective_config
from p3.parser_diagnostics import accepts, VLLM_RE, VERL_RE

ROOT = Path(__file__).resolve().parent


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = dict(ARM="broken", RUN_ID="test-pair", PROJ_DIR=str(ROOT),
                        P3_RUN_BASE=str(self.root / "runs"), P3_PYTHON=sys.executable)

    def task(self, ident="t1"):
        return Task(ident, "test", [Turn(1, 1, 1, True, "ok")])

    def test_default_smoke_saves_and_evaluates(self):
        _, env, args, acceptance = plan(self.env)
        self.assertIsNone(acceptance)
        self.assertIn("trainer.save_freq=1", args)
        self.assertIn("trainer.test_freq=1", args)
        self.assertIn("trainer.val_before_train=True", args)
        self.assertIn("test-pair/broken", env["CODE_EVAL_OUT"])
        self.assertRegex(env["RAY_TMPDIR"], r"^/root/autodl-tmp/rp3/[0-9a-f]{12}$")

    def test_negative_or_zero_frequency_refused(self):
        for key in ("EVAL_FREQ", "SAVE_FREQ", "KEEP_CKPT", "STEPS"):
            for value in ("-1", "0", "nonsense"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    plan({**self.env, key: value})

    def test_formal_dry_defaults_and_hydra_last_value(self):
        _, _, args, _ = plan({**self.env, "P3_MODE": "formal", "DRY": "1"})
        self.assertIn("trainer.save_freq=30", args)
        self.assertIn("trainer.test_freq=30", args)
        self.assertIn("trainer.total_training_steps=150", args)
        config = effective_config(["trainer.save_freq=-1", *args[2:]])
        self.assertEqual(config["trainer.save_freq"], "30")

    def test_smoke_missing_artifacts_cannot_pass(self):
        with self.assertRaises(OSError):
            check(self.root)

    def make_smoke_artifacts(self):
        (self.root / "eval").mkdir()
        launch = dict(run_id="run", arm="broken", argv=["bash", "runner",
            "trainer.total_training_steps=1", "trainer.test_freq=1"])
        (self.root / "launch.json").write_text(json.dumps(launch))
        for step in (0, 1):
            a = StepArtifact(self.root / "eval", step, "run/broken", "manifest")
            a.finish([self.task()], [], ["t1"], [], {
                "code/request_failures": 0, "code/native_weight_step": step,
                "code/native_requests": 1})

    def test_smoke_requires_checkpoint_and_does_not_certify_loading(self):
        self.make_smoke_artifacts()
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            check(self.root)
        ckpt = self.root / "ckpt/global_step_1/actor"
        ckpt.mkdir(parents=True)
        (ckpt / "fake-tensor-for-test").write_bytes(b"test-only")
        report = check(self.root)
        self.assertTrue(report["smoke_artifacts_complete"])
        self.assertFalse(report["checkpoint_load_verified"])
        self.assertFalse(report["formal_accepted"])
        self.assertEqual(report["native_weight_steps_attested"], [0, 1])

    def test_smoke_detects_modified_evaluation_file(self):
        self.make_smoke_artifacts()
        path = self.root / "eval/step_00000.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            check(self.root)

    def test_unaligned_end_step_refused(self):
        with self.assertRaises(ValueError):
            plan({**self.env, "STEPS": "2", "SAVE_FREQ": "3"})

    def test_formal_training_is_not_accidentally_authorized(self):
        with self.assertRaisesRegex(ValueError, "P3_ACCEPTANCE_RECORD"):
            plan({**self.env, "P3_MODE": "formal"})
        with self.assertRaises(ValueError):
            plan({**self.env, "STEPS": "150"})

    def test_existing_run_and_resume_preserve_checkpoint(self):
        root, _, _, _ = plan(self.env)
        (root / "ckpt").mkdir(parents=True)
        sentinel = root / "ckpt/precious"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            plan(self.env)
        with self.assertRaisesRegex(ValueError, "only valid"):
            plan({**self.env, "RESUME_FROM": str(sentinel.parent)})
        self.assertEqual(sentinel.read_text(), "keep")

    def test_resume_smoke_is_one_update_and_preserves_source(self):
        source = self.root / "source/global_step_1"
        (source / "actor").mkdir(parents=True)
        (source / "actor/model.pt").write_bytes(b"checkpoint")
        (source / "data.pt").write_bytes(b"dataloader")
        resume_env = {**self.env, "RUN_ID": "resume-test", "P3_MODE": "resume-smoke",
                      "RESUME_FROM": str(source)}
        _, _, args, _ = plan(resume_env)
        self.assertIn("trainer.total_training_steps=2", args)
        self.assertIn("trainer.resume_mode=resume_path", args)
        self.assertIn(f"trainer.resume_from_path={source}", args)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            plan({**resume_env, "STEPS": "3"})
        self.assertEqual((source / "actor/model.pt").read_bytes(), b"checkpoint")

    def test_resume_main_records_source_without_scope_error(self):
        source = self.root / "source/global_step_1"
        (source / "actor").mkdir(parents=True)
        (source / "actor/model.pt").write_bytes(b"checkpoint")
        (source / "data.pt").write_bytes(b"dataloader")
        env = {**self.env, "RUN_ID": "resume-main", "P3_MODE": "resume-smoke",
               "RESUME_FROM": str(source)}
        with patch.dict(os.environ, env, clear=True), \
             patch("p3.launch_arm.subprocess.call", return_value=1):
            self.assertEqual(launch_main(), 1)
        launch = json.loads((self.root / "runs/resume-main/broken/launch.json").read_text())
        self.assertEqual(launch["start_step"], 1)
        self.assertEqual(launch["resume_from"], str(source))
        self.assertEqual(launch["resume_checkpoint_files"]["actor/model.pt"]["sha256"],
                         __import__("hashlib").sha256(b"checkpoint").hexdigest())

    def make_formal_resume(self, **resume_changes):
        base_env = {**self.env, "ARM": "repaired", "P3_MODE": "formal", "DRY": "1"}
        _, _, base_args, _ = plan(base_env)
        base = self.root / "runs/test-pair/repaired"
        source = base / "ckpt/global_step_90"
        (source / "actor").mkdir(parents=True)
        (source / "actor/model.pt").write_bytes(b"model")
        (source / "actor/optim.pt").write_bytes(b"optimizer")
        (source / "actor/extra_state.pt").write_bytes(b"rng-scheduler")
        (source / "data.pt").write_bytes(b"dataloader")
        artifact = StepArtifact(base / "eval", 90, "test-pair/repaired", "manifest")
        artifact.finish([self.task()], [], ["t1"], [], {
            "code/request_failures": 0, "code/native_weight_step": 90,
            "code/native_requests": 1})
        (base.parent / "training_manifest.json").write_text("{}")
        (base / "launch.json").write_text(json.dumps({
            "run_id": "test-pair", "arm": "repaired", "mode": "formal",
            "argv": base_args, "source_sha256": {}}))
        resume_env = {**self.env, "ARM": "repaired", "P3_MODE": "formal-resume",
                      "DRY": "1", "RESUME_FROM": str(source), **resume_changes}
        root, _, args, _ = plan(resume_env)
        return source, root, args

    def test_formal_resume_is_isolated_and_config_equal(self):
        source, root, args = self.make_formal_resume()
        with patch.dict(os.environ, {"P3_RUN_BASE": str(self.root / "runs")}):
            proof = verify_formal_resume(source, root, ROOT, "test-pair", "repaired", args)
        self.assertEqual(root.name, "repaired_resume_00090")
        self.assertEqual(proof["resume_step"], 90)
        self.assertTrue(proof["normalized_config_equal"])
        self.assertIn("trainer.resume_mode=resume_path", args)
        self.assertIn(f"trainer.resume_from_path={source}", args)

    def test_formal_resume_refuses_scientific_config_drift(self):
        source, root, args = self.make_formal_resume(EVAL_FREQ="15")
        with patch.dict(os.environ, {"P3_RUN_BASE": str(self.root / "runs")}), \
             self.assertRaisesRegex(ValueError, "config drift"):
            verify_formal_resume(source, root, ROOT, "test-pair", "repaired", args)

    def test_formal_resume_requires_optimizer_and_rng_state(self):
        source, root, args = self.make_formal_resume()
        (source / "actor/optim.pt").unlink()
        with patch.dict(os.environ, {"P3_RUN_BASE": str(self.root / "runs")}), \
             self.assertRaisesRegex(ValueError, "optimizer"):
            verify_formal_resume(source, root, ROOT, "test-pair", "repaired", args)

    def test_bad_identity_and_path_override_refused(self):
        for changes in ({"RUN_ID": "../oops"}, {"RUN_ID": ""},
                        {"CODE_EVAL_OUT": "/tmp/shared"}):
            with self.assertRaises(ValueError):
                plan({**self.env, **changes})
        with self.assertRaisesRegex(ValueError, "AF_UNIX"):
            plan({**self.env, "P3_RAY_BASE": "/root/autodl-tmp/" + "x" * 70})

    def test_two_arms_share_manifest_not_output(self):
        _, broken, a, _ = plan(self.env)
        _, repaired, b, _ = plan({**self.env, "ARM": "repaired"})
        self.assertEqual(broken["CODE_EVAL_MANIFEST"], repaired["CODE_EVAL_MANIFEST"])
        self.assertNotEqual(broken["CODE_EVAL_OUT"], repaired["CODE_EVAL_OUT"])
        differences = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(len(differences), 2)  # parser + bookkeeping experiment name

    def test_real_dry_run_is_non_mutating(self):
        result = subprocess.run(["bash", str(ROOT / "p3/run_p3_arm.sh")],
            env={**os.environ, **self.env, "DRY": "1"}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[p3-command]", result.stdout)
        self.assertTrue(result.stdout.rfind("trainer.save_freq=1") >
                        result.stdout.rfind("trainer.save_freq=-1"))
        self.assertFalse((self.root / "runs").exists())

    def test_probe_reuse_refused_without_deletion(self):
        events = self.root / "events"
        events.mkdir()
        sentinel = events / "original.jsonl"
        sentinel.write_text("keep")
        result = subprocess.run(["bash", str(ROOT / "p3/run_p3_probe.sh")],
            env={**os.environ, "PROJ_DIR": str(ROOT), "P3_OUT": str(events),
                 "P3_CKPT_DIR": str(self.root / "ckpt"), "DRY": "0"},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_manifest_stable_and_drift_rejected(self):
        path = self.root / "manifest.json"
        first = bind_manifest(path, {"tasks": ["a", "b"]})
        self.assertEqual(first, bind_manifest(path, {"tasks": ["a", "b"]}))
        with self.assertRaises(ValueError):
            bind_manifest(path, {"tasks": ["a", "c"]})

    def test_complete_step_unique_and_hashed(self):
        a = StepArtifact(self.root, 0, "run/broken", "manifest-hash")
        a.finish([self.task()], [], ["t1"], [], {
            "code/request_failures": 0,
            "code/native_weight_step": 0,
            "code/native_requests": 1,
        })
        path = self.root / "step_00000.jsonl"
        original = path.read_bytes()
        self.assertEqual(json.loads(original)["run_id"], "run/broken")
        complete = json.loads((self.root / "step_00000.complete.json").read_text())
        self.assertTrue(complete["weight_sync_verified"])
        with self.assertRaises(FileExistsError):
            StepArtifact(self.root, 0, "another-run", "manifest-hash")
        self.assertEqual(original, path.read_bytes())

    def test_duplicate_missing_or_failed_step_never_complete(self):
        for i, (tasks, failures) in enumerate((([self.task(), self.task()], 0),
                                               ([], 0), ([self.task()], 1))):
            a = StepArtifact(self.root, i, "run", "manifest")
            with self.assertRaises(ValueError):
                a.finish(tasks, [], ["t1"], [], {"code/request_failures": failures})
            a.fail("expected test failure")
            self.assertFalse((self.root / f"step_{i:05d}.complete.json").exists())
            self.assertFalse((self.root / f"step_{i:05d}.jsonl").exists())
            with self.assertRaises(FileExistsError):
                StepArtifact(self.root, i, "run", "manifest")

    def test_complete_step_does_not_claim_weight_sync_without_attestation(self):
        a = StepArtifact(self.root, 3, "run", "manifest")
        a.finish([self.task()], [], ["t1"], [], {"code/request_failures": 0})
        complete = json.loads((self.root / "step_00003.complete.json").read_text())
        self.assertFalse(complete["weight_sync_verified"])

    def test_parser_single_and_two_capture_groups(self):
        payload = '{"name":"run_tests","arguments":{"code":"pass"}}'
        for rx in (VLLM_RE, VERL_RE):
            self.assertTrue(accepts(rx, f"<tool_call>{payload}</tool_call>"))
            self.assertFalse(accepts(rx, "<tool_call>broken</tool_call>"))
            self.assertFalse(accepts(rx, '<tool_call>["name","arguments"]</tool_call>'))
            self.assertFalse(accepts(rx, payload))
        self.assertTrue(accepts(VLLM_RE, f"<tool_call>{payload}"))
        self.assertFalse(accepts(VERL_RE, f"<tool_call>{payload}"))

    def test_base_url_does_not_duplicate_v1(self):
        self.assertEqual(_base_url("localhost:8000/v1/"), "http://localhost:8000/v1")

    def test_request_failure_is_fatal_in_required_mode(self):
        p = ChatPolicy(["http://invalid/v1"], "test", retries=0, raise_on_failure=True)
        with patch("urllib.request.urlopen", side_effect=OSError("test")):
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                p([[{"role": "user", "content": "test"}]])
        self.assertEqual(sum(p.failures.values()), 1)

    def test_native_rollout_policy_attests_weight_step(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return UserDict({"input_ids": [[1, 2]], "attention_mask": [[1, 1]]})
            def decode(self, ids, **kwargs): return "answer"
        class Client:
            async def generate(self, **kwargs):
                return SimpleNamespace(token_ids=[3], extra_fields={
                    "global_steps": 7, "min_global_steps": 7, "max_global_steps": 7})
        policy = NativeRolloutPolicy(Client(), Tokenizer(), expected_step=7, workers=2)
        self.assertEqual(policy([[{"role": "user", "content": "x"}]]), ["answer"])
        self.assertEqual(policy.observed_steps[7], 1)

    def test_native_rollout_policy_rejects_stale_weights(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): return [1]
            def decode(self, ids, **kwargs): return "stale"
        class Client:
            async def generate(self, **kwargs):
                return SimpleNamespace(token_ids=[2], extra_fields={"global_steps": 6})
        policy = NativeRolloutPolicy(Client(), Tokenizer(), expected_step=7, workers=1)
        with self.assertRaisesRegex(RuntimeError, "weight-step mismatch"):
            policy([[{"role": "user", "content": "x"}]])
        self.assertEqual(sum(policy.failures.values()), 1)

    def test_duplicate_probes_rejected(self):
        path = self.root / "probes.jsonl"
        path.write_text('{"task_id":"same"}\n' * 2)
        with self.assertRaises(ValueError):
            CodeEvalHook._load_probes(path)

    def test_hook_two_steps_same_manifest_and_refuses_repeat(self):
        recs = [("test", "t1", {"prompt": "test"})]
        with patch.dict(os.environ, {"CODE_EVAL_REQUIRE_REPAIR": "0"}, clear=True), \
             patch.object(CodeEvalHook, "_load_evalplus", return_value=recs), \
             patch("code_eval_hook.evaluate", return_value=[self.task()]):
            hook = CodeEvalHook(["localhost:8000"], "test", str(self.root),
                                probes_path=str(self.root / "absent"))
            hook.run(0)
            hook.run(1)
            with self.assertRaises(FileExistsError):
                hook.run(0)

    def test_required_repair_missing_and_duplicate_dataset_fail(self):
        recs = [("test", "t1", {"prompt": "test"})]
        with patch.dict(os.environ, {"CODE_EVAL_REQUIRE_REPAIR": "1"}, clear=True), \
             patch.object(CodeEvalHook, "_load_evalplus", return_value=recs):
            with self.assertRaisesRegex(ValueError, "Required repair"):
                CodeEvalHook(["localhost"], "test", str(self.root),
                             probes_path=str(self.root / "absent"))
        with patch.object(CodeEvalHook, "_load_evalplus", return_value=recs * 2):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                CodeEvalHook(["localhost"], "test", str(self.root))

    def patched_trainer(self):
        # Import is deliberately attempted without verl; install via injection.
        with contextlib.redirect_stdout(io.StringIO()):
            spec = importlib.util.spec_from_file_location("code_patch_under_test", ROOT / "code_patch.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            class Trainer:
                global_steps = 0
                config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(
                    model=SimpleNamespace(path="test")))
                async_rollout_manager = SimpleNamespace(server_addresses=["localhost:8000"])
                def _validate(self):
                    return {}
            module.install(Trainer)
        return Trainer()

    def test_later_eval_exception_is_not_swallowed(self):
        trainer = self.patched_trainer()
        with patch("code_eval_hook.CodeEvalHook") as hook, contextlib.redirect_stdout(io.StringIO()):
            hook.return_value.run.side_effect = [{"code/request_failures": 0}, RuntimeError("later")]
            trainer._validate()
            with self.assertRaisesRegex(RuntimeError, "training stopped"):
                trainer._validate()

    def test_reported_request_errors_not_marked_success(self):
        trainer = self.patched_trainer()
        with patch("code_eval_hook.CodeEvalHook") as hook, contextlib.redirect_stdout(io.StringIO()):
            hook.return_value.run.return_value = {"code/request_failures": 1}
            with self.assertRaises(RuntimeError):
                trainer._validate()
        self.assertFalse(getattr(trainer, "_code_eval_ok_once", False))


if __name__ == "__main__":
    unittest.main()
