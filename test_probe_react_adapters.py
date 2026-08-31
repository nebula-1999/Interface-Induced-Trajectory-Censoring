"""不依赖 GPU/datasets 的协议适配回归测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


# probe_react_full 的解析函数本身不依赖这两个重包；离线测试时提供最小桩。
datasets_stub = types.ModuleType("datasets")
datasets_stub.load_dataset = None
sys.modules.setdefault("datasets", datasets_stub)
sandbox_stub = types.ModuleType("sandbox")
sandbox_stub.run_tests = None
sys.modules.setdefault("sandbox", sandbox_stub)

MODULE_PATH = Path(__file__).with_name("probe_react_full.py")
spec = importlib.util.spec_from_file_location("probe_react_full_under_test", MODULE_PATH)
probe = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(probe)


class ParserAdapterTests(unittest.TestCase):
    def test_qwen_auto_keeps_legacy_parser(self):
        adapter = probe.parser_adapter("/models/Qwen2.5-Coder-7B-Instruct", "auto")
        self.assertEqual(adapter, "legacy")
        action, code, mode = probe.extract_react_response(
            "Here is the solution:\n```python\ndef f(): return 1\n```", adapter
        )
        self.assertFalse(action)
        self.assertEqual(code, "")
        self.assertEqual(mode, "unparsed")

    def test_deepseek_auto_accepts_direct_fenced_code_without_calling_it_action(self):
        adapter = probe.parser_adapter("/models/deepseek-coder-1.3b-instruct", "auto")
        self.assertEqual(adapter, "cross_family")
        action, code, mode = probe.extract_react_response(
            "Here is the solution:\n```python\ndef f(): return 1\n```", adapter
        )
        self.assertFalse(action)
        self.assertIn("def f", code)
        self.assertEqual(mode, "direct_fenced_code")

    def test_action_and_final_answer_paths_are_unchanged(self):
        action, code, mode = probe.extract_react_response(
            "Thought: try\nAction: run_tests\nAction Input: ```python\ndef f():\n return 1\n```",
            "legacy",
        )
        self.assertTrue(action)
        self.assertIn("def f", code)
        self.assertEqual(mode, "react_action")

        action, code, mode = probe.extract_react_response(
            "Final Answer: ```python\ndef f():\n return 2\n```", "legacy"
        )
        self.assertFalse(action)
        self.assertIn("return 2", code)
        self.assertEqual(mode, "final_answer")

    def test_native_fc_template_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "tokenizer_config.json"
            cfg.write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
            status, _ = probe.native_fc_template_status(tmp)
            self.assertEqual(status, "unsupported")

            cfg.write_text(
                json.dumps({"chat_template": "{% if tools %}{{ tools }}{% endif %}"}),
                encoding="utf-8",
            )
            status, _ = probe.native_fc_template_status(tmp)
            self.assertEqual(status, "supported")


if __name__ == "__main__":
    unittest.main()
