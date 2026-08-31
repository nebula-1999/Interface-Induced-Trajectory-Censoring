#!/usr/bin/env python3
"""调用意图的**唯一**判据。正文、表格、图片必须全部从这里取数。

此前存在三套数字（手工 18 / 严判据 21 / 轨迹标签 23），来源是不同时期的临时脚本。
分三档，报告中一律注明用的是哪一档：

  tight   严判据：{"name":"run_tests", ... "arguments" ... "code":"<字符串字面量含真实 Python>"}
          排除两类污染：① 复述注入的 schema（有 parameters/description 无 arguments）
                        ② 示意性伪代码（"code": 变量名，而非字符串字面量）
  strong  宽松强证据：轨迹 _fc_intent 标签为 json_named_call / xml_tool_call
  weak    弱启发：仅有 JSON 结构（json_arguments / json_code_obj），可能只是普通回答
"""
import json, os, re

TIGHT = re.compile(
    r'"name"\s*:\s*"run_tests".{0,200}?"arguments"\s*:\s*\{.{0,80}?"code"\s*:\s*"(.{0,4000}?)"\s*\}',
    re.S)
REAL = re.compile(r'\\n|def |class |return |import |lambda ')
STRONG_TAGS = {"json_named_call", "xml_tool_call"}
WEAK_TAGS = {"json_arguments", "json_code_obj"}


def is_tight_call(raw: str) -> bool:
    m = TIGHT.search(raw or "")
    return bool(m and REAL.search(m.group(1)))


def classify(path):
    """返回该臂的四个计数：parsed / tight / strong / weak（均以**首轮**为口径）。"""
    R = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    first = [(r["turns"] or [{}])[0] for r in R]
    return {
        "n": len(R),
        "parsed": sum(1 for t in first if t.get("action")),
        "tight": sum(1 for t in first if is_tight_call(t.get("raw_output") or "")),
        "strong": sum(1 for t in first if t.get("_fc_intent") in STRONG_TAGS),
        "weak": sum(1 for t in first if t.get("_fc_intent") in WEAK_TAGS),
        "final_ok": sum(r["final_ok"] for r in R),
        "n_err": sum(1 for r in R for t in r["turns"]
                     if (t.get("raw_output") or "").startswith("__ERROR__")),
    }


if __name__ == "__main__":
    D = os.path.join(os.path.dirname(__file__), "..", "runs", "final")
    print(f"{'规模':<8}{'n':>5}{'解析出':>8}{'严判据':>8}{'宽松强':>8}{'弱启发':>8}{'最终通过':>10}")
    print("-" * 60)
    for s in ["1.5B", "3B", "7B", "14B", "32B"]:
        c = classify(os.path.join(D, f"traj_v5_Qwen{s}_fc_intent.jsonl"))
        print(f"{s:<8}{c['n']:>5}{c['parsed']:>8}{c['tight']:>8}{c['strong']:>8}"
              f"{c['weak']:>8}{c['final_ok']:>10}")
