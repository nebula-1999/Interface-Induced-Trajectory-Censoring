#!/usr/bin/env python3
"""离线重解析矩阵：同一批原始输出，用不同 parser 规则重新解析。

目的：证明"服务端解析出 0"是 parser 与模型输出格式错配，
而不是我们的提取器写得差。全部离线，零 GPU。

四种规则：
  hermes_like     要求 <tool_call>…</tool_call> 包裹（vLLM 对 Qwen2.5 的推荐 parser）
  tools_like      要求 <tools>…</tools> 包裹（hanXen 专用 parser 的格式）
  bare_json       裸 JSON 对象，含 "name":"run_tests" 与 "arguments"
  tight           bare_json 且 arguments.code 为含真实 Python 的字符串字面量（本文判据）
"""
import json, os, re, sys

D = os.path.join(os.path.dirname(__file__), "..", "runs", "final")

RULES = {
    "hermes_like": re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>", re.S),
    "tools_like":  re.compile(r"<tools>\s*\{.*?\}\s*</tools>", re.S),
    "bare_json":   re.compile(r'\{[^{}]{0,200}"name"\s*:\s*"run_tests".{0,4000}?"arguments"', re.S),
}
TIGHT = re.compile(
    r'"name"\s*:\s*"run_tests".{0,200}?"arguments"\s*:\s*\{.{0,80}?"code"\s*:\s*"(.{0,4000}?)"\s*\}', re.S)
REAL = re.compile(r'\\n|def |class |return |import |lambda ')

def reparse(path):
    R = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    first = [(r["turns"] or [{}])[0] for r in R]
    out = {"n": len(R), "server": sum(1 for t in first if t.get("action"))}
    for name, pat in RULES.items():
        out[name] = sum(1 for t in first if pat.search(t.get("raw_output") or ""))
    out["tight"] = sum(1 for t in first
                       if (m := TIGHT.search(t.get("raw_output") or "")) and REAL.search(m.group(1)))
    return out

if __name__ == "__main__":
    print("同一批原始输出，四种 parser 规则离线重解析（首轮口径，n=100/档）")
    print(f"{'臂':<34}{'服务端':>8}{'hermes':>9}{'<tools>':>9}{'裸JSON':>9}{'严判据':>9}")
    print("-" * 80)
    arms = [(f"traj_v5_Qwen{s}_fc_intent.jsonl", f"Qwen-{s} (hermes 服务)") for s in
            ["1.5B", "3B", "7B", "14B", "32B"]]
    arms += [("traj_v6b_Qwen7B_fc_plugin.jsonl", "Qwen-7B (专用适配器服务)"),
             ("traj_v6_Llama8B_fc_strict.jsonl", "Llama-8B (llama3_json+strict)")]
    for f, lab in arms:
        p = os.path.join(D, f)
        if not os.path.exists(p): continue
        c = reparse(p)
        if c["server"] > 0:
            # parser 成功时调用进入结构化 tool_calls 字段、content 为空，
            # 对 content 再做离线重解析是**未定义**而非"零调用"。用 N/A，不用 0，
            # 否则表格本身会制造"适配器 84 但其他 parser 全 0"的视觉陷阱。
            print(f"{lab:<34}{c['server']:>8}{'N/A†':>9}{'N/A†':>9}{'N/A†':>9}{'N/A†':>9}")
        else:
            print(f"{lab:<34}{c['server']:>8}{c['hermes_like']:>9}{c['tools_like']:>9}"
                  f"{c['bare_json']:>9}{c['tight']:>9}")
    print("-" * 80)
    print("读法（重要）：")
    print("† 服务端成功解析时，调用被移入结构化 tool_calls 字段而 content 为空；")
    print("  对 content 的离线重解析因此是**未定义**，而非负结果。")
    print("  · 本矩阵只对服务端解析为 0 的臂有意义。")
    print("    当 parser 成功时，vLLM 把调用放进 tool_calls 而 content 为空，")
    print("    raw_output 里自然不含任何标签 —— 后两行的 0 是构造性的，")
    print("    **不表示没有调用**。")
    print("  · 前五行：hermes 列全 0 = 模型从不产出该格式；裸 JSON / 严判据列非零")
    print("    = 调用确实存在。两者之差即 parser 与输出格式错配吞掉的量。")
    print("  · 同一批字节、四种规则、离线重跑：排除『提取器写得差』这一解释。")
