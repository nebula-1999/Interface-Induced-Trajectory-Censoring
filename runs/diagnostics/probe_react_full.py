#!/usr/bin/env python3
"""ReAct 协议下的完整三层测试。

背景：function calling 下三层全否（0/1000 主动调用、0/256 重试、8-9/540 救回）；
换 ReAct 文本协议后**第一层跳到 100%**。但只验证了"会不会起步"，
后两层未知——若模型发一次 Action 拿到 Observation 就收手，
那 ReAct 只解决了起步，多轮 debug 依然不存在，训练照样没有有效信号。

三层：
  L1 发起率      —— 首轮是否产出 Action
  L2 持续率      —— 喂回**真实**的失败 Observation 后，是否再发 Action
  L3 救回率      —— 完整跑多轮，最终是否通过（对照 function calling 的 8-9/540）

不依赖 vLLM 的 tool parser：ReAct 是纯文本协议，自己解析 Action Input。
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request

from datasets import load_dataset

from sandbox import run_tests

# 2×2 设计：协议(fc/react) × 指令强度(optional/mandatory)。
# 此前 ReAct 写"必须先用 Action"、FC 写"你可以调用"，两个变量同时变了，
# 100% vs 0% 无法归因到协议本身。四格全跑才能分离。
FC_OPTIONAL = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。"""

FC_MANDATORY = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

**你必须先调用 run_tests 工具验证你的代码**，不要直接给出答案。
工具会返回测试通过情况与报错；若有失败，请修改代码并再次调用 run_tests。"""

REACT_OPTIONAL = """你是一个 Python 编程助手，可以按以下格式工作：

Thought: 你的思考
Action: run_tests
Action Input: ```python
<完整代码>
```
Observation: （由系统填写测试结果）

Thought: 我已确认代码正确
Final Answer: ```python
<最终代码>
```

你可以用 Action 提交代码验证，也可以直接给 Final Answer。"""

SYS = """你是一个 Python 编程助手，按以下格式逐步工作：

Thought: 你的思考
Action: run_tests
Action Input: ```python
<完整代码>
```
Observation: （由系统填写测试结果）

... 可以重复以上循环 ...

Thought: 我已确认代码正确
Final Answer: ```python
<最终代码>
```

**必须先用 Action 提交代码验证，再给 Final Answer。**"""

_ACTION = re.compile(r"^\s*Action\s*:\s*run_tests", re.M | re.I)
_CODE = re.compile(r"Action\s+Input\s*:\s*```(?:python|py)?\s*\n(.*?)```", re.S | re.I)
_FINAL = re.compile(r"Final\s+Answer\s*:\s*```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


FC_TOOLS = [{"type": "function", "function": {
    "name": "run_tests",
    "description": "把你写的 Python 代码交给测试运行，返回通过情况与报错。",
    "parameters": {"type": "object",
                   "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]

FC_SYS = """你是一个 Python 编程助手。请根据用户的描述实现所要求的函数。

你可以调用 run_tests 工具把代码交给测试运行，工具会返回通过情况与报错，
你可以据此修改代码再次提交。"""


def gen_fc(port, model, messages):
    """function calling 一轮。返回 (是否发起调用, 代码, 原始 message)。"""
    body = {"model": model, "temperature": 0.0, "max_tokens": 1024,
            "messages": messages, "tools": FC_TOOLS, "tool_choice": "auto"}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            msg = json.loads(r.read())["choices"][0]["message"]
    except Exception as e:
        return False, "", {"content": f"__ERROR__{e}"}
    tcs = msg.get("tool_calls") or []
    if tcs:
        args = tcs[0]["function"].get("arguments")
        try:
            code = (json.loads(args) if isinstance(args, str) else args).get("code", "")
        except Exception:
            code = ""
        return True, code, msg
    # 没发起调用时，退回从正文里取代码（模型多半直接给了答案）
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", msg.get("content") or "", re.S)
    return False, (m.group(1) if m else ""), msg


def gen(port, model, messages, max_tokens=1024):
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": messages, "stop": ["Observation:"]}   # 别让它自己编造 Observation
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return f"__ERROR__{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--strength", choices=["optional", "mandatory"],
                    default="mandatory", help="指令强度：可选 vs 必须")
    ap.add_argument("--protocol", choices=["react", "fc"], default="react",
                    help="react=ReAct 文本协议；fc=OpenAI function calling。"
                         "两者用同一批题、同一沙箱执行器，唯一差别是交互协议。")
    ap.add_argument("--out", default="", help="逐题 JSONL 输出路径")
    a = ap.parse_args()

    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    clean = json.load(open("clean_ids.json", encoding="utf-8"))["clean_index"]
    idxs = clean[:a.n]
    items = [(kc[i]["question"], kc[i]["test"], i) for i in idxs]

    l1 = l2_denom = l2 = rescued = n_err = 0
    fout = open(a.out, "w", encoding="utf-8") if a.out else None
    passed_final = passed_turn1 = 0
    turn_hist = {}

    for qi, (q, test, ds_id) in enumerate(items):
        is_fc = (a.protocol == "fc")
        sysmsg = ({("fc", "optional"): FC_OPTIONAL,
                   ("fc", "mandatory"): FC_MANDATORY,
                   ("react", "optional"): REACT_OPTIONAL,
                   ("react", "mandatory"): SYS})[(a.protocol, a.strength)]
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}]
        first_ok = final_ok = False
        turns = 0
        pending = None
        rec = {"i": qi, "clean_index": ds_id, "protocol": a.protocol,
               "strength": a.strength, "model": a.model, "turns": []}
        for t in range(1, a.max_turns + 1):
            if is_fc:
                if pending is not None:
                    has_action, code, msg = pending
                    pending = None
                else:
                    has_action, code, msg = gen_fc(a.port, a.model, msgs)
                out = msg.get("content") or ""
                if out.startswith("__ERROR__"):
                    n_err += 1
                    break
            else:
                out = pending if pending is not None else gen(a.port, a.model, msgs)
                pending = None
                if out.startswith("__ERROR__"):
                    n_err += 1
                    break
                has_action = bool(_ACTION.search(out))
                m = _CODE.search(out) if has_action else _FINAL.search(out)
                code = m.group(1) if m else ""
            if t == 1:
                l1 += has_action
            if not code:
                rec["turns"].append({"t": t, "action": has_action, "code": "", "passed": None})
                break
            turns = t
            res = run_tests(code, test, mode="pytest")
            final_ok = res.all_passed
            if t == 1:
                first_ok = res.all_passed
            rec["turns"].append({"t": t, "action": has_action, "code": code,
                                 "raw_output": out[:4000],
                                 "passed": res.passed, "total": res.total,
                                 "all_passed": res.all_passed,
                                 "obs": res.stderr[-400:]})
            if res.all_passed or not has_action:
                break
            if t == 1:
                l2_denom += 1
            obs = f"{res.passed}/{res.total} 个测试通过。\n{res.stderr[-500:]}"
            if is_fc:
                msgs += [msg, {"role": "tool", "tool_call_id":
                               (msg.get("tool_calls") or [{}])[0].get("id", "c0"),
                               "name": "run_tests", "content": obs}]
            else:
                msgs += [{"role": "assistant", "content": out},
                         {"role": "user", "content": f"Observation: {obs}"}]
            if t >= a.max_turns:
                break              # 最后一轮不必再生成，否则白烧一次推理
            nxt = gen_fc(a.port, a.model, msgs) if is_fc else gen(a.port, a.model, msgs)
            if t == 1 and (nxt[0] if is_fc else bool(_ACTION.search(nxt))):
                l2 += 1
            pending = nxt          # 下一轮直接消费，不重复生成
        turn_hist[turns] = turn_hist.get(turns, 0) + 1
        passed_turn1 += first_ok
        passed_final += final_ok
        if final_ok and not first_ok:
            rescued += 1
        rec.update(first_ok=first_ok, final_ok=final_ok, n_turns=turns)
        if fout:                      # 逐题落盘并 flush：中途关机也不会全丢
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
    if fout:
        fout.close()

    n = len(items)
    print(f"\n模型: {a.model}   协议={a.protocol}  强度={a.strength}  n={n}")
    print(f"请求错误: {n_err}" + ("   ⚠️ 非零，本组结果不可用" if n_err else ""))
    print(f"L1 首轮发起 Action     : {l1}/{n} = {l1/n:.0%}")
    print(f"L2 首轮失败后继续      : {l2}/{l2_denom if l2_denom else 1} "
          f"= {l2/l2_denom if l2_denom else 0:.0%}   （分母=首轮未通过数）")
    print(f"L3 首轮通过 / 最终通过 : {passed_turn1}/{n} = {passed_turn1/n:.0%}"
          f"  →  {passed_final}/{n} = {passed_final/n:.0%}")
    n_fail1 = n - passed_turn1
    print(f"   绝对救回率           : {rescued}/{n} = {rescued/n:.1%}")
    print(f"   **条件救回率**       : {rescued}/{n_fail1 if n_fail1 else 1} = "
          f"{rescued/n_fail1 if n_fail1 else 0:.1%}   （分母=首轮未通过数）")
    print(f"   ※ 绝对值跨规模不可比：大模型首轮通过率高，剩下的失败题更难")
    print(f"   轮次分布            : {dict(sorted(turn_hist.items()))}"
          f"   （0 = 未产出任何可执行代码）")
    print("\n判读：L2 高 → ReAct 真正解决了多轮，可以据此重跑训练；")
    print("      L2 低 → ReAct 只解决起步，多轮仍不存在，别改训练架构。")


if __name__ == "__main__":
    main()
