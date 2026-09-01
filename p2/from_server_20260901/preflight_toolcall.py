#!/usr/bin/env python3
"""跑任何 function-calling 评测之前，先花 5 秒确认接口是通的。

本文全部静默失败都会被下面两步之一抓住：
  1. auto 模式发一条 canonical 请求 —— 断言解析出调用、name 正确、arguments 可解析
  2. required 模式重跑一次做阳性对照 —— 区分「模型不调用」与「管线坏了」

用法:  python preflight_toolcall.py --port 8000 [--model <served-name>]
退出码: 0 全通过 / 1 auto 未解析出调用（可能是模型行为，看步骤 2）/ 2 管线故障
"""
import argparse, json, sys, urllib.request, urllib.error

TOOL = [{"type": "function", "function": {
    "name": "run_tests",
    "description": "Submit Python code to a test runner; returns pass/fail and errors.",
    "parameters": {"type": "object",
                   "properties": {"code": {"type": "string",
                                           "description": "Complete Python source."}},
                   "required": ["code"], "additionalProperties": False}}}]

def ask(port, model, choice):
    body = {"model": model, "temperature": 0.0, "max_tokens": 256,
            "messages": [{"role": "system", "content":
                          "You are a Python assistant. You may call run_tests to check your code."},
                         {"role": "user", "content": "Implement add(a, b) returning a + b."}],
            "tools": TOOL, "tool_choice": choice}
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0], None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode('utf-8','replace')[:300]}"
    except Exception as e:
        return None, str(e)

def check(ch, label):
    msg = ch["message"]; tcs = msg.get("tool_calls") or []
    if not tcs:
        print(f"  [{label}] tool_calls 为空  finish={ch.get('finish_reason')}")
        print(f"           content 前 120: {(msg.get('content') or '')[:120]!r}")
        return False
    fn = tcs[0].get("function") or {}
    name, args = fn.get("name"), fn.get("arguments")
    ok_name = name == "run_tests"
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
        ok_args = isinstance(parsed, dict) and isinstance(parsed.get("code"), str) and parsed["code"].strip()
    except Exception:
        parsed, ok_args = None, False
    print(f"  [{label}] tool_calls={len(tcs)}  name={name!r} {'✅' if ok_name else '❌ 期望 run_tests'}"
          f"  arguments {'✅ 含字符串 code' if ok_args else '❌ 不可解析或 code 非字符串'}")
    if not ok_name:
        print("           ← 模型调用了未声明的工具；parser 通常不校验工具名，"
              "会被记成一次有效调用")
    return ok_name and ok_args

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None, help="不传则从 /v1/models 取第一个")
    a = ap.parse_args()
    model = a.model
    if not model:
        try:
            model = json.loads(urllib.request.urlopen(
                f"http://localhost:{a.port}/v1/models", timeout=30).read())["data"][0]["id"]
        except Exception as e:
            print(f"❌ 无法连接 /v1/models: {e}"); return 2
    print(f"模型: {model}\n")

    ch, err = ask(a.port, model, "auto")
    if err:
        print(f"❌ auto 请求失败: {err}")
        if "enable-auto-tool-choice" in err or "tool_choice" in err:
            print("   → 服务端缺 --enable-auto-tool-choice 与 --tool-call-parser")
        return 2
    auto_ok = check(ch, "auto")

    ch2, err2 = ask(a.port, model, "required")
    if err2:
        print(f"❌ required 请求失败: {err2}"); return 2
    req_ok = check(ch2, "required")

    print()
    if auto_ok:
        print("✅ 通过：auto 模式下解析出合规调用，接口可用。"); return 0
    if req_ok:
        print("⚠️  auto 为 0 但 required 正常 —— 管线可用，是模型在 auto 下不产出该格式。")
        print("    这正是本文描述的静默错配：评测会记成『该模型不使用工具』。")
        print("    建议：检查原始 content 里是否已有格式完好的调用（离线重解析）。")
        return 1
    print("❌ auto 与 required 均未解析出合规调用 —— 管线故障，先修配置再跑评测。")
    return 2

if __name__ == "__main__":
    sys.exit(main())
