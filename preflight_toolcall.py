#!/usr/bin/env python3
"""Spend five seconds checking the interface before any function-calling evaluation.

One of these two checks catches every silent failure reported in the paper:
  1. Send one canonical auto request; require the right name and parseable arguments.
  2. Repeat in required mode to separate "the model did not call" from "the pipeline is broken."

Usage: python preflight_toolcall.py --port 8000 [--model <served-name>]
Exit codes: 0 pass / 1 auto produced no parsed call (inspect step 2) / 2 pipeline failure
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
        print(f"  [{label}] tool_calls is empty  finish={ch.get('finish_reason')}")
        print(f"           first 120 characters of content: {(msg.get('content') or '')[:120]!r}")
        return False
    fn = tcs[0].get("function") or {}
    name, args = fn.get("name"), fn.get("arguments")
    ok_name = name == "run_tests"
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
        ok_args = isinstance(parsed, dict) and isinstance(parsed.get("code"), str) and parsed["code"].strip()
    except Exception:
        parsed, ok_args = None, False
    print(f"  [{label}] tool_calls={len(tcs)}  name={name!r} {'✅' if ok_name else '❌ expected run_tests'}"
          f"  arguments {'✅ contains string code' if ok_args else '❌ unparseable or code is not a string'}")
    if not ok_name:
        print("           ← Undeclared tool: parsers may count it because they rarely validate names")
    return ok_name and ok_args

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None, help="default: first model returned by /v1/models")
    a = ap.parse_args()
    model = a.model
    if not model:
        try:
            model = json.loads(urllib.request.urlopen(
                f"http://localhost:{a.port}/v1/models", timeout=30).read())["data"][0]["id"]
        except Exception as e:
            print(f"❌ Could not connect to /v1/models: {e}"); return 2
    print(f"Model: {model}\n")

    ch, err = ask(a.port, model, "auto")
    if err:
        print(f"❌ auto request failed: {err}")
        if "enable-auto-tool-choice" in err or "tool_choice" in err:
            print("   → The server is missing --enable-auto-tool-choice and --tool-call-parser")
        return 2
    auto_ok = check(ch, "auto")

    ch2, err2 = ask(a.port, model, "required")
    if err2:
        print(f"❌ required request failed: {err2}"); return 2
    req_ok = check(ch2, "required")

    print()
    if auto_ok:
        print("✅ Passed: auto mode produced a compliant parsed call; the interface works."); return 0
    if req_ok:
        print("⚠️  auto produced zero calls but required works: the pipeline is functional, but the model does not emit this format under auto.")
        print("    This is the silent mismatch in the paper: the evaluation records that the model does not use tools.")
        print("    Recommendation: inspect raw content for a well-formed call and re-parse it offline.")
        return 1
    print("❌ Neither auto nor required produced a compliant parsed call. Fix the pipeline "
          "configuration before running the evaluation.")
    return 2

if __name__ == "__main__":
    sys.exit(main())
