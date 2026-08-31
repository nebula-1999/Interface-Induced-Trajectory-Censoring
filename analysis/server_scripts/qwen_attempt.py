import json, re
from collections import Counter
R=[json.loads(l) for l in open("traj_v3_Qwen7B_fc_nosuffix.jsonl",encoding="utf-8") if l.strip()]
def attempt(s):
    if not s: return None
    # 明确的工具调用意图：JSON 里带 name/arguments，或 name=run_tests
    if re.search(r'"name"\s*:\s*"run_tests"', s): return "json_toolcall"
    if re.search(r'"arguments"\s*:', s): return "json_arguments"
    if re.search(r'```json', s) and "run_tests" in s: return "json_block_named"
    if re.search(r'\{\s*"code"\s*:', s): return "json_code_obj"
    if "run_tests" in s: return "mentions_run_tests_only"
    return None
c=Counter(); ex={}
for r in R:
    t=(r["turns"] or [{}])[0]
    a=attempt(t.get("raw_output"))
    c[a or "no_attempt"]+=1
    if a and a not in ex: ex[a]=(t.get("raw_output") or "")
print(f"n={len(R)}  首轮是否在尝试调用工具：")
for k,v in c.most_common(): print(f"   {k:<26} {v}")
hard=sum(v for k,v in c.items() if k in ("json_toolcall","json_arguments","json_block_named","json_code_obj"))
print(f"\n  明确的工具调用意图（非 hermes 格式）: {hard}/{len(R)}")
print(f"  vLLM 实际解析出的 tool_calls        : {sum(1 for r in R if (r['turns'] or [{}])[0].get('action'))}/{len(R)}")
for k,v in ex.items():
    m=re.search(r'(```json.{0,200}|\{[^{}]{0,200}"name"[^{}]{0,200}\})', v, re.S)
    print(f"\n  --- {k} 样例 ---\n  {repr((m.group(1) if m else v[:200]))[:300]}")
