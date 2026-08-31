import json, re
f = "traj_v5_Qwen32B_fc_intent.jsonl"
R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
hits = [(r, (r["turns"] or [{}])[0]) for r in R
        if (r["turns"] or [{}])[0].get("_fc_intent") == "json_named_call"]
print(f"32B 强证据条数: {len(hits)}\n")

真调用 = schema回显 = 其他 = 0
for r, t in hits:
    s = t.get("raw_output") or ""
    m = re.search(r'\{[^{}]*"name"\s*:\s*"run_tests"[^{}]*(\{[^{}]*\})?[^{}]*\}', s, re.S)
    seg = m.group(0) if m else ""
    has_args = '"arguments"' in seg
    has_params = '"parameters"' in seg or '"description"' in seg or '"type": "object"' in seg
    has_realcode = bool(re.search(r'"code"\s*:\s*"[^"]*def ', s))
    if has_params and not has_args: schema回显 += 1
    elif has_args and has_realcode: 真调用 += 1
    else: 其他 += 1
print(f"  真调用（有 arguments 且 code 里含 def）: {真调用}")
print(f"  schema 回显（有 parameters/description，无 arguments）: {schema回显}")
print(f"  其他 / 判不准: {其他}")
print("\n--- 前 3 条命中的原文片段 ---")
for r, t in hits[:3]:
    s = t.get("raw_output") or ""
    i = s.find('"name"')
    print(f"\n[i={r['i']}] parse_mode={t.get('parse_mode')}")
    print("   ", repr(s[max(0, i-120):i+260]))
