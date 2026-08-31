import json
from collections import Counter
R = [json.loads(l) for l in open("traj_v3_Llama8B_fc.jsonl", encoding="utf-8") if l.strip()]
print(f"n={len(R)}  max_tokens={R[0].get('max_tokens')}  adapter={R[0].get('adapter')}")

first = [(r, (r["turns"] or [{}])[0]) for r in R]
empty = [(r, t) for r, t in first if not t.get("code")]
print(f"\n首轮空代码: {len(empty)}/{len(R)}   (v2 旧版是 26/100)")
if empty:
    print("  finish_reason:", Counter(t.get("_finish_reason") for _, t in empty))
    print("  arg_err      :", Counter(t.get("_fc_arg_err") for _, t in empty))
    print("  arg_keys     :", Counter(str(t.get("_fc_arg_keys")) for _, t in empty))
    print("  raw_args 长度:", [len(t.get("_fc_raw_args") or "") for _, t in empty][:12])
    for r, t in empty[:2]:
        print(f"\n  --- i={r['i']} 原始 arguments 尾部 ---")
        print("  ", repr((t.get("_fc_raw_args") or "")[-220:]))

print("\n全部首轮 finish_reason:", Counter(t.get("_finish_reason") for _, t in first))
tr = [t for r in R for t in r["turns"]]
print("全部轮次 finish_reason:", Counter(t.get("_finish_reason") for t in tr))
print("被 length 截断的轮次:", sum(1 for t in tr if t.get("_finish_reason") == "length"))
print(f"\n首轮通过 {sum(r['first_ok'] for r in R)}/{len(R)}   最终通过 {sum(r['final_ok'] for r in R)}/{len(R)}")
