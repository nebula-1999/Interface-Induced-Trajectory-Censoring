import json, glob, os
def load(f):
    return {r["clean_index"]: r for r in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip())}
fs = sorted(glob.glob("traj_v13_Llama8B_react_t06_s*.jsonl"))
print(f"已有 {len(fs)} 个 seed 臂")
if len(fs) < 2:
    raise SystemExit("不足两个，无法比对")
D = [load(f) for f in fs]
K = sorted(set(D[0]) & set(D[1]))
print(f"配对题目 {len(K)}   provenance: temp={D[0][K[0]].get('temperature')} seed={D[0][K[0]].get('seed')} vs seed={D[1][K[0]].get('seed')}")
same_final = sum(1 for k in K if D[0][k]["final_ok"] == D[1][k]["final_ok"])
same_code  = sum(1 for k in K if (D[0][k]["turns"] or [{}])[0].get("code") == (D[1][k]["turns"] or [{}])[0].get("code"))
print(f"  最终通过完全相同的题: {same_final}/{len(K)}")
print(f"  首轮代码逐字相同的题: {same_code}/{len(K)}   ← 若接近 100 说明采样没生效")
print(f"  各臂最终通过: {[sum(d[k]['final_ok'] for k in K) for d in D]}")
