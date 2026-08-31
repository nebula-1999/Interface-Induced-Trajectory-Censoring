import json, glob, os
from collections import Counter
def idset(f):
    try: return frozenset(json.loads(l)["clean_index"] for l in open(f,encoding="utf-8") if l.strip())
    except Exception: return None
print("=== 题目集合是否跨臂一致 ===")
sets = {}
for f in sorted(glob.glob("traj_v*.jsonl")):
    if "smoke" in f: continue
    s = idset(f)
    if s and len(s) == 100: sets.setdefault(s, []).append(os.path.basename(f))
print(f"  不同的 100 题集合数: {len(sets)}")
for i, (s, fs) in enumerate(sets.items()):
    print(f"   集合{i+1}: {len(fs)} 个臂  样例题号 {sorted(s)[:6]}")
    if len(sets) > 1: print(f"      文件: {fs[:4]}")

print("\n=== 各臂 provenance schema（字段是否齐全）===")
keys_seen = Counter()
for f in sorted(glob.glob("traj_v*.jsonl")):
    if "smoke" in f: continue
    try: r0 = json.loads(open(f,encoding="utf-8").readline())
    except Exception: continue
    keys_seen[frozenset(k for k in r0 if not k.startswith("turns"))] += 1
for i, (ks, n) in enumerate(keys_seen.most_common()):
    print(f"  schema{i+1} ({n} 个臂): {sorted(ks)}")

print("\n=== FC 臂的 max_tokens 实际值 ===")
for f in sorted(glob.glob("traj_v*.jsonl")):
    if "smoke" in f: continue
    try: r0 = json.loads(open(f,encoding="utf-8").readline())
    except Exception: continue
    if r0.get("protocol") == "fc":
        print(f"  {os.path.basename(f):<46} max_tokens={r0.get('max_tokens')} temp={r0.get('temperature')}")
