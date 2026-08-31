import json, glob, itertools
def load(f): return {r["clean_index"]: r for r in (json.loads(l) for l in open(f,encoding="utf-8") if l.strip())}
R=[load(f) for f in sorted(glob.glob("traj_v13_Llama8B_react_t06_s*.jsonl"))]
K=sorted(set(R[0]))
print("=== ReAct 三 seed：通过的是不是同一批题 ===")
sets=[frozenset(k for k in K if d[k]["final_ok"]) for d in R]
for i,j in itertools.combinations(range(3),2):
    inter=len(sets[i]&sets[j]); union=len(sets[i]|sets[j])
    print(f"  s{i+1} vs s{j+1}: 各 {len(sets[i])}/{len(sets[j])}  交集 {inter}  并集 {union}  Jaccard {inter/union:.3f}")
print(f"  三者交集 {len(sets[0]&sets[1]&sets[2])}   三者并集 {len(sets[0]|sets[1]|sets[2])}")
print()
print("=== 首轮 vs 最终：多轮是否把噪声洗掉 ===")
for i,d in enumerate(R):
    f1=frozenset(k for k in K if d[k]["first_ok"])
    print(f"  s{i+1}: 首轮 {len(f1)}  最终 {len(sets[i])}  救回 {len(sets[i]-f1)}  首轮过但最终没过 {len(f1-sets[i])}")
f1s=[frozenset(k for k in K if d[k]["first_ok"]) for d in R]
print(f"  首轮集合 Jaccard(s1,s2)={len(f1s[0]&f1s[1])/len(f1s[0]|f1s[1]):.3f}   最终集合 Jaccard(s1,s2)={len(sets[0]&sets[1])/len(sets[0]|sets[1]):.3f}")
print()
print("=== FC 三 seed 对照 ===")
F=[load(f) for f in sorted(glob.glob("traj_v13_Llama8B_fcstrict_t06_s*.jsonl"))]
fs=[frozenset(k for k in K if d[k]["final_ok"]) for d in F]
for i,j in itertools.combinations(range(3),2):
    inter=len(fs[i]&fs[j]); union=len(fs[i]|fs[j])
    print(f"  s{i+1} vs s{j+1}: 各 {len(fs[i])}/{len(fs[j])}  Jaccard {inter/union:.3f}")
