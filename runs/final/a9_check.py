import json, glob, statistics as st

# 【2026-08-31 更正】本脚本此前把含请求错误的臂纳入了均值/标准差。
# FC seed3 有 1 条并行工具调用被 vLLM 拒收，属缺失数据，
# 通过率相关统计一律排除；其错误率本身仍有效（见 final_table.py 表 3）。
_VALID_ONLY = True

def _n_err_file(f):
    import json
    try:
        R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    except Exception:
        return 0
    return sum(1 for r in R for t in r["turns"]
               if (t.get("raw_output") or "").startswith("__ERROR__"))

def _valid(f):
    return (not _VALID_ONLY) or _n_err_file(f) == 0

print("=== a9_fc_s3 的那条错误 ===")
R=[json.loads(l) for l in open("traj_v13_Llama8B_fcstrict_t06_s3.jsonl",encoding="utf-8") if l.strip()]
for r in R:
    for t in r["turns"]:
        s=t.get("raw_output") or ""
        if s.startswith("__ERROR__"):
            print(f"  clean_index={r['clean_index']} turn={t['t']}")
            print(f"  {s[:400]}")
print("\n=== A9 三 seed 结果 ===")
def stats(pat, lab):
    vals=[]
    for f in sorted(f for f in glob.glob(pat) if _valid(f)):
        D=[json.loads(l) for l in open(f,encoding="utf-8") if l.strip()]
        ok=sum(r["final_ok"] for r in D)
        first=sum(r["first_ok"] for r in D)
        errs=sum(1 for r in D for t in r["turns"] if (t.get("raw_output") or "").startswith("__ERROR__"))
        vals.append(ok)
        print(f"  {lab:<8} {f.split('_s')[-1][:1]}: 首轮 {first:>3}  最终 {ok:>3}  n_err={errs}")
    if len(vals)>1:
        print(f"  {lab:<8} 最终通过 均值={st.mean(vals):.1f} 标准差={st.stdev(vals):.2f} 范围=[{min(vals)},{max(vals)}]")
    return vals
r=stats("traj_v13_Llama8B_react_t06_s*.jsonl","ReAct")
f=stats("traj_v13_Llama8B_fcstrict_t06_s*.jsonl","FC")
if r and f:
    print(f"\n  差距（均值）: {st.mean(r)-st.mean(f):.1f} 个点")
    print(f"  最坏情况（ReAct最低 vs FC最高）: {min(r)} vs {max(f)} = {min(r)-max(f):+d}")
    print(f"  对照 temp=0 单次: ReAct 80 / FC 61  （注意 ReAct 那次实跑 1024）")
