import json, glob, os
KD = {"HumanEval/32", "Mbpp/599"}
print("=== ReAct run 评测曲线（统一 540 口径）===")
print(f"  {'step':>6}{'turn1':>8}{'final':>8}{'rescue':>8}{'repair':>9}{'n':>6}")
for f in sorted(glob.glob("/root/autodl-tmp/runs/code-eval-grpo-seed42/step_*.jsonl")):
    R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    m = [x for x in R if x["channel"] == "multi" and x["task_id"] not in KD]
    r = [x for x in R if x["channel"] == "repair" and x["task_id"] not in KD]
    if not m: continue
    t1 = sum(1 for x in m if x["turns"] and x["turns"][0]["all_passed"])
    fin = sum(1 for x in m if x["turns"] and x["turns"][-1]["all_passed"])
    rp = sum(1 for x in r if x["turns"] and x["turns"][-1]["all_passed"])
    st = os.path.basename(f)[5:10].lstrip("0") or "0"
    print(f"  {st:>6}{t1:>8}{fin:>8}{fin-t1:>8}{rp:>9}{len(m):>6}")
print("\n  历史 FC run 的救回全程: 5–10，末-首 +1/-1/0")
