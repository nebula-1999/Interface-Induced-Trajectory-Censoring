import json, glob, re
from collections import Counter
STRONG = {"json_named_call", "xml_tool_call"}   # 含 run_tests 名或 <tools>/<tool_call> 标签
WEAK   = {"json_arguments", "json_code_obj"}    # 只有 arguments/code 结构，可能只是普通 JSON 回答
print(f"{'规模':<8}{'解析出':>7}{'强证据':>8}{'弱启发':>8}{'合计':>7}{'直接给代码':>11}{'最终通过':>9}")
for f in sorted(glob.glob("traj_v5_Qwen*_fc_intent.jsonl"),
                key=lambda x: float(re.search(r"Qwen([\d.]+)B", x).group(1))):
    S = re.search(r"Qwen([\d.]+B)_", f).group(1)
    R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    first = [(r["turns"] or [{}])[0] for r in R]
    forms = Counter(t.get("_fc_intent") for t in first if t.get("_fc_intent"))
    strong = sum(v for k, v in forms.items() if k in STRONG)
    weak = sum(v for k, v in forms.items() if k in WEAK)
    direct = sum(1 for t in first if t.get("parse_mode") == "fc_direct_fenced_code")
    print(f"{S:<8}{sum(1 for t in first if t.get('action')):>7}{strong:>8}{weak:>8}"
          f"{strong+weak:>7}{direct:>11}{sum(r['final_ok'] for r in R):>9}")
    print(f"        形式明细: {dict(forms)}")
