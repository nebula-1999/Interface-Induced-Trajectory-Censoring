import json, glob, re
from collections import Counter

# 严判据：必须出现 {"name":"run_tests", ... "arguments" ... "code": "<字符串字面量>"}
# 且字符串里含真实 Python（def/class/return/import/lambda）。
# 排除两类污染：① 复述我注入的 schema（有 parameters/description 无 arguments）
#              ② 示意性伪代码（"code": 变量名，而非字符串字面量）
TIGHT = re.compile(
    r'"name"\s*:\s*"run_tests".{0,200}?"arguments"\s*:\s*\{.{0,80}?"code"\s*:\s*"(.{0,4000}?)"\s*\}',
    re.S)
REAL = re.compile(r'\\n|def |class |return |import |lambda ')

print(f"{'规模':<7}{'n':>5}{'解析出':>8}{'严判据真调用':>13}{'宽松强证据':>11}{'弱启发':>8}{'最终通过':>9}")
for f in sorted(glob.glob("traj_v5_Qwen*_fc_intent.jsonl"),
                key=lambda x: float(re.search(r"Qwen([\d.]+)B", x).group(1))):
    S = re.search(r"Qwen([\d.]+B)_", f).group(1)
    R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    first = [(r["turns"] or [{}])[0] for r in R]
    tight = 0
    for t in first:
        s = t.get("raw_output") or ""
        m = TIGHT.search(s)
        if m and REAL.search(m.group(1)):
            tight += 1
    forms = Counter(t.get("_fc_intent") for t in first if t.get("_fc_intent"))
    strong = forms.get("json_named_call", 0) + forms.get("xml_tool_call", 0)
    weak = forms.get("json_arguments", 0) + forms.get("json_code_obj", 0)
    print(f"{S:<7}{len(R):>5}{sum(1 for t in first if t.get('action')):>8}"
          f"{tight:>13}{strong:>11}{weak:>8}{sum(r['final_ok'] for r in R):>9}")
