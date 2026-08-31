#!/usr/bin/env python3
"""逐臂严格验收。脚本自带的 READY 门槛太松（Codex 指出四处），此处独立复核。

判定一条臂 OK，必须同时满足：
  1. 文件存在、每行 JSON 可解析
  2. 行数 == 期望 n（不是 ">0" 也不是 "存在即可"）
  3. request_error == 0
  4. provenance 全对：模型、协议、温度、max_tokens==2048、adapter
  5. manifest 里该臂 rc == 0
  6. 探针脚本 hash 与运行前钉下的一致（证明运行期未被修改）
缺失的臂计为 MISSING，同样导致整体不通过。
"""
import json, os, re, sys, hashlib, subprocess

EXPECT = [
    # (tag, 文件, 模型关键字, 协议, 温度)
    ("DS1.3b_react2048",  "traj_v13_DS1.3b_react.jsonl",  "deepseek-coder-1.3b", "react", 0.0),
    ("DS6.7b_react2048",  "traj_v13_DS6.7b_react.jsonl",  "deepseek-coder-6.7b", "react", 0.0),
    ("Llama32_1B_react2048", "traj_v13_Llama32_1B_react.jsonl", "Llama-3.2-1B", "react", 0.0),
    ("Llama32_3B_react2048", "traj_v13_Llama32_3B_react.jsonl", "Llama-3.2-3B", "react", 0.0),
]
for sd in (1, 2, 3):
    EXPECT.append((f"a9_react_s{sd}", f"traj_v13_Llama8B_react_t06_s{sd}.jsonl",
                   "Llama-3.1-8B", "react", 0.6))
for sd in (1, 2, 3):
    EXPECT.append((f"a9_fc_s{sd}", f"traj_v13_Llama8B_fcstrict_t06_s{sd}.jsonl",
                   "Llama-3.1-8B", "fc", 0.6))
N = 100

def manifest_rc(tag):
    try:
        txt = open("v13_manifest.txt", encoding="utf-8").read()
    except FileNotFoundError:
        return None
    m = re.search(rf"^{re.escape(tag)} rc=(\d+) lines=(\d+)", txt, re.M)
    return (int(m.group(1)), int(m.group(2))) if m else None

def hash_ok():
    try:
        pinned = {}
        for line in open("v13_pinned_hashes.txt", encoding="utf-8"):
            p = line.split()
            if len(p) == 2:
                pinned[p[1]] = p[0]
        cur = hashlib.sha256(open("probe_react_full.py", "rb").read()).hexdigest()
        return cur == pinned.get("probe_react_full.py"), cur[:16], pinned.get("probe_react_full.py", "?")[:16]
    except Exception as e:
        return False, "err", str(e)[:20]

hok, cur, pin = hash_ok()
print(f"探针 hash: 当前 {cur} / 钉定 {pin} → {'一致' if hok else '**已变更**'}")
print(f"{'臂':<22}{'n':>5}{'n_err':>7}{'rc':>4}  判定  说明")
print("-" * 92)
allok = hok
for tag, f, want_m, want_p, want_t in EXPECT:
    if not os.path.exists(f):
        print(f"{tag:<22}{'-':>5}{'-':>7}{'-':>4}  MISSING"); allok = False; continue
    bad = []
    try:
        R = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    except Exception as e:
        print(f"{tag:<22}{'-':>5}{'-':>7}{'-':>4}  BAD  json_error={e}"); allok = False; continue
    n = len(R)
    if n != N: bad.append(f"n={n}≠{N}")
    errs = sum(1 for r in R for t in r["turns"] if (t.get("raw_output") or "").startswith("__ERROR__"))
    if errs: bad.append(f"n_err={errs}")
    r0 = R[0] if R else {}
    if want_m not in str(r0.get("model", "")): bad.append(f"model={r0.get('model')}")
    if r0.get("protocol") != want_p: bad.append(f"protocol={r0.get('protocol')}")
    if abs(float(r0.get("temperature", -1)) - want_t) > 1e-9: bad.append(f"temp={r0.get('temperature')}")
    if int(r0.get("max_tokens", 0)) != 2048: bad.append(f"max_tokens={r0.get('max_tokens')}")
    if r0.get("adapter") != "cross_family": bad.append(f"adapter={r0.get('adapter')}")
    mr = manifest_rc(tag)
    rc = mr[0] if mr else None
    if mr is None: bad.append("manifest 无记录")
    elif rc != 0: bad.append(f"rc={rc}")
    verdict = "OK" if not bad else "BAD"
    if bad: allok = False
    print(f"{tag:<22}{n:>5}{errs:>7}{str(rc):>4}  {verdict:<5} {' '.join(bad)}")
print("-" * 92)
print("整体:", "全部通过" if allok else "**存在不通过项，不可作为统一配置结果引用**")
sys.exit(0 if allok else 1)
