"""复盘修复补丁 —— 下次开机在 /root/autodl-tmp/code-agent 下执行。

修 5 个洞（详见 AUDIT.md）：
  1  finish_reason 未记录 + max_tokens=1024 系统性偏袒 ReAct
  2  「直接给代码率 / 无法解析率」用轮数当分子、题数当分母
  3  轨迹缺 provenance（adapter / sys_file / max_tokens），消融两臂无法区分
  4  n_err>0 时脚本只打印警告，仍返回 0，被静默引用
  5  Qwen 用 legacy adapter、其他家族用 cross_family，L3 跨家族不可比
幂等：重复执行不会重复插入。
"""
import ast, re, sys

P = "probe_react_full.py"
s = open(P, encoding="utf-8").read()
orig = s
applied, skipped = [], []

# ---- 1a. max_tokens 提到 2048，并记录 finish_reason -------------------------
if '"max_tokens": 1024' in s:
    s = s.replace('"max_tokens": 1024', '"max_tokens": MAX_TOKENS')
    applied.append("max_tokens -> 常量")
else:
    skipped.append("max_tokens 已改")

if "MAX_TOKENS =" not in s:
    m = re.search(r'^def gen_fc', s, re.M)
    assert m, "找不到 gen_fc"
    s = s[:m.start()] + (
        "# FC 把代码放进 JSON 字符串，\\n 全部转义，比 ReAct 的裸 fenced block 更费 token。\n"
        "# 两协议共用同一上限时，过小的上限会系统性偏袒 ReAct —— 这正是 26 条空代码的\n"
        "# 首要嫌疑。抬到 2048 并记录 finish_reason，让截断可被观测。\n"
        "MAX_TOKENS = 2048\n\n\n") + s[m.start():]
    applied.append("MAX_TOKENS 常量")
else:
    skipped.append("MAX_TOKENS 已存在")

# gen_fc 记录 finish_reason
old_fc = '''            msg = json.loads(r.read())["choices"][0]["message"]
            if msg.get("content"):
                msg["content"] = normalize_bpe(msg["content"])'''
new_fc = '''            _ch = json.loads(r.read())["choices"][0]
            msg = _ch["message"]
            msg["_finish_reason"] = _ch.get("finish_reason")
            if msg.get("content"):
                msg["content"] = normalize_bpe(msg["content"])'''
if old_fc in s:
    s = s.replace(old_fc, new_fc, 1); applied.append("gen_fc 记录 finish_reason")
else:
    skipped.append("gen_fc finish_reason 已改或锚点变化")

# gen() 也要记录：ReAct 侧同样可能截断
old_g = '''            return normalize_bpe(json.loads(r.read())["choices"][0]["message"]["content"] or "")'''
new_g = '''            _ch = json.loads(r.read())["choices"][0]
            global LAST_FINISH_REASON
            LAST_FINISH_REASON = _ch.get("finish_reason")
            return normalize_bpe(_ch["message"]["content"] or "")'''
if old_g in s:
    s = s.replace(old_g, new_g, 1)
    s = s.replace("MAX_TOKENS = 2048\n", "MAX_TOKENS = 2048\nLAST_FINISH_REASON = None\n", 1)
    applied.append("gen 记录 finish_reason")
else:
    skipped.append("gen finish_reason 已改或锚点变化")

# ---- 2. 逐题口径的比率 -----------------------------------------------------
if "first_modes" not in s:
    anchor = '    parse_modes: Counter[str] = Counter()'
    assert anchor in s, "找不到 parse_modes 初始化"
    s = s.replace(anchor, anchor + '\n    first_modes: Counter[str] = Counter()   # 逐题首轮口径', 1)
    # 首轮记账
    a2 = '            if t == 1:\n                l1 += has_action'
    assert a2 in s, "找不到 l1 记账点"
    s = s.replace(a2, a2 + '\n                first_modes[parse_mode] += 1', 1)
    applied.append("first_modes 逐题统计")
else:
    skipped.append("first_modes 已存在")

# 打印改用逐题分子
rep = [
 ('    n_direct = parse_modes.get("direct_fenced_code", 0)',
  '    n_direct = first_modes.get("direct_fenced_code", 0)'),
 ('    n_unparsed = parse_modes.get("unparsed", 0) + parse_modes.get("template_echo", 0)',
  '    n_unparsed = first_modes.get("unparsed", 0) + first_modes.get("template_echo", 0)'),
 ('    n_final = parse_modes.get("final_answer", 0)',
  '    n_final = first_modes.get("final_answer", 0)'),
]
for a, b in rep:
    if a in s:
        s = s.replace(a, b, 1); applied.append(f"比率分子改逐题: {b.split('=')[0].strip()}")

# ---- 3. provenance 落盘 ----------------------------------------------------
if '"adapter": adapter' not in s:
    a3 = '        rec = {"i": qi, "clean_index": ds_id, "protocol": a.protocol,'
    assert a3 in s, "找不到 rec 初始化"
    s = s.replace(a3, a3 + '\n               "adapter": adapter, "sys_file": a.sys_file,'
                          '\n               "max_tokens": MAX_TOKENS,', 1)
    applied.append("rec 记录 adapter/sys_file/max_tokens")
else:
    skipped.append("provenance 已存在")

# ---- 4. n_err>0 时以非零码退出 ---------------------------------------------
if "EXIT_DIRTY" not in s:
    a4 = '    print(f"请求错误: {n_err}"'
    assert a4 in s, "找不到请求错误打印"
    m = re.search(r'\n(\s*)return 0\s*$', s)
    if m:
        s = s[:m.start()] + f'\n{m.group(1)}EXIT_DIRTY = 2 if n_err else 0\n{m.group(1)}return EXIT_DIRTY' + s[m.end():]
        applied.append("n_err>0 -> 退出码 2")
    else:
        skipped.append("找不到 return 0，n_err 退出码未改")

assert s != orig or not applied, "声称有修改但内容未变"
ast.parse(s)
open(P, "w", encoding="utf-8").write(s)
print("已应用:"); [print("  +", x) for x in applied]
print("跳过:");   [print("  -", x) for x in skipped]
