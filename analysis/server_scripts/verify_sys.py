import re
old = open("probe_toolcall2.py", encoding="utf-8").read()
m = re.search(r'SYSTEM\s*=\s*"""(.*?)"""', old, re.S)
assert m, "没找到 probe_toolcall2.py 的 SYSTEM"
legacy = m.group(1).strip()
mine = open("sys_legacy_train.txt", encoding="utf-8").read().strip()
print("与 0/1000 那批完全一致:", legacy == mine)
if legacy != mine:
    print("--- 原文 ---"); print(repr(legacy))
    print("--- 我的 ---"); print(repr(mine))
src = open("probe_react_full.py", encoding="utf-8").read()
m2 = re.search(r'FC_OPTIONAL\s*=\s*"""(.*?)"""', src, re.S)
fcopt = m2.group(1).strip()
print()
print("FC_OPTIONAL 是否为 legacy 去掉尾句:",
      legacy.replace("请给出完整的 Python 代码，包含所需的 import。", "").strip() == fcopt)
print("唯一差异 =", repr(legacy[len(fcopt.rstrip('。')):][:60]) if legacy.startswith(fcopt[:20]) else "见上")
