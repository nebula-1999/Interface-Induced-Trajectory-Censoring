"""精确解析 verl step 行的指标。

必须按**完整指标名**匹配：直接 findall("tool_calls/mean") 会命中
timing_s/agent_loop/tool_calls/mean —— 那是工具执行**耗时（秒）**，
不是调用次数。把耗时当次数报是硬错误。
"""
import re, sys

def parse(path):
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        return {}, 0, True
    series = {}
    n_step = 0
    for line in re.findall(r"step:\d+ - (.*)", txt):
        n_step += 1
        # verl 把部分指标值包成 np.float64(...) / np.int32(...)，
        # 裸数字正则会整条漏掉 —— num_turns 就是这样被漏读的。
        pat = (r"(?:^|\s-\s)([A-Za-z_][A-Za-z0-9_/@.]*):"
               r"(?:np\.(?:float64|float32|int64|int32)\()?(-?[0-9.eE+-]+)\)?")
        for m in re.finditer(pat, line):
            try:
                series.setdefault(m.group(1), []).append(float(m.group(2)))
            except ValueError:
                pass
    # 崩溃判定必须看位置：训练跑完后的 atexit / p.join() 清理噪声也会打 Traceback，
    # 把它算成崩溃会误废掉完整的数据（ReAct 10 步就被误判过一次）。
    # 只有出现在**最后一条 step 行之前**的异常才算真崩。
    last_step = 0
    for m in re.finditer(r"step:\d+ -", txt):
        last_step = m.end()
    crashed = False
    for m in re.finditer(r"Traceback|AssertionError|CUDA out of memory|OutOfMemoryError", txt):
        if m.start() < last_step or last_step == 0:
            crashed = True
            break
    return series, n_step, crashed

def show(path, label):
    s, n, crashed = parse(path)
    print(f"  {label}  步数={n}  {'⚠️崩溃' if crashed else '正常'}")
    exact = [k for k in s if k == "tool_calls/mean"]
    timing = [k for k in s if k.endswith("tool_calls/mean") and k != "tool_calls/mean"]
    for k in ["num_turns/mean", "num_turns/max", "response_length/mean",
              "global_seqlen/mean", "critic/rewards/mean", "timing_s/step"]:
        if k in s:
            v = s[k]; print(f"     {k:<26} 均={sum(v)/len(v):.3f}  首={v[0]:.3f}  末={v[-1]:.3f}")
    print(f"     真正的 tool_calls/mean（次数）: {'有' if exact else '不存在于本版 verl'}")
    for k in timing:
        v = s[k]; print(f"     {k}（耗时秒，非次数）: 均={sum(v)/len(v):.2f}")
    return s, n, crashed
