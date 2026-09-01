#!/usr/bin/env python3
"""在 BFCL 的 MODEL_CONFIG_MAPPING 里注册 Qwen2.5-Coder 尺寸梯子。

**为什么必须改 BFCL：** BFCL 出厂注册表里没有 Qwen2.5-Coder，而它是四个家族里
唯一表现出 censoring 的那一支（Llama 解析正常、Mistral 直接 400、DeepSeek 模板
不注入）。不注册就只能报 Llama，那是有效但很弱的阴性结果。

**改动范围（论文里要如实写这一段）：**
  · 只向 MODEL_CONFIG_MAPPING 追加模型条目
  · 复用 BFCL 自带的 QwenFCHandler，不新写 handler
  · 不改动任何评测、解析、判分逻辑
  · 幂等：重复执行不会重复追加
改动前后的文件 sha256 都会打印，随论文附录一起记录。

用法: python register_qwen_coder.py [--revert]
"""
import argparse, hashlib, re, shutil, sys
from pathlib import Path

CFG = Path("/root/bench-venv/lib/python3.12/site-packages/bfcl_eval/constants/model_config.py")
MARK_A = "    # === P1: Qwen2.5-Coder ladder (added by register_qwen_coder.py) ==="
MARK_B = "    # === end P1 additions ==="

SIZES = ["1.5B", "3B", "7B", "14B", "32B"]


def entry(sz):
    mid = f"Qwen/Qwen2.5-Coder-{sz}-Instruct"
    return f'''    "{mid}-FC": ModelConfig(
        model_name="{mid}",
        display_name="Qwen2.5-Coder-{sz}-Instruct (FC)",
        url="https://huggingface.co/{mid}",
        org="Qwen",
        license="apache-2.0",
        model_handler=QwenFCHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    ),'''


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()

    if not CFG.exists():
        sys.exit(f"找不到 {CFG}")
    src = CFG.read_text(encoding="utf-8")
    before = sha(CFG)

    if a.revert:
        if MARK_A not in src:
            print("没有本脚本的改动，无需回滚"); return
        src = re.sub(re.escape(MARK_A) + r".*?" + re.escape(MARK_B) + r"\n", "", src, flags=re.S)
        CFG.write_text(src, encoding="utf-8")
        print(f"已回滚   {before} → {sha(CFG)}")
        return

    if MARK_A in src:
        print(f"条目已存在（幂等），不重复追加。当前 sha256[:20] = {before}")
        return

    if "QwenFCHandler" not in src:
        sys.exit("★ 该 BFCL 版本没有 QwenFCHandler，模板不适用，停止")

    # 备份原件，便于核对与回滚
    bak = CFG.with_suffix(".py.p1-orig")
    if not bak.exists():
        shutil.copy2(CFG, bak)

    # 追加到 MODEL_CONFIG_MAPPING 字典末尾的右花括号之前
    m = list(re.finditer(r"\nMODEL_CONFIG_MAPPING\s*[:=]", src))
    if not m:
        sys.exit("★ 找不到 MODEL_CONFIG_MAPPING")
    start = m[-1].end()
    close = src.index("\n}", start)          # 字典结束
    block = "\n" + MARK_A + "\n" + "\n".join(entry(s) for s in SIZES) + "\n" + MARK_B
    src = src[:close] + block + src[close:]
    CFG.write_text(src, encoding="utf-8")

    print(f"已注册 {len(SIZES)} 个条目：")
    for s in SIZES:
        print(f"  Qwen/Qwen2.5-Coder-{s}-Instruct-FC   →  handler=QwenFCHandler（BFCL 自带）")
    print(f"\nmodel_config.py sha256[:20]   改前 {before}   改后 {sha(CFG)}")
    print(f"原件备份 {bak}")
    print("\n改动范围：仅追加模型条目。未改动任何评测 / 解析 / 判分逻辑。")


if __name__ == "__main__":
    main()
