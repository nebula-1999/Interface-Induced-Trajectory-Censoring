#!/usr/bin/env python3
"""离线补跑 P2 五条臂的**首轮**代码，得到本该被记录的首轮通过率。

**为什么可以这样补。** 产出这五条臂的机器没装 pytest，每次执行都返回
`No module named pytest`，于是通过率、救回率、轮次分布全部作废。但**第一轮的
生成不可能受影响**：probe_react_full.py 第 480-484 行，进入循环时上下文只有
`[system, user]`、`pending` 为 None，此刻还不存在任何 obs。所以把存下来的首轮
代码拿去跑真正的测试，得到的就是当初环境正常时本来会记录的那个值——不是估计。

**不能这样补的部分。** 第二轮起模型看到的是 `No module named pytest` 而不是真实
报错，据此改出的代码已经长歪。最终通过率、条件救回率、L2、轮次分布只能重跑，
本脚本不产出、也不应被拿来近似它们。

**判据与当初逐字相同**：同一个 `sandbox.run_tests(code, test, mode="pytest")`，
同一个 `all_passed` 口径（见 probe_react_full.py 第 546-548 行）。

产物是**派生量**，写进 `replay_turn1_*.jsonl`，绝不回写轨迹里的 first_ok。
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox import run_many, safe_workers, MEM_MB, TIMEOUT_S  # noqa: E402

P = Path("/root/autodl-tmp/p2")
SIZES = ["1.5B", "3B", "7B", "14B", "32B"]


def main() -> None:
    from datasets import load_dataset
    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")

    prov = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "pytest": __import__("pytest").__version__,
        "python": sys.version.split()[0],
        "sandbox_sha256": hashlib.sha256((P / "sandbox.py").read_bytes()).hexdigest()[:20],
        "workers": safe_workers(), "timeout_s": TIMEOUT_S, "mem_mb": MEM_MB,
        "note": "derived quantity: offline execution of stored turn-1 programs",
    }
    print(json.dumps(prov, ensure_ascii=False))
    print()
    print(f"{'尺寸':<7}{'n':>4}{'首轮有代码':>11}{'首轮通过':>10}{'通过率':>9}")
    print("-" * 42)

    summary = {}
    for sz in SIZES:
        src = P / "runs" / f"traj_p2_Qwen2.5-{sz}-Instruct_fc.jsonl"
        if not src.exists():
            print(f"{sz:<7} 轨迹缺失，跳过"); continue
        recs = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]

        idx, items = [], []
        for r in recs:
            code = ((r.get("turns") or [{}])[0].get("code") or "").strip()
            if code:
                idx.append(r)
                items.append((code, kc[r["clean_index"]]["test"]))

        results = run_many(items, mode="pytest") if items else []

        out = P / "runs" / f"replay_turn1_Qwen2.5-{sz}-Instruct.jsonl"
        n_pass = 0
        with out.open("w", encoding="utf-8") as fh:
            done = {id(r): res for r, res in zip(idx, results)}
            for r in recs:
                res = done.get(id(r))
                ok = bool(res and res.all_passed)
                n_pass += ok
                fh.write(json.dumps({
                    "i": r["i"], "clean_index": r["clean_index"],
                    "has_turn1_code": res is not None,
                    "passed": getattr(res, "passed", None),
                    "total": getattr(res, "total", None),
                    "turn1_all_passed": ok,
                }, ensure_ascii=False) + "\n")

        n = len(recs)
        summary[sz] = {"n": n, "with_code": len(items), "turn1_pass": n_pass}
        print(f"{sz:<7}{n:>4}{len(items):>11}{n_pass:>10}{n_pass / n:>8.0%}")

    (P / "runs" / "replay_turn1_summary.json").write_text(
        json.dumps({"provenance": prov, "summary": summary},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写出 replay_turn1_*.jsonl 与 replay_turn1_summary.json（均为派生量）")


if __name__ == "__main__":
    main()
