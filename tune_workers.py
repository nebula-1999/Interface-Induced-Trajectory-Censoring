#!/usr/bin/env python3
"""在 2 GiB cgroup 下扫并行度与启动方式，给 run_many 定参。

fork 让 worker 继承父进程内存（这里是整个 KodCode 数据集），在小 cgroup 里
是净负担；spawn 干净启动但要 pickle 传参。实测决定，不猜。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

from datasets import load_dataset

from sandbox import _one

def main() -> None:
    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    clean = json.load(open("clean_ids.json", encoding="utf-8"))["clean_index"][:40]
    items = [(kc[i]["solution"], kc[i]["test"], 10.0, 512) for i in clean]

    print(f"样本 {len(items)} 条\n{'方式':>6} {'并行':>4} {'耗时':>8} {'吞吐':>9}")
    print("-" * 32)
    best = None
    for method in ("fork", "spawn"):
        for w in (1, 2, 4, 8, 12):
            t0 = time.time()
            try:
                with ProcessPoolExecutor(max_workers=w,
                                         mp_context=mp.get_context(method)) as ex:
                    list(ex.map(_one, items, chunksize=2))
            except Exception as e:
                # 2 GiB cgroup 下并行度过高不是变慢，是 worker 被 OOM killer 杀掉
                print(f"{method:>6} {w:>4} {'':>8} {type(e).__name__}")
                continue
            el = time.time() - t0
            thr = len(items) / el
            print(f"{method:>6} {w:>4} {el:>7.1f}s {thr:>7.1f}/s")
            if best is None or thr > best[0]:
                best = (thr, method, w)

    if best is None:
        print("\n所有配置都崩了——降低 mem_mb 或换更小的样本")
        return
    thr, method, w = best
    print(f"\n最优: {method} × {w} → {thr:.1f}/s")
    print(f"全量 7669 条预计 {7669 / thr / 60:.0f} 分钟")


if __name__ == "__main__":
    main()
