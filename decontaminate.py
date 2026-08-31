#!/usr/bin/env python3
"""KodCode-Light-RL-10K 对 HumanEval+ / MBPP+ 的去污染。

MBPP 那个坑（29% 的 train 直接落在评测集里）的翻版检查。KodCode-Light-RL-10K
没有 benchmark_similarity 字段——那是 484K 的 KodCode-V1 才有的——所以自己做。

方法：n-gram 倒排索引 + containment，即 LLaMA / GPT-3 系列去污染的标准做法。
两个独立通道分别比对，任一命中都算可疑：

  A 问题通道：KodCode.question      vs  HumanEval+.prompt / MBPP+.prompt
  B 解法通道：KodCode.solution      vs  HumanEval+.canonical_solution / MBPP+.code

用 containment = |A ∩ B| / min(|A|, |B|) 而不是 Jaccard：训练题与评测题长度
差异大，Jaccard 会被长度差稀释，漏掉"短题被长题包含"这种真污染。

产出：contamination_report.json（逐条最大 containment）与 clean_ids.json。
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from datasets import load_dataset

# 分层 n：短评测题在大 n 下产生不了任何 n-gram（542 条里有 4 条完全落空、
# 224 条不足 10 个），必须回落到小 n，否则它们根本没被查过。
# 不再往 n=3 以下探——那一层里 "write a python function" 之类的通用短语
# 会让所有训练题全部命中。
NS_TEXT = [8, 5]
NS_CODE = [10, 6]
MIN_GRAMS = 5          # 低于这个数量的 n-gram，containment 的分母不可信

N_TEXT, N_CODE = NS_TEXT[0], NS_CODE[0]   # 供 audit_decontam 复用

THRESHOLDS = [0.10, 0.20, 0.30, 0.50, 0.80]


# ---------------------------------------------------------------------------
# 归一化

def norm_text(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split()


def norm_code(s: str) -> list[str]:
    """去 docstring / 注释后按标识符与符号切分。

    保留标识符（不做变量重命名归一化）——函数名和关键逻辑正是
    "同一道题"最强的指纹。
    """
    s = s or ""
    s = re.sub(r'""".*?"""', " ", s, flags=re.S)
    s = re.sub(r"'''.*?'''", " ", s, flags=re.S)
    s = re.sub(r"#.*", " ", s)
    return re.findall(r"[A-Za-z_]\w*|[^\sA-Za-z_0-9]", s)


def ngrams(toks: list[str], n: int) -> set[tuple]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


# ---------------------------------------------------------------------------
# 倒排索引

class Index:
    """ngram -> {eval_id}，外加每个 eval 条目的 ngram 总数用于算 containment。"""

    def __init__(self, n: int):
        self.n = n
        self.inv: dict[tuple, set[str]] = defaultdict(set)
        self.size: dict[str, int] = {}

    def add(self, eid: str, toks: list[str]) -> None:
        gs = ngrams(toks, self.n)
        if not gs:
            return
        self.size[eid] = len(gs)
        for g in gs:
            self.inv[g].add(eid)

    def best(self, toks: list[str]) -> tuple[float, str | None]:
        """返回 (最大 containment, 命中的 eval_id)。"""
        gs = ngrams(toks, self.n)
        if not gs:
            return 0.0, None
        hits: dict[str, int] = defaultdict(int)
        for g in gs:
            for eid in self.inv.get(g, ()):
                hits[eid] += 1
        if not hits:
            return 0.0, None
        best_c, best_e = 0.0, None
        for eid, c in hits.items():
            denom = min(len(gs), self.size[eid])
            v = c / denom if denom else 0.0
            if v > best_c:
                best_c, best_e = v, eid
        return best_c, best_e


class MultiIndex:
    """按 n 分层的倒排索引。

    每条评测题进入它能支撑的最大 n 层；训练题对每一层都查一遍，取最大
    containment。这样短题不再被静默跳过——漏检是覆盖率缺口，比误报严重得多。
    """

    def __init__(self, ns: list[int]):
        self.layers = {n: Index(n) for n in sorted(ns, reverse=True)}
        self.assigned: dict[str, int] = {}
        self.low_conf: set[str] = set()

    def add(self, eid: str, toks: list[str]) -> int | None:
        for n in self.layers:
            if len(toks) - n + 1 >= MIN_GRAMS:
                self.layers[n].add(eid, toks)
                self.assigned[eid] = n
                return n
        # 兜底：允许不足 MIN_GRAMS，但标为低置信
        for n in self.layers:
            if len(toks) >= n:
                self.layers[n].add(eid, toks)
                self.assigned[eid] = n
                self.low_conf.add(eid)
                return n
        return None

    def best(self, toks: list[str]) -> tuple[float, str | None]:
        bc, be = 0.0, None
        for ix in self.layers.values():
            c, e = ix.best(toks)
            if c > bc:
                bc, be = c, e
        return bc, be


# ---------------------------------------------------------------------------

def main() -> None:
    he = load_dataset("evalplus/humanevalplus", split="test")
    mb = load_dataset("evalplus/mbppplus", split="test")
    kc = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")

    idx_text = MultiIndex(NS_TEXT)
    idx_code = MultiIndex(NS_CODE)

    for r in he:
        eid = str(r["task_id"])
        idx_text.add(eid, norm_text(r["prompt"]))
        idx_code.add(eid, norm_code(r["prompt"] + "\n" + r["canonical_solution"]))
    for r in mb:
        eid = f"Mbpp/{r['task_id']}"
        idx_text.add(eid, norm_text(r["prompt"]))
        idx_code.add(eid, norm_code(r["code"]))

    n_eval = len(he) + len(mb)
    print(f"评测侧建索引：文本 {len(idx_text.assigned)}/{n_eval} 条、"
          f"代码 {len(idx_code.assigned)}/{n_eval} 条")
    for lbl, ix in (("文本", idx_text), ("代码", idx_code)):
        from collections import Counter as _C
        print(f"  {lbl}分层: {dict(_C(ix.assigned.values()))}  "
              f"低置信(n-gram<{MIN_GRAMS}): {len(ix.low_conf)} 条")
    miss = n_eval - min(len(idx_text.assigned), len(idx_code.assigned))
    print(f"  {'✅ 全覆盖' if miss == 0 else f'⚠️ 仍有 {miss} 条漏检'}")
    print(f"训练侧待检 {len(kc)} 条\n")

    rows = []
    for i, r in enumerate(kc):
        ct, et = idx_text.best(norm_text(r["question"]))
        cc, ec = idx_code.best(norm_code(r["solution"]))
        rows.append({
            "i": i,
            "question_id": r.get("question_id"),
            "subset": r.get("subset"),
            "text_c": round(ct, 4), "text_hit": et,
            "code_c": round(cc, 4), "code_hit": ec,
            "max_c": round(max(ct, cc), 4),
        })
        if (i + 1) % 2500 == 0:
            print(f"  ...{i + 1}/{len(kc)}")

    print(f"\n{'=' * 64}\n阈值扫描（任一通道 containment 超过阈值即标记）")
    print(f"{'阈值':>6} | {'问题通道':>8} | {'解法通道':>8} | "
          f"{'任一命中':>8} | {'剩余干净':>8}")
    print("-" * 64)
    summary = {}
    for t in THRESHOLDS:
        nt = sum(1 for r in rows if r["text_c"] >= t)
        nc = sum(1 for r in rows if r["code_c"] >= t)
        na = sum(1 for r in rows if r["max_c"] >= t)
        summary[str(t)] = {"text": nt, "code": nc, "any": na,
                           "clean": len(rows) - na}
        print(f"{t:>6.2f} | {nt:>8d} | {nc:>8d} | {na:>8d} | "
              f"{len(rows) - na:>8d}")

    print(f"\n{'=' * 64}\ncontainment 最高的 8 条（人工抽检用）")
    for r in sorted(rows, key=lambda r: -r["max_c"])[:8]:
        print(f"  max={r['max_c']:.3f} text={r['text_c']:.3f} "
              f"code={r['code_c']:.3f} subset={r['subset']:15s} "
              f"hit={r['code_hit'] or r['text_hit']}")

    with open("contamination_report.json", "w", encoding="utf-8") as f:
        json.dump({"n_text": N_TEXT, "n_code": N_CODE,
                   "summary": summary, "rows": rows}, f, ensure_ascii=False)
    # 用最严的 0.10：供给（约 8.6K）远大于需求（2400），误杀零成本，
    # 而"我们用了最保守的阈值"在报告和面试里都是干净的答案。
    keep = [r["i"] for r in rows if r["max_c"] < 0.10]
    with open("clean_ids.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": 0.10, "clean_index": keep}, f)
    print(f"\n已写出 contamination_report.json 与 clean_ids.json"
          f"（阈值 0.10，保留 {len(keep)} 条）")


if __name__ == "__main__":
    main()
