# P1：在标准 tool-use benchmark 上做五层测量

**给 Codex 的交接说明。** 目标机器：**旧卡 `autodl-code`**（不是 P2 那台）。

## 这个实验在答什么

审稿人一定会问的一句话：

> 「这只是你自建 harness 的毛病，标准 benchmark 上不存在。」

P1 就是答这句。如果 BFCL / tau-bench 在同样的服务端配置下也出现
「模型产出了合格调用、服务端解析为零」，那么受影响的就不是我们的实验，
**是这些 benchmark 报出来的数字本身**。

两种结果都值得报：
- **出现 censoring** → benchmark 的测量效度问题，本文范围扩大
- **不出现** → 说明问题局限于自建 harness，论文范围要相应收窄，
  这一条必须写进 Limitations

## 核心设计：不要改 benchmark 的代码

五层测量需要的「模型产出了但服务端没解析走」那份文本，
**只在 parser 失败时才留在 `content` 里**——解析成功会把它搬进
`tool_calls` 并清空 `content`（本文 §5.2 那些 N/A 就是这么来的）。

所以在 HTTP 层记录就拿到了全部原料：

```
BFCL / tau-bench  →  :8001 记录代理（透明转发）  →  :8000 vLLM
```

benchmark 只需把 `OPENAI_BASE_URL` 指到 `http://127.0.0.1:8001/v1`。
**代理必须逐字节透明**——改任何东西，测的就不是 benchmark 的真实行为了。
脚本里有两道 preflight：一道直连 vLLM，一道经代理，两道都必须过。

## 文件

| | |
|---|---|
| `toolcall_proxy.py` | 透明记录代理，流式/非流式都处理（166 行） |
| `analyze_p1.py` | 五层漏斗，复用 `analysis/intent.py` 的**同一个**判据函数 |
| `run_p1.sh` | 编排：起服务 → 起代理 → 双 preflight → 跑 benchmark → 拆 |

## Codex 需要做的三件事

脚本已经写好，但有两处需要落地——**这两处我没法离线写死，因为要看
benchmark 仓库当时的实际接口**：

### 1. 推送载荷

```bash
ssh autodl-code 'mkdir -p /root/autodl-tmp/p1/{logs,analysis}'
scp p1/toolcall_proxy.py p1/analyze_p1.py p1/run_p1.sh \
    preflight_toolcall.py  autodl-code:/root/autodl-tmp/p1/
scp analysis/intent.py     autodl-code:/root/autodl-tmp/p1/analysis/
ssh autodl-code 'chmod +x /root/autodl-tmp/p1/run_p1.sh'
```

### 2. 装 benchmark，并各写一个 `run_*.sh`

```bash
# BFCL
git clone https://github.com/ShishirPatil/gorilla /root/autodl-tmp/p1/bfcl
# tau-bench
git clone https://github.com/sierra-research/tau-bench /root/autodl-tmp/p1/tau-bench
```

`run_p1.sh` 会以 `bash run_bfcl.sh <标签>` 和 `bash run_tau.sh <标签>` 调用它们，
调用时已设好 `OPENAI_BASE_URL` 与 `OPENAI_API_KEY`。这两个小脚本要 Codex 按
benchmark 当时的 CLI 写，形如：

```bash
# run_bfcl.sh —— 放在 /root/autodl-tmp/p1/bfcl/
#!/usr/bin/env bash
set -e
bfcl generate --model probe --test-category simple,parallel,multiple \
     --num-threads 1 --result-dir "./result_$1"
bfcl evaluate --model probe --result-dir "./result_$1" --score-dir "./score_$1"
```

**要点：**
- 模型名必须是 `probe`（vLLM 的 `--served-model-name` 钉死了它）
- `--num-threads 1`。并发会让代理日志与 benchmark 评分难以按序对齐，
  而且本实验不赶时间
- 先用**最小子集**跑通（BFCL 的 `simple` 一类即可），确认代理日志非空、
  `analyze_p1.py` 出得来表，再放全量。不要一上来跑全量

### 3. 起跑

```bash
ssh autodl-code 'cd /root/autodl-tmp/p1 && source /root/autodl-tmp/env.sh && \
  setsid --fork ./run_p1.sh </dev/null >p1_boot.log 2>&1'
ssh autodl-code 'tail -f /root/autodl-tmp/p1/p1_run.log'
```

## 验收标准

跑完必须满足，否则结果不可用：

1. `p1/logs/*.jsonl` 非空，且 `n_tools_offered > 0` 的记录占多数
   （若为 0，说明 benchmark 根本没走 tools 分支，配置错了）
2. 两道 preflight 都 rc=0（尤其是经代理那道——它证明代理透明）
3. `analyze_p1.py` 的「服务端解析」列非全 0 也非全 100%
   （全 0 说明 parser 配错，全 100% 说明这条臂没有 censoring，是有效的阴性结果）
4. HTTP 错误数为 0；非 0 的臂按本文 §3.3 的规则记为不可用，不参与比较

## 不要做的事

- **不要**为 P1 另写一套意图判据。全文所有意图统计出自
  `analysis/intent.py` 的同一个函数，另起炉灶跨实验就不可比了
- **不要**改代理让它「顺便修一下」解析。代理只记录，不干预
- **不要**并发跑多条臂。单卡 80G，两个模型同时起会 OOM，
  而且日志会交叉
