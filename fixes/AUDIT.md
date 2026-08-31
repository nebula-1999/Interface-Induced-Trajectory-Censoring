# 2026-08-29/30 跨家族实验复盘

结论先行：**昨天报出的两张表都不能直接用**。一处是统计口径错误，一处是实验配置
系统性偏袒 ReAct，还有一处是我引用了脚本自己判定为不可用的数据。

数据位置：`code_agent/runs/xfam2/`（v2，问题数据，保留作对照）。
修复补丁：`code_agent/fixes/fix_audit.py`、重跑队列 `code_agent/fixes/run_v3.sh`。

---

## 洞 1（最严重）26 条空代码撑起了全部显著性

Llama-3.1-8B 配对 McNemar：

| | n | ReAct | FC | b/c | p |
|---|---|---|---|---|---|
| 全部题目 | 100 | 81% | 50% | 38/7 | 3.1e-06 |
| 剔除 FC 首轮空代码的 26 条 | 74 | 81% | **68%** | 17/7 | **0.064** |

剔掉就不显著了。而这 26 条**无法归因**：旧 `gen_fc` 里

```python
except Exception:
    code = ""
```

把「JSON 解析失败 / key 名不对 / 模型真给空」压成同一个空串，且未保存原始
arguments。**在归因清楚之前，p=3.1e-06 不可写入任何材料。**

## 洞 2 max_tokens 系统性偏袒 ReAct

`gen()` 与 `gen_fc()` 都用 `max_tokens=1024`，看似公平，实则不是：FC 把代码放进
JSON 字符串，换行全部转义成 `\n`，**同一份代码在 FC 下更费 token**。共用上限时 FC
更容易被截断，截断的 tool_call 是残缺 JSON → 解析失败 → 空代码 → 计为失败。

这正好是洞 1 那 26 条最可能的成因。而 `finish_reason` 被丢弃（只取
`choices[0]["message"]`），导致该假设**用现有数据无法证伪**。

修复：记录 `finish_reason`，上限抬到 2048。**因此所有跨协议比较都必须重跑**，
新旧数据不可混用。

> **【2026-08-31 更正】本条修复当时并未生效，方向也判断反了。**
>
> `fix_audit.py` 只替换了 `gen_fc()` 里的字面量 `1024`，而 ReAct 侧的 `gen()`
> 是**函数默认参数** `max_tokens=1024`，未被匹配；落盘的 provenance 却统一写
> `MAX_TOKENS`，于是 **ReAct 实跑 1024、FC 实跑 2048，而两者都记录成 2048**。
> 也就是说，真实情形与本条最初的判断**相反**——获得更长生成预算的是 FC，不是 ReAct。
>
> 2026-08-31 11:20 才真正修复（`gen` 默认改为 `MAX_TOKENS`）。随后以真实 2048
> 重跑三个主 ReAct 臂（v14），结果：Llama 80→**80**、Qwen 74→**74**、
> Mistral 32→**33**。**上限不对称的实证影响为零**——这些编程题的 ReAct 响应
> 中位长度约 1000 token，1024 本就基本够用。
>
> 受影响的 7 个臂及其替代数据见 `runs/final/ERRATA.md`。
>
> 教训与本文件第 3、4 条同源：**声称修好而未验证生效，比不修更危险**——
> 它同时污染了数据和对数据的信任。此后所有修复均以外部验证器核对，
> 不采信"补丁已应用"这一步。

## 洞 3 比率的分子是轮数、分母是题数

`parse_modes[parse_mode] += 1` 每轮累加，打印时却除以题数 `n`：

| 模型 | 脚本报「直接给代码率」 | 逐题实际 |
|---|---|---|
| Mistral-7B react | 67% | **54%** |
| DS-6.7B react | 59% | 59%（恰好单轮，未暴露） |

多轮模型被系统性抬高，理论上可超过 100%。「无法解析率」同源。
`L1 严格 Action 发起率` 只在 `t==1` 累加，**不受影响**。

## 洞 4 我引用了脚本自判不可用的数据

`xfam2_run.log:68` —— Mistral-7B FC 臂：

```
请求错误: 3   ⚠️ 非零，本组结果不可用
```

脚本正确地打了警告，但仍返回 0，我照样把它的 L1=2%、最终 28% 填进了对比表。
修复：`n_err>0` 时以退出码 2 结束，让调用方无法静默忽略。

## 洞 5 Qwen 与其他家族的判定口径不同

`parser_adapter()`：Qwen → `legacy`（严格 Action/Final Answer），其他家族 →
`cross_family`（允许直接 fenced code 作为有效作答）。

于是**同一个行为在 Qwen 上记为失败、在 Llama 上记为成功**。L1 不受影响（在
adapter 分支之前算），但 **L3 首轮/最终通过率跨家族不可比**。已排入队列的
`qwenfc2.sh` 没传 `--parser-adapter`，会直接把这个偏差带进判决实验。

修复：`run_v3.sh` 全部显式 `--parser-adapter cross_family`。

## 洞 6 轨迹缺 provenance

`rec` 只存 `protocol` / `strength`，不存 `adapter`、`sys_file`、`max_tokens`。
消融的 B 臂（无尾句）与 C 臂（含尾句）**除文件名外无法区分**。已修。

## 洞 7 取样不是随机

`idxs = clean[:a.n]` —— 取 7669 条里的前 100 条。跨臂配对因此成立（这点是好的），
但样本对题库不具代表性。**暂不改动**：改了会切断与全部历史数据的可比性。
定案后另做一次固定随机种子的复制实验。

## 洞 8 运维

- DS-1.3B 本地只有 93/100：关机守护「满 100 行立即杀队列关机」，同步循环 45s
  一轮，卡在最后一次同步之后。完整文件在数据盘上，开机 rsync 即可补。
- 关机守护应先 `sync` 并留 60s 缓冲再断电。

---

## 下次开机执行顺序

```bash
scp code_agent/fixes/{fix_audit.py,run_v3.sh} autodl-code:/root/autodl-tmp/code-agent/
ssh autodl-code 'cd /root/autodl-tmp/code-agent && python fix_audit.py'   # 必须看输出确认锚点全中
ssh autodl-code 'cd /root/autodl-tmp/code-agent && setsid --fork ./run_v3.sh </dev/null >v3_run.log 2>&1'
```

`fix_audit.py` 是对**服务器上已打过两次补丁的版本**做的，本地这份 `probe_react_full.py`
是打补丁之前的快照，因此**补丁的锚点未能离线验证**。执行后必须逐条核对「已应用/跳过」
输出，任何一条锚点未命中都要先查清再跑。
