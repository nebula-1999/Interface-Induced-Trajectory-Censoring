# P2：Qwen2.5-Instruct 尺寸梯子（§5.2 的阴性对照）

## 这个实验在问什么

§5.2 现在只能说：**在 Qwen2.5-Coder 家族内**，更大的 checkpoint 在同一错配接口下
有单调更大的绝对低估（0 → 4 → 21 → 36 → 80，服务端恒为 0）。

审稿人会问：这是「规模现象」还是「Coder 这一支碰巧不行」？

换一个家族答不了这个问题——四家里 Llama 解析得好好的、Mistral 直接 400、
DeepSeek 模板根本不注入，**只有 parser 层的错配才会产生 censoring**。

所以对照要在**同一血统内**做：Qwen2.5-**Instruct** 同尺寸梯子。
两者共用同一份 chat template（Coder 的模板是从 Instruct 继承的，这正是
§5.1 那个假阳性的成因），差别在 Instruct **在 tool-call token 上训练过**。

| | Coder（已有） | Instruct（本实验） |
|---|---|---|
| chat template | 继承自 Instruct | 原生 |
| tool-call token 训练 | ❌ | ✅ |
| 预期 server-parsed | 0，各尺寸 | 非零，且 ≈ emitted |

**若结果如预期**，错配就被锁死在「是否受过该格式训练」这一个变量上，
而不是「某个家族碰巧不行」。这比换家族强，也正是审稿人会要的那个对照。

**若 Instruct 也是 0**，那反而是更强的发现——说明问题出在 hermes parser
与整个 Qwen2.5 系的模板约定之间，范围比论文现在写的更大。两种结果都值得报。

## 跑法

```bash
# 1. 装公钥（你自己跑，要输密码）
ssh-copy-id -i ~/.ssh/autodl.pub -p 57821 root@connect.nma1.seetacloud.com

# 2. 推送载荷 + 起跑（见 deploy_p2.sh）
bash p2/deploy_p2.sh
```

## 设计约束（不要改）

1. **vLLM 必须是 0.27.1。** 本实验测的就是 parser 行为，换版本 hermes 的解析
   逻辑可能就变了，两条梯子直接不可比。
2. **同一题集。** `clean_ids.json` 随行，探针取 `clean[:100]`，与 50 个已有臂一致。
   脚本开头会算 prompt 哈希——**旧机跑 P0 时要同样算一次比对**，
   防的是 KodCode 上游改版导致题集漂移。这是待核验项，不是已核验。
3. **`temperature=0`、`seed=0`、`fc-schema terse`、`tool_choice=auto`** ——
   与 §5.2 的 Coder 臂逐项对齐。任何一项不同，对照就废了。
4. **每条臂先 preflight。** `preflight_toolcall.py` 几秒钟跑完，配置坏了立刻停。
   这个项目最贵的教训就是配置静默失效后烧掉几小时。

## 结果怎么读

```bash
python3 analysis/intent.py            # 与 Coder 臂同一个判据函数
python3 p2/compare_ladders.py         # 两条梯子并排
```

**不要**为 P2 另写一套意图判据。全文所有意图统计出自 `analysis/intent.py`
的同一个函数，P2 若另起炉灶，跨实验的数字就不可比了。
