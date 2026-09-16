# P3 正式重跑前验收：未通过

核查日期：2026-09-15。基准：本地 HEAD `9cfaaad` 与当前文件。

## 结论

**当前不具备启动 150-step 正式配对训练的条件。**

已经完成本地源码检查、历史数据完整性检查、纯函数测试和异常路径模拟；没有启动 GPU 训练，没有部署修改，没有执行 launcher 中的目录清理，也没有改动旧实验产物。

服务器方面，已有 SSH 别名 `autodl-code` 的连接最终返回 `Connection refused`。这不能区分关机、端口变化或 SSH 服务不可用。显卡、磁盘、服务器实际部署代码、真实训练和 checkpoint 恢复均未验收。

验收不是只检查 `save_freq` 字面值：必须确认评测真实发生、使用正确权重、记录身份唯一、保存完整且能恢复。当前多项已在本地确认失败。

## 1. 验收表

| 项目 | 结论 | 证据与影响 |
|---|---|---|
| launcher shell 语法 | 通过 | 两个 runner 均通过 `bash -n`；不代表行为安全 |
| 定期评测配置 | 部分符合 | arm wrapper 默认 EVAL_FREQ=30、val_before_train=True；未阻止传入负频率，真实触发未验证 |
| checkpoint 默认保存 | **失败** | `run_p3_arm.sh:42` 仍默认 SAVE_FREQ=-1 |
| 恢复模式 | **失败** | `run_p3_probe.sh:113` 仍为 trainer.resume_mode=disable |
| 重跑保护旧数据 | **失败** | `run_p3_arm.sh:68–75` 固定同臂目录并递归清理 events/ckpt；probe 也会清理事件文件 |
| step 0 唯一性 | **失败** | 旧文件 2035 行，只有996个唯一(step,channel,task_id)键，全部键重复 |
| 评测失败后停止 | **失败** | 首次成功后，下一次评测异常被捕获后允许返回；模拟已复现 |
| 请求错误验收 | **失败** | 评测报告 request_failures=1 仍被记为成功评测；模拟已复现 |
| 评测实际使用更新后 LoRA | **未验证** | 请求只指定模型路径，无可审计的权重/adapter版本凭据；不能凭端点地址相同推断 |
| 自主工具调用评测 | 范围不匹配 | 当前 CodeEvalHook 使用普通消息生成，由评测器自动测试并反馈；可测条件修复，不能冒充原生 FC 调用 |
| 离线 verl parser 重放 | **失败** | 已知合法信封在辅助函数中得到 vLLM=True、verl=False |
| 原生工具 parser 基础提取 | 有限通过 | 直接调用现有4个纯函数测试通过；未测试真实verl注册与Ray进程 |
| 分解评测基本逻辑 | 有限通过 | 直接调用现有8个测试函数通过，包括三项mock batch/monkeypatch |
| verifier 本地纯逻辑 | 有限通过 | 摘要抗普通文本误匹配、skipped/xfail计分母两项检查通过 |
| 真实 pytest 执行环境 | **未验证** | 本地检查的两个Python均没有pytest；没有据此推断服务器依赖状况 |
| broken 原文哈希 | 通过 | 19,202个extract事件，哈希不一致0 |
| 轨迹身份与训练step关联 | 不完整 | 上述extract事件均无task_id、step字段，无法直接按题/step作学习分析 |
| checkpoint 完整加载/恢复 | **未验证** | 服务器不可连接；未重新确认15GB历史目录是否完整 |

## 2. 保存与目录是明确阻断项

当前有效路径：

```text
run_p3_arm.sh
  SAVE_FREQ defaults to -1
  RUN_ROOT=/root/p3_formal/$ARM
  deletes events and checkpoints in that arm directory
  calls run_p3_probe.sh
    trainer.resume_mode=disable
```

`DRY=1` 只出现在使用注释里，没有对应的实际分支。本次没有以所谓dry-run方式运行launcher，避免触发真实删除或训练。

需要修改后再验收：

- 正式运行必须显式保存，拒绝非正的保存/评测频率；打印并落盘最终配置。
- 每次实验使用独立run_id。重试/恢复必须明确选择已有checkpoint，不清理它。
- 输出和checkpoint路径明确落到预期数据盘；不使用旧磁盘余量注释作为事实。
- 验证恢复后global step、训练状态及必要随机/数据状态衔接；不能仅检查文件存在。

## 3. 旧 step 0 分母污染已经证实

检查文件：

`p3/results/eval_archive/code-eval-p3-repaired-ABORTED-2008/step_00000.jsonl`

| 通道 | 记录行数 | 不同task_id |
|---|---:|---:|
| multi | 1108 | 542 |
| repair | 927 | 454 |
| 合计 | 2035 | 按channel区分后996个键 |

以 `(step, channel, task_id)` 为键：

- 996个键全部重复。
- 953个键出现2次，43个键出现3次。
- 相对每键一条，多出1039行。
- 没有run_id字段；重复究竟来自哪些历史运行/重试，不能仅从这些键确定。

因此，`20/1108` 不能作为已验证的独立题目基线，相关置信区间及“≥29/1108就判上涨”的规则不能继续使用。

**也不能直接把分母改成542。**需要先确认每次完整评测的记录边界和归属；随意取每题第一条/最后一条会引入另一个未经验证的选择。

根因链条可在代码看到：`CODE_EVAL_OUT` 按arm但不按run区分，`eval_decompose.dump` 使用append，记录没有run/trial身份，读取端可按task_id覆盖重复行。

## 4. 评测异常可能再次留下“训练完成但测量缺失”

对 `code_patch.install` 的函数定义做了独立内存模拟，不导入或运行真实verl、不连接模型：

```text
第一次评测成功 → 第二次人为抛出评测异常
结果：LATER_EVAL_EXCEPTION_SWALLOWED = True

评测返回 code/request_failures=1
结果：仍设置 _code_eval_ok_once=True
```

目前代码只把“从未成功过时的异常”作为致命错误。后续异常会打印后继续；请求失败还可能由ChatPolicy转换为空输出，再被当作任务失败。

需要制定正式评测有效性门控：任何必需评测缺失、请求故障或样本集合不一致，都不能得到“完整学习对照通过”的标记。可以保存已有训练进度，但不能静默将受污染的评测纳入曲线。

## 5. 解析辅助指标的错误已复现

检查对象：`p3/rollout_probe.py::_accepts`。

已知合规的教学输入：带闭合tool_call信封、run_tests名称、合法JSON参数的代码调用。

```text
VLLM_RE replay: True
VERL_RE replay: False
```

原因：VLLM正则有两个捕获组，findall返回tuple；VERL正则只有一个捕获组，返回str。辅助函数统一取`m[0]`，对后者取到了首字符，而不是整个JSON。

这是离线“would accept”判据错误，不是证明真实verl parser拒绝了该输入。实际accepted日志与这个辅助指标要分开。

另需独立校验tight候选的JSON/schema/工具名。不能因为regex命中，就宣布每条都语义完整。

## 6. 已通过的有限检查

系统Python与桌面捆绑Python均没有pytest。本次未安装新依赖，使用标准库直接执行现有测试函数：

- `test_qwen_tools_parser.py`：4项通过。
- `test_eval_decompose.py`：8项通过；monkeypatch参数由可恢复的`unittest.mock.patch.object`适配，其他函数直接调用。
- `sandbox._parse` 普通伪造文本场景和`Result`分母检查：2项通过。
- 两个runner的shell语法检查通过。

这些测试没有执行生成代码，没有测试Linux资源限制、真实pytest子进程、Ray分发、真实模型生成、权重同步或恢复。不能用“14项通过”替代GPU端到端验收。

对broken正式原文另核对：

```text
extract events                 19202
accepted calls                     7
events with accepted calls          7
accepted calls in tight events      7
text SHA256 mismatches              0
events with task_id                 0
events with step                    0
```

7次接受确实都发生于tight标记事件，但tight仍是启发式标签，不因此成为经过独立验证的合法调用总体。

## 7. 修复后真实验收的最小流程

在新的独立验收目录下执行，不复用历史broken/repaired目录：

1. 固定小评测manifest，确认step0实际输出，并保存唯一身份和配置。
2. 一个完整训练update后，确认step1评测读到当前权重/LoRA；不能只以生成文本不同为凭据。
3. 保存完整checkpoint，等待完成，校验文件、可加载性和实际占用。
4. 从该checkpoint恢复，继续到step2；确认不是重新从step0训练，且新记录不重复旧记录。
5. 确认工具执行结果真正进入下一轮输入；给执行/observation/生成事件配上可联结身份。
6. 两条臂都走相同的评测/保存/恢复验收。将“条件反馈修复评测”与“自主工具交互评测”明确区分。
7. 用正式batch、rollout数和长度配置完成至少一个完整step，检查峰值资源；小规模smoke成功不保证正式形状不OOM。
8. 只有这些证据齐全，才放行150步。历史15GB checkpoint和旧GPU时估计均不能替代这次运行证据。

## 8. 当前下一步

先修复已确认的本地阻断项，再做本地回归；服务器可连接后进行上述最小真实验收。当前报告是**不通过报告**，不是部署或正式训练许可。

## 9. 同日修复记录（不覆盖上面的失败历史）

已修复：正式配置保存/评测默认30及正值检查、真实dry-run、唯一RUN_ID及禁止覆盖、
两臂训练/评测内容manifest、step非追加落盘和完整性门控、后续评测异常不再吞掉、
请求错误致命处理、repair缺失拒绝、正则单捕获组错误、工具标记按run隔离。
新事件附run_id/arm/diagnostic_version；task/step及observation关联尚未完成。

新增标准库回归入口 `python3 -m unittest test_p3_acceptance -v`。
本地复测：24项新回归通过；另外12项原有parser/分解评测测试直接调用通过，
2项verifier纯逻辑检查通过。两份shell语法与`git diff --check`通过。
未安装pytest，也没有把这些结果当作真实pytest/verl/GPU测试通过。
具体行为和未完成实机验收见 `RUNBOOK_20260915.md`。
正式150步与显式恢复当前均保留阻断，避免在权重同步和恢复未验收时又烧一整轮。
历史评测文件、原文与论文数字没有修改。
