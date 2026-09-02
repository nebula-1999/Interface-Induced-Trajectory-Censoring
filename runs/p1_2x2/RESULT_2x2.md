# BFCL v4 上的 template × parser 2×2 消融（2026-09-02）

回应的质疑：§5.3 同时替换了 parser 与其配对的 chat template，因此 0→196
无法在两者之间分摊。四格中对角线两格此前已有，本次补上两个非对角格。

同一权重、同一 200 题、同一 seed / 解码 / 执行器 / 评分器，只改服务端配置。

## 首请求解析（每题第一次 HTTP 请求）

| | documented 模板 | dedicated 模板 |
|---|---|---|
| **hermes parser** | ① **0**/200 | ② **0**/200 |
| **dedicated parser** | ③ **0**/200 | ④ **196**/200 |

四格 HTTP 错误均为 0。

## BFCL 官方评分器

| cell | `simple_python` | `multi_turn_base` |
|---|---|---|
| ① documented tpl + hermes | 0.00 | 0.00 |
| ② dedicated tpl + hermes | **0.00** | **0.00** |
| ③ documented tpl + dedicated parser | **0.00** | **0.00** |
| ④ dedicated tpl + dedicated parser | 0.96 | 0.19 |

## 读法

**效应不可分摊，因为它不是可加的。** 单独换模板：0。单独换 parser：0。
两者同时换：0.96 / 0.19。这不是「主要由 parser 驱动」也不是「主要由模板驱动」，
而是**纯交互**——匹配契约的任何一侧而不匹配另一侧，收益精确为零。

这比原先「adapter 级归因」的表述更强，也更贴合本文的核心命题：失效的不是
某个组件，而是**模型—序列化—parser 这份三方契约**。修 parser 不修模板，
或修模板不修 parser，在这个基准上都一分不涨。

## 与原稿措辞的关系

原 §5.3 写「本实验无法在两者之间分摊效应」，现在可以更准确地写：
**已做 2×2，效应在两个主效应上均为零，全部集中在交互项。**

---

## 有效性验证（2026-09-02 补）

「0/0/0/196 太干净，会不会是 flag 根本没生效」——这个质疑是对的，
而且「参数没生效」产生的也是 0。三重验证：

### 1. 参数确实抵达 vLLM

各格启动时的 `non-default args`：

| cell | chat_template | tool_call_parser | plugin |
|---|---|---|---|
| ① | （无 = documented） | hermes | — |
| ② | dedicated | hermes | — |
| ③ | （无 = documented） | qwen2_5_coder | 已加载 |
| ④ | dedicated | qwen2_5_coder | 已加载 |

### 2. 输出的变化模式符合 2×2 预期

比较各格**首请求的返回内容**（200 个共同用例）：

- **③ vs ①**（同模板、只换 parser）：内容相同 **187/200**。模板没变 → 模型输入
  没变 → temperature=0 下输出应几乎全同。13 条差异是已知的多轮 + 8 线程非确定性。
- **② vs ①**（同 parser、只换模板）：内容相同 **0/200**。模板确实改变了模型输入。

若模板参数是摆设，② 应与 ① 全同。实测全不同。

### 3. 机制可直接目视

同一道题（`multi_turn_base_10`），两种模板下模型发出的内容：

```
documented:  ```json
             {"name": "mkdir", "arguments": {"dir_name": "Projects"}}
             ```

dedicated:   <tools>
             {"name": "mkdir", "arguments": {"dir_name": "Projects"}}
             </tools>
```

**载荷一字不差，包装层完全不同。**

### 结论：四个 0 各有各的正确理由

| cell | 模型发出的包装层 | parser 接受的包装层 | 结果 |
|---|---|---|---|
| ① | ` ```json ` | `<tool_call>` | 0（parser 行为正确） |
| ② | `<tools>` | `<tool_call>` | 0（parser 行为正确） |
| ③ | ` ```json ` | `<tools>` | 0（parser 行为正确） |
| ④ | `<tools>` | `<tools>` | **196** |

没有任何组件是坏的。模板决定模型发出哪种信封，parser 决定接受哪种信封，
公开基准上得 0.00 还是 0.96，取决于两者是否**同意**。

这也解释了 §5.2 的观察：documented 模板下 Coder 各尺寸 `no_envelope` 恒为
100/100，**因为那份模板本身不要求任何信封**。
