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
