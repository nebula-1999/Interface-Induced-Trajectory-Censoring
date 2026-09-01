# 题集一致性核验

P2 在**另一台机器**上跑，而探针是现场 `load_dataset("KodCode/...")` 加
`clean_ids.json[:100]` 取题的。若 KodCode 上游改版，两台机器取到的 100 题
就会不同，"50 个臂共用一个题集"这条随即失效，P2 与 §5.2 也不再可比。

因此在两台机器上各算一次 `clean[:100]` 的 prompt 哈希：

```
旧卡 autodl-code   clean[:100] prompt sha256 = cd69faed3c50f18ef4e51dd77fa34ec9
新卡 autodl-p2     clean[:100] prompt sha256 = cd69faed3c50f18ef4e51dd77fa34ec9
                                              ^ 一致（2026-09-01 核验）
```

复算：

```bash
python p2/hash_items.py clean_ids.json
```

**这是已核验项，不是待核验项。** 核验窗口很窄——两台机器必须同时开着；
旧卡关机后就没法再算了。
