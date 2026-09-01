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

2026-09-02 又在**本机**独立复算了第三次，两台 GPU 机都已关停：

```
旧卡 autodl-code   (2026-09-01) = cd69faed3c50f18ef4e51dd77fa34ec9
新卡 autodl-p2     (2026-09-01) = cd69faed3c50f18ef4e51dd77fa34ec9
本机 macOS         (2026-09-02) = cd69faed3c50f18ef4e51dd77fa34ec9   ← 三处一致
```

同时把**上游数据集的 revision 也钉住了**，这比哈希更早暴露漂移：

```
KodCode/KodCode-Light-RL-10K  x-repo-commit = dcf78a8bbba9a613b596ce993c4921a38687dfcc
```

这个 commit 与 P2 跑批时缓存里用的那一版逐字相同（见 `runs/p2/p2_run.log` 里
`Found the latest cached dataset configuration ... /dcf78a8bbba9a613b596ce993c4921a38687dfcc`）。

**这是已核验项，不是待核验项。**（探针每次运行仍会打印一行「★ 待核验」，那是
脚本里写死的提醒文案，不代表本项未完成——读日志时别被它误导。）

复算不再需要 GPU 机器，也不再有"窗口"：

```bash
# 机器上（有 datasets 库时）
python p2/hash_items.py clean_ids.json

# 任何一台机器，只要能连 HF：直接读裸 parquet，不依赖 datasets 的行序
curl -sL -o kodcode.parquet \
  https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K/resolve/main/data/train-00000-of-00001.parquet
python - <<'EOF'
import hashlib, json, pyarrow.parquet as pq
q = pq.read_table("kodcode.parquet").column("question").to_pylist()
h = hashlib.sha256()
for i in json.load(open("clean_ids.json"))["clean_index"][:100]:
    h.update((q[i] or "").encode())
print(h.hexdigest()[:32])
EOF
```

两条路径给出同一个值，说明该哈希不依赖 `datasets` 的加载行为，只依赖 parquet 的行序。
