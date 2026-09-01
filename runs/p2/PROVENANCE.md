# P2 Instruct 尺寸梯子 —— 产物来源

跑于 2026-09-01，实例 `autodl-p2`（单卡 A800 80G），当晚关机。

## 产出这五条臂的确切代码

| 文件 | sha256（前 16 位） | 说明 |
|---|---|---|
| `probe_react_full.py` | `f8b1559cfda179c9` | **五条臂全部由这一份产出**，见 `p2/from_server_20260901/` |
| `run_p2.sh` | 见 git | 1.5B/7B/14B/32B 与 3B 之间改过（加幂等跳过），**探针未动** |
| `clean_ids.json` | 随行于 `p2/from_server_20260901/` | 题集取 `clean[:100]` |

仓库根目录的 `probe_react_full.py`（`51a0f07e11681`6d4…）**与之不同**：它只改了
两处打印文字和一段 docstring（把「格式完好但 parser 不认」这句无依据的断言换成
中性陈述），判据逻辑一字未动。这份改动**刻意没有部署**，以保证五条臂同源。
要复现本目录数据，用 `p2/from_server_20260901/probe_react_full.py`。

## 配置

`protocol=fc  strength=optional  fc-schema=terse  parser-adapter=cross_family`
`temperature=0  seed=0  max_tokens=2048  n=100`
vLLM 服务端：`--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192`，
不加 `--served-model-name`（模型 id 即路径，与探针 `--model` 传的字符串一致）。

题集 prompt 哈希 `cd69faed3c50f18ef4e51dd77fa34ec9`。
★ 待核验：旧机跑 P0 时应同样算一次比对，防 KodCode 上游改版导致题集漂移。

## 已知的一处不同源

3B 是补跑的（首轮因半截下载的模型被跳过），跑于 22:03，其余四条跑于 19:07–21:51。
探针与服务端配置完全一致，中间只改了 `run_p2.sh` 的臂调度逻辑。
