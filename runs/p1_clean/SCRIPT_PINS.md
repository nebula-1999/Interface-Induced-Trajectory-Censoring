# 这批产物是哪一版脚本跑出来的

`bfcl_run/provenance.txt` 是 `run_p1.sh` 自动生成的，而当时那张 sha256 清单
**漏了 `run_p1.sh` 自己和 `validate_p1.py`**——也就是说它偏偏没钉住产出它的脚本。
（`runs/p1/`（恢复版）的 provenance 里有那两行，是 2026-09-01 恢复时手工补的，
不是脚本产出的。）清单已在事后补全，但**本目录这批产物跑在补全之前**，所以在这里
把对应关系写死：

| 文件 | sha256 | 说明 |
|---|---|---|
| `p1/run_p1.sh` | `ec78a9f068cb97a1a377260e3f860ee9f68a6d622ca5b9654fc05a7ad0941f80` | 产出本目录的那一版 |
| `p1/validate_p1.py` | `d28384dfde6862b82b74a5c5d5eac3ba5b807a21a5a7b01cc747f3366dd9c004` | 与 provenance 中一致 |

其余文件（`bfcl_registration.py`、`toolcall_proxy.py`、`analyze_p1.py`、两个
plugin 文件、manifest）的哈希见 `bfcl_run/provenance.txt`，与仓库中同名文件一致。

**怎么核**：`shasum -a 256 p1/run_p1.sh` 应得到上表第一行。若不符，说明该脚本在
本批产物之后又被改过——查 git 历史，本文件写于 commit 之时。

**与 `runs/p1/` 的区别**：那批是 2026-09-01 21:13 经人工恢复的产物，其
provenance 钉的 `run_p1.sh` 是 `0358c8f3…`（本仓库不再保留该版本，它与
`ec78a9f0…` 的差别只有一处：归档时是否一并清理 `smoke_<臂>` 的 project root）。
`runs/p1/` 保留作对照，不作为论文引用来源。
