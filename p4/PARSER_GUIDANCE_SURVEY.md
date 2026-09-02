# 普查：关于 tool-call parser 的指引到底写在哪（2026-09-02，零算力）

## 缘起

预注册 v1 曾写「接口错配不是罕见事故，稀少的反而是模板与 parser 恰好同意」。
该句因证据不足被删——只普查了 chat template，**没有普查各家推荐的 parser**。
本文件补上缺的那一半。

## 方法

对九个家族取 HuggingFace 模型卡（`README.md`），检索是否出现
`--tool-call-parser`，以及是否提到 vLLM。零算力，纯文本检索，可复算。

## 结果

| 家族 | 模型卡提到 `--tool-call-parser` | 模型卡提到 vLLM |
|---|---|---|
| Qwen2.5-Coder | **否** | 是 |
| Qwen2.5-Instruct | **否** | 是 |
| Qwen3 | **否** | 是 |
| Granite 3.1 | **否** | 否 |
| GLM-4 | **否** | 是 |
| Mistral-7B-v0.3 | **否** | 是 |
| DeepSeek-Coder-V2 | **否** | 是 |
| Phi-4 | **否** | 否 |
| Llama-3.1-8B | 不可访问（gated，401） | — |

**八张可访问的模型卡中，零张提及该开关；其中六张明确建议用 vLLM 部署。**

## 可以主张的

**关于 tool-call parser 的指引只存在于服务栈一侧，不在模型一侧。**
模型发布方告诉你用 vLLM，却不告诉你该配哪个 parser；而 parser 的选择决定了
模型的工具调用是否可见（§5.2、§5.3、本文 2×2）。照模型卡操作的人，
**收不到任何"这个选择要紧"的信号**——错配因此不是疏忽，是文档分工的产物。

这与 §5.1 的观察互补：那里说的是"按官方文档配置，四个家族各在一层失败"；
这里说明**为什么**照做也会失败——所谓"官方文档"分散在两方，各自假定对方会写。

## **不能**主张的

1. **不能**据此说"部署中的错配普遍存在"。本普查只看文档，没有测量真实部署。
2. **不能**说各家 parser 与其模板普遍不匹配。vLLM 为约四十个家族提供了专用
   parser；只要按其文档选对，多数家族本就是匹配的。Granite 即一例
   （vLLM 文档推荐 `granite`，我们故意配 hermes 才得到 0）。
3. 模型卡不提该开关，也**不等于**发布方没有别处的说明（如各自的部署文档、
   博客）。本普查的边界就是模型卡。

## 复算

```bash
python - <<'PY'
import urllib.request, re
pat = re.compile(r'--tool[- ]call[- ]parser[= ]+([A-Za-z0-9_.\-]+)')
for repo in ["Qwen/Qwen2.5-Coder-7B-Instruct", "ibm-granite/granite-3.1-8b-instruct", ...]:
    md = urllib.request.urlopen(f"https://huggingface.co/{repo}/raw/main/README.md").read().decode()
    print(repo, sorted(set(pat.findall(md))), "vllm" in md.lower())
PY
```
