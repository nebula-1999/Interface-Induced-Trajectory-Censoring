4. Positive control: the pipeline works

Before attributing a zero to a model we verify the measuring instrument. On the *same live
server*, same model, same tools, same system prompt, changing only `tool_choice`:

```
tool_choice = auto      → finish_reason = stop        tool_calls = 0
                          content: "好的，下面是一个实现 count_distinct 的示例代码：```python…"

tool_choice = required  → finish_reason = tool_calls  tool_calls = 1
                          name = run_tests
                          arguments = {"code": "def count_distinct(nums):\n    return len(set(nums))"}
```

Under `required`, `hermes` parses correctly — server flag, parser and chat template are all
functional. The zero under `auto` is therefore model behaviour, not misconfiguration.

**Bound on this control.** `required` uses constrained decoding, so it establishes only
that *the pipeline is capable*; it does not establish that the model spontaneously knows the
`<tool_call>` format. The two must be stated separately.

---
