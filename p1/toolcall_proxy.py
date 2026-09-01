#!/usr/bin/env python3
"""P1：在 benchmark harness 与 vLLM 之间做透明记录代理。

为什么是代理而不是改 harness：五层测量需要的「模型产出了但服务端没解析出来」
那份文本，只在 parser **失败**时留在 `content` 里——成功解析会把它搬进
`tool_calls` 并清空 `content`（本文 §5.2 那些 N/A 就是这么来的）。
因此只要在 HTTP 层完整记下请求与响应，五层就齐了，不需要 fork BFCL 或
tau-bench 的任何代码：把它们的 base_url 指到本代理即可。

代理必须是**透明**的：原样转发、原样返回、不改任何字节。否则测的就不是
它们的真实行为了。

用法:
    python p1/toolcall_proxy.py --listen 8001 --upstream http://127.0.0.1:8000 \
        --log p1/logs/bfcl_qwen7b.jsonl --tag bfcl/Qwen-7B
然后让 benchmark 连 http://127.0.0.1:8001/v1
"""
import argparse, json, os, time, uuid
from aiohttp import web, ClientSession, ClientTimeout

CHAT = "/v1/chat/completions"


def _merge_stream(chunks):
    """把 SSE 增量重组成一个响应体，字段与非流式对齐。"""
    content, calls, finish = [], {}, None
    for c in chunks:
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                slot = calls.setdefault(i, {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return {"content": "".join(content),
            "tool_calls": [calls[k] for k in sorted(calls)],
            "finish_reason": finish}


def _flatten(resp):
    """非流式响应 → 与 _merge_stream 同构的字典。"""
    ch = (resp.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return {"content": msg.get("content") or "",
            "tool_calls": [{"name": (t.get("function") or {}).get("name", ""),
                            "arguments": (t.get("function") or {}).get("arguments", "")}
                           for t in (msg.get("tool_calls") or [])],
            "finish_reason": ch.get("finish_reason")}


async def handler(request: web.Request):
    app = request.app
    body = await request.read()
    url = app["upstream"] + request.path_qs
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    t0 = time.time()

    async with ClientSession(timeout=ClientTimeout(total=app["timeout"])) as sess:
        async with sess.request(request.method, url, data=body, headers=headers) as up:
            ct = up.headers.get("Content-Type", "")
            streaming = "text/event-stream" in ct

            if not streaming:
                raw = await up.read()
                resp = web.Response(body=raw, status=up.status,
                                    headers={k: v for k, v in up.headers.items()
                                             if k.lower() not in ("content-encoding",
                                                                  "content-length",
                                                                  "transfer-encoding")})
                if request.path == CHAT:
                    try:
                        _log(app, body, _flatten(json.loads(raw)), up.status, t0,
                             request_headers=request.headers)
                    except Exception as e:
                        _log(app, body, {"_parse_error": repr(e),
                                         "_raw": raw[:2000].decode("utf-8", "replace")},
                             up.status, t0, request_headers=request.headers)
                return resp

            # 流式：边转发边收集，绝不缓冲整包（否则改变了 harness 观察到的时序）
            out = web.StreamResponse(status=up.status,
                                     headers={"Content-Type": ct,
                                              "Cache-Control": "no-cache"})
            await out.prepare(request)
            chunks, buf = [], b""
            async for part in up.content.iter_any():
                await out.write(part)
                buf += part
                while b"\n\n" in buf:
                    ev, buf = buf.split(b"\n\n", 1)
                    for line in ev.split(b"\n"):
                        if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                            try:
                                chunks.append(json.loads(line[6:]))
                            except Exception:
                                pass
            await out.write_eof()
            if request.path == CHAT:
                _log(app, body, _merge_stream(chunks), up.status, t0, streamed=True,
                     request_headers=request.headers)
            return out


def _log(app, req_body, flat, status, t0, streamed=False, request_headers=None):
    try:
        req = json.loads(req_body)
    except Exception:
        req = {}
    tools = req.get("tools") or []
    rec = {
        "id": uuid.uuid4().hex[:12],
        "ts": round(time.time(), 3),
        "latency_s": round(time.time() - t0, 3),
        "tag": app["tag"],
        # The custom BFCL handler adds these two headers.  They make aggregate HTTP
        # observations joinable with BFCL's per-case execution and score files even
        # when the benchmark issues concurrent requests.
        "case_id": (request_headers or {}).get("X-P1-Case-ID"),
        "request_index": _as_int((request_headers or {}).get("X-P1-Request-Index")),
        "status": status,
        "streamed": streamed,
        "model": req.get("model"),
        "tool_choice": req.get("tool_choice"),
        "n_tools_offered": len(tools),
        "tool_names_offered": [(t.get("function") or {}).get("name") for t in tools][:20],
        "messages_n": len(req.get("messages") or []),
        # 五层测量的原料：解析出来的调用，以及**没被解析走**的正文
        "parsed_tool_calls": flat.get("tool_calls", []),
        "content": flat.get("content", ""),
        "finish_reason": flat.get("finish_reason"),
    }
    if "_parse_error" in flat:
        rec["_parse_error"] = flat["_parse_error"]
        rec["_raw"] = flat.get("_raw")
    with open(app["logpath"], "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    app["n"] += 1
    if app["n"] % 50 == 0:
        print(f"  [proxy] {app['n']} 次调用已记录 → {app['logpath']}", flush=True)


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8001)
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--log", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app["upstream"] = a.upstream.rstrip("/")
    app["logpath"] = a.log
    app["tag"] = a.tag
    app["timeout"] = a.timeout
    app["n"] = 0
    app.router.add_route("*", "/{tail:.*}", handler)
    print(f"记录代理: :{a.listen}  →  {app['upstream']}")
    print(f"日志: {a.log}   标签: {a.tag or '(无)'}")
    print(f"把 benchmark 的 base_url 指到 http://127.0.0.1:{a.listen}/v1")
    web.run_app(app, port=a.listen, print=None)


if __name__ == "__main__":
    main()
