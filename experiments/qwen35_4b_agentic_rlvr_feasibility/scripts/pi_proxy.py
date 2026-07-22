"""Logging reverse proxy between pi-coding-agent and vLLM, to capture pi's EXACT prompts.

Why a proxy instead of parsing pi's --mode json events: those events contain the assistant messages
and tool results, but NOT pi's system prompt or its tool JSON schemas -- both of which live only in
the HTTP request. SFT rows built without them would train the model on a context it never sees at
deploy time, which is the same class of mistake as the harness mismatch this experiment already hit
(our harness measured 0.486 where pi measures 0.810). The proxy records the request verbatim, so
harvested rows are exactly what pi puts in front of the model.

    pi  --->  proxy :8421  --->  vLLM :8420
              (logs request + reassembled completion)

Streaming is preserved: chunks are relayed to pi as they arrive and simultaneously accumulated, so
pi behaves identically whether or not the proxy is in the path.

Each log line is one completed exchange:
    {"t": <unix>, "messages": [...], "tools": [...], "completion": {"content": str,
     "reasoning_content": str, "tool_calls": [...]}}
Conversations are reconstructed downstream by prefix-chaining (a turn's `messages` extends the
previous turn's), so episodes can run concurrently without needing per-episode tagging.
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

LOCK = threading.Lock()


def _accumulate(chunks):
    """Reassemble an OpenAI SSE stream into a single completion dict."""
    content, reasoning, calls = [], [], {}
    finish = None
    for raw in chunks:
        for line in raw.split("\n"):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                d = json.loads(payload)
            except Exception:
                continue
            for ch in d.get("choices") or []:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for tc in delta.get("tool_calls") or []:
                    i = tc.get("index", 0)
                    slot = calls.setdefault(i, {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
    return {"content": "".join(content), "reasoning_content": "".join(reasoning),
            "finish_reason": finish,
            "tool_calls": [calls[k] for k in sorted(calls)]}


def make_handler(upstream, logpath):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _relay(self, method):
            body = b""
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n:
                body = self.rfile.read(n)
            req = Request(upstream + self.path, data=body if n else None, method=method)
            for k, v in self.headers.items():
                if k.lower() not in ("host", "content-length", "connection"):
                    req.add_header(k, v)
            try:
                resp = urlopen(req, timeout=1200)
            except Exception as exc:
                msg = json.dumps({"error": str(exc)}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                return

            ctype = resp.headers.get("Content-Type", "application/json")
            streaming = "event-stream" in ctype
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype)
            if streaming:
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            chunks, raw = [], b""
            if streaming:
                while True:
                    part = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
                    if not part:
                        break
                    chunks.append(part.decode("utf-8", "replace"))
                    self.wfile.write(hex(len(part))[2:].encode() + b"\r\n" + part + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                raw = resp.read()
                self.wfile.write(raw)

            if "chat/completions" not in self.path:
                return
            try:
                reqd = json.loads(body or b"{}")
            except Exception:
                return
            if streaming:
                comp = _accumulate(chunks)
            else:
                try:
                    choice = json.loads(raw)["choices"][0]
                    m = choice.get("message", {})
                    comp = {"content": m.get("content") or "",
                            "reasoning_content": m.get("reasoning_content") or "",
                            "finish_reason": choice.get("finish_reason"),
                            "tool_calls": [{"name": (c.get("function") or {}).get("name", ""),
                                            "arguments": (c.get("function") or {}).get("arguments", "")}
                                           for c in (m.get("tool_calls") or [])]}
                except Exception:
                    return
            # Per-call size metrics for the token-cap mechanism readout. NOTE: the server runs
            # WITHOUT --reasoning-parser, so <think> text arrives as ordinary CONTENT deltas --
            # `content` length is the runaway-generation signal; `reasoning_content` will be empty.
            tool_arg_chars = sum(len(c.get("arguments") or "") for c in comp.get("tool_calls") or [])
            rec = {"t": time.time(), "messages": reqd.get("messages") or [],
                   "tools": reqd.get("tools") or [], "completion": comp,
                   "max_completion_tokens": reqd.get("max_completion_tokens") or reqd.get("max_tokens"),
                   "finish_reason": comp.get("finish_reason"),
                   "n_content_chars": len(comp.get("content") or ""),
                   "n_reasoning_chars": len(comp.get("reasoning_content") or ""),
                   "n_tool_arg_chars": tool_arg_chars}
            with LOCK:
                with open(logpath, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")

        def do_POST(self):
            self._relay("POST")

        def do_GET(self):
            self._relay("GET")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8421)
    ap.add_argument("--upstream", default="http://127.0.0.1:8420")
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    print(f"pi proxy :{a.port} -> {a.upstream} | logging to {a.log}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.upstream, a.log)).serve_forever()


if __name__ == "__main__":
    main()
