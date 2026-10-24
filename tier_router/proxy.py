"""A local proxy between Claude Code and the Anthropic API that picks the
model for each new prompt with tev1 (classify.py).

Claude Code is pointed at it with ANTHROPIC_BASE_URL. Requests and the
credentials on them are passed to the API unchanged, except for the `model`
of the main conversation's requests:

- A request that starts a turn (its last message is text the user typed)
  is classified; the turn's tool-result follow-ups reuse that choice, so a
  turn never changes model halfway and its prompt cache holds.
- A choice close to a threshold keeps the conversation's current model:
  switching model starts the prompt cache over.
- Claude Code's own background calls (no tools, or already Haiku) and
  anything that is not a messages call go through untouched.
- If tev1 is unreachable, the request keeps the model Claude Code asked for.
  If the API refuses a request we re-modelled, it is sent once more as it
  came.

  python3 -m tier_router.proxy [--port 8787]
"""

import argparse
import hashlib
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import classify

UPSTREAM = urllib.parse.urlsplit(os.environ.get("TIER_ROUTER_UPSTREAM", "https://api.anthropic.com"))
MODELS = {
    "haiku": os.environ.get("TIER_ROUTER_HAIKU", "claude-haiku-5-5"),
    "sonnet": os.environ.get("TIER_ROUTER_SONNET", "claude-sonnet-5-5"),
    "opus": os.environ.get("TIER_ROUTER_OPUS", "claude-opus-5-5"),
}
# How far past a threshold a score must be to switch a conversation's model.
HYSTERESIS = float(os.environ.get("TIER_ROUTER_HYSTERESIS", "0.03"))
LOG = Path(os.environ.get("TIER_ROUTER_LOG", Path.home() / ".cache/tier-router/decisions.jsonl"))
# Credential errors are the same whichever model the request names, so a retry
# with the model Claude Code asked for can only repeat them.
NO_RETRY = {401, 403}
HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding",
               "accept-encoding", "proxy-connection", "upgrade", "te", "trailer"}

_conversations = {}  # conversation id -> {"tier", "score", "seen"}
_lock = threading.Lock()
_weights = classify.load_weights()


def _text(content):
    """The user-typed text of a message, without Claude Code's reminders."""
    if isinstance(content, str):
        parts = [content]
    else:
        parts = [b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text"]
    text = "\n".join(parts)
    while "<system-reminder>" in text:
        a = text.index("<system-reminder>")
        b = text.find("</system-reminder>", a)
        text = text[:a] + (text[b + len("</system-reminder>"):] if b >= 0 else "")
    return text.strip()


def _conversation_id(messages):
    first = next((m for m in messages if m.get("role") == "user"), None)
    seed = json.dumps(first.get("content") if first else "", sort_keys=True)[:4000]
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def _turn(messages):
    """(prompt, context, starts_turn) for a main-loop request."""
    # Claude Code puts a `system` message (the environment) after the
    # prompt; only the conversation's own messages say whose turn it is.
    messages = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not messages or messages[-1].get("role") != "user":
        return "", "", False
    last = messages[-1]
    prompt = _text(last.get("content"))
    has_tool_result = isinstance(last.get("content"), list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in last["content"])
    context = ""
    for m in reversed(messages[:-1]):
        if m.get("role") == "assistant":
            context = _text(m.get("content"))
            if context:
                break
    if prompt and not has_tool_result:
        return prompt, context, True
    # A tool-result follow-up: the turn's prompt is the last typed text.
    for m in reversed(messages[:-1]):
        if m.get("role") == "user":
            t = _text(m.get("content"))
            if t and not (isinstance(m.get("content"), list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])):
                return t, "", False
    return "", "", False


def _log(entry):
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def route(body):
    """The model to send `body` to, and a log entry saying why."""
    asked = body.get("model", "")
    entry = {"t": round(time.time(), 3), "asked": asked}
    if not body.get("tools") or "haiku" in asked:
        entry["why"] = "background"
        return asked, entry
    messages = body.get("messages") or []
    conv = _conversation_id(messages)
    prompt, context, starts = _turn(messages)
    entry.update(conv=conv, starts_turn=starts, prompt=prompt[:120])
    with _lock:
        known = _conversations.get(conv)
    if known and not starts:
        entry.update(why="same turn", tier=known["tier"])
        return MODELS[known["tier"]], entry
    if not prompt:
        entry["why"] = "no prompt found"
        return asked, entry
    started = time.time()
    try:
        tier, score, _ = classify.classify(prompt, context, _weights)
    except Exception as e:  # tev1 down, slow, or answering oddly
        entry.update(why=f"tev1 failed: {e}", ms=int((time.time() - started) * 1000))
        return asked, entry
    entry.update(ms=int((time.time() - started) * 1000), score=round(score, 3), picked=tier)
    if known and tier != known["tier"]:
        edge = min(abs(score - _weights["t_sonnet"]), abs(score - _weights["t_opus"]))
        if edge < HYSTERESIS:
            entry["why"] = "kept: too close to switch"
            tier = known["tier"]
    entry.setdefault("why", "classified")
    entry["tier"] = tier
    with _lock:
        _conversations[conv] = {"tier": tier, "score": score, "seen": time.time()}
        if len(_conversations) > 500:
            for k, _ in sorted(_conversations.items(), key=lambda kv: kv[1]["seen"])[:100]:
                del _conversations[k]
    return MODELS[tier], entry


def _dump_shape(data):
    """Debugging: the request's message structure, text cut short."""
    def cut(v):
        if isinstance(v, str):
            return v[:150] + (f"…(+{len(v) - 150})" if len(v) > 150 else "")
        if isinstance(v, list):
            return [cut(x) for x in v]
        if isinstance(v, dict):
            return {k: cut(x) for k, x in v.items()}
        return v
    shape = {"model": data.get("model"), "tools": len(data.get("tools") or []),
             "keys": sorted(data), "messages": cut(data.get("messages"))}
    with open(os.environ["TIER_ROUTER_DUMP"], "a") as f:
        f.write(json.dumps(shape) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet; decisions go to the log
        pass

    def _forward(self, body):
        cls = http.client.HTTPSConnection if UPSTREAM.scheme == "https" else http.client.HTTPConnection
        conn = cls(UPSTREAM.hostname, UPSTREAM.port, timeout=600)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        headers["Content-Length"] = str(len(body))
        conn.request(self.command, self.path, body=body, headers=headers)
        return conn, conn.getresponse()

    def _relay(self, conn, resp):
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_HEADERS:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
        finally:
            conn.close()

    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        sent = body
        entry = None
        if self.command == "POST" and self.path.split("?")[0] == "/v1/messages" and body:
            try:
                data = json.loads(body)
                if os.environ.get("TIER_ROUTER_DUMP"):
                    _dump_shape(data)
                model, entry = route(data)
                if model and model != data.get("model"):
                    data["model"] = model
                    sent = json.dumps(data).encode()
                entry["sent"] = model
            except (ValueError, KeyError) as e:
                entry = {"t": round(time.time(), 3), "why": f"unparsed: {e}"}
        try:
            conn, resp = self._forward(sent)
            if sent is not body and 400 <= resp.status < 500 and resp.status not in NO_RETRY:
                # Refused as re-modelled (a parameter that model lacks, or one
                # the plan does not include): send it as it came.
                # Auth errors (401, 403) go back to the client as they are.
                detail = resp.read()[:300].decode(errors="replace")
                conn.close()
                if entry is not None:
                    entry.update(retried=resp.status, error=detail)
                conn, resp = self._forward(body)
            if entry is not None:
                _log(entry)
            self._relay(conn, resp)
        except (OSError, http.client.HTTPException) as e:
            if entry is not None:
                entry["upstream_error"] = str(e)
                _log(entry)
            try:
                msg = json.dumps({"type": "error", "error": {"type": "api_error",
                                  "message": f"tier-router could not reach the API: {e}"}}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            except OSError:
                pass

    do_POST = do_GET = do_PUT = do_DELETE = do_PATCH = _handle


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=int(os.environ.get("TIER_ROUTER_PORT", "8787")))
    a = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    server.daemon_threads = True
    print(f"tier-router on http://127.0.0.1:{a.port} -> {UPSTREAM.geturl()}; "
          f"models {MODELS}; log {LOG}", file=sys.stderr, flush=True)
    def warm():
        # Load tev1 now, so the first prompt does not wait ~40 s for it.
        try:
            classify.ask_tev1("warm up", timeout=120)
        except Exception as e:
            print(f"tier-router: tev1 not ready ({e}); prompts keep their model until it is",
                  file=sys.stderr, flush=True)
    threading.Thread(target=warm, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
