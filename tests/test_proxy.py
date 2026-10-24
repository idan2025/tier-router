"""End-to-end check of the proxy against a fake Anthropic API on localhost.
Needs tev1 running in Ollama; spends no Claude tokens.

  python3 -I tests/test_proxy.py
"""
import json, os, subprocess, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
seen = []


class FakeAPI(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append({"model": body["model"], "auth": self.headers.get("x-api-key")})
        if body.get("metadata", {}).get("deny") == body["model"]:
            msg = b'{"type":"error","error":{"type":"authentication_error","message":"bad key"}}'
            self.send_response(401); self.send_header("Content-Length", str(len(msg))); self.end_headers()
            self.wfile.write(msg); return
        if body.get("metadata", {}).get("refuse") == body["model"]:
            msg = b'{"type":"error","error":{"type":"invalid_request_error","message":"nope"}}'
            self.send_response(400); self.send_header("Content-Length", str(len(msg))); self.end_headers()
            self.wfile.write(msg); return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for ev in ("message_start", "content_block_delta", "message_stop"):
            data = f"event: {ev}\ndata: {json.dumps({'type': ev, 'model': body['model']})}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data)); self.wfile.flush(); time.sleep(0.05)
        self.wfile.write(b"0\r\n\r\n")


def post(body):
    req = urllib.request.Request("http://127.0.0.1:8788/v1/messages", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "x-api-key": "secret-123"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read().decode()


def main():
    api = ThreadingHTTPServer(("127.0.0.1", 8799), FakeAPI)
    threading.Thread(target=api.serve_forever, daemon=True).start()
    log = ROOT / "tests/.decisions.jsonl"
    log.unlink(missing_ok=True)
    env = {**os.environ, "TIER_ROUTER_UPSTREAM": "http://127.0.0.1:8799", "TIER_ROUTER_LOG": str(log),
           "TIER_ROUTER_HAIKU": "h", "TIER_ROUTER_SONNET": "s", "TIER_ROUTER_OPUS": "o"}
    proxy = subprocess.Popen([sys.executable, "-s", "-m", "tier_router.proxy", "--port", "8788"],
                             cwd=ROOT, env=env, stderr=subprocess.PIPE, text=True)
    try:
        print(proxy.stderr.readline().strip())
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:8788/", timeout=1)
            except urllib.error.HTTPError:
                break
            except OSError:
                time.sleep(0.2)
        tools = [{"name": "Bash", "input_schema": {"type": "object"}}]
        first = {"role": "user", "content": [{"type": "text", "text": "<system-reminder>ctx</system-reminder>merge it"}]}
        checks = []

        def check(name, body, want):
            n = len(seen)
            status, text = post(body)
            got = seen[n]["model"] if len(seen) > n else None
            ok = got == want and status == 200 and "message_stop" in text and seen[n]["auth"] == "secret-123"
            checks.append(ok)
            print(f"{'ok ' if ok else 'BAD'} {name}: sent {got!r}, wanted {want!r}")

        check("background call passes through",
              {"model": "claude-haiku-5-5", "messages": [first]}, "claude-haiku-5-5")
        # As Claude Code sends it: reminders in the user message, the prompt
        # its last text block, and a `system` message after it.
        env = {"role": "system", "content": [{"type": "text", "text": "# Environment ..."}]}
        check("'merge it' starts a turn -> haiku",
              {"model": "claude-opus-5-5", "tools": tools, "messages": [first, env]}, "h")
        follow = [first, {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "Bash", "input": {}}]},
                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "merged"}]}]
        check("tool-result follow-up keeps the turn's model",
              {"model": "claude-opus-5-5", "tools": tools, "messages": follow + [env]}, "h")
        hard = follow + [{"role": "assistant", "content": [{"type": "text", "text": "Merged PR #12."}]},
                         {"role": "user", "content": "the vpn drops randomly when the laptop switches from wifi to ethernet, find out why and fix it"}]
        check("a hard new prompt in the same conversation -> opus",
              {"model": "claude-opus-5-5", "tools": tools, "messages": hard + [env]}, "o")
        check("a re-modelled request the API refuses is sent again as it came",
              {"model": "claude-opus-5-5", "tools": tools, "metadata": {"refuse": "h"},
               "messages": [{"role": "user", "content": "what's the default port for postgres?"}]},
              "h")
        print("    then:", seen[-1]["model"])
        checks[-1] = checks[-1] and seen[-2]["model"] == "h" and seen[-1]["model"] == "claude-opus-5-5"
        # A credential error is the same for any model: pass it on, do not retry.
        n = len(seen)
        try:
            post({"model": "claude-opus-5-5", "tools": tools, "metadata": {"deny": "h"},
                  "messages": [{"role": "user", "content": "what's the default port for postgres?"}]})
            status = 200
        except urllib.error.HTTPError as err:
            status = err.code
        ok = status == 401 and len(seen) == n + 1
        checks.append(ok)
        print(f"{'ok ' if ok else 'BAD'} a 401 is passed on without a retry: status {status}, {len(seen) - n} request(s)")
        print("\nlog:")
        for l in open(log):
            e = json.loads(l)
            print(" ", {k: e[k] for k in ("why", "tier", "score", "ms", "sent") if k in e})
        print("\nALL OK" if all(checks) else "\nFAILED")
        return 0 if all(checks) else 1
    finally:
        proxy.terminate()
        api.shutdown()


if __name__ == "__main__":
    sys.exit(main())
