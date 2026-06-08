import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


UPSTREAM_BASE_URL = "http://163.228.113.120:8000/v1"
MODEL_NAME = f"openai/gpt-5.2"
API_KEY = "EMPTY"


def fix_openai_compatible_response(data):
    """把服务端返回的 message.reasoning 补到标准的 message.content。"""
    for choice in data.get("choices", []):
        message = choice.get("message") or {}
        if message.get("content") is None and message.get("reasoning"):
            message["content"] = message["reasoning"]
    return data


class CompatProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def _forward(self):
        upstream_url = f"{UPSTREAM_BASE_URL}{self.path.removeprefix('/v1')}"
        body = None

        if self.command in {"POST", "PUT", "PATCH"}:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length else b""
            body_data = json.loads(raw_body.decode("utf-8")) if raw_body else {}

            # CrewAI 使用 openai/gpt-5.2 标识 provider；转发给 vLLM 时去掉 openai/ 前缀。
            model = body_data.get("model")
            if isinstance(model, str) and model.startswith("openai/"):
                body_data["model"] = model.removeprefix("openai/")

            # 这个简单代理处理普通 JSON 响应，不处理 SSE stream。
            body_data["stream"] = False
            body = json.dumps(body_data, ensure_ascii=False).encode("utf-8")

        request = urllib.request.Request(
            upstream_url,
            data=body,
            method=self.command,
            headers={
                "Content-Type": "application/json",
                "Authorization": self.headers.get("Authorization", f"Bearer {API_KEY}"),
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = response.status
                response_body = response.read()
        except urllib.error.HTTPError as error:
            status = error.code
            response_body = error.read()

        content_type = "application/json"
        try:
            response_data = json.loads(response_body.decode("utf-8"))
            if self.path.endswith("/chat/completions"):
                response_data = fix_openai_compatible_response(response_data)
            response_body = json.dumps(response_data, ensure_ascii=False).encode("utf-8")
        except Exception:
            content_type = "text/plain; charset=utf-8"

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def start_compat_proxy():
    server = ThreadingHTTPServer(("127.0.0.1", 0), CompatProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/v1", MODEL_NAME, API_KEY
