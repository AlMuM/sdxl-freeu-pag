"""盲评网页的自托管服务端（只用 Python 标准库）。

把 04b_build_online_eval.py 生成的 online_eval/ 放到 --site 指定的目录，
所有页面和接口都挂在一段随机口令路径下（/<token>/），没有完整链接的人打不开。
每位评审的结果存成 <results>/results_<名字>.json，格式与离线版导出的一致，
拉回本地放进 eval_results/ 即可直接运行 05_analyze.py。

用法（服务器上）：
    python3 eval_server.py --site site --results results --port 8080 --token <随机串>
接口：
    GET  /<token>/api/ping
    GET  /<token>/api/answers?rater=<名字>       该评审已保存的全部答案
    POST /<token>/api/answer   {"rater", "prompt_id", "answer"}
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

NAME_RE = re.compile(r"^[\w\-]{1,32}$")  # \w 在 Python 里包含中文
PID_RE = re.compile(r"^p\d{3}$")
SLOT_RE = re.compile(r"^[A-F]$")
MAX_BODY = 16 * 1024
lock = threading.Lock()


def clean_answer(a: dict) -> dict:
    """只保留已知字段，防止写入任意内容。"""
    slot = lambda v: v if isinstance(v, str) and SLOT_RE.match(v) else None
    tags = {}
    for s, lst in (a.get("tags") or {}).items():
        if SLOT_RE.match(str(s)) and isinstance(lst, list):
            tags[s] = [str(t)[:20] for t in lst[:5]]
    return {"best": slot(a.get("best")), "worst": slot(a.get("worst")), "tie": bool(a.get("tie")),
            "tags": tags, "ms": int(a.get("ms") or 0), "updatedAt": int(a.get("updatedAt") or 0)}


def make_handler(site: Path, results: Path, token: str):
    prefix = f"/{token}/"

    class Handler(BaseHTTPRequestHandler):
        server_version = "eval/1"

        def log_message(self, fmt, *args):  # 只记方法和状态，不记口令路径
            print(f"{datetime.now():%m-%d %H:%M:%S} {self.command} {args[1] if len(args) > 1 else ''}", flush=True)

        def send_json(self, code: int, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def route(self):
            url = urlparse(self.path)
            if url.path == f"/{token}":  # 补上结尾斜杠，保证相对路径的图片能加载
                self.send_response(301)
                self.send_header("Location", prefix)
                self.end_headers()
                return None, None
            if not url.path.startswith(prefix):
                self.send_error(404)
                return None, None
            return unquote(url.path[len(prefix):]), parse_qs(url.query)

        def results_file(self, rater: str) -> Path:
            return results / f"results_{rater}.json"

        def do_GET(self):
            rel, query = self.route()
            if rel is None:
                return
            if rel == "api/ping":
                return self.send_json(200, {"ok": True})
            if rel == "api/answers":
                rater = (query.get("rater") or [""])[0]
                if not NAME_RE.match(rater):
                    return self.send_json(400, {"error": "bad rater"})
                f = self.results_file(rater)
                data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"answers": {}}
                return self.send_json(200, data["answers"])
            root = site.resolve()
            path = (root / (rel or "index.html")).resolve()
            if root not in path.parents or not path.is_file():  # 拒绝 ../ 越界访问
                return self.send_error(404)
            body = path.read_bytes()
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/"):
                ctype += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache" if path.suffix == ".html" else "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            rel, _ = self.route()
            if rel is None:
                return
            if rel != "api/answer":
                return self.send_error(404)
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                return self.send_json(413, {"error": "body size"})
            try:
                body = json.loads(self.rfile.read(length))
                rater, pid = str(body["rater"]), str(body["prompt_id"])
                answer = clean_answer(body["answer"])
            except (ValueError, KeyError, TypeError, AttributeError):
                return self.send_json(400, {"error": "bad json"})
            if not NAME_RE.match(rater) or not PID_RE.match(pid):
                return self.send_json(400, {"error": "bad rater or prompt_id"})
            with lock:
                f = self.results_file(rater)
                data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"rater": rater, "answers": {}}
                data["answers"][pid] = answer
                data["exported_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(f)
            self.send_json(200, {"ok": True})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--token", required=True)
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.site, args.results, args.token))
    print(f"serving {args.site} on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
