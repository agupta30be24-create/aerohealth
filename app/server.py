"""AeroHealth 3D server: serves the Three.js frontend (web/) and runs the trained model on uploaded files.

Run from the repo root:   python app/server.py            -> http://localhost:8765
                          python app/server.py --port 9000

Endpoints
  GET  /                    3D frontend (web/index.html)
  GET  /examples/<name>     the example upload files
  GET  /api/health          {"ok": true}
  POST /api/predict?model=auto|FD001|FD002|FD003|FD004   body = engine sensor file (C-MAPSS .txt or .csv)

Uploads go through exactly the same code as the NASA fleets (src/export_frontend.build_payload) and the Streamlit
dashboard: Preprocessor fitted on training engines only -> trained XGBoost -> calibrated risk -> SHAP.
Standard library only (no extra packages). Binds to localhost by default.
"""
import argparse
import json
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import aerohealth as ah  # noqa: E402
import export_frontend as ef  # noqa: E402

MAX_UPLOAD = 50 * 1024 * 1024
EXAMPLES = {p.name: p for p in (ROOT / "examples").glob("*.csv")}
_models, _lock = {}, threading.Lock()


def trained(fd):
    with _lock:
        if fd not in _models:
            print(f"training {fd} model on its training engines (first request only)...", flush=True)
            _models[fd] = ef.TrainedModel(fd)
        return _models[fd]


def predict(data, model="auto"):
    df, notes = ah.read_engine_file(data)
    suggestion = ah.suggest_model(df)
    fd = suggestion[0] if model == "auto" else model
    if fd not in ah.DATASETS:
        raise ah.UploadError(f"Unknown model '{model}'. Use auto or one of {', '.join(ah.DATASETS)}.")
    return ef.upload_payload(trained(fd), df, notes, suggestion)


class Handler(SimpleHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._json(200, {"ok": True, "datasets": ah.DATASETS})
        if path.startswith("/examples/"):
            p = EXAMPLES.get(path.split("/")[-1])
            if not p:
                return self._json(404, {"error": "No such example."})
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/predict":
            return self._json(404, {"error": "Not found."})
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0:
            return self._json(400, {"error": "The file is empty."})
        if size > MAX_UPLOAD:
            return self._json(413, {"error": "File too large (limit 50 MB)."})
        data = self.rfile.read(size)
        model = parse_qs(url.query).get("model", ["auto"])[0]
        try:
            return self._json(200, predict(data, model))
        except ah.UploadError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:  # unexpected: report, do not crash the server
            return self._json(500, {"error": f"Could not analyse this file ({type(e).__name__}: {e})."})

    def log_message(self, fmt, *args):
        if "/api/" in str(args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only if others on your network should reach it")
    a = ap.parse_args()
    server = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(ROOT / "web")))
    print(f"AeroHealth 3D: http://localhost:{a.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
