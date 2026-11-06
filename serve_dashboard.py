"""Serve the dashboard on your own computer.

  python serve_dashboard.py            # http://127.0.0.1:8060
  python serve_dashboard.py --port 9000 --rebuild

--rebuild runs build_dashboard.py first (use it after changing docs/dashboard_template.html or the data).
Only the docs/ folder is served, and only to this computer.
"""
import argparse
import functools
import http.server
import socketserver
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, default=8060)
p.add_argument('--rebuild', action='store_true')
p.add_argument('--no-browser', action='store_true')
a = p.parse_args()

root = Path(__file__).resolve().parent
if a.rebuild:
    subprocess.run([sys.executable, 'build_dashboard.py'], cwd=root, check=True)
docs = root / 'docs'
if not (docs / 'dashboard.html').exists():
    raise SystemExit('docs/dashboard.html is missing. Run: python build_dashboard.py')


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.path = '/dashboard.html'
        super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, *args):
        pass


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('127.0.0.1', a.port), functools.partial(Handler, directory=str(docs))) as srv:
    url = f'http://127.0.0.1:{a.port}'
    print(f'Tsetlin Pair Lab running at {url}  (Ctrl+C to stop)')
    if not a.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('stopped')
