"""flywheel-cams: both cameras side by side in the browser, to aim them. No Rerun, no robot.

    scripts/cams.sh            # then open http://localhost:8642

Run it next to `scripts/teleop.sh` (arms only) to practise the task while looking at what the policy will see.
Stop it before recording: a camera can only be read by one program at a time.
"""

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PORT = 8642
PAGE = b"""<!doctype html><meta charset=utf-8><title>cameras</title>
<body style="margin:0;background:#08080a;display:grid;place-items:center;height:100vh">
<img src="/stream" style="max-width:100vw;max-height:100vh">"""


class Feed:
    """Reads the cameras in one thread and keeps the latest side-by-side JPEG."""

    def __init__(self, cams, size=(640, 480), fps=15):
        self.names = list(cams)
        self.caps = []
        for name, index in cams.items():
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            if not cap.isOpened():
                sys.exit(f"camera {name} (index {index}) did not open: is another program using it?")
            self.caps.append(cap)
        self.size, self.period, self.jpeg = size, 1 / fps, b""
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            t0 = time.monotonic()
            tiles = []
            for name, cap in zip(self.names, self.caps):
                ok, frame = cap.read()
                frame = cv2.resize(frame, self.size) if ok else np.zeros((self.size[1], self.size[0], 3), np.uint8)
                cv2.putText(frame, name if ok else f"{name}: no frame", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (26, 122, 255), 2)
                tiles.append(frame)
            self.jpeg = cv2.imencode(".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            time.sleep(max(self.period - (time.monotonic() - t0), 0))


def main():
    cams = dict(zip(("top", "wrist"), map(int, sys.argv[1:3]))) if len(sys.argv) >= 3 else {"top": 1, "wrist": 0}
    feed = Feed(cams)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + feed.jpeg + b"\r\n")
                    time.sleep(feed.period)
            except (BrokenPipeError, ConnectionResetError):
                pass

    print(f"cameras {cams} → http://localhost:{PORT}   (Ctrl+C to stop)")
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
