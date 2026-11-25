"""r14 (D7 diagnosis): a fixed-rate MJPEG source for Studio's IP cameras, to measure the duplicates Studio's own
sampling adds. Serves /cameras/overview/mjpeg and /cameras/wrist/mjpeg at exactly FPS frames per second (paced on a
monotonic clock), every frame different everywhere (pre-generated noise frames, 640x480, cycled; a frame never repeats
its predecessor). usage: fake_mjpeg.py PORT FPS"""
import sys, time, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np, cv2
PORT, FPS = int(sys.argv[1]), float(sys.argv[2])
rng = np.random.default_rng(0)
FRAMES = [cv2.imencode(".jpg", rng.integers(0, 255, (480, 640, 3), dtype=np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes() for _ in range(64)]
B = b"frame"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if not self.path.endswith("/mjpeg"): self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={B.decode()}"); self.end_headers()
        t0, k = time.monotonic(), 0
        try:
            while True:
                jpg = FRAMES[k % len(FRAMES)]
                self.wfile.write(b"--" + B + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                k += 1; nxt = t0 + k / FPS; time.sleep(max(0.0, nxt - time.monotonic()))
        except (BrokenPipeError, ConnectionResetError):
            return
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
