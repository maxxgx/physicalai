"""True staleness of the plugin's camera streams as a 30 Hz latest-frame recorder sees them (Studio's IP camera: a reader
thread drains the MJPEG stream into a latest-frame slot; the recorder samples the slot on its own clock). A sample is
stale when it holds the same received frame as the previous sample. Runs the autopilot (drive) meanwhile.
usage: stale_probe.py [--sim http://127.0.0.1:8931] [--seconds 60] [--hz 30]"""
import argparse, json, threading, time, urllib.request
import cv2
ap = argparse.ArgumentParser(); ap.add_argument("--sim", default="http://127.0.0.1:8931"); ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--hz", type=float, default=30); a = ap.parse_args()
def post(path, body):
    urllib.request.urlopen(urllib.request.Request(a.sim + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"}, method="POST"), timeout=10).read()
post("/episode/auto-reset", {"enabled": True, "dwell_s": 2.0}); post("/autopilot", {"mode": "drive"})
latest = {}; stop = threading.Event()
def reader(cam):
    c = cv2.VideoCapture(f"{a.sim}/cameras/{cam}/mjpeg"); n = 0
    while not stop.is_set():
        ok, f = c.read()
        if ok:
            n += 1; latest[cam] = (n, time.monotonic(), cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype("float32"))
for cam in ("overview", "wrist"): threading.Thread(target=reader, args=(cam,), daemon=True).start()
time.sleep(3)
res = {cam: {"samples": 0, "stale": 0, "received": 0, "small_diff": 0} for cam in ("overview", "wrist")}; prev = {}; first = {}; prev_img = {}
t0 = time.monotonic(); k = 0
while time.monotonic() - t0 < a.seconds:
    k += 1; due = t0 + k / a.hz
    while time.monotonic() < due: time.sleep(0.0005)
    for cam in res:
        n, _, img = latest.get(cam, (0, 0, None)); first.setdefault(cam, n)
        if img is not None and cam in prev_img and abs(img - prev_img[cam]).mean() < 0.25: res[cam]["small_diff"] += 1
        prev_img[cam] = img
        res[cam]["samples"] += 1; res[cam]["stale"] += int(prev.get(cam) == n); prev[cam] = n; res[cam]["received"] = n - first[cam]
stop.set(); post("/autopilot", {"mode": "off"})
for cam, r in res.items():
    print(f"{cam}: received {r['received'] / a.seconds:.1f} fps; {r['samples']} samples at {a.hz:.0f} Hz, stale {r['stale']} ({r['stale'] / r['samples'] * 100:.1f} %); mean |diff| < 0.25 at 160x120 (dscheck's test, before Studio's video coding) {r['small_diff']} ({r['small_diff'] / r['samples'] * 100:.1f} %)")
