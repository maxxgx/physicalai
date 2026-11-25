"""Episode intake: copy the real robot episodes into REAL2SIM_ROOT/episodes/ (optional SRC) and report METADATA ONLY, so a
reviewer can lock the held-out episodes before anyone looks at a frame. No video frame is decoded: LeRobot metadata
(meta/info.json, meta/episodes/*.parquet, meta/tasks.parquet) and the container headers (ffprobe -show_streams /
-show_format, which read no pixels). Writes episodes/INTAKE.json and prints a summary.
usage: intake_episodes.py [SRC]   (SRC: a LeRobot dataset folder or a folder of them; copied, never moved)"""
import json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(os.environ.get("REAL2SIM_ROOT", "~/real2sim-data/bucket-v2")).expanduser()
DEST = ROOT / "episodes"
if len(sys.argv) > 1:
    src = Path(sys.argv[1]).expanduser()
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / src.name
    if target.exists():
        raise SystemExit(f"{target} exists: not overwriting an intake")
    shutil.copytree(src, target)
    print("copied", src, "->", target)

datasets = sorted({p.parent.parent for p in DEST.rglob("meta/info.json")})
report = {"root": str(DEST), "datasets": [], "note": "metadata only: no frame decoded (ffprobe headers, LeRobot meta)"}


def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return {"error": out.stderr.strip()[:200]}
    j = json.loads(out.stdout); s = next((x for x in j.get("streams", []) if x.get("codec_type") == "video"), {})
    return {"codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"), "width": s.get("width"), "height": s.get("height"),
            "avg_frame_rate": s.get("avg_frame_rate"), "nb_frames": s.get("nb_frames"),
            "duration_s": float(j.get("format", {}).get("duration", 0) or 0), "size_mb": round(int(j.get("format", {}).get("size", 0) or 0) / 1e6, 1)}


for ds in datasets:
    info = json.loads((ds / "meta/info.json").read_text())
    entry = {"path": str(ds), "fps": info.get("fps"), "total_episodes": info.get("total_episodes"), "total_frames": info.get("total_frames"),
             "robot_type": info.get("robot_type"), "features": {k: {kk: v.get(kk) for kk in ("dtype", "shape", "names") if kk in v}
                                                                for k, v in info.get("features", {}).items()}}
    try:
        import pandas as pd
        eps = sorted((ds / "meta/episodes").rglob("*.parquet"))
        if eps:
            e = pd.concat([pd.read_parquet(p) for p in eps])
            cols = [c for c in ("episode_index", "length", "tasks") if c in e.columns]
            entry["episodes"] = [{c: (r[c].tolist() if hasattr(r[c], "tolist") else r[c]) for c in cols} for _, r in e[cols].iterrows()]
        if (ds / "meta/tasks.parquet").exists():
            entry["tasks"] = pd.read_parquet(ds / "meta/tasks.parquet").index.tolist()
    except ImportError:
        entry["episodes_note"] = "pandas missing: run with the Studio venv's python for the episode table"
    entry["videos"] = {str(v.relative_to(ds)): probe(v) for v in sorted((ds / "videos").rglob("*.mp4"))}
    report["datasets"].append(entry)

(DEST / "INTAKE.json").write_text(json.dumps(report, indent=1, default=str))
for e in report["datasets"]:
    cams = sorted({Path(k).parts[1].replace("observation.images.", "") for k in e["videos"] if len(Path(k).parts) > 1})
    res = sorted({(v.get("width"), v.get("height"), v.get("avg_frame_rate")) for v in e["videos"].values()})
    print(f"{e['path']}: {e['total_episodes']} episodes, {e['total_frames']} frames at {e['fps']} fps; cameras {cams}; streams {res}")
    for ep in e.get("episodes", [])[:50]:
        print("  episode", ep.get("episode_index"), "length", ep.get("length"), "frames", f"({(ep.get('length') or 0) / (e['fps'] or 30):.1f} s)", ep.get("tasks"))
print("wrote", DEST / "INTAKE.json", "| datasets:", len(report["datasets"]))
