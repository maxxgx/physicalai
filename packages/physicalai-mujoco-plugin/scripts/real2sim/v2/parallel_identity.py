"""r18 5(a) (coordinator): the exact parallel planner gives the same episodes as the serial one. Runs baseline_r16.py on
the same seeds with PHYSICALAI_PLAN_PARALLEL=0 and =N (in-process planning, PHYSICALAI_PLAN_PROCESS=0, so the comparison
is the planner alone) and compares per seed: success, failure, sim_s, glitch, max penetration and unclean reasons; also
the wall time and real-time factor (sim_s / wall_s) per seed. Run on an otherwise idle machine for the timings.
usage: parallel_identity.py SEEDS(comma list) N OUT_DIR"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ("success", "failure", "sim_s", "glitch", "max_penetration_m", "unclean_reasons")


def run(seeds: str, n: int, out: Path) -> tuple[dict, float]:
    env = dict(os.environ, SEEDS=seeds, PHYSICALAI_PLAN_PARALLEL=str(n), PHYSICALAI_PLAN_PROCESS="0")
    log = out / f"p{n}.log"; t0 = time.perf_counter()
    with open(log, "w") as fh:
        subprocess.run([sys.executable, "-B", str(HERE / "baseline_r16.py"), "0", "0", "default", str(out / f"p{n}.json")],
                       env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
    rows = {r["seed"]: r for r in (json.loads(x) for x in open(log) if x.startswith('{"seed"'))}
    return rows, round(time.perf_counter() - t0, 1)


def main() -> None:
    seeds, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    serial, w0 = run(seeds, 0, out)
    par, w1 = run(seeds, n, out)
    per = []
    for s in sorted(serial):
        a, b = serial[s], par.get(s)
        diff = {k: [a[k], b[k]] for k in KEYS if b is None or a[k] != b[k]} if b is not None else {"missing": True}
        per.append({"seed": s, "identical": not diff, "diff": diff, "sim_s": a["sim_s"],
                    "wall_s": [a.get("wall_s"), b.get("wall_s") if b else None],
                    "rtf": [round(a["sim_s"] / a["wall_s"], 2) if a.get("wall_s") else None,
                            round(b["sim_s"] / b["wall_s"], 2) if b and b.get("wall_s") else None]})
    res = {"method": __doc__.split("usage")[0].strip(), "workers": n, "seeds": len(per),
           "identical": sum(p["identical"] for p in per), "wall_total_s": {"serial": w0, f"parallel_{n}": w1}, "per_seed": per}
    json.dump(res, open(out / "identity.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("workers", "seeds", "identical", "wall_total_s")}))
    sys.exit(0 if res["identical"] == res["seeds"] else 1)


if __name__ == "__main__":
    main()
