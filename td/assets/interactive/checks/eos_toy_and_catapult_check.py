#!/usr/bin/env python3
"""Check eos_toy_and_catapult.html against Python implementations.

Run from anywhere:  python td/assets/interactive/checks/eos_toy_and_catapult_check.py

1. Extracts the <script id="sim"> block (the figure's simulation code), runs it
   under Node for several parameter sets, and compares every iterate with
   a Python port (same PRNG, same update order).
2. Checks that the Python port of the toy map is the map used in the notebook
   (the `toy_run` code cell of td/build_td2_3_edge_of_stability.py), by running
   both noise-free and comparing.
3. Checks the catapult maps against explicit gradient descent on (u, v).
4. Checks the physics the figure's captions state: S hovers at 2/eta with
   <x^2> = sigma^2 = 10 c (eta/2)^6; abrupt start from y0 = 8 at eta = 0.4
   diverges while a T = 1600 warmup does not; catapult phases at 1.5 / 3 / 4.2.
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HTML = HERE.parent / "eos_toy_and_catapult.html"
BUILDER = HERE.parents[2] / "build_td2_3_edge_of_stability.py"
C = 62.5
fails = 0


def ok(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg)
    fails += 0 if cond else 1


# ------------------------------------------------------------ Python ports
def mulberry32(seed):
    a = seed & 0xFFFFFFFF

    def imul(x, y):
        return (x * y) & 0xFFFFFFFF

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = imul(t ^ (t >> 15), t | 1)
        t = (t ^ ((t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return nxt


def gaussians(K, seed):
    u = mulberry32(seed)
    out = np.empty(K)
    for k in range(K):
        u1 = 1 - u()
        u2 = u()
        out[k] = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return out


def toy_run_port(K, eta, y0, x0, ramp, T, noise, seed):
    xs = np.full(K, np.nan); ys = np.full(K, np.nan)
    xi = gaussians(K, seed)
    x, y, div = x0, y0, False
    for k in range(K):
        e = eta * min(1, (k + 1) / T) if ramp else eta
        xs[k], ys[k] = x, y
        xn = (1 - e * y) * x + noise * xi[k]
        yn = y - e * (0.5 * x * x - 5 * C / y**6)
        x, y = xn, yn
        if not (math.isfinite(x) and math.isfinite(y)) or y <= 0 or abs(x) > 1e3:
            div = True
            break
    return xs, ys, div


def catapult_port(K, f0, a0, n, eta):
    f = np.full(K + 1, np.nan); al = np.full(K + 1, np.nan)
    ff, a, div = f0, a0, False
    for k in range(K + 1):
        f[k], al[k] = ff, a
        if k == K:
            break
        b = eta * eta * ff * ff / n
        ff, a = ff * (1 - a + b), a + b * (a - 4)
        if not math.isfinite(ff) or abs(ff) > 1e100:
            div = True
            f[k + 1] = np.nan
            break
    return f, al, div


# ------------------------------------------------------------ 1. JS vs Python
sim = re.search(r'<script id="sim">(.*?)</script>', HTML.read_text(), re.S).group(1)
toy_cases = [
    dict(K=4000, eta=0.4, y0=3.0, x0=1e-3, ramp=False, T=400, noise=1e-3, seed=1),
    dict(K=4000, eta=0.3, y0=3.0, x0=1e-3, ramp=False, T=400, noise=1e-3, seed=1),
    dict(K=4000, eta=0.6, y0=2.0, x0=1e-3, ramp=False, T=400, noise=0.0, seed=1),
    dict(K=4000, eta=0.4, y0=8.0, x0=1e-3, ramp=False, T=400, noise=1e-3, seed=1),
    dict(K=4000, eta=0.4, y0=8.0, x0=1e-3, ramp=True, T=1600, noise=1e-3, seed=1),
]
cat_cases = [dict(K=150, f0=1.0, a0=a, n=n, eta=1.0) for a in (0.5, 1.5, 2.5, 3.0, 3.9, 4.2) for n in (100, 1000, 100000)]
harness = sim + "\nconst out = {toy: [], cat: []};\n"
harness += f"for (const p of {json.dumps(toy_cases)}) {{ const r = toyRun(p); out.toy.push({{xs: Array.from(r.xs), ys: Array.from(r.ys), div: r.diverged}}); }}\n"
harness += f"for (const p of {json.dumps(cat_cases)}) {{ const r = catapultRun(p); out.cat.push({{f: Array.from(r.f), al: Array.from(r.al), div: r.diverged}}); }}\n"
harness += "process.stdout.write(JSON.stringify(out, (k, v) => (typeof v === 'number' && !isFinite(v)) ? null : v));\n"
res = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=True)
js = json.loads(res.stdout)


def arr(v):
    return np.array([np.nan if a is None else a for a in v], dtype=float)


for p, r in zip(toy_cases, js["toy"]):
    xs, ys, div = toy_run_port(**p)
    jx, jy = arr(r["xs"]), arr(r["ys"])
    same_nan = np.array_equal(np.isnan(xs), np.isnan(jx))
    m = ~np.isnan(xs)
    err = np.max(np.abs(xs[m] - jx[m]) + np.abs(ys[m] - jy[m])) if m.any() else 0.0
    ok(same_nan and div == r["div"] and err < 1e-9,
       f"toy JS == Python  eta={p['eta']} y0={p['y0']} ramp={p['ramp']} noise={p['noise']}: max|diff|={err:.1e}, diverged={div}")
for p, r in zip(cat_cases, js["cat"]):
    f, al, div = catapult_port(**p)
    jf, ja = arr(r["f"]), arr(r["al"])
    m = ~np.isnan(f)
    rel = np.max(np.abs(f[m] - jf[m]) / (1 + np.abs(f[m])) + np.abs(al[m] - ja[m]) / (1 + np.abs(al[m])))
    ok(np.array_equal(np.isnan(f), np.isnan(jf)) and div == r["div"] and rel < 1e-12,
       f"catapult JS == Python  a0={p['a0']} n={p['n']}: max rel diff={rel:.1e}, diverged={div}")

# ------------------------------------------------------------ 2. port == notebook's toy map
ns = {}
src = BUILDER.read_text()
sys.path.insert(0, str(BUILDER.parent))
spec = {}
exec(compile(src.replace('raise SystemExit(run(CELLS, "td2_3_edge_of_stability", "td2_3_edge_of_stability"))', "pass"), str(BUILDER), "exec"), spec)
toy_cells = [c["source"] for c in spec["CELLS"] if c["kind"] == "code" and "def toy_run" in c["source"]]
ok(len(toy_cells) == 1, "found the notebook's toy_run cell")
exec("import numpy as np\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n" + toy_cells[0].split("# ---- demo")[0], ns)
for eta, y0 in [(0.3, 3.0), (0.4, 3.0), (0.5, 3.0), (0.4, 8.0)]:
    nx, ny = ns["toy_run"](np.full(3000, eta), y0=y0, x0=1e-3, noise=0.0)
    px, py, _ = toy_run_port(3000, eta, y0, 1e-3, False, 1, 0.0, 1)
    m = ~np.isnan(px)
    same = np.array_equal(np.isnan(nx), np.isnan(px)) and np.max(np.abs(nx[m] - px[m]) + np.abs(ny[m] - py[m])) < 1e-12
    ok(same, f"notebook toy_run == figure map (noise-free) eta={eta} y0={y0}")

# ------------------------------------------------------------ 3. catapult maps vs explicit GD
rng = np.random.default_rng(0)
n = 500
u = rng.standard_normal(n); v = rng.standard_normal(n)
for a0 in (1.5, 3.0, 3.9):
    lam0 = (u @ u + v @ v) / n
    eta = a0 / lam0
    uu, vv = u.copy(), v.copy()
    f, al, _ = catapult_port(30, v @ u / math.sqrt(n), a0, n, eta)
    err = 0.0
    for k in range(30):
        fk = vv @ uu / math.sqrt(n)
        ak = eta * (uu @ uu + vv @ vv) / n
        err = max(err, abs(fk - f[k]) / (1 + abs(f[k])), abs(ak - al[k]))
        uu, vv = uu - eta * fk * vv / math.sqrt(n), vv - eta * fk * uu / math.sqrt(n)
    ok(err < 1e-9, f"catapult maps == explicit GD on (u, v), eta*lambda0={a0}: max err {err:.1e}")

# ------------------------------------------------------------ 4. stated physics
for eta in (0.3, 0.35, 0.4, 0.45, 0.5):
    xs, ys, div = toy_run_port(4000, eta, 3.0, 1e-3, False, 1, 1e-3, 1)
    S = ys[-1000:].mean(); x2 = (xs[-1000:] ** 2).mean(); s2 = 10 * C * (eta / 2) ** 6
    ok(not div and abs(S - 2 / eta) < 0.01 * 2 / eta and abs(x2 - s2) < 0.1 * s2,
       f"toy eta={eta}: mean S={S:.4f} vs 2/eta={2/eta:.4f}; <x^2>={x2:.5f} vs sigma^2={s2:.5f}")
_, _, d_jump = toy_run_port(4000, 0.4, 8.0, 1e-3, False, 1, 1e-3, 1)
xs, ys, d_warm = toy_run_port(4000, 0.4, 8.0, 1e-3, True, 1600, 1e-3, 1)
ok(d_jump and not d_warm and abs(ys[-1000:].mean() - 5) < 0.05,
   f"sharp start y0=8, eta=0.4: no warmup diverges ({d_jump}); T=1600 warmup survives ({not d_warm}), final mean S={ys[-1000:].mean():.3f}")
for a0, phase in ((1.5, "lazy"), (3.0, "catapult"), (4.2, "divergent")):
    f, al, div = catapult_port(150, 1.0, a0, 1000, 1.0)
    fin = al[~np.isnan(al)][-1]
    good = {"lazy": (not div) and abs(fin - a0) < 0.01, "catapult": (not div) and fin < 2 and np.nanmax(f**2) > 100,
            "divergent": div}[phase]
    ok(good, f"catapult a0={a0}: {phase} (final eta*lambda={fin:.3g}, diverged={div}, peak f^2={np.nanmax(f**2):.3g})")

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
raise SystemExit(1 if fails else 0)
