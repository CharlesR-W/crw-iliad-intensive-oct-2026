"""Check the curves drawn by ../mode_explorer.html against independent solutions.

Runs the page's <script id="mathlib"> block in node, then compares:
  1. N=2, d=0: the page's a(t) and half-time vs the logistic closed form;
  2. N=2, d>0: the page's a(t) vs the closed-form t(a) for tau a' = sqrt(4a^2+d^2)(s-a);
  3. any N, d: the page's a(t) and kernel k(t) vs scipy solve_ivp on the per-layer system;
  4. balanced N>=3: the page vs scipy on tau a' = N a^(2-2/N)(s-a) (the reduced ODE);
  5. initial conditions: prod w = a0 and w_{j+1}^2 - w_j^2 = d.
Run: python td/assets/interactive/checks/mode_explorer_check.py
"""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

HTML = Path(__file__).resolve().parent.parent / "mode_explorer.html"
lib = re.search(r'<script id="mathlib">([\s\S]*?)</script>', HTML.read_text()).group(1)

cases = [  # (N, s, a0, d)
    (2, 3.0, 1e-3, 0.0), (2, 0.75, 1e-4, 0.0), (2, 2.0, 1e-2, 0.5), (2, 1.0, 1e-4, 10.0),
    (2, 3.0, 1e-4, 100.0), (2, 1.0, 7.9, 0.0), (3, 2.0, 1e-3, 0.0), (4, 3.0, 1e-2, 0.0),
    (6, 1.0, 1e-3, 0.0), (3, 1.5, 1e-3, 2.0), (4, 1.0, 1e-4, 0.3), (1, 2.0, 1e-3, 0.0),
]
js = lib + r"""
const L = module.exports, cases = %s, out = [];
for (const [N, s, a0, d] of cases) {
  const r = L.integrateMode(N, s, a0, d, 1e7);
  const T = r.tEnd, ts = [];
  for (let i = 1; i <= 12; i++) ts.push(T * i / 12);
  out.push({N, s, a0, d, T, w0: L.initLayers(N, a0, d), ts,
            a: ts.map(t => L.interp(r.t, r.a, t)), k: ts.map(t => L.interp(r.t, r.k, t)),
            th: L.halfTime(r, s), nsteps: r.t.length});
}
console.log(JSON.stringify(out));
""" % json.dumps(cases)
js = "const module = {exports: {}};\n" + js
res = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)

def per_layer(t, w, s):
    a = np.prod(w)
    return np.array([np.prod(np.delete(w, j)) for j in range(len(w))]) * (s - a)

worst = {"a_vs_scipy": 0.0, "k_vs_scipy": 0.0, "closed": 0.0, "half": 0.0, "reduced": 0.0, "init": 0.0}
for r in res:
    N, s, a0, d, ts = r["N"], r["s"], r["a0"], r["d"], np.array(r["ts"])
    w0 = np.array(r["w0"])
    worst["init"] = max(worst["init"], abs(np.prod(w0) / a0 - 1), *(abs(np.diff(w0**2) - d)) if N > 1 else [0])
    sol = solve_ivp(per_layer, [0, ts[-1]], w0, args=(s,), method="DOP853", rtol=1e-11, atol=1e-14, dense_output=True)
    W = sol.sol(ts)
    a_ref = np.prod(W, 0)
    k_ref = np.array([sum(np.prod(np.delete(W[:, i], j))**2 for j in range(N)) for i in range(len(ts))])
    ea = np.max(np.abs(np.array(r["a"]) - a_ref)) / s
    ek = np.max(np.abs(np.array(r["k"]) / k_ref - 1))
    worst["a_vs_scipy"] = max(worst["a_vs_scipy"], ea); worst["k_vs_scipy"] = max(worst["k_vs_scipy"], ek)
    line = f"N={N} s={s} a0={a0:g} d={d:g}: T={r['T']:.4g} steps={r['nsteps']} |a-scipy|/s={ea:.1e} |k/k_scipy-1|={ek:.1e}"
    if N == 2:
        a_js = np.array(r["a"])
        if d == 0:
            e = np.exp(2 * s * ts); ref = s * e / (e - 1 + s / a0)
            ec = np.max(np.abs(a_js - ref)) / s
            if a0 < s / 2:
                th_ref = np.log(s / a0 - 1) / (2 * s); eh = abs(r["th"] / th_ref - 1)
                worst["half"] = max(worst["half"], eh); line += f" half-time rel err={eh:.1e}"
        else:
            R = np.sqrt(4 * s * s + d * d)
            F = lambda x: np.log((4 * s * x + d * d + R * np.sqrt(4 * x * x + d * d)) / (2 * s - 2 * x)) / R
            m = np.abs(a_js - s) > 1e-6 * s
            ec = np.max(np.abs(F(a_js[m]) - F(a0) - ts[m]) * np.abs(s - a_js[m]) * np.sqrt(4 * a_js[m]**2 + d * d)) / s  # time error -> a error
        worst["closed"] = max(worst["closed"], ec); line += f" |a-closed|/s={ec:.1e}"
    if N >= 3 and d == 0:
        sol2 = solve_ivp(lambda t, a: N * np.abs(a)**(2 - 2 / N) * (s - a), [0, ts[-1]], [a0], method="DOP853", rtol=1e-11, atol=1e-14, dense_output=True)
        er = np.max(np.abs(np.array(r["a"]) - sol2.sol(ts)[0])) / s
        worst["reduced"] = max(worst["reduced"], er); line += f" |a-reduced ODE|/s={er:.1e}"
    print(line)
print("worst:", {k: f"{v:.1e}" for k, v in worst.items()})
tol = {"a_vs_scipy": 1e-4, "k_vs_scipy": 1e-3, "closed": 1e-4, "half": 1e-3, "reduced": 1e-4, "init": 1e-9}
bad = [k for k in tol if worst[k] > tol[k]]
print("FAIL: " + ", ".join(bad) if bad else "all mode_explorer checks passed")
sys.exit(1 if bad else 0)
