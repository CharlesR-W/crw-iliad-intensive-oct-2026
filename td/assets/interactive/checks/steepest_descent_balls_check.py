#!/usr/bin/env python3
"""Check the math encoded in ../steepest_descent_balls.html.

Extracts the JavaScript between `// BEGIN MATH` and `// END MATH`, runs it under node on
random gradients and ellipse settings, and compares with an independent numpy computation:

1. trust-region step: brute-force minimum of <g, D> over 20k points on the boundary of each
   ball of radius eta (l2, linf, l1, Mahalanobis {D : D^T M D <= eta^2});
2. penalty step: brute-force minimum of <g, D> + ||D||^2 / (2 eta) over a grid of step lengths
   along every sampled boundary direction;
3. dual norms against closed forms; Mahalanobis ellipse semi-axes against (sqrt kappa, 1/sqrt kappa);
4. the near-tie flags (2% tolerance) for exact ties, a near tie and random cases.

Run: python td/assets/interactive/checks/steepest_descent_balls_check.py   (needs node; ~5 s)
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HTML = Path(__file__).resolve().parent.parent / "steepest_descent_balls.html"
src = HTML.read_text()
math_js = re.search(r"// BEGIN MATH.*?\n(.*?)// END MATH", src, re.S).group(1)

rng = np.random.default_rng(1)
cases = []
for _ in range(300):
    ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.1, 1.6)
    cases.append({"g": [rad * np.cos(ang), rad * np.sin(ang)], "kappa": float(rng.uniform(1, 6)),
                  "phi": float(rng.uniform(0, 180)), "eta": float(rng.uniform(0.2, 1.2))})
cases += [{"g": [0.7, 0.7], "kappa": 2.0, "phi": 10.0, "eta": 1.0},      # exact l1 tie
          {"g": [0.9, 0.0], "kappa": 2.0, "phi": 10.0, "eta": 1.0},      # exact linf tie
          {"g": [1.0, 0.995], "kappa": 2.0, "phi": 10.0, "eta": 1.0},    # near l1 tie, not linf
          {"g": [1.0, 0.5], "kappa": 2.0, "phi": 10.0, "eta": 1.0}]      # no tie

harness = math_js + """
const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = cases.map(c => {
  const M = mahalanobisM(c.kappa, c.phi), r = {M};
  for (const n of ["l2", "linf", "l1", "mah"]) {
    r[n] = {trust: step(c.g, n, M, c.eta, "trust"), pen: step(c.g, n, M, c.eta, "pen")};
  }
  return r;
});
console.log(JSON.stringify(out));
"""
res = json.loads(subprocess.run(["node", "-e", harness], input=json.dumps(cases), capture_output=True,
                                text=True, check=True).stdout)

t = np.linspace(0, 2 * np.pi, 20000, endpoint=False)
circ = np.stack([np.cos(t), np.sin(t)], 1)


def unit_boundary(norm, M):
    if norm == "l2":
        return circ
    if norm == "linf":
        return circ / np.abs(circ).max(1, keepdims=True)
    if norm == "l1":
        return circ / np.abs(circ).sum(1, keepdims=True)
    q = np.einsum("ni,ij,nj->n", circ, M, circ)
    return circ / np.sqrt(q)[:, None]


def dual(g, norm, M):
    return {"l2": np.linalg.norm(g), "linf": np.abs(g).sum(), "l1": np.abs(g).max(),
            "mah": np.sqrt(g @ np.linalg.solve(M, g))}[norm]


worst = 0.0
for c, r in zip(cases, res):
    g, eta = np.array(c["g"]), c["eta"]
    M = np.array(r["M"])
    # ellipse geometry
    p = np.deg2rad(c["phi"]); u = np.array([np.cos(p), np.sin(p)]); w = np.array([-u[1], u[0]])
    assert abs(u @ M @ u - 1 / c["kappa"]) < 1e-12 and abs(w @ M @ w - c["kappa"]) < 1e-12
    for norm in ("l2", "linf", "l1", "mah"):
        B = unit_boundary(norm, M)
        # trust region
        D = np.array(r[norm]["trust"]["delta"])
        best = (eta * B @ g).min()
        assert g @ D <= best + 1e-9, (norm, c)
        nrm = {"l2": np.linalg.norm(D), "linf": np.abs(D).max(), "l1": np.abs(D).sum(),
               "mah": np.sqrt(D @ M @ D)}[norm]
        assert abs(nrm - eta) < 1e-9, (norm, nrm)
        assert abs(r[norm]["trust"]["dual"] - dual(g, norm, M)) < 1e-9
        # penalty: min over directions of min over lengths k of k <g,b> + k^2/(2 eta)
        Dp = np.array(r[norm]["pen"]["delta"])
        gb = B @ g
        kstar = np.clip(-eta * gb, 0, None)          # optimal length along each unit direction b
        brute = (kstar * gb + kstar**2 / (2 * eta)).min()
        nrm_p = {"l2": np.linalg.norm(Dp), "linf": np.abs(Dp).max(), "l1": np.abs(Dp).sum(),
                 "mah": np.sqrt(Dp @ M @ Dp)}[norm]
        val = Dp @ g + nrm_p**2 / (2 * eta)
        assert val <= brute + 1e-9, (norm, val, brute)
        assert abs(val + eta * dual(g, norm, M) ** 2 / 2) < 1e-9
        worst = max(worst, val - brute)
# near-tie flags: exact ties, near ties inside the 2% tolerance, and clear non-ties
flags = [(r["l1"]["trust"]["nearTie"], r["linf"]["trust"]["nearTie"]) for r in res[-4:]]
assert flags[0] == (True, False)      # |g1| = |g2| exactly
assert flags[1] == (False, True)      # g2 = 0 exactly
assert flags[2] == (True, False)      # near l1 tie
assert flags[3] == (False, False)
for c, r in zip(cases[:-4], res[:-4]):
    g = np.abs(c["g"])
    assert r["linf"]["trust"]["nearTie"] == (g.min() <= 0.02 * g.max())
    assert r["l1"]["trust"]["nearTie"] == (abs(g[0] - g[1]) <= 0.02 * g.max())
# Mahalanobis penalty step equals -eta M^{-1} g ; l2 penalty step equals -eta g
for c, r in zip(cases, res):
    g, eta, M = np.array(c["g"]), c["eta"], np.array(r["M"])
    assert np.allclose(r["mah"]["pen"]["delta"], -eta * np.linalg.solve(M, g))
    assert np.allclose(r["l2"]["pen"]["delta"], -eta * g)
    assert np.allclose(r["linf"]["pen"]["delta"], -eta * np.abs(g).sum() * np.sign(g))
print(f"ok: {len(cases)} cases, 4 norms, trust-region and penalty steps match brute force "
      f"(largest penalty-objective gap {worst:.2e}, should be <= 0)")
