#!/usr/bin/env python3
"""Check the math encoded in ../filter_explorer.html (DA1 §6).

Extracts the JavaScript between `// BEGIN MATH` and `// END MATH`, runs it under node on a grid
of (s, eta, K, lambda), and compares with independent numpy computations:

1. GD filter: simulate the response recursion r_{t+1} = (1 - eta s) r_t - eta g with r_0 = 0,
   frozen curvature s and gradient g = 1, and compare -r_K with F_gd(s, eta, K), including s = 0,
   negative s and the unstable region eta s > 2;
2. SOURCE filter against (1 - exp(-eta K s)) / s (and eta K at s = 0), damped filter against
   1 / (s + lambda), IF against 1 / s;
3. F(0) = eta K for GD and SOURCE, 1 / lambda for the damped filter;
4. at s = 1 / (eta K): SOURCE = (1 - 1/e) eta K ~ 0.632 eta K, ridge with lambda = 1/(eta K) = 0.5 eta K;
5. GD -> 1/s as K grows when 0 < eta s < 2.

Run: python da/assets/interactive/checks/filter_explorer_check.py   (needs node; ~1 s)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HTML = Path(__file__).resolve().parent.parent / "filter_explorer.html"
math_js = re.search(r"// BEGIN MATH.*?\n(.*?)// END MATH", HTML.read_text(), re.S).group(1)

rng = np.random.default_rng(0)
cases = []
for _ in range(400):
    eta = 10 ** rng.uniform(-3, 0)
    K = int(round(10 ** rng.uniform(0, 3)))
    T = eta * K
    s = float(rng.choice([0.0, 1 / T, rng.uniform(-1, 6) / T, rng.uniform(0, 2.5) / eta, 1e-14]))
    cases.append({"s": s, "eta": eta, "K": K, "lam": 10 ** rng.uniform(-3, 2)})

harness = math_js + """
const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));
console.log(JSON.stringify(cases.map(c => ({
  ifv: F_if(c.s), damp: F_damped(c.s, c.lam), gd: F_gd(c.s, c.eta, c.K), src: F_source(c.s, c.eta, c.K)}))));
"""
res = json.loads(subprocess.run(["node", "-e", harness], input=json.dumps(cases), capture_output=True,
                                text=True, check=True).stdout)

fails = []


def close(a, b, rtol=1e-9, atol=0.0):
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


for c, r in zip(cases, res):
    s, eta, K, lam = c["s"], c["eta"], c["K"], c["lam"]
    rr = 0.0
    for _ in range(K):
        rr = (1 - eta * s) * rr - eta
    gd_ref = -rr
    src_ref = eta * K if s == 0 else -np.expm1(-eta * K * s) / s
    if not close(r["gd"], gd_ref, 1e-8, 1e-12 * abs(eta * K)):
        fails.append(("gd", c, r["gd"], gd_ref))
    if not close(r["src"], src_ref, 1e-9):
        fails.append(("src", c, r["src"], src_ref))
    if s + lam != 0 and not close(r["damp"], 1 / (s + lam), 1e-12):
        fails.append(("damp", c, r["damp"], 1 / (s + lam)))
    for k in ("gd", "src", "damp") + (("ifv",) if s != 0 else ()):
        if r[k] is None and not (k == "damp" and s + lam == 0):   # JSON null = NaN or infinity
            fails.append((k + " not finite", c, r[k], None))
    if s != 0 and r["ifv"] is not None and not close(r["ifv"], 1 / s, 1e-12):
        fails.append(("if", c, r["ifv"], 1 / s))

# F(0), the crossover and the K -> infinity limit, through the same JS functions
extra = math_js + """
const T = 0.1 * 10, out = {};
out.gd0 = F_gd(0, 0.1, 10); out.src0 = F_source(0, 0.1, 10); out.damp0 = F_damped(0, 0.25);
out.src_cross = F_source(1 / T, 0.1, 10) / T; out.ridge_cross = F_damped(1 / T, 1 / T) / T;
out.gd_inf = F_gd(5, 0.1, 5000); out.gd_unstable = F_gd(25, 0.1, 10);
console.log(JSON.stringify(out));
"""
e = json.loads(subprocess.run(["node", "-e", extra], capture_output=True, text=True, check=True).stdout)
checks = [
    ("F_gd(0) = eta K", close(e["gd0"], 1.0)),
    ("F_source(0) = eta K", close(e["src0"], 1.0)),
    ("F_damped(0) = 1/lambda", close(e["damp0"], 4.0)),
    ("SOURCE at s = 1/(eta K) is (1 - 1/e) eta K", close(e["src_cross"], 1 - np.exp(-1))),
    ("ridge with lambda = 1/(eta K) at s = 1/(eta K) is 0.5 eta K", close(e["ridge_cross"], 0.5)),
    ("GD -> 1/s for 0 < eta s < 2 as K grows", close(e["gd_inf"], 1 / 5)),
    ("GD for eta s > 2 is not 1/s (overshoot)", not close(e["gd_unstable"], 1 / 25, 0.1)),
]
for name, ok in checks:
    print(f"[{'ok' if ok else 'FAIL'}] {name}")
print(f"[{'ok' if not fails else 'FAIL'}] {len(cases)} random cases: JS filters vs numpy (GD by simulating the recursion)")
for f in fails[:10]:
    print("   ", f)
sys.exit(1 if fails or not all(ok for _, ok in checks) else 0)
