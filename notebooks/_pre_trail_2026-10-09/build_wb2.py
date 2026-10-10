#!/usr/bin/env python3
"""Build WB2 (statistical learning).

Cells live in per-section modules:
  wb2_s1.py   opening, §1.A–1.D (risks, least squares, bias–variance, regularization)
  wb2_s1e.py  §1.E kernels
  wb2_s1f.py  §1.F information bottleneck and rate–distortion
  wb2_s2.py   §2 theorems (NFL, UAT, VC, Zhang et al., PAC-Bayes)
  wb2_s3.py   §3 overfitting and double descent
  wb2_s4.py   §4 Bayes, MLE, C vs H, generalization measures
Numerical checks are in checks/.
"""
from nbkit import run
import wb2_s1 as s1
import wb2_s1e as s1e
import wb2_s1f as s1f
import wb2_s2 as s2
import wb2_s3 as s3
import wb2_s4 as s4

CELLS = (
    s1.OPENING
    + s1.S1ABC
    + s1.REG_OPENER + s1.REG_PRIOR + s1.REG_L1 + s1.REG_FILTER + s1.REG_NOISE + s1.REG_DROPOUT + s1.REG_FLAT
    + s1e.KERNELS
    + s1f.IB_OPENER + s1f.IB_RD + s1f.IB_PLANE
    + s2.S2_AB + s2.S2_VC + s2.S2_C2 + s2.S2_D + s2.S2_PACBAYES
    + s3.S3_ABC + s3.S3_BENIGN + s3.S3_EF
    + s4.S4
)

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb2", "02_statistical_learning"))
