#!/usr/bin/env python3
"""Build WB2: How Deep Learning Changed Statistical Learning Theory.

Shape (2026-10-09 trail restructure): a keystone section, then an annotated trail.
Each section opens with a one-sentence purpose and a short rated reading list,
then `### Exercises: ...` subsections whose exercises (numbered k.m) state
their own notation, each followed by its folded solution.

Cells live in per-section modules:
  wb2_keystone.py        opening and route; §1 keystone: vacuous bounds (Zhang et al.,
                         Chan's posts) and double descent in least squares
  wb2_classical.py       §2 reference cases (gap, excess risk, OLS, bias–variance);
                         §3 no free lunch, universal approximation, VC dimension
  wb2_regularization.py  §4 ridge as prior / constraint / spectral filter, dropout, flatness
  wb2_interpolation.py   §5 kernel ridge and the ridgeless limit, benign overfitting
  wb2_pacbayes.py        §6 PAC-Bayes
  wb2_bayes_sgd.py       §7 Bayes and MLE generalization error; §8 gradient noise vs
                         curvature, SGD vs posterior, fantastic generalization measures
  wb2_ib.py              §9 the information-bottleneck debate
The pre-restructure modules (wb2_s*.py, wb2_new_*.py) are in _pre_trail_2026-10-09/.
Numerical checks are in checks/ (they use the old exercise numbering).
"""
from nbkit import run
import wb2_keystone
import wb2_classical
import wb2_regularization
import wb2_interpolation
import wb2_pacbayes
import wb2_bayes_sgd
import wb2_ib

CELLS = (
    wb2_keystone.CELLS
    + wb2_classical.CELLS
    + wb2_regularization.CELLS
    + wb2_interpolation.CELLS
    + wb2_pacbayes.CELLS
    + wb2_bayes_sgd.CELLS
    + wb2_ib.CELLS
)

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb2", "02_statistical_learning"))
