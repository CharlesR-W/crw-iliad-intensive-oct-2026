# WB6 new exercises: technical check

Read-only audit of `notebooks/build_wb6.py` §§9–11, 2026-10-08, while the editorial owner is active. No source edits. Algebra independently derived; numerical examples recomputed with Python's standard library. This is a focused exercise check, not certification of the whole reading catalogue.

## 9.1: quadratic overlap — correct with assumptions now present

The current source already says $c_1(0)\ne0$ and some bulk $c_i(0)\ne0$ in part (c). With $0<\mu\le\Lambda/2$ and $0<\eta<2/\Lambda$, the threshold $2/(\Lambda+\mu)<\eta<2/\Lambda$ is correct. At equality the overlap stays at its initial value, strictly between zero and one. The formulas for gradient overlap and per-mode one-step decrease are correct when the gradient is nonzero; the overlap is undefined at a stationary point.

Numerical checks:

| step | overlap | outlier decrease | bulk decrease | bulk fraction |
|---|---:|---:|---:|---:|
| 0 | 0.90917356 | 0.99500000 | 19.68229301 | 0.95187958 |
| 100 | 0.98679221 | 0.13330978 | 0.35330805 | 0.72604831 |

The crossover of bulk and outlier decrease occurs at $t\approx148.489$, consistent with “about 150 steps.” No substantive correction needed. Optionally specify $g\ne0$ for the overlap definition and state the equality case in the answer.

## 10.1: correct identities, but narrow the curvature conclusion

Parts (a)–(c), norm growth, and the first-order directional step are algebraically correct for a twice-differentiable scale-invariant loss and $w\ne0$. However, radial scaling at fixed direction does not prove monotonically decreasing curvature during training, since the direction also changes.

Replace the setup's first sentence with:

> Assume the loss is twice differentiable near a nonzero incoming weight vector $w$. For batch normalization using the current batch's statistics, with nonzero preactivation variance and with the denominator's $\varepsilon$ set to zero, positive rescaling of $w$ leaves the normalized output unchanged. This need not hold with frozen inference statistics or with an added weight penalty.

The existing $L(\alpha w)=L(w)$ assumption then supports the exercises. In (d), replace “what happens to the curvature ... as training proceeds?” with:

> Using (b), separate the effect of changing $\|w\|$ from that of changing its unit direction $u=w/\|w\|$. What effective step size multiplies the directional gradient?

Replace “As the norm grows, the curvature a fixed $\eta$ meets shrinks” with:

> Exactly, $H(w_t)=\|w_t\|^{-2}H(u_t)$. Norm growth lowers this radial factor and the effective directional step size $\eta/\|w_t\|^2$; it does not guarantee decreasing Hessian eigenvalues along training because $H(u_t)$ also changes.

The distinction between batch statistics and fixed inference statistics follows directly from the BN transform in [Ioffe & Szegedy](https://arxiv.org/abs/1502.03167). The effective-rate connection is also the subject of [van Laarhoven](https://arxiv.org/abs/1706.05350).

## 11.1: support proof correct; qualify genericity

The determinant-polynomial argument is correct. Setting non-permutation entries to zero is a legitimate witness that the polynomial is nonzero; it proves that almost every continuous assignment on the permitted support is invertible, including assignments with all permitted entries nonzero.

The definition should say **“has rank at most $r$ if it can be written $\Delta W=AB^T$”**. LoRA parameterization permits lower rank as well.

Part (c)'s inference about actual training updates is conditional, not automatic. Replace its final conclusion with:

> Thus sparsity alone does not imply low rank. A support with a perfect matching and generic continuous coefficients is full-rank almost surely; learned supports and coefficients can be structured or correlated, so the empirical rank remains a separate measurement.

If retaining the random-support threshold sentence, explicitly specify independent Bernoulli support and sufficiently large $n$, rather than the unquantified “realistic $n$.” The proof in (b) alone is sufficient for the lesson and does not establish that trained updates follow that random model.

## 11.2: boundary, normal-number scope, and rounding hypotheses

The paper indeed uses an absolute $10^{-5}$ tolerance on bfloat16 checkpoint values: [Mukherjee et al., §2.2](https://arxiv.org/html/2505.11711). The [RISC-V BF16 format description](https://github.com/riscv/riscv-bfloat16/blob/main/doc/riscv-bfloat16-format.adoc) specifies normal/subnormal formats; the normal significand has seven stored fraction bits.

1. **Part (a):** specify finite, nonzero **normal** values and adjacent representable values within one magnitude binade. The spacing inside $[2^e,2^{e+1})$ is $2^{e-7}$, but at an exact power of two the neighbor toward zero is only $2^{e-8}$ away (except the smallest-normal boundary). Subnormal spacing is constant, $2^{-133}$, if subnormals are supported rather than flushed to zero.
2. **Part (b):** the strict cutoff is wrong at $|w|=2^{-9}$. Its next neighbor toward zero is $2^{-17}=7.62939453125\times10^{-6}$ away, within tolerance; its neighbor away from zero is $2^{-16}=1.52587890625\times10^{-5}$ away, outside tolerance. Replace the conclusion with: **“For a finite representable starting value, some distinct stored value can lie within tolerance iff $|w|\le2^{-9}$. At the boundary only the neighboring value toward zero qualifies. For $|w|>2^{-9}$, unchanged under this tolerance means identical stored values.”** Tiny values and zero also have distinct representable neighbors inside the tolerance.
3. **Part (c):** the half-spacing threshold assumes the initial high-precision value is exactly the representable value $w$. State that assumption; otherwise the distance from the initial master value to a rounding boundary depends on its position within the bin. Also, for $|w|\le2^{-9}$ changing the stored value does not necessarily exceed the absolute tolerance, so restrict the half-spacing argument to larger magnitudes and exclude power-of-two boundaries. At magnitude approximately $0.02$, use the exactly representable $w=0.02001953125$ (BF16 rounding of $0.02$); half-spacing is $2^{-14}=6.103515625\times10^{-5}$, about $0.3049\%$ of $|w|$. The printed approximate numerical value is otherwise correct.
4. Replace the universal “unchanged means moved by less than about 0.3%” with **“In this example, starting at the center of the rounding bin, a net movement below about 0.3% leaves the stored value unchanged. This does not imply zero gradient or no movement during training.”** Part (d) should similarly say **“below the relevant rounding/tolerance threshold”**, not impose 0.3% on every parameter magnitude. Per-step rounding and final-only rounding are correctly distinguished in the current text.

All corrections above concern mathematical scope and exactness, not design or topic selection. Final source recheck remains necessary after integration.

## Targeted integration recheck

Re-read the owner's completed §§9–11 against the findings above. The substantive corrections are present: nonzero projections and equality case in 9.1; BN batch-statistic assumptions and radial-versus-directional Hessian distinction in 10.1; rank-at-most definition and qualified genericity in 11.1; normal-number scope, inclusive $2^{-9}$ cutoff, representable initial master weight, and corrected $0.02$ example in 11.2. The worked numerical values remain correct.

One small residual wording issue remains in **11.2(d)**: the parenthetical “about 0.2–0.4% of their size” again applies the relative rounding estimate to all parameters, although (c) carefully restricted it to values above the tolerance cutoff and away from powers of two. Replace **“less than the rounding threshold (about 0.2–0.4% of their size)”** with **“less than their rounding or absolute-tolerance threshold”**. This is the same scope correction requested above, not a new audit finding. No other blocking issue remains from this focused audit. No builder source was edited during this recheck.

Root applied the exact residual 11.2(d) replacement after recheck.
