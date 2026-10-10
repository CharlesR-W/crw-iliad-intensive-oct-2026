# L1: Linear networks (outline and speaker notes)

**Length:** about 18 min in full. A compressed path of about 10 min is at the end.
**Hands off to:** `td1_linear_networks` (NB1).
**Purpose:** give everyone the mode picture: what changes between a shallow and a two-layer linear model, why modes are learned as sigmoids ordered by strength, what sets the plateau, and which lessons from this model are derived and which are planted. NB2 refers back to this picture, and `td2_7` restates it.

**Notation on the board (matches NB1).** Whitened inputs, $\mathbb E[xx^\top]=I$. Target correlation $\Sigma=\mathbb E[yx^\top]=USV^\top$, singular values $s_\alpha$. End-to-end map $W=W_2W_1$, error $E=\Sigma-W$, mode strength $a_\alpha=u_\alpha^\top Wv_\alpha$. **Time:** $\tau=1/\eta$ with $t$ counting steps, so gradient flow is $\tau\dot\theta=-\nabla L$ (Saxe's convention). If anyone asks: NB2 uses $t=k\eta$, under which $\tau$ simply drops out ($\dot a=2a(s-a)$, as `td2_7` writes it).

Each segment lists: time, the claim or picture, at most two board equations, what to show, and where the notebook picks it up. Nothing here is derived on the board. The derivations are NB1's exercises.

---

## 0. Why study a model that can't represent anything (1 min)

**Claim.** A deep linear network computes one matrix, so depth adds no expressivity. But its gradient dynamics are nonlinear, and it is the simplest multilayer model where gradient descent can be solved exactly. Several things seen in real training show up here in closed form: long plateaus followed by sudden drops, the strongest structure learned first, saddle-to-saddle trajectories, and a tangent kernel that either moves or stays frozen depending on the initialization.

**Speaker notes.**

- Say up front that everything is a statement about this model (square loss, linear net, whitened inputs). How far it transfers is the last segment's topic, and NB1 §6 is devoted to it.
- *(Room for the instructor's own framing.)*

---

## 1. Shallow versus two-layer: same loss, different geometry (4 min)

**Claim.** With whitened inputs, the loss depends on the data only through $\Sigma$, and it is the same function of $W$ for both models. Factorizing $W=W_2W_1$ changes the geometry in which $W$ descends. A rescaling symmetry gives a conserved imbalance $D=W_2^\top W_2-W_1W_1^\top$. From aligned initial conditions the modes decouple, and each one obeys a scalar ODE.

**Board.**
$$
\text{shallow: }\ \tau\dot a=s-a\qquad\qquad\text{two-layer: }\ \tau\dot a=\sqrt{4a^2+d^2}\,(s-a)
$$
(NB1 eq. 1.4. Here $d=c^2-b^2$ is the mode's conserved imbalance.)

**Picture to describe.** One axis per mode, so the problem is a stack of independent one-dimensional problems. The only thing distinguishing the two models is the prefactor, which is how fast that mode is allowed to move. Say that this prefactor is the tangent kernel restricted to the mode (segment 3).

**Show.** `assets/interactive/mode_explorer.html`, preset **rich (small balanced init)**, with **show shallow model (N = 1) dashed** ticked. The dashed shallow curves are plain exponentials that all start moving at once. The solid two-layer curves sit flat, then jump.

**Speaker notes.**

- Linear regression has no plateaus. Every plateau in the two-layer model comes from the prefactor being small at the start.
- Aligned initial conditions are an assumption in §1.3. §1.5 shows that small random weights align by themselves before the loss moves (input and output weights grow fastest along $(v_1,u_1)$). The first mode's alignment is derived; that later modes follow is checked numerically, not proved. Say this in one sentence. The NB1 code cell "watching alignment happen before the loss moves" shows it.
- **Picked up in:** NB1 §1.1 to §1.5 (Exercises 1.1 to 1.6).

---

## 2. Sigmoids ordered by strength, and the plateau (4 min)

**Claim.** Small, balanced initialization ($d=0$, small $a_0>0$) gives a logistic curve per mode. The plateau lasts about $(\tau/2s)\ln(s/a_0)$, and the rise takes a time of order $\tau/s$. With a common small $a_0$, modes finish their plateaus in order of decreasing $s$. The loss is a staircase.

**Board.**
$$
\tau\dot a=2a(s-a),\qquad a(t)=\frac{s\,e^{2st/\tau}}{e^{2st/\tau}-1+s/a_0},\qquad t_{1/2}=\frac{\tau}{2s}\ln\Big(\frac{s}{a_0}-1\Big)\approx\frac{\tau}{2s}\ln\frac{s}{a_0}
$$

**Picture to describe.** Plot $a/s$ against $t/\tau$. Draw three S-curves with $s_1>s_2>s_3$, each flat near zero and then rising, with the strongest rising first. Below them, the loss as a staircase with one drop per mode.

**Show.** The same explorer. Move the **initial $a_0$ (log₁₀)** slider: plateaus lengthen only logarithmically. Move the **ratio** slider: when the singular values are close, the steps merge into one drop.

**Speaker notes.**

- The ordering is by $s$, and the dependence on $a_0$ is only logarithmic at depth 2. Visible separate steps need real gaps in the spectrum: NB1 Exercise 6.1 gives the criterion $\delta s/s\gg 4.4/\ln(1/a_0)$, which matters in segment 4.
- *Depth, if time allows (30 s):* for balanced depth $N\ge3$, $\tau\dot a=Na^{2-2/N}(s-a)$. The origin becomes a degenerate saddle and the plateau grows as a power, $\propto a_0^{-(1-2/N)}$, instead of a log. Preset **deep (N = 4)**. This is NB1 §2, a side branch nothing later depends on.
- *Landscape, if asked:* every critical point is a partial SVD reconstruction, and there are no spurious minima. That says nothing about how long training spends near each saddle (NB1 §3.4). It is also a side branch.
- For random small initialization, NB1 writes the half-time as $(\tau/2s)\ln(4s/\Delta)$. That is the same formula with an effective seed $a_0=\Delta/4$, so it doesn't need a separate board line.
- **Picked up in:** NB1 §1.4 and Exercise 1.5 (code: integrate the mode ODEs against the closed forms); §2 for depth; §3 for the landscape.

---

## 3. The lazy/rich dial is imbalance, not scale (4 min)

**Claim.** How $W$ moves is a linear operator applied to the error, $\tau\dot W=K_t[E]$. $K_t$ is the linear network's NTK. On a mode it equals the prefactor from segment 1. If the kernel grows from near zero (small, balanced start), learning is rich: plateaus, sigmoids, ordering by $s$. If the kernel is large and barely moves (large imbalance), learning is lazy: an exponential at a common rate, with no plateaus and no ordering.

**Board.**
$$
\tau\dot W=W_2W_2^\top E+E\,W_1^\top W_1=:K_t[E]
$$
(NB1 eq. 4.1.) For a balanced aligned mode the kernel is $2a$, which grows as the mode is learned. For $|d|\gg s$ it is $\approx|d|$, which stays fixed.

**Picture to describe.** The explorer's second panel, the kernel per mode against time on a log scale. Rich curves climb by orders of magnitude. Lazy curves are flat lines.

**Show.** Presets **lazy (large imbalance)**, then **mixed (d between the s's)**, then **large balanced init (not lazy)**. The last one is the point of the segment.

**Speaker notes.**

- **The common trap.** "Large initialization gives lazy training" is true of large *Gaussian* initialization and false of large *balanced* initialization. In a balanced network the function is the square of the per-layer scale, so large balanced weights mean a large initial function. Removing the excess changes the kernel by order 100% (NB1 §5.3).
- Why Gaussian initialization is lazy: its input and output weight vectors in hidden space are nearly orthogonal. NB1's per-mode invariant $\Delta$ counts imbalance *and* misalignment, and for Gaussian weights of variance $\sigma^2$ at width $h$, $\Delta\approx2\sigma^2h$. The explorer's slider is the aligned special case $\Delta=|d|$. Mention $\Delta$ by name only, so students recognise it in §4.3.
- Mixed regime: modes with $s\gg\Delta/2$ are learned richly and finish first, while modes with $s\ll\Delta/2$ relax lazily. This matches Tu, Aranguri and Jacot's threshold for wide Gaussian initialization.
- Chizat, Oyallon and Bach's point, that laziness comes from scaling rather than width as such, is the general version. The imbalance version is specific to this model (NB1 §6 item 6).
- Pointers for later: general-network NTK and infinite width are the Physics of Deep Learning day (NB1 §7 links it). The same kernel reappears in `td2_6` as the Gauss–Newton matrix.
- **Picked up in:** NB1 §4 (the end-to-end flow, balanced kernels, the invariant $\Delta$) and §5 (rich, lazy, mixed), with the code cells "the invariant $\Delta$ for five initializations" and "stagewise versus exponential, and which kernels move".

---

## 4. Three heuristics, and what transfers (4 min)

**Claim.** Several widely quoted lessons were derived or popularized in this model. For each one, separate what follows from the mathematics from what the data or initialization put in. The aim is a fair accounting, not a debunking. NB1 §6 uses the same three headings for every item: *Derived*, *Planted*, *Transfer*.

No board equation. If useful, write the three headings as three columns and fill them in aloud.

**Heuristic 1: stagewise learning ordered by singular value (and semantic development).**

- *Derived:* from small initialization, modes are learned in order of decreasing $s$, with sigmoidal transitions. A shallow net shows no stages.
- *Planted:* small initialization; real gaps in the spectrum; whitened inputs, which is what makes the SVD of $\Sigma$ the right object. In Saxe–McClelland–Ganguli's semantic-development model, the tree-structured data are built so that coarse splits carry the larger singular values. "Coarse before fine" is then "larger $s$ before smaller $s$". The dynamics add the timing, the sigmoidal shape and the deep/shallow contrast.
- *Transfer:* incremental, roughly stagewise learning is seen in some nonlinear settings, but the "modes" are then not an SVD of the data.
- *Show:* NB1 Exercise 6.2 (code). Tree data against flat random data: the tree's right singular vectors are its splits, and the flat data's are meaningless mixtures. The loss curves show stages only when the hierarchy is strong.

**Heuristic 2: small initialization gives saddle plateaus and low-rank-first learning.**

- *Derived:* the trajectory passes near $\Sigma_{\{1\}},\Sigma_{\{1,2\}},\dots$ in order, and effective rank grows one step at a time. This is a joint property of initialization scale, factorized parametrization and gradient flow, not of the loss: the shallow model has the same loss and learns all modes at once.
- *Planted:* the small initialization itself. Also, for whitened full-rank regression the endpoint is $\Sigma$ regardless of the path, so "low-rank bias" appears only through early stopping or in underdetermined problems (matrix completion, sensing).
- *Transfer:* proved pieces are the first escape and the first greedy rank-one phase. The full sequence is conjectured in general. The nonlinear analogues are in NB1 §7's readings.

**Heuristic 3: imbalance between layers decides lazy versus rich.**

- *Derived:* large $\Delta$ freezes the kernel. Small $\Delta$ and small $a_0$ give the stagewise regime.
- *Planted:* "large" must mean large imbalance, not large weights (segment 3).
- *Transfer:* the scaling version is general. The imbalance version can differ in piecewise-linear networks.

**What does not transfer, said once.** The exact solution needs decoupled (aligned) modes, whitened inputs and linear features. Nonlinear nets are not deep linear nets. The claim that survives is that some of their *phenomenology* matches.

**Speaker notes.**

- The three heuristics above are NB1 §6 items 1 (with item 2, semantic development, folded in), 3 and 6. §6 has two more items for those who want them: item 4, orthogonal initialization and dynamical isometry (no small-kernel plateau, but half of random orthogonal starts in square nets cannot reach $\Sigma$ under gradient flow); and item 5, generalization dynamics (Advani–Saxe–Sompolinsky; the linear-least-squares parts are derived in the MDL day's WB2: early stopping as a spectral filter, Exercise 4.3, and double descent in the keystone).
- Present each *Derived* and *Planted* entry with the same weight. Students can weigh them.
- *(Room for the instructor's own view on which heuristics matter.)*
- **Picked up in:** NB1 §6, Exercises 6.1 to 6.3.

---

## 5. Hand-off (1 min)

- **NB1's short path (about 1 to 1.5 h, from its opening cell):** §1.1 to 1.5 with Exercise 1.4 and the §1 code; then §4.1 and §4.3; then §5.2 to 5.4 and the initialization-comparison cell; then items 1, 3 and 6 of §6. §2 (depth) and §3 (landscape) are side branches.
- **If you'd rather do exactly solvable theory all day:** NB1 §7, "the analytic escape hatch". It links the Physics of Deep Learning day's notebooks and the infinite-width readings.
- **Where this returns in NB2:** the mode picture in `td2_7` (what is learned first); the kernel $K_t$ in `td2_6` (the Gauss–Newton matrix is the NTK seen from parameter space); parametrization and muP in `td2_4` §4.5.

---

## Compressed path (about 10 min)

Use this if the guest lecture takes the morning slot. NB1 §1 is self-contained, so skipping L1 entirely also works. In that case, the §7.1 walkthrough of `td2_7` restates the mode picture.

| Segment | Keep | Drop |
|---|---|---|
| 0 | one sentence | the rest |
| 1 (2 min) | the board pair: shallow vs two-layer scalar ODE; explorer with the shallow curves dashed | conservation, alignment |
| 2 (3 min) | the logistic and $t_{1/2}$; strongest first; staircase | depth, landscape |
| 3 (3 min) | the kernel prefactor; presets **lazy** and **large balanced init (not lazy)** | $\Delta$, mixed regime |
| 4 (2 min) | one sentence per heuristic: *derived* vs *planted*; "NB1 §6 does this properly" | the examples |
| 5 | NB1's short path | — |
