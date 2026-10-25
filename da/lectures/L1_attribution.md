# L1: What did this training example do? (outline and speaker notes)

**Length:** about 40 min, 61 slides; many are 20-second slides (questions, takeaways, single equations). A compressed path of about 25 min is at the end.
**Slides:** [`L1_attribution.html`](L1_attribution.html). Arrows or space step through slides and build-ups, N shows the speaker notes on the slide, S opens reveal.js's speaker view, Esc the overview. For a PDF, open the page with `?print-pdf` and print from Chrome or Chromium. Figures are made by `assets/L1/make_figures.py`; the formulas and toy numbers are checked in `../checks/L1_checks.py`.
**Slide rules.** Every slide is one of: one equation plus one sentence; one figure plus one sentence; at most three bullets of at most twelve words; a question. Each section opens with a question slide and ends with a one-line takeaway. Numbers, citations, caveats and vocabulary are in the speaker notes (and below).
**Comes after:** nothing; this lecture opens the day. **Hands off to:** DA1 (`da1_core`) in the morning, then L2 and DA2 (`da2_methods`) after lunch.
**Purpose:** give everyone the question, the formalism (data weights $\beta\to\pi(\beta)\to\phi$), and the classical influence function derived by the implicit function theorem and tested against exact deletion in least squares. Then what each fix for networks changes (damping is a proximal objective, Gauss–Newton, the PBRF), and the habit of asking which $\pi$, which target, which check. DA1 develops each segment; the lecture's job is the question, the formalism, and the habit of asking "which question does this method answer?".

**Notation on the board (matches DA1 §2).** Training examples $z_i$, $i=1,\dots,n$, with per-example losses $L_i(w)$. Data weights $\beta_i$ and the **summed** objective $L(w;\beta)=\sum_i\beta_iL_i(w)$; $\beta=\mathbf 1$ is the real run. Observable $\phi(w)$ (for example a test loss), "the trained model" $\pi(\beta)$. Influence $\mathcal I_i=\partial\phi(\pi(\beta))/\partial\beta_i$ at $\beta=\mathbf 1$; deletion is $\beta_i:1\to0$, so it changes $\phi$ by about $-\mathcal I_i$. A helpful example has negative influence on a test loss. Hessian $H=\nabla^2_wL(w^*;\mathbf 1)$, gradients $g_i=\nabla L_i(w^*)$ and $g_\phi=\nabla\phi(w^*)$. For §3: least squares with $A=\sum_jx_jx_j^\top$, residual $r_j=y_j-x_j^\top w^*$, leverage $h_j=x_j^\top A^{-1}x_j$. For §4: outputs $f_i(w)$, Jacobian $J_i$, loss Hessian in the outputs $H_{y,i}$, Gauss–Newton $G=\sum_iJ_i^\top H_{y,i}J_i$, damping $\lambda$, checkpoint $w^s$.

Slide-level fixes from a pedagogy critique moved several items (non-transitivity, the eigenbasis slide, the $G\neq H$ example, Bae's five gaps) into speaker notes.

Each segment lists time, slides (one line each, as shown), the board, speaker notes, and where the notebooks pick it up. Links are relative to `da/lectures/`.

---

## 0. Opening (2 min), slides 1–5

1. **Title:** *"What did this training example do?"* From a causal question to one formula, and what that formula actually answers.
2. **Question:** *Which training examples made the model do this?*
3. **Bullets (builds), "Which pretraining texts make this response likely?":** query, a dialogue in which an AI assistant says it does not consent to being shut down. / 810M model: top sequences share words with the query ("continue", "existing"), but not its meaning. / 52B model: top sequences are about wanting to keep existing: an AI pleading with its crew; a person surviving in a desert.
4. **Figure** (diagram, builds), "Causal analysis of a training run": interpretability (input, layers, output) against data attribution (training data, optimizer steps, weights, behaviour). *"Ablating a layer costs a forward pass. Ablating an example costs a training run."*
5. **Bullets (builds):** the plan, six lines: counterfactuals are subtle; data weights make deletion differentiable; derive the influence function and test it on regression; what breaks for networks and what the fixes answer; how to check an attribution; another $\pi$, the training run, and a map of the afternoon.

**Speaker notes.**
- About 40 minutes, 61 slides. Everything derived here is derived again, more slowly, in DA1.
- Slide 2: let it sit. Everyone has an intuition for the answer; the lecture is about why the intuition needs a formalism.
- Slide 3: these scores concern the response's likelihood under smaller pretrained models, not why some assistant chose it. Grosse et al. (2023), Fig. 1 and §5.3, influence functions (EK-FAC, DA2 stop 1) on pretrained models of two sizes. The influential sequences get more abstract with scale. This is the kind of answer people want ("this behaviour came from imitating these kinds of documents") and exactly the kind of claim this morning teaches you to check. Do not over-read it: Grosse et al. present it as consistent with role-play by imitation, at a high level of abstraction. Nobody retrained the 52B model without those documents.
- Slide 4: mechanistic interpretability asks which components of a forward pass caused an output, and tests it by ablating, many times. Data attribution asks which examples of a training run caused a behaviour of the final model. Retraining costs as much as training, so attribution can rarely afford even one ablation; most methods *predict* retraining without doing it, and all the conceptual difficulties of causal reasoning come along.
- Slide 5: parts 3 and 4 are the derivations and the bulk of the time. Parts 5 and 6 are short; DA1 §5 and §6 have you do them.
- **Picked up in:** [DA1 §1](../da1_core.html#da1-1), [DA2 stop 1](../da2_methods.html#da2-stop-1) (the shutdown example).

---

## 1. Counterfactuals (3 min), slides 6–9

6. **Question:** *"Example $i$ caused it if deleting $i$ and retraining changes it."* What could go wrong?
7. **Bullets (builds), "Two causes, each sufficient":** the croissant table. Premium butter and premium yeast give fluffy; standard butter with premium yeast, and premium butter with standard yeast, are still fluffy; both standard gives flat. *"Leave-one-out credits neither. In training: a behaviour taught by many redundant examples."*
8. **Equation (builds), Shapley values:** $\varphi_i=\mathbb E_{\text{random order}}[v(S\cup\{i\})-v(S)]$, $S$ the examples before $i$. *"Croissants: butter $\tfrac12$, yeast $\tfrac12$. The price: each $v(S)$ is a retraining, for up to $2^{n-1}$ subsets."*
9. **Takeaway:** *"The influence of example $i$" is ambiguous until you say which counterfactual.*

**Speaker notes.**
- Slide 6 is leave-one-out (LOO): Lewis's counterfactual dependence, applied to training data. Ask the room for a failure before showing slide 7.
- The bakery: premium butter makes more steam, premium yeast makes more gas, and either alone is enough for fluffy croissants. Downgrading only one changes nothing, so LOO says neither caused the fluffiness, though both did.
- Non-transitivity (cut from the slides; tell it if there is time): a boulder falls toward a hiker, she ducks, she survives. Ducking depends on the boulder and survival on ducking, but survival does not depend on the boulder: without it she survives anyway. Dependence does not chain (Ned Hall's example, as told by Hitchcock 2001; Mueller 2024 §3). In training: an example can be necessary for a feature that a capability uses, yet deleting it changes no output.
- Shapley: $v(S)$ is the behaviour of a model trained on subset $S$. Shuffle the $n$ examples uniformly at random and add them one by one; $\varphi_i$ is the expected jump in $v$ when $i$ arrives. Equivalently (DA1 §1), $\varphi_i=\frac1n\sum_{S\subseteq[n]\setminus\{i\}}\binom{n-1}{|S|}^{-1}[v(S\cup\{i\})-v(S)]$. In the bakery $v(S)=1$ if $S$ contains butter or yeast; LOO gives 0 to everything, Shapley gives butter and yeast ½ each, flour and sugar 0.
- Shapley attributes over finite coalitions: it is not the derivative at the full training set that the rest of the day computes. Practical versions sample permutations (Ghorbani & Zou 2019). It still splits credit between sufficient causes, where one could argue each deserves all of it.
- **Picked up in:** [DA1 §1](../da1_core.html#da1-1) (Mueller 2024 is the reading, §4; written about interpretability, and the same failures apply).

---

## 2. The formalism: data weights and targets (4 min), slides 10–16

10. **Question:** *How do you make "delete example $i$" something you can differentiate?*
11. **Equation:** $L(w;\beta)=\sum_{i=1}^n\beta_iL_i(w)$. *"$\beta=\mathbf 1$ is the real run; $\beta_i:1\to0$ deletes example $i$."*
12. **Equation:** $\beta\xrightarrow{\pi}\pi(\beta)\xrightarrow{\phi}\phi(\pi(\beta))$. *"$\pi$, 'the trained model', is a modelling choice. $\phi$ is the behaviour: a test loss, a logit."*
13. **Equation:** $\mathcal I_i=\partial\phi(\pi(\beta))/\partial\beta_i\,|_{\beta=\mathbf 1}$. *"Deleting $i$ changes $\phi$ by about $-\mathcal I_i$. On a test loss, a helpful example has $\mathcal I_i<0$."*
14. **Figure** `cube.png`: the cube of data weights for three examples: the real run at the all-ones corner, a derivative arrow, the leave-one-out corner one edge away, a subset-deletion corner two edges away. *"The derivative; leave-one-out; deleting a large random subset."*
15. **Figure** `response.png`: slope of a regression fit as one point's weight goes from 1 to 0: the exact curve, its tangent at 1, the two endpoints. *"One point of a 15-point regression: the tangent predicts a change 5.7× too small."*
16. **Takeaway:** *Influence is the derivative of a behaviour in one example's weight, through a chosen $\pi$.*

**Speaker notes.**
- Slide 10: deletion is discrete; calculus needs a continuous knob.
- Slide 11: $w$ are the parameters, $L_i$ the loss on training example $z_i$. Note the **summed** loss, not the mean: curvatures and damping constants are in summed-loss units all day. For an argmin it doesn't matter (the minimizer of $cL$ is that of $L$); for SGD we write the $1/b$ explicitly.
- Slide 12: $\phi(w)$ is a scalar observable of the trained parameters; we also call it (or its test input) the query. $\pi(\beta)$ is what training on weights $\beta$ produces. Is it the argmin? This seed's last iterate? A posterior? Methods differ mainly in this choice.
- Slide 13, conventions said once: summed loss; upweighting derivative; deletion is $\beta_i:1\to0$; helpful has negative influence on a test loss. Koh and Liang upweight by $\varepsilon$ on the mean loss, so their $\mathcal I_{\rm up}$ is $n$ times ours ([DA1 Exercise 2.1](../da1_core.html#da1-2-ext-kl)). TracIn, TRAK and SOURCE report helpful as positive.
- Slide 14: local derivative $\mathcal I_i$. Leave-one-out $\phi(\pi(\mathbf 1-e_i))-\phi(\pi(\mathbf 1))$ equals $-\mathcal I_i$ only to first order. Subset deletion $\phi(\pi(\mathbf 1_S))$ for a random $S$, typically half the data; first-order prediction $\phi(\pi(\mathbf 1))-\sum_{j\notin S}\mathcal I_j$.
- Slide 15: the far point B of DA1 §3's regression (one feature plus an intercept). The black curve is the exact refit as its weight $\beta_B$ goes from 1 to 0; the dashed line is the tangent at 1, the influence-function prediction. Slope 0.256 at $\beta=1$; LOO 0.674; the tangent predicts 0.330. Ratio of changes 5.72. §3 explains the number.
- **Picked up in:** [DA1 §2](../da1_core.html#da1-2).

---

## 3. The classical influence function (11 min), slides 17–33

17. **Question:** *If training lands at a minimum, where does the minimum go when you reweight one example?*
18. **Equation (builds), warm-up, "The influence of a point on a mean":** $L(\mu;\beta)=\frac12\sum_j\beta_j(z_j-\mu)^2$, then $\mu(\beta)=\sum_j\beta_jz_j/\sum_j\beta_j$, then $\partial\mu/\partial\beta_i|_{\mathbf 1}=(z_i-\bar z)/n$. *"A point's pull on the mean is its distance from the mean, over $n$."*
19. **Equation:** $\nabla_wL(w^*(\beta);\beta)=0$ for all $\beta$ near $\mathbf 1$. *"Implicit function theorem: if $H=\nabla^2_wL$ is invertible at $w^*$, this defines a smooth $w^*(\beta)$. So differentiate the equation, not the training algorithm."*
20. **Equation (builds), "Two places $\beta_i$ enters":** $\nabla_wL=\sum_j\beta_j\nabla L_j$, then $\nabla^2_wL\,\partial w^*/\partial\beta_i+\nabla L_i(w^*)=0$ (through $w^*$, explicitly). *"At $\beta=\mathbf 1$: $H$ is the Hessian, $g_i=\nabla L_i(w^*)$ is $i$'s gradient."*
21. **Equation (builds):** $\partial w^*/\partial\beta_i=-H^{-1}g_i$, boxed: $\mathcal I_i=-g_\phi^\top H^{-1}g_i$. *"$g_\phi=\nabla\phi(w^*)$; the chain rule through $\phi$. Needs $H$ invertible."*
22. **Equation:** $H=n$, $g_i=-(z_i-\bar z)\Rightarrow-H^{-1}g_i=(z_i-\bar z)/n$. *"For the mean, $L_i=\frac12(z_i-\mu)^2$."* It recovers the warm-up.
23. **Figure** `pull.png`: contours of an elongated quadratic bowl; the pull $-g_i$ points mostly along the stiff direction, the move $-H^{-1}g_i$ is rotated toward the flat direction. *"Flat directions move most. Then $g_\phi$ reads off what $\phi$ cares about."*
24. **Figure** `wells.png`: a double well; lowering one example's weight tilts it so the global minimum jumps to the other well while the IF follows the local branch. *"The IF follows the nearby minimum; retraining from scratch may land in another basin."*
25. **Takeaway:** *One linear solve per query: $\mathcal I_i=-g_\phi^\top H^{-1}g_i$.*
26. **Question:** *How well does a derivative predict deleting a whole example?*
27. **Equation (builds), "Everything is explicit":** $L_j=\frac12(y_j-w^\top x_j)^2$, $A=\sum_jx_jx_j^\top$; then $H=A$, $g_j=-x_jr_j$, $r_j=y_j-x_j^\top w^*$. *"The IF says deleting $j$ moves the fit by $-A^{-1}x_jr_j$."*
28. **Equation (builds), "Exact deletion, step 1":** subtract the two sets of normal equations to get $A\delta=x_j(x_j^\top\delta-r_j)$, $c:=x_j^\top\delta-r_j$. *"$\delta=w^*_{(-j)}-w^*$ is the exact change. So $\delta=cA^{-1}x_j$: only the size $c$ is unknown."*
29. **Equation (builds), "Exact deletion, step 2":** $x_j^\top\delta=c\,h_j$, so $c=h_jc-r_j\Rightarrow c=-r_j/(1-h_j)$. *"$h_j$ is the leverage of example $j$."*
30. **Equation (builds), "Exact deletion is the IF, magnified":** $w^*_{(-j)}-w^*=-A^{-1}x_jr_j/(1-h_j)$, then $\frac1{1-h_j}=1+h_j+h_j^2+\cdots$ with the leading 1 marked "IF".
31. **Equation:** $h_j=x_j^\top A^{-1}x_j\in[0,1]$, $\sum_jh_j=p$. *"$p$: the number of parameters. At $h_j=1$ the point alone pins down a direction."*
32. **Figure** `leverage.png`: left, a regression with points A (in the bulk) and B (far out), refit against IF prediction for each; right, exact over predicted change against leverage, on the curve $1/(1-h)$. Title: *"Low leverage: close. High leverage: far too small."* Caption: *"Point B has leverage $h=0.825$, so $1/(1-h)=5.7$: the puzzle from slide 15."*
33. **Takeaway:** *The IF is least reliable for exactly the points you most want to ask about. And this model is convex and converged.*

**Speaker notes.**
- Slide 17: here $\pi$ is "the minimizer". Start with the smallest possible model.
- Slide 18: quotient rule, $\partial\mu/\partial\beta_i=(z_i\sum_j\beta_j-\sum_j\beta_jz_j)/(\sum_j\beta_j)^2$, at $\beta=\mathbf 1$ equal to $(nz_i-n\bar z)/n^2$. Robust statistics asked this first (Hampel 1974): move a fraction $\varepsilon$ of the mass to $z$ and the mean moves at rate $z-\mu$. Keep $(z_i-\bar z)/n$ in mind: it comes back in §4.
- Slide 19: $w^*=w^*(\mathbf 1)$ is the trained model. We assume a branch of stationary points exists and is smooth; the implicit function theorem says when.
- Slide 20: chain rule on $F(w,\beta)=\sum_j\beta_j\nabla L_j(w)$: the $w$-dependence gives $\partial F/\partial w\cdot\partial w^*/\partial\beta_i$, and $\partial F/\partial\beta_i=\nabla L_i$.
- Slide 21: this is Koh and Liang (2017) §2, in our convention. Deleting $i$ changes $\phi$ by about $-\mathcal I_i=g_\phi^\top H^{-1}g_i$. In practice: solve $Hu=g_\phi$ once per query, then $\mathcal I_i=-u^\top g_i$ for every training example. L2 is about doing that solve when $H$ has $10^{18}$ entries.
- Slide 22: $\partial L_i/\partial\mu=-(z_i-\mu)$, second derivative 1 per example, so $H=n$ in summed units.
- Slide 23: three factors of $-g_\phi^\top H^{-1}g_i$. $g_i$ is how example $i$ pulls on the parameters (its descent direction is $-g_i$); $H^{-1}$ converts force to displacement, a lot along small eigenvalues and little along large ones; $g_\phi$ projects the displacement onto the observable. This is why the gradient dot product $-g_\phi^\top g_i$ (the IF with $H$ replaced by $I$) can be badly wrong: it ignores the rotation. §5 shows it.
- Slide 24: the theorem needs only $H$ invertible, and gives the unique smooth branch of stationary points through $w^*$. To call that branch "the retrained model" also needs $w^*$ to be a strict minimum and retraining to stay in this basin; both hold automatically for a strictly convex loss. The tilt in the figure is exaggerated. [DA1 Exercise 3.3](../da1_core.html#da1-3-ext-ift) has a version with numbers: $L_1=(w^2-1)^2$, $L_2=0.10w$, $L_3=-0.11w$; lowering $\beta_3$ to 0.8 moves the global minimum to the other well.
- Slide 26: back to the 5.7× puzzle. Linear regression is the one model where we can do the deletion exactly, in closed form.
- Slide 27: $w^*=A^{-1}\sum_jx_jy_j$. The influence prediction of the deletion change is $-\partial w^*/\partial\beta_j=H^{-1}g_j=-A^{-1}x_jr_j$. Upweighting $j$ pulls the fit toward $j$, along $A^{-1}x_j$, in proportion to the residual. [DA1 Exercise 3.1](../da1_core.html#da1-ex3-1) is the required exercise of the morning; you may want to do it before looking at slides 28–30.
- Slides 28–29: expand the left side and use $Aw^*=\sum_ix_iy_i$: $A\delta-x_jx_j^\top w^*-x_jx_j^\top\delta=-x_jy_j$, then collect. For step 2, take the inner product of $\delta=cA^{-1}x_j$ with $x_j$, then substitute into $c=x_j^\top\delta-r_j$.
- Slide 30: the IF keeps the leading 1. The higher terms are the higher-order Taylor terms of the response in $\beta_j$, which linear regression happens to sum in closed form (Sherman–Morrison; [DA1 Exercise 3.2](../da1_core.html#da1-3-ext-series) gives the whole response curve, the black line of slide 15).
- Slide 31: at $h_j=1$, deleting the point leaves the fit undetermined: $(A-x_jx_j^\top)A^{-1}x_j=x_j(1-h_j)=0$, so the reduced design is singular. $h_j$ is the diagonal of the hat matrix $XA^{-1}X^\top$, an orthogonal projection of rank $p$: hence $0\le h_j\le1$ and the trace is $p$, so the average leverage is $p/n$. With $\hat\Sigma=A/n$, $h_j=\frac1nx_j^\top\hat\Sigma^{-1}x_j$, a squared length in the metric set by the data; with an intercept, $\frac1n$ plus $\frac1n$ times the squared Mahalanobis distance of the other features from their mean.
- Slide 32: left, removing A ($h=0.068$, blue) or B ($h=0.825$, red), with the exact refit (solid) and the IF prediction (dotted); for A both are visually close to the full fit (correction factor 1.07). Right, every point's ratio of exact to predicted change in slope lies on $1/(1-h)$. 15 points, $p=2$, so the leverages sum to 2.
- **Picked up in:** [DA1 §3](../da1_core.html#da1-3), [DA1 Exercise 3.1](../da1_core.html#da1-ex3-1).

---

## 4. Neural networks: damping, Gauss–Newton, PBRF (11 min), slides 34–47

34. **Question (bullets build):** *At a network checkpoint, which assumptions survive?* $H$ invertible? No: flat directions. / $H$ positive? No: negative curvature. / $\nabla L=0$? No: training stopped early.
35. **Figure** `valley.png`: the hyperbola $ab=\bar z$ in the $(a,b)$ plane; at a minimizer, $v$ points across the valley and $v_\perp$ along it. *"A toy: predict $ab$, losses $L_i=\frac12(z_i-ab)^2$. Along the valley, $H$ has a zero eigenvalue."*
36. **Equation (builds), "Damping is exact for a proximal question":** $L(w;\beta)+\frac\lambda2\|w-w^*\|^2$ has Hessian $H+\lambda I$ at $w^*$, so $\mathcal I_i^\lambda=-g_\phi^\top(H+\lambda I)^{-1}g_i$. *"Not a numerical patch: reweight $i$, while paying to move away from $w^*$."*
37. **Equation (builds), "The gradient points across the valley":** $H=nvv^\top$, $g_i=-(z_i-\bar z)v\Rightarrow\mathcal I_i^\lambda=(z_i-\bar z)\,g_\phi^\top v/(n\|v\|^2+\lambda)$. *"$g_i$ has no flat component, so $\lambda\to0$ is finite. For $\phi=ab$ the limit is $(z_i-\bar z)/n$: the mean again."*
38. **Figure** `ab_damping.png`: damped IF against $\lambda$ for four minimizers; for $\phi=a^2$ the plateaus differ by minimizer, for $\phi=ab$ they coincide. Title: *"Finite limits, but which one?"* Caption: *"$\phi=a^2$ changes along the valley: its answer depends on the minimizer."*
39. **Equation (builds), "Split the Hessian through the outputs":** $L=\sum_i\ell_i(f_i(w))$, $\nabla L=\sum_iJ_i^\top\nabla_f\ell_i$; then $H=G+{}$residual term, $G=\sum_iJ_i^\top H_{y,i}J_i$. *"$f_{ic}$: output $c$ on example $i$. $J_i=\partial f_i/\partial w$. $H_{y,i}=\nabla^2_f\ell_i$. Product rule: two terms."*
40. **Equation (builds), "$G$ can't have negative eigenvalues":** $\ell_i$ convex in $f\Rightarrow G\succeq0\Rightarrow G+\lambda I\succ0$ for every $\lambda>0$. *"With $H$, an eigenvalue $s<0$ makes $1/(s+\lambda)$ blow up at $\lambda=-s$."*
41. **Question:** *The checkpoint need not minimize the training loss.* So what is $-g_\phi^\top(G+\lambda I)^{-1}g_i$ the derivative of?
42. **Figure** `bregman.png`: a convex loss curve with its tangent at $f^s$; the vertical gap between curve and tangent at $f$ is $D(f,f^s)$. *"$f^s$: the checkpoint's outputs. $D(f,f^s)=\ell(f)-\ell(f^s)-\nabla\ell(f^s)^\top(f-f^s)$: zero gradient at $f^s$, Hessian $H_y$."*
43. **Equation (builds), "Reweight, paying for every move":** $Q(w;\beta)=\sum_j(\beta_j-1)L_j(w)+\sum_jD_j(f_j(w),f_j(w^s))+\frac\lambda2\|w-w^s\|^2$ (the reweighting, the output cost, the weight cost). *"$w^s$: the saved checkpoint. The minimizer of $Q$ is the proximal Bregman response function, PBRF$(\beta)$."*
44. **Equation (builds), "The checkpoint is optimal by construction":** $Q(w;\mathbf 1)\ge0$ with equality at $w^s$; $\nabla^2_wQ(w^s)=G+\lambda I$; $\partial_{\beta_i}\nabla_wQ=g_i$. *"No residual term: $D_j$ has zero gradient in $f$ at $f^s$."*
45. **Equation:** $\partial\phi(\mathrm{PBRF}(\beta))/\partial\beta_i|_{\beta=\mathbf 1}=-g_\phi^\top(G+\lambda I)^{-1}g_i$. *"Exact, for a question that isn't retraining. Slide 3's LLM results used an approximation of it."*
46. **Figure** `proximity.png`: two scatter plots for a 25-parameter MLP, damped-IF predictions against proximal retraining (Spearman 0.995) and against plain retraining (Spearman 0.19). Title: *"The damped IF answers its own question."* Caption: *"A 25-parameter MLP: it ranks proximal retraining almost perfectly, plain retraining hardly at all."*
47. **Takeaway:** *Each fix changes the question. The damped Gauss–Newton IF answers the PBRF question exactly.*

**Speaker notes.**
- Slide 34: three standard modifications turn $-g_\phi^\top H^{-1}g_i$ into the formula used in practice, $-g_\phi^\top(G+\lambda I)^{-1}g_i$. The theme: each fix changes the question being answered. Bae et al. (2022) §§1–4 is the full argument.
- Slide 35: networks have continuous families of minimizers: rescaling symmetries, unused units, more parameters than constraints. The $a\cdot b$ model is the smallest example: minimizers are the hyperbola $ab=\bar z$; rescaling $(a,b)\to(\alpha a,b/\alpha)$ moves along it. At a minimizer $H=nvv^\top$ with $v=(b^*,a^*)$: rank one, so $H^{-1}$ does not exist. The SLT day uses the same model.
- Slide 36: at a stationary $w^*$ the proximal term has zero gradient, so $w^*$ is still stationary for the new objective at $\beta=\mathbf 1$, and the derivation of §3 goes through with $H+\lambda I$. For $H\succeq0$, increasing $\lambda$ shrinks the parameter response; the scalar influence can still grow or change sign.
- Slide 37: in $H$'s eigenbasis $(s_k,u_k)$, $(H+\lambda I)^{-1}g_i=\sum_kc_ku_k/(s_k+\lambda)$ with $g_i=\sum_kc_ku_k$. On a flat direction ($s_k=0$) the factor is $1/\lambda$, which matters only if $c_k\neq0$; then the response grows like $1/\lambda$. If $g_i$ has no flat component, the $\lambda\to0$ limit is the pseudoinverse answer $-H^+g_i$. Here $v=(b^*,a^*)$ is an eigenvector of $H$ with eigenvalue $n\|v\|^2$ and $g_i$ is parallel to $v$, so nothing diverges. For $\phi=ab$ (what the model computes, constant along the valley), $g_\phi^\top v=\|v\|^2$ and the limit $(z_i-\bar z)/n$ is the same at every minimizer. For $\phi=a^2$, $g_\phi=(2a^*,0)$, $g_\phi^\top v=2a^*b^*$, and the limit $2\bar z(z_i-\bar z)/(n\|v\|^2)$ depends on the minimizer.
- Slide 38: each curve has a plateau for $\lambda\ll n\|v\|^2$, then $1/\lambda$ decay. Circles: the derivative of retraining by gradient flow warm-started at that minimizer, by finite differences; they match the plateaus because gradient flow conserves $a^2-b^2$ and moves along $v$. Retraining from a random init could land anywhere on the new hyperbola, so for $a^2$ there is no retraining-independent answer, and damping is what picks one.
- Slide 39: $f_i(w)\in\mathbb R^C$ is the network's outputs on example $i$. Differentiating twice: the second derivative hits either the $J_i$ (giving the residual term) or the $\nabla_f\ell_i$ (giving $G$). The residual term vanishes when the output gradients do (squared error at interpolation) or when the model is linear in its parameters.
- Slide 40: squared error and softmax cross-entropy are both convex in the outputs. Near $\lambda=-s$ the damped Hessian IF gives huge responses of either sign. A caution (cut from the slides): $G\neq H$ even at a strict minimum. One parameter, $f(w)=w^2$, target $-1$, squared error: $L=\frac12(w^2+1)^2$ has a strict minimum at $w=0$, where $J=f'(0)=0$ so $G=0$, while the residual $(f+1)f''=2$ gives all of $H=2$. So the Gauss–Newton influence need not equal the Hessian influence even at a converged minimum.
- Slide 41: at a non-stationary $w^s$ the implicit-function derivation has nothing to start from. Bae et al. (2022): "If influence functions are the answer, then what is the question?" Construct an objective whose optimum is the checkpoint by design.
- Slide 42: for squared error $D=\frac12\|f-f^s\|^2$. In general it measures how far the outputs moved, in the loss's own geometry, and is non-negative for convex $\ell$.
- Slide 43: Bae et al. (2022) §5. $w^s$ need not be stationary. The first term is zero at $\beta=\mathbf 1$ and adds or removes example $j$'s loss as $\beta_j$ moves.
- Slide 44: the Hessian in $w$ of $D_j(f_j(w),f_j^s)$ at $w^s$ is $J_j^\top H_{y,j}J_j$ plus $\sum_c(\partial D_j/\partial f_c)\nabla^2_wf_{jc}$, and the second sum is zero because $\nabla_fD_j=0$ at $f^s$. That is where $G$ comes from: not an approximation, an exact Hessian of a different objective.
- Slide 45: in words, how would $\phi$ change if $i$ were reweighted, while moving the outputs on the training set and moving the parameters were both penalized? This is the formula DA2 stop 1 computes at LLM scale (with EK-FAC for $G$).
- Slide 46: this uses the damped *Hessian* IF of slide 36, not Gauss–Newton; the objective has a fixed ridge $\frac{10^{-3}}2\|w\|^2$ in every refit, included in $H$. Bae et al. identify five gaps between the IF and leave-one-out retraining. Changes of question: warm start (retraining starts from $w^s$, not a fresh init), proximity (moving the parameters is penalized), non-convergence (the Bregman term stands in for the loss). Errors: linearization, the approximate solve. For networks the changes of question dominate (their Fig. 4). Setup: one hidden layer of 8 tanh units, 20 noisy points of $\sin2x$, trained to a minimum by Newton's method; $\phi$ is the squared error at one test input; $\lambda$ is 1% of the top Hessian eigenvalue. Deleting each point: damped-IF prediction against proximal retraining (the IF's own target, made finite) and against plain retraining from $w^*$. The Hessian's eigenvalues run from $5\times10^{-4}$ to 106: without the proximal term, a deletion moves the parameters far along nearly flat directions (across deletions, the vector of plain-retraining changes has 49 times the norm of the IF predictions). Which one you want is a choice of question, not a matter of accuracy.
- **Picked up in:** [DA1 §4](../da1_core.html#da1-4), [DA1 Exercise 4.1](../da1_core.html#da1-4-ext-ab) (the $a\cdot b$ toy), [DA1 §4 extension](../da1_core.html#da1-4-ext-mlp) (the MLP), [TD 2.6 §6.1](../../td/td2_6_curvature.html#td26-1) (same decomposition), L2 (how to multiply by $G$).

---

## 5. Checking (4 min), slides 48–52

48. **Question:** *How would you know an attribution is right?*
49. **Bullets (builds), "Target and estimator":** Target: the question. A choice of $\pi$, and of one of the three targets. / Estimator: how you compute the answer. / They fail separately, so they need separate checks.
50. **Bullets (builds), "From least to most demanding":** finite differences of the method's own target. / Leave-one-out retraining. / Retraining on random half-subsets.
51. **Equation:** $\mathrm{LDS}=\mathrm{Spearman}(\{\phi(S_k)\},\{\sum_{i\in S_k}\mathcal I_i\})$. *"$S_k$: a random half of the data; retrain on it and measure $\phi(S_k)$. It checks the ranking, not the size."*
52. **Bullets (builds), table, "Different checks rank methods differently":** exact IF, damped IF, gradient dot product against check 1 (own target, relative error), check 2 (LOO, Spearman), check 3 (LDS): exact IF $3\times10^{-7}$, 0.96, 0.83; damped IF $3\times10^{-7}$ (proximal), 0.82, 0.83; gradient dot product —, 0.36, 0.56. *"Both IFs nail their own target; LOO prefers the exact IF; LDS can't tell them apart."*

**Speaker notes.**
- Slide 48: every method outputs a number per training example. Which question do you test it against?
- Slide 49: we have met two targets for one formula: the damped IF is the derivative of the proximal (PBRF) response, not of retraining. The table makes it concrete: one method computes its own target to seven digits and still predicts leave-one-out worse than another.
- Slide 50: check 1 tests the estimator only (reweight by $\pm\varepsilon$, recompute $\pi$, difference $\phi$). Check 2 tests one finite deletion. Check 3 tests large deletions, far from $\beta=\mathbf 1$: it rewards estimators that extrapolate, which need not be those that differentiate exactly. For networks, retraining with a new seed moves $\phi$ too. Single-model targets (this run, this seed: unrolling) differ from distributional ones (averaged over seeds: datamodels, TRAK's protocol).
- Slide 51: $S_1,\dots,S_m$ are random subsets, typically half the data; $\phi(S_k)$ is the behaviour after retraining on $S_k$. To first order $\phi(S)-\phi(\text{all})=-\sum_{j\notin S}\mathcal I_j$, which is $\sum_{i\in S}\mathcal I_i$ up to a constant. Averaged over test points. Park et al. (2023) Definition 2.4. Multiplying every score by 10, or any increasing transform of the predictions, leaves the LDS unchanged.
- Slide 52: logistic regression, $n=80$: pick the check that matches your question. DA1 §5's setup is $\ell_2$-regularized logistic regression, 80 examples in 10 dimensions with feature scales spanning 1.5 decades, 10 test losses as observables; Newton solver, so every check is deterministic. Damping $\lambda$ is the median Hessian eigenvalue. The damped IF is off by 48% against the plain local derivative, but matches its own proximal target to $3\times10^{-7}$: a different target, not a defect. It is worse on LOO and ties on LDS. Calibration slopes on the half-subsets are 12 to 35, far from 1: a high LDS doesn't license "deleting these changes the loss by this much". The gradient dot product ignores the $H^{-1}$ rotation of slide 23; the Hessian's eigenvalues here span a factor of about 360.
- **Picked up in:** [DA1 Exercises 5.1–5.2](../da1_core.html#da1-ex5-1), where you compute this table.

---

## 6. Another $\pi$: unrolled GD, filters, the afternoon's map (4 min), slides 53–59

53. **Question:** *What if $\pi$ is this training run, not an argmin?*
54. **Equation:** $\partial w/\partial\beta_i=-F(M)g_i=-\sum_kF(s_k)c_ku_k$. *"Many estimators filter the curvature spectrum. Classical IF: $F(s)=1/s$. Damped: $1/(s+\lambda)$. Now one more."*
55. **Equation (builds), "Differentiate $K$ steps instead of an argmin":** $r_t=\partial w_t/\partial\beta_i$; $r_{t+1}=(I-\eta H)r_t-\eta g_i$, $r_0=0$; $r_K=-\eta\sum_{k<K}(I-\eta H)^kg_i$; $F(s)=\eta\sum_{k<K}(1-\eta s)^k=(1-(1-\eta s)^K)/s$, $F(0)=\eta K$. *"Constant curvature and gradient give a geometric series; per eigenvalue, a filter."*
56. **Figure** `filters.png`: log-log plot of filters: $1/s$ diverges at small $s$; the damped filter and the $K$-step GD filter flatten to a finite value. Title: *"Same shape, different meanings."* Caption: *"$\lambda$ is a proximal penalty; $\eta K$ is training time. Both tame the flat directions."*
57. **Bullets (builds), table, "DA2's methods differ mainly in what $\pi$ is":** classical IF, the stationary point near the trained $w^*$ / PBRF, a proximal optimum around the checkpoint / unrolling, this run's last iterate, same seed and order / Bayesian IF, a local posterior / datamodels, a fitted predictor of retraining.
58. **Takeaway:** *Training time can play the role of damping. Same shape, different question.*
59. **Takeaway:** *Before trusting an attribution, ask which $\pi$, which target, and which check.*

**Speaker notes.**
- Slide 53: the simplest case of unrolling, and a lens on the afternoon's methods. If short of time, skip to the map (slide 57) and say: the methods differ in $\pi$.
- Slide 54: $M$ is a curvature matrix with eigenpairs $(s_k,u_k)$; $g_i=\sum_kc_ku_k$. Methods differ in $F$ and in which matrix $M$ is ($H$, $G$, a segment's average Hessian).
- Slide 55: assume $H$ and $g_i$ constant over the $K$ steps. On the range of $H$ the sum is $r_K=-[I-(I-\eta H)^K]H^{-1}g_i$; on its null space, $r_K=-\eta Kg_i$. For $0<\eta s<2$ the filter tends to $1/s$ as $K\to\infty$. This is the simplest case of unrolling (DA2 stop 2); SOURCE's filter $(1-e^{-\eta Ks})/s$ is its continuous-time cousin.
- Slide 56: plotted with $\lambda=1$ and $\eta K=1$. Every filter but the classical one is finite at $s=0$: $F(0)=1/\lambda$, $\eta K$, $1/\gamma$ (local Bayesian IF at Gaussian leading order) or $\tau^2/\sigma_{\rm noise}^2$ (Bayesian linear regression in $X^\top X$ units). For GD, directions with $s\ll1/(\eta K)$ haven't responded yet: influence depends on how long you train. What the lens hides: general unrolling has changing curvature and non-commuting step Jacobians; the Bayesian IF beyond leading order sees non-quadratic structure; datamodels have no such form. Shapes are similar; questions are not.
- Slide 57: the classical IF, the PBRF and (the simplest case of) unrolling were derived in this lecture. The rest are DA2: unrolling is stop 2 (TracIn, SOURCE, MAGIC); the Bayesian IF is stop 3, where $\phi$ is read as $\mathbb E_{p_\beta}\phi$ under $p_\beta\propto\exp(-L(w;\beta)-\frac\gamma2\|w-w^*\|^2)$; datamodels are stop 5. TRAK is not a row: it is an analytic estimator judged against the datamodel-style subset target.
- **Picked up in:** [DA1 §6](../da1_core.html#da1-6) (a table of filters, and [the interactive filter explorer](../da1_core.html#da1-6-figure)), [DA2 stop 2](../da2_methods.html#da2-stop-2) (unrolling).

---

## 7. The day and readings (1 min), slides 60–61

60. **Bullets (builds), "How to use the day":** Morning: DA1, §1–§6. Do Exercises 3.1, 5.1 and 5.2. / After lunch: L2, curvature at scale. / Then: the DA2 tour. Read each capsule; go deep anywhere.
61. **Readings** (five, scored Importance / Length).

**Speaker notes.**
- DA1 is one path of about 90 minutes; everything under "Extension" is optional. DA2's six stops: IFs at scale (EK-FAC), unrolling, Bayesian IFs, influence over training, datamodels and TRAK, open problems. Read every capsule in order (about 30 min), then go deep through the paper or the stop's walkthrough.
- Links: [DA1](../da1_core.html), [DA2](../da2_methods.html#da2-stop-1), [L2](L2_curvature_at_scale.html).
- Reading sections: Koh & Liang §2 (derivation, mean-loss convention) and §4 (checks against LOO); Bae et al. §§1–4 (the five gaps) and §5 (PBRF), Fig. 4; Grosse et al. §5.3 with Figs. 1 and 15–16 (read selectively: 119 pages); Park et al. §2 (LDS, Definition 2.4) and §3; Mueller §3–4.
- Also in DA1: Ghorbani & Zou 2019 (Data Shapley); Basu, Pope & Feizi 2021 (IFs are fragile); Ilyas et al. 2022 (Datamodels).
- **Picked up in:** [DA1](../da1_core.html), [DA2](../da2_methods.html#da2-stop-1).

Readings (on the slide):
- Koh & Liang 2017, [*Understanding Black-box Predictions via Influence Functions*](https://arxiv.org/abs/1703.04730). ●●●●● / ●●○○○.
- Bae et al. 2022, [*If Influence Functions are the Answer, Then What is the Question?*](https://arxiv.org/abs/2209.05364). ●●●●● / ●●●○○.
- Grosse et al. 2023, [*Studying LLM Generalization with Influence Functions*](https://arxiv.org/abs/2308.03296). ●●●●● / ●●●●○.
- Park et al. 2023, [*TRAK: Attributing Model Behavior at Scale*](https://arxiv.org/abs/2303.14186). ●●●●○ / ●●●○○.
- Mueller 2024, [*Missed Causes and Ambiguous Effects*](https://arxiv.org/abs/2407.04690). ●●●○○ / ●●○○○.

---

## Compressed path (about 25 min)

| Segment | Show | Skip (say in one sentence if needed) |
|---|---|---|
| 0 (2 min) | 2, 3, 5 | 1, 4 ("ablating an example costs a training run") |
| 1 (2 min) | 6, 7, 9 | 8 (say "Shapley averages over subsets, at the price of many retrainings") |
| 2 (3 min) | 11, 12, 13, 15, 16 | 10, 14 ("deletion of one, or of a large subset, is a different target from the derivative") |
| 3 (6 min) | 19, 20, 21, 30, 31, 32, 33 | 17, 18 (the warm-up mean), 22, 23, 24 (the wells figure), 25, 26–29 (say "exact deletion is the IF divided by $1-h_j$") |
| 4 (6 min) | 34, 36, 39, 40, 41, 45, 46, 47 | 35, 37, 38 (the $a\cdot b$ toy details), 42, 43, 44 (the Bregman and PBRF derivation: say "an objective whose optimum is the checkpoint, with Hessian $G+\lambda I$") |
| 5 (3 min) | 49, 51, 52 | 48, 50 |
| 6 (2 min) | 57, 59 | 53, 54, 55, 56, 58 (everything in §6 except the map) |
| 7 (1 min) | 60, 61 | none |
