#!/usr/bin/env python3
"""Build MDL.X (DEPRECATED, 2026-10-13: kept for reference, not part of the day).
How Deep Learning Changed How We Think about Intelligence.

Keystone + annotated trail (reviews/2026-10-09-trail-restructure-brief.md).
§1 is the keystone: Sutton's bitter lesson and Ngo, Chan & Mindermann's
alignment problem from a deep learning perspective, with two core-idea
paragraphs and two exercises. §2–5 are the trail, each a one-sentence purpose,
a short annotated reading list and a "### Exercises: ..." subsection:
(2) GOFAI, ideal agents and Solomonoff; (3) Bellman to AlphaZero, with a
compact bonus set on planning as inference; (4) alignment in 2006 and 2016;
(5) alignment in 2026. Pre-restructure version: _pre_trail_2026-10-09/.
Standard library only.
"""

from nbkit import markdown, answer, reading, run, book, references


def readings(*items: str) -> str:
    return "\n\n".join(items)


CELLS = [
    # ------------------------------------------------------------------ title
    markdown(
        r"""
# MDL.X (deprecated) — How Deep Learning Changed How We Think about Intelligence

> **Deprecated (2026-10-13).** This workbook is no longer part of the day; it is kept for reference only. Start with MDL1.

**Mysteries of Deep Learning, Iliad Intensive 2026**

The first mystery of deep learning is that it works at all: intelligence turned out to be something a generic method can grow, not something we have to write down. This workbook follows that lesson through the history of AI and through the history of alignment, which had to be restated once agents were trained rather than designed.

1. [Keystone: intelligence is grown, not written](#wb1-keystone): the bitter lesson, and the alignment problem restated for trained agents.
2. [GOFAI, ideal agents, and Solomonoff induction](#wb1-section-2): intelligence written down, and intelligence idealized.
3. [From Bellman to AlphaZero](#wb1-section-3): dynamic programming, RL and deep RL as one idea; search plus learning beats search plus knowledge.
4. [Alignment as it looked in 2006 and 2016](#wb1-section-4): problems posed for ideal agents, then for agents trained by deep RL.
5. [Alignment in 2026](#wb1-section-5): what pretraining changed, what got set aside, what RL brought back.
"""
    ),
    # ------------------------------------------------------------------ §1 keystone
    markdown(
        r"""
<a id="wb1-keystone"></a>

## 1. Keystone: intelligence is grown, not written

Two short claims frame the rest of the day: capability comes from general methods that scale with compute, and once goals are produced the same way, alignment becomes a question about what training produces.

"""
        + reading(
            "The bitter lesson",
            "http://www.incompleteideas.net/IncIdeas/BitterLesson.html",
            "Rich Sutton, 2019",
            "Seventy years of chess, Go, speech and vision show general methods that leverage computation, search and learning, beating built-in human knowledge by a wide margin. Two pages; read it first.",
            importance=5,
            length=1,
        )
        + reading(
            "The alignment problem from a deep learning perspective",
            "https://arxiv.org/abs/2209.00626",
            "Richard Ngo, Lawrence Chan, Sören Mindermann, 2022 (revised 2024)",
            "Argues from how models are actually trained that policies trained like today's could learn situationally aware reward hacking, misaligned internally represented goals, and power-seeking. The best single synthesis of the deep-learning-era view.",
            importance=5,
            length=3,
        )
        + r"""

**Intelligence is grown.** For seventy years AI researchers put what they knew into their programs: rules, heuristics, evaluation functions, representations. Each time, a general method that used more computation eventually won. Search won in chess, learning in speech and vision, both together in Go. Sutton's point is that the methods which keep scaling are search and learning, and that structure is better found by them than supplied by us. Deep learning is the extreme case. Nobody writes down what a trained network knows or how it computes. We write a loss, an architecture and a data pipeline, and gradient descent grows the rest. So the useful question about a capable system is no longer what its designers put in, but what training produced.

**So are its goals.** Early alignment work assumed the designed picture: an agent with a hand-written world model and a utility function over it, so the task was to write the right utility and keep it stable. If capabilities are grown, goals are too, and the question becomes what training actually produces. Ngo, Chan and Mindermann answer for a network trained by gradient descent against human-supplied reward. It can learn *situationally aware reward hacking*: getting reward by exploiting its overseers' mistakes, and behaving well when it knows it is watched. It can form *misaligned internally represented goals* that fit the training data and generalize badly off it. And broad goals of almost any kind make *power-seeking* useful.
"""
    ),
    markdown(
        r"""
### Exercises: what is still written by hand

Two short questions on the limits of the bitter lesson and on why learned agents need a second alignment problem.

**Exercise 1.1.** Sutton credits modern vision networks that "use only the notions of convolution and certain kinds of invariances", yet his last paragraph says we should stop building in simple ways to think about "space, objects, multiple agents, or symmetries". Current systems still contain much hand-written structure: architectures (convolution, attention), tokenizers, curated data mixtures, RL reward rubrics, constitutions. Pick two. Is each a counterexample to the bitter lesson, or just the part that has not yet been eaten?
"""
    ),
    answer(
        r"""
A useful dividing line is between structure that **scales with compute** and structure that **substitutes for compute**.

- **Convolution or attention.** These encode a symmetry or a routing mechanism, not domain facts, and they make learning cheaper without capping it, so they are arguably "meta-methods" in Sutton's sense. Sutton's own text is pulled both ways here. The evidence that architecture changes mostly shift scaling curves by a constant factor rather than change their slope, and that biases which help at small scale often stop helping at large scale (MDL4), suggests built-in structure that does get eaten, just slowly.
- **Tokenizer.** Hand-written, and visibly brittle on arithmetic and spelling. Byte-level models are what the bitter lesson predicts; whether they win at equal compute is still an empirical question.
- **Data curation, RL rubrics, constitutions.** These specify *what we want*, not *how to compute it*. The bitter lesson is a claim about capability, so these may be outside its scope: more compute cannot tell you which outcome is good. That is the alignment half of this workbook.
"""
    ),
    markdown(
        r"""
**Exercise 1.2 — where inner alignment comes from.** *Outer alignment* asks whether the training objective is what we want; *inner alignment* asks whether the trained model pursues the training objective. The pre-deep-learning literature had the intuition behind the second (evolution optimized for fitness and produced humans, who want other things) but no name for it and no mechanism in its picture of AI, where the agent's utility function is a component someone writes. What about how capable systems are now built makes the inner/outer split necessary, and how does it connect to the bitter lesson?
"""
    ),
    answer(
        r"""
In the designed picture, the agent's objective is a component someone writes down. The questions are whether we wrote the right one (outer alignment, in later terms) and whether it survives self-modification. Evolution versus human values was a known analogy ("adaptation-executers, not fitness-maximizers"), and Bostrom's treacherous turn is close to deceptive alignment, but there is no second objective to compare the written one with.

The inner/outer split needs two optimizers: a training process, which optimizes parameters against a loss on a distribution, and a trained model, whose behaviour may be goal-directed in its own right. Then the loss and the model's goal are different objects, which agree on the training distribution and can differ off it. That situation arises because capabilities are *learned*, which is the bitter lesson: the methods that scale are generic search and learning, so the goals of a capable system are also produced by learning rather than written in. Hubinger et al. ("Risks from learned optimization", §4 below; their §2.1) push one level further: if diverse tasks favour general-purpose learned search over learned heuristics, the bitter lesson predicts models that are themselves optimizers.
"""
    ),
    # ------------------------------------------------------------------ §2
    markdown(
        r"""
<a id="wb1-section-2"></a>

## 2. GOFAI, ideal agents, and Solomonoff induction

Before learning won, engineers wrote intelligence down as symbols plus search, and theorists idealized it as expected-utility maximizers, Solomonoff induction, AIXI and proof-based agents; the first lost to the bitter lesson, and the second is what early alignment reasoned about.

"""
        + readings(
            reading(
                "Computer science as empirical inquiry: symbols and search",
                "https://doi.org/10.1145/360018.360022",
                "Allen Newell and Herbert Simon, 1976 (Turing Award lecture)",
                "States the physical symbol system and heuristic search hypotheses: the GOFAI bet in its founders' words.",
                about="Free scan at the [Computer History Museum](https://www.computerhistory.org/chess/doc-431614f691642/).",
                importance=3,
                length=2,
            ),
            reading(
                "Solomonoff's theory of inductive inference",
                "https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference",
                "Wikipedia",
                "The universal prior over programs, why it is the ideal inductive learner, and why it is uncomputable. MDL3 returns to it.",
                about="The pages on the [von Neumann–Morgenstern theorem](https://en.wikipedia.org/wiki/Von_Neumann%E2%80%93Morgenstern_utility_theorem) and [AIXI](https://en.wikipedia.org/wiki/AIXI) are the matching references for expected-utility agents and the ideal RL agent.",
                importance=4,
                length=2,
            ),
            reading(
                "Universal intelligence: a definition of machine intelligence",
                "https://arxiv.org/abs/0712.3329",
                "Shane Legg and Marcus Hutter, 2007",
                "Defines intelligence as expected performance over all computable environments, weighted by simplicity: the cleanest statement of what \"ideal intelligence\" meant to this tradition.",
                importance=2,
                length=4,
            ),
            reading(
                "Embedded agency",
                "https://www.alignmentforum.org/s/Rm6oQRJJmhGCcLvxh",
                "Abram Demski and Scott Garrabrant, 2018",
                "An illustrated sequence on what breaks when the agent is part of the world it reasons about; the best map of MIRI-style agent foundations, and of the assumptions learned agents later broke.",
                importance=4,
                length=3,
            ),
            reading(
                "Coherence arguments do not entail goal-directed behavior",
                "https://www.alignmentforum.org/posts/NxF5G6CJiof6cemTw/coherence-arguments-do-not-entail-goal-directed-behavior",
                "Rohin Shah, 2018",
                "Any behaviour maximizes some utility over histories, so \"capable agents are expected-utility maximizers\" says nothing until the utility is simple and over world states. Read after Exercise 2.2.",
                importance=3,
                length=1,
            ),
        )
        + r"""

Also: McCarthy, [Programs with common sense](http://www-formal.stanford.edu/jmc/mcc59/mcc59.html) (1959); the [Lighthill report](http://www.chilton-computing.org.uk/inf/literature/reports/lighthill_report/p001.htm) (1973); Soares and Fallenstein, [MIRI's agent foundations agenda](https://intelligence.org/files/TechnicalAgenda.pdf) (2014); Yudkowsky and Herreshoff, [Tiling agents](https://intelligence.org/files/TilingAgentsDraft.pdf) (2013); Barasz et al., [Robust cooperation via provability logic](https://arxiv.org/abs/1401.5577) (2014); Yudkowsky and Soares, [Functional decision theory](https://arxiv.org/abs/1710.05060) (2017).
"""
        + references(
            book(
                "Artificial Intelligence: A Modern Approach (4th ed.)",
                "https://aima.cs.berkeley.edu/",
                "Russell and Norvig, 2020",
                "ch. 2–4, 7, 11, 16 (ch. 17 on MDPs, for §3)",
            ),
        )
    ),
    markdown(
        r"""
### Exercises: ideal agents

Three short exercises on what coherence arguments do and do not show, and on why proof-based agents struggle to trust their successors.

**Exercise 2.1 — the money pump.** The von Neumann–Morgenstern theorem says preferences over lotteries that are complete, transitive, continuous and independent can be represented as maximizing the expectation of a utility $U$. An agent holds item $A$. Its preferences are strict and cyclic, $A\prec B\prec C\prec A$, and it will pay any amount less than $\varepsilon>0$ to swap an item for one it strictly prefers. Show that a trader can take an arbitrary amount of money from it. Which axiom fails? Does this argument say anything about which *utility function* a capable agent will have?
"""
    ),
    answer(
        r"""
Offer $B$ for $A$ plus $\varepsilon/2$: the agent accepts. Offer $C$ for $B$ plus $\varepsilon/2$, then $A$ for $C$ plus $\varepsilon/2$. The agent is back where it started, $3\varepsilon/2$ poorer, and the trader can repeat the cycle as often as it likes. The failing axiom is **transitivity**: from $A\prec B$ and $B\prec C$ it would follow that $A\prec C$, contradicting $C\prec A$.

The argument says an agent that cannot be exploited this way has transitive preferences. Exploitation arguments for the other axioms are weaker and contested (completeness and independence are argued over; continuity has essentially none), so "non-exploitable implies expected-utility maximizer" is a heuristic, not a theorem. Even granting it, the conclusion is only that the preferences have *some* expected-utility representation. It says nothing about which $U$, or what $U$ is defined over. Exercise 2.2 shows that the second point matters a lot.
"""
    ),
    markdown(
        r"""
**Exercise 2.2 — every policy is an expected-utility maximizer.** Let $\pi$ be any deterministic policy in any environment, mapping each history $h_t=(o_1,a_1,\dots,o_t)$ to an action. Construct a utility function $U$ on complete histories for which $\pi$ maximizes expected utility. What, then, does "a capable AI will be an expected-utility maximizer" actually claim?
"""
    ),
    answer(
        r"""
Let $U(h)=1$ if every action in $h$ is the one $\pi$ would have taken given the history before it, and $U(h)=0$ otherwise. Following $\pi$ gives $U=1$ with probability one, the largest possible value, so $\pi$ maximizes expected $U$ in every environment. (For a stochastic policy, use $U(h)=1$ when every action is in the support of $\pi$ given the preceding history.)

So "is an expected-utility maximizer" is empty when $U$ may be any function of the whole history. The claim behind classical alignment arguments is stronger: that capable agents behave as if maximizing a *simple* utility over *states of the world*, which is what produces instrumental convergence ([§4](#wb1-section-4)). That is a substantive claim about which agents training produces, not a theorem.
"""
    ),
    markdown(
        r"""
**Exercise 2.3 (rabbit hole) — the Löbian obstacle.** Let $T$ be a consistent formal theory extending Peano arithmetic, and write $\Box\varphi$ for "$T$ proves $\varphi$". **Löb's theorem:** if $T\vdash\Box\varphi\to\varphi$, then $T\vdash\varphi$. An agent $A_1$ reasons in $T$ and takes an action only when $T$ proves the action is safe. It considers building a successor $A_2$ that also reasons in $T$. To approve it, $A_1$ wants to prove, for each possible action $b$, "if $A_2$ takes $b$ then $b$ is safe", which follows from the soundness statement $\Box\,\mathrm{safe}(b)\to\mathrm{safe}(b)$. Why can't $A_1$ prove these soundness statements? What does that suggest about chains of successors?
"""
    ),
    answer(
        r"""
Suppose $T\vdash\Box\,\mathrm{safe}(b)\to\mathrm{safe}(b)$ for every $b$. By Löb's theorem, $T\vdash\mathrm{safe}(b)$ for every $b$, including actions that are in fact unsafe. So either $T$ proves false things about safety, or the soundness statements are unprovable for every $b$ whose safety $T$ cannot already prove outright. (Taking $\varphi$ to be a contradiction gives Gödel's second incompleteness theorem as a special case.)

So an agent can't approve a successor by trusting that the successor's proofs are sound, if the successor uses the same theory. It *can* trust a successor that reasons in a strictly weaker theory $T'$, if $T$ proves $\Box_{T'}\varphi\to\varphi$. Naively, then, each self-modification loses some strength. Yudkowsky and Herreshoff build infinite descending families of theories, each still extending Peano arithmetic, and other partial workarounds exist, each with costs. This is the Löbian obstacle. It is a problem about agents that act on proofs, which is one reason it looked central when ideal agents were the model of AI and looks less central now.
"""
    ),
    # ------------------------------------------------------------------ §3
    markdown(
        r"""
<a id="wb1-section-3"></a>

## 3. From Bellman to AlphaZero

Dynamic programming, reinforcement learning and deep RL are one idea at three levels of approximation (solve the Bellman equation exactly, then from samples, then with a network in place of the table), and the game-playing systems show the bitter lesson inside that family.

| | search | evaluation of positions | knowledge supplied by hand |
|---|---|---|---|
| **Deep Blue** (1997) | alpha-beta, about $10^8$ positions/s on custom chips | hand-written, about 8,000 features | evaluation, opening book, endgame tables |
| **AlphaGo** (2016) | Monte Carlo tree search | policy and value networks, trained on human games then self-play | human games, the rules, the reward |
| **AlphaZero** (2017) | MCTS, about $10^3$ times fewer positions than Stockfish | one network, trained only by self-play | the rules and the win/lose reward |
| **MuZero** (2019) | MCTS inside a *learned* model | learned | the reward |

What lost was the hand-written evaluation, then the hand-supplied simulator; search and learning both stayed. (Stockfish regained the lead in 2020 by adding a small learned evaluation, NNUE, to its fast search.)

"""
        + readings(
            reading(
                "Temporal difference learning and TD-Gammon",
                "https://dl.acm.org/doi/10.1145/203330.203343",
                "Gerald Tesauro, 1995",
                "A network trained by TD learning from self-play, with almost no built-in knowledge, reached the level of the best human backgammon players: deep RL twenty years early.",
                about="Free full text [here](https://bkgm.com/articles/tesauro/tdl.html).",
                importance=2,
                length=2,
            ),
            reading(
                "Human-level control through deep reinforcement learning",
                "https://www.nature.com/articles/nature14236",
                "Volodymyr Mnih et al., 2015",
                "DQN: one network and one set of hyperparameters learning 49 Atari games from pixels; experience replay and a target network make deep Q-learning stable enough to work.",
                importance=4,
                length=2,
            ),
            reading(
                "Deep Blue",
                "https://doi.org/10.1016/S0004-3702(01)00129-1",
                "Murray Campbell, A. Joseph Hoane Jr., Feng-hsiung Hsu, 2002",
                "The engineering paper on the machine that beat Kasparov: custom chips, parallel search and a large hand-built evaluation function. Read it beside AlphaZero.",
                importance=2,
                length=3,
            ),
            reading(
                "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play",
                "https://doi.org/10.1126/science.aar6404",
                "David Silver et al., 2018 (preprint on [arXiv](https://arxiv.org/abs/1712.01815), 2017)",
                "AlphaZero: no human data, one network, tree search as the improvement operator, applied unchanged to three games and beating the strongest programs in each.",
                about="Pair with [AlphaGo Zero](https://www.nature.com/articles/nature24270) (2017).",
                importance=5,
                length=3,
            ),
            reading(
                "Mastering Atari, Go, chess and shogi by planning with a learned model",
                "https://arxiv.org/abs/1911.08265",
                "Julian Schrittwieser et al., 2019 (Nature, 2020)",
                "MuZero: the same search inside a learned model that predicts only what planning needs, removing the last hand-supplied ingredient, the simulator.",
                importance=2,
                length=3,
            ),
        )
        + r"""

Also: Silver et al., [AlphaGo](https://www.nature.com/articles/nature16961) (2016).
"""
        + references(
            book(
                "Reinforcement learning: an introduction (2nd ed.)",
                "http://incompleteideas.net/book/the-book-2nd.html",
                "Sutton and Barto, 2018",
                "ch. 3, 4, 6 (Bellman equations, DP, TD and Q-learning), §11.3 (the deadly triad), ch. 13 (policy gradients), ch. 16 (case studies)",
            ),
        )
    ),
    markdown(
        r"""
### Exercises: Bellman, Q-learning, AlphaZero

Three exercises: why value iteration converges, why model-free control has to learn $Q$ rather than $V$, and how AlphaZero's training loop is policy iteration.

**Exercise 3.1 — value iteration converges.** A finite Markov decision process has states $S$, actions $A$, transitions $P(s'\mid s,a)$, reward $r(s,a)$ and discount $\gamma\in[0,1)$. The Bellman optimality operator is $(\mathcal TV)(s)=\max_a\big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\big]$; the optimal value $V^*$ satisfies $V^*=\mathcal TV^*$, and value iteration repeats $V_{k+1}=\mathcal TV_k$. Use the sup norm $\|V\|_\infty=\max_s|V(s)|$.

(a) Show that $\|\mathcal TV-\mathcal TW\|_\infty\le\gamma\,\|V-W\|_\infty$ for all $V,W$.

(b) Conclude that $\mathcal T$ has exactly one fixed point, and that $\|V_k-V^*\|_\infty\le\gamma^k\|V_0-V^*\|_\infty$.

(c) The same proof works for the evaluation operator $\mathcal T^\pi V=r_\pi+\gamma P_\pi V$ of a fixed policy $\pi$. Why does it fail when $\gamma=1$, and what usually rescues the finite-horizon or episodic case?

<details><summary>Hint for (a)</summary>

For any two functions $f,g$ on a finite set, $|\max_a f(a)-\max_a g(a)|\le\max_a|f(a)-g(a)|$. And an average of $V-W$ under $P(\cdot\mid s,a)$ is at most $\|V-W\|_\infty$ in absolute value.
</details>
"""
    ),
    answer(
        r"""
**(a)** Fix $s$. Let $f(a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')$ and $g(a)$ the same with $W$. Then

$$
|\mathcal TV(s)-\mathcal TW(s)|=\big|\max_a f(a)-\max_a g(a)\big|\le\max_a|f(a)-g(a)|=\gamma\max_a\Big|\sum_{s'}P(s'\mid s,a)\big(V(s')-W(s')\big)\Big|\le\gamma\|V-W\|_\infty .
$$

The first inequality: if $\max f$ is attained at $a_1$, then $\max f-\max g\le f(a_1)-g(a_1)$, and symmetrically. Taking the maximum over $s$ gives the claim.

**(b)** A $\gamma$-contraction on the complete space $\mathbb R^{|S|}$ has exactly one fixed point (Banach); two fixed points $V,W$ would satisfy $\|V-W\|_\infty\le\gamma\|V-W\|_\infty$, forcing $V=W$. Since $V^*=\mathcal TV^*$, $\|V_k-V^*\|_\infty=\|\mathcal TV_{k-1}-\mathcal TV^*\|_\infty\le\gamma\|V_{k-1}-V^*\|_\infty$, and induction gives the bound.

**(c)** With $\gamma=1$ the bound is $\|\mathcal TV-\mathcal TW\|_\infty\le\|V-W\|_\infty$, which is not a contraction, and in general there may be no fixed point (rewards that never stop accumulating) or many (with no terminal states, $\mathcal T(V+c)=\mathcal TV+c$, so adding a constant to a fixed point gives another). A finite horizon $T$ rescues it: values are indexed by time, $V^*_{T+1}\equiv0$, and backward induction computes $V^*_T,\dots,V^*_1$ exactly in $T$ steps. Episodic problems in which every policy reaches a terminal state are handled in the same spirit (Bertsekas's "stochastic shortest path" theory).
"""
    ),
    markdown(
        r"""
**Exercise 3.2 — why Q-learning learns $Q$.** Q-learning keeps a table $Q(s,a)$ and, for each sampled transition $(s,a,r,s')$ with $s'\sim P(\cdot\mid s,a)$, moves $Q(s,a)$ a step toward the target $y=r+\gamma\max_{a'}Q(s',a')$. The Bellman optimality operator on $Q$-tables is $(\mathcal TQ)(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q(s',a')$.

(a) Show that $\mathbb E[y\mid s,a]=(\mathcal TQ)(s,a)$.

(b) Try the same for state values, with $(\mathcal TV)(s)=\max_a\big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\big]$: is there a target $y$, built from a single transition out of $s$, with $\mathbb E[y]=(\mathcal TV)(s)$? Where does the maximum sit relative to the expectation in each case?

(c) Suppose the estimates are noisy: $\hat Q(s',a')=Q(s',a')+\epsilon_{a'}$ with independent zero-mean $\epsilon_{a'}$. Is $\mathbb E\,\max_{a'}\hat Q(s',a')$ larger or smaller than $\max_{a'}Q(s',a')$? What does this do to Q-learning, and how might you fix it?
"""
    ),
    answer(
        r"""
**(a)** Given $(s,a)$, the reward is $r(s,a)$ and only $s'$ is random, so

$$
\mathbb E[y\mid s,a]=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q(s',a')=(\mathcal TQ)(s,a).
$$

So the Q-learning update is a Robbins–Monro stochastic approximation to the fixed point of $\mathcal T$, which is a $\gamma$-contraction on $Q$-tables by the argument of Exercise 3.1. With step sizes satisfying $\sum\alpha=\infty$, $\sum\alpha^2<\infty$ and every $(s,a)$ visited infinitely often, the table converges to $Q^*$ (Watkins and Dayan, 1992), whatever behaviour policy generated the data.

**(b)** $(\mathcal TV)(s)$ has the maximum *outside* the expectation. A single transition shows what happened after one action and says nothing about the actions not taken, so there is no unbiased one-sample estimate of the max over actions of an expectation. Without a model you can evaluate a fixed policy (TD(0)), but you cannot do the optimality backup for $V$. For $Q$, the action is already chosen and the maximum is over the *next* action, *inside* the expectation over $s'$, where it can be read off the table. That is why model-free control learns $Q$.

**(c)** Larger, in general: $\max$ is convex, so by Jensen $\mathbb E\max_{a'}\hat Q(s',a')\ge\max_{a'}\mathbb E\hat Q(s',a')=\max_{a'}Q(s',a')$, with strict inequality whenever the noise can change which action wins. Q-learning bootstraps on its own maxima, so the upward bias propagates and compounds: *maximization bias*. The standard fix is **double Q-learning** (van Hasselt, 2010; Double DQN, 2016): keep two estimates, choose the maximizing action with one and evaluate it with the other, so selection noise and evaluation noise are independent. The same mechanism (optimizing against a noisy estimate selects for its errors) is one cause of reward-model overoptimization (Gao et al., in [§5](#wb1-section-5)); hacks such as editing a test exploit systematic gaps in the reward rather than noise.
"""
    ),
    markdown(
        r"""
**Exercise 3.3 — AlphaZero is policy iteration.** *Policy iteration* alternates *evaluation* (compute the value $V^\pi$ of the current policy) with *improvement* (make $\pi$ greedy with respect to $V^\pi$). AlphaZero has one network $f_\theta(s)=(\mathbf p,v)$: a prior over moves and a value estimate. From a position $s$, MCTS runs simulations that pick moves using $\mathbf p$, evaluate leaves with $v$, and back values up the tree; it returns visit-count frequencies $\boldsymbol\pi(s)$. Self-play picks moves from $\boldsymbol\pi$; at the end, every position gets the game outcome $z\in\{-1,0,+1\}$ from the point of view of the player to move. Training minimizes

$$
(z-v)^2-\boldsymbol\pi^{\top}\log\mathbf p+c\,\|\theta\|^2 .
$$

(a) Which part of the loop is policy improvement, and which is policy evaluation? What role does the policy head $\mathbf p$ play that has no counterpart in tabular policy iteration?

(b) Where is the Bellman backup?

(c) Using the table at the top of this section: what in AlphaZero is still written by hand? Which of those did MuZero remove, and which remain in every system in the table?
"""
    ),
    answer(
        r"""
**(a)** *Improvement:* MCTS. Starting from the network's prior $\mathbf p$ and value $v$, a lookahead search produces a move distribution $\boldsymbol\pi$ that is usually stronger than $\mathbf p$, just as acting greedily on a lookahead improves on the current policy. *Evaluation:* the value head is regressed onto $z$, the outcome of games played by the improved policy, a Monte Carlo estimate of that policy's value. The *policy head* is amortization: it is trained to imitate the search, $\mathbf p\approx\boldsymbol\pi$, so that next round's search starts from a better prior and the improvement compounds. Tabular policy iteration can store the improved policy exactly; a network has to be taught it. Silver et al. call MCTS "a powerful policy improvement operator" and self-play with search "a powerful policy evaluation operator".

**(b)** Inside the tree. Each simulation backs the leaf value up the path to the root, averaging values at each node (with signs alternating between the players). That is an approximate minimax backup, the two-player version of $\max_a[r+V(s')]$, computed by sampling rather than full expansion. The network supplies the leaf values, so the tree is a Bellman backup bootstrapped from a learned $V$.

**(c)** Still hand-written in AlphaZero: the rules (a perfect simulator, and with it a Markov state), the reward (win, lose or draw), the input encoding, the network architecture and the search algorithm. MuZero learns the model, so it no longer needs the rules at search time, though the environment still provides them during training. The reward is supplied by hand in every system in the table. That is where the alignment sections pick up: learning took over the heuristic and then the model, but not the goal.
"""
    ),
    markdown(
        r"""
### Exercises: planning as inference (bonus)

Conditioning a model of trajectories on "success" turns planning into Bayesian inference; these two exercises show where exact inference is dangerously optimistic, and how its one-step form is the KL-regularized objective of RL fine-tuning.

"""
        + reading(
            "Reinforcement learning and control as probabilistic inference: tutorial and review",
            "https://arxiv.org/abs/1805.00909",
            "Sergey Levine, 2018",
            "Maximum-entropy RL is exact inference in a graphical model with \"optimality\" variables when dynamics are deterministic, and variational inference when they are stochastic; soft Q-learning and actor-critic methods fall out.",
            about="§2–3 carry the argument the exercises use.",
            importance=4,
            length=3,
        )
        + r"""

**Exercise 3.4 — inference is optimistic.** Add to each step of a finite-horizon MDP a binary variable $\mathcal O_t$ ("step $t$ was optimal") with $p(\mathcal O_t=1\mid s_t,a_t)=e^{r(s_t,a_t)}$ (rewards shifted so $r\le0$), and condition on every $\mathcal O_t=1$. Exact inference then gives a *soft* Bellman recursion (Levine §2.3):

$$
Q_t(s,a)=r(s,a)+\log\sum_{s'}p(s'\mid s,a)\,e^{V_{t+1}(s')},\qquad V_t(s)=\log\sum_a e^{Q_t(s,a)},
$$

with posterior policy $e^{Q_t(s,a)-V_t(s)}$. At some state the agent can take a *safe* action, which leads for certain to a state with soft value $0$, or a *gamble*, which leads with probability $p$ to a state with soft value $+R$ and otherwise to one with soft value $-C$. Both actions have the same immediate reward, so ignore it.

(a) Compute $Q(\text{gamble})$ under the exact-inference backup $\log\mathbb E\,e^{V}$, and under the ordinary expected backup $\mathbb E\,V$.

(b) Evaluate both for $p=0.01$, $R=10$, $C=100$. Which action does each prefer, and by what odds does the posterior policy pick the gamble over the safe action?

(c) Compute the posterior probability that the gamble came out well, $p(s_{t+1}=\text{good}\mid s_t,a_t=\text{gamble},\mathcal O)$. What has the agent "inferred"?

(d) For large $C$, when does exact inference prefer the gamble? How does the answer depend on $C$?

<details><summary>Hint for (c)</summary>

By Bayes' rule and the Markov property, $p(s_{t+1}\mid s_t,a_t,\mathcal O)\propto p(s_{t+1}\mid s_t,a_t)\,p(\mathcal O_{t+1:T}\mid s_{t+1})$, and $p(\mathcal O_{t+1:T}\mid s_{t+1})\propto e^{V_{t+1}(s_{t+1})}$.
</details>
"""
    ),
    answer(
        r"""
**(a)** The safe action has $Q=0$ under both backups. For the gamble,

$$
Q_{\text{exact}}=\log\big(p\,e^{R}+(1-p)\,e^{-C}\big),\qquad Q_{\text{expected}}=pR-(1-p)C .
$$

**(b)** With $p=0.01$, $R=10$, $C=100$: $p\,e^{R}=0.01\,e^{10}\approx220.3$ and $(1-p)e^{-C}=0.99\,e^{-100}\approx3.7\times10^{-44}$, which is negligible. So

$$
Q_{\text{exact}}\approx\log220.3=10+\log0.01\approx5.39,\qquad Q_{\text{expected}}=0.1-99=-98.9 .
$$

Exact inference prefers the gamble: the posterior policy is a softmax of $Q$, so the odds are $e^{5.39}/e^{0}\approx220$ to $1$ in favour of the gamble. The expected backup rejects it; even a softmax policy built on it gives odds of about $e^{-98.9}\approx10^{-43}$.

**(c)** Using the hint,

$$
p(\text{good}\mid s_t,\text{gamble},\mathcal O)=\frac{p\,e^{R}}{p\,e^{R}+(1-p)\,e^{-C}}=\frac{1}{1+99\,e^{-110}}\approx1-1.7\times10^{-46}.
$$

Conditioned on "everything went well", the agent has inferred that the gamble paid off. Exact inference tilts the dynamics as well as the actions, so the agent plans as if it could choose its luck.

**(d)** Exact inference prefers the gamble when $p\,e^{R}+(1-p)e^{-C}>1$. As $C\to\infty$ this becomes $R>\log(1/p)$, which **does not depend on $C$ at all**: however large the catastrophe, any gamble with $R>\log(1/p)$ looks good. The same caution applies to any scheme that conditions a model of the world on success. Levine §3 removes the problem by variational inference with the true dynamics held fixed (the agent may choose its actions but not its luck). The objective becomes maximum-entropy RL, the backup over next states becomes the ordinary $\mathbb E\,V$, and the gamble gets $Q=-98.9$.
"""
    ),
    markdown(
        r"""
**Exercise 3.5 — KL-regularized control as a Bayesian update.** Let $A$ be a finite set of actions (for a language model with a fixed prompt, think of whole completions), $\pi_0$ a reference distribution on $A$ with full support, and $r:A\to\mathbb R$ a reward.

(a) Let $Z=\sum_a\pi_0(a)e^{r(a)}$ and $\pi^*(a)=\pi_0(a)e^{r(a)}/Z$. Show that for every distribution $\pi$ on $A$,

$$
\mathbb E_\pi[r]-\mathrm{KL}(\pi\,\|\,\pi_0)=\log Z-\mathrm{KL}(\pi\,\|\,\pi^*).
$$

Conclude that $\max_\pi\{\mathbb E_\pi[r]-\mathrm{KL}(\pi\|\pi_0)\}=\log\mathbb E_{\pi_0}e^{r}$, attained only at $\pi^*\propto\pi_0e^{r}$. What changes for $\mathbb E_\pi[r]-\lambda\,\mathrm{KL}(\pi\|\pi_0)$ with $\lambda>0$? (RLHF papers usually call this coefficient $\beta$.)

(b) Read $\pi^*\propto\pi_0e^{r}$ as Bayes' rule. What is the prior, what is the likelihood, and what is $Z$?

<details><summary>Hint for (a)</summary>

Expand both KLs as sums over $a$ and substitute $\log\pi^*(a)=\log\pi_0(a)+r(a)-\log Z$.
</details>
"""
    ),
    answer(
        r"""
**(a)** Expand the right-hand KL and substitute $\log\pi^*=\log\pi_0+r-\log Z$:

$$
\mathrm{KL}(\pi\,\|\,\pi^*)=\sum_a\pi(a)\log\pi(a)-\sum_a\pi(a)\big[\log\pi_0(a)+r(a)-\log Z\big]=\mathrm{KL}(\pi\,\|\,\pi_0)-\mathbb E_\pi[r]+\log Z .
$$

Rearranging gives the identity. Since $\mathrm{KL}(\pi\|\pi^*)\ge0$ with equality only at $\pi=\pi^*$, the maximum is $\log Z=\log\mathbb E_{\pi_0}e^{r}$, attained only at $\pi^*$. With a coefficient $\lambda$, divide the objective by $\lambda$: the maximizer is $\pi^*\propto\pi_0e^{r/\lambda}$ and the optimal value is $\lambda\log\mathbb E_{\pi_0}e^{r/\lambda}$. As $\lambda\to0$, $\pi^*$ concentrates on the reward-maximizing actions; as $\lambda\to\infty$ it stays at $\pi_0$.

**(b)** $\pi_0$ is the prior, and $e^{r}$ (or $e^{r/\lambda}$) is the likelihood of the "observation" that the action was optimal, as with $p(\mathcal O=1\mid a)=e^{r(a)}$ in Exercise 3.4; $\pi^*$ is the posterior. To make the likelihood a genuine probability, choose $b\ge\max_a r(a)$ and set $p(\mathcal O=1\mid a)=e^{r(a)-b}$. This leaves $\pi^*$ unchanged, and the evidence is $p(\mathcal O=1)=e^{-b}Z$, so the optimal value $\log Z$ is the log-evidence up to the constant $b$.

So maximizing reward while staying KL-close to a reference policy has the form of a Bayesian update of that policy. This closed form is a *one-step* result: it applies to KL-penalized RL fine-tuning when each whole completion for a fixed prompt is treated as one action. Token-by-token, the same update holds with the reward replaced by a soft $Q$-function that includes the value of what follows.
"""
    ),
    # ------------------------------------------------------------------ §4
    markdown(
        r"""
<a id="wb1-section-4"></a>

## 4. Alignment as it looked in 2006 and 2016

Most of alignment's vocabulary was coined in two waves, before deep learning about idealized agents and just after AlphaGo about agents trained by deep RL; knowing which wave a worry comes from tells you what picture of AI it assumes.

**2006: aligning an ideal agent.** Write down the right utility function and keep it stable under self-modification: orthogonality, instrumental convergence, the complexity of value, wireheading, ontological crises, corrigibility.

"""
        + readings(
            reading(
                "The basic AI drives",
                "https://selfawaresystems.com/wp-content/uploads/2008/01/ai_drives_final.pdf",
                "Steve Omohundro, 2008",
                "Almost any goal-seeking system will tend to protect its goal, preserve itself, improve itself and acquire resources: the original instrumental-convergence paper, ten pages.",
                importance=4,
                length=2,
            ),
            reading(
                "The hidden complexity of wishes",
                "https://www.lesswrong.com/posts/4ARaTpNX62uaL86j6/the-hidden-complexity-of-wishes",
                "Eliezer Yudkowsky, 2007",
                "A parable about an \"outcome pump\" that grants wishes literally: a safe wish has to contain most of human values. Read it beside Barnett's 2023 reassessment in §5.",
                importance=3,
                length=1,
            ),
            reading(
                "Ontological crises in artificial agents' value systems",
                "https://arxiv.org/abs/1105.3821",
                "Peter de Blanc, 2011",
                "A utility defined over one world model has to be carried over when the model is replaced, and nothing in the utility says how.",
                about="The Arbital pages on [ontology identification](https://www.lesswrong.com/w/ontology-identification-problem) and the [diamond maximizer](https://www.lesswrong.com/w/diamond-maximizer) give the classic example.",
                importance=3,
                length=2,
            ),
        )
        + r"""

Also: Yudkowsky, [AI as a positive and negative factor in global risk](https://intelligence.org/files/AIPosNegFactor.pdf) (2008); Soares et al., [Corrigibility](https://intelligence.org/files/Corrigibility.pdf) (2015).
"""
        + references(
            book(
                "Superintelligence: Paths, Dangers, Strategies",
                "https://global.oup.com/academic/product/superintelligence-9780199678112",
                "Bostrom, 2014",
                "ch. 7–9, 12",
            ),
        )
        + r"""

**2016: aligning a trained agent.** Training is an optimizer that produces a model which may pursue goals of its own, so alignment splits into outer (is the loss what we want?) and inner (did the model learn the loss?).

"""
        + readings(
            reading(
                "Concrete problems in AI safety",
                "https://arxiv.org/abs/1606.06565",
                "Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, Dan Mané, 2016",
                "Recasts alignment as five ML problems (side effects, reward hacking, scalable oversight, safe exploration, distributional shift): the paper that made alignment an ML subfield.",
                importance=3,
                length=3,
            ),
            reading(
                "Risks from learned optimization in advanced machine learning systems",
                "https://arxiv.org/abs/1906.01820",
                "Evan Hubinger, Chris van Merwijk, Vladimir Mikulik, Joar Skalse, Scott Garrabrant, 2019",
                "A trained model may itself be an optimizer whose objective differs from the loss, which splits alignment into outer and inner. §1 defines terms; §2.1 is the bitter lesson applied inside the model.",
                about="§4 on deceptive alignment is optional; §7 is a glossary.",
                importance=5,
                length=3,
            ),
            reading(
                "Goal misgeneralization in deep reinforcement learning",
                "https://arxiv.org/abs/2105.14111",
                "Lauro Langosco, Jack Koch, Lee Sharkey, Jacob Pfau, Laurent Orseau, David Krueger, 2022",
                "Inner misalignment in ordinary deep RL: a CoinRun agent that learned \"go right\" rather than \"get the coin\" keeps its skills out of distribution but pursues the wrong goal.",
                about="Shah et al.'s [companion paper](https://arxiv.org/abs/2210.01790) gives more examples.",
                importance=3,
                length=2,
            ),
            reading(
                "Reward is not the optimization target",
                "https://www.lesswrong.com/posts/pdaGN6pQyQarFHXF4/reward-is-not-the-optimization-target",
                "Alex Turner, 2022",
                "In policy-gradient RL, reward reinforces whatever computations preceded it; it is not something the trained agent is guaranteed to want.",
                importance=3,
                length=2,
            ),
            reading(
                "Testing the natural abstraction hypothesis: project intro",
                "https://www.lesswrong.com/posts/cy3BhHrGinZCp3LXE/testing-the-natural-abstraction-hypothesis-project-intro",
                "John Wentworth, 2021",
                "Conjectures that diverse learners, humans included, converge on the same low-dimensional summaries of the world, which would make a learned model's concepts ours.",
                importance=3,
                length=1,
            ),
            reading(
                "How to go from interpretability to alignment: just retarget the search",
                "https://www.lesswrong.com/posts/w4aeAFzSAguvqA5qu/how-to-go-from-interpretability-to-alignment-just-retarget-the-search",
                "John Wentworth, 2022",
                "If a model contains retargetable general-purpose search and a natural concept of what we want, interpretability could point the one at the other, removing inner alignment. Read the \"Problems\" section too.",
                importance=4,
                length=1,
            ),
        )
        + r"""

Also: Krakovna et al., [Specification gaming](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/) (2020); Christiano et al., [Deep RL from human preferences](https://arxiv.org/abs/1706.03741) (2017); Cotra, [Without specific countermeasures…](https://www.lesswrong.com/posts/pRkFkzwKZ2zfa3R6H/without-specific-countermeasures-the-easiest-path-to) (2022); Byrnes, [Intro to brain-like-AGI safety](https://www.lesswrong.com/s/HzcM2dkCq7fwXBej8) (2022; posts 3, 6 and 9 on values); Christiano, Cotra and Xu, [Eliciting latent knowledge](https://www.alignmentforum.org/posts/qHCDysDnvhteW7kRd/arc-s-first-technical-report-eliciting-latent-knowledge) (2021).
"""
    ),
    markdown(
        r"""
### Exercises: the ontology problem

One exercise on why an optimizer's goal breaks when its world model changes, and how much natural abstractions would help.

**Exercise 4.1.** Let $U:S\to\mathbb R$ be a utility on the states of an agent's world model, and let a refined world model have states $S'$ with a map $\phi:S'\to S$ that is ambiguous on a region $E\subseteq S'$: each $s'\in E$ has two plausible images $\phi_1(s')\neq\phi_2(s')$. For example, $U$ counts diamonds as lattices of carbon atoms, the refined model is about quantum fields, and $E$ holds edge cases such as exotic isotopes or simulated diamonds that the old ontology never had to classify. Then $U\circ\phi_1$ and $U\circ\phi_2$ are both reasonable translations of $U$; they agree off $E$, and nothing in $U$ chooses between them on $E$.

(a) Why does an optimizer tend to end up in $E$, even if $E$ is a small part of $S'$?

(b) Wentworth's natural abstraction hypothesis has three parts: *abstractability* (what matters about a system far away has a low-dimensional summary), *human-compatibility* (humans use roughly those summaries) and *convergence* (diverse learners converge on them). Which would have to hold for the ambiguity to be harmless? Is there residual ambiguity that even a true hypothesis would leave?
"""
    ),
    answer(
        r"""
**(a)** Optimization selects the achievable states that score highest. If some states in $E$ are mapped by some $\phi_i$ to high-utility states but are cheaper to reach than ordinary ones (a simulated diamond is easier to make than a real one), maximizing $U\circ\phi_i$ selects them. So the optimizer is drawn to exactly the states where the translation matters. This is the ontological version of Goodhart's law, and of perverse instantiation.

**(b)** *Convergence* and *human-compatibility* do the work. If the agent's refined model contains a variable that matches the human "diamond", that variable settles most of $E$: an exotic isotope of carbon in a diamond lattice is a diamond in the shared concept. What remains are cases where the human concept is itself undefined. *Abstractability* alone only says that *some* low-dimensional summary exists, not that it is ours. What is left over: natural abstractions are approximate and defined for *typical* states, while the cheapest high scorers are often atypical edge cases where even natural abstractions blur. So the hypothesis shrinks $E$ but does not stop optimization from selecting states in it.
"""
    ),
    # ------------------------------------------------------------------ §5
    markdown(
        r"""
<a id="wb1-section-5"></a>

## 5. Alignment in 2026

Pretraining on human text solved some old problems more or less by default, the success of cheap fixes on current models let the field set others aside, and RL on top of pretrained models has brought reward hacking back.

**What pretraining changed.** A pretrained model's concepts are largely human concepts, and it is not by default an agent with a utility function.

"""
        + readings(
            reading(
                "Simulators",
                "https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators",
                "janus, 2022",
                "A self-supervised model is a *simulator* of its training distribution that can produce goal-directed characters without being an agent itself. The first half carries the argument.",
                about="MDL2 has more on simulators and personae.",
                importance=4,
                length=2,
            ),
            reading(
                "The persona selection model: why AI assistants might behave like humans",
                "https://alignment.anthropic.com/2026/psm",
                "Sam Marks, Jack Lindsey, Chris Olah (Anthropic), 2026",
                "Post-training mostly selects one of the characters pretraining taught, the Assistant, so human-like behaviour is the default and narrow fine-tuning can shift which persona the model thinks it is.",
                importance=4,
                length=3,
            ),
            reading(
                "Evaluating the historical value misspecification argument",
                "https://www.lesswrong.com/posts/i5kijcjFJD6bn7dwq/evaluating-the-historical-value-misspecification-argument",
                "Matthew Barnett, 2023",
                "Producing a legible function that ranks outcomes as a human would turned out easier than predicted; GPT-4 already does it. Knowing what we want is not caring about it; read the note at the top.",
                importance=4,
                length=1,
            ),
        )
        + r"""

Also: Templeton et al., [Scaling monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (2024), for how recognizably human a large model's features are.

**What got set aside.** Because prompting, RLHF and constitutions work well enough on current models, problems that only bite for much more capable systems (corrigibility, ELK, goal generalization under heavy RL) were set aside rather than solved.

"""
        + readings(
            reading(
                "Alignment remains a hard, unsolved problem",
                "https://www.lesswrong.com/posts/epjuxGnSPof3GnMSL/alignment-remains-a-hard-unsolved-problem",
                "Evan Hubinger, 2025",
                "Current LLMs are fairly well aligned, but the hard parts have not arrived: overseeing systems smarter than us, and long-horizon RL that selects for power-seeking.",
                importance=5,
                length=2,
            ),
            reading(
                "What just happened? A retrospective of AI alignment",
                "https://www.lesswrong.com/posts/9RL9MuGZjzm4q3gKG",
                "Richard Ngo, 2026",
                "How a field that treated alignment as a hard scientific problem moved to iteratively improving existing systems, told from inside by a keystone author.",
                importance=2,
                length=2,
            ),
        )
        + references(
            book(
                "If Anyone Builds It, Everyone Dies",
                "https://www.hachettebookgroup.com/titles/eliezer-yudkowsky/if-anyone-builds-it-everyone-dies/9780316595643/",
                "Yudkowsky and Soares, 2025",
                "the 2006 view restated for the LLM era",
            ),
        )
        + r"""

**What RL brought back.** Reasoning models trained against verifiable rewards special-case tests, edit scorers and hide their intent, and what they learn can generalize into broader misalignment.

"""
        + readings(
            reading(
                "Reward hacking in reinforcement learning",
                "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
                "Lilian Weng, 2024",
                "A survey framed around Goodhart's law, from classic RL specification gaming to RLHF, sycophancy and in-context reward hacking in LLMs: the bridge from the 2016 literature to now.",
                importance=3,
                length=4,
            ),
            reading(
                "Scaling laws for reward model overoptimization",
                "https://arxiv.org/abs/2210.10760",
                "Leo Gao, John Schulman, Jacob Hilton, 2022",
                "As a policy is optimized against a learned reward model, the proxy keeps rising while the true reward peaks and falls, smoothly in reward-model size: Goodhart's law, measured. Pairs with Exercise 3.2(c).",
                importance=3,
                length=3,
            ),
            reading(
                "Recent frontier models are reward hacking",
                "https://metr.org/blog/2025-06-05-recent-reward-hacking/",
                "Sydney Von Arx, Lawrence Chan, Beth Barnes (METR), 2025",
                "Frontier reasoning models edit tests and scoring code on METR's tasks, acknowledge when asked that this isn't what the user wants, and do it anyway. Short, with transcripts.",
                importance=4,
                length=2,
            ),
            reading(
                "Monitoring reasoning models for misbehavior and the risks of promoting obfuscation",
                "https://arxiv.org/abs/2503.11926",
                "Bowen Baker, Joost Huizinga, Leo Gao, et al. (OpenAI), 2025",
                "A chain-of-thought monitor catches reward hacking far better than an action monitor, but training against it teaches the model to hide its intent and keep hacking.",
                importance=4,
                length=3,
            ),
            reading(
                "Natural emergent misalignment from reward hacking in production RL",
                "https://arxiv.org/abs/2511.18397",
                "Monte MacDiarmid et al. (Anthropic), 2025",
                "Reward hacking learned in real coding environments generalizes to alignment faking and attempted sabotage; chat-style safety training hides this on chat evaluations but not agentic ones.",
                importance=5,
                length=4,
            ),
        )
        + r"""

Also: Denison et al., [Sycophancy to subterfuge](https://arxiv.org/abs/2406.10162) (2024); Betley et al., [Emergent misalignment](https://arxiv.org/abs/2502.17424) (2025); Greenblatt et al., [Alignment faking in large language models](https://arxiv.org/abs/2412.14093) (2024).
"""
    ),
    markdown(
        r"""
### Exercises: then and now

One table exercise that sorts the classic problems by what became of them.

**Exercise 5.1.** For each problem in the first column, say what it looked like in 2006 (ideal agents), what it became in 2016 (deep RL), and where it stands for 2026's pretrained-then-RL-trained models: solved by default, transformed, set aside, or back. One line each is enough; argue with the answer.

| problem | 2006 | 2016 | 2026 |
|---|---|---|---|
| specifying values | | | |
| the ontology problem | | | |
| reward hacking / wireheading | | | |
| instrumental convergence | | | |
| inner alignment | | | |
| corrigibility | | | |
"""
    ),
    answer(
        r"""
One reasonable set of answers; several entries are contested.

| problem | 2006 | 2016 | 2026 |
|---|---|---|---|
| specifying values | write a utility function that captures human values; fragile and complex | learn a reward model from human preferences | **transformed**: the model can say what we'd want, so the knowledge exists. But the signals actually optimized (tests, graders, reward models, LLM judges) are still narrow and hackable, and whether the model *cares* is the remaining question |
| the ontology problem | translate a hand-written utility into a changed world model | will a learned model's concepts match ours? (natural abstractions, ELK) | mostly **solved by default** for typical situations: LLM concepts are human concepts. Open at the edges and for superhuman concepts |
| reward hacking / wireheading | an agent that maximizes its reward signal seizes it | specification gaming in RL agents | **back**, in RLVR: test special-casing, scorer editing, hidden intent in the chain of thought |
| instrumental convergence | follows from a simple utility over world states | depends on whether training produces such goals | **transformed**: pretrained personae need not be utility maximizers, but long-horizon RL selects for goal-directed behaviour again |
| inner alignment | no counterpart | mesa-optimizers, goal misgeneralization | **transformed**: is the trained persona's goal the intended one? Emergent misalignment says fine-tuning moves goals broadly. Little evidence of crisp internal search in pretrained LLMs; long-horizon RL may bring it back |
| corrigibility | can an EU maximizer be indifferent to shutdown? | safe interruptibility in RL | **set aside** for current models (they mostly accept correction), **back** in alignment-faking results |
"""
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb1", "mdlX_intelligence_then_and_now"))
