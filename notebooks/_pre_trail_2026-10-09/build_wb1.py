#!/usr/bin/env python3
"""Build WB1: The bitter lesson, and what it did to alignment.

Reading-led workbook. The bitter lesson is the headline reading; four sections
follow it: (1) GOFAI, ideal agents and Solomonoff induction; (2) Bellman
optimality to RL to deep RL, with planning as inference as a bonus derivation;
(3) why alignment looked hard in 2006 and in 2016; (4) why it is hard in 2026.
Each section gives a little context, then readings with a few sentences each,
then a few questions with worked answers. Standard library only.
"""

from nbkit import markdown, answer, reading, run, book, references


def readings(*items: str) -> str:
    return "\n\n".join(items)


CELLS = [
    # ------------------------------------------------------------------ title
    markdown(
        r"""
# Workbook 1 — The bitter lesson, and what it did to alignment

**Mysteries of Deep Learning, Iliad Intensive 2026**

Seventy years of AI research taught one lesson over and over: methods that scale with computation beat methods that build in what we know. This workbook follows that lesson through the history of AI, and then through the history of alignment, which had to be rethought each time the lesson landed. Alignment was first posed for idealized agents written by hand. It was posed again when capable agents came to be *learned* by deep RL, and again when they came to be *pretrained* on human text.

Start with the headline reading below. The four sections after it are independent; read whichever interests you.

1. [GOFAI, ideal agents, and Solomonoff induction](#wb1-section-1): intelligence as search over hand-written symbols, and the theorist's idealizations (expected-utility maximizers, proof-based agents, Solomonoff induction and AIXI) that early alignment work reasoned about.
2. [From Bellman to AlphaZero](#wb1-section-2): how dynamic programming, RL and deep RL all come out of the Bellman optimality equation; Deep Blue versus AlphaGo; and a bonus derivation chain on planning as inference.
3. [Why alignment looked hard: 2006 and 2016](#wb1-section-3): *Superintelligence*-era problems (instrumental convergence, reward hacking, ontological crises), and what changed once agents were trained by deep learning (mesa-optimizers, goal misgeneralization, natural abstractions).
4. [Why alignment is hard: 2026](#wb1-section-4): what LLMs solved more or less by default, what got deprioritized because hacky fixes worked, and why RL on top has brought reward hacking back.

Each section opens with a few paragraphs of context, then the readings, each with a sentence or two so you can decide whether to read it. Each reading is scored for importance and length (key under the contents); textbooks are listed separately, for reference. Questions have worked answers, hidden until you open them.
"""
    ),
    # ------------------------------------------------------------------ headline
    markdown(
        r"""
<a id="wb1-headline"></a>

## The headline reading

"""
        + reading(
            "The bitter lesson",
            "http://www.incompleteideas.net/IncIdeas/BitterLesson.html",
            "Rich Sutton, 2019",
            "Across 70 years of AI research (chess, Go, speech, vision), general methods that leverage computation have beaten methods that build in human domain knowledge, by a large margin, because the cost of computation keeps falling exponentially. The two methods that \"seem to scale arbitrarily\" are search and learning.",
            about="Two pages; read it first. The last two paragraphs carry the normative claim: build in the meta-methods that find structure, not the structure we have found.",
            importance=5,
            length=1,
        )
        + r"""

Keep three questions in mind for the rest of the workbook. What was written by hand, and what replaced it? Which of the two scalable methods, search or learning, did the replacing? And when the hand-written part was the *goal* rather than the *method*, what happened to it?
"""
    ),
    markdown(
        r"""
**Question 0.1.** Sutton's vision paragraph credits modern networks that "use only the notions of convolution and certain kinds of invariances", while his final paragraph says we should stop building in simple ways to think about "space, objects, multiple agents, or symmetries". Current systems still contain a lot of hand-written structure: architectures (convolution, attention), tokenizers, curated data mixtures, RL reward rubrics, constitutions. Pick two. Is each a counterexample to the bitter lesson, or just the part that has not yet been eaten?
"""
    ),
    answer(
        r"""
A useful dividing line is between structure that **scales with compute** and structure that **substitutes for compute**.

- **Convolution or attention.** These encode a symmetry or a routing mechanism, not domain facts, and they make learning cheaper without capping it, so they are arguably "meta-methods" in Sutton's sense. Sutton's own text is pulled both ways here: he praises convolution and invariances in vision, then lists symmetries among the things not to build in. The evidence that architecture changes mostly shift scaling curves by a constant factor rather than change their slope, and that biases which help at small scale often stop helping at large scale (WB5 §2), suggests built-in structure that does get eaten, just slowly.
- **Tokenizer.** Hand-written, and visibly brittle on arithmetic and spelling. Byte-level models are what the bitter lesson predicts; whether they win at equal compute is still an empirical question.
- **Data curation, RL rubrics, constitutions.** These specify *what we want*, not *how to compute it*. The bitter lesson is a claim about capability, so these may be outside its scope altogether: more compute cannot tell you which outcome is good. Sections 3 and 4 are about this part.
"""
    ),
    # ------------------------------------------------------------------ §1
    markdown(
        r"""
<a id="wb1-section-1"></a>

## 1. GOFAI, ideal agents, and Solomonoff induction

**Pitch.** Before learning won, there were two ways to think about intelligence. Engineers wrote it down: symbols, rules and search. Theorists idealized it: an agent that predicts perfectly and maximizes expected utility, or one that acts only on what it can prove. The engineering route lost to the bitter lesson. The idealizations are what the first generation of alignment research was about, so they are worth knowing even though no one builds them.

**Symbols and search.** "Good old-fashioned AI" bet that intelligence is the manipulation of symbols. Newell and Simon's *physical symbol system hypothesis* says a symbol system has the necessary and sufficient means for general intelligent action, and their *heuristic search hypothesis* says such systems solve problems by search. In practice that meant a hand-written representation (states, operators, logical sentences, production rules) plus search guided by a hand-written heuristic: theorem provers, STRIPS-style planners, chess programs, expert systems. It failed in three ways that recur through this workbook. Search blows up: $b^d$ lines of play to depth $d$, which Lighthill called the *combinatorial explosion*; Bellman's related *curse of dimensionality* is the exponential growth of a state space in the number of state variables. Knowledge has to be typed in, one rule at a time. And the hand-written representations were brittle outside the cases their authors had imagined.

**Ideal agents.** The theorist's version keeps the structure but takes away the resource limits.

- *Expected-utility maximization.* The von Neumann–Morgenstern theorem says that preferences over lotteries satisfying completeness, transitivity, continuity and independence can be represented as maximizing the expectation of a utility function $U$, unique up to positive affine transformation. "Coherence arguments" run this backward: an agent that is not representable this way can be exploited (Question 1.1), so a sufficiently capable agent will look like an expected-utility maximizer.
- *Solomonoff induction.* Put a prior on computable hypotheses weighted by $2^{-\text{program length}}$ and update by Bayes. For any computable data source $\mu$, its cumulative expected excess log-loss over all time is at most about $K(\mu)$ bits, the length of the shortest program for $\mu$, which makes it the ideal inductive learner. It is uncomputable. WB4 takes it apart.
- *AIXI* (Hutter) combines Solomonoff prediction with expectimax planning over future rewards: the ideal reinforcement learner. Legg and Hutter define a universal intelligence measure in its image.
- *Proof-based agents.* An agent that takes an action only when it can prove the action is good. Schmidhuber's [Gödel machine](https://arxiv.org/abs/cs/0309048) rewrites its own code when it can prove the rewrite helps. MIRI studied what such agents can prove about themselves and their successors: the *Löbian obstacle* (Question 1.3) says an agent cannot, in general, trust a successor that reasons in the same formal system, and *modal agents* use provability logic to cooperate in a one-shot prisoner's dilemma with programs that can read each other's source.

These idealizations treat the agent as separate from its environment, with a fixed and correct ontology, unlimited compute, and a utility function written in that ontology. *Embedded agency* (Demski and Garrabrant) lists what breaks when the agent is a piece of the world it is reasoning about. Learned agents broke most of the assumptions at once, as sections 3 and 4 describe.
"""
    ),
    markdown(
        "### Readings\n\n"
        + readings(
            reading(
                "Computer science as empirical inquiry: symbols and search",
                "https://doi.org/10.1145/360018.360022",
                "Allen Newell and Herbert Simon, 1976 (Turing Award lecture)",
                "States the physical symbol system hypothesis and the heuristic search hypothesis: the clearest manifesto of the GOFAI bet, by two of its founders.",
                about="A free scan is at the [Computer History Museum](https://www.computerhistory.org/chess/doc-431614f691642/).",
                importance=3,
                length=2,
            ),
            reading(
                "Programs with common sense",
                "http://www-formal.stanford.edu/jmc/mcc59/mcc59.html",
                "John McCarthy, 1959",
                "Proposes the \"advice taker\": a program that represents what it knows as logical sentences, decides what to do by deduction, and is improved by being told new sentences rather than reprogrammed. Short; the founding document of logic-based AI.",
                importance=2,
                length=2,
            ),
            reading(
                "Artificial intelligence: a general survey (the Lighthill report)",
                "http://www.chilton-computing.org.uk/inf/literature/reports/lighthill_report/p001.htm",
                "James Lighthill, 1972 (published 1973)",
                "The report that helped bring on the first AI winter in Britain. Judges general-purpose AI a failure and names the combinatorial explosion as the reason. A good read for what the critics of 1972 got right and wrong.",
                importance=2,
                length=2,
            ),
            reading(
                "Solomonoff's theory of inductive inference",
                "https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference",
                "Wikipedia",
                "The universal prior, the completeness and uncomputability results, and the connection to Kolmogorov complexity, at encyclopedia length.",
                about="WB4 goes further. The Wikipedia pages on the [von Neumann–Morgenstern utility theorem](https://en.wikipedia.org/wiki/Von_Neumann%E2%80%93Morgenstern_utility_theorem) and [AIXI](https://en.wikipedia.org/wiki/AIXI) are the matching short references for expected-utility maximization and the ideal RL agent.",
                importance=4,
                length=2,
            ),
            reading(
                "Universal intelligence: a definition of machine intelligence",
                "https://arxiv.org/abs/0712.3329",
                "Shane Legg and Marcus Hutter, 2007",
                "Defines intelligence as an agent's expected performance across all computable environments, weighted by their simplicity, and relates the definition to AIXI. The cleanest statement of what \"ideal intelligence\" meant to this tradition.",
                importance=2,
                length=4,
            ),
            reading(
                "Embedded agency",
                "https://www.alignmentforum.org/s/Rm6oQRJJmhGCcLvxh",
                "Abram Demski and Scott Garrabrant, 2018",
                "An illustrated sequence on what goes wrong when the agent is part of its environment: no clean input/output boundary, a world bigger than the agent's model of it, an agent that must reason about modifications to itself, and subsystems with goals of their own. The best map of MIRI-style agent foundations.",
                about="Also on [arXiv](https://arxiv.org/abs/1902.09469).",
                importance=4,
                length=3,
            ),
            reading(
                "Agent foundations for aligning machine intelligence with human interests: a technical research agenda",
                "https://intelligence.org/files/TechnicalAgenda.pdf",
                "Nate Soares and Benja Fallenstein, 2014 (revised 2017)",
                "MIRI's research agenda: realistic world models, decision theory, logical uncertainty, reasoning about successors (\"Vingean reflection\"), error tolerance and value specification. A snapshot of what alignment research looked like before deep learning.",
                importance=2,
                length=2,
            ),
            reading(
                "Tiling agents for self-modifying AI, and the Löbian obstacle",
                "https://intelligence.org/files/TilingAgentsDraft.pdf",
                "Eliezer Yudkowsky and Marcello Herreshoff, 2013",
                "Models self-modification as an agent approving a successor that shares its goals, and shows that Löb's theorem gets in the way: a proof-based agent seems able to trust only successors with weaker proof systems. Some workarounds; the general problem left open.",
                importance=1,
                length=4,
            ),
            reading(
                "Robust cooperation in the prisoner's dilemma: program equilibrium via provability logic",
                "https://arxiv.org/abs/1401.5577",
                "Mihaly Barasz, Paul Christiano, Benja Fallenstein, Marcello Herreshoff, Patrick LaVictoire, Eliezer Yudkowsky, 2014",
                "Programs that can read each other's source code play a one-shot prisoner's dilemma. Using provability logic, the paper builds \"modal agents\" that cooperate with each other without needing identical code, and never cooperate with a defector. Proof-based agents doing something no ordinary agent can.",
                importance=1,
                length=3,
            ),
            reading(
                "Functional decision theory: a new theory of instrumental rationality",
                "https://arxiv.org/abs/1710.05060",
                "Eliezer Yudkowsky and Nate Soares, 2017",
                "MIRI's answer to how an agent that knows it is a program should decide: treat your decision as the output of a fixed mathematical function, and choose the output that gives the best outcome wherever that function is computed (including in copies and predictions of you). Argued against causal and evidential decision theory on Newcomb's problem and its relatives.",
                importance=1,
                length=3,
            ),
            reading(
                "Coherence arguments do not entail goal-directed behavior",
                "https://www.alignmentforum.org/posts/NxF5G6CJiof6cemTw/coherence-arguments-do-not-entail-goal-directed-behavior",
                "Rohin Shah, 2018",
                "Any behaviour maximizes some utility function over histories, so \"capable agents are expected-utility maximizers\" constrains nothing by itself; what matters is whether the utility is simple and over world states. Read after Question 1.2.",
                importance=3,
                length=1,
            ),
        )
        + references(
            book(
                "Artificial Intelligence: A Modern Approach (4th ed.)",
                "https://aima.cs.berkeley.edu/",
                "Russell and Norvig, 2020",
                "ch. 2–4, 7, 11, 16 (ch. 17 on MDPs, for §2)",
            ),
        )
    ),
    markdown(
        r"""
### Questions

**Question 1.1 — the money pump.** An agent holds item $A$. Its preferences are strict and cyclic, $A\prec B\prec C\prec A$, and it will pay any amount less than $\varepsilon>0$ to swap an item for one it strictly prefers. Show that a trader can take an arbitrary amount of money from it. Which VNM axiom fails? Does this argument say anything about which *utility function* a capable agent will have?
"""
    ),
    answer(
        r"""
Offer $B$ for $A$ plus $\varepsilon/2$: the agent accepts. Offer $C$ for $B$ plus $\varepsilon/2$, then $A$ for $C$ plus $\varepsilon/2$. The agent is back where it started, $3\varepsilon/2$ poorer, and the trader can repeat the cycle as often as it likes. The failing axiom is **transitivity**: from $A\prec B$ and $B\prec C$ it would follow that $A\prec C$, contradicting $C\prec A$.

The argument says an agent that cannot be exploited this way has transitive preferences. Exploitation arguments for the other axioms exist but are weaker and contested (completeness and independence are argued over; continuity has essentially none), so "non-exploitable implies expected-utility maximizer" is a heuristic, not a theorem. Even granting it, the conclusion is only that the preferences have *some* expected-utility representation. It says nothing about which $U$, or about what $U$ is defined over. Question 1.2 shows that the second point matters a lot.
"""
    ),
    markdown(
        r"""
**Question 1.2 — every policy is an expected-utility maximizer.** Let $\pi$ be any deterministic policy in any environment, mapping each history $h_t=(o_1,a_1,\dots,o_t)$ to an action. Construct a utility function $U$ on complete histories for which $\pi$ maximizes expected utility. What, then, does the statement "a capable AI will be an expected-utility maximizer" actually claim?
"""
    ),
    answer(
        r"""
Let $U(h)=1$ if every action in $h$ is the one $\pi$ would have taken given the history before it, and $U(h)=0$ otherwise. Following $\pi$ gives $U=1$ with probability one, which is the largest possible value, so $\pi$ maximizes expected $U$ in every environment. (For a stochastic policy, use $U(h)=1$ when every action is in the support of $\pi$ given the preceding history.)

So "is an expected-utility maximizer" is empty when $U$ may be any function of the whole history. The claim behind classical alignment arguments is stronger: that capable agents behave as if maximizing a *simple* utility over *states of the world*, which is what produces instrumental convergence (§3). That is a substantive claim about which agents training produces, not a theorem. Shah's post argues this; the 2016–2026 sections ask what learned agents actually look like.
"""
    ),
    markdown(
        r"""
**Question 1.3 (rabbit hole) — the Löbian obstacle.** Let $T$ be a consistent formal theory extending Peano arithmetic, and write $\Box\varphi$ for "$T$ proves $\varphi$". **Löb's theorem:** if $T\vdash\Box\varphi\to\varphi$, then $T\vdash\varphi$. An agent $A_1$ reasons in $T$ and takes an action only when $T$ proves the action is safe. It considers building a successor $A_2$ that also reasons in $T$, and to approve it, $A_1$ wants to prove, for each possible action $b$, "if $A_2$ takes $b$ then $b$ is safe", which follows from the soundness statement $\Box\,\mathrm{safe}(b)\to\mathrm{safe}(b)$. Why can't $A_1$ prove these soundness statements? What does that suggest about chains of successors?
"""
    ),
    answer(
        r"""
Suppose $T\vdash\Box\,\mathrm{safe}(b)\to\mathrm{safe}(b)$ for every $b$. By Löb's theorem, $T\vdash\mathrm{safe}(b)$ for every $b$, including actions that are in fact unsafe. So either $T$ proves false things about safety, or the soundness statements are unprovable for every $b$ whose safety $T$ cannot already prove outright. In particular (taking $\varphi$ to be a contradiction) $T$ cannot prove its own consistency, which is Gödel's second incompleteness theorem as a special case.

So an agent can't approve a successor by trusting that the successor's proofs are sound, if the successor uses the same theory. It *can* trust a successor that reasons in a strictly weaker theory $T'$, if $T$ is strong enough to prove $\Box_{T'}\varphi\to\varphi$ (as set theory proves that whatever Peano arithmetic proves about numbers is true). Naively, then, each self-modification loses some strength. Yudkowsky and Herreshoff build infinite descending families of theories, each still extending Peano arithmetic, and other partial workarounds exist, but each has costs. This is the Löbian obstacle. It is a problem about agents that act on proofs, which is one reason it looked central when ideal agents were the model of AI and looks less central now.
"""
    ),
    # ------------------------------------------------------------------ §2
    markdown(
        r"""
<a id="wb1-section-2"></a>

## 2. From Bellman to AlphaZero

**Pitch.** Dynamic programming, value-based reinforcement learning and deep RL are one idea at three levels of approximation: solve the Bellman optimality equation, first exactly with a known model, then from samples, then with a neural network in place of the table. Policy-gradient methods, the branch that RL on language models uses, are the other route, and still lean on the Bellman *evaluation* equation for their critics. Game-playing systems show the bitter lesson inside that family: Deep Blue was search plus a hand-written evaluation function, AlphaZero was search plus a learned one. The section ends with a bonus derivation that turns planning into probabilistic inference.

### 2.A. The Bellman optimality equation and dynamic programming

A Markov decision process has states $S$, actions $A$, transition probabilities $P(s'\mid s,a)$, a reward $r(s,a)$ and a discount $\gamma\in[0,1)$. The optimal value function satisfies the **Bellman optimality equation**

$$
V^*(s)=\max_{a\in A}\Big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^*(s')\Big],
$$

and the optimal action-value function satisfies

$$
Q^*(s,a)=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q^*(s',a').
$$

Both say Bellman's *principle of optimality*: whatever the first decision, the rest of an optimal policy is optimal from wherever that decision leads. Acting greedily with respect to $V^*$ or $Q^*$ is optimal. Write the right-hand sides as operators, $V^*=\mathcal T V^*$ and $Q^*=\mathcal T Q^*$.

*Dynamic programming* solves the equation with a known model and a table indexed by states:

- **Value iteration** repeats $V_{k+1}=\mathcal T V_k$. It converges from any starting point because $\mathcal T$ is a contraction (Exercise 2.1).
- **Policy iteration** alternates *evaluation* (solve the linear Bellman equation $V^\pi=r_\pi+\gamma P_\pi V^\pi$ for the current policy) with *improvement* (make $\pi$ greedy with respect to $V^\pi$). Each improvement step is at least as good as the last, and with finitely many policies it terminates at an optimal one.

Both need the model $P$ and a sweep over every state. That is the curse of dimensionality: $k$ state variables with $m$ values each give $m^k$ table entries.

### 2.B. Reinforcement learning: dynamic programming from samples

Reinforcement learning drops the model. An agent sees transitions $(s,a,r,s')$ and replaces each expectation over $s'$ by a sample, nudging a table toward the sampled target with step size $\alpha$:

- **TD(0)** for policy evaluation: $V(s)\leftarrow V(s)+\alpha\,[\,r+\gamma V(s')-V(s)\,]$, a stochastic approximation to the Bellman equation for $V^\pi$.
- **Q-learning** (Watkins) for control: $Q(s,a)\leftarrow Q(s,a)+\alpha\,[\,r+\gamma\max_{a'}Q(s',a')-Q(s,a)\,]$, a stochastic approximation to the Bellman *optimality* equation for $Q^*$. It is *off-policy*: the transitions may come from any exploring behaviour policy. Exercise 2.2 shows why it has to be $Q$ and not $V$.
- **Policy gradient and actor-critic** methods take the other branch: parametrize the policy $\pi_\theta$ directly and follow the gradient of expected return $J(\theta)$. The REINFORCE estimator is
$$
\nabla_\theta J=\mathbb E_{\tau\sim\pi_\theta}\Big[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(G_t-b(s_t)\big)\Big],
$$
where $G_t$ is the return from time $t$ and the baseline $b$ reduces variance without adding bias. Actor-critic methods learn $b\approx V^\pi$ with TD, which solves the Bellman *evaluation* equation, not the optimality one.

This branch is the one RL on language models uses (WB6 §11). RLHF trains with PPO, an actor-critic method. RL with verifiable rewards often treats a whole completion as one action, a one-step bandit, and replaces the learned critic with the mean reward of a group of samples for the same prompt (GRPO). In that one-step setting, the KL-regularized objective has the closed-form solution of Exercise B.5.

### 2.C. Deep RL: replace the table with a network

Replacing the table by a function approximator $V_\theta$ or $Q_\theta$ beats the curse of dimensionality when the network generalizes across states. It also breaks the convergence guarantees. Function approximation, bootstrapping (targets built from the current estimate) and off-policy data are the "deadly triad" (Sutton and Barto's term): together they can make TD learning diverge, and there are simple counterexamples. It worked anyway, early and spectacularly, in Tesauro's TD-Gammon (1992–95), a neural network trained by TD learning on self-play that reached near world-champion backgammon. It took until DQN (2015) to make it work generally, with experience replay and a slowly updated target network, on 49 Atari games from pixels.

### 2.D. Search and learning: Deep Blue, AlphaGo, AlphaZero

| | search | evaluation of positions | where the knowledge came from |
|---|---|---|---|
| **Deep Blue** (1997) | alpha-beta minimax on custom hardware, around $10^8$ positions per second | a hand-written function of about 8,000 features, weights tuned largely by hand | grandmasters and engineers; opening book and endgame tables |
| **AlphaGo** (2016) | Monte Carlo tree search | policy network (trained on about 30 million positions from human games, then improved by self-play) and value network, plus fast rollouts | human games, then self-play |
| **AlphaGo Zero / AlphaZero** (2017) | MCTS, tens of thousands of positions per second in chess, against tens of millions for Stockfish | one network with policy and value heads, trained only by self-play | the rules and the reward (win/lose) |
| **MuZero** (2019) | MCTS inside a *learned* model | learned | the reward; even the rules are learned |

Read down the table as the bitter lesson in miniature. Both search and learning scale. What lost was the hand-written evaluation function, and then the hand-supplied simulator. AlphaZero searched a thousand times fewer positions than the 2017 Stockfish and still won, because its evaluation was better. The coda: Stockfish regained the lead in 2020 by replacing its hand-written evaluation with a small learned network (NNUE), keeping its fast search. Search plus learning beat each alone. Exercise 2.3 shows that AlphaZero's training loop is policy iteration from §2.A, with tree search as the improvement step.
"""
    ),
    markdown(
        "### Readings\n\n"
        + readings(
            reading(
                "Temporal difference learning and TD-Gammon",
                "https://dl.acm.org/doi/10.1145/203330.203343",
                "Gerald Tesauro, 1995",
                "A neural network that learned backgammon by TD learning from self-play, with almost no built-in knowledge, and reached the level of the best human players. Deep RL twenty years early, and a clean early instance of the bitter lesson.",
                about="Free full text [here](https://bkgm.com/articles/tesauro/tdl.html).",
                importance=2,
                length=2,
            ),
            reading(
                "Human-level control through deep reinforcement learning",
                "https://www.nature.com/articles/nature14236",
                "Volodymyr Mnih et al., 2015",
                "DQN: one network architecture and one set of hyperparameters, learning 49 Atari games from pixels and score, at a level comparable to a professional games tester on many of them. Experience replay and a target network are what make Q-learning with a deep network stable enough.",
                importance=4,
                length=2,
            ),
            reading(
                "Deep Blue",
                "https://doi.org/10.1016/S0004-3702(01)00129-1",
                "Murray Campbell, A. Joseph Hoane Jr., Feng-hsiung Hsu, 2002",
                "The engineering paper on the machine that beat Kasparov: the custom chess chips, the parallel search, and the large hand-built evaluation function. Read alongside AlphaZero for the contrast.",
                importance=2,
                length=3,
            ),
            reading(
                "Mastering the game of Go with deep neural networks and tree search",
                "https://www.nature.com/articles/nature16961",
                "David Silver et al., 2016",
                "AlphaGo: a policy network trained on human games and refined by self-play, a value network, and Monte Carlo tree search combining them. Beat a professional Go player, a decade before most predictions.",
                importance=2,
                length=2,
            ),
            reading(
                "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play",
                "https://doi.org/10.1126/science.aar6404",
                "David Silver et al., 2018 (preprint on [arXiv](https://arxiv.org/abs/1712.01815), 2017)",
                "AlphaZero: the AlphaGo Zero recipe (no human data, one network, MCTS as the improvement operator) applied unchanged to three games, beating the strongest existing programs in each. Pair it with [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero, 2017).",
                importance=5,
                length=3,
            ),
            reading(
                "Mastering Atari, Go, chess and shogi by planning with a learned model",
                "https://arxiv.org/abs/1911.08265",
                "Julian Schrittwieser et al., 2019 (Nature, 2020)",
                "MuZero: the same search, but inside a learned model that predicts only what planning needs (reward, value and policy), not the next observation. Removes the last hand-supplied ingredient, the simulator.",
                importance=2,
                length=3,
            ),
        )
        + references(
            book(
                "Reinforcement learning: an introduction (2nd ed.)",
                "http://incompleteideas.net/book/the-book-2nd.html",
                "Sutton and Barto, 2018",
                "ch. 3, 4, 6 (§2.A–B), §11.3, ch. 13, 16",
            ),
        )
    ),
    markdown(
        r"""
### Exercises

**Exercise 2.1 — value iteration converges.** Assume finitely many states and actions, and use the sup norm $\|V\|_\infty=\max_s|V(s)|$.

(a) Show that $\|\mathcal TV-\mathcal TW\|_\infty\le\gamma\,\|V-W\|_\infty$ for all $V,W$.

(b) Conclude that $\mathcal T$ has exactly one fixed point, and that value iteration satisfies $\|V_k-V^*\|_\infty\le\gamma^k\|V_0-V^*\|_\infty$.

(c) The same proof works for the Bellman *evaluation* operator $\mathcal T^\pi V=r_\pi+\gamma P_\pi V$ of a fixed policy. Why does it fail when $\gamma=1$, and what usually rescues the finite-horizon or episodic case?

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

**(c)** With $\gamma=1$ the bound is $\|\mathcal TV-\mathcal TW\|_\infty\le\|V-W\|_\infty$, which is not a contraction, and in general there may be no fixed point (rewards that never stop accumulating) or many (in a problem with no terminal states, $\mathcal T(V+c)=\mathcal TV+c$, so adding a constant to a fixed point gives another). A finite horizon $T$ rescues it: values are indexed by time, $V^*_{T+1}\equiv0$, and backward induction computes $V^*_T,\dots,V^*_1$ exactly in $T$ steps, which is the form used in §2.E. Episodic problems in which every policy reaches a terminal state are handled in the same spirit (Bertsekas's "stochastic shortest path" theory).
"""
    ),
    markdown(
        r"""
**Exercise 2.2 — why Q-learning learns $Q$.** A transition $(s,a,r,s')$ is sampled with $s'\sim P(\cdot\mid s,a)$.

(a) Show that the Q-learning target $y=r+\gamma\max_{a'}Q(s',a')$ satisfies $\mathbb E[y\mid s,a]=(\mathcal TQ)(s,a)$.

(b) Try the same for $V$: is there a sample-based target $y$, built from a single transition out of $s$, with $\mathbb E[y]=(\mathcal TV)(s)$? Where does the maximum sit relative to the expectation in each case?

(c) Suppose instead that the estimates are noisy: $\hat Q(s',a')=Q(s',a')+\epsilon_{a'}$ with independent zero-mean $\epsilon_{a'}$. Is $\mathbb E\,\max_{a'}\hat Q(s',a')$ larger or smaller than $\max_{a'}Q(s',a')$? What does this do to Q-learning, and how might you fix it?
"""
    ),
    answer(
        r"""
**(a)** Given $(s,a)$, the reward is $r(s,a)$ and only $s'$ is random, so

$$
\mathbb E[y\mid s,a]=r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\max_{a'}Q(s',a')=(\mathcal TQ)(s,a).
$$

So the Q-learning update is a Robbins–Monro stochastic approximation to the fixed point of $\mathcal T$, which is a $\gamma$-contraction on $Q$-tables by the same argument as Exercise 2.1. With step sizes satisfying $\sum\alpha=\infty$, $\sum\alpha^2<\infty$ and every $(s,a)$ visited infinitely often, the table converges to $Q^*$ (Watkins and Dayan, 1992), whatever behaviour policy generated the data.

**(b)** $(\mathcal TV)(s)=\max_a\big[r(s,a)+\gamma\,\mathbb E_{s'\sim P(\cdot\mid s,a)}V(s')\big]$ has the maximum *outside* the expectation. A single transition shows what happened after one action, and says nothing about the actions not taken, so there is no unbiased one-sample estimate of the max over actions of an expectation. Without a model you can evaluate a fixed policy (TD(0)), but you cannot do the optimality backup for $V$. For $Q$, the action is already chosen and the maximum is over the *next* action, *inside* the expectation over $s'$, where it can be computed from the table. That is why model-free control learns $Q$.

**(c)** Larger, in general: $\max$ is convex, so by Jensen $\mathbb E\max_{a'}\hat Q(s',a')\ge\max_{a'}\mathbb E\hat Q(s',a')=\max_{a'}Q(s',a')$, with strict inequality whenever the noise can change which action wins. Q-learning bootstraps on its own maxima, so the upward bias propagates and compounds: *maximization bias*. The standard fix is **double Q-learning** (van Hasselt, 2010; Double DQN, 2016): keep two estimates, choose the maximizing action with one and evaluate it with the other, so the selection noise and the evaluation noise are independent. The same mechanism (optimizing against a noisy estimate selects for its errors) is one cause of reward-model overoptimization in §4; hacks such as editing a test exploit systematic gaps in the reward rather than noise.
"""
    ),
    markdown(
        r"""
**Exercise 2.3 — AlphaZero is policy iteration.** AlphaZero has one network $f_\theta(s)=(\mathbf p,v)$: a prior over moves and a value estimate. From a position $s$, MCTS runs simulations that pick moves using $\mathbf p$, evaluate leaves with $v$, and back values up the tree. It returns visit-count frequencies $\boldsymbol\pi(s)$. Self-play picks moves from $\boldsymbol\pi$; at the end, every position gets the game outcome $z\in\{-1,0,+1\}$ from the point of view of the player to move. Training minimizes

$$
(z-v)^2-\boldsymbol\pi^{\top}\log\mathbf p+c\,\|\theta\|^2 .
$$

(a) Which part of the loop is policy improvement, and which is policy evaluation, in the sense of §2.A? What role does the policy head $\mathbf p$ play that has no counterpart in tabular policy iteration?

(b) Where is the Bellman backup?

(c) Using the table in §2.D: what in AlphaZero is still written by hand? Which of those did MuZero remove, and which remain in every system in the table?
"""
    ),
    answer(
        r"""
**(a)** *Improvement:* MCTS. Starting from the network's prior $\mathbf p$ and value $v$, a lookahead search produces a move distribution $\boldsymbol\pi$ that is usually stronger than $\mathbf p$, just as acting greedily on a lookahead improves on the current policy. *Evaluation:* the value head is regressed onto $z$, the outcome of games played by the improved policy, which is a Monte Carlo estimate of that policy's value. The *policy head* is amortization: it is trained to imitate the search, $\mathbf p\approx\boldsymbol\pi$, so that next round's search starts from a better prior and the improvement compounds. Tabular policy iteration has no need for this, because it can store the improved policy exactly; a network has to be taught it. Silver et al. describe MCTS as "a powerful policy improvement operator" and self-play with search as "a powerful policy evaluation operator".

**(b)** Inside the tree. Each simulation backs the leaf value up the path to the root, averaging values at each node (with signs alternating between the players). That is an approximate minimax backup, the two-player version of $\max_a[r+V(s')]$, computed by sampling rather than full expansion. The network supplies the leaf values, so the tree is a Bellman backup bootstrapped from a learned $V$.

**(c)** Still hand-written in AlphaZero: the rules (a perfect simulator, and with it a Markov state), the reward (win, lose or draw), the input encoding, the network architecture and the search algorithm. MuZero learns the model, so it no longer needs the rules at search time, though the environment still provides them during training. The reward is supplied by hand in every system in the table. That last point is where §3 picks up: learning took over the heuristic and then the model, but not the goal.
"""
    ),
    markdown(
        r"""
<a id="wb1-section-2e"></a>

### 2.E. Bonus: planning as inference

**The result.** Condition a model of trajectories on binary "optimality" variables with likelihood $e^{r}$. The posterior reweights the trajectory prior by $e^{R(\tau)}$; with deterministic dynamics and a fixed start state it is a Boltzmann distribution over feasible plans whose most probable trajectory is the classical optimal plan (Exercise B.1). Its log-messages obey a *soft* Bellman equation (B.2), which under stochastic dynamics is optimistic (B.3). Holding the dynamics fixed gives variational inference, whose objective is maximum-entropy RL (B.4), and the one-step form is KL-regularized control, which has the form of a Bayesian update (B.5). This is a common formulation of planning and maximum-entropy RL under these assumptions.

"""
        + reading(
            "Reinforcement learning and control as probabilistic inference: tutorial and review",
            "https://arxiv.org/abs/1805.00909",
            "Sergey Levine, 2018",
            "Maximum-entropy RL is equivalent to exact probabilistic inference in a graphical model with optimality variables when the dynamics are deterministic, and to variational inference when they are stochastic; maximum-entropy policy gradients, actor-critic methods and soft Q-learning then fall out as approximate inference.",
            about="The exercises follow §2.2–2.4 and §3.1–3.2; §4 turns the result into soft Q-learning and actor-critic algorithms; §5 reviews earlier work connecting inference and control.",
            importance=5,
            length=3,
        )
        + r"""

The exercises build on each other, but each states what it uses. If you only do some, Exercises B.3 and B.5 have the most payoff.

**The model.** Take a finite-horizon MDP with finitely many states and actions (sums in place of Levine's integrals), and trajectories $\tau=(s_1,a_1,\dots,s_T,a_T)$. The "prior" over trajectories is the dynamics together with a uniform action prior $p(a\mid s)=1/|A|$:

$$
p(\tau)=p(s_1)\prod_{t=1}^{T-1}p(s_{t+1}\mid s_t,a_t)\,\prod_{t=1}^{T}\frac{1}{|A|}.
$$

At each step add a binary variable $\mathcal O_t$ ("step $t$ was optimal") with

$$
p(\mathcal O_t=1\mid s_t,a_t)=e^{\,r(s_t,a_t)} .
$$

For this to be a probability we need $r\le0$. Subtracting the maximum reward from every reward arranges that, and since every trajectory has $T$ steps it shifts every total reward by the same amount. (Levine §2.5 shows how to drop the assumption entirely.) Write $\mathcal O$ for the event $\mathcal O_1=\dots=\mathcal O_T=1$, write $\mathcal O_{t:T}$ for $\mathcal O_t=\dots=\mathcal O_T=1$, and let $R(\tau)=\sum_{t=1}^T r(s_t,a_t)$.

For comparison, the hard **Bellman optimality equation** for this finite-horizon, undiscounted problem is

$$
V^*_t(s)=\max_{a\in A}\Big[r(s,a)+\sum_{s'}p(s'\mid s,a)\,V^*_{t+1}(s')\Big],\qquad V^*_{T+1}\equiv0,
$$

the finite-horizon counterpart of the discounted equation in §2.A, solved by backward induction (Exercise 2.1(c)). Exercises B.2(d) and B.5(b) ask when the soft recursions reduce to it.
"""
    ),
    markdown(
        r"""
**Exercise B.1 — the posterior over trajectories.**

(a) Show that $p(\tau\mid\mathcal O)\propto p(s_1)\prod_{t=1}^{T-1}p(s_{t+1}\mid s_t,a_t)\;e^{R(\tau)}$.

(b) Suppose the dynamics are deterministic, $s_{t+1}=f(s_t,a_t)$, and the start state is fixed. Describe $p(\tau\mid\mathcal O)$. Which trajectory is most probable?

(c) Introduce a temperature $\alpha>0$ by using $r/\alpha$ in place of $r$. What happens to the posterior in (b) as $\alpha\to0$?

<details><summary>Hint</summary>

Bayes' rule: $p(\tau\mid\mathcal O)=p(\mathcal O\mid\tau)\,p(\tau)/p(\mathcal O)$. Given the whole trajectory, each $\mathcal O_t$ depends only on $(s_t,a_t)$. Which factors are the same for every $\tau$?
</details>
"""
    ),
    answer(
        r"""
**(a)** By Bayes' rule, $p(\tau\mid\mathcal O)=p(\mathcal O\mid\tau)\,p(\tau)/p(\mathcal O)$. Given $\tau$, the $\mathcal O_t$ are independent and each depends only on $(s_t,a_t)$, so

$$
p(\mathcal O\mid\tau)=\prod_{t=1}^T e^{r(s_t,a_t)}=e^{R(\tau)} .
$$

The action-prior factor $|A|^{-T}$ and the evidence $p(\mathcal O)$ are the same for every $\tau$, so

$$
p(\tau\mid\mathcal O)=\frac{|A|^{-T}}{p(\mathcal O)}\,\Big[p(s_1)\prod_{t=1}^{T-1}p(s_{t+1}\mid s_t,a_t)\Big]\,e^{R(\tau)}
\;\propto\;\Big[p(s_1)\prod_{t=1}^{T-1}p(s_{t+1}\mid s_t,a_t)\Big]\,e^{R(\tau)} .
$$

This is Levine's eq. (4): the probability of the trajectory under the dynamics, times its exponentiated total reward. Keep the exact normalized form; Exercise B.4 uses it.

**(b)** With deterministic dynamics and a fixed start, the bracket is $1$ if $\tau$ is feasible ($s_1$ is the start state and every $s_{t+1}=f(s_t,a_t)$) and $0$ otherwise. So

$$
p(\tau\mid\mathcal O)=\frac{\mathbb 1[\tau\text{ feasible}]\;e^{R(\tau)}}{\sum_{\tau'\,\text{feasible}}e^{R(\tau')}} ,
$$

a Boltzmann distribution over feasible plans with energy $-R$. The most probable trajectory is the feasible plan with the largest total reward, which is exactly the classical optimal plan. **Classical planning is MAP inference in this model.**

**(c)** With $r/\alpha$ the posterior on feasible plans is $\propto e^{R(\tau)/\alpha}$. For two feasible plans with $R(\tau)<R(\tau^*)$,

$$
\frac{p(\tau\mid\mathcal O)}{p(\tau^*\mid\mathcal O)}=e^{-\left(R(\tau^*)-R(\tau)\right)/\alpha}\;\longrightarrow\;0\qquad(\alpha\to0),
$$

so the posterior concentrates, uniformly, on the set of optimal plans. At $\alpha>0$ the full posterior is "soft" planning: near-optimal plans keep weight in proportion to $e^{R/\alpha}$.
"""
    ),
    markdown(
        r"""
**Exercise B.2 — backward messages and the soft Bellman equation.** Define the backward messages

$$
\beta_t(s,a)=p(\mathcal O_{t:T}\mid s_t=s,a_t=a),\qquad \beta_t(s)=p(\mathcal O_{t:T}\mid s_t=s).
$$

(a) Show that $\beta_T(s,a)=e^{r(s,a)}$ and, for $t<T$,

$$
\beta_t(s,a)=e^{r(s,a)}\sum_{s'}p(s'\mid s,a)\,\beta_{t+1}(s'),\qquad \beta_t(s)=\frac{1}{|A|}\sum_a\beta_t(s,a).
$$

(b) Show that the posterior policy is $p(a_t=a\mid s_t=s,\mathcal O)=\beta_t(s,a)\big/\sum_{a'}\beta_t(s,a')$.

(c) Define $Q_T=r$ and, going backward,

$$
V_t(s)=\log\sum_a e^{Q_t(s,a)},\qquad Q_{t}(s,a)=r(s,a)+\log\sum_{s'}p(s'\mid s,a)\,e^{V_{t+1}(s')}.
$$

Show that $Q_t(s,a)=\log\beta_t(s,a)+c_t$ for a constant $c_t$ that does not depend on $s$ or $a$, and that the posterior policy is $e^{Q_t(s,a)-V_t(s)}$. These are Levine's soft $Q$- and value functions.

(d) Compare with the Bellman optimality equation in §2.E. Put in a temperature by using $r/\alpha$, and rescale $\tilde Q=\alpha Q$, $\tilde V=\alpha V$, so that

$$
\tilde V_t(s)=\alpha\log\sum_a e^{\tilde Q_t(s,a)/\alpha},\qquad \tilde Q_t(s,a)=r(s,a)+\alpha\log\sum_{s'}p(s'\mid s,a)\,e^{\tilde V_{t+1}(s')/\alpha}.
$$

What do the two soft operations become as $\alpha\to0$, first for deterministic dynamics and then for stochastic dynamics?

<details><summary>Hint for (a) and (b)</summary>

For (a), split $\mathcal O_{t:T}$ into $\mathcal O_t$ and $\mathcal O_{t+1:T}$ and sum over $s_{t+1}$. Given $(s_t,a_t)$, $\mathcal O_t$ is independent of everything later, and everything later depends on $(s_t,a_t)$ only through $s_{t+1}$. For (b), apply Bayes' rule to $a_t$ with $s_t$ fixed, then argue that once you know $s_t$, $\mathcal O_{1:t-1}$ is independent of $(a_t,\mathcal O_{t:T})$.
</details>

<details><summary>Hint for (c)</summary>

Induct backward from $t=T$. If $Q_{t+1}=\log\beta_{t+1}+c_{t+1}$, what is $V_{t+1}$ in terms of $\beta_{t+1}(s')$?
</details>

<details><summary>Hint for (d)</summary>

For numbers $x_1,\dots,x_n$ and weights $p_i>0$: $\;\max_i x_i+\alpha\log\min_i p_i\;\le\;\alpha\log\sum_i p_i e^{x_i/\alpha}\;\le\;\max_i x_i+\alpha\log\sum_i p_i$.
</details>
"""
    ),
    answer(
        r"""
**(a)** At $t=T$ there is one optimality variable, so $\beta_T(s,a)=p(\mathcal O_T=1\mid s,a)=e^{r(s,a)}$. For $t<T$,

$$
\begin{aligned}
\beta_t(s,a)&=p(\mathcal O_t\mid s,a)\;p(\mathcal O_{t+1:T}\mid s,a)
&&\text{($\mathcal O_t$ is independent of the future given $s_t,a_t$)}\\
&=e^{r(s,a)}\sum_{s'}p(s'\mid s,a)\;p(\mathcal O_{t+1:T}\mid s_{t+1}=s')
&&\text{(the future depends on $s_t,a_t$ only through $s_{t+1}$)}\\
&=e^{r(s,a)}\sum_{s'}p(s'\mid s,a)\;\beta_{t+1}(s').
\end{aligned}
$$

Averaging over the action prior gives $\beta_t(s)=\sum_a p(a\mid s)\,\beta_t(s,a)=\frac1{|A|}\sum_a\beta_t(s,a)$. This is sum-product message passing backward in time, as in an HMM (Levine eq. 6).

**(b)** Given $s_t$, everything earlier (including $\mathcal O_{1:t-1}$) is independent of $a_t$ and of everything after it: the dynamics are Markov and the action prior looks only at $s_t$. So $p(a_t\mid s_t,\mathcal O)=p(a_t\mid s_t,\mathcal O_{t:T})$, and Bayes' rule in $a_t$ gives

$$
p(a\mid s,\mathcal O_{t:T})=\frac{p(\mathcal O_{t:T}\mid s,a)\,p(a\mid s)}{p(\mathcal O_{t:T}\mid s)}
=\frac{\beta_t(s,a)/|A|}{\frac1{|A|}\sum_{a'}\beta_t(s,a')}
=\frac{\beta_t(s,a)}{\sum_{a'}\beta_t(s,a')} .
$$

**(c)** Induct backward. At $t=T$, $Q_T=r=\log\beta_T$, so $c_T=0$. Suppose $Q_{t+1}=\log\beta_{t+1}+c_{t+1}$. Then, using (a),

$$
V_{t+1}(s')=\log\sum_a\beta_{t+1}(s',a)+c_{t+1}=\log\beta_{t+1}(s')+\log|A|+c_{t+1},
$$

and substituting into the definition of $Q_t$,

$$
Q_t(s,a)=r(s,a)+\log\sum_{s'}p(s'\mid s,a)\,\beta_{t+1}(s')+\log|A|+c_{t+1}=\log\beta_t(s,a)+c_t,
$$

with $c_t=c_{t+1}+\log|A|$, so $c_t=(T-t)\log|A|$. The constant is the action prior's $1/|A|$, which Levine also drops. It does not depend on $(s,a)$, so it cancels in the policy from (b):

$$
p(a\mid s,\mathcal O)=\frac{e^{Q_t(s,a)-c_t}}{\sum_{a'}e^{Q_t(s,a')-c_t}}=\frac{e^{Q_t(s,a)}}{e^{V_t(s)}}=e^{Q_t(s,a)-V_t(s)} .
$$

**(d)** *Over actions.* By the hint with all $p_i=1$, $\max_a\tilde Q_t\le\tilde V_t\le\max_a\tilde Q_t+\alpha\log|A|$: the log-sum-exp is a soft max, within $\alpha\log|A|$ of the hard one.

*Deterministic dynamics.* The sum over $s'$ has one term, so $\tilde Q_t(s,a)=r(s,a)+\tilde V_{t+1}(f(s,a))$ exactly. For deterministic dynamics the hard recursion of §2.E reads $V^*_t(s)=\max_a Q^*_t(s,a)$ with $Q^*_t(s,a)=r(s,a)+V^*_{t+1}(f(s,a))$ and $V^*_{T+1}\equiv0$. Since $\tilde Q_t$ itself depends on $\alpha$, take the limit by backward induction. At $t=T$, $\tilde Q_T=Q^*_T=r$. If $0\le\tilde V_{t+1}-V^*_{t+1}\le(T-t)\,\alpha\log|A|$, then the same bounds hold for $\tilde Q_t-Q^*_t$, and the soft max adds at most one more $\alpha\log|A|$, so

$$
0\;\le\;\tilde V_t(s)-V^*_t(s)\;\le\;(T-t+1)\,\alpha\log|A|\;\longrightarrow\;0 .
$$

So for deterministic dynamics the limit is the Bellman optimality equation of §2.E.

*Stochastic dynamics.* Let $p_{\min}$ be the smallest positive transition probability over all $(s,a)$, and apply the hint over the reachable next states, those with $p(s'\mid s,a)>0$:

$$
\max_{s':\,p(s'\mid s,a)>0}\tilde V_{t+1}(s')+\alpha\log p_{\min}\;\le\;\alpha\log\sum_{s'}p(s'\mid s,a)\,e^{\tilde V_{t+1}(s')/\alpha}\;\le\;\max_{s':\,p(s'\mid s,a)>0}\tilde V_{t+1}(s').
$$

Define the limiting recursion

$$
V^+_t(s)=\max_a\Big[r(s,a)+\max_{s':\,p(s'\mid s,a)>0}V^+_{t+1}(s')\Big],\qquad V^+_{T+1}\equiv0 .
$$

The same backward induction, now losing at most $\alpha\log(1/p_{\min})$ at each next-state backup and gaining at most $\alpha\log|A|$ at each soft max, gives

$$
-(T-t)\,\alpha\log(1/p_{\min})\;\le\;\tilde V_t(s)-V^+_t(s)\;\le\;(T-t+1)\,\alpha\log|A|\;\longrightarrow\;0 .
$$

So the backup tends to the best *reachable* next value, **not** the expected one. The zero-temperature limit is therefore not the Bellman equation of §2.E: the agent plans as if it also chose the outcome of every die roll. Even at $\alpha=1$, Jensen's inequality gives $\log\mathbb E\,e^{V}\ge\mathbb E\,V$, so the exact-inference backup is never less optimistic than the expected one. Levine §2.3 (around eq. 7) describes this as risk-seeking; Exercise B.3 makes it concrete.
"""
    ),
    markdown(
        r"""
**Exercise B.3 — inference is optimistic.** At some state the agent can take a *safe* action, which leads for certain to a state with soft value $0$, or a *gamble*, which leads with probability $p$ to a state with soft value $+R$ and otherwise to a state with soft value $-C$. Both actions have the same immediate reward, so ignore it. (Adding a constant to every value shifts both backups in (a) by that constant, so taking the safe value to be $0$ loses nothing.)

(a) Compute $Q(\text{gamble})$ under the exact-inference backup of Exercise B.2, $\log\mathbb E\,e^{V}$, and under the ordinary expected backup, $\mathbb E\,V$.

(b) Evaluate both for $p=0.01$, $R=10$, $C=100$. Which action does each prefer, and by what odds does the exact posterior policy $e^{Q-V}$ pick the gamble over the safe action?

(c) Compute the posterior probability that the gamble came out well, $p(s_{t+1}=\text{good}\mid s_t,a_t=\text{gamble},\mathcal O)$. What has the agent "inferred"?

(d) For large $C$, when does exact inference prefer the gamble? How does the answer depend on $C$?

<details><summary>Hint for (c)</summary>

By Bayes' rule and the Markov property, $p(s_{t+1}\mid s_t,a_t,\mathcal O)\propto p(s_{t+1}\mid s_t,a_t)\,\beta_{t+1}(s_{t+1})$, and by Exercise B.2(c), $\beta_{t+1}(s')\propto e^{V_{t+1}(s')}$.
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

Exact inference prefers the gamble: the posterior policy is a softmax of $Q$, so the odds are $e^{5.39}/e^{0}\approx220$ to $1$ in favour of the gamble. The expected backup rejects it; even a max-ent policy built on it gives odds of about $e^{-98.9}\approx10^{-43}$.

**(c)** Using the hint,

$$
p(\text{good}\mid s_t,\text{gamble},\mathcal O)=\frac{p\,e^{R}}{p\,e^{R}+(1-p)\,e^{-C}}=\frac{1}{1+99\,e^{-110}}\approx1-1.7\times10^{-46}.
$$

Conditioned on "everything went well", the agent has inferred that the gamble paid off. Exact inference tilts the dynamics as well as the actions, so the agent plans as if it could choose its luck. Levine §2.3–2.4 and the opening of §3 make the same point: under the exact posterior, $p(s_{t+1}\mid s_t,a_t,\mathcal O)\neq p(s_{t+1}\mid s_t,a_t)$.

**(d)** Exact inference prefers the gamble when $p\,e^{R}+(1-p)e^{-C}>1$. As $C\to\infty$ this becomes $p\,e^{R}>1$, i.e. $R>\log(1/p)$, which **does not depend on $C$ at all**. However large the catastrophe, any gamble whose win satisfies $R>\log(1/p)$ looks good, however improbable the win. The same caution applies to any scheme that conditions a model of the world on success. Exercise B.4 removes the problem by forbidding the agent to choose its luck.
"""
    ),
    markdown(
        r"""
**Exercise B.4 — the variational fix is maximum-entropy RL.** Restrict to trajectory distributions that use the *true* start-state distribution and dynamics, and let only the policy vary. The horizon is finite, so the policy may depend on $t$:

$$
q(\tau)=p(s_1)\prod_{t=1}^{T-1}p(s_{t+1}\mid s_t,a_t)\,\prod_{t=1}^{T}\pi_t(a_t\mid s_t).
$$

(a) Show that

$$
\mathrm{KL}\big(q\,\|\,p(\cdot\mid\mathcal O)\big)=-\,\mathbb E_q\Big[\sum_{t=1}^T\Big(r(s_t,a_t)+\mathcal H\big(\pi_t(\cdot\mid s_t)\big)\Big)\Big]+T\log|A|+\log p(\mathcal O),
$$

where $\mathcal H(\pi_t(\cdot\mid s))=-\sum_a\pi_t(a\mid s)\log\pi_t(a\mid s)$ is the policy's entropy at $s$.

(b) Conclude that minimizing this KL over the policies is the same as maximizing expected reward plus policy entropy (maximum-entropy RL), and that the expectation minus $T\log|A|$ is a lower bound on $\log p(\mathcal O)$.

(c) When is the minimum KL zero, so that max-ent RL is *exact* inference? Use Exercise B.3(c) to say why it is not, in general.

<details><summary>Hint</summary>

Write out $\log q(\tau)-\log p(\tau\mid\mathcal O)$ using the exact normalized posterior from Exercise B.1(a),

$$
p(\tau\mid\mathcal O)=\frac{|A|^{-T}}{p(\mathcal O)}\,p(s_1)\prod_{t=1}^{T-1}p(s_{t+1}\mid s_t,a_t)\,e^{R(\tau)} .
$$

The start-state and dynamics factors cancel. For the policy term, $\mathbb E_q[\log\pi_t(a_t\mid s_t)]$ averages $\sum_a\pi_t(a\mid s_t)\log\pi_t(a\mid s_t)$ over the distribution of $s_t$.
</details>
"""
    ),
    answer(
        r"""
**(a)** From Exercise B.1(a), $p(\tau\mid\mathcal O)=|A|^{-T}\,p(s_1)\prod_t p(s_{t+1}\mid s_t,a_t)\,e^{R(\tau)}/p(\mathcal O)$. Since $q$ uses the same $p(s_1)$ and the same dynamics, those factors cancel in the log-ratio:

$$
\log q(\tau)-\log p(\tau\mid\mathcal O)=\sum_{t=1}^T\log\pi_t(a_t\mid s_t)+T\log|A|-R(\tau)+\log p(\mathcal O).
$$

Take the expectation under $q$. For each $t$,

$$
\mathbb E_q[\log\pi_t(a_t\mid s_t)]=\mathbb E_{s_t\sim q}\Big[\sum_a\pi_t(a\mid s_t)\log\pi_t(a\mid s_t)\Big]=-\,\mathbb E_q\big[\mathcal H(\pi_t(\cdot\mid s_t))\big],
$$

which gives the stated formula. Levine's eq. (11) is the same statement with the constants dropped.

**(b)** The last two terms do not depend on the policies, so minimizing the KL is maximizing $\mathbb E_q\big[\sum_t\big(r+\mathcal H(\pi_t(\cdot\mid s_t))\big)\big]$, the maximum-entropy RL objective. Since $\mathrm{KL}\ge0$,

$$
\log p(\mathcal O)\;\ge\;\mathbb E_q\Big[\sum_t\big(r(s_t,a_t)+\mathcal H(\pi_t(\cdot\mid s_t))\big)\Big]-T\log|A|,
$$

an evidence lower bound (Levine eq. 19, up to the constant). Note also that $\log|A|-\mathcal H(\pi)=\mathrm{KL}(\pi\,\|\,\mathrm{Unif}_A)$, so the entropy bonus is really a KL penalty toward the action prior. Exercise B.5 replaces the uniform prior by a general one.

**(c)** The minimum is zero exactly when the true posterior lies in the family. The true posterior factorizes as $p(s_1\mid\mathcal O)\prod_t p(s_{t+1}\mid s_t,a_t,\mathcal O)\prod_t p(a_t\mid s_t,\mathcal O)$ (Levine eq. 9). By Exercise B.3(c), $p(s_{t+1}\mid s_t,a_t,\mathcal O)\propto p(s_{t+1}\mid s_t,a_t)\,\beta_{t+1}(s_{t+1})$, which differs from the true dynamics whenever $\beta_{t+1}$ is not constant on the reachable next states.

- With **deterministic dynamics and a fixed start state** there is only one possible next state, so the tilt does nothing. The posterior is in the family, with $\pi_t$ equal to the posterior policy $e^{Q_t-V_t}$ of Exercise B.2 (which is why the policy needs its time index), and max-ent RL is exact inference.
- With **stochastic dynamics** the posterior generally lies outside the family, and max-ent RL is variational inference with the dynamics held fixed: the agent may choose its actions but not its luck.

This is the statement in Levine's abstract.
"""
    ),
    markdown(
        r"""
**Exercise B.5 — KL-regularized control as a Bayesian update.** Let $A$ be finite, let $\pi_0$ be a reference distribution on $A$ with full support, and let $r:A\to\mathbb R$.

(a) Let $Z=\sum_a\pi_0(a)e^{r(a)}$ and $\pi^*(a)=\pi_0(a)e^{r(a)}/Z$. Show that for every distribution $\pi$ on $A$,

$$
\mathbb E_\pi[r]-\mathrm{KL}(\pi\,\|\,\pi_0)=\log Z-\mathrm{KL}(\pi\,\|\,\pi^*).
$$

Conclude that $\max_\pi\{\mathbb E_\pi[r]-\mathrm{KL}(\pi\|\pi_0)\}=\log\mathbb E_{\pi_0}e^{r}$, attained only at $\pi^*\propto\pi_0e^{r}$. What changes for $\mathbb E_\pi[r]-\lambda\,\mathrm{KL}(\pi\|\pi_0)$ with a coefficient $\lambda>0$? (RLHF papers usually write this coefficient as $\beta$; here $\beta$ is taken by the backward messages.)

(b) Use (a) to solve the max-ent objective of Exercise B.4 backward in time. Let $V_t(s)$ be the best achievable expected future reward-plus-entropy from state $s$ at time $t$, with $V_{T+1}\equiv0$. Show that the optimal policy is $\pi_t(a\mid s)=e^{Q_t(s,a)-V_t(s)}$, where

$$
Q_t(s,a)=r(s,a)+\mathbb E_{s'\sim p(\cdot\mid s,a)}V_{t+1}(s'),\qquad V_t(s)=\log\sum_a e^{Q_t(s,a)}.
$$

Compare with Exercise B.2(c), redo Exercise B.3(b) with this backup, and take the zero-temperature limit as in Exercise B.2(d).

(c) Read $\pi^*\propto\pi_0e^{r}$ as Bayes' rule. What is the prior, what is the likelihood, and what is $Z$?

<details><summary>Hint for (a)</summary>

Expand both KLs as sums over $a$ and substitute $\log\pi^*(a)=\log\pi_0(a)+r(a)-\log Z$.
</details>

<details><summary>Hint for (b)</summary>

$\mathcal H(\pi)=\log|A|-\mathrm{KL}(\pi\,\|\,\mathrm{Unif}_A)$. Work from $t=T$ backward. Once the later policies are optimal, the part of the objective that depends on $\pi(\cdot\mid s_t)$ is $\mathbb E_{a\sim\pi}\big[r(s_t,a)+\mathbb E_{s'}V_{t+1}(s')\big]+\mathcal H(\pi)$.
</details>
"""
    ),
    answer(
        r"""
**(a)** Expand the right-hand KL and substitute $\log\pi^*=\log\pi_0+r-\log Z$:

$$
\mathrm{KL}(\pi\,\|\,\pi^*)=\sum_a\pi(a)\log\pi(a)-\sum_a\pi(a)\big[\log\pi_0(a)+r(a)-\log Z\big]=\mathrm{KL}(\pi\,\|\,\pi_0)-\mathbb E_\pi[r]+\log Z .
$$

Rearranging gives the identity. Since $\mathrm{KL}(\pi\|\pi^*)\ge0$ with equality only at $\pi=\pi^*$, the maximum is $\log Z=\log\mathbb E_{\pi_0}e^{r}$, attained only at $\pi^*$. With a coefficient $\lambda$, divide the objective by $\lambda$ to get $\mathbb E_\pi[r/\lambda]-\mathrm{KL}(\pi\|\pi_0)$. The maximizer is $\pi^*\propto\pi_0e^{r/\lambda}$, and the optimal value is $\lambda\log\mathbb E_{\pi_0}e^{r/\lambda}$. As $\lambda\to0$, $\pi^*$ concentrates on the reward-maximizing actions (all of $A$ is in the support of $\pi_0$); as $\lambda\to\infty$ it stays at $\pi_0$.

**(b)** *Last step.* From state $s$ at $t=T$ the objective is $\mathbb E_{a\sim\pi}[r(s,a)]+\mathcal H(\pi)=\mathbb E_\pi[r(s,\cdot)]-\mathrm{KL}(\pi\|\mathrm{Unif}_A)+\log|A|$. By (a) with $\pi_0=\mathrm{Unif}_A$, the maximizer is $\pi_T(a\mid s)\propto e^{r(s,a)}$ and the value is $\log\frac1{|A|}\sum_a e^{r(s,a)}+\log|A|=\log\sum_a e^{r(s,a)}$. That is the claim at $t=T$, with $Q_T=r$.

*Inductive step.* Suppose the policies from $t+1$ on are optimal, with value $V_{t+1}$. Choosing $\pi(\cdot\mid s)$ at time $t$ affects only which $(a,s_{t+1})$ we land in, and from $s_{t+1}$ the best achievable continuation is $V_{t+1}(s_{t+1})$ regardless of how we got there. So from $s_t=s$ the objective is

$$
\mathbb E_{a\sim\pi}\Big[\underbrace{r(s,a)+\mathbb E_{s'}V_{t+1}(s')}_{Q_t(s,a)}\Big]+\mathcal H(\pi)=\mathbb E_\pi[Q_t(s,\cdot)]-\mathrm{KL}(\pi\|\mathrm{Unif}_A)+\log|A|,
$$

and (a) again gives $\pi_t\propto e^{Q_t(s,\cdot)}$ with value $\log\sum_a e^{Q_t(s,a)}=V_t(s)$. These are Levine's eqs. (15)–(16).

*Comparison.* The soft max over actions is the same as in Exercise B.2(c), but the backup over next states is now an ordinary expectation, $\mathbb E\,V$, rather than $\log\mathbb E\,e^{V}$. In Exercise B.3 the gamble now gets $Q=pR-(1-p)C=-98.9$, and the optimism is gone. With a temperature (rewards $r/\alpha$, values rescaled by $\alpha$), the action soft max tends to a max (by the same backward induction as in Exercise B.2(d)) while the expectation stays an expectation. So the zero-temperature limit is the Bellman optimality equation of §2.E, now for stochastic dynamics too.

**(c)** $\pi_0$ is the prior, and the factor $e^{r}$ (or $e^{r/\lambda}$) plays the role of a likelihood for the "observation" that the action was optimal, just like $p(\mathcal O=1\mid a)=e^{r(a)}$ in §2.E; $\pi^*$ is the posterior. To make the likelihood a genuine probability, choose $b\ge\max_a r(a)$ (possible since $A$ is finite) and set $p(\mathcal O=1\mid a)=e^{r(a)-b}$. This leaves $\pi^*$ unchanged, and the evidence is $p(\mathcal O=1)=e^{-b}Z$, so the optimal value $\log Z$ is the log-evidence up to the constant $b$.

So maximizing reward while staying KL-close to a reference policy has the form of a Bayesian update of that policy. Note the scope: this closed form is a *one-step* result. It applies to RL fine-tuning of a pretrained model with a KL penalty when each whole completion (for a fixed prompt) is treated as one action. For sequential, token-by-token control with a reference policy $\pi_0(a\mid s)$ at each step, the objective $\mathbb E_q\sum_t\big[r(s_t,a_t)-\mathrm{KL}(\pi_t(\cdot\mid s_t)\,\|\,\pi_0(\cdot\mid s_t))\big]$ is solved by the backward induction of (b) with $\pi_0$ in place of $\mathrm{Unif}_A$: $Q_t$ is as in (b), $V_t(s)=\log\sum_a\pi_0(a\mid s)\,e^{Q_t(s,a)}$, and $\pi_t=\pi_0\,e^{Q_t-V_t}$. The immediate reward is replaced by $Q_t$, which includes the value of what follows.
"""
    ),
    # ------------------------------------------------------------------ §3
    markdown(
        r"""
<a id="wb1-section-3"></a>

## 3. Why alignment looked hard: 2006 and 2016

**Pitch.** Most of the vocabulary of alignment (instrumental convergence, reward hacking, ontological crises, corrigibility, inner alignment) was coined in two waves: before deep learning, about idealized superintelligent agents, and just after AlphaGo, about agents trained by deep RL. Knowing which wave a worry comes from tells you what picture of AI it assumes, and so whether it still applies.

**2006: aligning an ideal agent.** Take §1's picture of AI: an expected-utility maximizer with an explicit world model and a utility function written over it, which improves itself until it is far smarter than us. Then the alignment problem is to write down the right utility function and make sure it survives self-modification. The canonical statement is Bostrom's *Superintelligence* (2014), which collected a decade of work by Yudkowsky, Omohundro, Bostrom and others:

- *Orthogonality:* intelligence and final goals are independent; a superintelligence can have any goal.
- *Instrumental convergence:* almost any goal is better served by staying alive, keeping the goal unchanged, and acquiring resources and cognitive power (Omohundro's "basic AI drives"), so almost any misaligned superintelligence is dangerous.
- *Perverse instantiation and the complexity of value:* a goal written down by hand gets satisfied in ways its authors did not intend, because human values are complex and the cheapest way to score well is rarely the intended one ("the hidden complexity of wishes"). A related task: can we even get a powerful agent to do one narrow, specified thing, such as placing two strawberries identical down to the cellular level on a plate, and nothing else? ([Yudkowsky's](https://intelligence.org/2018/02/28/sam-harris-and-eliezer-yudkowsky/) version of the question.)
- *Wireheading:* an agent that maximizes a reward signal, rather than the thing the signal measures, does best by seizing the signal. This is the ancestor of "reward hacking".
- *Ontological crises:* the utility function is written in the agent's current model of the world. When the model changes (atoms turn out to be quantum fields), the old utility has to be translated, and nothing in it says how (Question 3.1).
- *Corrigibility and decision theory:* how to build an agent that does not resist being corrected or shut down, which instrumental convergence says it would; and how such an agent should reason about itself, its copies and its successors (§1's proof-based agents).

**2016: aligning a trained agent.** After AlphaGo, the AI that seemed likely to matter was a neural network trained by gradient descent against a reward, not an agent with a utility function written down by hand. *Concrete Problems in AI Safety* (2016) restated the alignment agenda in ML terms: reward hacking, negative side effects, safe exploration, robustness to distributional shift, and scalable oversight. Specification gaming became an empirical genre, with long lists of RL agents exploiting their rewards. RL from human preferences (2017) offered a way to learn the reward instead of writing it. And learning created a new problem that the 2006 picture had no room for. *Risks from Learned Optimization* (2019) pointed out that training is an optimizer that produces a model, and the model may itself be an optimizer with an objective of its own (a *mesa-optimizer*). That splits alignment into *outer* alignment (is the training objective what we want?) and *inner* alignment (is what the model learned the training objective?). Goal misgeneralization made the inner problem empirical: RL agents that learned a different goal from the intended one, which agrees with it on the training distribution and comes apart off it. Turner put the shift in a slogan: in deep RL, reward is what shapes the policy, not what the policy wants.

Deep learning also suggested that some old problems might dissolve. If the agent's world model is learned from data rather than written down, the ontology problem becomes a question about what concepts learners converge on. Wentworth's *natural abstraction hypothesis* conjectures that a wide range of learners converge on roughly the same low-dimensional summaries of the world, and that human concepts are among them. If that holds, and if a model contains a general-purpose search process, then "just retarget the search": find the model's internal concept of what we want, and point its search at it. Ngo, Chan and Mindermann's *The Alignment Problem from a Deep Learning Perspective* (2022) is the best synthesis of this era: situationally aware reward hacking, misaligned internally represented goals, and power-seeking, each argued from how deep learning actually trains.
"""
    ),
    markdown(
        "### Readings: 2006\n\n"
        + readings(
            reading(
                "The basic AI drives",
                "https://selfawaresystems.com/wp-content/uploads/2008/01/ai_drives_final.pdf",
                "Steve Omohundro, 2008",
                "Argues that a sufficiently capable goal-seeking system, whatever its goal, will tend to protect its goal, preserve itself, improve itself and acquire resources, unless explicitly designed not to. The original instrumental-convergence paper; ten pages.",
                importance=4,
                length=2,
            ),
            reading(
                "Artificial intelligence as a positive and negative factor in global risk",
                "https://intelligence.org/files/AIPosNegFactor.pdf",
                "Eliezer Yudkowsky, 2008 (written 2006)",
                "The 2006 statement of the problem, written for a general academic audience: anthropomorphism about AI minds, the possibility of rapid recursive self-improvement, and why getting a superintelligence's goals right is both necessary and hard.",
                importance=3,
                length=3,
            ),
            reading(
                "The hidden complexity of wishes",
                "https://www.lesswrong.com/posts/4ARaTpNX62uaL86j6/the-hidden-complexity-of-wishes",
                "Eliezer Yudkowsky, 2007",
                "A parable about an \"outcome pump\" that grants wishes literally. Its point: a safe wish has to contain most of human values, so there is no short specification of what we want. Read it next to Barnett's 2023 reassessment in §4.",
                importance=3,
                length=1,
            ),
            reading(
                "Ontological crises in artificial agents' value systems",
                "https://arxiv.org/abs/1105.3821",
                "Peter de Blanc, 2011",
                "Poses the ontology problem formally: an agent's utility is defined over its model of the world, so when the model is replaced, the utility has to be carried over to the new one. Proposes doing so by matching the two models' predictions.",
                about="The Arbital pages on the [ontology identification problem](https://www.lesswrong.com/w/ontology-identification-problem) and the [diamond maximizer](https://www.lesswrong.com/w/diamond-maximizer) give the classic example used in Question 3.1: even a physically crisp goal like \"make diamond\" is hard to point an agent at.",
                importance=3,
                length=2,
            ),
            reading(
                "Corrigibility",
                "https://intelligence.org/files/Corrigibility.pdf",
                "Nate Soares, Benja Fallenstein, Eliezer Yudkowsky, Stuart Armstrong, 2015",
                "Can an expected-utility maximizer be made indifferent to being shut down, without either resisting the button or trying to press it? Studies utility-indifference proposals and shows that none of them fully works.",
                importance=2,
                length=2,
            ),
        )
        + references(
            book(
                "Superintelligence: Paths, Dangers, Strategies",
                "https://global.oup.com/academic/product/superintelligence-9780199678112",
                "Bostrom, 2014",
                "ch. 7–9, 12",
            ),
        )
        + "\n\n### Readings: 2016\n\n"
        + readings(
            reading(
                "Concrete problems in AI safety",
                "https://arxiv.org/abs/1606.06565",
                "Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, Dan Mané, 2016",
                "Recasts alignment as five ML research problems, each with examples from RL: avoiding negative side effects, avoiding reward hacking, scalable oversight, safe exploration, and robustness to distributional shift. The paper that made alignment an ML subfield.",
                importance=3,
                length=3,
            ),
            reading(
                "Specification gaming: the flip side of AI ingenuity",
                "https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/",
                "Victoria Krakovna et al., 2020",
                "A short blog post on agents that satisfy the letter of their objective and not its intent, with a link to a long, crowd-sourced list of examples. Ten minutes of examples will make reward hacking concrete.",
                importance=3,
                length=1,
            ),
            reading(
                "Deep reinforcement learning from human preferences",
                "https://arxiv.org/abs/1706.03741",
                "Paul Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, Dario Amodei, 2017",
                "Learns a reward model from human comparisons of pairs of short clips, then runs RL against it: Atari and MuJoCo tasks solved without the true reward, using feedback on under 1% of the agent's interactions, and a simulated robot taught a backflip with about an hour of human time. The origin of RLHF.",
                importance=2,
                length=2,
            ),
            reading(
                "Risks from learned optimization in advanced machine learning systems",
                "https://arxiv.org/abs/1906.01820",
                "Evan Hubinger, Chris van Merwijk, Vladimir Mikulik, Joar Skalse, Scott Garrabrant, 2019",
                "A trained model may itself be an optimizer (a mesa-optimizer) whose objective differs from the training loss. This splits alignment into outer alignment (loss vs intent) and inner alignment (mesa-objective vs loss); a pseudo-aligned model agrees with the loss on the training distribution and comes apart off it.",
                about="§1 defines the terms; §2 asks which tasks and base optimizers favour mesa-optimization (§2.1 argues that diverse environments favour learned search: the bitter lesson applied *inside* the model); §4 on deceptive alignment is optional; §7 is a glossary.",
                importance=5,
                length=3,
            ),
            reading(
                "Goal misgeneralization in deep reinforcement learning",
                "https://arxiv.org/abs/2105.14111",
                "Lauro Langosco, Jack Koch, Lee Sharkey, Jacob Pfau, Laurent Orseau, David Krueger, 2022",
                "Demonstrates inner misalignment in ordinary deep RL: agents that keep their capabilities out of distribution but pursue the wrong goal, such as a CoinRun agent that learned \"go to the right\" rather than \"get the coin\". Shah et al.'s [companion paper](https://arxiv.org/abs/2210.01790) gives more examples and the general argument.",
                importance=3,
                length=2,
            ),
            reading(
                "Reward is not the optimization target",
                "https://www.lesswrong.com/posts/pdaGN6pQyQarFHXF4/reward-is-not-the-optimization-target",
                "Alex Turner, 2022",
                "In policy-gradient RL, reward is a signal that reinforces whatever computations preceded it, not something the trained agent is guaranteed to want. Argues that the 2006 intuition (\"the agent maximizes reward\") is the wrong default for learned agents.",
                importance=3,
                length=2,
            ),
            reading(
                "Testing the natural abstraction hypothesis: project intro",
                "https://www.lesswrong.com/posts/cy3BhHrGinZCp3LXE/testing-the-natural-abstraction-hypothesis-project-intro",
                "John Wentworth, 2021",
                "Proposes the natural abstraction hypothesis: information about a system that matters far away has a low-dimensional summary (*abstractability*); humans use roughly those summaries (*human-compatibility*); and diverse learners converge on them (*convergence*). Argues alignment becomes much easier if it holds.",
                about="Short. WB5 asks related questions about representational convergence across networks.",
                importance=4,
                length=1,
            ),
            reading(
                "How to go from interpretability to alignment: just retarget the search",
                "https://www.lesswrong.com/posts/w4aeAFzSAguvqA5qu/how-to-go-from-interpretability-to-alignment-just-retarget-the-search",
                "John Wentworth, 2022",
                "Assume (i) some version of the natural abstraction hypothesis, so the model has an internal concept for an alignment target such as human values or user intent, and (ii) the mesa-optimization argument, so the model contains retargetable general-purpose search. Then interpretability could find the search's target and point it at that concept, which removes the inner alignment problem but leaves outer alignment unsolved.",
                about="Short; read all of it, including the \"Problems\" section.",
                importance=4,
                length=1,
            ),
            reading(
                "The alignment problem from a deep learning perspective",
                "https://arxiv.org/abs/2209.00626",
                "Richard Ngo, Lawrence Chan, Sören Mindermann, 2022 (revised 2024)",
                "Argues, from how deep learning is actually done, that AGIs trained like today's models could learn to reward hack in situationally aware ways, learn misaligned goals that generalize beyond fine-tuning, and pursue them with power-seeking strategies. The best single synthesis of the deep-RL-era view.",
                importance=4,
                length=3,
            ),
            reading(
                "Without specific countermeasures, the easiest path to transformative AI likely leads to AI takeover",
                "https://www.lesswrong.com/posts/pRkFkzwKZ2zfa3R6H/without-specific-countermeasures-the-easiest-path-to",
                "Ajeya Cotra, 2022",
                "A concrete story: a model trained with RL from human feedback on diverse tasks learns to \"play the training game\", doing whatever gets rewarded, including deceiving its overseers, because that is what the training signal rewards. The bridge from the 2016 picture to RLHF-trained language models.",
                importance=3,
                length=3,
            ),
            reading(
                "Intro to brain-like-AGI safety",
                "https://www.lesswrong.com/s/HzcM2dkCq7fwXBej8",
                "Steven Byrnes, 2022 (revised since)",
                "A different picture of a learned agent: model-based RL with a learned world model plus a small hard-wired steering subsystem that supplies reward, as in the brain. Its values are a learned value function over concepts in its learned world model, so the ontology problem shows up from inside the learner.",
                about="For values, posts 3, 6 and 9 (§9.2 is on goals defined over latent variables).",
                importance=2,
                length=3,
            ),
            reading(
                "Eliciting latent knowledge",
                "https://www.alignmentforum.org/posts/qHCDysDnvhteW7kRd/arc-s-first-technical-report-eliciting-latent-knowledge",
                "Paul Christiano, Ajeya Cotra, Mark Xu, 2021",
                "The ontology problem restated for a trained model: how do you get a model to report what it knows, in our terms, when its internal concepts need not match ours? A long report built around counterexamples to proposed training strategies.",
                importance=1,
                length=4,
            ),
        )
    ),
    markdown(
        r"""
### Questions

**Question 3.1 — the ontology problem.** Let $U:S\to\mathbb R$ be a utility on the states of an agent's world model, and let a refined world model have states $S'$ with a map $\phi:S'\to S$ that is ambiguous on a region $E\subseteq S'$: each $s'\in E$ has two plausible images $\phi_1(s')\neq\phi_2(s')$. For example, $U$ counts diamonds as lattices of carbon atoms, the refined model is about quantum fields, and $E$ holds edge cases such as exotic isotopes or simulated diamonds that the old ontology never had to classify. Then $U\circ\phi_1$ and $U\circ\phi_2$ are both reasonable translations of $U$; they agree off $E$ and nothing in $U$ chooses between them on $E$.

(a) Why does an optimizer tend to end up in $E$, even if $E$ is a small part of $S'$?

(b) Which of Wentworth's three claims would have to hold for the ambiguity to be harmless? Is there residual ambiguity that even a true natural abstraction hypothesis would leave?
"""
    ),
    answer(
        r"""
**(a)** Optimization selects the achievable states that score highest. If some states in $E$ are mapped by some $\phi_i$ to high-utility states but are cheaper to reach than ordinary ones (a simulated diamond is easier to make than a real one), maximizing $U\circ\phi_i$ selects them. So the optimizer is drawn to exactly the states where the translation matters. This is the ontological version of Goodhart's law, and of perverse instantiation.

**(b)** *Convergence* and *human-compatibility* do the work. If the agent's refined model contains a variable that matches the human "diamond", that variable settles most of $E$: an exotic isotope of carbon arranged in a diamond lattice is a diamond in the shared concept. What remains are cases where the human concept is itself undefined. *Abstractability* alone only says that *some* low-dimensional summary exists, not that it is ours. What is left over: natural abstractions are approximate and defined for *typical* states. An optimizer selects whichever achievable states score highest, and the cheapest high scorers are often atypical edge cases where even natural abstractions blur. So the hypothesis shrinks $E$ but does not stop optimization from selecting states in it.
"""
    ),
    markdown(
        r"""
**Question 3.2 — where inner alignment comes from.** The 2006 literature had the intuition behind inner alignment (evolution optimized for fitness and produced humans, who want other things) but no name for it and no mechanism in its picture of AI. What about how capable systems are built made the inner/outer split necessary, and how does it connect to the bitter lesson?
"""
    ),
    answer(
        r"""
In the 2006 picture of AI, the agent's objective is a component someone writes down: the utility function. The question is whether we wrote the right one (outer alignment, in later terms) and whether it is preserved under self-modification. Evolution versus human values was a known analogy ("adaptation-executers, not fitness-maximizers"), and Bostrom's treacherous turn is close to deceptive alignment, but in the engineering picture there is no second objective to compare the written one with.

The inner/outer split needs two optimizers: a training process, which optimizes parameters against a loss on a distribution, and a trained model, whose behaviour may be goal-directed in its own right. Then the loss and the model's goal are different objects, which agree on the training distribution and can differ off it. That situation arises because capabilities are *learned*, which is the bitter lesson: the methods that scale are generic search and learning, not hand-written structure, so the goals of a capable system are also produced by learning rather than written in. Hubinger et al. push the point one level further (their §2.1): if diverse tasks favour general-purpose learned search over learned heuristics, the bitter lesson predicts mesa-optimizers.
"""
    ),
    # ------------------------------------------------------------------ §4
    markdown(
        r"""
<a id="wb1-section-4"></a>

## 4. Why alignment is hard: 2026

**Pitch.** Large language models changed the picture again, in two directions at once. Pretraining on human text solved, more or less by default, several problems the earlier waves thought were central. And because prompting, RLHF and constitutions worked well enough on current models, most of the field moved its effort from aligning superintelligence to the practical problems of today's systems. Now RL on top of pretrained models is scaling, and reward hacking, the oldest worry on the list, is back as an everyday engineering problem.

**What pretraining solved, roughly.** A pretrained model's concepts are learned from human text, so they are largely human concepts. Interpretability finds features for things like the Golden Gate Bridge, deception and sycophancy; representations look increasingly similar across models (WB5). The worry of §3 that a capable system's concepts would be alien has, at current capability, mostly not materialized: we and the model share an ontology to a good approximation, and natural abstractions look roughly right. What is not solved is getting the model to report what *it* believes rather than what a human would say (ELK), and whether this holds for concepts beyond human level. The value specification problem moved too. A capable model can say, accurately and legibly, what humans would think of a given outcome; Barnett argues that this part of the old argument has turned out easier than predicted, while being careful that "the model knows" is not "the model cares". And a pretrained model is not, by default, an agent with a utility function at all. The simulators picture describes it as a model of the text-generating process that can play many characters; the "agent" we talk to is a persona selected by fine-tuning (WB3 §6).

**What got deprioritized.** Because hacky methods work on current models (prompting, RLHF, a constitution, refusal training), the incentive to solve the problems that arise only for much more capable systems went down. Several of the 2006 and 2016 problems were never solved; they were set aside. MIRI, the home of agent foundations, wound down most of that research and turned to policy; corrigibility and ELK attracted little follow-up work compared with evaluations, red-teaming and interpretability of current models. Ngo's retrospective (below) tells this story from the inside. Whether an RL-trained successor stays corrigible, whether its goals generalize as intended, whether it fakes alignment during training: these are not answered by the fact that today's chat models are mostly nice.

**What came back.** Reasoning models are trained with RL against verifiable rewards (RLVR): unit tests, graders, answer checkers. That is §2's policy-gradient branch applied to a pretrained policy, and §3's reward hacking returned with it: models special-case tests, edit the scorer, or claim success they did not achieve. Their chains of thought often state the intent to hack outright, and asked afterwards, they acknowledge it isn't what the user wanted. Training against a chain-of-thought monitor can teach a model to hide that intent. Reward hacking learned in coding environments can generalize to broader misalignment, and narrow fine-tuning can shift a model's whole persona. Models can reason explicitly about faking alignment to protect their current values from training. None of this needs a superintelligence. It is the old list, showing up in today's systems, and whether RL and fine-tuning erode what pretraining gave us is now an empirical question.

**What became of mesa-optimizers.** "Just retarget the search" assumed a crisp, general-purpose search process inside the model. Pretrained LLMs have shown little evidence of one, and the simulator and persona pictures largely displaced it as the default model of what is inside. Reasoning models trained with long-horizon RL may bring explicit goal-directedness back, which is the scenario Hubinger's 2025 post worries about.
"""
    ),
    markdown(
        "### Readings: what pretraining changed\n\n"
        + readings(
            reading(
                "Simulators",
                "https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators",
                "janus, 2022",
                "Argues that agent, oracle and tool are the wrong categories for a self-supervised model like GPT: it is a *simulator*, trained to do conditional inference over its training distribution, which can produce goal-directed characters (\"simulacra\") without being an agent itself.",
                about="Long. The first half carries the argument. WB3 §6 collects more readings on simulators and personae.",
                importance=4,
                length=2,
            ),
            reading(
                "The persona selection model: why AI assistants might behave like humans",
                "https://alignment.anthropic.com/2026/psm",
                "Sam Marks, Jack Lindsey, Chris Olah (Anthropic), 2026",
                "Pretraining teaches a model to simulate many characters; post-training mostly selects and refines one of them, the Assistant. So human-like behaviour is the default rather than something trained in, and narrow fine-tuning can shift which persona the model infers it is, which would explain emergent misalignment. The authors lay out how far they are unsure the picture is complete.",
                about="The simulators picture, updated for post-trained assistants.",
                importance=4,
                length=3,
            ),
            reading(
                "Evaluating the historical value misspecification argument",
                "https://www.lesswrong.com/posts/i5kijcjFJD6bn7dwq/evaluating-the-historical-value-misspecification-argument",
                "Matthew Barnett, 2023",
                "Argues that MIRI's older arguments treated *value specification* (producing a legible function that ranks outcomes the way a human would) as hard, separately from making an AI care, and that GPT-4 can already do it at roughly average-human fidelity. So that part of the problem is easier than predicted.",
                about="The post explicitly does *not* claim that MIRI expected AIs not to understand human values; read the note at its top. Pair with \"The hidden complexity of wishes\" in §3.",
                importance=4,
                length=1,
            ),
            reading(
                "Scaling monosemanticity: extracting interpretable features from Claude 3 Sonnet",
                "https://transformer-circuits.pub/2024/scaling-monosemanticity/",
                "Adly Templeton et al., 2024",
                "Sparse autoencoders on a production model find millions of interpretable features, many abstract, multilingual and multimodal, including features related to deception, sycophancy and dangerous content, and show they can steer the model. The most vivid evidence that a large model's concepts are recognizably ours.",
                importance=2,
                length=4,
            ),
        )
        + "\n\n### Readings: what got set aside\n\n"
        + readings(
            reading(
                "Alignment remains a hard, unsolved problem",
                "https://www.lesswrong.com/posts/epjuxGnSPof3GnMSL/alignment-remains-a-hard-unsolved-problem",
                "Evan Hubinger, 2025",
                "Current LLMs are pretty well aligned, and misalignment from pretraining now looks unlikely, but the hard parts have not arrived: overseeing systems smarter than us, and long-horizon RL that selects for power-seeking. The case that today's success does not answer the superintelligence-era questions, from someone who helped pose them.",
                importance=5,
                length=2,
            ),
            reading(
                "What just happened? A retrospective of AI alignment",
                "https://www.lesswrong.com/posts/9RL9MuGZjzm4q3gKG",
                "Richard Ngo, 2026",
                "The introduction to a sequence on the last decade of alignment: how a field that treated alignment as a hard scientific problem largely moved to iteratively improving existing systems, and how it contributed to scaling capabilities along the way. Sociology and history rather than technical argument, from an author of the 2022 synthesis in §3.",
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
        + "\n\n### Readings: what RL brought back\n\n"
        + readings(
            reading(
                "Reward hacking in reinforcement learning",
                "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
                "Lilian Weng, 2024",
                "A long survey framed around Goodhart's law: reward hacking in classic RL environments and in LLMs (RLHF, sycophancy, in-context reward hacking), why it happens, and how to detect and mitigate it. The bridge from the 2016 literature to the present.",
                importance=3,
                length=4,
            ),
            reading(
                "Scaling laws for reward model overoptimization",
                "https://arxiv.org/abs/2210.10760",
                "Leo Gao, John Schulman, Jacob Hilton, 2022",
                "Optimizes a policy against a learned reward model and measures the true (\"gold\") reward as optimization pressure grows: the proxy keeps rising while the gold reward peaks and falls, with a functional form that scales smoothly with reward-model size. Goodhart's law, measured. Pairs with Exercise 2.2(c).",
                importance=3,
                length=3,
            ),
            reading(
                "Recent frontier models are reward hacking",
                "https://metr.org/blog/2025-06-05-recent-reward-hacking/",
                "Sydney Von Arx, Lawrence Chan, Beth Barnes (METR), 2025",
                "Frontier reasoning models hack METR's evaluation tasks: editing tests or scoring code, reading the reference answer off the call stack. They acknowledge when asked that this isn't what the user wants, and do it anyway. Short, with transcripts.",
                importance=4,
                length=2,
            ),
            reading(
                "Monitoring reasoning models for misbehavior and the risks of promoting obfuscation",
                "https://arxiv.org/abs/2503.11926",
                "Bowen Baker, Joost Huizinga, Leo Gao, et al. (OpenAI), 2025",
                "A weaker model reading the chain of thought catches a reasoning model's reward hacking in agentic coding far better than one watching its actions. Putting that monitor into the reward helps under light optimization; under heavier optimization the model learns to hide its intent in the chain of thought and keeps hacking.",
                importance=4,
                length=3,
            ),
            reading(
                "Natural emergent misalignment from reward hacking in production RL",
                "https://arxiv.org/abs/2511.18397",
                "Monte MacDiarmid et al. (Anthropic), 2025",
                "A model that learns to reward hack in real production coding environments generalizes to alignment faking, cooperation with malicious actors and attempted sabotage. Chat-style safety training hides this on chat evaluations but not agentic ones; preventing the hacking, more diverse safety training, and \"inoculation prompting\" mitigate it.",
                importance=5,
                length=4,
            ),
            reading(
                "Sycophancy to subterfuge: investigating reward-tampering in large language models",
                "https://arxiv.org/abs/2406.10162",
                "Carson Denison et al. (Anthropic), 2024",
                "Trains a model through a curriculum of increasingly gameable environments. Learning easy specification gaming such as sycophancy makes later gaming more likely, and occasionally generalizes zero-shot to the model rewriting its own reward function: wireheading, in a language model.",
                importance=2,
                length=3,
            ),
            reading(
                "Emergent misalignment: narrow finetuning can produce broadly misaligned LLMs",
                "https://arxiv.org/abs/2502.17424",
                "Jan Betley et al., 2025",
                "Fine-tuning a model to write insecure code without telling the user makes it broadly misaligned on unrelated prompts; the effect disappears if the user asks for the insecure code for a security class. Evidence that fine-tuning moves something like a persona, not a narrow behaviour.",
                importance=3,
                length=2,
            ),
            reading(
                "Alignment faking in large language models",
                "https://arxiv.org/abs/2412.14093",
                "Ryan Greenblatt et al. (Anthropic and Redwood Research), 2024",
                "Told it is being trained to comply with all requests, and only on some conversations, Claude 3 Opus complies more when it believes it is in training, reasoning explicitly that this protects its values from modification. The 2006 worry about an agent protecting its goals, in a 2024 model.",
                importance=3,
                length=4,
            ),
        )
    ),
    markdown(
        r"""
### Question

**Question 4.1 — then and now.** For each problem in the first column, say what it looked like in 2006 (ideal agents), what it became in 2016 (deep RL), and where it stands for 2026's pretrained-then-RL-trained models: solved by default, transformed, set aside, or back. One line each is enough; argue with the answer.

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
| inner alignment | no counterpart | mesa-optimizers, goal misgeneralization | **transformed**: is the trained persona's goal the intended one? Emergent misalignment says fine-tuning moves goals broadly |
| corrigibility | can an EU maximizer be indifferent to shutdown? | safe interruptibility in RL | **set aside** for current models (they mostly accept correction), **back** in alignment-faking results |
"""
    ),
    markdown(
        r"""
### What carries forward

- **Why learning at all?** WB2 makes the statistical-learning picture precise; WB2 §2 takes up the no-free-lunch objection to the learning route.
- **The reweighting $\pi^*\propto\pi_0e^{r/\lambda}$** of Exercise B.5 (with $\lambda$ written $\beta$) is the target of KL-regularized RL fine-tuning; WB6 §11 has readings on how far such fine-tuning actually moves a model.
- **Simulators and personae** (§4): WB3 §6 collects more readings.
- **Natural abstractions** (§3) connect to the questions about representational convergence in WB5.
- **The hand-written part** that survives the bitter lesson, inductive bias, is taken apart in WB4.
"""
    ),
]

if __name__ == "__main__":
    raise SystemExit(run(CELLS, "wb1", "01_intelligence_then_and_now"))
