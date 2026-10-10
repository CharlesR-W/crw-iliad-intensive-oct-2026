# WB3 reading-centred revision, 2026-10-09

Instructor feedback (CRW): make WB3 reading-centred; readings must answer the section's question; keep only simple useful derivations and Fermi estimates; no discussion questions.

Changes (builder: notebooks/build_wb3.py; previous version in notebooks/_old_wb2/*2026-10-09*):
- §1: + Sutskever "An observation on generalization", gwern "Why does pretraining work?", sentiment neuron, GPT-2, Karpathy RNNs. Ex 1.1 restructured: labels vs. images (per-sample bits) vs. text.
- §2: rewritten around text as already-abstracted/compressed data; JEPA as attempt to get abstraction from video. New Fermi Ex 2.1 (LeCun's child-vs-LLM bytes, then bits per event: sentence vs video). Readings: LeCun AMI paper, V-JEPA 2, Sora report.
- §3: NTP = code length (stated, arithmetic-coding exercise removed), every position a target, causal transformer parallel training, hardware. Readings: Weng GLM, Hooker, Wang et al. 2022, Horace He, Delétang, Illustrated Transformer, BERT. Vaswani only cited for the table.
- §4: geometric DL alone, as the opposite thesis. Bronstein ICLR 2021 keynote, GBC ch. 9 §9.2/9.4, GDL book, ViT, Smith et al. 2023. Ex 4.1/4.2 kept (trimmed).
- §5: + EleutherAI cookbook / Transformer Math 101, nostalgebraist "chinchilla's wild implications".
- §6: readings only: janus, nostalgebraist 2021 + "the void", gwern prompts-as-programming, Anthropic PSM, Assistant Axis. Removed Waluigi, RLHF exercises 6.1/6.2, RL "how far" readings (already in WB6 §11).
- Cross-refs updated in build_wb1.py, build_wb6.py (§11 head), README.md.

Checks: links verified by research subagents (openai.com pages via search only, 403 on fetch). Independent Opus review found no arithmetic errors in 1.1/2.1/4.1/4.2; its wording/accuracy fixes (sentiment-neuron description, 270 wpm bytes, LeCun 2025 source, RoPE symmetry, KV-cache wording, allocation exponents, 2 nats/token) applied. Codex review attempted twice; the process was killed (exit 144) with no output.
