# ML Coding Round: Implement-From-Scratch (Frontier-Lab Style)

**Format:** 3 timed parts, 90 minutes total. Closed-book — this is the round
labs run *from memory*.
**Where this shows up:** OpenAI research engineer / MTS coding rounds, Anthropic
MLE loops, and small independent labs (Liquid AI, Zyphra, Arcee) whose whole
business is efficient training + inference. Per public interview reports, the
canonical asks are: multi-head attention (with GQA), KV-cache decoding,
top-k/top-p sampling, and optimizer/training-loop mechanics — implemented in
PyTorch with basic tensor ops only, often in 25–35 minutes per topic with
little to no debugging time.

## The three parts

| Part | File | Time | What it tests |
|------|------|------|---------------|
| 1 | `part1_attention.py` | ~25 min | scaled dot-product attention, bool masks, causal masking, MHA/GQA/MQA head plumbing |
| 2 | `part2_sampling.py` | ~35 min | KV-cache attention (the mask offset!), top-k / top-p filtering, temperature sampling, incremental `generate` |
| 3 | `part3_optim.py` | ~30 min | AdamW from scratch (must match `torch.optim.AdamW`), warmup+cosine LR schedule, exact gradient accumulation |

Each file's docstrings are the full spec. The parts are independent — if you
stall on one, move on.

## Rules of engagement

1. **90 minutes on the clock, all three parts.** Real rounds are shorter per
   topic but you also get one topic; this simulates a full onsite morning.
2. **No AI assistance, no HF/torch source-diving.** PyTorch API docs are OK
   (real interviews allow "what's the signature of `scatter_`" level lookups).
3. Banned inside your implementations: `F.scaled_dot_product_attention`,
   `nn.MultiheadAttention`, `torch.optim`, HF `generate`. Allowed: everything
   basic — `einsum`, `topk`, `sort`, `cumsum`, `multinomial`, `masked_fill`.
4. Grade a part only when you finish it (or its time is up):

   ```
   uv run python tests/grade_part1.py     # from this directory
   uv run python tests/grade_part2.py
   uv run python tests/grade_part3.py
   ```

   **Do not open anything in `tests/` — the graders contain reference
   implementations.** Grader output is your interviewer feedback: in a real
   round you'd get one or two "are you sure about the masked rows?" nudges,
   which is what the failure messages emulate.
5. `SOLUTIONS.md` has reference implementations + the rubric. Open it last.

## Scoring yourself (28 checks total)

- **Strong hire:** all three parts green in 90 min.
- **Hire:** parts 1 and 2 green, part 3 mostly green, minor tolerance misses.
- **On the bubble:** any *conceptual* miss — attention leaking across the
  causal boundary, coupled weight decay, mean-of-means accumulation — even if
  everything else passes. These are the exact things interviewers probe.

Speed matters: labs explicitly filter on "codes MHA fluently, from memory,
without debugging." If part 1 takes 40+ minutes, that's the signal to drill.
