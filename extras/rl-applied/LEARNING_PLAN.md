# RL Applied: 3-Week Learning Plan

**Who this is for.** You (Jasmine) already have the theory — policy gradients, reward
modeling, DPO, RLVR from the tinker-cookbook lectures — but you've never written a
training loop. This plan closes that gap by building an **RLHF-in-miniature pipeline on
your own research problem**: your α-sweep activation-steering images + your preference
judgments → a reward model → a policy that picks α.

**Design constraints.** Pure numpy, CPU, no external data required (every script has a
synthetic mode), every file < 200 lines. Torch is deliberately absent in Week 1: writing
the gradient by hand is the fastest way to make `log-prob × advantage` stop being a slogan.

---

## Week 1 — Close the theory→code gap: REINFORCE from scratch

**Folder:** `week1_reinforce/`

| Day | Task |
|---|---|
| 1–2 | `bandit.py` — 10-armed bandit. Fill in: softmax sampling, the REINFORCE gradient, a running-mean baseline. |
| 3 | Run the baseline-vs-no-baseline experiment. Look at gradient-norm variance in your own logs, not a textbook. |
| 4–5 | `gridworld.py` — stateful REINFORCE. Fill in: returns-to-go, per-state value baseline, the tabular policy-gradient update. |

**Checkpoints**
- [ ] Bandit agent concentrates >85% probability on the best arm.
- [ ] You can state, from your own run logs, why the no-baseline gradient variance is ~5–20× larger when rewards have a big constant offset.
- [ ] Gridworld episode length converges to the optimal 8 with the V(s) baseline — and you can explain why the no-baseline run collapses instead of learning.

**You're done when** you can write the softmax REINFORCE update
`θ += lr · (G_t − b) · (onehot(a) − π(·|s))` from memory, derive it from
`∇θ log π(a|s) · advantage`, and explain the variance/baseline tradeoff using numbers
from your own runs.

## Week 2 — Bradley-Terry reward model on your pilot data

**Folder:** `week2_reward_model/`

| Day | Task |
|---|---|
| 1 | `python3 make_pairs.py --synthetic` — understand the data format (embeddings + pairs CSV). |
| 2–3 | `reward_model.py` — fill in the BT loss `−log σ(s_winner − s_loser)`, its gradient, and the accuracy eval. Get ~79% val accuracy on synthetic data (near the ~84% rater-consistency ceiling that `make_pairs.py` prints). |
| 4–5 | Record real preference pairs from your α-sweep pilot images (see the week-2 README), run `make_pairs.py` on them, retrain, report held-out accuracy. |

**Checkpoints**
- [ ] Synthetic mode: val accuracy ≈ 79% — well above chance (50%) and close to the printed rater-consistency ceiling (~84%) — and RM score correlates >0.85 with the hidden true quality within each context.
- [ ] Real mode: RM trained on ~80% of your judgments, accuracy reported on the held-out 20%.

**You're done when** you have a number: *"a linear head on CLIP embeddings predicts my
held-out preference judgments with X% accuracy."* **This is your grant milestone** —
"validate automatic scores against human labels" — with a train/val protocol behind it.

## Week 3 — Close the loop: a policy that picks α

**Folder:** `week3_alpha_policy/`

| Day | Task |
|---|---|
| 1–2 | `alpha_policy.py` — contextual-bandit REINFORCE over 9 discretized α values, maximizing the RM score. Same three TODO concepts as Week 1. |
| 3 | Run against the synthetic RM; verify per-context α converges near the true optimum. |
| 4–5 | Swap in your real Week-2 RM head; compare the policy's α choices against your manual picks. |

**Checkpoints**
- [ ] Synthetic: policy's modal α per context lands on the grid point nearest the true α*.
- [ ] Policy's average RM score beats both uniform-random α and a single fixed hand-tuned α.
- [ ] End-to-end run: `make_pairs.py --synthetic` → `reward_model_solution.py` → `alpha_policy.py` completes on your machine with no external data.

**You're done when** you can run **preferences → reward model → policy** end-to-end on
your own data and articulate where reward hacking would show up (the policy exploiting RM
errors at α values with little preference coverage).

---

## Map to the standard tinker-cookbook RLHF pipeline

| Standard pipeline | Your miniature | Lecture it grounds |
|---|---|---|
| SFT (base competence) | The steering direction already exists; the α grid is your "supervised" starting point | lec2 (IFT/RM/rejection) |
| Preference collection | Your pairwise judgments on α-sweep images | lec8 (preferences) |
| Reward model (BT on pairs) | Linear head on CLIP embeddings, BT loss | reward-modeling lecture |
| PPO/GRPO against the RM | Tabular REINFORCE over α against your RM | lec3–4 (policy gradients, RL implementation) |
| KL penalty / reward hacking | Entropy of the α-policy; over-optimizing a noisy RM | overoptimization lecture |
| DPO shortcut | Skips the RM: on your problem, "prefer the winner's α directly" | lec6 (DPO) |

After Week 3, re-read the cookbook recipes: every component (rollout, advantage,
RM head, KL-to-reference) will map onto ~20 lines you wrote yourself.

## Week 4 (optional) — One LLM-scale rep

Do a single DPO run on **Qwen2.5-0.5B-Instruct** with **TRL's `DPOTrainer`** on a free
Colab T4: load `trl`, `peft`, a small preference set (e.g. a few thousand pairs from
`trl-lib/ultrafeedback_binarized`), LoRA rank 8, ~30 min of training, then eyeball
chosen-vs-rejected logprob margins. The point is not a good model — it's seeing that the
DPO loss you know from lec6 is the same `−log σ(β·Δlogratio)` shape as your Week-2 BT
loss, at scale. One afternoon, then stop.
