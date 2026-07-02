# Week 1 — REINFORCE from scratch (pure numpy, CPU, no framework)

You know the math (lec3–4). This week you write it. Two exercises, in order.
No torch on purpose: for a softmax policy the gradient of `log π(a)` has a
closed form (`onehot(a) − π`), so you implement `∇log π × advantage` literally
instead of trusting autograd. When you later read a torch loss like
`-(logprobs * advantages).mean()`, you'll know exactly what it expands to.

## 1. `bandit.py` — multi-armed bandit (do this first)

One state, 10 arms, softmax policy over arms. Three TODOs:

| TODO | What it teaches |
|---|---|
| 1. Policy sampling | A policy is just a distribution you sample; params → softmax → `rng.choice` |
| 2. Log-prob × advantage update | THE policy-gradient estimator. Derive `∇θ log π(a) = onehot(a) − π` on paper first |
| 3. Running-mean baseline | Baselines don't bias the gradient (`E[b·∇log π] = 0`) but slash its variance |

Run `python3 bandit.py`. **Expected output when correct** (compare `bandit_solution.py`):
- avg reward climbs from ~5.5 to ~7.0 (the best arm's mean); prob on best arm > 0.85
  in both runs;
- gradient-norm variance without the baseline is **~5–20× larger**. The arm means sit
  around +5 deliberately: the baseline's job is to subtract that uninformative offset.
  This is exactly why RLHF implementations subtract a value estimate (or group mean,
  in GRPO) before multiplying by log-probs.

## 2. `gridworld.py` — stateful REINFORCE

5×5 grid, start corner → goal corner, −1 per step, +10 at goal. Four TODOs:

| TODO | What it teaches |
|---|---|
| 1. State-conditioned sampling | Policy tables: `π(·|s) = softmax(θ[s])` |
| 2. Returns-to-go | Credit assignment: an action is judged by everything after it, discounted |
| 3. Tabular PG update | Same estimator as the bandit, applied per-(s,a,G) along a trajectory |
| 4. Learned V(s) baseline | The tabular ancestor of PPO's value head / the critic |

Run `python3 gridworld.py`. **Expected output when correct:**
- WITH baseline: avg episode length ~10 in the first 300 episodes, ~8 (optimal) after;
  greedy path length = 8.
- WITHOUT baseline: the run typically **collapses** (length pinned at 200). Don't skip
  the post-mortem — every return is negative, so every action actually taken gets
  pushed down, and the policy can lock into a degenerate loop. This failure mode is
  the visceral version of "advantage normalization matters."

## Order and pacing

Bandit TODOs 1–2 (an hour or two, mostly the paper derivation) → TODO 3 + variance
experiment (an hour) → gridworld (half a day; TODO 2 off-by-one bugs are the classic
time sink — check `Gs[-1]` equals the final reward).

Done when the LEARNING_PLAN Week-1 checkboxes hold **and** you can rewrite the bandit
update from memory. Everything in Weeks 2–3 reuses these exact three ideas.
