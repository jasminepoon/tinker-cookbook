# Week 3 — Close the loop: a policy that picks α

`alpha_policy.py` is the "PPO stage" of your miniature RLHF pipeline, with every
inessential complication stripped away:

| Full RLHF (tinker-cookbook recipes) | This week |
|---|---|
| Prompt from the dataset | Context c (one of 4 "prompts") |
| LLM generates a response | Pick α from 9 grid values; the world "renders" an embedding |
| Reward model scores the response | `r = w_rm · embedding` (your week-2 head) |
| PPO/GRPO with per-prompt-relative advantage | Tabular REINFORCE with a per-context baseline |
| KL penalty to the reference model | Not needed — the action space is 9 knob values, nowhere to collapse to nonsense (but see reward hacking below) |

Because the environment is one-step (pick α → get reward → done), this is a
**contextual bandit** — RLHF itself is usually treated as exactly that (one action
= one full response). The three TODOs are your week-1 bandit code with one twist
each: state-conditioned sampling, and the baseline becomes per-context (the tabular
version of GRPO's group-relative advantage).

## Run it

```bash
python3 alpha_policy.py             # exercise; solution: alpha_policy_solution.py
```

Works out of the box: if `../week2_reward_model/data/rm_head.npz` is missing, it
builds a ground-truth synthetic RM. If you've run week 2, it **automatically loads
your trained head** — no code change; that's the swap-in. (Both must use the same
world/seed; both call `make_world(seed=0)`.)

**Expected output (solution):** RM reward climbs over ~8000 steps; all 4 contexts land
within one grid step of the true α*; expected true quality ≈ −0.013 for the policy vs
−0.135 for a fixed manual α=0.2 and −0.319 for uniform random.

## Comparing against your manual picks (real data)

With your real week-2 RM: for each of your pilot prompts, record the α you chose by
hand. Then (1) check where your α sits in the policy's learned distribution for that
context, and (2) compare RM scores of policy-α images vs your manual-α images. Three
outcomes, all informative: **agreement** (pipeline validated — automatic scoring
reproduces your tuning), **policy wins per the RM but you disagree looking at the
images** (reward hacking — the RM is being over-optimized where your preference data
is thin; the week-2 accuracy told you it's ~80%-right, and the policy seeks out the
20%), or **policy finds a genuinely better α you hadn't tried** (the payoff).

To use your real α grid/contexts, edit `ALPHA_GRID` / `C` in week 2's `make_pairs.py`
(both weeks import from there) and replace `embed_synthetic` with a lookup into your
real per-(context, α) image embeddings.

## Where to go next

Re-read tinker-cookbook lec3–4 and the RLHF recipes: rollout loop, advantage,
baseline, RM head are now each ~10 lines you've written. The natural extensions —
Gaussian policy over continuous α (REINFORCE with a mean/std head), an explicit
entropy bonus or KL-to-uniform to resist RM over-optimization — are the same
upgrades PPO makes, and good week-4 side quests before or instead of the Colab DPO rep.
