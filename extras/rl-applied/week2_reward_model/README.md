# Week 2 — A Bradley-Terry reward model on your own preferences

This is the "RM" stage of RLHF, in miniature: your pairwise judgments → a scalar
scorer. Featurizer = frozen CLIP image embeddings; the trained part is a single
linear head `s = w·e`, exactly analogous to the scalar head on a frozen/finetuned
LLM backbone in the cookbook's reward-modeling lecture.

## Run synthetic mode first (10 minutes, zero data)

```bash
python3 make_pairs.py --synthetic     # fake steering world -> data/
python3 reward_model.py               # your exercise (2 TODOs + training loop)
python3 reward_model_solution.py      # reference
```

The synthetic world mimics your real setup: 4 "prompts" (contexts), a steering
knob α, embeddings that move along a direction as α changes, hidden quality
peaking at a per-context α*, and a noisy Bradley-Terry "rater" (you) labeling
same-context pairs. Note the printed **rater-consistency ceiling (~0.84)**: near-tie
pairs are coin flips even for a perfect model. Expected result: **val accuracy ~0.79**,
sitting between chance (0.50) and that ceiling.

Two things worth noticing in the solution's output:
- The RM score correlates >0.85 with true quality **within** each context but is
  meaningless **across** contexts — same-context pairs only identify scores up to a
  per-context constant. LLM reward models have the same property per-prompt; it's why
  RLHF advantages are computed relative to other samples of the same prompt.
- Watch the train/val accuracy gap for overfitting (64-dim head, hundreds of pairs is
  fine; with very few real pairs, raise `L2`).

## Recording preference pairs from your α-sweep pilot images

1. Put the pilot images in one folder, named so context and α are recoverable,
   e.g. `prompt03_alpha+0.35.png`.
2. Make `judgments.csv` with header `imgA,imgB,winner` — winner is the filename of
   the image closer to your target quality. Compare **within the same prompt/context,
   different α** (that's the decision the week-3 policy has to make).
3. Practical labeling tips: randomize left/right when you view them; do 100–300 pairs
   (about an hour — pairs per context matter more than total); re-judge ~20 duplicated
   pairs to measure your own consistency — that number is your accuracy ceiling.
4. Then:

```bash
pip install sentence-transformers --break-system-packages   # only needed this once
python3 make_pairs.py --images ./pilot_images --csv ./judgments.csv
python3 reward_model_solution.py
```

## What validation accuracy means for the grant

The held-out accuracy is literally the milestone *"validate automatic scores
against human labels"*: train on ~80% of your judgments, report agreement on the
untouched 20%. Report it as: **"an automatic scorer (CLIP + linear head) agrees with
held-out human preference judgments X% of the time (chance 50%, rater self-consistency
Y%)"** — where Y comes from your re-judged duplicates. X ≥ ~70% with real data is a
solid pilot result; also report pairs-per-context and the train/val protocol.
