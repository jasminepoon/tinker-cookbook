"""Bradley-Terry reward model: linear head on image embeddings -- EXERCISE.

The RM assigns each image a scalar score s = w . embedding. Bradley-Terry says
your judgments are noisy comparisons of those scores:
    P(a preferred over b) = sigmoid(s_a - s_b)
Training = maximum likelihood on your pairs. This is EXACTLY the loss used for
LLM reward models (and the shape inside DPO); only the featurizer differs.

Run after make_pairs.py has populated ./data:
    python3 make_pairs.py --synthetic && python3 reward_model.py
"""
import csv
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VAL_FRAC = 0.2
LR = 2.0
L2 = 1e-3
ITERS = 4000
SEED = 0


def load_data():
    E = np.load(os.path.join(DATA, "embeddings.npy")).astype(np.float64)
    with open(os.path.join(DATA, "pairs.csv")) as f:
        pairs = np.array([[int(r["idx_a"]), int(r["idx_b"]), int(r["winner"])]
                          for r in csv.DictReader(f)])
    return E, pairs


def log_sigmoid(x):
    """Numerically stable log(sigmoid(x)). Provided so numerics aren't the exercise."""
    return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))


def sigmoid(x):
    return np.exp(log_sigmoid(x))


def make_features(E, pairs):
    """X[k] = e_winner - e_loser for pair k, so the BT margin is simply X @ w.
    (winner column: 0 means idx_a won, 1 means idx_b won.) Provided."""
    delta = E[pairs[:, 0]] - E[pairs[:, 1]]        # e_a - e_b
    sign = 1.0 - 2.0 * pairs[:, 2]                 # +1 if a won, -1 if b won
    return delta * sign[:, None]


def bt_loss_and_grad(w, X, l2=L2):
    """TODO 1: the Bradley-Terry loss and its gradient.
      margin m = X @ w                     (score of winner minus loser)
      loss    = -mean(log_sigmoid(m)) + 0.5 * l2 * ||w||^2
      grad    = ?
    HINT: d/dm[-log sigmoid(m)] = -(1 - sigmoid(m)), and dm/dw = X, so
      grad = -X.T @ (1 - sigmoid(m)) / len(X) + l2 * w.
    Derive that derivative yourself before pasting the hint. Compare with the
    DPO loss (lec6): same -log sigmoid(margin), margin built from log-ratios."""
    raise NotImplementedError("TODO 1: BT loss + gradient")


def accuracy(w, X):
    """TODO 2: fraction of pairs where the model agrees with the human,
    i.e. margin X @ w > 0. (Count exact ties as half.)"""
    raise NotImplementedError("TODO 2: pairwise accuracy")


def main():
    E, pairs = load_data()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pairs))
    n_val = int(len(pairs) * VAL_FRAC)
    val, train = pairs[perm[:n_val]], pairs[perm[n_val:]]
    X_tr, X_va = make_features(E, train), make_features(E, val)
    print(f"{len(train)} train pairs, {len(val)} val pairs, D={E.shape[1]}")

    w = np.zeros(E.shape[1])
    for it in range(ITERS + 1):
        # ----------------------------------------------------------------
        # TODO 3: full-batch gradient DESCENT on the BT loss.
        # Log loss / train acc / val acc every 1000 iters -- watch whether
        # train and val accuracy diverge (that gap is your overfitting
        # signal; more pairs or more L2 is the fix).
        # ----------------------------------------------------------------
        raise NotImplementedError("TODO 3: training loop body")

    val_acc = accuracy(w, X_va)
    print(f"\nFINAL held-out accuracy: {val_acc:.3f}  (chance = 0.500)")
    print("-> grant milestone number: RM predicts held-out human preferences "
          f"{100 * val_acc:.1f}% of the time.")
    np.savez(os.path.join(DATA, "rm_head.npz"), w=w)
    print(f"saved RM head -> {DATA}/rm_head.npz (week 3 loads this)")
    # Success on synthetic data: val acc ~0.79 vs the ~0.84 rater-consistency
    # ceiling printed by make_pairs.py. If you hit ~0.5, check the sign of your
    # gradient; if train >> val, you are overfitting.


if __name__ == "__main__":
    main()
