"""Bradley-Terry reward model: linear head on image embeddings -- SOLUTION.

Run after make_pairs.py has populated ./data:
    python3 make_pairs.py --synthetic && python3 reward_model_solution.py
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
    """Numerically stable log(sigmoid(x))."""
    return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))


def sigmoid(x):
    return np.exp(log_sigmoid(x))


def make_features(E, pairs):
    """X[k] = e_winner - e_loser for pair k, so the BT margin is X @ w."""
    delta = E[pairs[:, 0]] - E[pairs[:, 1]]        # e_a - e_b
    sign = 1.0 - 2.0 * pairs[:, 2]                 # +1 if a won, -1 if b won
    return delta * sign[:, None]


def bt_loss_and_grad(w, X, l2=L2):
    """Bradley-Terry: loss = -mean log sigmoid(margin), margin = X @ w.
    d/dm [-log sigmoid(m)] = -(1 - sigmoid(m)), so grad = -X^T (1-sig)/M."""
    m = X @ w
    loss = -log_sigmoid(m).mean() + 0.5 * l2 * w @ w
    grad = -(X.T @ (1.0 - sigmoid(m))) / len(X) + l2 * w
    return loss, grad


def accuracy(w, X):
    """A pair is 'correct' if the model scores the human winner higher
    (exact ties, e.g. at w = 0, count as half)."""
    m = X @ w
    return float((m > 0).mean() + 0.5 * (m == 0).mean())


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
        loss, grad = bt_loss_and_grad(w, X_tr)
        if it % 1000 == 0:
            print(f"iter {it:4d}  BT loss {loss:.4f}  "
                  f"train acc {accuracy(w, X_tr):.3f}  val acc {accuracy(w, X_va):.3f}")
        w -= LR * grad

    val_acc = accuracy(w, X_va)
    print(f"\nFINAL held-out accuracy: {val_acc:.3f}  (chance = 0.500)")
    print("-> grant milestone number: RM predicts held-out human preferences "
          f"{100 * val_acc:.1f}% of the time.")

    np.savez(os.path.join(DATA, "rm_head.npz"), w=w)
    print(f"saved RM head -> {DATA}/rm_head.npz (week 3 loads this)")

    meta_path = os.path.join(DATA, "meta.npz")
    if os.path.exists(meta_path):  # synthetic mode: we know the hidden quality
        meta = np.load(meta_path)
        q, ctx, s = meta["quality"], meta["contexts"], E @ w
        # Pairs compare within a context, so the RM is only identified up to a
        # per-context constant (same reason RLHF RMs are per-prompt relative!).
        # Correlate within each context, not globally.
        corrs = [np.corrcoef(s[ctx == c], q[ctx == c])[0, 1]
                 for c in np.unique(ctx)]
        print("[synthetic sanity check] within-context corr(RM score, true "
              f"quality) = {np.round(corrs, 3)}")


if __name__ == "__main__":
    main()
