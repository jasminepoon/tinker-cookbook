"""Contextual-bandit REINFORCE over the steering knob alpha -- EXERCISE.

This closes the RLHF loop on your problem:
    your preferences (wk 2 data) -> reward model (wk 2) -> policy (this file).
Each step: a context ("prompt") arrives, the policy picks one of 9 discrete
alpha values, the world "generates an image" (an embedding), the RM scores it,
and REINFORCE nudges the policy. All three TODOs are week-1 concepts, reused.

Run: python3 alpha_policy.py     (works with no week-2 artifacts: it falls
back to a ground-truth synthetic RM. After week 2, it auto-loads your head.)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "week2_reward_model"))
from make_pairs import ALPHA_GRID, C, embed_synthetic, make_world, quality  # noqa: E402

RM_PATH = os.path.join(HERE, "..", "week2_reward_model", "data", "rm_head.npz")
N_ALPHA = len(ALPHA_GRID)          # 9 alphas in [-0.7, 0.7]
STEPS = 8000
LR = 0.2
B_EMA = 0.05          # per-context baseline EMA rate
MANUAL_ALPHA = 0.2    # stand-in for "one hand-tuned alpha used everywhere"
SEED = 0


def load_rm(world):
    """Provided. Week-2 trained head if present, else ground-truth linear RM."""
    if os.path.exists(RM_PATH):
        w, src = np.load(RM_PATH)["w"], "trained week-2 RM head (data/rm_head.npz)"
    else:
        A = np.vstack([world["mu"], world["v"], world["u"][None]])
        b = np.concatenate([-world["alpha_star"] ** 2, 2 * world["alpha_star"], [-1.0]])
        w, src = np.linalg.lstsq(A, b, rcond=None)[0], "ground-truth synthetic RM"
    print(f"reward model: {src}")
    return w / np.linalg.norm(w)   # RM scale is arbitrary; whiten for stable LR


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def train(world, w_rm, seed=SEED):
    rng = np.random.default_rng(seed)
    theta = np.zeros((C, N_ALPHA))     # one softmax over alphas per context
    baseline = np.zeros(C)             # per-context running-mean reward
    rew = []
    for t in range(STEPS):
        c = int(rng.integers(C))                       # a "prompt" arrives
        # ----------------------------------------------------------------
        # TODO 1: sample an alpha index k from softmax(theta[c]), build the
        # embedding, and get the reward:
        #     e = embed_synthetic(world, c, ALPHA_GRID[k], rng)
        #     r = float(w_rm @ e)
        # ----------------------------------------------------------------
        raise NotImplementedError("TODO 1: sample alpha, get RM reward")
        # ----------------------------------------------------------------
        # TODO 2: REINFORCE update on theta[c] with advantage r - baseline[c].
        # Identical in shape to bandit.py TODO 2. Why is the baseline
        # PER-CONTEXT here? (Hint: contexts have different mean RM scores --
        # same reason RLHF advantages are relative to the same prompt.)
        # ----------------------------------------------------------------
        raise NotImplementedError("TODO 2: log-prob x advantage update")
        # ----------------------------------------------------------------
        # TODO 3: update the per-context baseline EMA toward r.
        # ----------------------------------------------------------------
        raise NotImplementedError("TODO 3: per-context baseline")
        rew.append(r)
        if (t + 1) % 2000 == 0:
            print(f"step {t + 1:5d}  avg RM reward (last 2000): "
                  f"{np.mean(rew[-2000:]):+.3f}")
    return theta


def evaluate(world, w_rm, theta):
    """Provided: compares the learned policy to a fixed manual alpha and to
    uniform-random alpha on TRUE quality (known in the synthetic world)."""
    Q = np.array([[quality(world, c, a) for a in ALPHA_GRID] for c in range(C)])
    pi = np.array([softmax(theta[c]) for c in range(C)])
    print("\nper-context result (grid step = 0.175):")
    for c in range(C):
        k = int(np.argmax(pi[c]))
        print(f"  context {c}: policy alpha = {ALPHA_GRID[k]:+.3f} "
              f"(p={pi[c, k]:.2f})   true alpha* = {world['alpha_star'][c]:+.2f}")
    pol = float((pi * Q).sum(axis=1).mean())
    man = float(np.mean([quality(world, c, MANUAL_ALPHA) for c in range(C)]))
    print(f"\nexpected TRUE quality: policy {pol:+.4f} | manual alpha={MANUAL_ALPHA} "
          f"{man:+.4f} | uniform {float(Q.mean()):+.4f}")
    # Success: every context's alpha within one grid step of alpha*, and
    # policy quality well above both baselines (solution gets ~-0.013).


if __name__ == "__main__":
    world = make_world(seed=0)         # must match week 2's world (same seed)
    w_rm = load_rm(world)
    theta = train(world, w_rm)
    evaluate(world, w_rm, theta)
