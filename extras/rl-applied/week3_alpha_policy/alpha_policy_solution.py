"""Contextual-bandit REINFORCE over the steering knob alpha -- SOLUTION.

Works out of the box (builds a ground-truth synthetic RM if week 2's trained
head isn't there):     python3 alpha_policy_solution.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "week2_reward_model"))
from make_pairs import ALPHA_GRID, C, embed_synthetic, make_world, quality  # noqa: E402

RM_PATH = os.path.join(HERE, "..", "week2_reward_model", "data", "rm_head.npz")
N_ALPHA = len(ALPHA_GRID)
STEPS = 8000
LR = 0.2
B_EMA = 0.05          # per-context baseline EMA rate
MANUAL_ALPHA = 0.2    # stand-in for "one hand-tuned alpha used everywhere"
SEED = 0


def load_rm(world):
    """Prefer the week-2 trained head; otherwise build the ground-truth linear
    RM by solving for w with  w . E[embed(c, a)] = quality(c, a)  exactly."""
    if os.path.exists(RM_PATH):
        w, src = np.load(RM_PATH)["w"], "trained week-2 RM head (data/rm_head.npz)"
    else:
        A = np.vstack([world["mu"], world["v"], world["u"][None]])
        b = np.concatenate([-world["alpha_star"] ** 2, 2 * world["alpha_star"], [-1.0]])
        w, src = np.linalg.lstsq(A, b, rcond=None)[0], "ground-truth synthetic RM"
    # RM scale is arbitrary (only orderings matter); normalize so LR is stable.
    # Real RLHF stacks whiten/clip rewards for the same reason.
    print(f"reward model: {src}")
    return w / np.linalg.norm(w)


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
        probs = softmax(theta[c])
        k = int(rng.choice(N_ALPHA, p=probs))          # policy picks an alpha
        e = embed_synthetic(world, c, ALPHA_GRID[k], rng)   # "generate image"
        r = float(w_rm @ e)                            # RM scores it
        adv = r - baseline[c]
        onehot = np.zeros(N_ALPHA)
        onehot[k] = 1.0
        theta[c] += LR * adv * (onehot - probs)        # week-1 update, verbatim
        baseline[c] += B_EMA * (r - baseline[c])
        rew.append(r)
        if (t + 1) % 2000 == 0:
            print(f"step {t + 1:5d}  avg RM reward (last 2000): "
                  f"{np.mean(rew[-2000:]):+.3f}")
    print(f"(first 500 steps averaged {np.mean(rew[:500]):+.3f})")
    return theta


def evaluate(world, w_rm, theta):
    """Exact expected true quality of: the learned policy, uniform-random alpha,
    and one fixed manual alpha -- averaged over contexts."""
    Q = np.array([[quality(world, c, a) for a in ALPHA_GRID] for c in range(C)])
    pi = np.array([softmax(theta[c]) for c in range(C)])
    print("\nper-context result (grid step = 0.175):")
    ok = 0
    for c in range(C):
        k = int(np.argmax(pi[c]))
        a_star = world["alpha_star"][c]
        hit = abs(ALPHA_GRID[k] - a_star) <= 0.176
        ok += hit
        print(f"  context {c}: policy alpha = {ALPHA_GRID[k]:+.3f} "
              f"(p={pi[c, k]:.2f})   true alpha* = {a_star:+.2f}   "
              f"{'OK' if hit else 'MISS'}")
    print(f"{ok}/{C} contexts within one grid step of the true optimum")
    pol = float((pi * Q).sum(axis=1).mean())
    uni = float(Q.mean())
    man = float(np.mean([quality(world, c, MANUAL_ALPHA) for c in range(C)]))
    print(f"\nexpected TRUE quality (0 is perfect):")
    print(f"  learned policy       : {pol:+.4f}")
    print(f"  fixed manual alpha={MANUAL_ALPHA} : {man:+.4f}")
    print(f"  uniform random alpha : {uni:+.4f}")
    print("The policy adapts alpha per context; a single manual alpha can't.")


if __name__ == "__main__":
    world = make_world(seed=0)         # must match week 2's world (same seed)
    w_rm = load_rm(world)
    theta = train(world, w_rm)
    evaluate(world, w_rm, theta)
