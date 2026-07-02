"""REINFORCE on a 5x5 gridworld -- SOLUTION.

Run: python3 gridworld_solution.py
"""
import numpy as np

SIZE = 5
N_STATES = SIZE * SIZE
N_ACTIONS = 4                 # 0=up 1=down 2=left 3=right
START, GOAL = 0, N_STATES - 1
MAX_STEPS = 200               # generous: a random walk must be able to find the goal
GAMMA = 0.99
LR = 0.15
V_LR = 0.1                    # baseline (value table) learning rate
EPISODES = 1500
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def step(state, action):
    """Deterministic move; walking into a wall stays put.
    Reward: -1 per step, +10 on reaching the goal."""
    r, c = divmod(state, SIZE)
    dr, dc = MOVES[action]
    r2, c2 = min(max(r + dr, 0), SIZE - 1), min(max(c + dc, 0), SIZE - 1)
    s2 = r2 * SIZE + c2
    done = s2 == GOAL
    return s2, (10.0 if done else -1.0), done


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def rollout(theta, rng):
    """One episode. Returns lists of states, actions, rewards."""
    s, traj = START, []
    for _ in range(MAX_STEPS):
        probs = softmax(theta[s])
        a = rng.choice(N_ACTIONS, p=probs)
        s2, r, done = step(s, a)
        traj.append((s, a, r))
        s = s2
        if done:
            break
    return traj


def returns_to_go(rewards):
    """G_t = r_t + GAMMA * G_{t+1}, computed backward."""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        out.append(G)
    return out[::-1]


def train(use_baseline, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros((N_STATES, N_ACTIONS))
    V = np.zeros(N_STATES)                      # per-state value baseline
    lengths = []
    for _ in range(EPISODES):
        traj = rollout(theta, rng)
        states = [t[0] for t in traj]
        actions = [t[1] for t in traj]
        Gs = returns_to_go([t[2] for t in traj])
        for s, a, G in zip(states, actions, Gs):
            adv = G - V[s] if use_baseline else G
            probs = softmax(theta[s])
            onehot = np.zeros(N_ACTIONS)
            onehot[a] = 1.0
            theta[s] += LR * adv * (onehot - probs)   # REINFORCE, tabular
            if use_baseline:
                V[s] += V_LR * (G - V[s])             # move baseline toward G
        lengths.append(len(traj))
    return theta, np.array(lengths)


def greedy_path_length(theta):
    s, steps, seen = START, 0, set()
    while s != GOAL and steps < MAX_STEPS and s not in seen:
        seen.add(s)
        s, _, _ = step(s, int(np.argmax(theta[s])))
        steps += 1
    return steps if s == GOAL else None


if __name__ == "__main__":
    for use_baseline in (True, False):
        theta, lengths = train(use_baseline)
        name = "WITH baseline (advantage = G - V(s))" if use_baseline else "WITHOUT baseline"
        print(f"\n=== {name} ===")
        for i in range(0, EPISODES, 300):
            print(f"episodes {i:4d}-{i+299:4d}: avg length {lengths[i:i+300].mean():5.1f}")
        gp = greedy_path_length(theta)
        print(f"greedy-policy path length: {gp} (optimal = 8)")
    print("\nLesson: returns are almost always negative here, so without a baseline")
    print("every sampled action gets pushed DOWN, and the policy typically collapses")
    print("into a degenerate loop that never finds the goal. The V(s) baseline gives")
    print("better-than-expected actions a POSITIVE advantage -- a well-conditioned")
    print("signal -- and the same algorithm then solves the task in a few hundred")
    print("episodes. This is the variance/baseline tradeoff, live.")
