"""REINFORCE on a 5x5 gridworld -- EXERCISE.

New vs. the bandit: STATE. The policy is now a table theta[state, action],
rewards arrive over a whole trajectory, and the "reward" for an action is the
discounted RETURN-TO-GO from that timestep, not the immediate reward.

Fill in TODOs 1-4, then run: python3 gridworld.py
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
    Reward: -1 per step, +10 on reaching the goal. (Given -- not a TODO.)"""
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
    """One episode. Returns a list of (state, action, reward)."""
    s, traj = START, []
    for _ in range(MAX_STEPS):
        # ------------------------------------------------------------------
        # TODO 1: sample action a from softmax(theta[s]) -- same as bandit
        # TODO 1, except the logits are now indexed by the current state.
        # ------------------------------------------------------------------
        raise NotImplementedError("TODO 1: state-conditioned sampling")
        s2, r, done = step(s, a)
        traj.append((s, a, r))
        s = s2
        if done:
            break
    return traj


def returns_to_go(rewards):
    """TODO 2: G_t = r_t + GAMMA * G_{t+1}. Return a list, same length,
    where element t is the discounted return from timestep t onward.
    HINT: iterate over `reversed(rewards)` accumulating G, then reverse."""
    raise NotImplementedError("TODO 2: returns-to-go")


def train(use_baseline, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.zeros((N_STATES, N_ACTIONS))
    V = np.zeros(N_STATES)                      # per-state value baseline
    lengths = []
    for _ in range(EPISODES):
        traj = rollout(theta, rng)
        Gs = returns_to_go([t[2] for t in traj])
        for (s, a, _), G in zip(traj, Gs):
            # --------------------------------------------------------------
            # TODO 3: the tabular REINFORCE update for this (s, a, G):
            #   adv  = G - V[s]   (or just G if not use_baseline)
            #   grad of log pi(a|s) wrt theta[s] = onehot(a) - softmax(theta[s])
            #   theta[s] += LR * adv * that
            # --------------------------------------------------------------
            raise NotImplementedError("TODO 3: policy-gradient update")
            # --------------------------------------------------------------
            # TODO 4: if use_baseline, move V[s] toward G:
            #   V[s] += V_LR * (G - V[s])
            # This is a learned per-state value baseline -- the tabular
            # ancestor of the critic in actor-critic / the value head in PPO.
            # --------------------------------------------------------------
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
        print(f"greedy-policy path length: {greedy_path_length(theta)} (optimal = 8)")
    # Success: WITH baseline converges to avg length ~8 and greedy path = 8.
    # WITHOUT baseline it will likely collapse (length pinned at MAX_STEPS).
    # Exercise: explain the collapse. What sign is every return? What does
    # that do to every action the policy actually took?
