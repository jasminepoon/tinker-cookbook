"""REINFORCE on a 10-armed Gaussian bandit -- SOLUTION.

Run: python3 bandit_solution.py
"""
import numpy as np

K = 10          # number of arms
STEPS = 5000
LR = 0.1
BASELINE_EMA = 0.05


class Bandit:
    """Arm i pays N(mean_i, 1). Means sit around +5: the offset is deliberate --
    it makes the no-baseline gradient variance painfully visible."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.means = 5.0 + rng.permutation(np.linspace(-2.0, 2.0, K))
        self.best_arm = int(np.argmax(self.means))
        self.rng = np.random.default_rng(seed + 1)

    def pull(self, arm):
        return self.means[arm] + self.rng.normal(0.0, 1.0)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def train(use_baseline, seed=0):
    rng = np.random.default_rng(seed)
    env = Bandit(seed=seed + 100)
    theta = np.zeros(K)          # policy params: pi = softmax(theta)
    baseline = 0.0
    rewards, grad_norms = [], []

    for _ in range(STEPS):
        # --- sample an action from the softmax policy ---
        probs = softmax(theta)
        a = rng.choice(K, p=probs)
        r = env.pull(a)

        # --- REINFORCE update ---
        # grad_theta log pi(a) for a softmax policy = onehot(a) - probs.
        # Scale by the advantage (r - baseline) and ascend.
        adv = (r - baseline) if use_baseline else r
        onehot = np.zeros(K)
        onehot[a] = 1.0
        grad = adv * (onehot - probs)
        theta += LR * grad

        # --- running-mean baseline (exponential moving average of reward) ---
        if use_baseline:
            baseline += BASELINE_EMA * (r - baseline)

        rewards.append(r)
        grad_norms.append(float(np.linalg.norm(grad)))

    return env, theta, np.array(rewards), np.array(grad_norms)


def report(name, env, theta, rewards, grad_norms):
    probs = softmax(theta)
    print(f"\n=== {name} ===")
    print(f"best arm: {env.best_arm} (mean payout {env.means[env.best_arm]:.2f})")
    print(f"avg reward  first 100 steps: {rewards[:100].mean():.3f}")
    print(f"avg reward   last 100 steps: {rewards[-100:].mean():.3f}")
    print(f"policy prob on best arm: {probs[env.best_arm]:.3f}")
    # variance measured late in training, once the policy has sharpened
    print(f"gradient-norm variance (last 1000 steps): {grad_norms[-1000:].var():.4f}")
    return grad_norms[-1000:].var()


if __name__ == "__main__":
    out_b = train(use_baseline=True)
    out_n = train(use_baseline=False)
    var_b = report("WITH baseline", *out_b)
    var_n = report("WITHOUT baseline", *out_n)
    print(f"\nvariance ratio (no-baseline / baseline): {var_n / max(var_b, 1e-9):.1f}x")
    print("Lesson: the baseline changes no expectation (E[b * grad log pi] = 0),")
    print("but subtracting the ~5.0 reward offset shrinks per-step gradient noise.")
