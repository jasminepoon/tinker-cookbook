"""Build (image embeddings, preference pairs) for the Bradley-Terry reward model.

Synthetic mode (no data, no GPU, no extra deps):
    python3 make_pairs.py --synthetic

Real mode (your alpha-sweep pilot images + your judgments):
    python3 make_pairs.py --images ./pilot_images --csv ./judgments.csv
  judgments.csv columns: imgA,imgB,winner   (filenames; winner = imgA or imgB)
  Needs sentence-transformers or open_clip_torch installed (only for this step).

Outputs into --out (default ./data):
    embeddings.npy  (N, D) float32     one row per image
    pairs.csv       idx_a,idx_b,winner (winner: 0 = a preferred, 1 = b preferred)
    items.csv       bookkeeping (name / context / alpha per row)
    meta.npz        synthetic mode only: world params + true quality (used by week 3)
"""
import argparse
import csv
import os

import numpy as np

# ---- the synthetic "steering world" (also imported by week 3) ---------------
# An image embedding is  e = mu_c + alpha * v_c + alpha^2 * u + noise :
# a per-context base vector, a context-specific steering direction scaled by
# the knob alpha, and a shared curvature direction. True quality peaks at a
# context-dependent alpha* -- like a style knob with a sweet spot per prompt.
D, C = 64, 4
ALPHA_GRID = np.linspace(-0.7, 0.7, 9)
ALPHA_STAR = np.array([-0.4, -0.1, 0.2, 0.5])          # hidden optimum per context
EMBED_NOISE = 0.05


def make_world(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "mu": rng.normal(0, 1, (C, D)),
        "v": rng.normal(0, 1, (C, D)) / np.sqrt(D),
        "u": rng.normal(0, 1, D) / np.sqrt(D),
        "alpha_star": ALPHA_STAR,
    }


def embed_synthetic(world, c, a, rng):
    return world["mu"][c] + a * world["v"][c] + a * a * world["u"] \
        + EMBED_NOISE * rng.normal(size=D)


def quality(world, c, a):
    return -(a - world["alpha_star"][c]) ** 2


# ---- synthetic dataset -------------------------------------------------------
def gen_synthetic(out, n_items=600, n_pairs=800, beta=10.0, seed=0):
    rng = np.random.default_rng(seed + 1)
    world = make_world(seed)
    ctx = rng.integers(0, C, n_items)
    alph = rng.choice(ALPHA_GRID, n_items)
    E = np.stack([embed_synthetic(world, c, a, rng) for c, a in zip(ctx, alph)])
    q = np.array([quality(world, c, a) for c, a in zip(ctx, alph)])

    # pairs compare images of the SAME context (same prompt, different alpha),
    # judged by a noisy Bradley-Terry rater: P(a wins) = sigmoid(beta * (qa - qb))
    pairs, p_as = [], []
    while len(pairs) < n_pairs:
        i, j = rng.integers(0, n_items, 2)
        if i == j or ctx[i] != ctx[j]:
            continue
        p_a = 1.0 / (1.0 + np.exp(-beta * (q[i] - q[j])))
        pairs.append((i, j, 0 if rng.random() < p_a else 1))
        p_as.append(p_a)
    # No model can beat the rater's own consistency: near-tie pairs are coin
    # flips. This ceiling is what a PERFECT reward model would score.
    ceiling = np.maximum(p_as, 1.0 - np.asarray(p_as)).mean()

    write_outputs(out, E.astype(np.float32), pairs,
                  items=[(k, f"synthetic_{k}", int(ctx[k]), float(alph[k]))
                         for k in range(n_items)])
    np.savez(os.path.join(out, "meta.npz"), contexts=ctx, alphas=alph,
             quality=q, **world)
    print(f"[synthetic] {n_items} items, {len(pairs)} pairs, "
          f"{C} contexts, true alpha* = {ALPHA_STAR}")
    print(f"[synthetic] rater-consistency accuracy ceiling: {ceiling:.3f} "
          "(a perfect RM scores this, not 1.0)")


# ---- real dataset ------------------------------------------------------------
def embed_images(paths):
    try:
        from PIL import Image
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("clip-ViT-B-32")
        return np.asarray(model.encode([Image.open(p).convert("RGB") for p in paths]))
    except ImportError:
        raise SystemExit(
            "Real mode needs CLIP embeddings. Install one of:\n"
            "  pip install sentence-transformers   (used here)\n"
            "  pip install open_clip_torch pillow  (then adapt embed_images)\n"
            "Or run with --synthetic first.")


def gen_real(out, images_dir, csv_path):
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f)]
    names = sorted({r["imgA"] for r in rows} | {r["imgB"] for r in rows})
    idx = {n: k for k, n in enumerate(names)}
    E = embed_images([os.path.join(images_dir, n) for n in names])
    pairs = [(idx[r["imgA"]], idx[r["imgB"]],
              0 if r["winner"].strip() == r["imgA"] else 1) for r in rows]
    write_outputs(out, E.astype(np.float32), pairs,
                  items=[(idx[n], n, "", "") for n in names])
    print(f"[real] embedded {len(names)} images (D={E.shape[1]}), {len(pairs)} pairs")


# ---- shared output writer ----------------------------------------------------
def write_outputs(out, E, pairs, items):
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "embeddings.npy"), E)
    with open(os.path.join(out, "pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx_a", "idx_b", "winner"])
        w.writerows(pairs)
    with open(os.path.join(out, "items.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "name", "context", "alpha"])
        w.writerows(items)
    print(f"wrote embeddings.npy {E.shape}, pairs.csv, items.csv -> {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--images", help="folder of pilot images (real mode)")
    ap.add_argument("--csv", help="pairwise judgments csv (real mode)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--n-pairs", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.synthetic:
        gen_synthetic(args.out, n_pairs=args.n_pairs, seed=args.seed)
    elif args.images and args.csv:
        gen_real(args.out, args.images, args.csv)
    else:
        ap.error("use --synthetic, or --images DIR --csv FILE")
