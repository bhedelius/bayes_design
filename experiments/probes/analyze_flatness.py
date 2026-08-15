"""Does spread across reaction states carry information beyond the level?

H3 (Albery-Knowles) predicts evolved enzymes have a *level* catalytic profile.
The design field's filter maximises quality at each state independently, which is
a claim about the level, not the spread. So the hypothesis is only interesting if
spread separates evolved from designed enzymes **after** the level is accounted
for - otherwise "flat" is just a restatement of "good everywhere".

Reported, natural vs designed:
  1. each predictor alone: AUC and a rank-sum p
  2. the conditional test: a likelihood-ratio test of
     logit(natural) ~ level  against  logit(natural) ~ level + spread
     The 1-df LRT on the added term is the H3 test.

Sign convention: H3 predicts natural enzymes are FLATTER, i.e. lower spread, so a
negative spread coefficient supports it.
"""

import argparse
import glob
import json

import numpy as np
from scipy import stats

STATES = ["apo", "TI1", "AEI", "TI2"]


def auc(pos, neg):
    """Probability a random positive outranks a random negative."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def logistic_ll(beta, X, y):
    z = X @ beta
    # stable log-likelihood
    return float(np.sum(y * z - np.logaddexp(0, z)))


def fit_logistic(X, y, iters=200):
    """Newton-IRLS with a small ridge term; returns (beta, loglik)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (y - p) - 1e-6 * beta
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta, logistic_ll(beta, X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="placer_flatness_*.json")
    ap.add_argument("--out", default="flatness_analysis.json")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(a.glob)):
        rows.extend(json.load(open(f)))
    if not rows:
        raise SystemExit("no results found")
    for r in rows:
        v = [r[f"{s}_triad"] for s in STATES]
        r["level"] = float(np.mean(v))
        r["spread"] = float(np.std(v, ddof=1))
        r["minstate"] = float(np.min(v))

    rows.sort(key=lambda r: (r["tier"], -r["level"]))
    print("=" * 88)
    print(f"{'label':<22}{'tier':<13}" + "".join(f"{s:>7}" for s in STATES)
          + f"{'level':>8}{'spread':>8}{'min':>7}")
    print("=" * 88)
    for r in rows:
        print(f"{r['label'][:21]:<22}{r['tier']:<13}"
              + "".join(f"{r[f'{s}_triad']:>7.3f}" for s in STATES)
              + f"{r['level']:>8.3f}{r['spread']:>8.3f}{r['minstate']:>7.3f}")

    nat = [r for r in rows if r["tier"] == "natural"]
    des = [r for r in rows if r["tier"].startswith("design")]
    print(f"\nn natural = {len(nat)}, n designed = {len(des)}")
    if len(nat) < 3 or len(des) < 3:
        raise SystemExit("too few in a group for the comparison")

    print("\n" + "=" * 88)
    print("Each predictor alone (AUC = P(natural ranks higher); 0.5 = no signal)")
    print("=" * 88)
    print(f"{'predictor':<12}{'natural':>18}{'designed':>18}{'AUC':>8}{'rank-sum p':>13}")
    res = {}
    for k in ["level", "spread", "minstate"] + [f"{s}_triad" for s in STATES]:
        x = np.array([r[k] for r in nat])
        y = np.array([r[k] for r in des])
        p = stats.mannwhitneyu(x, y).pvalue
        A = auc(x, y)
        res[k] = {"natural_mean": float(x.mean()), "designed_mean": float(y.mean()),
                  "auc": A, "p": float(p)}
        print(f"{k:<12}{x.mean():>10.3f}+/-{x.std(ddof=1) / np.sqrt(len(x)):<6.3f}"
              f"{y.mean():>10.3f}+/-{y.std(ddof=1) / np.sqrt(len(y)):<6.3f}{A:>8.3f}{p:>13.4f}")

    # ---- the conditional test -------------------------------------------------
    lab = np.array([1.0] * len(nat) + [0.0] * len(des))
    lev = np.array([r["level"] for r in nat + des])
    spr = np.array([r["spread"] for r in nat + des])
    zs = lambda v: (v - v.mean()) / (v.std(ddof=1) + 1e-12)  # noqa: E731
    ones = np.ones(len(lab))
    X0 = np.column_stack([ones, zs(lev)])
    X1 = np.column_stack([ones, zs(lev), zs(spr)])
    b0, ll0 = fit_logistic(X0, lab)
    b1, ll1 = fit_logistic(X1, lab)
    lr = 2 * (ll1 - ll0)
    p_lr = float(stats.chi2.sf(max(lr, 0), 1))

    print("\n" + "=" * 88)
    print("Conditional test: does spread add anything once level is in the model?")
    print("=" * 88)
    print(f"  level only        : loglik = {ll0:.4f}   beta_level  = {b0[1]:+.3f}")
    print(f"  level + spread    : loglik = {ll1:.4f}   beta_level  = {b1[1]:+.3f}"
          f"   beta_spread = {b1[2]:+.3f}")
    print(f"  likelihood-ratio  : chi2(1) = {lr:.3f}   p = {p_lr:.4f}")
    print("\n  H3 predicts natural enzymes are FLATTER -> beta_spread < 0.")
    if p_lr < 0.05 and b1[2] < 0:
        print("  => spread adds information beyond level, in the direction H3 predicts.")
    elif p_lr < 0.05:
        print("  => spread adds information, but in the OPPOSITE direction to H3.")
    else:
        print("  => spread adds nothing beyond level. H3 not supported by this test.")

    res["conditional"] = {"ll_level": ll0, "ll_level_spread": ll1, "lr_chi2": float(lr),
                          "p": p_lr, "beta_level": float(b1[1]), "beta_spread": float(b1[2])}
    res["n_natural"], res["n_designed"] = len(nat), len(des)
    json.dump({"rows": rows, "stats": res}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
