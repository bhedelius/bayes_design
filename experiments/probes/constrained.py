"""BayesDesign as a constraint rather than an objective.

Instead of decoding argmax of the ratio, let ProteinMPNN pick the residue subject
to the BayesDesign score staying within `delta` nats of the best available score
at that position:

    pick(i) = argmax_a  p_struct(i, a)   s.t.  log bayes(i, a) >= max_a' log bayes(i, a') - delta

delta = 0   -> pure BayesDesign (the current decode)
delta = inf -> pure ProteinMPNN

Sweeping delta traces the trade between ProteinMPNN likelihood and BayesDesign
score. A per-position delta (e.g. tight in the core, loose at the surface) tests
whether the constraint can be applied selectively.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
FLEX = dict(zip("ARNDCQEGHILKMFPSTWYV",
                [0.357, 0.529, 0.463, 0.511, 0.346, 0.493, 0.497, 0.544, 0.323, 0.462,
                 0.365, 0.466, 0.295, 0.314, 0.509, 0.507, 0.444, 0.305, 0.420, 0.386]))
FLEX_V = np.array([FLEX[c] for c in AA])
WCM = np.array([c in "WCM" for c in AA])


def load(dirpath, balance=0.002):
    """Concatenate per-shard tables and conditionals, preserving shard order."""
    shards = sorted(glob.glob(os.path.join(dirpath, "probs_*.npz")),
                    key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    frames, ps, pq = [], [], []
    for path in shards:
        i = int(path.rsplit("_", 1)[1].split(".")[0])
        z = np.load(path)
        frames.append(pd.read_csv(os.path.join(dirpath, f"residues_{i}.csv")))
        ps.append(z["p_struct"])
        pq.append(z["p_seq"])
    d = pd.concat(frames, ignore_index=True)
    p_struct, p_seq = np.concatenate(ps), np.concatenate(pq)
    assert len(d) == len(p_struct), f"{len(d)} rows vs {len(p_struct)} conditionals"
    bayes = (p_struct + balance) / (p_seq + balance)
    bayes /= bayes.sum(1, keepdims=True)
    return d, p_struct, p_seq, bayes


def constrained_pick(p_struct, bayes, delta):
    """delta: scalar or per-position array of allowed nats below the best bayes score."""
    lb = np.log(bayes + 1e-12)
    floor = lb.max(1, keepdims=True) - np.asarray(delta).reshape(-1, 1)
    masked = np.where(lb >= floor, p_struct, -np.inf)
    return masked.argmax(1)


def report(name, idx, d, p_struct, bayes, native):
    lb, lps = np.log(bayes + 1e-12), np.log(p_struct + 1e-12)
    r = np.arange(len(idx))
    return {
        "setting": name,
        "recovery": (idx == native).mean(),
        "W/C/M": WCM[idx].mean(),
        "flex": FLEX_V[idx].mean(),
        "logP_mpnn": lps[r, idx].mean(),
        "logP_bayes": lb[r, idx].mean(),
        "agree_mpnn": (idx == p_struct.argmax(1)).mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--balance", type=float, default=0.002)
    a = ap.parse_args()

    d, p_struct, p_seq, bayes = load(a.dir, a.balance)
    native = np.array([AA.index(c) for c in d["aa"]])
    # Burial quartile within each structure; Q4 = most buried.
    d["burial_q"] = d.groupby("pdb")["nbrs"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
    d["Bq"] = d.groupby("pdb")["bfac"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))

    print(f"{len(d)} residues, {d.pdb.nunique()} structures\n")

    print("=== Uniform constraint: sweep delta (nats of BayesDesign score given up) ===")
    rows = []
    for delta in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]:
        idx = constrained_pick(p_struct, bayes, delta)
        rows.append({**report(f"delta={delta}", idx, d, p_struct, bayes, native)})
    t = pd.DataFrame(rows).set_index("setting")
    print(t.round(4).to_string())
    print("\n  delta=0 is the current BayesDesign decode; delta=inf is plain ProteinMPNN.")

    print("\n=== Where does a uniform constraint bind? (delta = 1.0) ===")
    idx = constrained_pick(p_struct, bayes, 1.0)
    d["_changed"] = idx != p_struct.argmax(1)
    print(d.groupby("burial_q")["_changed"].mean().rename("fraction overridden vs MPNN")
          .rename(index=lambda q: f"burial Q{q + 1}" + (" (exposed)" if q == 0 else " (buried)" if q == 3 else ""))
          .round(3).to_string())

    print("\n=== Burial-graded constraint: tight in the core, loose at the surface ===")
    grades = {
        "uniform delta=1.0": np.full(len(d), 1.0),
        "core-tight  (Q4:0.25 -> Q1:3.0)": d["burial_q"].map({3: 0.25, 2: 0.75, 1: 1.5, 0: 3.0}).to_numpy(float),
        "surface-tight (Q1:0.25 -> Q4:3.0)": d["burial_q"].map({0: 0.25, 1: 0.75, 2: 1.5, 3: 3.0}).to_numpy(float),
    }
    rows = []
    for name, delta in grades.items():
        idx = constrained_pick(p_struct, bayes, delta)
        rec = report(name, idx, d, p_struct, bayes, native)
        # Does the design still reproduce the native rigidity gradient?
        f = pd.Series(FLEX_V[idx], index=d.index)
        per = f.groupby([d.pdb, d.Bq]).mean().unstack()
        rec["flex_slope_Q1toQ4"] = (per[3] - per[0]).mean()
        rows.append(rec)
    nat_per = pd.Series(FLEX_V[native], index=d.index).groupby([d.pdb, d.Bq]).mean().unstack()
    print(pd.DataFrame(rows).set_index("setting").round(4).to_string())
    print(f"\n  native flex_slope_Q1toQ4 = {(nat_per[3] - nat_per[0]).mean():+.4f}")


if __name__ == "__main__":
    main()
