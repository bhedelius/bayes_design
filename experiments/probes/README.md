# Probes

Instruments for dissecting and extending the BayesDesign objective, and for asking
whether any of it reaches enzyme catalysis. The work began as a B-factor question
and moved well past it; what is kept here is the measurements and, deliberately,
the hypotheses that died and the controls that killed them.

Nothing here is a design method yet. The honest summary is that the design-side
findings are solid and the catalysis-side findings are uniformly negative.

---

## Part 1 — What the BayesDesign objective is

`log p(seq|struct) − log p(seq)` is the **pointwise mutual information** between
sequence and structure:

```
log [ p(seq|struct) / p(seq) ] = log [ p(seq,struct) / (p(seq)·p(struct)) ]
```

Not a free energy. This framing predicts the observed behaviour instead of merely
describing it, because PMI has known properties:

- **PMI discounts whatever is generic.** Burial is the biggest thing both terms
  share, so burial is what cancels.
- **PMI is biased toward rare events** — the textbook problem. The preference for
  Trp/Cys/Met is that bias, not a stability effect.
- **`PMI^k` is the standard correction**, already exposed as the denominator weight
  in `CombinedModel`.

### The denominator cancels burial because it can *see* burial

Per-protein Spearman between CB neighbour count and expected Kyte-Doolittle
hydropathy — "does this term put hydrophobics in the core?" (399 structures):

| term | ρ with burial |
|---|---|
| native sequence | +0.358 ± 0.005 |
| ProteinMPNN `p(a\|struct,seq)` | +0.483 ± 0.005 |
| ProtXLNet `p(a\|seq)` | +0.335 ± 0.008 |
| BayesDesign ratio | +0.209 ± 0.013 |

ProtXLNet reads burial from hydrophobic patterning with no structure at all, so
division cancels it: only 43% of the numerator's burial signal survives. Sweeping
the denominator from a background frequency table to full ProtXLNet confirms the
mechanism — a position-independent denominator knows nothing about burial and the
ratio keeps 102% of it.

### Constraining beats optimising

Let ProteinMPNN pick the residue subject to the BayesDesign score staying within
`delta` nats of the best available (`constrained.py`):

| delta | % Bayes score kept | % MPNN likelihood kept | sum | W/C/M |
|---|---|---|---|---|
| 0 (current decode) | 100% | 0% | 100% | 13.3% |
| 1.0 | 75.4% | 59.4% | **135%** | 4.8% |
| 1.5 | 60.3% | 78.6% | **139%** | 3.9% |
| inf (ProteinMPNN) | 0% | 100% | 100% | 2.8% |

The sum exceeding 100% is the point: the last quarter of the objective costs ~60%
of the likelihood and drives the composition pathology. It also restores the
surface/core contrast for free (spread 0.046 → 0.069; native is 0.062).

### Ensemble averaging beats the sequence denominator outright

Replacing `p(seq)` with an average over Gaussian-perturbed backbones
(`ensemble_average.py`) — a product of experts, i.e. K terms of weight 1/K:

| objective | recovery | W/C/M | burial signal | profile sim |
|---|---|---|---|---|
| ProteinMPNN | 50.5% | 2.3% | +0.495 | 0.966 |
| **BayesDesign** | 27.5% | **17.2%** | **+0.103** | 0.854 |
| **ensemble avg (1.7 Å)** | **20.3%** | **0.4%** | **+0.644** | 0.943 |

More native signal removed *and* structural signal amplified rather than
destroyed. Averaging keeps what every conformer agrees on; dividing by a sequence
model destroys it. This is a one-row change in `OBJECTIVES` — no new model.

The generator does not matter: ANM normal modes and Gaussian noise trace the same
frontier at matched recovery (`mode_ensemble.py`), so physical realism buys
nothing here and "ensemble" is a misleading name for the mechanism.

### Design is local

Redesigning a 20-residue patch and re-scoring untouched positions (`locality.py`):

| shell | MPNN pick changed | Δ flexibility |
|---|---|---|
| 0–6 Å | 22.5% | −0.0004 ± 0.0018 |
| 10–12 Å | 8.7% | +0.0004 ± 0.0023 |
| >20 Å | 1.8% | −0.0014 ± 0.0014 |
| *inside the patch* | — | *−0.0138 ± 0.0046* |

Neighbours reshuffle; no shell shows a flexibility shift distinguishable from zero.
Rigidification does not propagate, so piecewise design is safe.

**Safe radius around an active site ≈ 15 Å** (`active_site_shells.py`): mutating 8
positions flips catalytic-residue identity 16.7% of the time at 6–8 Å and 0% beyond
15 Å, and **ProteinMPNN is ~2× safer than BayesDesign** at every close shell.

### The term library is ~2-dimensional

Correlating centred log-probabilities across every available backend
(`term_redundancy.py`): the three MPNN variants correlate 0.94–0.95 and agree on
74–77% of top-1 picks; ProtXLNet correlates ~0.53 with them.

**Participation ratio 1.95 of 5**, top two directions carrying 87.4% of variance.
The eight modes in `OBJECTIVES` are largely reparameterisations of a structure axis
and a sequence axis. Models trained on the same task with the same data land on the
same axis regardless of architecture — growing the space needs different *training
objectives*, not more architectures.

---

## Part 2 — Catalysis: a uniformly negative result

Four hypotheses, from enzymology rather than from what was convenient to measure.

| hypothesis | source | outcome |
|---|---|---|
| **H1 preorganization** | Warshel | **untestable from static structures** |
| **H2 conditional frustration** | Wolynes / Ferreiro | not supported |
| **H3 landscape flatness** | Albery & Knowles | never tested |
| **H4 null** | — | consistent with everything, not established |

### H1 could not be measured, and that is the finding

Warshel's reorganization energy is an *ensemble-averaged* property. Two attempts:

- **Ligand-masking** (`preorganization.py`) measured the wrong thing — holo-vs-apo
  is not GS-vs-TS. It turned out to measure ligand *complementarity*.
- **Apo/holo coordinates** (`apo_holo_preorg.py`) were swamped by artifacts:
  round-0 KE07 structures of the same variant differ by 0.089 Å and 3.9 Å depending
  on which crystal, and global RMSD grows with round. Artifacts exceed the signal.

The crystallographers who solved these structures report *"a high degree of
preorganization"* in the evolved variant. H1 is probably true and simply not
visible in single crystals. **Ensemble generators (BioEmu, AlphaFlow) now make this
reachable without MD** — the obvious next test.

### The one metric that validated does not predict catalysis

`KL[p(a|struct,ligand) ‖ p(a|struct,ligand masked)]` detects active sites robustly:
30 structures, four designed lineages, **z = 5.9–119.9** against nulls matched for
both patch size *and* burial, all at the 100th percentile. Only 17% of its variance
is explained by distance to the ligand, so it is not a repackaged distance cutoff —
it identifies which pocket residues are *energetically coupled* to the ligand.

It nonetheless fails to rank catalysts. Pre-registered test on the Gln50 series —
same backbone, same ligand, one position varied, external activity labels from
Kries et al. (packed Gln/Met/Phe beat water-filled His/Ser/Ala/Lys):

```
packed  0.1487    watery  0.1584    Mann-Whitney p = 0.571
ranking (best->worst): Q50K > Q50S > Q50F > Q50M > Q50(wt) > Q50H > Q50A
```

The worse group scores *higher*; wild-type Gln50, the best catalyst, ranks 5th of 7.

### Nothing else separates them either

A panel of six model quantities plus a crystallographic water count (`q50_panel.py`)
all interleave the two groups. **The water count — the positive control on the
labels — also fails**, so that run is uninformative about the models rather than
damning of them: a failed control is not a failed hypothesis.

Electrostatics (`electric_field.py`), PDB2PQR/AMBER charges, potential and field at
the oxyanion nitrogen N1: best descriptor p = 0.229, no separation. Q50K's lysine
makes the potential ~40 units *more* favourable for stabilising negative charge and
it is in the worse group — naive electrostatics failing in an interpretable way,
since catalysis is about *differential* stabilisation.

### Anti-naturalness: proposed and killed by its own control

Sequence likelihood anti-correlates with k_cat across the HG3 series
(r = −0.937, n = 3), and evolved positions score far below average. It looked like
"catalysis is bought with anti-naturalness".

The control (`naturalness_control.py`) refutes it:

| test | result |
|---|---|
| vs random-mutation null | mean **z = +2.84**, 13/14 variants positive |
| active site vs rest | +0.113 ± 0.272 (n = 13) |

Evolution's mutations are **more** natural than random ones, and show no site
concentration. The mutated positions are simply positions where the model holds
strong preferences, so any substitution scores badly. The original correlation
tracks **mutation count**, not catalysis.

This also puts the KE07 ProteinMPNN result (ρ = −0.604, p = 0.013, 30×
site-concentrated) under suspicion — it never had the same null.

### Catalytic residues are more natural, not less

53 enzymes, 79 UniProt-annotated catalytic residues (`catalytic_pmi.py`):

| detector | AUC |
|---|---|
| conservation (**most** natural under sequence) | **0.634** |
| −PMI | 0.579 |
| −`log p(struct)` (**least** natural under structure) | 0.444 — *below chance* |

Frustration theory predicts functional sites are frustrated, and that is well
supported — but **ProteinMPNN cannot test it**: trained on a PDB full of enzymes
with their catalytic residues in place, it reproduces them as patterns. Testing
frustration needs a folding-only energy (AWSEM Frustratometer), with 0.634 as the
number to beat.

### `log p(seq|struct)` is not an energy

Full 3L×3L Hessian via autograd (`hessian.py`). 82% of *single-residue*
displacements sit at a local maximum, but only ~59% of *collective* modes are
stable, and a uniform-spring elastic network beats the learned Hessian 3× at
predicting B-factors (+0.544 vs +0.174). Combining them is worse than ANM alone.
Invariant across all four training-noise checkpoints (0.02–0.30 Å), so this is not
a regularisation artifact.

---

## Methodological notes

Two failure modes recurred and are worth stating:

1. **A failed proxy is not a failed hypothesis.** Several hypotheses were declared
   refuted when the operationalisation was the weak link — H1 most clearly.
2. **Run the boring control first.** The anti-naturalness claim survived twenty
   minutes past the point a random-mutation null would have killed it.

Also: these probes use one-out conditionals, a *pseudo*-likelihood.
`Σᵢ log p(aᵢ|seq₋ᵢ) ≠ log p(seq)`. The repo's default
`bidirectional_autoregressive` decode telescopes correctly, so decode-time
magnitudes may differ from what is measured here.

---

## Scripts

```
fetch_ids.py            RCSB search -> ids.txt (30% identity cluster reps)
prep_structures.py      download PDBs -> structures400.json
score_conditionals.py   per-residue one-out conditionals (+ --dump-probs)
analyze_bfactor.py      the original B-factor statistics
plot_flexibility.py     flexibility_vs_bfactor.png
constrained.py          constraint-vs-objective sweep
ensemble_average.py     ensemble averaging vs sequence denominator
mode_ensemble.py        ANM-mode vs Gaussian ensemble generators
geometric_specificity.py blurred-geometry denominator (collapses onto MPNN axis)
term_redundancy.py      effective dimensionality of the term library
locality.py             patch redesign -> perturbation vs 3D distance
curvature.py / hessian.py  second-order structure of log p; ANM baseline
fetch_catalytic.py      UniProt ACT_SITE + PDBe SIFTS -> catalytic.json
catalytic_pmi.py        catalytic residues in the (structure, sequence) plane
active_site_shells.py   safe-radius curve around active sites
ke07_stiffness.py       KE07 trajectory: active-site stiffness
ke07_site_vs_rest.py    site-specific vs global likelihood control
preorganization.py      ligand-masking KL along evolution trajectories
validate_reorg.py       positive control + noise floor for that metric
apo_holo_preorg.py      H1 via apo/holo coordinates (confounded)
catalysis_hypotheses.py H1/H2 on matched design->evolved pairs
q50_supervised.py       pre-registered test on the labelled Gln50 series
q50_panel.py            discriminator panel + water-count control
electric_field.py       PDB2PQR charges -> potential/field at the oxyanion
kemp_kinetics.py        Kemp sequences + measured kinetics
plm_vs_kcat.py          does sequence likelihood track k_cat?
naturalness_control.py  random-mutation null (killed the above)
run_ribo.sh             shard scoring across 8 GPUs
```

Datasets: 399 X-ray structures ≤2.0 Å, single chain, no non-polymer entities,
70–250 residues, 30% identity cluster representatives (54,751 residues); 53 enzymes
with 79 annotated catalytic residues; 30 Kemp eliminase structures across four
designed lineages with measured or ordinal kinetics.

Large outputs (`residues*.csv`, `probs/`, `structures*.json`, `pdbs*/`) are
gitignored; regenerate with `run_ribo.sh` (~6 min on 8 GPUs).

---

## Open

1. **H1 with ensemble generators.** BioEmu gives equilibrium ensembles in minutes;
   preorganization becomes measurable as an ensemble-averaged field.
2. **H3 landscape flatness** — untested. KE07 has apo, substrate and product
   structures across rounds.
3. **Frustration with AWSEM**, not a PDB-trained model. Baseline to beat: 0.634.
4. **A benchmark with actives *and* inactives.** Every catalysis test here used
   variants that all work, which is the hardest possible discrimination and not the
   one that matters. Design campaigns report both outcomes.
5. **Ensemble averaging into `OBJECTIVES`** — the one result ready to use.
