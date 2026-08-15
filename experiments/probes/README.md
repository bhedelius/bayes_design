# Probes

Instruments for dissecting and extending the BayesDesign objective, and for asking
whether any of it reaches enzyme catalysis. The work began as a B-factor question
and moved well past it; what is kept here is the measurements and, deliberately,
the hypotheses that died and the controls that killed them.

Nothing here is a design method yet. The design-side findings are solid. The
catalysis side was uniformly negative until H3 (landscape flatness), which is the
one positive and is reported in Part 3 with its caveats attached.

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
| **H3 landscape flatness** | Albery & Knowles | **supported** — see Part 3 |
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

## Part 3 — The differential objective, and why the target moved

Every metric in Part 2 scores a **single state**. Rate does not depend on a single
state: `ΔΔG‡ = ΔG(TS) − ΔG(GS)`. The natural fix is a two-context objective —
score the sequence with the ligand in TS geometry, minus the same sequence with it
in GS geometry — which `CombinedModel` already expresses (`ligand_selectivity` is
the same shape, ±1 weights over two contexts).

LigandMPNN sees a ligand as coordinates plus element indices with **no bond graph**
(`Y`, `Y_t`, `Y_m`). So a GS→TS change is *purely* a coordinate displacement, and
feasibility becomes one measurable curve, answerable before building any poses.

### The model resolves 0.1 Å; the coordinates do not

`ts_sensitivity.py`, on 5RGE (HG3.17) with the TSA 6NT. Ceiling = removing the
ligand outright (KL 0.350 nats at the site). Floor = the same perturbation read
beyond 15 Å.

| displacement | KL at site | % of ceiling | KL beyond 15 Å |
|---|---|---|---|
| 0.1 Å | 0.0023 | 0.7% | 0.0000 |
| 0.3 Å | 0.0339 | 9.7% | 0.0000 |
| 0.5 Å | 0.0562 | 16.1% | 0.0000 |
| 1.0 Å | 0.2417 | 69.1% | 0.0000 |
| 1.5 Å | 0.3932 | 112.5% | 0.0001 |

Signal clears 3× the spatial floor at **0.2 Å**, well inside the 0.2–0.5 Å a proton
transfer moves heavy atoms. Rigid rotation is more sensitive than translation
(0.5 Å RMS → 42.7% of ceiling) because it changes which atoms face which residues.
Past ~1.5 Å the curve exceeds the ceiling: a *misplaced* ligand misleads the model
more than an absent one.

So sensitivity is not the binding constraint — **pose accuracy is**. 6NT's own
crystallographic uncertainty is `u_rms = √(B/8π²) = 0.44 Å`, comparable to the
displacement being resolved. A GS/TS differential built on crystal poses is
dominated by pose error, not by the chemistry.

This one is worth flagging against the Part 2 pattern: the boring control
*validated* the idea rather than killing it, and refuted the pessimistic
prediction that motivated running it.

### The target moved: the bottleneck is the cycle, not the first step

Lauko *et al.*, *Science* 2025 (designed serine hydrolases, RFdiffusion +
LigandMPNN + **PLACER** ensembles) tested ~509 designs across three rounds:

| round | tested | FP-labeled (site reactive) | turnover |
|---|---|---|---|
| 1 | 129 | 3 (2.3%) | 2 (1.6%) |
| 2 | 192 | 33 (17.2%) | 10 (5.2%) |
| 3 | 132 | **111 (84%)** | **2 (1.5%)** |

Round 3 forms a reactive active site 84% of the time and turns over 1.5% of the
time. **The first half-reaction is close to solved; the catalytic cycle is not.**

That relocates the problem. A GS→TS differential optimises the step that already
works. Turnover requires the *whole* cycle, and — Albery & Knowles — a fast enzyme
has a **level** free-energy profile: no state bound much more tightly than the
others. Maximising binding at any single state, including the TS, digs a
thermodynamic pit and yields a stoichiometric binder, not a catalyst.

This is **H3, still the untested hypothesis**, and it now has a benchmark. The
objective it implies is not a maximum or a difference but a *flatness*: minimise
the spread of the ligand-conditioned score across the reaction states. That is a
K-context `CombinedModel` whose terms compute a variance rather than a sum, and it
makes a falsifiable split prediction:

- single-state score at the TS predicts **FP labeling** but not turnover;
- flatness across states predicts **turnover**, conditional on FP labeling.

PLACER already scores geometric preorganisation at each state *independently* and
requires it to be high everywhere. Nobody scores the spread across states, or
penalises over-stabilisation of any one intermediate. That is the gap.

### The paper's own figure already leans this way

PLACER models four states — **apo, TI1, AEI, TI2** — and reports the fraction of
ensemble frames with all key hydrogen bonds simultaneously formed. Fig. S3
compares designs against 13 natural hydrolases by K-S test:

| state | apo | TI1 | AEI | TI2 |
|---|---|---|---|---|
| designs vs natural, p | 0.09 | **4.9e-7** | 0.03 | **1e-4** |

Designs are indistinguishable from natural enzymes in the **apo** state and diverge
sharply at the **intermediates**. That is an uneven profile — right at the start,
wrong through the middle — which is the shape H3 predicts, measured by the authors
without being framed as a flatness claim.

### Status of the benchmark: worse than expected

The per-design table (~509 sequences with pass/fail outcomes) **is not published**.
Checked and ruled out: the bioRxiv `.docx` supplement (figures only, no tables),
the *Science* SM (Table S1 lists only the ~18 *kinetically or structurally
analysed* designs across 11 rounds), and the Zenodo archive
(`10.5281/zenodo.14642126` — 125 MB, code only, mirrors the GitHub repo).

So the actives/inactives benchmark cannot be assembled by scraping. Two routes
remain: run PLACER ourselves on the deposited design structures (9DED, 9DEE, 9DEF,
9DEG, 9DEH, 9MRB) and score flatness against their published kinetics, or ask the
authors for the screening table.

### H3 tested: evolved hydrolases are flatter, and the level says nothing

First direct test of H3, and the first positive result on the catalysis side.

PLACER generates conformational ensembles for a protein with a covalent reaction
intermediate grafted onto the catalytic serine. Four states are scored per enzyme
(`make_reaction_states.py`, `placer_flatness.py`): **apo, TI1, AEI, TI2**. The
per-state readout is the fraction of 50 ensemble frames with the catalytic triad
intact — His NE2 to Ser Ogamma, His ND1 to the Asp/Glu carboxylate — a definition
identical in all four states, which is what makes the spread comparable.

11 natural serine hydrolases against 6 deposited designs from Lauko 2025:

| predictor | natural | designed | rank-sum p |
|---|---|---|---|
| **level** (mean over states) | 0.848 ± 0.080 | 0.867 ± 0.037 | 0.51 |
| **spread** (SD over states) | **0.059 ± 0.019** | **0.162 ± 0.043** | 0.096 |

The level does not separate them at all — designs are marginally *higher*. The
spread separates them roughly threefold, in the direction H3 predicts.

Conditional test, which is the one that matters:

| | |
|---|---|
| LR chi2(1), level+spread vs level | 13.69 |
| asymptotic p | 0.0002 |
| **permutation p** (5000 shuffles) | **0.0024** |
| leave-one-out AUC | 0.000 → 0.818 |

The fitted coefficients are large (beta_spread = −10.4), the signature of
quasi-separation, so the asymptotic p is not trustworthy on its own; the
permutation null (median chi2 = 0.66) is what carries the claim.

**The caveats, which are substantial:**

- n = 17, and the 6 designs are the *crystallised* ones, i.e. successes. This is
  natural-vs-designed as a proxy for good-vs-bad catalyst, not measured k_cat.
- **Internal control is a wash, leaning against**: within the design series,
  spread vs round gives Spearman +0.309 (p = 0.55, n = 6). H3 expects later,
  better rounds to be flatter — this leans the other way, far from significantly.
- Every enzyme is modelled with the same butyryl acyl group, which is wrong for
  chymotrypsin, AChE and the lipases specifically.
- Subtilisin is an outlier (level 0.135) and 1SBT is a low-resolution structure.

So: a real positive on a pre-registered prediction with the controls attached,
and not yet a result about catalysis until it is repeated against measured rates.

### The bug that produced the first, opposite answer

Worth recording because it nearly became a finding. The first run had 8 of 12
natural enzymes scoring **exactly 0.000** at all three ligand states with apo
≈ 1.0, and the analysis duly reported H3 unsupported.

Chymotrypsin and AChE obviously form these intermediates — that is their
mechanism — so the zeros were the tell. PLACER crops ~600 atoms around the
ligand, and under `--mutate` alone it centres that crop on whatever heteroatom
exists. Structures whose only heteroatom was the chloride placeholder got a crop
nowhere near the active site: the triad atoms were simply absent, and `measure()`
scored every frame as "not formed". The run exited 0 and printed nothing.

Passing `--target_res` in every state fixes it; chymotrypsin goes from 0.000 to
1.000 across all four. `measure()` now raises when triad atoms are missing from
more than 20% of frames, rather than silently returning zero — an absent atom and
a broken interaction must not be the same number.

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
make_reaction_states.py derive TI1/AEI from the one PLACER ships
placer_flatness.py      four-state PLACER ensembles, triad-intact fractions
analyze_flatness.py     level vs spread, with permutation + LOO controls
ts_sensitivity.py       ligand-displacement resolution curve (GS/TS feasibility)
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
