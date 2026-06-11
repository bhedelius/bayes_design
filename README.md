# BayesDesign

<img src="https://github.com/dellacortelab/bayes_design/blob/master/data/figs/bayes_design.png?raw=true" alt="BayesDesign" width="700"/>

BayesDesign designs protein sequences for a target backbone by optimizing for
**stability and conformational specificity** rather than mere sequence
recapitulation. See the [paper](https://doi.org/10.1038/s41598-023-42032-1)
(Stern et al., *Scientific Reports*, 2023) and
[preprint](https://www.biorxiv.org/content/10.1101/2022.12.28.521825v1).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dellacortelab/bayes_design/blob/master/examples/BayesDesign.ipynb)

## How it works

Most inverse-folding models give `p(sequence | structure)` — the probability of
a sequence *given* a backbone. What we actually want for a stable, specific
design is `p(structure | sequence)` — how strongly a sequence prefers the target
fold over all alternatives. By Bayes' rule:

```
p(structure | sequence)  ∝  p(sequence | structure) / p(sequence)
```

So BayesDesign divides a structure-conditioned model by a structure-free
sequence model (a "marginal") and decodes from the result. The marginal cancels
out generic amino-acid propensity, leaving the signal that is specific to the
fold.

This repo generalizes that single ratio into a **weighted log-linear combination
of models** (`CombinedModel`):

```
p(design)  ∝  exp( Σ_i  w_i · log p_i )     # normalized over the 20 amino acids
```

Positive weights are "numerator" terms (favor) and negative weights are
"denominator" terms (penalize). Each design objective (a "mode") is just a list
of `(model, weight)` terms — see [Design modes](#design-modes-objectives). This
makes negative/multi-state design (e.g. prefer one fold or ligand over another)
and model ensembles fall out of the same machinery.

## Installation

```
pip install -e .            # runtime
pip install -e .[dev]       # + dev tools (ruff, mypy, pytest)
pip install -e .[esm]       # + ESM-IF1 backend (fair-esm, torch-geometric, ...)
```

This installs two console commands: `bayes-design` (design) and
`bayes-design-experiment` (evaluation/analysis).

The MPNN-based modes need model weights, downloaded once:

```
bash src/bayes_design/ligand_mpnn/get_model_params.sh src/bayes_design/ligand_mpnn/model_params
```

(The ProteinMPNN weights used by the default `bayes_design` mode ship with the
repo. ESM-IF1 weights download automatically on first use.)

## Quickstart

Redesign a backbone (PDB id `6MRR`), keeping residues 67–68 fixed:

```
bayes-design --model_name bayes_design --protein_id 6MRR \
    --decode_order n_to_c --decode_algorithm beam --n_beams 128 \
    --fixed_positions 67 68
```

Designs are written to `./results/`. Useful options:

| option | meaning |
|--------|---------|
| `--model_name` | which design mode/objective (see below) |
| `--protein_id` | PDB id of the target backbone (downloaded if absent) |
| `--fixed_positions` | 1-indexed residue ranges to keep, e.g. `3 10 14 14` |
| `--decode_order` | `n_to_c`, `proximity`, `reverse_proximity`, `random` |
| `--decode_algorithm` | `greedy`, `beam` (`--n_beams`), `sample` (`--temperature`) |
| `--n_designs` | number of sequences to sample |

## Models

Each model implements one `ProbabilityModel` interface (`p(next residue | context)`),
so they are interchangeable as terms of an objective.

| model | gives | role |
|-------|-------|------|
| **ProteinMPNN** | `p(seq \| backbone)` | structure-conditioned likelihood (message-passing GNN over the backbone graph) |
| **LigandMPNN** | `p(seq \| backbone, ligand)` | like ProteinMPNN but also conditions on non-protein atoms (ligands, metals, nucleic acids) |
| **SolubleMPNN** | `p(seq \| backbone)` | ProteinMPNN retrained on soluble proteins (biases away from hydrophobic surfaces) |
| **ESM-IF1** | `p(seq \| backbone)` | an independent inverse-folding model (GVP-transformer); useful as an alternative or ensemble likelihood |
| **ProtXLNet** | `p(seq)` | structure-free protein language model — the marginal in the denominator |

Auxiliary (not design terms): **PSSM** (position-specific scoring matrix, used by
the analysis commands) and **trRosetta** (structure predictor, used as an
evaluation oracle).

## Design modes (objectives)

Select with `--model_name`. Defined declaratively in `OBJECTIVES`
(`src/bayes_design/model.py`) — adding a mode is one row.

| mode | objective | use it to… |
|------|-----------|------------|
| `bayes_design` | ProteinMPNN / ProtXLNet | design for stability & conformational specificity (**default**) |
| `bayes_design_ligand` | LigandMPNN / ProtXLNet | design the fold while accounting for a bound ligand |
| `bayes_design_ligand_specificity` | LigandMPNN(+lig) / LigandMPNN(−lig) | optimize *specificity* for the ligand (the effect the ligand adds) |
| `bayes_design_soluble` | SolubleMPNN / ProtXLNet | steer a design toward solubility |
| `bayes_design_esm_if1` | ESM-IF1 / ProtXLNet | use ESM-IF1 as the likelihood instead of ProteinMPNN |
| `bayes_design_ensemble` | (ProteinMPNN · ESM-IF1) / ProtXLNet | combine two inverse-folding models (product-of-experts) for robustness |
| `fold_specificity` | ProteinMPNN(main) / ProteinMPNN(decoy) | negative design: prefer the target fold over a decoy (`--decoy_id`) |
| `ligand_selectivity` | LigandMPNN(main) / LigandMPNN(ligand B) | design to bind the target ligand, not an off-target (`--ligand_file_b`) |

Notes:
- **Ligand modes** read ligand atoms from the target PDB's HETATM records by
  default; override with `--ligand_file <pdb>`. They need the LigandMPNN weights
  (see Installation).
- **ESM-IF1 modes** need `pip install -e .[esm]`. ESM-IF1 decodes autoregressively
  in sequence order, so run them with `--decode_order n_to_c`.
- **Multi-state modes** (`fold_specificity`, `ligand_selectivity`) score the same
  designed residues against two structural contexts; the contexts must describe
  the same chain (e.g. alternative conformations, or the same protein with
  different ligands).

Example — design a binding pocket for the ligand in a bound structure:

```
bayes-design --model_name bayes_design_ligand --protein_id <ligand-bound PDB> \
    --decode_order n_to_c --decode_algorithm beam
```

## Evaluating designs

`bayes-design-experiment` scores and filters designed sequences (log-probability
under a model, PSSMs, histograms). For example:

```
bayes-design-experiment compare_seq_metric --metric log_prob --model_name bayes_design \
    --protein_id 1PIN --fixed_positions 34 34 --sequences MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG
```

See [`experiments.md`](experiments.md) for the full set of analysis recipes.

## Performance

On a V100 GPU, greedy decoding predicts ~10 residues/s; beam search with 128
beams predicts ~1 residue every 2s.

## Docker

```
git clone https://github.com/dellacortelab/bayes_design.git
docker build -t bayes_design -f ./bayes_design/dependencies/Dockerfile ./bayes_design
docker run -dit --gpus all --name bayes_dev --rm \
    -v $(pwd)/bayes_design:/code -v $(pwd)/bayes_design/data:/data bayes_design
docker exec -it bayes_dev /bin/bash
```

## Development

```
ruff format src tests && ruff check src tests   # lint + format
mypy src/bayes_design                           # type-check
pytest                                          # tests (set BAYES_RUN_SLOW=1 for the model regression)
```

Vendored third-party model code (`protein_mpnn/`, `ligand_mpnn/`) is excluded
from linting and type-checking.

## Citation

```bibtex
@Article{Stern2023,
  author  = {Stern, Jacob A. and Free, Tyler J. and Stern, Kimberlee L. and Gardiner, Spencer and Dalley, Nicholas A. and Bundy, Bradley C. and Price, Joshua L. and Wingate, David and Della Corte, Dennis},
  title   = {A probabilistic view of protein stability, conformational specificity, and design},
  journal = {Scientific Reports},
  year    = {2023},
  volume  = {13},
  number  = {1},
  pages   = {15493},
  doi     = {10.1038/s41598-023-42032-1},
}
```

## License

See [LICENSE](LICENSE).
