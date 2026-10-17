"""Experiment and analysis entry points.

No eager re-exports: the submodules have divergent dependencies (``analysis``
needs matplotlib, ``scrmsd_experiment`` needs the full experiments extra), while
``scrmsd`` is a lightweight, core-only toolkit. Import submodules directly, e.g.
``from bayes_design.experiments.scrmsd import calculate_rmsd`` or
``from bayes_design.experiments.analysis import viz_probs``.
"""
