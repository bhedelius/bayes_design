"""
Evaluation for Seq Design Models that maximize p(struct=S|seq) - i.e. sequence
design models made to design a sequence that matches a backbone.
"""

import math

import numpy as np
import torch

from bayes_design.utils import AMINO_ACID_ORDER


def evaluate_log_prob(seq, prob_model, decode_order, fixed_position_mask, mask_type, structure=None, exclude_aa=None):
    """
    Evaluate the log probability of the structure for the sequence under a
    sequence to structure model. This measures p(struct|seq) for a designed sequence.
    We expect p(seq|struct)/p(seq) to be better than p(seq|struct) because
    p(seq|struct)/p(seq) seeks to model p(struct|seq). This may be a flawed metric
    because the seq-to-struct network may only key in on a few important residues
    for global structure, possibly failing to correctly model the probabilities of
    residues that are less important for folding.

    You should call evaluate_log_prob with a sequence designed with p(seq|struct)
    and with a sequence designed with p(seq|struct)/p(seq) and compare the log
    probabilities.

    If fixed positions are provided, they are excluded from the log probability calculation.

    Args:
        sequence (str): a string (no spaces) representing an amino acid sequence
        seq_to_struct_model (nn.Module): a network taking as input a
    """
    if exclude_aa is None:
        exclude_aa = []
    n_fixed_positions = int(fixed_position_mask.sum())
    n_predicted_positions = len(seq) - n_fixed_positions

    # Decode order defines the order in which the masked positions are predicted
    token_to_decode = torch.tensor(decode_order[n_fixed_positions:])
    probs = prob_model(
        seq=[seq] * n_predicted_positions,
        struct=structure,
        decode_order=decode_order,
        token_to_decode=token_to_decode,
        mask_type=mask_type,
    )
    for aa in exclude_aa:
        probs[:, AMINO_ACID_ORDER.index(aa)] = 0
    probs = probs / probs.sum(dim=1, keepdim=True)

    log_prob = 0
    for i, tok in enumerate(token_to_decode):
        aa = seq[tok]
        seq_idx = AMINO_ACID_ORDER.index(aa)
        log_prob += math.log(probs[i, seq_idx])

    return log_prob


def evaluate_sequence_log_prob(
    seq, prob_model, structure=None, decode_order=None, mask_type="bidirectional_mlm", exclude_aa=None
):
    """Teacher-forced summed log-likelihood of a full sequence under one model.

    Scores every position (no fixed positions) against the given sequence. Call it
    with a single backend (not a CombinedModel) to get an un-normalized per-term
    log-likelihood -- e.g. log p(seq|struct) from ProteinMPNN or log p(seq) from a
    marginal -- for calibration against measured stabilities.
    """
    if decode_order is None:
        decode_order = list(range(len(seq)))
    fixed_position_mask = np.zeros(len(seq))
    return evaluate_log_prob(
        seq=seq,
        prob_model=prob_model,
        decode_order=decode_order,
        fixed_position_mask=fixed_position_mask,
        mask_type=mask_type,
        structure=structure,
        exclude_aa=exclude_aa,
    )


def predict_delta_g(log_p_struct_given_seq, log_p_seq, coef):
    """Apply a fitted linear calibration ``dG = a*log p(seq|struct) + b*log p(seq) + c``.

    ``coef`` is ``(a, b, c)`` as fit by the calibration script. In the review's terms
    the denominator weight is ``lambda = -b/a`` and the inverse temperature ``beta = 1/a``.
    """
    a, b, c = coef
    return a * log_p_struct_given_seq + b * log_p_seq + c


def evaluate_rmsd(sequence, seq_to_struct_model, targ_struct):
    """
    Evaluate RMSD between the "true structure" (computationally predicted from
    the true sequence) and the predicted structure for a designed sequence. This
    is a good metric for evaluating p(struct|seq), but has a performance ceiling.
    You can only match the RMSD. Ideally, we want to design a sequence with RMSD
    as close to 0 as possible, but with higher stability than the original sequence
    (measured by evaluate_log_prob).
    """


def evaluate_perplexity(
    seq, prob_model, decode_order, structure=None, fixed_position_mask=None, mask_type=None, exclude_aa=None
):
    """
    Evaluate the perplexity of the sequence under the model. If fixed positions are provided, they
    are excluded from the perplexity calculation.
    """
    if exclude_aa is None:
        exclude_aa = []
    log_prob = evaluate_log_prob(
        seq=seq,
        prob_model=prob_model,
        decode_order=decode_order,
        structure=structure,
        fixed_position_mask=fixed_position_mask,
        mask_type=mask_type,
        exclude_aa=exclude_aa,
    )
    n_fixed_positions = fixed_position_mask.sum()
    n = len(seq) - n_fixed_positions
    perplexity = np.exp(-1 / n * log_prob)
    return perplexity


def evaluate_recapitulation(pred_seq, orig_seq):
    """
    Evaluate % recapitulation of the original sequence. This is a bad metric for
    measuring p(struct|seq) because it really measures p(seq|struct).
    """
    matches = [p == o for p, o in zip(pred_seq, orig_seq)]
    return sum(matches) / len(matches)


def evaluate_stability():
    """
    This is the best metric, as it measures p(struct|seq), without being subject
    to the approximations of using a computational model for p(struct|seq). But it
    must be based on experimental results.
    """
    pass


metric_dict = {"log_prob": evaluate_log_prob, "perplexity": evaluate_perplexity}
