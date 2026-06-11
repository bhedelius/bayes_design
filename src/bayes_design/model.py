import os
import pickle as pkl
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tr_rosetta_pytorch import trRosettaNetwork
from tr_rosetta_pytorch.cli import DEFAULT_MODEL_PATH
from tr_rosetta_pytorch.utils import preprocess
from transformers import XLNetLMHeadModel, XLNetTokenizer

from .protein_mpnn.protein_mpnn_utils import ProteinMPNN
from .utils import AMINO_ACID_ORDER


class ProbabilityModel(nn.Module, ABC):
    """Shared interface so any model is interchangeable as a term in a
    `CombinedModel`. Structure/ligand context is bound at construction, keeping
    ``forward(seq, struct, decode_order, token_to_decode, ...) -> (N x 20)``
    uniform across models (the decode loop relies on this)."""

    @abstractmethod
    def forward(
        self, seq, struct, decode_order, token_to_decode, mask_type="bidirectional_autoregressive", temperature=1.0
    ):
        raise NotImplementedError


class XLNetWrapper(ProbabilityModel):
    def __init__(self, model_name="Rostlab/prot_xlnet", device=None):
        super().__init__()
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model = XLNetLMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)

        xlnet_vocab_dict = self.tokenizer.get_vocab()
        xlnet_vocab_dict["▁X"] = xlnet_vocab_dict["X"]
        self.canonical_idx_to_xlnet_idx = torch.tensor([xlnet_vocab_dict["▁" + aa] for aa in AMINO_ACID_ORDER])

    def forward(
        self, seq, decode_order, token_to_decode, struct=None, mask_type="bidirectional_autoregressive", temperature=1.0
    ):
        """Accept an amino acid sequence, return class probabilities for the next token
        Args:
            seq (len N list of len L_seq str): a string representation of an amino
                acid sequence with unknown residues indicated with a dash (-)
            decode_order (len L list): list of the order of indices to decode.
                This determines the values in the permutation mask. Each index
                attends to all indices that occur previously in the decoding_order.
            token_to_decode (len N tensor): index in the range [0, L-1] indicating which
                token to predict for each item in the batch.
        Returns:
            probs ((20) torch.Tensor): a vector of probabilities for the next
                token
        """
        # Replace rare amino acids with "X"
        seq = [re.sub(r"[UZOB]", "X", s) for s in seq]
        # Huggingface XLNet expects a space-separated sequence
        seq = [" ".join(s) for s in seq]
        # Mask '-' tokens
        seq = [re.sub(r"-", "<mask>", s) for s in seq]
        input_ids = self.tokenizer(seq, return_tensors="pt", add_special_tokens=False)["input_ids"]
        seq_len = input_ids.shape[-1]

        n_tokens = len(token_to_decode)

        # perm_mask should be the mask for query stream attention (don't allow items to see self), because the xlnet
        # implementation subtracts an identity matrix from perm_mask to get the content stream attention, where items are
        # masked from seeing self. The target_mapping ensures that we use query stream attention when predicting
        # the target elements https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/xlnet/modeling_xlnet.py#L1172
        # In query stream attention you should always be masked from yourself because tokens are always masked from themselves during training
        perm_mask = torch.ones(
            (n_tokens, seq_len, seq_len), dtype=torch.float
        )  # perm_mask[0, j, k] = 1 means that the jth token cannot see the kth token
        if mask_type == "unidirectional_autoregressive":  # Allow each token to see tokens preceding it in decode order
            for i, tok in enumerate(token_to_decode):
                token_to_decode_idx = decode_order.index(tok)
                # +1 because we want to include the tokens that the token_to_decode can see
                for j, idx in enumerate(decode_order[: token_to_decode_idx + 1]):
                    perm_mask[i, idx, decode_order[:j]] = 0.0
            # decode_order: [2, 1, 0]
            # i = 0 -> decode pos 2
            #   j = 0, idx = 2
            #           [1, 1, 1]
            #           [1, 1, 1]
            #           [1, 1, 1]
            # i = 1 -> decode pos 1
            #   j = 0, idx = 2
            #   j = 1, idx = 1
            #       perm_mask[1, 1, [2]] = 0.
            #           [1, 1, 1]
            #           [1, 1, 0]
            #           [1, 1, 1]
            # i = 2 -> decode pos 0
            #   j = 0, idx = 2
            #   j = 1, idx = 1
            #       perm_mask[2, 1, [2]] = 0.
            #           [1, 1, 1]
            #           [1, 1, 0]
            #           [1, 1, 1]
            #   j = 2, idx = 0
            #       perm_mask[2, 0, [2, 1]] = 0.
            #           [1, 0, 0]
            #           [1, 1, 0]
            #           [1, 1, 1]
        elif mask_type == "bidirectional_autoregressive":
            for i, tok in enumerate(token_to_decode):
                token_to_decode_idx = decode_order.index(tok)
                # Iterate over all tokens up to and including tok
                for prev_tok_1 in decode_order[: token_to_decode_idx + 1]:
                    # Iterate over all tokens preceding tok
                    for prev_tok_2 in decode_order[:token_to_decode_idx]:
                        if prev_tok_1 == prev_tok_2:
                            # Never let a token see itself in query-stream attention
                            continue
                        # Allow all tokens up to and including tok to see tokens preceding tok
                        perm_mask[i, prev_tok_1, prev_tok_2] = 0.0

            #       [1, 1, 1, 1]
            #       [1, 1, 1, 1]
            #       [1, 1, 1, 1]
            #       [1, 1, 1, 1]
            #
            #       [1, 1, 1, 1]
            #       [1, 1, 1, 1]
            #       [1, 1, 1, 0]
            #       [1, 1, 1, 1]
            #
            #       [1, 1, 1, 1]
            #       [1, 1, 0, 0]
            #       [1, 1, 1, 0]
            #       [1, 1, 0, 1]
            #
            #       [1, 0, 0, 0]
            #       [1, 1, 0, 0]
            #       [1, 0, 1, 0]
            #       [1, 0, 0, 1]

        elif mask_type == "bidirectional_mlm":
            for i, tok in enumerate(token_to_decode):
                perm_mask[i, :, torch.arange(seq_len) != tok] = (
                    0.0  # Allow full bidirectional context (masked-language-model-style. this is not autoregressive)
                )
                # In query stream attention, tokens are always masked from themselves
                perm_mask[i, torch.arange(seq_len), torch.arange(seq_len)] = 1.0

                # [1, 0, 1]
                # [0, 1, 1]
                # [0, 0, 1]

                # [1, 1, 0]
                # [0, 1, 0]
                # [0, 1, 1]

                # [1, 0, 0]
                # [1, 1, 0]
                # [1, 0, 1]

        target_mapping = torch.zeros(
            (n_tokens, 1, seq_len), dtype=torch.float
        )  # Shape [batch_size=n_tokens, num_tokens_to_predict=1, seq_length=n_tokens]
        for i, tok in enumerate(token_to_decode):
            target_mapping[i, 0, tok] = 1.0  # Predict token tok at batch position i

        with torch.inference_mode():
            out = self.model(
                input_ids.to(self.device),
                perm_mask=perm_mask.to(self.device),
                target_mapping=target_mapping.to(self.device),
            )
            # logits has shape [batch_size, 1, config.vocab_size] (1 is num_tokens_to_predict)
            index_corrected_logits = out.logits[:, 0, self.canonical_idx_to_xlnet_idx]
            # Ignore the last entry, corresponding to 'X'
            index_corrected_logits = index_corrected_logits[:, :-1]
            # Get temperature-adjusted probabilities
            probs = torch.nn.functional.softmax(index_corrected_logits / temperature, dim=-1)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs


class ProteinMPNNWrapper(ProbabilityModel):
    def __init__(self, device=None):
        super().__init__()
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        backbone_noise = 0.00  # Std of Gaussian noise added to backbone atoms
        # v_48_030 = version with 48 edges, 0.30A noise (package-relative path)
        checkpoint_path = Path(__file__).parent / "protein_mpnn" / "vanilla_model_weights" / "v_48_030.pt"
        hidden_dim = 128
        num_layers = 3
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print("Number of edges:", checkpoint["num_edges"])
        noise_level_print = checkpoint["noise_level"]
        print(f"Training noise level: {noise_level_print}")
        self.model = ProteinMPNN(
            num_letters=21,
            node_features=hidden_dim,
            edge_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            augment_eps=backbone_noise,
            k_neighbors=checkpoint["num_edges"],
        )
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print("Model loaded")

    def forward(self, seq, struct, decode_order, token_to_decode, mask_type, temperature=1.0):
        """Accept an amino acid sequence and protein structure
        coordinates, return class probabilities for the next token
        Args:
            seq (len N list of len L_seq str): a list of string representations of an amino
                acid sequence.
            struct ((L x 4 x 3) torch.Tensor): batch_size x seq_length x
                num_atoms x num_coordinates tensor
            decode_order (len L list): list of the order of indices to decode.
                This determines the values in the permutation mask. Each index
                attends to all indices that occur previously in the decoding_order.
            token_to_decode (len N tensor): tensor of indices in the range [0, L-1] indicating which
                token to decode next.
        Returns:
            probs ((N x 20) torch.Tensor): a vector of probabilities for the next
                token
        """

        N = len(token_to_decode)
        L = len(seq[0])
        assert L == struct.shape[0], "Sequence length must match the number of residues in the provided structure"

        # Convert amino acid character to index
        seq = [re.sub(r"-", "X", s) for s in seq]
        seq = torch.tensor([[AMINO_ACID_ORDER.index(aa) for aa in s] for s in seq]).to(self.device)
        with torch.no_grad():
            if mask_type != "bidirectional_mlm":
                decode_order = torch.tensor(decode_order).expand(N, L)
            elif mask_type == "bidirectional_mlm":
                decode_order = torch.tensor(
                    np.array(
                        [
                            np.append(np.delete(decode_order, decode_order.index(tok)).tolist(), tok)
                            for tok in token_to_decode
                        ]
                    )
                )

            struct = struct.expand(N, *struct.shape).to(self.device)
            # Default values
            # `mask` masks positions that are missing structural information
            mask = torch.isfinite(torch.sum(struct, (2, 3))).to(torch.float32)
            isnan = torch.isnan(struct)
            struct[isnan] = 0.0
            chain_M = torch.ones(N, L).float().to(self.device)
            chain_encoding_all = torch.ones(N, L).float().to(self.device)
            residue_idx = torch.arange(L).expand(N, L).to(self.device)

            _, logits = self.model(
                X=struct,
                S=seq,
                mask=mask,
                chain_M=chain_M,
                residue_idx=residue_idx,
                chain_encoding_all=chain_encoding_all,
                use_input_decoding_order=True,
                randn=None,
                decoding_order=decode_order.to(self.device),
            )
            # Get temperature-adjusted probabilities
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            # N x L x 20
            # Ignore last entry, corresponding to 'X'
            probs = probs[:, :, :-1]
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Note that unlike XLNet, ProteinMPNN gives probabilities for all residues, not just the one to decode.
        # So here we extract the probabilities for the residue to decode.
        return probs[range(len(token_to_decode)), token_to_decode]
        # N x 20


class MPNNWrapper(ProbabilityModel):
    """Structure-conditioned model backed by the vendored LigandMPNN code.

    A single class covers the LigandMPNN ``model_type`` variants we use:
    ``ligand_mpnn`` (p(seq | struct, ligand)), ``soluble_mpnn``, and a
    context-bound ``protein_mpnn``. Its structural context (the parsed input dict
    from `bayes_design.utils.get_ligand`) is bound at construction, so the same
    model can be used against different structures/ligands as different terms of a
    `CombinedModel` (multi-state / negative design). ``use_ligand=False`` zeroes
    the ligand-atom mask via ``featurize(use_atom_context=False)`` — the ligand-free
    denominator for ligand-specificity.

    Decode order is honored by crafting ``randn`` so LigandMPNN's internal
    ``argsort((chain_mask + 1e-4) * |randn|)`` reproduces it, and every feature is
    pre-batched to B=N (with batch_size=1) so the model's internal repeats no-op.
    """

    def __init__(self, device=None, model_type="ligand_mpnn", checkpoint_path=None, context=None, use_ligand=True):
        super().__init__()
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        from .ligand_mpnn import data_utils as ligand_data_utils
        from .ligand_mpnn.model_utils import ProteinMPNN as LigandMPNN

        self._featurize = ligand_data_utils.featurize
        self.model_type = model_type
        self.context = context
        self.use_ligand = use_ligand
        self._features = None

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Ligand checkpoints store atom_context_num; protein/soluble ones do not.
        self.atom_context_num = checkpoint.get("atom_context_num", 1)
        self.model = LigandMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=checkpoint["num_edges"],
            atom_context_num=self.atom_context_num,
            model_type=model_type,
            device=self.device,
            ligand_mpnn_use_side_chain_context=False,
        )
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"{model_type} loaded (use_ligand={use_ligand})")

    def _static_features(self):
        """Featurize the bound structure/ligand once and cache it (it is constant
        for the run; only S/chain_mask/randn vary per forward call)."""
        if self._features is None:
            if self.context is None:
                raise ValueError("MPNNWrapper requires a parsed context; see bayes_design.utils.get_ligand().")
            self._features = self._featurize(
                self.context,
                number_of_ligand_atoms=self.atom_context_num,
                use_atom_context=self.use_ligand,
                model_type=self.model_type,
            )
        return self._features

    def forward(
        self, seq, struct, decode_order, token_to_decode, mask_type="bidirectional_autoregressive", temperature=1.0
    ):
        N = len(token_to_decode)
        L = len(seq[0])

        seq = [re.sub(r"-", "X", s) for s in seq]
        seq = torch.tensor([[AMINO_ACID_ORDER.index(aa) for aa in s] for s in seq]).to(self.device)

        with torch.no_grad():
            if mask_type != "bidirectional_mlm":
                decode_orders = torch.tensor(decode_order).expand(N, L).clone()
            else:
                decode_orders = torch.tensor(
                    np.array(
                        [
                            np.append(np.delete(decode_order, decode_order.index(tok)).tolist(), tok)
                            for tok in token_to_decode
                        ]
                    )
                )
            decode_orders = decode_orders.to(self.device)

            base = self._static_features()
            assert base["S"].shape[1] == L, "Sequence length must match the number of residues in the context"

            def _batch(t):
                return t.expand(N, *t.shape[1:]).clone().to(self.device)

            feature_dict = {
                k: (_batch(v) if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == 1 else v)
                for k, v in base.items()
            }
            feature_dict["S"] = seq
            feature_dict["chain_mask"] = torch.ones(N, L, device=self.device)
            feature_dict["batch_size"] = 1
            feature_dict["symmetry_residues"] = [[]]

            # Craft randn so argsort(|randn|) == decode_order for each row.
            randn = torch.zeros(N, L, device=self.device)
            ranks = torch.arange(1, L + 1, device=self.device, dtype=randn.dtype)
            randn.scatter_(1, decode_orders.long(), ranks.expand(N, L))
            feature_dict["randn"] = randn

            logits = self.model.score(feature_dict, use_sequence=True)["logits"]  # N x L x 21
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
            probs = probs[:, :, :-1]  # drop 'X'
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs[range(N), token_to_decode]


class ESMIF1Wrapper(ProbabilityModel):
    """ESM-IF1 inverse folding likelihood, p(seq | struct).

    Uses the call-time backbone coordinates (N, CA, C from ``struct[:, :3]``); no
    context is bound. ESM-IF1's decoder is autoregressive in sequence index, so a
    single teacher-forced forward gives, at each position, p(residue_i | struct,
    residues_{<i}). This matches the default ``n_to_c`` decode order (each position
    conditions on its index-predecessors); other decode orders are approximated.
    Requires the optional ``esm`` extra (``pip install -e .[esm]``).
    """

    def __init__(self, device=None):
        super().__init__()
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        import esm
        from esm.inverse_folding.util import CoordBatchConverter

        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.model = self.model.to(self.device).eval()
        self.batch_converter = CoordBatchConverter(self.alphabet)
        # Map our 20 canonical amino acids (excluding 'X') to ESM-IF1 logit columns.
        self.canonical_idx_to_esmif_idx = torch.tensor(
            [self.alphabet.get_idx(aa) for aa in AMINO_ACID_ORDER[:-1]], device=self.device
        )
        print("ESM-IF1 loaded")

    def forward(
        self, seq, struct, decode_order, token_to_decode, mask_type="bidirectional_autoregressive", temperature=1.0
    ):
        N = len(token_to_decode)
        coords = struct[:, :3, :].detach().cpu().numpy()  # L x 3 x 3 (N, CA, C)
        seqs = [re.sub(r"-", "X", s) for s in seq]
        batch = [(coords, None, s) for s in seqs]

        with torch.no_grad():
            coords_b, confidence, _, tokens, padding_mask = self.batch_converter(batch, device=self.device)
            prev_output_tokens = tokens[:, :-1]
            logits, _ = self.model.forward(coords_b, padding_mask, confidence, prev_output_tokens)
            # logits: N x alphabet x (L+1); residue position idx -> logits[:, :, idx]
            tok = token_to_decode.to(self.device)
            sel = logits[torch.arange(N, device=self.device), :, tok]  # N x alphabet
            sel = sel[:, self.canonical_idx_to_esmif_idx]  # N x 20
            probs = torch.nn.functional.softmax(sel / temperature, dim=-1)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs


class CombinedModel(ProbabilityModel):
    """Weighted log-linear combination of probability models, renormalized.

    ``p ∝ exp(Σ_i w_i · log(p_i + balance))`` over the 20 amino acids. Positive
    weights are "numerator" terms, negative weights "denominator" terms. The
    two-term ``[(num, +1), (den, -1)]`` case reproduces the original BayesDesign
    ratio ``(p_num + b) / (p_den + b)``; more terms express product-of-experts and
    multi-state / negative design (see `OBJECTIVES`).
    """

    def __init__(self, terms, device=None, balance=0.002):
        super().__init__()
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.terms = terms  # list of (ProbabilityModel, float weight)
        self.balance = balance

    def forward(
        self, seq, struct, decode_order, token_to_decode, mask_type="bidirectional_autoregressive", temperature=1.0
    ):
        log_score = None
        for model, weight in self.terms:
            probs = model(
                seq=seq,
                struct=struct,
                decode_order=decode_order,
                token_to_decode=token_to_decode,
                mask_type=mask_type,
                temperature=temperature,
            )
            term = weight * torch.log(probs + self.balance)
            log_score = term if log_score is None else log_score + term

        log_score = log_score - log_score.max(dim=-1, keepdim=True).values
        score = torch.exp(log_score)
        return score / score.sum(dim=-1, keepdim=True)


class BayesDesign(CombinedModel):
    """Backward-compatible 2-term objective: p(seq|struct) / p(seq).

    Defaults to ProteinMPNN / XLNet, matching the original BayesDesign, but accepts
    any numerator/denominator `ProbabilityModel`s.
    """

    def __init__(self, numerator=None, denominator=None, device=None, bayes_balance_factor=0.002, **kwargs):
        num = numerator if numerator is not None else ProteinMPNNWrapper(device=device)
        den = denominator if denominator is not None else XLNetWrapper(device=device)
        super().__init__(terms=[(num, 1.0), (den, -1.0)], device=device, balance=bayes_balance_factor)


class TrRosettaWrapper:
    def __init__(self, data_location="./data/msa", database="./data/uniref30/UniRef30_2022_02", device=None):

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        model_files = [*Path(DEFAULT_MODEL_PATH).glob("*.pt")]
        self.models = []
        for model_file in model_files[:1]:
            trRosetta = trRosettaNetwork(filters=64, kernel=3, num_layers=61)
            trRosetta = trRosetta.to(self.device)
            trRosetta.load_state_dict(torch.load(model_file, map_location=self.device))
            trRosetta.eval()
            self.models.append(trRosetta)

        self.data_location = data_location
        os.makedirs(self.data_location, exist_ok=True)

        self.database = database

    def __call__(self, seq, seq_id=""):
        """Pass a sequence through the trRosetta model and return the distogram
        Args:
            seq (str): a space-separated string representing the amino acid sequence
        Returns:
            distance ((L x L x 37) torch.Tensor): a distogram representing distance bin probabilities
        """
        seq_path = os.path.join(self.data_location, f"{seq_id}_{seq}.txt")
        seq_msa_path = os.path.join(self.data_location, f"msa_{seq_id}_{seq}.txt")
        # Make a fasta file with the sequence
        with open(seq_path, "w") as f:
            f.write(">\n" + "".join(seq.split()))
        # Get an MSA for the sequence
        if not os.path.exists(seq_msa_path):
            subprocess.run(
                [
                    "/root/hh-suite/bin/hhblits",
                    "-v",
                    "0",
                    "-i",
                    f"{seq_path}",
                    "-oa3m",
                    f"{seq_msa_path}",
                    "-d",
                    f"{self.database}",
                ]
            )
        x = preprocess(seq_msa_path).to(self.device)
        outputs = []
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                outputs.append(output)
            averaged_outputs = [
                torch.stack(model_output).mean(dim=0).cpu().numpy().squeeze(0).transpose(1, 2, 0)
                for model_output in zip(*outputs)
            ]
            # prob_theta, prob_phi, prob_distance, prob_omega
            output_dict = dict(zip(["theta", "phi", "dist", "omega"], averaged_outputs))
            distance = output_dict["dist"]
            distance = distance.squeeze()

        return distance


class PSSM:
    def __init__(self, pssm_path):
        # load pssm from pickle
        with open(pssm_path, "rb") as f:
            self.pssm = pkl.load(f)

    def __call__(self, seq, struct, decode_order, token_to_decode, mask_type):
        return self.pssm[token_to_decode, :]


_LIGAND_DIR = Path(__file__).parent / "ligand_mpnn" / "model_params"


def _backend_xlnet(device, context=None, **kwargs):
    return XLNetWrapper(device=device)


def _backend_protein_mpnn(device, context=None, **kwargs):
    # Legacy vanilla ProteinMPNN — uses the call-time struct, not a bound context.
    return ProteinMPNNWrapper(device=device)


def _backend_protein_mpnn_ctx(device, context=None, **kwargs):
    # Context-bound ProteinMPNN (via LigandMPNN code) for multi-state objectives.
    return MPNNWrapper(
        device=device,
        model_type="protein_mpnn",
        checkpoint_path=_LIGAND_DIR / "proteinmpnn_v_48_020.pt",
        context=context,
        use_ligand=False,
    )


def _backend_ligand_mpnn(device, context=None, use_ligand=True, **kwargs):
    return MPNNWrapper(
        device=device,
        model_type="ligand_mpnn",
        checkpoint_path=_LIGAND_DIR / "ligandmpnn_v_32_010_25.pt",
        context=context,
        use_ligand=use_ligand,
    )


def _backend_soluble_mpnn(device, context=None, **kwargs):
    return MPNNWrapper(
        device=device,
        model_type="soluble_mpnn",
        checkpoint_path=_LIGAND_DIR / "solublempnn_v_48_020.pt",
        context=context,
        use_ligand=False,
    )


def _backend_esm_if1(device, context=None, **kwargs):
    # Uses the call-time struct (N, CA, C), not a bound context.
    return ESMIF1Wrapper(device=device)


# Each backend builder takes (device, context, **kwargs) and returns a ProbabilityModel.
BACKENDS = {
    "xlnet": _backend_xlnet,
    "protein_mpnn": _backend_protein_mpnn,
    "protein_mpnn_ctx": _backend_protein_mpnn_ctx,
    "ligand_mpnn": _backend_ligand_mpnn,
    "soluble_mpnn": _backend_soluble_mpnn,
    "esm_if1": _backend_esm_if1,
}

# An objective is a list of term specs {backend, weight, context?, **backend_kwargs}.
# Adding a design mode is one row here; adding a model is one BACKENDS entry.
# Invariant: the natural-sequence prior cancels only when the sum of the
# numerator (positive) weights equals the magnitude of the denominator (negative)
# weights. e.g. the ensemble below averages two likelihoods (0.5 + 0.5) against one
# marginal (-1); a product-of-experts variant would be +1, +1 against -2.
OBJECTIVES = {
    # p(struct|seq)        ∝ p(seq|struct)         / p(seq)
    "bayes_design": [
        {"backend": "protein_mpnn", "weight": 1.0},
        {"backend": "xlnet", "weight": -1.0},
    ],
    # p(struct,ligand|seq) ∝ p(seq|struct,ligand)  / p(seq)
    "bayes_design_ligand": [
        {"backend": "ligand_mpnn", "weight": 1.0, "context": "main", "use_ligand": True},
        {"backend": "xlnet", "weight": -1.0},
    ],
    # p(ligand|seq,struct) ∝ p(seq|struct,ligand)  / p(seq|struct)   [same LigandMPNN, ligand masked]
    "bayes_design_ligand_specificity": [
        {"backend": "ligand_mpnn", "weight": 1.0, "context": "main", "use_ligand": True},
        {"backend": "ligand_mpnn", "weight": -1.0, "context": "main", "use_ligand": False},
    ],
    # Solubility-steered fold design (SolubleMPNN as the likelihood).
    "bayes_design_soluble": [
        {"backend": "soluble_mpnn", "weight": 1.0, "context": "main"},
        {"backend": "xlnet", "weight": -1.0},
    ],
    # Multi-state / negative design: prefer fold A (main) over decoy fold B.
    "fold_specificity": [
        {"backend": "protein_mpnn_ctx", "weight": 1.0, "context": "main"},
        {"backend": "protein_mpnn_ctx", "weight": -1.0, "context": "decoy"},
    ],
    # Ligand selectivity: bind the main ligand, not ligand B.
    "ligand_selectivity": [
        {"backend": "ligand_mpnn", "weight": 1.0, "context": "main", "use_ligand": True},
        {"backend": "ligand_mpnn", "weight": -1.0, "context": "ligand_b", "use_ligand": True},
    ],
    # ESM-IF1 inverse-folding likelihood instead of ProteinMPNN.
    "bayes_design_esm_if1": [
        {"backend": "esm_if1", "weight": 1.0},
        {"backend": "xlnet", "weight": -1.0},
    ],
    # Product-of-experts likelihood: ProteinMPNN and ESM-IF1 together.
    "bayes_design_ensemble": [
        {"backend": "protein_mpnn", "weight": 0.5},
        {"backend": "esm_if1", "weight": 0.5},
        {"backend": "xlnet", "weight": -1.0},
    ],
}


def _build_term(spec, device, contexts):
    spec = dict(spec)
    backend = spec.pop("backend")
    weight = spec.pop("weight")
    context = contexts.get(spec.pop("context", None))
    model = BACKENDS[backend](device=device, context=context, **spec)
    return model, weight


def build_objective(name, device=None, contexts=None, bayes_balance_factor=0.002):
    contexts = contexts or {}
    terms = [_build_term(spec, device, contexts) for spec in OBJECTIVES[name]]
    return CombinedModel(terms, device=device, balance=bayes_balance_factor)


def build_model(name, device=None, contexts=None, bayes_balance_factor=0.002):
    """Single entry point used by the CLI: build any objective or backend by name."""
    if name in OBJECTIVES:
        return build_objective(name, device=device, contexts=contexts, bayes_balance_factor=bayes_balance_factor)
    if name in BACKENDS:
        return BACKENDS[name](device=device, context=(contexts or {}).get("main"))
    return model_dict[name](device=device)


def objective_uses_context(name):
    """Whether the named objective/backend conditions on a parsed structure context."""
    if name in OBJECTIVES:
        return any(spec.get("context") for spec in OBJECTIVES[name])
    return name in ("ligand_mpnn", "soluble_mpnn", "protein_mpnn_ctx")


model_dict = {
    "xlnet": XLNetWrapper,
    "protein_mpnn": ProteinMPNNWrapper,
    "ligand_mpnn": MPNNWrapper,
    "soluble_mpnn": MPNNWrapper,
    "esm_if1": ESMIF1Wrapper,
    "bayes_design": BayesDesign,
    "bayes_design_ligand": CombinedModel,
    "bayes_design_ligand_specificity": CombinedModel,
    "bayes_design_soluble": CombinedModel,
    "bayes_design_esm_if1": CombinedModel,
    "bayes_design_ensemble": CombinedModel,
    "fold_specificity": CombinedModel,
    "ligand_selectivity": CombinedModel,
    "pssm": PSSM,
    "trRosetta": TrRosettaWrapper,
}
