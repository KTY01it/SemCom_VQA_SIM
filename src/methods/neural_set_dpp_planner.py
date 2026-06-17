from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.nst_features import build_nst_v2_features


class SetDPPNeuralPlanner(nn.Module):
    """
    NSP-v4 SetDPP.

    Difference from NSP-v1/v2/v3:
      - encodes the whole candidate set with TransformerEncoder
      - predicts contextual quality score q_i
      - predicts diversity embedding z_i
      - predicts field mask logits
    """

    def __init__(
        self,
        object_vocab_size: int,
        relation_vocab_size: int,
        feature_dim: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        diversity_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.object_vocab_size = int(object_vocab_size)
        self.relation_vocab_size = int(relation_vocab_size)
        self.feature_dim = int(feature_dim)

        self.object_emb = nn.Embedding(self.object_vocab_size, emb_dim, padding_idx=0)
        self.relation_emb = nn.Embedding(self.relation_vocab_size, emb_dim, padding_idx=0)

        in_dim = emb_dim * 3 + feature_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.set_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.quality_head = nn.Linear(hidden_dim, 1)
        self.mask_head = nn.Linear(hidden_dim, 3)
        self.diversity_head = nn.Linear(hidden_dim, diversity_dim)

    def forward(self, subject_id, relation_id, object_id, features, key_padding_mask=None):
        """
        subject_id/relation_id/object_id: [B, M]
        features: [B, M, F]
        key_padding_mask: [B, M], True for padding
        """
        s = self.object_emb(subject_id)
        r = self.relation_emb(relation_id)
        o = self.object_emb(object_id)

        x = torch.cat([s, r, o, features], dim=-1)
        h = self.input_proj(x)

        h = self.set_encoder(h, src_key_padding_mask=key_padding_mask)

        quality_logits = self.quality_head(h).squeeze(-1)
        mask_logits = self.mask_head(h)
        diversity = F.normalize(self.diversity_head(h), dim=-1)

        return quality_logits, mask_logits, diversity


def safe_shifted_id(x: Any, vocab_size: int) -> int:
    try:
        out = int(x) + 1
    except Exception:
        out = 0

    if out < 0:
        out = 0
    if out >= vocab_size:
        out = vocab_size - 1

    return out


def load_setdpp_model(path: str) -> SetDPPNeuralPlanner:
    ckpt = torch.load(path, map_location="cpu")

    model = SetDPPNeuralPlanner(
        object_vocab_size=int(ckpt["object_vocab_size"]),
        relation_vocab_size=int(ckpt["relation_vocab_size"]),
        feature_dim=int(ckpt["feature_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def encode_candidate_set(
    model: SetDPPNeuralPlanner,
    triplets: List[Dict[str, Any]],
    question: str | None,
    keywords: Iterable[str] | None,
    question_type: str | None,
    max_candidates: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = min(len(triplets), max_candidates)

    subject_ids = []
    relation_ids = []
    object_ids = []
    feats = []

    for i, t in enumerate(triplets[:n]):
        subject_ids.append(safe_shifted_id(t.get("subject_id", 0), model.object_vocab_size))
        relation_ids.append(safe_shifted_id(t.get("relation_id", 0), model.relation_vocab_size))
        object_ids.append(safe_shifted_id(t.get("object_id", 0), model.object_vocab_size))

        f = build_nst_v2_features(
            triplet=t,
            question=question,
            keywords=keywords,
            question_type=question_type,
            rank_index=i,
            n_top=max(1, n),
        )
        feats.append(f)

    subject = torch.tensor([subject_ids], dtype=torch.long)
    relation = torch.tensor([relation_ids], dtype=torch.long)
    obj = torch.tensor([object_ids], dtype=torch.long)
    features = torch.tensor([feats], dtype=torch.float32)

    return subject, relation, obj, features


@torch.no_grad()
def predict_set(
    model: SetDPPNeuralPlanner,
    triplets: List[Dict[str, Any]],
    question: str | None,
    keywords: Iterable[str] | None,
    question_type: str | None,
    max_candidates: int,
):
    subject, relation, obj, features = encode_candidate_set(
        model=model,
        triplets=triplets,
        question=question,
        keywords=keywords,
        question_type=question_type,
        max_candidates=max_candidates,
    )

    quality_logits, mask_logits, diversity = model(subject, relation, obj, features)

    quality = quality_logits[0].cpu()
    mask_probs = torch.sigmoid(mask_logits[0]).cpu()
    diversity = diversity[0].cpu()

    return quality, mask_probs, diversity


def setdpp_select(
    quality: torch.Tensor,
    diversity: torch.Tensor,
    n_top: int,
    beta: float,
) -> List[int]:
    """
    Greedy learned MMR/DPP-like selection:
      score_i = quality_i - beta * max cosine(z_i, selected)
    """
    n = int(quality.shape[0])
    remaining = list(range(n))
    selected: List[int] = []

    while remaining and len(selected) < n_top:
        best_idx = remaining[0]
        best_score = -1e18

        for idx in remaining:
            if not selected:
                red = 0.0
            else:
                sims = []
                for j in selected:
                    sims.append(float(torch.dot(diversity[idx], diversity[j]).item()))
                red = max(sims) if sims else 0.0

            score = float(quality[idx].item()) - float(beta) * red

            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected
