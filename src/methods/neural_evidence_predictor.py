from __future__ import annotations

import torch
import torch.nn as nn


class NeuralEvidencePredictor(nn.Module):
    """
    NSP-v6 RAMS evidence predictor.

    Inputs:
      - subject/relation/object ids
      - dense question-triplet features
      - question embedding
      - triplet embedding
      - interaction features: |q-t|, q*t

    Outputs:
      - evidence_logit: probability that this triplet is answer/evidence-bearing
      - mask_logits: probability of keeping subject/relation/object fields
    """

    def __init__(
        self,
        object_vocab_size: int,
        relation_vocab_size: int,
        dense_dim: int,
        text_dim: int,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.object_vocab_size = int(object_vocab_size)
        self.relation_vocab_size = int(relation_vocab_size)
        self.dense_dim = int(dense_dim)
        self.text_dim = int(text_dim)

        self.object_emb = nn.Embedding(self.object_vocab_size, emb_dim, padding_idx=0)
        self.relation_emb = nn.Embedding(self.relation_vocab_size, emb_dim, padding_idx=0)

        id_dim = emb_dim * 3
        text_interaction_dim = text_dim * 4
        in_dim = id_dim + dense_dim + text_interaction_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.evidence_head = nn.Linear(hidden_dim, 1)
        self.mask_head = nn.Linear(hidden_dim, 3)

    def forward(self, subject_id, relation_id, object_id, dense_features, q_emb, t_emb):
        s = self.object_emb(subject_id)
        r = self.relation_emb(relation_id)
        o = self.object_emb(object_id)

        text_feat = torch.cat(
            [
                q_emb,
                t_emb,
                torch.abs(q_emb - t_emb),
                q_emb * t_emb,
            ],
            dim=-1,
        )

        x = torch.cat([s, r, o, dense_features, text_feat], dim=-1)
        h = self.backbone(x)

        evidence_logit = self.evidence_head(h).squeeze(-1)
        mask_logits = self.mask_head(h)

        return evidence_logit, mask_logits


def safe_shifted_id(x, vocab_size: int) -> int:
    try:
        out = int(x) + 1
    except Exception:
        out = 0

    if out < 0:
        out = 0
    if out >= vocab_size:
        out = vocab_size - 1

    return out


def load_evidence_predictor(path: str) -> NeuralEvidencePredictor:
    ckpt = torch.load(path, map_location="cpu")

    model = NeuralEvidencePredictor(
        object_vocab_size=int(ckpt["object_vocab_size"]),
        relation_vocab_size=int(ckpt["relation_vocab_size"]),
        dense_dim=int(ckpt["dense_dim"]),
        text_dim=int(ckpt["text_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model
