from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class NeuralSemanticPolicyV7(nn.Module):
    """
    V7.1 Evidence-Aware Teacher-Distilled Set Policy.

    Pure-AI inference:
      candidate SG set -> teacher selection logits + evidence logits + field mask logits

    No DBSS, no Slot-Guard rule, no answer label at inference.
    """

    def __init__(
        self,
        object_vocab_size: int,
        relation_vocab_size: int,
        dense_dim: int,
        text_dim: int,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.object_vocab_size = int(object_vocab_size)
        self.relation_vocab_size = int(relation_vocab_size)
        self.dense_dim = int(dense_dim)
        self.text_dim = int(text_dim)
        self.hidden_dim = int(hidden_dim)

        self.object_emb = nn.Embedding(self.object_vocab_size, emb_dim, padding_idx=0)
        self.relation_emb = nn.Embedding(self.relation_vocab_size, emb_dim, padding_idx=0)

        input_dim = emb_dim * 3 + dense_dim + text_dim * 4

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.set_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.select_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        subject_id: torch.Tensor,
        relation_id: torch.Tensor,
        object_id: torch.Tensor,
        dense_features: torch.Tensor,
        q_emb: torch.Tensor,
        t_emb: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.object_emb(subject_id)
        r = self.relation_emb(relation_id)
        o = self.object_emb(object_id)

        q = q_emb[:, None, :].expand_as(t_emb)

        text_feat = torch.cat(
            [
                q,
                t_emb,
                torch.abs(q - t_emb),
                q * t_emb,
            ],
            dim=-1,
        )

        x = torch.cat([s, r, o, dense_features, text_feat], dim=-1)
        h = self.input_proj(x)

        padding_mask = ~candidate_mask.bool()
        h = self.set_encoder(h, src_key_padding_mask=padding_mask)

        select_logits = self.select_head(h).squeeze(-1)
        evidence_logits = self.evidence_head(h).squeeze(-1)
        mask_logits = self.mask_head(h)

        select_logits = select_logits.masked_fill(~candidate_mask.bool(), -1e9)
        evidence_logits = evidence_logits.masked_fill(~candidate_mask.bool(), -1e9)

        return select_logits, evidence_logits, mask_logits


def load_v7_policy(path: str) -> NeuralSemanticPolicyV7:
    ckpt = torch.load(path, map_location="cpu")

    model = NeuralSemanticPolicyV7(
        object_vocab_size=int(ckpt["object_vocab_size"]),
        relation_vocab_size=int(ckpt["relation_vocab_size"]),
        dense_dim=int(ckpt["dense_dim"]),
        text_dim=int(ckpt["text_dim"]),
        emb_dim=int(ckpt.get("emb_dim", 64)),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        num_layers=int(ckpt.get("num_layers", 3)),
        num_heads=int(ckpt.get("num_heads", 4)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model
