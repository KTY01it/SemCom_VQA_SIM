from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class NeuralSemanticTokenizer(nn.Module):
    """
    DBSS-NST v1.

    Input:
      subject_id, relation_id, object_id

    Output:
      logits for keep_subject, keep_relation, keep_object
    """

    def __init__(
        self,
        num_objects: int,
        num_relations: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.object_emb = nn.Embedding(num_objects + 1, emb_dim, padding_idx=0)
        self.relation_emb = nn.Embedding(num_relations + 1, emb_dim, padding_idx=0)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, subject_id, relation_id, object_id):
        s = self.object_emb(subject_id)
        r = self.relation_emb(relation_id)
        o = self.object_emb(object_id)
        x = torch.cat([s, r, o], dim=-1)
        return self.mlp(x)


def safe_id(x: Any) -> int:
    try:
        return max(0, int(x))
    except Exception:
        return 0


def load_nst_model(path: str) -> NeuralSemanticTokenizer:
    ckpt = torch.load(path, map_location="cpu")

    model = NeuralSemanticTokenizer(
        num_objects=int(ckpt["num_objects"]),
        num_relations=int(ckpt["num_relations"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_keep_mask(
    model: NeuralSemanticTokenizer,
    triplet: Dict[str, Any],
    threshold: float = 0.5,
) -> Tuple[Dict[str, bool], list[float]]:
    s = torch.tensor([safe_id(triplet.get("subject_id", 0)) + 1], dtype=torch.long)
    r = torch.tensor([safe_id(triplet.get("relation_id", 0)) + 1], dtype=torch.long)
    o = torch.tensor([safe_id(triplet.get("object_id", 0)) + 1], dtype=torch.long)

    logits = model(s, r, o)
    probs = torch.sigmoid(logits)[0].cpu().tolist()

    keep = {
        "subject": bool(probs[0] >= threshold),
        "relation": bool(probs[1] >= threshold),
        "object": bool(probs[2] >= threshold),
    }

    if not any(keep.values()):
        keep["object"] = True

    return keep, probs
