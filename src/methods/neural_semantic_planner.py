from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from src.methods.nst_features import build_nst_v2_features


class NeuralSemanticPlanner(nn.Module):
    """
    NSP-v1: neural semantic planner.

    It predicts:
      - select_logit: whether a candidate triplet should be transmitted
      - mask_logits: keep_subject / keep_relation / keep_object

    This model does not depend on DBSS at inference time.
    """

    def __init__(
        self,
        object_vocab_size: int,
        relation_vocab_size: int,
        feature_dim: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.object_vocab_size = int(object_vocab_size)
        self.relation_vocab_size = int(relation_vocab_size)
        self.feature_dim = int(feature_dim)

        self.object_emb = nn.Embedding(self.object_vocab_size, emb_dim, padding_idx=0)
        self.relation_emb = nn.Embedding(self.relation_vocab_size, emb_dim, padding_idx=0)

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.backbone = nn.Sequential(
            nn.Linear(emb_dim * 3 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.select_head = nn.Linear(hidden_dim, 1)
        self.mask_head = nn.Linear(hidden_dim, 3)

    def forward(self, subject_id, relation_id, object_id, features):
        s = self.object_emb(subject_id)
        r = self.relation_emb(relation_id)
        o = self.object_emb(object_id)
        f = self.feature_proj(features)

        x = torch.cat([s, r, o, f], dim=-1)
        h = self.backbone(x)

        select_logit = self.select_head(h).squeeze(-1)
        mask_logits = self.mask_head(h)

        return select_logit, mask_logits


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


def load_nsp_model(path: str) -> NeuralSemanticPlanner:
    ckpt = torch.load(path, map_location="cpu")

    model = NeuralSemanticPlanner(
        object_vocab_size=int(ckpt["object_vocab_size"]),
        relation_vocab_size=int(ckpt["relation_vocab_size"]),
        feature_dim=int(ckpt["feature_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_candidate(
    model: NeuralSemanticPlanner,
    triplet: Dict[str, Any],
    question: str | None,
    keywords: Iterable[str] | None,
    question_type: str | None,
    rank_index: int,
    n_top: int,
    mask_threshold: float = 0.40,
) -> Tuple[float, Dict[str, bool], List[float]]:
    s = torch.tensor(
        [safe_shifted_id(triplet.get("subject_id", 0), model.object_vocab_size)],
        dtype=torch.long,
    )
    r = torch.tensor(
        [safe_shifted_id(triplet.get("relation_id", 0), model.relation_vocab_size)],
        dtype=torch.long,
    )
    o = torch.tensor(
        [safe_shifted_id(triplet.get("object_id", 0), model.object_vocab_size)],
        dtype=torch.long,
    )

    feats = build_nst_v2_features(
        triplet=triplet,
        question=question,
        keywords=keywords,
        question_type=question_type,
        rank_index=rank_index,
        n_top=n_top,
    )
    feat_tensor = torch.tensor([feats], dtype=torch.float32)

    select_logit, mask_logits = model(s, r, o, feat_tensor)

    select_prob = float(torch.sigmoid(select_logit)[0].cpu().item())
    mask_probs = torch.sigmoid(mask_logits)[0].cpu().tolist()

    keep = {
        "subject": bool(mask_probs[0] >= mask_threshold),
        "relation": bool(mask_probs[1] >= mask_threshold),
        "object": bool(mask_probs[2] >= mask_threshold),
    }

    if not any(keep.values()):
        keep["object"] = True

    return select_prob, keep, mask_probs
