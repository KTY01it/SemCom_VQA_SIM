from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from src.methods.nst_features import build_nst_v2_features


class QuestionConditionedNST(nn.Module):
    """
    NST-v2: question-conditioned semantic field mask predictor.

    Input:
      subject_id, relation_id, object_id
      dense question/triplet/rank features

    Output:
      logits for keep_subject, keep_relation, keep_object
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

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, subject_id, relation_id, object_id, features):
        s = self.object_emb(subject_id)
        r = self.relation_emb(relation_id)
        o = self.object_emb(object_id)
        f = self.feature_proj(features)

        x = torch.cat([s, r, o, f], dim=-1)
        return self.mlp(x)


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


def load_nst_v2_model(path: str) -> QuestionConditionedNST:
    ckpt = torch.load(path, map_location="cpu")

    model = QuestionConditionedNST(
        object_vocab_size=int(ckpt["object_vocab_size"]),
        relation_vocab_size=int(ckpt["relation_vocab_size"]),
        feature_dim=int(ckpt["feature_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_keep_mask_v2(
    model: QuestionConditionedNST,
    triplet: Dict[str, Any],
    question: str | None,
    keywords: List[str] | None,
    question_type: str | None,
    rank_index: int,
    n_top: int,
    threshold: float = 0.5,
) -> Tuple[Dict[str, bool], List[float]]:
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

    logits = model(s, r, o, feat_tensor)
    probs = torch.sigmoid(logits)[0].cpu().tolist()

    keep = {
        "subject": bool(probs[0] >= threshold),
        "relation": bool(probs[1] >= threshold),
        "object": bool(probs[2] >= threshold),
    }

    if not any(keep.values()):
        keep["object"] = True

    return keep, probs
