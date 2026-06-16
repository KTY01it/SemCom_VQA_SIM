import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.methods.neural_semantic_planner import NeuralSemanticPlanner
from src.methods.nst_features import nst_v2_feature_names


def shifted_ids(series, vocab_size: int):
    x = torch.tensor(series.clip(lower=0).values + 1, dtype=torch.long)
    x = torch.clamp(x, min=0, max=vocab_size - 1)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/nsp/nsp_train.csv")
    parser.add_argument("--out", default="results/nsp/nsp_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-loss-weight", type=float, default=0.50)
    parser.add_argument("--bit-lambda", type=float, default=0.05)
    parser.add_argument("--object-vocab-size", type=int, default=1118)
    parser.add_argument("--relation-vocab-size", type=int, default=208)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    feat_cols = nst_v2_feature_names()
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing NSP feature columns: {missing}")

    subject = shifted_ids(df["subject_id"], args.object_vocab_size)
    relation = shifted_ids(df["relation_id"], args.relation_vocab_size)
    obj = shifted_ids(df["object_id"], args.object_vocab_size)

    features = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    select_y = torch.tensor(df["select_label"].values, dtype=torch.float32)
    mask_y = torch.tensor(
        df[["keep_subject", "keep_relation", "keep_object"]].values,
        dtype=torch.float32,
    )

    ds = TensorDataset(subject, relation, obj, features, select_y, mask_y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = NeuralSemanticPlanner(
        object_vocab_size=args.object_vocab_size,
        relation_vocab_size=args.relation_vocab_size,
        feature_dim=len(feat_cols),
    )

    pos = float(select_y.sum().item())
    neg = float(len(select_y) - pos)
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_sel = 0.0
        total_mask = 0.0
        total_keep = 0.0
        total_select_prob = 0.0

        for s, r, o, feat, sy, my in dl:
            select_logit, mask_logits = model(s, r, o, feat)

            select_loss = F.binary_cross_entropy_with_logits(
                select_logit,
                sy,
                pos_weight=pos_weight,
            )

            # Learn masks more strongly for positive candidates,
            # but still give weak supervision to negatives.
            mask_weight = (0.25 + 0.75 * sy).unsqueeze(-1)
            raw_mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits,
                my,
                reduction="none",
            )
            mask_loss = (raw_mask_loss * mask_weight).mean()

            select_prob = torch.sigmoid(select_logit)
            mask_prob = torch.sigmoid(mask_logits)

            # Expected transmitted fields only matter for selected candidates.
            expected_keep = (select_prob.unsqueeze(-1) * mask_prob).mean()

            loss = (
                select_loss
                + args.mask_loss_weight * mask_loss
                + args.bit_lambda * expected_keep
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(s)
            total_sel += select_loss.item() * len(s)
            total_mask += mask_loss.item() * len(s)
            total_keep += expected_keep.item() * len(s)
            total_select_prob += select_prob.mean().item() * len(s)

        n = len(ds)
        print({
            "epoch": epoch,
            "loss": total_loss / n,
            "select_loss": total_sel / n,
            "mask_loss": total_mask / n,
            "expected_keep": total_keep / n,
            "avg_select_prob": total_select_prob / n,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "object_vocab_size": args.object_vocab_size,
        "relation_vocab_size": args.relation_vocab_size,
        "feature_dim": len(feat_cols),
        "feature_cols": feat_cols,
    }, out_path)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
