import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from src.methods.neural_semantic_planner import NeuralSemanticPlanner
from src.methods.nst_features import nst_v2_feature_names


def shifted_ids(series, vocab_size: int):
    x = torch.tensor(series.clip(lower=0).values + 1, dtype=torch.long)
    x = torch.clamp(x, min=0, max=vocab_size - 1)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/nsp/nsp_v3_listwise_train.csv")
    parser.add_argument("--out", default="results/nsp/nsp_v3_listwise.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-loss-weight", type=float, default=0.50)
    parser.add_argument("--bce-loss-weight", type=float, default=0.20)
    parser.add_argument("--bit-lambda", type=float, default=0.05)
    parser.add_argument("--object-vocab-size", type=int, default=1118)
    parser.add_argument("--relation-vocab-size", type=int, default=208)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    feat_cols = nst_v2_feature_names()

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing NSP listwise feature columns: {missing}")

    model = NeuralSemanticPlanner(
        object_vocab_size=args.object_vocab_size,
        relation_vocab_size=args.relation_vocab_size,
        feature_dim=len(feat_cols),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    groups = []
    for _, g in df.groupby("group_id", sort=False):
        groups.append(g.reset_index(drop=True))

    print("num_groups:", len(groups))
    print("num_rows:", len(df))

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_rank = 0.0
        total_bce = 0.0
        total_mask = 0.0
        total_keep = 0.0
        total_select_prob = 0.0
        total_items = 0

        # Shuffle group order.
        perm = torch.randperm(len(groups)).tolist()

        for gi in perm:
            g = groups[gi]

            subject = shifted_ids(g["subject_id"], args.object_vocab_size)
            relation = shifted_ids(g["relation_id"], args.relation_vocab_size)
            obj = shifted_ids(g["object_id"], args.object_vocab_size)
            features = torch.tensor(g[feat_cols].values, dtype=torch.float32)

            teacher_prob = torch.tensor(g["teacher_prob"].values, dtype=torch.float32)
            select_y = torch.tensor(g["select_label"].values, dtype=torch.float32)
            mask_y = torch.tensor(
                g[["keep_subject", "keep_relation", "keep_object"]].values,
                dtype=torch.float32,
            )

            select_logit, mask_logits = model(subject, relation, obj, features)

            # Listwise KL / cross entropy:
            log_p = F.log_softmax(select_logit, dim=0)
            rank_loss = -(teacher_prob * log_p).sum()

            # Auxiliary BCE to keep absolute positive/negative supervision.
            bce_loss = F.binary_cross_entropy_with_logits(select_logit, select_y)

            mask_weight = (0.25 + 0.75 * select_y).unsqueeze(-1)
            raw_mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits,
                mask_y,
                reduction="none",
            )
            mask_loss = (raw_mask_loss * mask_weight).mean()

            select_prob = torch.sigmoid(select_logit)
            mask_prob = torch.sigmoid(mask_logits)
            expected_keep = (select_prob.unsqueeze(-1) * mask_prob).mean()

            loss = (
                rank_loss
                + args.bce_loss_weight * bce_loss
                + args.mask_loss_weight * mask_loss
                + args.bit_lambda * expected_keep
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            n = len(g)
            total_loss += float(loss.item()) * n
            total_rank += float(rank_loss.item()) * n
            total_bce += float(bce_loss.item()) * n
            total_mask += float(mask_loss.item()) * n
            total_keep += float(expected_keep.item()) * n
            total_select_prob += float(select_prob.mean().item()) * n
            total_items += n

        print({
            "epoch": epoch,
            "loss": total_loss / max(1, total_items),
            "rank_loss": total_rank / max(1, total_items),
            "bce_loss": total_bce / max(1, total_items),
            "mask_loss": total_mask / max(1, total_items),
            "expected_keep": total_keep / max(1, total_items),
            "avg_select_prob": total_select_prob / max(1, total_items),
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
