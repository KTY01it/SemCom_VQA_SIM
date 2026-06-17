import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from src.methods.neural_set_dpp_planner import SetDPPNeuralPlanner
from src.methods.nst_features import nst_v2_feature_names


def shifted_ids(series, vocab_size: int):
    x = torch.tensor(series.clip(lower=0).values + 1, dtype=torch.long)
    x = torch.clamp(x, min=0, max=vocab_size - 1)
    return x


def diversity_regularizer(diversity, select_label):
    """
    Penalize high cosine similarity among teacher-selected items.
    diversity: [1, M, D], normalized
    select_label: [M]
    """
    pos_idx = torch.nonzero(select_label > 0.5, as_tuple=False).flatten()
    if len(pos_idx) <= 1:
        return torch.tensor(0.0, dtype=torch.float32)

    z = diversity[0, pos_idx]  # [P, D]
    sim = z @ z.t()

    eye = torch.eye(sim.shape[0], dtype=torch.bool)
    off_diag = sim[~eye]

    # Penalize positive similarity. Negative similarity is okay.
    return torch.relu(off_diag).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/nsp/nsp_v4_setdpp_train.csv")
    parser.add_argument("--out", default="results/nsp/nsp_v4_setdpp.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-loss-weight", type=float, default=0.50)
    parser.add_argument("--bce-loss-weight", type=float, default=0.20)
    parser.add_argument("--div-loss-weight", type=float, default=0.10)
    parser.add_argument("--bit-lambda", type=float, default=0.05)
    parser.add_argument("--object-vocab-size", type=int, default=1118)
    parser.add_argument("--relation-vocab-size", type=int, default=208)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    feat_cols = nst_v2_feature_names()

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing SetDPP feature columns: {missing}")

    model = SetDPPNeuralPlanner(
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
        total_div = 0.0
        total_keep = 0.0
        total_items = 0

        perm = torch.randperm(len(groups)).tolist()

        for gi in perm:
            g = groups[gi]

            subject = shifted_ids(g["subject_id"], args.object_vocab_size).unsqueeze(0)
            relation = shifted_ids(g["relation_id"], args.relation_vocab_size).unsqueeze(0)
            obj = shifted_ids(g["object_id"], args.object_vocab_size).unsqueeze(0)
            features = torch.tensor(g[feat_cols].values, dtype=torch.float32).unsqueeze(0)

            teacher_prob = torch.tensor(g["teacher_prob"].values, dtype=torch.float32)
            select_y = torch.tensor(g["select_label"].values, dtype=torch.float32)
            mask_y = torch.tensor(
                g[["keep_subject", "keep_relation", "keep_object"]].values,
                dtype=torch.float32,
            )

            quality_logits, mask_logits, diversity = model(subject, relation, obj, features)

            q = quality_logits[0]          # [M]
            mlog = mask_logits[0]          # [M, 3]

            log_p = F.log_softmax(q, dim=0)
            rank_loss = -(teacher_prob * log_p).sum()

            bce_loss = F.binary_cross_entropy_with_logits(q, select_y)

            mask_weight = (0.25 + 0.75 * select_y).unsqueeze(-1)
            raw_mask_loss = F.binary_cross_entropy_with_logits(
                mlog,
                mask_y,
                reduction="none",
            )
            mask_loss = (raw_mask_loss * mask_weight).mean()

            select_prob = torch.sigmoid(q)
            mask_prob = torch.sigmoid(mlog)

            expected_keep = (select_prob.unsqueeze(-1) * mask_prob).mean()
            div_loss = diversity_regularizer(diversity, select_y)

            loss = (
                rank_loss
                + args.bce_loss_weight * bce_loss
                + args.mask_loss_weight * mask_loss
                + args.div_loss_weight * div_loss
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
            total_div += float(div_loss.item()) * n
            total_keep += float(expected_keep.item()) * n
            total_items += n

        print({
            "epoch": epoch,
            "loss": total_loss / max(1, total_items),
            "rank_loss": total_rank / max(1, total_items),
            "bce_loss": total_bce / max(1, total_items),
            "mask_loss": total_mask / max(1, total_items),
            "div_loss": total_div / max(1, total_items),
            "expected_keep": total_keep / max(1, total_items),
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
