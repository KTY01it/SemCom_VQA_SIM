import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.methods.neural_semantic_reranker import SemanticRerankerNSP


def pairwise_margin_loss(scores, select_y, utility, margin: float):
    pos_idx = torch.nonzero(select_y > 0.5, as_tuple=False).flatten()
    neg_idx = torch.nonzero(select_y <= 0.5, as_tuple=False).flatten()

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return torch.tensor(0.0, device=scores.device)

    # Hard negatives: high utility but not selected.
    neg_util = utility[neg_idx]
    k = min(len(neg_idx), max(4, len(pos_idx) * 3))
    hard_neg_local = torch.topk(neg_util, k=k).indices
    hard_neg_idx = neg_idx[hard_neg_local]

    pos_scores = scores[pos_idx][:, None]
    neg_scores = scores[hard_neg_idx][None, :]

    return torch.relu(margin - pos_scores + neg_scores).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/nsp/nsp_v5_semantic_train.pt")
    parser.add_argument("--out", default="results/nsp/nsp_v5_semantic.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-loss-weight", type=float, default=0.50)
    parser.add_argument("--bce-loss-weight", type=float, default=0.20)
    parser.add_argument("--pair-loss-weight", type=float, default=0.50)
    parser.add_argument("--bit-lambda", type=float, default=0.05)
    parser.add_argument("--pair-margin", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu")

    dense_dim = int(data["dense_features"].shape[1])
    text_dim = int(data["text_dim"])

    model = SemanticRerankerNSP(
        object_vocab_size=int(data["object_vocab_size"]),
        relation_vocab_size=int(data["relation_vocab_size"]),
        dense_dim=dense_dim,
        text_dim=text_dim,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    group_id = data["group_id"]
    groups = []
    for gid in torch.unique(group_id, sorted=True):
        idx = torch.nonzero(group_id == gid, as_tuple=False).flatten()
        groups.append(idx)

    print("num_groups:", len(groups))
    print("num_rows:", len(group_id))
    print("device:", args.device)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_rank = 0.0
        total_pair = 0.0
        total_bce = 0.0
        total_mask = 0.0
        total_keep = 0.0
        total_items = 0

        perm = torch.randperm(len(groups)).tolist()

        for gi in perm:
            idx = groups[gi]

            s = data["subject_id"][idx].to(args.device)
            r = data["relation_id"][idx].to(args.device)
            o = data["object_id"][idx].to(args.device)
            dense = data["dense_features"][idx].to(args.device)
            q_emb = data["q_emb"][idx].to(args.device).float()
            t_emb = data["t_emb"][idx].to(args.device).float()

            teacher_prob = data["teacher_prob"][idx].to(args.device)
            select_y = data["select_label"][idx].to(args.device)
            utility = data["utility_score"][idx].to(args.device)
            mask_y = data["mask_label"][idx].to(args.device)

            scores, mask_logits = model(s, r, o, dense, q_emb, t_emb)

            rank_loss = -(teacher_prob * F.log_softmax(scores, dim=0)).sum()
            bce_loss = F.binary_cross_entropy_with_logits(scores, select_y)
            pair_loss = pairwise_margin_loss(
                scores=scores,
                select_y=select_y,
                utility=utility,
                margin=args.pair_margin,
            )

            mask_weight = (0.25 + 0.75 * select_y).unsqueeze(-1)
            raw_mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits,
                mask_y,
                reduction="none",
            )
            mask_loss = (raw_mask_loss * mask_weight).mean()

            select_prob = torch.sigmoid(scores)
            mask_prob = torch.sigmoid(mask_logits)
            expected_keep = (select_prob.unsqueeze(-1) * mask_prob).mean()

            loss = (
                rank_loss
                + args.bce_loss_weight * bce_loss
                + args.pair_loss_weight * pair_loss
                + args.mask_loss_weight * mask_loss
                + args.bit_lambda * expected_keep
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            n = len(idx)
            total_loss += float(loss.item()) * n
            total_rank += float(rank_loss.item()) * n
            total_pair += float(pair_loss.item()) * n
            total_bce += float(bce_loss.item()) * n
            total_mask += float(mask_loss.item()) * n
            total_keep += float(expected_keep.item()) * n
            total_items += n

        print({
            "epoch": epoch,
            "loss": total_loss / max(1, total_items),
            "rank_loss": total_rank / max(1, total_items),
            "pair_loss": total_pair / max(1, total_items),
            "bce_loss": total_bce / max(1, total_items),
            "mask_loss": total_mask / max(1, total_items),
            "expected_keep": total_keep / max(1, total_items),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "object_vocab_size": int(data["object_vocab_size"]),
        "relation_vocab_size": int(data["relation_vocab_size"]),
        "dense_dim": dense_dim,
        "text_dim": text_dim,
        "dense_feature_names": data["dense_feature_names"],
        "text_model": data["text_model"],
    }, out_path)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
