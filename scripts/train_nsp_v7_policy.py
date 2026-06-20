import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.methods.neural_semantic_policy_v7 import NeuralSemanticPolicyV7


def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / mask.sum().clamp_min(eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--select-loss-weight", type=float, default=0.7)
    parser.add_argument("--evidence-loss-weight", type=float, default=1.2)
    parser.add_argument("--mask-loss-weight", type=float, default=0.7)
    parser.add_argument("--budget-loss-weight", type=float, default=0.05)
    parser.add_argument("--pair-loss-weight", type=float, default=0.3)
    parser.add_argument("--answer-pair-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu")

    if "answer_label" not in data:
        raise RuntimeError("Dataset has no answer_label. Rebuild with updated build_nsp_v7_teacher_dataset.py.")

    model = NeuralSemanticPolicyV7(
        object_vocab_size=int(data["object_vocab_size"]),
        relation_vocab_size=int(data["relation_vocab_size"]),
        dense_dim=int(data["dense_dim"]),
        text_dim=int(data["text_dim"]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    n = data["subject_id"].shape[0]
    indices = torch.arange(n)

    cand_total = float(data["candidate_mask"].sum().item())

    select_pos = float(data["select_label"].sum().item())
    select_neg = max(1.0, cand_total - select_pos)
    select_pos_weight = torch.tensor(select_neg / max(1.0, select_pos), device=args.device)

    answer_pos = float(data["answer_label"].sum().item())
    answer_neg = max(1.0, cand_total - answer_pos)
    answer_pos_weight = torch.tensor(answer_neg / max(1.0, answer_pos), device=args.device)

    print("num_groups:", n)
    print("select_positive_ratio:", select_pos / max(1.0, cand_total))
    print("answer_positive_ratio:", answer_pos / max(1.0, cand_total))
    print("select_pos_weight:", float(select_pos_weight.item()))
    print("answer_pos_weight:", float(answer_pos_weight.item()))
    print("device:", args.device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = indices[torch.randperm(n)]

        totals = {
            "loss": 0.0,
            "select_loss": 0.0,
            "evidence_loss": 0.0,
            "mask_loss": 0.0,
            "budget_loss": 0.0,
            "pair_loss": 0.0,
            "answer_pair_loss": 0.0,
            "expected_selected": 0.0,
            "expected_fields": 0.0,
        }
        seen = 0

        for start in range(0, n, args.batch_size):
            idx = perm[start: start + args.batch_size]

            sid = data["subject_id"][idx].to(args.device)
            rid = data["relation_id"][idx].to(args.device)
            oid = data["object_id"][idx].to(args.device)
            dense = data["dense_features"][idx].to(args.device)
            q_emb = data["q_emb"][idx].to(args.device).float()
            t_emb = data["t_emb"][idx].to(args.device).float()
            cand_mask = data["candidate_mask"][idx].to(args.device).bool()
            select_y = data["select_label"][idx].to(args.device)
            answer_y = data["answer_label"][idx].to(args.device)
            mask_y = data["mask_label"][idx].to(args.device)

            select_logits, evidence_logits, mask_logits = model(
                sid, rid, oid, dense, q_emb, t_emb, cand_mask
            )

            raw_select_loss = F.binary_cross_entropy_with_logits(
                select_logits,
                select_y,
                pos_weight=select_pos_weight,
                reduction="none",
            )
            select_loss = masked_mean(raw_select_loss, cand_mask.float())

            raw_evidence_loss = F.binary_cross_entropy_with_logits(
                evidence_logits,
                answer_y,
                pos_weight=answer_pos_weight,
                reduction="none",
            )
            evidence_loss = masked_mean(raw_evidence_loss, cand_mask.float())

            # Mask supervision: teacher-selected and answer-bearing candidates matter most.
            important = torch.maximum(select_y, answer_y)
            mask_weight = cand_mask.float().unsqueeze(-1) * (
                0.05 + 0.95 * important.unsqueeze(-1)
            )
            raw_mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits,
                mask_y,
                reduction="none",
            )
            mask_loss = (raw_mask_loss * mask_weight).sum() / mask_weight.sum().clamp_min(1.0)

            select_prob = torch.sigmoid(select_logits) * cand_mask.float()
            mask_prob = torch.sigmoid(mask_logits) * cand_mask.float().unsqueeze(-1)

            expected_selected = select_prob.sum(dim=1)
            target_selected = select_y.sum(dim=1)
            selected_loss = F.mse_loss(expected_selected, target_selected)

            expected_fields = (select_prob.unsqueeze(-1) * mask_prob).sum(dim=(1, 2))
            target_fields = mask_y.sum(dim=(1, 2))
            field_budget_loss = F.mse_loss(expected_fields, target_fields)

            budget_loss = selected_loss + field_budget_loss

            pair_losses = []
            answer_pair_losses = []

            for b in range(select_logits.shape[0]):
                valid = cand_mask[b]

                pos = valid & (select_y[b] > 0.5)
                neg = valid & (select_y[b] < 0.5)

                if pos.any() and neg.any():
                    pos_mean = select_logits[b][pos].mean()
                    neg_mean = select_logits[b][neg].mean()
                    pair_losses.append(F.relu(1.0 - pos_mean + neg_mean))

                ans = valid & (answer_y[b] > 0.5)
                non_ans = valid & (answer_y[b] < 0.5)

                if ans.any() and non_ans.any():
                    ans_mean = evidence_logits[b][ans].mean()
                    non_ans_mean = evidence_logits[b][non_ans].mean()
                    answer_pair_losses.append(F.relu(1.0 - ans_mean + non_ans_mean))

            pair_loss = torch.stack(pair_losses).mean() if pair_losses else torch.tensor(0.0, device=args.device)
            answer_pair_loss = torch.stack(answer_pair_losses).mean() if answer_pair_losses else torch.tensor(0.0, device=args.device)

            loss = (
                args.select_loss_weight * select_loss
                + args.evidence_loss_weight * evidence_loss
                + args.mask_loss_weight * mask_loss
                + args.budget_loss_weight * budget_loss
                + args.pair_loss_weight * pair_loss
                + args.answer_pair_loss_weight * answer_pair_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = len(idx)
            totals["loss"] += float(loss.item()) * bs
            totals["select_loss"] += float(select_loss.item()) * bs
            totals["evidence_loss"] += float(evidence_loss.item()) * bs
            totals["mask_loss"] += float(mask_loss.item()) * bs
            totals["budget_loss"] += float(budget_loss.item()) * bs
            totals["pair_loss"] += float(pair_loss.item()) * bs
            totals["answer_pair_loss"] += float(answer_pair_loss.item()) * bs
            totals["expected_selected"] += float(expected_selected.mean().item()) * bs
            totals["expected_fields"] += float(expected_fields.mean().item()) * bs
            seen += bs

        print({
            "epoch": epoch,
            **{k: v / max(1, seen) for k, v in totals.items()},
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "object_vocab_size": int(data["object_vocab_size"]),
        "relation_vocab_size": int(data["relation_vocab_size"]),
        "dense_dim": int(data["dense_dim"]),
        "text_dim": int(data["text_dim"]),
        "text_model": data["text_model"],
        "dense_feature_names": data["dense_feature_names"],
        "emb_dim": 64,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "max_candidates": int(data["max_candidates"]),
        "n_top": int(data["n_top"]),
    }, out_path)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
