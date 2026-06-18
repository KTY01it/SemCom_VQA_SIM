import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.methods.neural_evidence_predictor import NeuralEvidencePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/nsp/nsp_v6_rams_train.pt")
    parser.add_argument("--out", default="results/nsp/nsp_v6_rams.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-loss-weight", type=float, default=0.50)
    parser.add_argument("--bit-lambda", type=float, default=0.02)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = torch.load(args.data, map_location="cpu")

    dense_dim = int(data["dense_features"].shape[1])
    text_dim = int(data["text_dim"])

    model = NeuralEvidencePredictor(
        object_vocab_size=int(data["object_vocab_size"]),
        relation_vocab_size=int(data["relation_vocab_size"]),
        dense_dim=dense_dim,
        text_dim=text_dim,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n = len(data["answer_label"])
    answer_pos = float(data["answer_label"].sum().item())
    answer_neg = float(n - answer_pos)

    pos_weight = torch.tensor(
        [answer_neg / max(1.0, answer_pos)],
        dtype=torch.float32,
        device=args.device,
    )

    print("num_rows:", n)
    print("answer_positive_ratio:", answer_pos / max(1.0, n))
    print("answer_pos_weight:", float(pos_weight.item()))
    print("device:", args.device)

    indices = torch.arange(n)

    for epoch in range(1, args.epochs + 1):
        model.train()

        perm = indices[torch.randperm(n)]

        total_loss = 0.0
        total_evi = 0.0
        total_mask = 0.0
        total_keep = 0.0
        total_seen = 0

        for start in range(0, n, args.batch_size):
            idx = perm[start: start + args.batch_size]

            s = data["subject_id"][idx].to(args.device)
            r = data["relation_id"][idx].to(args.device)
            o = data["object_id"][idx].to(args.device)
            dense = data["dense_features"][idx].to(args.device)
            q_emb = data["q_emb"][idx].to(args.device).float()
            t_emb = data["t_emb"][idx].to(args.device).float()
            answer_y = data["answer_label"][idx].to(args.device)
            mask_y = data["mask_label"][idx].to(args.device)

            evidence_logit, mask_logits = model(s, r, o, dense, q_emb, t_emb)

            evidence_loss = F.binary_cross_entropy_with_logits(
                evidence_logit,
                answer_y,
                pos_weight=pos_weight,
            )

            # Give answer-positive triplets stronger mask supervision.
            mask_weight = (0.25 + 0.75 * answer_y).unsqueeze(-1)
            raw_mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits,
                mask_y,
                reduction="none",
            )
            mask_loss = (raw_mask_loss * mask_weight).mean()

            mask_prob = torch.sigmoid(mask_logits)
            expected_keep = mask_prob.mean()

            loss = (
                evidence_loss
                + args.mask_loss_weight * mask_loss
                + args.bit_lambda * expected_keep
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = len(idx)
            total_loss += float(loss.item()) * bs
            total_evi += float(evidence_loss.item()) * bs
            total_mask += float(mask_loss.item()) * bs
            total_keep += float(expected_keep.item()) * bs
            total_seen += bs

        print({
            "epoch": epoch,
            "loss": total_loss / max(1, total_seen),
            "evidence_loss": total_evi / max(1, total_seen),
            "mask_loss": total_mask / max(1, total_seen),
            "expected_keep": total_keep / max(1, total_seen),
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
