import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.methods.neural_semantic_tokenizer_v2 import QuestionConditionedNST
from src.methods.nst_features import nst_v2_feature_names


def shifted_ids(series, vocab_size: int):
    x = torch.tensor(series.clip(lower=0).values + 1, dtype=torch.long)
    x = torch.clamp(x, min=0, max=vocab_size - 1)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/nst/nst_v2_train.csv")
    parser.add_argument("--out", default="results/nst/nst_v2_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bit-lambda", type=float, default=0.50)
    parser.add_argument("--object-vocab-size", type=int, default=1118)
    parser.add_argument("--relation-vocab-size", type=int, default=208)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    feat_cols = nst_v2_feature_names()
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing NST-v2 feature columns: {missing}")

    subject = shifted_ids(df["subject_id"], args.object_vocab_size)
    relation = shifted_ids(df["relation_id"], args.relation_vocab_size)
    obj = shifted_ids(df["object_id"], args.object_vocab_size)

    features = torch.tensor(df[feat_cols].values, dtype=torch.float32)

    y = torch.tensor(
        df[["keep_subject", "keep_relation", "keep_object"]].values,
        dtype=torch.float32,
    )

    ds = TensorDataset(subject, relation, obj, features, y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = QuestionConditionedNST(
        object_vocab_size=args.object_vocab_size,
        relation_vocab_size=args.relation_vocab_size,
        feature_dim=len(feat_cols),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_bce = 0.0
        total_keep = 0.0

        for s, r, o, feat, target in dl:
            logits = model(s, r, o, feat)

            bce = F.binary_cross_entropy_with_logits(logits, target)
            probs = torch.sigmoid(logits)
            keep_ratio = probs.mean()

            loss = bce + args.bit_lambda * keep_ratio

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(s)
            total_bce += bce.item() * len(s)
            total_keep += keep_ratio.item() * len(s)

        n = len(ds)
        print({
            "epoch": epoch,
            "loss": total_loss / n,
            "bce": total_bce / n,
            "avg_keep_prob": total_keep / n,
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
