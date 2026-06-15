import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.methods.neural_semantic_tokenizer import NeuralSemanticTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/nst/nst_train.csv")
    parser.add_argument("--out", default="results/nst/nst_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bit-lambda", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    subject = torch.tensor(df["subject_id"].clip(lower=0).values + 1, dtype=torch.long)
    relation = torch.tensor(df["relation_id"].clip(lower=0).values + 1, dtype=torch.long)
    obj = torch.tensor(df["object_id"].clip(lower=0).values + 1, dtype=torch.long)

    y = torch.tensor(
        df[["keep_subject", "keep_relation", "keep_object"]].values,
        dtype=torch.float32,
    )

    num_objects = int(max(subject.max().item(), obj.max().item()) + 1)
    num_relations = int(relation.max().item() + 1)

    ds = TensorDataset(subject, relation, obj, y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = NeuralSemanticTokenizer(
        num_objects=num_objects,
        num_relations=num_relations,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_bce = 0.0
        total_keep = 0.0

        for s, r, o, target in dl:
            logits = model(s, r, o)
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
        "num_objects": num_objects,
        "num_relations": num_relations,
    }, out_path)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
