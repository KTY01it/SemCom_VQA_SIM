import argparse
import json
from pathlib import Path

from sentence_transformers import CrossEncoder
from sentence_transformers import InputExample
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    examples = []

    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            examples.append(
                InputExample(
                    texts=[rec["question"], rec["triplet_text"]],
                    label=float(rec["label"]),
                )
            )

    print("num_examples:", len(examples))

    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        max_length=args.max_length,
    )

    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=str(out_dir),
        show_progress_bar=True,
    )

    # Explicit final save. Do not rely on CrossEncoder.fit(output_path=...) here.
    model.save(str(out_dir))

    if not out_dir.exists():
        raise RuntimeError(f"Save failed: {out_dir} does not exist")

    if not ((out_dir / "model.safetensors").exists() or (out_dir / "pytorch_model.bin").exists()):
        raise RuntimeError(f"Save failed: no model weights found in {out_dir}")

    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
