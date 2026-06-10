import json
from pathlib import Path

from src.eval.proxy_metric_contract import all_contracts
from src.utils.config import load_yaml


def main() -> None:
    cfg = load_yaml("configs/experiment.yaml")
    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "proxy_metric_contract.json"

    contracts = {
        "primary_metric": "validated_answer_related_semantic_coverage",
        "short_name": "validated proxy answerability",
        "claim_boundary": (
            "This metric measures answer-related semantic evidence coverage "
            "after communication, validation, and invalid-packet drop. "
            "It is not full VQA accuracy."
        ),
        "contracts": all_contracts(),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
