import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def read_csv(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def to_float(x, default=None):
    if x is None:
        return default

    if isinstance(x, str) and x.strip() == "":
        return default

    try:
        return float(x)
    except ValueError:
        return default


def get_metric(row, preferred_key: str, fallback_key: str, default=None):
    """
    Use validated metric first. Fall back to raw metric for backward compatibility.
    """
    value = to_float(row.get(preferred_key), default=None)
    if value is not None:
        return value
    return to_float(row.get(fallback_key), default=default)


def filter_rows(rows, **conditions):
    out = []

    for row in rows:
        keep = True

        for key, expected in conditions.items():
            if str(row.get(key)) != str(expected):
                keep = False
                break

        if keep:
            out.append(row)

    return out


def print_table(title: str, rows: List[List[Any]], headers: List[str]) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)

    table = [headers] + rows
    widths = [max(len(str(r[i])) for r in table) for i in range(len(headers))]

    for idx, r in enumerate(table):
        line = "  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers)))
        print(line)

        if idx == 0:
            print("-" * len(line))


def summarize_sg_answerability(answer_rows):
    """
    Main table:
    SG strict proxy answerability after channel.
    Prefer validated metric after dropping invalid semantic packets.
    """
    out = []

    for channel in ["awgn", "rayleigh"]:
        for n_top in ["3", "6", "9", "12", "15", "18"]:
            selected = filter_rows(
                answer_rows,
                semantic_type="sg",
                channel=channel,
                n_top=n_top,
            )

            by_method = {r["ranking_method"]: r for r in selected}

            if not all(m in by_method for m in ["original", "do", "go"]):
                continue

            original = get_metric(
                by_method["original"],
                "answerability_after_strict_validated",
                "answerability_after_strict",
            )
            do = get_metric(
                by_method["do"],
                "answerability_after_strict_validated",
                "answerability_after_strict",
            )
            go = get_metric(
                by_method["go"],
                "answerability_after_strict_validated",
                "answerability_after_strict",
            )

            go_gain_vs_original = go - original
            go_gain_vs_do = go - do

            out.append([
                channel,
                n_top,
                f"{original:.4f}",
                f"{do:.4f}",
                f"{go:.4f}",
                f"{go_gain_vs_original:+.4f}",
                f"{go_gain_vs_do:+.4f}",
                f"{to_float(by_method['go'].get('valid_packet_rate'), default=0.0):.4f}",
                f"{to_float(by_method['go'].get('invalid_packet_rate'), default=0.0):.4f}",
            ])

    print_table(
        title="Table 1 — SG strict proxy answerability after channel, validated/drop-invalid",
        headers=[
            "channel",
            "n_top",
            "Original-SG",
            "DO-SG",
            "GO-SG",
            "GO-Orig",
            "GO-DO",
            "GO valid",
            "GO invalid",
        ],
        rows=out,
    )


def summarize_bbox_answerability(answer_rows):
    """
    BBox has no relation.
    Use loose proxy answerability after dropping invalid BBox packets.
    """
    out = []

    for channel in ["awgn", "rayleigh"]:
        for n_top in ["3", "6", "9", "12", "15", "18"]:
            selected = filter_rows(
                answer_rows,
                semantic_type="bbox",
                channel=channel,
                n_top=n_top,
            )

            by_method = {r["ranking_method"]: r for r in selected}

            if not all(m in by_method for m in ["original", "do", "go"]):
                continue

            original = get_metric(
                by_method["original"],
                "answerability_after_loose_validated",
                "answerability_after_loose",
            )
            do = get_metric(
                by_method["do"],
                "answerability_after_loose_validated",
                "answerability_after_loose",
            )
            go = get_metric(
                by_method["go"],
                "answerability_after_loose_validated",
                "answerability_after_loose",
            )

            go_gain_vs_original = go - original
            go_gain_vs_do = go - do

            out.append([
                channel,
                n_top,
                f"{original:.4f}",
                f"{do:.4f}",
                f"{go:.4f}",
                f"{go_gain_vs_original:+.4f}",
                f"{go_gain_vs_do:+.4f}",
                f"{to_float(by_method['go'].get('valid_packet_rate'), default=0.0):.4f}",
                f"{to_float(by_method['go'].get('invalid_packet_rate'), default=0.0):.4f}",
            ])

    print_table(
        title="Table 2 — BBox loose proxy answerability after channel, validated/drop-invalid",
        headers=[
            "channel",
            "n_top",
            "Original-BBox",
            "DO-BBox",
            "GO-BBox",
            "GO-Orig",
            "GO-DO",
            "GO valid",
            "GO invalid",
        ],
        rows=out,
    )


def summarize_before_after_drop(answer_rows):
    """
    Show how much the channel damages validated answerability.
    Focus on GO.
    """
    out = []

    for semantic_type in ["sg", "bbox"]:
        for channel in ["awgn", "rayleigh"]:
            for n_top in ["3", "6", "9", "12", "15", "18"]:
                selected = filter_rows(
                    answer_rows,
                    semantic_type=semantic_type,
                    ranking_method="go",
                    channel=channel,
                    n_top=n_top,
                )

                if not selected:
                    continue

                r = selected[0]

                if semantic_type == "sg":
                    before = to_float(r["answerability_before_strict"])
                    after_raw = to_float(r["answerability_after_strict"])
                    after_validated = get_metric(
                        r,
                        "answerability_after_strict_validated",
                        "answerability_after_strict",
                    )
                else:
                    before = to_float(r["answerability_before_loose"])
                    after_raw = to_float(r["answerability_after_loose"])
                    after_validated = get_metric(
                        r,
                        "answerability_after_loose_validated",
                        "answerability_after_loose",
                    )

                drop_validated = before - after_validated

                out.append([
                    semantic_type,
                    channel,
                    n_top,
                    f"{before:.4f}",
                    f"{after_raw:.4f}",
                    f"{after_validated:.4f}",
                    f"{drop_validated:+.4f}",
                    f"{to_float(r['ber']):.6f}",
                    f"{to_float(r['packet_error_rate']):.4f}",
                    f"{to_float(r.get('valid_packet_rate'), default=0.0):.4f}",
                ])

    print_table(
        title="Table 3 — Channel-induced proxy answerability drop for GO, validated/drop-invalid",
        headers=[
            "type",
            "channel",
            "n_top",
            "before",
            "after_raw",
            "after_valid",
            "drop_valid",
            "BER",
            "PER",
            "valid_rate",
        ],
        rows=out,
    )


def summarize_latency(answer_rows):
    """
    Compare SG vs BBox communication latency under GO.
    """
    out = []

    for channel in ["awgn", "rayleigh"]:
        for n_top in ["3", "6", "9", "12", "15", "18"]:
            sg_rows = filter_rows(
                answer_rows,
                semantic_type="sg",
                ranking_method="go",
                channel=channel,
                n_top=n_top,
            )
            bbox_rows = filter_rows(
                answer_rows,
                semantic_type="bbox",
                ranking_method="go",
                channel=channel,
                n_top=n_top,
            )

            if not sg_rows or not bbox_rows:
                continue

            sg = sg_rows[0]
            bbox = bbox_rows[0]

            sg_bits = to_float(sg["avg_source_bits"])
            bbox_bits = to_float(bbox["avg_source_bits"])
            sg_t = to_float(sg["t_com_sec"])
            bbox_t = to_float(bbox["t_com_sec"])

            sg_ans = get_metric(
                sg,
                "answerability_after_strict_validated",
                "answerability_after_strict",
            )
            bbox_ans = get_metric(
                bbox,
                "answerability_after_loose_validated",
                "answerability_after_loose",
            )

            out.append([
                channel,
                n_top,
                f"{sg_bits:.1f}",
                f"{bbox_bits:.1f}",
                f"{sg_t:.6f}",
                f"{bbox_t:.6f}",
                f"{bbox_t / sg_t:.2f}x" if sg_t > 0 else "NA",
                f"{sg_ans:.4f}",
                f"{bbox_ans:.4f}",
            ])

    print_table(
        title="Table 4 — SG vs BBox communication cost under GO, validated/drop-invalid",
        headers=[
            "channel",
            "n_top",
            "SG bits",
            "BBox bits",
            "SG t_com",
            "BBox t_com",
            "BBox/SG",
            "SG ans",
            "BBox ans",
        ],
        rows=out,
    )


def summarize_best_setting(answer_rows):
    """
    Find best validated answerability under each semantic type/channel/method.
    """
    grouped = defaultdict(list)

    for r in answer_rows:
        key = (
            r["semantic_type"],
            r["channel"],
            r["ranking_method"],
        )
        grouped[key].append(r)

    out = []

    for key, group in sorted(grouped.items()):
        semantic_type, channel, method = key

        best_row = None
        best_score = -1.0

        for r in group:
            if semantic_type == "sg":
                score = get_metric(
                    r,
                    "answerability_after_strict_validated",
                    "answerability_after_strict",
                    default=-1.0,
                )
            else:
                score = get_metric(
                    r,
                    "answerability_after_loose_validated",
                    "answerability_after_loose",
                    default=-1.0,
                )

            if score > best_score:
                best_score = score
                best_row = r

        if best_row is None:
            continue

        out.append([
            semantic_type,
            channel,
            method,
            best_row["n_top"],
            f"{best_score:.4f}",
            f"{to_float(best_row['avg_source_bits']):.1f}",
            f"{to_float(best_row['t_com_sec']):.6f}",
            f"{to_float(best_row.get('valid_packet_rate'), default=0.0):.4f}",
        ])

    print_table(
        title="Table 5 — Best validated proxy answerability setting per method",
        headers=[
            "type",
            "channel",
            "method",
            "best_n_top",
            "best_valid_after",
            "bits",
            "t_com",
            "valid_rate",
        ],
        rows=out,
    )

def summarize_image_baseline(image_rows):
    """
    Compare raw image transmission against GO semantic transmission.

    This is a communication-cost baseline only.
    It does not claim image-level VQA accuracy.
    """
    comparison_rows = [
        r for r in image_rows
        if r.get("row_type") == "image_vs_semantic"
    ]

    if not comparison_rows:
        print("\nNo image_vs_semantic rows found in image_baseline.csv")
        return

    out = []

    for channel in ["awgn", "rayleigh"]:
        for semantic_type in ["sg", "bbox"]:
            selected = [
                r for r in comparison_rows
                if r.get("channel") == channel and r.get("semantic_type") == semantic_type
            ]

            selected = sorted(selected, key=lambda r: float(r["n_top"]))

            for r in selected:
                out.append([
                    channel,
                    semantic_type,
                    str(int(float(r["n_top"]))),
                    f"{to_float(r['image_bits']):.0f}",
                    f"{to_float(r['semantic_bits']):.1f}",
                    f"{to_float(r['image_to_semantic_bit_ratio']):.1f}x",
                    f"{to_float(r['image_t_com_sec']):.3f}",
                    f"{to_float(r['semantic_t_com_sec']):.6f}",
                    f"{to_float(r['image_to_semantic_latency_ratio']):.1f}x",
                    f"{to_float(r['semantic_answerability_validated']):.4f}",
                ])

    print_table(
        title="Table 6 — Raw Image Transmission vs GO semantic transmission",
        headers=[
            "channel",
            "type",
            "n_top",
            "image_bits",
            "semantic_bits",
            "bit_ratio",
            "image_t",
            "semantic_t",
            "latency_ratio",
            "semantic_ans",
        ],
        rows=out,
    )
    
def summarize_total_latency_breakdown(latency_rows):
    """
    Summarize paper-style total latency:
        t_total = max(t_question_parser, t_image_processing + t_com + t_decode)
                  + t_answer_reasoning

    Uses results/latency_breakdown.csv.
    """
    comparison_rows = [
        r for r in latency_rows
        if r.get("row_type") == "image_vs_semantic_total"
    ]

    if not comparison_rows:
        print("\nNo image_vs_semantic_total rows found in latency_breakdown.csv")
        return

    out = []

    for channel in ["awgn", "rayleigh"]:
        for semantic_type in ["sg", "bbox"]:
            selected = [
                r for r in comparison_rows
                if r.get("channel") == channel and r.get("semantic_type") == semantic_type
            ]
            selected = sorted(selected, key=lambda r: float(r["n_top"]))

            for r in selected:
                out.append([
                    channel,
                    semantic_type,
                    str(int(float(r["n_top"]))),
                    f"{to_float(r['image_t_total_ms']):.2f}",
                    f"{to_float(r['semantic_t_total_ms']):.2f}",
                    f"{to_float(r['image_to_semantic_total_ratio']):.2f}x",
                    f"{to_float(r['image_t_com_ms']):.2f}",
                    f"{to_float(r['semantic_t_com_ms']):.3f}",
                    f"{to_float(r['semantic_answerability_validated']):.4f}",
                    f"{to_float(r['semantic_valid_packet_rate']):.4f}",
                ])

    print_table(
        title="Table 7 — Paper-style total latency: Raw Image vs GO semantic transmission",
        headers=[
            "channel",
            "type",
            "n_top",
            "image_total_ms",
            "semantic_total_ms",
            "total_ratio",
            "image_com_ms",
            "semantic_com_ms",
            "semantic_ans",
            "valid_rate",
        ],
        rows=out,
    )
    
def main() -> None:
    answer_path = Path("results/answerability_sweep.csv")
    image_path = Path("results/image_baseline.csv")

    answer_rows = read_csv(answer_path)

    print(f"Loaded answerability rows: {len(answer_rows)} from {answer_path}")
    print("Primary metric: *_validated answerability after invalid-packet drop.")

    summarize_sg_answerability(answer_rows)
    summarize_bbox_answerability(answer_rows)
    summarize_before_after_drop(answer_rows)
    summarize_latency(answer_rows)
    summarize_best_setting(answer_rows)

    if image_path.exists():
        image_rows = read_csv(image_path)
        print(f"\nLoaded image baseline rows: {len(image_rows)} from {image_path}")
        summarize_image_baseline(image_rows)
    else:
        print(f"\nImage baseline missing: {image_path}")

    latency_path = Path("results/latency_breakdown.csv")
    if latency_path.exists():
        latency_rows = read_csv(latency_path)
        print(f"\nLoaded latency breakdown rows: {len(latency_rows)} from {latency_path}")
        summarize_total_latency_breakdown(latency_rows)
    else:
        print(f"\nLatency breakdown missing: {latency_path}")
        

if __name__ == "__main__":
    main()