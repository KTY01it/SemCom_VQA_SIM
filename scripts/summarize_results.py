import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any


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
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

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
    SG strict answerability after channel.
    This is the strongest task-oriented proxy for relation-aware VQA.
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

            original = to_float(by_method["original"]["answerability_after_strict"])
            do = to_float(by_method["do"]["answerability_after_strict"])
            go = to_float(by_method["go"]["answerability_after_strict"])

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
            ])

    print_table(
        title="Table 1 — SG strict answerability after channel",
        headers=[
            "channel",
            "n_top",
            "Original-SG",
            "DO-SG",
            "GO-SG",
            "GO-Orig",
            "GO-DO",
        ],
        rows=out,
    )


def summarize_bbox_answerability(answer_rows):
    """
    BBox has no relation.
    Use loose answerability: answer object/attribute appears in delivered bboxes.
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

            original = to_float(by_method["original"]["answerability_after_loose"])
            do = to_float(by_method["do"]["answerability_after_loose"])
            go = to_float(by_method["go"]["answerability_after_loose"])

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
            ])

    print_table(
        title="Table 2 — BBox loose answerability after channel",
        headers=[
            "channel",
            "n_top",
            "Original-BBox",
            "DO-BBox",
            "GO-BBox",
            "GO-Orig",
            "GO-DO",
        ],
        rows=out,
    )


def summarize_before_after_drop(answer_rows):
    """
    Show how much the channel damages answerability.
    Focus on GO, because GO is the proposed ranking.
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
                    after = to_float(r["answerability_after_strict"])
                else:
                    before = to_float(r["answerability_before_loose"])
                    after = to_float(r["answerability_after_loose"])

                drop = before - after

                out.append([
                    semantic_type,
                    channel,
                    n_top,
                    f"{before:.4f}",
                    f"{after:.4f}",
                    f"{drop:+.4f}",
                    f"{to_float(r['ber']):.6f}",
                    f"{to_float(r['packet_error_rate']):.4f}",
                ])

    print_table(
        title="Table 3 — Channel-induced answerability drop for GO",
        headers=[
            "type",
            "channel",
            "n_top",
            "before",
            "after",
            "drop",
            "BER",
            "PER",
        ],
        rows=out,
    )


def summarize_latency(answer_rows):
    """
    Compare SG vs BBox latency under GO.
    Ranking does not change packet size much, so GO is enough.
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

            out.append([
                channel,
                n_top,
                f"{sg_bits:.1f}",
                f"{bbox_bits:.1f}",
                f"{sg_t:.6f}",
                f"{bbox_t:.6f}",
                f"{bbox_t / sg_t:.2f}x" if sg_t > 0 else "NA",
            ])

    print_table(
        title="Table 4 — SG vs BBox communication cost under GO",
        headers=[
            "channel",
            "n_top",
            "SG bits",
            "BBox bits",
            "SG t_com",
            "BBox t_com",
            "BBox/SG",
        ],
        rows=out,
    )


def summarize_best_setting(answer_rows):
    """
    Find best answerability_after under each semantic type/channel/method.
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
                score = to_float(r["answerability_after_strict"], default=-1.0)
            else:
                score = to_float(r["answerability_after_loose"], default=-1.0)

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
        ])

    print_table(
        title="Table 5 — Best answerability setting per method",
        headers=[
            "type",
            "channel",
            "method",
            "best_n_top",
            "best_after",
            "bits",
            "t_com",
        ],
        rows=out,
    )


def main() -> None:
    answer_path = Path("results/answerability_sweep.csv")

    answer_rows = read_csv(answer_path)

    print(f"Loaded answerability rows: {len(answer_rows)} from {answer_path}")

    summarize_sg_answerability(answer_rows)
    summarize_bbox_answerability(answer_rows)
    summarize_before_after_drop(answer_rows)
    summarize_latency(answer_rows)
    summarize_best_setting(answer_rows)


if __name__ == "__main__":
    main()
