import csv
from pathlib import Path
from typing import Any, Dict, List


def read_csv(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    value = to_float(row.get(preferred_key), default=None)
    if value is not None:
        return value
    return to_float(row.get(fallback_key), default=default)


def print_table(title: str, rows: List[List[Any]], headers: List[str]) -> None:
    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)

    table = [headers] + rows
    widths = [max(len(str(r[i])) for r in table) for i in range(len(headers))]

    for idx, r in enumerate(table):
        line = "  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers)))
        print(line)
        if idx == 0:
            print("-" * len(line))


def index_rows(rows):
    out = {}
    for r in rows:
        key = (
            r["semantic_type"],
            r["channel"],
            r["snr_db"],
            r["n_top"],
            r["coding_mode"],
        )
        out[key] = r
    return out


def summarize_sg_ldpc_gain(rows):
    idx = index_rows(rows)
    out = []

    for channel in ["awgn", "rayleigh"]:
        for snr_db in ["4.0", "6.0", "8.0", "10.0", "12.0"]:
            for n_top in ["3", "6", "9", "12"]:
                k_uncoded = ("sg", channel, snr_db, n_top, "uncoded")
                k_ldpc = ("sg", channel, snr_db, n_top, "ldpc_like")

                if k_uncoded not in idx or k_ldpc not in idx:
                    continue

                u = idx[k_uncoded]
                l = idx[k_ldpc]

                u_ans = get_metric(
                    u,
                    "answerability_after_strict_validated",
                    "answerability_after_strict",
                )
                l_ans = get_metric(
                    l,
                    "answerability_after_strict_validated",
                    "answerability_after_strict",
                )

                u_ber = to_float(u["decoded_ber"])
                l_ber = to_float(l["decoded_ber"])

                u_per = to_float(u["packet_error_rate"])
                l_per = to_float(l["packet_error_rate"])

                u_t = to_float(u["coded_t_com_sec"])
                l_t = to_float(l["coded_t_com_sec"])

                out.append([
                    channel,
                    snr_db,
                    n_top,
                    f"{u_ans:.4f}",
                    f"{l_ans:.4f}",
                    f"{l_ans - u_ans:+.4f}",
                    f"{u_ber:.5f}",
                    f"{l_ber:.5f}",
                    f"{u_per:.4f}",
                    f"{l_per:.4f}",
                    f"{to_float(u.get('valid_packet_rate'), default=0.0):.4f}",
                    f"{to_float(l.get('valid_packet_rate'), default=0.0):.4f}",
                    f"{l_t / u_t:.2f}x" if u_t > 0 else "NA",
                ])

    print_table(
        title="Table 1 — GO-SG LDPC-like gain: validated strict proxy answerability",
        headers=[
            "channel",
            "snr",
            "n_top",
            "uncoded_ans",
            "ldpc_ans",
            "ans_gain",
            "uncoded_BER",
            "ldpc_BER",
            "uncoded_PER",
            "ldpc_PER",
            "uncoded_valid",
            "ldpc_valid",
            "latency_x",
        ],
        rows=out,
    )


def summarize_bbox_ldpc_gain(rows):
    idx = index_rows(rows)
    out = []

    for channel in ["awgn", "rayleigh"]:
        for snr_db in ["4.0", "6.0", "8.0", "10.0", "12.0"]:
            for n_top in ["3", "6", "9", "12"]:
                k_uncoded = ("bbox", channel, snr_db, n_top, "uncoded")
                k_ldpc = ("bbox", channel, snr_db, n_top, "ldpc_like")

                if k_uncoded not in idx or k_ldpc not in idx:
                    continue

                u = idx[k_uncoded]
                l = idx[k_ldpc]

                u_ans = get_metric(
                    u,
                    "answerability_after_loose_validated",
                    "answerability_after_loose",
                )
                l_ans = get_metric(
                    l,
                    "answerability_after_loose_validated",
                    "answerability_after_loose",
                )

                u_ber = to_float(u["decoded_ber"])
                l_ber = to_float(l["decoded_ber"])

                u_per = to_float(u["packet_error_rate"])
                l_per = to_float(l["packet_error_rate"])

                u_t = to_float(u["coded_t_com_sec"])
                l_t = to_float(l["coded_t_com_sec"])

                out.append([
                    channel,
                    snr_db,
                    n_top,
                    f"{u_ans:.4f}",
                    f"{l_ans:.4f}",
                    f"{l_ans - u_ans:+.4f}",
                    f"{u_ber:.5f}",
                    f"{l_ber:.5f}",
                    f"{u_per:.4f}",
                    f"{l_per:.4f}",
                    f"{to_float(u.get('valid_packet_rate'), default=0.0):.4f}",
                    f"{to_float(l.get('valid_packet_rate'), default=0.0):.4f}",
                    f"{l_t / u_t:.2f}x" if u_t > 0 else "NA",
                ])

    print_table(
        title="Table 2 — GO-BBox LDPC-like gain: validated loose proxy answerability",
        headers=[
            "channel",
            "snr",
            "n_top",
            "uncoded_ans",
            "ldpc_ans",
            "ans_gain",
            "uncoded_BER",
            "ldpc_BER",
            "uncoded_PER",
            "ldpc_PER",
            "uncoded_valid",
            "ldpc_valid",
            "latency_x",
        ],
        rows=out,
    )


def summarize_rayleigh_best(rows):
    idx = index_rows(rows)
    out = []

    for semantic_type in ["sg", "bbox"]:
        for snr_db in ["8.0", "10.0", "12.0"]:
            best = None

            for n_top in ["3", "6", "9", "12"]:
                k_uncoded = (semantic_type, "rayleigh", snr_db, n_top, "uncoded")
                k_ldpc = (semantic_type, "rayleigh", snr_db, n_top, "ldpc_like")

                if k_uncoded not in idx or k_ldpc not in idx:
                    continue

                u = idx[k_uncoded]
                l = idx[k_ldpc]

                if semantic_type == "sg":
                    u_ans = get_metric(
                        u,
                        "answerability_after_strict_validated",
                        "answerability_after_strict",
                    )
                    l_ans = get_metric(
                        l,
                        "answerability_after_strict_validated",
                        "answerability_after_strict",
                    )
                else:
                    u_ans = get_metric(
                        u,
                        "answerability_after_loose_validated",
                        "answerability_after_loose",
                    )
                    l_ans = get_metric(
                        l,
                        "answerability_after_loose_validated",
                        "answerability_after_loose",
                    )

                gain = l_ans - u_ans

                if best is None or gain > best["gain"]:
                    best = {
                        "semantic_type": semantic_type,
                        "snr_db": snr_db,
                        "n_top": n_top,
                        "uncoded_ans": u_ans,
                        "ldpc_ans": l_ans,
                        "gain": gain,
                        "uncoded_t": to_float(u["coded_t_com_sec"]),
                        "ldpc_t": to_float(l["coded_t_com_sec"]),
                        "ldpc_success": to_float(l["ldpc_block_success_rate"]),
                        "uncoded_valid": to_float(u.get("valid_packet_rate"), default=0.0),
                        "ldpc_valid": to_float(l.get("valid_packet_rate"), default=0.0),
                    }

            if best is not None:
                out.append([
                    best["semantic_type"],
                    best["snr_db"],
                    best["n_top"],
                    f"{best['uncoded_ans']:.4f}",
                    f"{best['ldpc_ans']:.4f}",
                    f"{best['gain']:+.4f}",
                    f"{best['uncoded_valid']:.4f}",
                    f"{best['ldpc_valid']:.4f}",
                    f"{best['ldpc_t'] / best['uncoded_t']:.2f}x",
                    f"{best['ldpc_success']:.4f}",
                ])

    print_table(
        title="Table 3 — Best LDPC-like gains on Rayleigh channel, validated/drop-invalid",
        headers=[
            "type",
            "snr",
            "best_n_top",
            "uncoded_ans",
            "ldpc_ans",
            "gain",
            "uncoded_valid",
            "ldpc_valid",
            "latency_x",
            "ldpc_success",
        ],
        rows=out,
    )


def main():
    path = Path("results/answerability_sweep_ldpc.csv")
    rows = read_csv(path)

    print(f"Loaded rows: {len(rows)} from {path}")
    print("Primary metric: *_validated answerability after invalid-packet drop.")

    summarize_sg_ldpc_gain(rows)
    summarize_bbox_ldpc_gain(rows)
    summarize_rayleigh_best(rows)


if __name__ == "__main__":
    main()