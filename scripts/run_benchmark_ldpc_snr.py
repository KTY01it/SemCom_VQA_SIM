import argparse
import csv
import random
from pathlib import Path

from src.comm.bpsk import bpsk_demodulate_hard, bpsk_modulate
from src.comm.channel import awgn_channel, rayleigh_channel
from src.comm.latency import communication_latency_sec
from src.data.gqa_subset import GQACommSubset
from src.eval.metrics import bit_error_rate, packet_error_rate
from src.eval.proxy_metrics import (
    answer_hit_triplets,
    keyword_hit_rate_triplets,
)
from src.eval.semantic_metrics import (
    sg_field_accuracy,
    sg_triplet_exact_match_rate,
)
from src.eval.delivered_proxy import (
    delivered_triplet_answer_hit,
    delivered_triplet_keyword_hit_rate,
)
from src.eval.evidence_metrics import (
    coverage_ratio,
    redundancy_ratio,
    unique_concept_count,
)
from src.semantic.packet_codec import (
    SGTripletPacket,
    decode_sg_triplets,
    encode_sg_triplets,
)
from src.semantic.ranking import (
    build_object_frequency,
    build_relation_frequency,
    rank_triplets_do,
    rank_triplets_go,
    rank_triplets_original,
)
from src.methods.dbss import dbss_select_triplets
from src.utils.config import load_yaml

# IMPORTANT:
# If your ldpc_codec.py uses different function names,
# change only these two imports and the two calls in transmit_bits_ldpc().
from src.comm.ldpc_codec import LDPCConfig, SystematicLDPC

def make_ldpc_codec() -> SystematicLDPC:
    return SystematicLDPC(
        LDPCConfig(
            k=256,
            m=256,
            col_weight=3,
            max_iter=30,
            seed=123,
        )
    )
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LDPC-coded SNR benchmark for SG semantic selection methods."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["random", "original", "do", "go", "dbss"],
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["awgn", "rayleigh"],
        choices=["awgn", "rayleigh"],
    )
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=[-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16],
    )
    parser.add_argument("--n-top", type=int, default=9)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default="results/benchmark_ldpc/main_snr_sg_ldpc_dbss.csv",
    )
    return parser.parse_args()


def transmit_bits_ldpc(bits, channel_type, snr_db, seed, perfect_csi, codec):
    """
    Source bits -> LDPC encode -> BPSK -> channel -> hard demod -> LDPC decode.

    Returns:
        rx_bits: decoded source bits, same length as original source bits
        coded_bits: transmitted LDPC-coded bits
        rx_coded_bits: received hard coded bits before LDPC decoding
        block_success_rate: LDPC block-level success ratio
    """
    coded_bits, original_len = codec.encode(bits)

    symbols = bpsk_modulate(coded_bits)

    if channel_type == "awgn":
        rx_symbols = awgn_channel(symbols, snr_db=snr_db, seed=seed)
    elif channel_type == "rayleigh":
        rx_symbols = rayleigh_channel(
            symbols,
            snr_db=snr_db,
            seed=seed,
            perfect_csi=perfect_csi,
        )
    else:
        raise ValueError(f"Unknown channel_type: {channel_type}")

    rx_coded_bits = bpsk_demodulate_hard(rx_symbols)
    rx_bits, block_success_rate = codec.decode(rx_coded_bits, original_len)

    return rx_bits, coded_bits, rx_coded_bits, block_success_rate


def select_triplets(
    triplets,
    method,
    keywords,
    object_freq,
    relation_freq,
    question="",
    n_top=9,
    snr_db=8.0,
    channel_type="awgn",
    seed=0,
):
    if method == "random":
        out = list(triplets)
        random.seed(seed)
        random.shuffle(out)
        return out

    if method == "original":
        return rank_triplets_original(triplets)

    if method == "do":
        return rank_triplets_do(triplets, object_freq, relation_freq)

    if method == "go":
        return rank_triplets_go(triplets, keywords, object_freq, relation_freq)

    if method == "dbss":
        return dbss_select_triplets(
            triplets=triplets,
            question=question,
            keywords=keywords,
            n_top=n_top,
            snr_db=snr_db,
            channel_type=channel_type,
        )

    if method == "dbss_no_coverage":
        return dbss_select_triplets(
            triplets=triplets,
            question=question,
            keywords=keywords,
            n_top=n_top,
            snr_db=snr_db,
            channel_type=channel_type,
            alpha=1.0,
            beta=0.0,
            gamma=0.25,
            lamb=0.75,
            mu=0.05,
        )

    if method == "dbss_no_redundancy":
        return dbss_select_triplets(
            triplets=triplets,
            question=question,
            keywords=keywords,
            n_top=n_top,
            snr_db=snr_db,
            channel_type=channel_type,
            alpha=1.0,
            beta=1.0,
            gamma=0.25,
            lamb=0.0,
            mu=0.05,
        )

    raise ValueError(f"Unknown SG ranking method: {method}")


def run_sg_ldpc(
    sg_rows,
    sample_by_qid,
    object_freq,
    relation_freq,
    method,
    channel_type,
    snr_db,
    n_top,
    seed,
    perfect_csi,
    bandwidth_hz,
):
    ber_list = []
    coded_ber_list = []
    per_list = []
    exact_list = []
    field_acc_list = []
    keyword_hit_list = []
    answer_hit_list = []
    delivered_keyword_hit_list = []
    delivered_answer_hit_list = []
    latency_list = []
    source_bit_list = []
    coded_bit_list = []
    coverage_list = []
    redundancy_list = []
    unique_concept_list = []
    ldpc_block_success_list = []

    codec = make_ldpc_codec()

    for sample_idx, row in enumerate(sg_rows):
        qid = row["question_id"]
        sample = sample_by_qid[qid]

        ranked = select_triplets(
            triplets=row["triplets"],
            method=method,
            keywords=sample["keywords"],
            object_freq=object_freq,
            relation_freq=relation_freq,
            question=sample.get("question", ""),
            n_top=n_top,
            snr_db=snr_db,
            channel_type=channel_type,
            seed=seed + sample_idx,
        )

        selected = ranked[:n_top]
        if not selected:
            continue

        tx_packets = [
            SGTripletPacket(
                subject_id=t["subject_id"],
                relation_id=t["relation_id"],
                object_id=t["object_id"],
            )
            for t in selected
        ]

        tx_bits = encode_sg_triplets(tx_packets)

        rx_bits, coded_bits, rx_coded_bits, block_success_rate = transmit_bits_ldpc(
            tx_bits,
            channel_type=channel_type,
            snr_db=snr_db,
            seed=seed + sample_idx,
            perfect_csi=perfect_csi,
            codec=codec,
        )

        ldpc_block_success_list.append(block_success_rate)

        rx_packets = decode_sg_triplets(rx_bits, num_triplets=len(tx_packets))

        ber_list.append(bit_error_rate(tx_bits, rx_bits))
        coded_ber_list.append(bit_error_rate(coded_bits, rx_coded_bits))
        per_list.append(packet_error_rate(tx_bits, rx_bits, packet_size_bits=48))
        exact_list.append(sg_triplet_exact_match_rate(tx_packets, rx_packets))
        field_acc_list.append(sg_field_accuracy(tx_packets, rx_packets))

        keyword_hit_list.append(keyword_hit_rate_triplets(selected, sample["keywords"]))
        answer_hit_list.append(answer_hit_triplets(selected, sample["answer"]))

        delivered_keyword_hit_list.append(
            delivered_triplet_keyword_hit_rate(
                selected_triplets=selected,
                tx_packets=tx_packets,
                rx_packets=rx_packets,
                keywords=sample["keywords"],
            )
        )
        delivered_answer_hit_list.append(
            delivered_triplet_answer_hit(
                selected_triplets=selected,
                tx_packets=tx_packets,
                rx_packets=rx_packets,
                answer=sample["answer"],
            )
        )

        coverage_list.append(
            coverage_ratio(
                selected_units=selected,
                question=sample.get("question", ""),
                keywords=sample.get("keywords", []),
            )
        )
        redundancy_list.append(redundancy_ratio(selected))
        unique_concept_list.append(unique_concept_count(selected))

        source_bit_list.append(len(tx_bits))
        coded_bit_list.append(len(coded_bits))

        # For communication latency under LDPC-coded transmission,
        # use coded bit length because coded bits are actually transmitted.
        latency_list.append(
            communication_latency_sec(
                num_bits=len(coded_bits),
                bandwidth_hz=bandwidth_hz,
                snr_db=snr_db,
            )
        )

    if not ber_list:
        raise RuntimeError("No valid SG samples were evaluated.")

    return {
        "semantic_type": "sg",
        "ranking_method": method,
        "coding": "ldpc",
        "channel": channel_type,
        "snr_db": snr_db,
        "n_top": n_top,
        "num_samples": len(ber_list),
        "avg_source_bits": sum(source_bit_list) / len(source_bit_list),
        "avg_coded_bits": sum(coded_bit_list) / len(coded_bit_list),
        "avg_num_units": (sum(source_bit_list) / len(source_bit_list)) / 48.0,
        "coded_ber": sum(coded_ber_list) / len(coded_ber_list),
        "ldpc_block_success_rate": sum(ldpc_block_success_list) / len(ldpc_block_success_list),
        "ber": sum(ber_list) / len(ber_list),
        "packet_error_rate": sum(per_list) / len(per_list),
        "semantic_exact_match": sum(exact_list) / len(exact_list),
        "semantic_field_accuracy": sum(field_acc_list) / len(field_acc_list),
        "proxy_keyword_hit_rate": sum(keyword_hit_list) / len(keyword_hit_list),
        "proxy_answer_hit_rate": sum(answer_hit_list) / len(answer_hit_list),
        "delivered_keyword_hit_rate": sum(delivered_keyword_hit_list) / len(delivered_keyword_hit_list),
        "delivered_answer_hit_rate": sum(delivered_answer_hit_list) / len(delivered_answer_hit_list),
        "coverage_ratio": sum(coverage_list) / len(coverage_list),
        "redundancy_ratio": sum(redundancy_list) / len(redundancy_list),
        "unique_concept_count": sum(unique_concept_list) / len(unique_concept_list),
        "t_com_sec": sum(latency_list) / len(latency_list),
    }


def main():
    args = parse_args()

    cfg = load_yaml("configs/experiment.yaml")
    seed = int(cfg["project"]["seed"]) if args.seed is None else args.seed
    perfect_csi = bool(cfg["channel"]["perfect_csi"])
    bandwidth_hz = float(cfg["latency"]["bandwidth_hz"])

    ds = GQACommSubset(Path(cfg["data"]["root"]))

    samples = ds.load_samples(limit=args.num_samples)
    sample_by_qid = {s["question_id"]: s for s in samples}
    sg_rows = ds.load_sg_triplets(limit=args.num_samples)

    all_samples_for_freq = ds.load_samples()
    object_freq = build_object_frequency(all_samples_for_freq)
    relation_freq = build_relation_frequency(all_samples_for_freq)

    out_rows = []

    for channel_type in args.channels:
        for snr_db in args.snrs:
            for method in args.methods:
                row = run_sg_ldpc(
                    sg_rows=sg_rows,
                    sample_by_qid=sample_by_qid,
                    object_freq=object_freq,
                    relation_freq=relation_freq,
                    method=method,
                    channel_type=channel_type,
                    snr_db=snr_db,
                    n_top=args.n_top,
                    seed=seed,
                    perfect_csi=perfect_csi,
                    bandwidth_hz=bandwidth_hz,
                )
                out_rows.append(row)
                print(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
