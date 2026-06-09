from __future__ import annotations

from dataclasses import dataclass

from src.comm.latency import communication_latency_sec


@dataclass(frozen=True)
class TotalLatencyConfig:
    bandwidth_hz: float = 100_000.0

    # Paper-style device throughput.
    # Jetson TX2 peak throughput used as approximation.
    device_tflops: float = 1.33

    # Paper-style computation complexity.
    bbox_extraction_tflops: float = 0.44
    sg_generation_tflops: float = 0.02

    # Lightweight overhead placeholders.
    question_parser_ms: float = 0.0
    answer_reasoning_ms: float = 0.0
    ranking_ms: float = 0.1
    channel_decode_ms: float = 0.0

    # Raw image preparation. Keep zero unless you model JPEG/raw preprocessing.
    raw_image_processing_ms: float = 0.0


def tflops_to_ms(work_tflops: float, device_tflops: float) -> float:
    """
    Convert computation work in TFLOPs to latency in milliseconds.

    latency_sec = work_tflops / device_tflops
    latency_ms  = 1000 * latency_sec
    """
    if device_tflops <= 0:
        raise ValueError(f"device_tflops must be positive, got {device_tflops}")
    return 1000.0 * float(work_tflops) / float(device_tflops)


def semantic_processing_latency_ms(
    semantic_type: str,
    cfg: TotalLatencyConfig,
) -> float:
    """
    Estimate end-device image/semantic processing latency.

    image:
        raw image transmission baseline. No semantic extraction.

    bbox:
        BBox extraction + ranking.

    sg:
        BBox extraction + SG generation + ranking.
        SG generation is assumed to build on object/BBox extraction.
    """
    if semantic_type == "image":
        return float(cfg.raw_image_processing_ms)

    if semantic_type == "bbox":
        return (
            tflops_to_ms(cfg.bbox_extraction_tflops, cfg.device_tflops)
            + cfg.ranking_ms
        )

    if semantic_type == "sg":
        return (
            tflops_to_ms(cfg.bbox_extraction_tflops, cfg.device_tflops)
            + tflops_to_ms(cfg.sg_generation_tflops, cfg.device_tflops)
            + cfg.ranking_ms
        )

    raise ValueError(f"Unknown semantic_type: {semantic_type}")


def total_latency_breakdown(
    semantic_type: str,
    num_bits: int | float,
    snr_db: float,
    cfg: TotalLatencyConfig,
) -> dict:
    """
    Paper-style total latency:

        t_total = max(t_question_parser,
                      t_image_processing + t_com + t_channel_decode)
                  + t_answer_reasoning

    All returned latency fields are in milliseconds except t_com_sec.
    """
    t_com_sec = communication_latency_sec(
        num_bits=int(round(float(num_bits))),
        bandwidth_hz=cfg.bandwidth_hz,
        snr_db=float(snr_db),
    )
    t_com_ms = 1000.0 * t_com_sec

    t_question_parser_ms = float(cfg.question_parser_ms)
    t_image_processing_ms = semantic_processing_latency_ms(semantic_type, cfg)
    t_channel_decode_ms = float(cfg.channel_decode_ms)
    t_answer_reasoning_ms = float(cfg.answer_reasoning_ms)

    uplink_branch_ms = t_image_processing_ms + t_com_ms + t_channel_decode_ms

    t_total_ms = (
        max(t_question_parser_ms, uplink_branch_ms)
        + t_answer_reasoning_ms
    )

    return {
        "t_question_parser_ms": t_question_parser_ms,
        "t_image_processing_ms": t_image_processing_ms,
        "t_com_ms": t_com_ms,
        "t_com_sec": t_com_sec,
        "t_channel_decode_ms": t_channel_decode_ms,
        "t_answer_reasoning_ms": t_answer_reasoning_ms,
        "t_uplink_branch_ms": uplink_branch_ms,
        "t_total_ms": t_total_ms,
    }
