from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageBaselineConfig:
    height: int = 480
    width: int = 320
    channels: int = 3
    bits_per_value: int = 32


def raw_float32_image_bits(
    height: int = 480,
    width: int = 320,
    channels: int = 3,
    bits_per_value: int = 32,
) -> int:
    """
    Paper-style raw image baseline:
        image_bits = bits_per_value * H * W * C

    For H=480, W=320, C=3, float32:
        32 * 480 * 320 * 3 = 14,745,600 bits
    """
    return int(bits_per_value * height * width * channels)


def image_file_size_bits(image_path: str | Path) -> int:
    """
    Optional JPEG/PNG file-size baseline.
    Use this only when actual image files are available.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing image file: {path}")
    return int(path.stat().st_size * 8)


def compression_ratio(image_bits: float, semantic_bits: float) -> float:
    """
    Ratio > 1 means image transmission is larger than semantic transmission.
    """
    if semantic_bits <= 0:
        return 0.0
    return float(image_bits / semantic_bits)

