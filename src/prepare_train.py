"""Materialize data/train from caption selections, downscaled for the 768 crop."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ImageOps import exif_transpose

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import IMAGE_SUFFIXES, load_config, validate_image_caption_pairs

logger = logging.getLogger("food_studio.prepare_train")

JPEG_QUALITY = 90


@dataclass(frozen=True)
class CaptionSelection:
    image_id: str
    source: str
    custom_caption: str | None
    include: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write included caption selections into data/train as downscaled JPEGs."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser.parse_args()


def scaled_size(width: int, height: int, short_edge: int) -> tuple[int, int]:
    """Return dimensions after a no-upscale short-side cap, preserving aspect ratio."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    if short_edge < 1:
        raise ValueError("short_edge must be positive")
    shortest = min(width, height)
    if shortest <= short_edge:
        return width, height
    scale = short_edge / shortest
    return max(1, round(width * scale)), max(1, round(height * scale))


def downscale_training_image(image: Image.Image, short_edge: int) -> Image.Image:
    image = exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    new_width, new_height = scaled_size(width, height, short_edge)
    if (new_width, new_height) == (width, height):
        return image
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def load_caption_selections(path: Path) -> list[CaptionSelection]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Caption selections not found: {path}. Review captions in the UI first."
        )
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    raw = data.get("selections") if isinstance(data, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} must contain a non-empty selections object")

    selections: list[CaptionSelection] = []
    for image_id, item in raw.items():
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError(f"{path} has an invalid selection id")
        if not isinstance(item, dict):
            raise ValueError(f"{path} selection {image_id!r} must be an object")
        source = item.get("source")
        if source not in {"codex", "grok", "custom"}:
            raise ValueError(f"{path} selection {image_id!r} has invalid source {source!r}")
        custom = item.get("customCaption")
        if custom is not None and not isinstance(custom, str):
            raise ValueError(f"{path} selection {image_id!r} customCaption must be a string or null")
        include = item.get("include")
        if not isinstance(include, bool):
            raise ValueError(f"{path} selection {image_id!r} include must be a boolean")
        selections.append(
            CaptionSelection(
                image_id=image_id,
                source=source,
                custom_caption=custom.strip() if isinstance(custom, str) else None,
                include=include,
            )
        )
    return selections


def find_source_image(source_dir: Path, image_id: str) -> Path:
    matches = [
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.stem == image_id
    ]
    if not matches:
        raise FileNotFoundError(f"No source image in {source_dir} for {image_id!r}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple source images for {image_id!r}: {names}")
    return matches[0]


def caption_text(selection: CaptionSelection, captions_dir: Path) -> str:
    if selection.source == "custom":
        if not selection.custom_caption:
            raise ValueError(f"Included image {selection.image_id!r} has an empty custom caption")
        return selection.custom_caption
    caption_path = captions_dir / selection.source / f"{selection.image_id}.txt"
    if not caption_path.is_file():
        raise FileNotFoundError(
            f"Included image {selection.image_id!r} is missing {caption_path}"
        )
    caption = caption_path.read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError(f"Empty caption file: {caption_path}")
    return caption


def clear_stem(train_dir: Path, stem: str) -> None:
    for path in train_dir.iterdir():
        if path.is_file() and path.stem == stem:
            path.unlink()


def write_training_copy(
    source_path: Path,
    destination_path: Path,
    short_edge: int,
    caption: str,
) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    with Image.open(source_path) as image:
        original_size = image.size
        resized = downscale_training_image(image, short_edge)
        resized.save(destination_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        written_size = resized.size
    destination_path.with_suffix(".txt").write_text(f"{caption}\n", encoding="utf-8")
    return original_size, written_size, source_path.stat().st_size, destination_path.stat().st_size


def prepare_train_dir(
    source_dir: Path,
    train_dir: Path,
    selections_path: Path,
    short_edge: int,
) -> list[Path]:
    selections = [item for item in load_caption_selections(selections_path) if item.include]
    if not selections:
        raise ValueError(f"No included images in {selections_path}")

    train_dir.mkdir(parents=True, exist_ok=True)
    included_ids = {item.image_id for item in selections}
    captions_dir = selections_path.parent
    written: list[Path] = []
    source_bytes = 0
    train_bytes = 0

    logger.info(
        "Writing %s training copies to %s (short side <= %s, no upscale)",
        len(selections),
        train_dir,
        short_edge,
    )
    for selection in sorted(selections, key=lambda item: item.image_id):
        source_path = find_source_image(source_dir, selection.image_id)
        caption = caption_text(selection, captions_dir)
        clear_stem(train_dir, selection.image_id)
        destination = train_dir / f"{selection.image_id}.jpg"
        original_size, written_size, before, after = write_training_copy(
            source_path, destination, short_edge, caption
        )
        source_bytes += before
        train_bytes += after
        written.append(destination)
        logger.info(
            "%s  %sx%s %0.2fMB -> %sx%s %0.2fMB",
            selection.image_id,
            original_size[0],
            original_size[1],
            before / (1024 * 1024),
            written_size[0],
            written_size[1],
            after / (1024 * 1024),
        )

    for path in list(train_dir.iterdir()):
        if path.is_file() and path.stem not in included_ids:
            path.unlink()

    validate_image_caption_pairs(train_dir)
    logger.info(
        "Wrote %s pairs (%0.1fMB source -> %0.1fMB in %s)",
        len(written),
        source_bytes / (1024 * 1024),
        train_bytes / (1024 * 1024),
        train_dir,
    )
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    dataset: dict[str, Any] = config["dataset"]
    prepare_train_dir(
        source_dir=dataset["source_dir"],
        train_dir=dataset["train_dir"],
        selections_path=dataset["selections_path"],
        short_edge=dataset["train_short_edge"],
    )


if __name__ == "__main__":
    main()
