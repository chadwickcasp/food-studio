"""Generate matched baseline/LoRA samples and side-by-side comparison grids."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont

from src.experiment import (
    ValidationPrompt,
    current_render_settings,
    load_config,
    load_json_object,
    load_validation_prompts,
    render_request_matches,
    repo_relative,
    sample_record_matches,
    write_json,
)

logger = logging.getLogger("food_studio.inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched FLUX.2 Klein evaluation.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Prompt JSON to render. Defaults to inference.validation_prompts from the config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit sample directory. Required by the development-checkpoint runner.",
    )
    parser.add_argument(
        "--lora",
        type=Path,
        default=None,
        help="Directory or safetensors file for the trained adapter. Omit for the baseline.",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Build comparison grids from existing baseline and LoRA sample folders.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Keep an existing image when the saved manifest matches its prompt, seed, "
            "inference settings, and adapter. Render every other sample."
        ),
    )
    return parser.parse_args()


def torch_dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pipeline(config: dict, lora_path: Path | None):
    # The pinned Diffusers checkout is installed on the training VM. Importing it
    # here keeps local unit tests able to load this module without that package.
    from diffusers import Flux2KleinPipeline

    model = config["model"]
    inference = config["inference"]
    dtype = torch_dtype(inference["precision"])
    device = pick_device()
    if inference["precision"] == "bf16" and device.type == "mps":
        raise ValueError("bf16 is not supported on MPS. Use CUDA or set inference.precision to fp16.")

    pipeline = Flux2KleinPipeline.from_pretrained(
        model["id"],
        revision=model["revision"],
        torch_dtype=dtype,
    )
    if lora_path is not None:
        resolved = lora_path.expanduser().resolve()
        if resolved.is_file():
            pipeline.load_lora_weights(str(resolved.parent), weight_name=resolved.name)
        else:
            pipeline.load_lora_weights(str(resolved))
        scale = config["lora"]["adapter_scale"]
        if scale != 1.0:
            pipeline.set_adapters(["default"], adapter_weights=[scale])
    if inference["cpu_offload"] and device.type == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline, device


def sample_dir(config: dict, lora_path: Path | None) -> Path:
    root: Path = config["output"]["samples_dir"]
    return root / ("lora" if lora_path is not None else "baseline")


def generate_samples(
    config: dict,
    lora_path: Path | None,
    *,
    prompts_path: Path | None = None,
    output_dir: Path | None = None,
    reuse_existing: bool = False,
) -> Path:
    prompt_file = prompts_path or config["inference"]["validation_prompts"]
    prompts = load_validation_prompts(prompt_file)
    out_dir = output_dir.expanduser().resolve() if output_dir is not None else sample_dir(config, lora_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    run_start = perf_counter()
    inference = config["inference"]
    current = current_render_settings(config, lora_path)
    records = []
    previous_manifest = load_json_object(out_dir / "manifest.json") if reuse_existing else None
    if previous_manifest is None:
        previous_manifest = {}
    previous_samples = previous_manifest.get("samples", [])
    if not isinstance(previous_samples, list):
        previous_samples = []
    previous_records = {
        record.get("id"): record
        for record in previous_samples
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    }
    request_matches = bool(previous_manifest) and render_request_matches(previous_manifest, current)

    def existing_sample_matches(item: ValidationPrompt) -> bool:
        image_path = out_dir / f"{item.prompt_id}.png"
        record = previous_records.get(item.prompt_id)
        return request_matches and image_path.is_file() and sample_record_matches(record, item)

    to_generate = [item for item in prompts if not existing_sample_matches(item)]
    pipeline = None
    device = None
    if to_generate:
        pipeline, device = load_pipeline(config, lora_path)

    for item in prompts:
        image_path = out_dir / f"{item.prompt_id}.png"
        reused = existing_sample_matches(item)
        if reused:
            logger.info("Reusing %s (seed=%s) -> %s", item.prompt_id, item.seed, image_path)
            sample_runtime = previous_records[item.prompt_id].get("runtime_seconds")
        else:
            logger.info("Generating %s (seed=%s) -> %s", item.prompt_id, item.seed, image_path)
            generator = torch.Generator(device="cpu").manual_seed(item.seed)
            sample_start = perf_counter()
            image = pipeline(
                prompt=item.prompt,
                height=inference["height"],
                width=inference["width"],
                num_inference_steps=inference["num_inference_steps"],
                guidance_scale=inference["guidance_scale"],
                generator=generator,
                max_sequence_length=inference["max_sequence_length"],
                text_encoder_out_layers=tuple(inference["text_encoder_out_layers"]),
            ).images[0]
            image.save(image_path)
            sample_runtime = round(perf_counter() - sample_start, 3)
        records.append(
            {
                "id": item.prompt_id,
                "prompt_group_id": item.prompt_group_id,
                "prompt": item.prompt,
                "seed": item.seed,
                "image": repo_relative(image_path),
                "runtime_seconds": sample_runtime,
                "reused_existing": reused,
            }
        )

    if device is None:
        gpu_name = previous_manifest.get("gpu_name")
    elif device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = None
    manifest = {
        "created_utc": started.isoformat(),
        "prompts_path": str(prompt_file),
        "device": str(device) if device is not None else previous_manifest.get("device"),
        "gpu_name": gpu_name,
        "runtime_seconds": round(perf_counter() - run_start, 3),
        "generated_sample_count": len(to_generate),
        "reused_sample_count": len(prompts) - len(to_generate),
        **current,
        "negative_prompt": None,
        "samples": records,
    }
    write_json(out_dir / "manifest.json", manifest)
    logger.info("Wrote %s samples and manifest to %s", len(records), out_dir)
    return out_dir


def _label_bar(width: int, text: str, height: int = 36) -> Image.Image:
    bar = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(bar)
    font = ImageFont.load_default()
    draw.text((10, 10), text, fill=(240, 240, 240), font=font)
    return bar


def write_pair(baseline: Image.Image, lora: Image.Image, title: str, path: Path) -> Image.Image:
    height = max(baseline.height, lora.height)
    left = baseline.resize((int(baseline.width * height / baseline.height), height))
    right = lora.resize((int(lora.width * height / lora.height), height))
    gap = 8
    canvas = Image.new("RGB", (left.width + right.width + gap, height + 36), (8, 8, 8))
    header = f"{title}    left=baseline    right=lora"
    canvas.paste(_label_bar(canvas.width, header), (0, 0))
    canvas.paste(left, (0, 36))
    canvas.paste(right, (left.width + gap, 36))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return canvas


def write_comparisons(config: dict) -> Path:
    prompts = load_validation_prompts(config["inference"]["validation_prompts"])
    baseline_dir = config["output"]["samples_dir"] / "baseline"
    lora_dir = config["output"]["samples_dir"] / "lora"
    out_dir: Path = config["output"]["comparisons_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    pair_images: list[Image.Image] = []
    for item in prompts:
        baseline_path = baseline_dir / f"{item.prompt_id}.png"
        lora_path = lora_dir / f"{item.prompt_id}.png"
        if not baseline_path.is_file() or not lora_path.is_file():
            missing.append(item.prompt_id)
            continue
        with Image.open(baseline_path) as baseline, Image.open(lora_path) as lora:
            pair_path = out_dir / f"{item.prompt_id}.png"
            pair_images.append(
                write_pair(
                    baseline.convert("RGB"),
                    lora.convert("RGB"),
                    f"{item.prompt_id}  seed={item.seed}",
                    pair_path,
                )
            )

    if missing:
        raise FileNotFoundError(
            "Missing matched sample images for: "
            + ", ".join(missing)
            + f". Expected files in {baseline_dir} and {lora_dir}."
        )

    sheet_width = max(image.width for image in pair_images)
    sheet_height = sum(image.height for image in pair_images) + 8 * (len(pair_images) - 1)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (8, 8, 8))
    y = 0
    for image in pair_images:
        sheet.paste(image, (0, y))
        y += image.height + 8
    contact_path = out_dir / "contact_sheet.png"
    sheet.save(contact_path)
    write_json(
        out_dir / "manifest.json",
        {
            "baseline_dir": str(baseline_dir),
            "lora_dir": str(lora_dir),
            "pairs": [item.prompt_id for item in prompts],
            "contact_sheet": str(contact_path),
        },
    )
    logger.info("Wrote comparisons to %s", out_dir)
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    prompt_file = args.prompts.expanduser().resolve() if args.prompts else config["inference"]["validation_prompts"]
    load_validation_prompts(prompt_file)

    if args.compare_only:
        if args.prompts is not None or args.output_dir is not None or args.reuse_existing:
            raise ValueError("--compare-only uses the configured validation prompts and default sample directories.")
        write_comparisons(config)
        return

    generate_samples(
        config,
        args.lora,
        prompts_path=prompt_file,
        output_dir=args.output_dir,
        reuse_existing=args.reuse_existing,
    )
    baseline_dir = config["output"]["samples_dir"] / "baseline"
    lora_dir = config["output"]["samples_dir"] / "lora"
    using_default_directories = args.prompts is None and args.output_dir is None
    both_sample_dirs_exist = baseline_dir.is_dir() and lora_dir.is_dir()
    if using_default_directories and args.lora is not None and both_sample_dirs_exist:
        write_comparisons(config)


if __name__ == "__main__":
    main()
