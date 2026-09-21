"""Config loading, dataset pair checks, and LoRA safety assertions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Copied from huggingface/diffusers@80c7ed262aeffbeb43ef13ae04baeb9b84515a69
# examples/dreambooth/train_dreambooth_lora_flux2_klein.py when --lora_layers is unset.
OFFICIAL_KLEIN_LORA_TARGET_MODULES: list[str] = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "to_qkv_mlp_proj",
    *[f"single_transformer_blocks.{i}.attn.to_out" for i in range(24)],
]


@dataclass(frozen=True)
class ImageCaptionPair:
    image_path: Path
    caption_path: Path
    caption: str


@dataclass(frozen=True)
class ValidationPrompt:
    prompt_id: str
    prompt: str
    seed: int


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{key} must be a mapping")
    return value


def _require_str(section: Mapping[str, Any], key: str, label: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _require_int(section: Mapping[str, Any], key: str, label: str, *, minimum: int = 1) -> int:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(section: Mapping[str, Any], key: str, label: str, *, minimum: float | None = None) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return number


def _require_bool(section: Mapping[str, Any], key: str, label: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _require_int_list(section: Mapping[str, Any], key: str, label: str) -> list[int]:
    value = section.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list of integers")
    layers = list(value)
    if not layers or any(not isinstance(item, int) or isinstance(item, bool) for item in layers):
        raise ValueError(f"{label} must be a non-empty list of integers")
    return layers


def load_config(config_path: Path) -> dict[str, Any]:
    """Load config.yaml, check required fields, and resolve relative paths."""
    path = config_path.expanduser().resolve()
    raw = load_yaml(path)

    model = _require_mapping(raw, "model")
    dataset = _require_mapping(raw, "dataset")
    output = _require_mapping(raw, "output")
    lora = _require_mapping(raw, "lora")
    training = _require_mapping(raw, "training")
    inference = _require_mapping(raw, "inference")

    model_id = _require_str(model, "id", "model.id")
    model_revision = _require_str(model, "revision", "model.revision")
    diffusers_revision = _require_str(raw, "diffusers_revision", "diffusers_revision")

    precision = _require_str(training, "precision", "training.precision").lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("training.precision must be one of: bf16, fp16, fp32")
    inference_precision = _require_str(inference, "precision", "inference.precision").lower()
    if inference_precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("inference.precision must be one of: bf16, fp16, fp32")

    resolution = _require_int(training, "resolution", "training.resolution")
    if resolution % 16 != 0:
        raise ValueError("training.resolution must be divisible by 16")
    width = _require_int(inference, "width", "inference.width")
    height = _require_int(inference, "height", "inference.height")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("inference width and height must be divisible by 16")

    config = {
        "model": {"id": model_id, "revision": model_revision},
        "diffusers_revision": diffusers_revision,
        "dataset": {"train_dir": resolve_path(path, _require_str(dataset, "train_dir", "dataset.train_dir"))},
        "output": {
            "checkpoints_dir": resolve_path(
                path, _require_str(output, "checkpoints_dir", "output.checkpoints_dir")
            ),
            "samples_dir": resolve_path(path, _require_str(output, "samples_dir", "output.samples_dir")),
            "comparisons_dir": resolve_path(
                path, _require_str(output, "comparisons_dir", "output.comparisons_dir")
            ),
            "observations_path": resolve_path(
                path, _require_str(output, "observations_path", "output.observations_path")
            ),
        },
        "lora": {
            "rank": _require_int(lora, "rank", "lora.rank"),
            "alpha": _require_int(lora, "alpha", "lora.alpha"),
            "dropout": _require_float(lora, "dropout", "lora.dropout", minimum=0.0),
            "adapter_scale": _require_float(lora, "adapter_scale", "lora.adapter_scale", minimum=0.0),
            "target_modules": list(OFFICIAL_KLEIN_LORA_TARGET_MODULES),
        },
        "training": {
            "seed": _require_int(training, "seed", "training.seed", minimum=0),
            "precision": precision,
            "resolution": resolution,
            "batch_size": _require_int(training, "batch_size", "training.batch_size"),
            "gradient_accumulation_steps": _require_int(
                training, "gradient_accumulation_steps", "training.gradient_accumulation_steps"
            ),
            "gradient_checkpointing": _require_bool(
                training, "gradient_checkpointing", "training.gradient_checkpointing"
            ),
            "cache_latents": _require_bool(training, "cache_latents", "training.cache_latents"),
            "cpu_offload": _require_bool(training, "cpu_offload", "training.cpu_offload"),
            "use_8bit_adam": _require_bool(training, "use_8bit_adam", "training.use_8bit_adam"),
            "learning_rate": _require_float(training, "learning_rate", "training.learning_rate", minimum=0.0),
            "lr_scheduler": _require_str(training, "lr_scheduler", "training.lr_scheduler"),
            "lr_warmup_steps": _require_int(training, "lr_warmup_steps", "training.lr_warmup_steps", minimum=0),
            "max_train_steps": _require_int(training, "max_train_steps", "training.max_train_steps"),
            "checkpoint_steps": _require_int(training, "checkpoint_steps", "training.checkpoint_steps"),
            "max_grad_norm": _require_float(training, "max_grad_norm", "training.max_grad_norm", minimum=0.0),
            "weighting_scheme": _require_str(training, "weighting_scheme", "training.weighting_scheme"),
            "max_sequence_length": _require_int(
                training, "max_sequence_length", "training.max_sequence_length"
            ),
            "text_encoder_out_layers": _require_int_list(
                training, "text_encoder_out_layers", "training.text_encoder_out_layers"
            ),
            "center_crop": _require_bool(training, "center_crop", "training.center_crop"),
            "dataloader_workers": _require_int(
                training, "dataloader_workers", "training.dataloader_workers", minimum=0
            ),
        },
        "inference": {
            "width": width,
            "height": height,
            "num_inference_steps": _require_int(
                inference, "num_inference_steps", "inference.num_inference_steps"
            ),
            "guidance_scale": _require_float(
                inference, "guidance_scale", "inference.guidance_scale", minimum=0.0
            ),
            "scheduler": _require_str(inference, "scheduler", "inference.scheduler"),
            "precision": inference_precision,
            "cpu_offload": _require_bool(inference, "cpu_offload", "inference.cpu_offload"),
            "validation_prompts": resolve_path(
                path, _require_str(inference, "validation_prompts", "inference.validation_prompts")
            ),
            "max_sequence_length": _require_int(
                inference, "max_sequence_length", "inference.max_sequence_length"
            ),
            "text_encoder_out_layers": _require_int_list(
                inference, "text_encoder_out_layers", "inference.text_encoder_out_layers"
            ),
        },
        "config_path": path,
    }
    return config


def validate_image_caption_pairs(train_dir: Path) -> list[ImageCaptionPair]:
    """Require a 1:1 image/.txt pairing with non-empty UTF-8 captions."""
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}. Put image+caption pairs in this folder."
        )

    images = sorted(
        path
        for path in train_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    captions = sorted(
        path for path in train_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    )

    image_stems = {path.stem: path for path in images}
    caption_stems = {path.stem: path for path in captions}

    missing_captions = sorted(stem for stem in image_stems if stem not in caption_stems)
    missing_images = sorted(stem for stem in caption_stems if stem not in image_stems)
    problems: list[str] = []
    if missing_captions:
        problems.append(
            "Images without a matching .txt caption: " + ", ".join(missing_captions)
        )
    if missing_images:
        problems.append(
            "Captions without a matching image: " + ", ".join(missing_images)
        )

    pairs: list[ImageCaptionPair] = []
    empty_captions: list[str] = []
    for stem in sorted(image_stems.keys() & caption_stems.keys()):
        caption_path = caption_stems[stem]
        caption = caption_path.read_text(encoding="utf-8").strip()
        if not caption:
            empty_captions.append(caption_path.name)
            continue
        pairs.append(
            ImageCaptionPair(
                image_path=image_stems[stem],
                caption_path=caption_path,
                caption=caption,
            )
        )
    if empty_captions:
        problems.append("Empty captions: " + ", ".join(empty_captions))
    if problems:
        raise ValueError(
            "Malformed training dataset in "
            f"{train_dir}. Each image needs exactly one non-empty UTF-8 .txt with the same stem.\n"
            + "\n".join(problems)
        )
    if not pairs:
        raise ValueError(
            f"No image/caption pairs found in {train_dir}. "
            "Add files such as 0001.jpg and 0001.txt before training."
        )
    return pairs


def load_validation_prompts(path: Path) -> list[ValidationPrompt]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation prompt file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or "prompts" not in data:
        raise ValueError(f"{path} must be a JSON object with a 'prompts' list")
    raw_prompts = data["prompts"]
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError(f"{path} must contain a non-empty prompts list")

    prompts: list[ValidationPrompt] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_prompts):
        if not isinstance(item, dict):
            raise ValueError(f"prompts[{index}] must be an object")
        prompt_id = item.get("id")
        prompt = item.get("prompt")
        seed = item.get("seed")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompts[{index}].id must be a non-empty string")
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate validation prompt id: {prompt_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompts[{index}].prompt must be a non-empty string")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"prompts[{index}].seed must be an integer")
        seen_ids.add(prompt_id)
        prompts.append(ValidationPrompt(prompt_id=prompt_id.strip(), prompt=prompt.strip(), seed=seed))
    return prompts


def assert_lora_only_trainable(
    transformer_named_params: Iterable[tuple[str, Any]],
    text_encoder_named_params: Iterable[tuple[str, Any]],
    vae_named_params: Iterable[tuple[str, Any]],
) -> list[str]:
    """Fail unless the only trainable tensors are LoRA adapter weights."""
    text_trainable = [name for name, param in text_encoder_named_params if param.requires_grad]
    vae_trainable = [name for name, param in vae_named_params if param.requires_grad]
    if text_trainable:
        raise RuntimeError(
            "Text encoder parameters are trainable; freeze the text encoder. "
            f"Examples: {text_trainable[:5]}"
        )
    if vae_trainable:
        raise RuntimeError(
            f"VAE parameters are trainable; freeze the VAE. Examples: {vae_trainable[:5]}"
        )

    trainable = [name for name, param in transformer_named_params if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. LoRA adapters were not applied to the transformer.")
    non_lora = [name for name in trainable if "lora" not in name.lower()]
    if non_lora:
        raise RuntimeError(
            "Trainable parameters include original transformer weights. "
            "Only LoRA adapter tensors should require grad. "
            f"Examples: {non_lora[:8]}"
        )
    return trainable


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
