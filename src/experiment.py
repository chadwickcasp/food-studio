"""Config loading, dataset pair checks, and LoRA safety assertions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
RENDER_MANIFEST_FIELDS = (
    "model_id",
    "model_revision",
    "diffusers_revision",
    "adapter_scale",
    "width",
    "height",
    "num_inference_steps",
    "guidance_scale",
    "scheduler",
    "precision",
    "max_sequence_length",
    "text_encoder_out_layers",
)
ROOT = Path(__file__).resolve().parents[1]

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
    prompt_group_id: str
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

    train_short_edge = _require_int(dataset, "train_short_edge", "dataset.train_short_edge")
    if train_short_edge < resolution:
        raise ValueError("dataset.train_short_edge must be >= training.resolution")

    train_dir = resolve_path(path, _require_str(dataset, "train_dir", "dataset.train_dir"))
    config = {
        "model": {"id": model_id, "revision": model_revision},
        "diffusers_revision": diffusers_revision,
        "dataset": {
            "train_dir": train_dir,
            "train_short_edge": train_short_edge,
            "selections_path": train_dir.parent / "captions" / "selections.json",
            "source_dir": train_dir.parent,
        },
        "output": {
            "checkpoints_dir": resolve_path(
                path, _require_str(output, "checkpoints_dir", "output.checkpoints_dir")
            ),
            "development_dir": resolve_path(
                path, _require_str(output, "development_dir", "output.development_dir")
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
            "development_prompts": resolve_path(
                path, _require_str(inference, "development_prompts", "inference.development_prompts")
            ),
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
    seen_group_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    for index, item in enumerate(raw_prompts):
        if not isinstance(item, dict):
            raise ValueError(f"prompts[{index}] must be an object")
        prompt_id = item.get("id")
        prompt = item.get("prompt")
        seed = item.get("seed")
        seeds = item.get("seeds")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompts[{index}].id must be a non-empty string")
        prompt_id = prompt_id.strip()
        if prompt_id in seen_group_ids:
            raise ValueError(f"Duplicate validation prompt id: {prompt_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompts[{index}].prompt must be a non-empty string")
        if seed is not None and seeds is not None:
            raise ValueError(f"prompts[{index}] must define either 'seed' or 'seeds', not both")
        if seeds is None:
            seeds = [seed]
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(not isinstance(value, int) or isinstance(value, bool) for value in seeds)
        ):
            raise ValueError(f"prompts[{index}].seeds must be a non-empty list of integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"prompts[{index}].seeds must not contain duplicates")

        seen_group_ids.add(prompt_id)
        for seed_index, seed_value in enumerate(seeds):
            # Preserve the original sample ID and filename for the first seed so an
            # expanded development run can reuse already-rendered checkpoint images.
            sample_id = prompt_id if seed_index == 0 else f"{prompt_id}-seed-{seed_value}"
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate prompt sample id: {sample_id}")
            seen_sample_ids.add(sample_id)
            prompts.append(
                ValidationPrompt(
                    prompt_id=sample_id,
                    prompt_group_id=prompt_id,
                    prompt=prompt.strip(),
                    seed=seed_value,
                )
            )
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def lora_weights_file(lora_path: Path | None) -> Path | None:
    if lora_path is None:
        return None
    resolved = lora_path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    weights = resolved / LORA_WEIGHTS_NAME
    if not weights.is_file():
        raise FileNotFoundError(f"No LoRA weights file in {resolved}. Expected {LORA_WEIGHTS_NAME}.")
    return weights


def current_render_settings(config: Mapping[str, Any], lora_path: Path | None) -> dict[str, Any]:
    """Inference settings and adapter identity recorded on a sample manifest."""
    inference = config["inference"]
    weights = lora_weights_file(lora_path)
    if lora_path is None:
        adapter_path = None
        adapter_scale = None
    else:
        adapter_path = str(lora_path.expanduser().resolve())
        adapter_scale = config["lora"]["adapter_scale"]
    return {
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "diffusers_revision": config["diffusers_revision"],
        "adapter_path": adapter_path,
        "adapter_scale": adapter_scale,
        "adapter_sha256": None if weights is None else file_sha256(weights),
        "width": inference["width"],
        "height": inference["height"],
        "num_inference_steps": inference["num_inference_steps"],
        "guidance_scale": inference["guidance_scale"],
        "scheduler": inference["scheduler"],
        "precision": inference["precision"],
        "cpu_offload": inference["cpu_offload"],
        "max_sequence_length": inference["max_sequence_length"],
        "text_encoder_out_layers": list(inference["text_encoder_out_layers"]),
    }


def _outputs_suffix(path: str) -> str | None:
    parts = Path(path).parts
    if "outputs" not in parts:
        return None
    index = parts.index("outputs")
    return Path(*parts[index:]).as_posix()


def adapters_match(previous_path: object, current_path: object) -> bool:
    if previous_path is None or current_path is None:
        return previous_path is None and current_path is None
    if not isinstance(previous_path, str) or not isinstance(current_path, str):
        return False
    if previous_path == current_path:
        return True
    previous = Path(previous_path)
    current = Path(current_path)
    if previous.exists() and current.exists():
        return previous.resolve() == current.resolve()
    previous_suffix = _outputs_suffix(previous_path)
    current_suffix = _outputs_suffix(current_path)
    return previous_suffix is not None and previous_suffix == current_suffix


def render_request_matches(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """True when a saved manifest describes the render that is about to run."""
    for field in RENDER_MANIFEST_FIELDS:
        if field not in previous or previous[field] != current[field]:
            return False
    if not adapters_match(previous.get("adapter_path"), current.get("adapter_path")):
        return False
    if "cpu_offload" in previous and previous["cpu_offload"] != current["cpu_offload"]:
        return False
    # Older manifests have no adapter hash. They can match once; the rewrite records
    # adapter_sha256, and a later change to the weights file will miss.
    if "adapter_sha256" in previous and previous["adapter_sha256"] != current["adapter_sha256"]:
        return False
    return True


def sample_record_matches(record: Mapping[str, Any] | None, item: ValidationPrompt) -> bool:
    if not isinstance(record, Mapping):
        return False
    return (
        record.get("id") == item.prompt_id
        and record.get("prompt") == item.prompt
        and record.get("seed") == item.seed
    )


def saved_render_matches(
    output_dir: Path,
    prompts: Sequence[ValidationPrompt],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    lora_path: Path | None,
) -> bool:
    current = current_render_settings(config, lora_path)
    if not render_request_matches(manifest, current):
        return False
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        return False
    records = {record.get("id"): record for record in samples if isinstance(record, dict)}
    for item in prompts:
        image_path = output_dir / f"{item.prompt_id}.png"
        if not image_path.is_file():
            return False
        if not sample_record_matches(records.get(item.prompt_id), item):
            return False
    return True


def load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a JSON object")
    return loaded


def repo_relative(path: Path) -> str:
    """Return a checkout-relative POSIX path so manifests stay valid on another machine."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
