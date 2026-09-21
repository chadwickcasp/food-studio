"""Train a transformer-only LoRA on FLUX.2 Klein Base 4B.

The training recipe and default LoRA target modules follow the official Diffusers
example at huggingface/diffusers@80c7ed262aeffbeb43ef13ae04baeb9b84515a69
examples/dreambooth/train_dreambooth_lora_flux2_klein.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    offload_models,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module

from src.experiment import (
    ImageCaptionPair,
    assert_lora_only_trainable,
    load_config,
    validate_image_caption_pairs,
    write_json,
)

logger = logging.getLogger("food_studio.train")


@dataclass
class Models:
    tokenizer: Any
    noise_scheduler: Any
    noise_scheduler_copy: Any
    vae: Any
    transformer: Any
    text_encoder: Any


@dataclass
class StepCaches:
    latents: list[torch.Tensor | None]
    prompts: list[torch.Tensor | None]
    text_ids: list[torch.Tensor | None]
    bn_mean: torch.Tensor
    bn_std: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a FLUX.2 Klein transformer LoRA.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help='Resume from a checkpoint directory, or "latest".',
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reject_unsupported_precision(precision: str) -> None:
    if precision == "bf16" and torch.backends.mps.is_available() and not torch.cuda.is_available():
        raise ValueError("bf16 training is not supported on MPS. Use a CUDA GPU or set training.precision to fp16.")


def preprocess_image(image: Image.Image, size: int, center_crop: bool) -> torch.Tensor:
    image = exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    scale = max(size / height, size / width)
    image = TF.resize(
        image,
        [round(height * scale), round(width * scale)],
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    if center_crop:
        image = TF.center_crop(image, [size, size])
    else:
        top, left, crop_h, crop_w = transforms.RandomCrop.get_params(image, output_size=(size, size))
        image = TF.crop(image, top, left, crop_h, crop_w)
    return TF.normalize(TF.to_tensor(image), [0.5], [0.5])


class CaptionImageDataset(Dataset):
    def __init__(self, pairs: list[ImageCaptionPair], size: int, center_crop: bool) -> None:
        self.pairs = pairs
        self.size = size
        self.center_crop = center_crop

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        pair = self.pairs[index]
        with Image.open(pair.image_path) as image:
            pixel_values = preprocess_image(image, self.size, self.center_crop)
        return {"index": index, "pixel_values": pixel_values, "caption": pair.caption}


def collate(examples: list[dict[str, torch.Tensor | str | int]]) -> dict[str, torch.Tensor | list[str] | list[int]]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {
        "indices": [int(example["index"]) for example in examples],
        "pixel_values": pixel_values,
        "captions": [str(example["caption"]) for example in examples],
    }


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def save_lora(transformer: torch.nn.Module, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    Flux2KleinPipeline.save_lora_weights(
        str(output_dir),
        transformer_lora_layers=get_peft_model_state_dict(transformer),
        **_collate_lora_metadata({"transformer": transformer}),
    )
    weights_path = output_dir / "pytorch_lora_weights.safetensors"
    if not weights_path.is_file():
        raise RuntimeError(f"Expected LoRA weights at {weights_path}")
    return weights_path


def reload_lora(transformer: torch.nn.Module, lora_config: LoraConfig, weights_dir: Path) -> None:
    has_adapter = hasattr(transformer, "peft_config") and "default" in getattr(transformer, "peft_config", {})
    if not has_adapter:
        transformer.add_adapter(lora_config)
    state = Flux2KleinPipeline.lora_state_dict(str(weights_dir))
    transformer_state = {
        key.replace("transformer.", ""): value
        for key, value in state.items()
        if key.startswith("transformer.")
    }
    incompatible = set_peft_model_state_dict(
        transformer,
        convert_unet_state_dict_to_peft(transformer_state),
        adapter_name="default",
    )
    unexpected = getattr(incompatible, "unexpected_keys", None)
    if unexpected:
        raise RuntimeError(f"LoRA reload produced unexpected keys: {unexpected}")


def latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    checkpoints = [path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda path: int(path.name.split("-")[1]))[-1]


def sigmas_for_timesteps(noise_scheduler, timesteps: torch.Tensor, n_dim: int, dtype: torch.dtype) -> torch.Tensor:
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule = noise_scheduler.timesteps.to(timesteps.device)
    step_indices = [(schedule == timestep).nonzero().item() for timestep in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def load_models(config: dict[str, Any], device: torch.device, weight_dtype: torch.dtype) -> Models:
    model_cfg = config["model"]
    offload = config["training"]["cpu_offload"]
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        model_cfg["id"], subfolder="tokenizer", revision=model_cfg["revision"]
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_cfg["id"], subfolder="scheduler", revision=model_cfg["revision"]
    )
    vae = AutoencoderKLFlux2.from_pretrained(
        model_cfg["id"], subfolder="vae", revision=model_cfg["revision"], torch_dtype=weight_dtype
    )
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_cfg["id"], subfolder="transformer", revision=model_cfg["revision"], torch_dtype=weight_dtype
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_cfg["id"], subfolder="text_encoder", revision=model_cfg["revision"], torch_dtype=weight_dtype
    )
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.to(dtype=weight_dtype, device="cpu" if offload else device)
    text_encoder.to(dtype=weight_dtype, device="cpu" if offload else device)
    transformer.to(device=device, dtype=weight_dtype)
    if config["training"]["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()
    return Models(
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        noise_scheduler_copy=copy.deepcopy(noise_scheduler),
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
    )


def attach_lora(models: Models, config: dict[str, Any]) -> LoraConfig:
    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        init_lora_weights="gaussian",
        target_modules=lora_cfg["target_modules"],
    )
    models.transformer.add_adapter(lora_config)
    trainable = assert_lora_only_trainable(
        models.transformer.named_parameters(),
        models.text_encoder.named_parameters(),
        models.vae.named_parameters(),
    )
    logger.info("Trainable LoRA parameters: %s", len(trainable))
    return lora_config


def make_dataloader(pairs: list[ImageCaptionPair], config: dict[str, Any], *, shuffle: bool) -> DataLoader:
    training = config["training"]
    dataset = CaptionImageDataset(pairs, training["resolution"], training["center_crop"])
    generator = torch.Generator().manual_seed(training["seed"]) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=training["dataloader_workers"],
        generator=generator,
    )


def cache_latents_and_prompts(
    models: Models,
    dataloader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    weight_dtype: torch.dtype,
    *,
    show_progress: bool,
) -> StepCaches:
    training = config["training"]
    text_pipeline = Flux2KleinPipeline.from_pretrained(
        config["model"]["id"],
        vae=None,
        transformer=None,
        tokenizer=models.tokenizer,
        text_encoder=models.text_encoder,
        scheduler=None,
        revision=config["model"]["revision"],
    )
    bn_mean = models.vae.bn.running_mean.view(1, -1, 1, 1).detach().to(dtype=weight_dtype)
    bn_std = torch.sqrt(models.vae.bn.running_var.view(1, -1, 1, 1) + models.vae.config.batch_norm_eps).detach().to(
        dtype=weight_dtype
    )
    n = len(dataloader.dataset)
    caches = StepCaches(
        latents=[None] * n,
        prompts=[None] * n,
        text_ids=[None] * n,
        bn_mean=bn_mean,
        bn_std=bn_std,
    )
    for batch in tqdm(dataloader, desc="Caching latents and prompts", disable=not show_progress):
        with torch.no_grad():
            if training["cache_latents"]:
                with offload_models(models.vae, device=device, offload=training["cpu_offload"]):
                    pixels = batch["pixel_values"].to(device=device, dtype=models.vae.dtype, non_blocking=True)
                    latents = models.vae.encode(pixels).latent_dist.mode()
                for offset, index in enumerate(batch["indices"]):
                    caches.latents[index] = latents[offset : offset + 1].detach().cpu()
            with offload_models(text_pipeline, device=device, offload=training["cpu_offload"]):
                prompt_embeds, text_ids = text_pipeline.encode_prompt(
                    prompt=batch["captions"],
                    max_sequence_length=training["max_sequence_length"],
                    text_encoder_out_layers=tuple(training["text_encoder_out_layers"]),
                )
            for offset, index in enumerate(batch["indices"]):
                caches.prompts[index] = prompt_embeds[offset : offset + 1].detach().cpu()
                caches.text_ids[index] = text_ids[offset : offset + 1].detach().cpu()
    if training["cache_latents"] and any(item is None for item in caches.latents):
        raise RuntimeError("VAE latent cache is incomplete.")
    if any(item is None for item in caches.prompts) or any(item is None for item in caches.text_ids):
        raise RuntimeError("Prompt embedding cache is incomplete.")
    if training["cache_latents"]:
        models.vae.to("cpu")
        models.vae = None
    text_pipeline.to("cpu")
    models.text_encoder = None
    models.tokenizer = None
    free_memory()
    return caches


def make_optimizer(transformer: torch.nn.Module, config: dict[str, Any], accelerator: Accelerator):
    training = config["training"]
    params = [param for param in transformer.parameters() if param.requires_grad]
    if training["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("training.use_8bit_adam is true but bitsandbytes is not installed.") from exc
        optimizer = bnb.optim.AdamW8bit(params, lr=training["learning_rate"])
    else:
        optimizer = torch.optim.AdamW(params, lr=training["learning_rate"])
    scheduler_name = training["lr_scheduler"]
    if scheduler_name == "constant" and training["lr_warmup_steps"] > 0:
        scheduler_name = "constant_with_warmup"
    lr_scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=training["lr_warmup_steps"] * accelerator.num_processes,
        num_training_steps=training["max_train_steps"] * accelerator.num_processes,
    )
    return optimizer, lr_scheduler


def register_checkpoint_hooks(accelerator: Accelerator, lora_config: LoraConfig) -> None:
    def save_model_hook(models, weights, output_dir) -> None:
        if not accelerator.is_main_process:
            if weights:
                weights.clear()
            return
        saved = False
        for model in models:
            unwrapped = unwrap_model(accelerator, model)
            if isinstance(unwrapped, Flux2Transformer2DModel) or hasattr(unwrapped, "peft_config"):
                save_lora(unwrapped, Path(output_dir))
                saved = True
        if not saved:
            raise ValueError("No LoRA transformer found while saving a checkpoint.")
        if weights:
            weights.clear()

    def load_model_hook(models, input_dir) -> None:
        while models:
            unwrapped = unwrap_model(accelerator, models.pop())
            reload_lora(unwrapped, lora_config, Path(input_dir))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def maybe_resume(accelerator: Accelerator, checkpoints_dir: Path, resume: str | None) -> int:
    if not resume:
        return 0
    resume_path = latest_checkpoint(checkpoints_dir) if resume == "latest" else Path(resume)
    if resume_path is None:
        logger.info("No checkpoint found; starting a new run.")
        return 0
    accelerator.load_state(str(resume_path))
    global_step = int(resume_path.name.split("-")[1])
    logger.info("Resumed from %s", resume_path)
    return global_step


def batch_loss(
    batch: dict[str, Any],
    models: Models,
    caches: StepCaches,
    config: dict[str, Any],
    accelerator: Accelerator,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    training = config["training"]
    indices = batch["indices"]
    prompt_embeds = torch.cat([caches.prompts[index] for index in indices], dim=0).to(device=device, dtype=weight_dtype)
    text_ids = torch.cat([caches.text_ids[index] for index in indices], dim=0).to(device=device)
    if training["cache_latents"]:
        model_input = torch.cat([caches.latents[index] for index in indices], dim=0).to(
            device=device, dtype=weight_dtype
        )
    else:
        with offload_models(models.vae, device=device, offload=training["cpu_offload"]):
            pixels = batch["pixel_values"].to(device=device, dtype=models.vae.dtype)
            model_input = models.vae.encode(pixels).latent_dist.mode()

    model_input = Flux2KleinPipeline._patchify_latents(model_input)
    mean = caches.bn_mean.to(device=model_input.device, dtype=model_input.dtype)
    std = caches.bn_std.to(device=model_input.device, dtype=model_input.dtype)
    model_input = (model_input - mean) / std
    model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(model_input.device)
    noise = torch.randn_like(model_input)
    u = compute_density_for_timestep_sampling(
        weighting_scheme=training["weighting_scheme"],
        batch_size=model_input.shape[0],
        device=model_input.device,
    )
    timestep_indices = (u * models.noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = models.noise_scheduler_copy.timesteps[timestep_indices].to(model_input.device)
    sigmas = sigmas_for_timesteps(models.noise_scheduler_copy, timesteps, model_input.ndim, model_input.dtype)
    packed = Flux2KleinPipeline._pack_latents((1.0 - sigmas) * model_input + sigmas * noise)

    unwrapped = unwrap_model(accelerator, models.transformer)
    guidance = None
    if getattr(unwrapped.config, "guidance_embeds", False):
        guidance = torch.full((model_input.shape[0],), 1.0, device=device, dtype=weight_dtype)
    model_pred = models.transformer(
        hidden_states=packed,
        timestep=timesteps / 1000,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=model_input_ids,
        return_dict=False,
    )[0]
    model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred[:, : packed.size(1)], model_input_ids)
    weighting = compute_loss_weighting_for_sd3(training["weighting_scheme"], sigmas=sigmas)
    target = noise - model_input
    return torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    ).mean()


def run_training_loop(
    models: Models,
    caches: StepCaches,
    dataloader: DataLoader,
    optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    config: dict[str, Any],
    checkpoints_dir: Path,
    global_step: int,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> int:
    training = config["training"]
    progress = tqdm(
        range(training["max_train_steps"]),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    log_path = checkpoints_dir / "train_log.jsonl"
    models.transformer.train()
    while global_step < training["max_train_steps"]:
        for batch in dataloader:
            with accelerator.accumulate(models.transformer):
                loss = batch_loss(batch, models, caches, config, accelerator, device, weight_dtype)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(models.transformer.parameters(), training["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                logs = {"loss": float(loss.detach().item()), "lr": lr_scheduler.get_last_lr()[0]}
                progress.set_postfix(**logs)
                if accelerator.is_main_process:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"step": global_step, **logs}) + "\n")
                    if global_step % training["checkpoint_steps"] == 0:
                        save_path = checkpoints_dir / f"checkpoint-{global_step}"
                        accelerator.save_state(str(save_path))
                        logger.info("Saved checkpoint to %s", save_path)
                if global_step >= training["max_train_steps"]:
                    break
        if global_step >= training["max_train_steps"]:
            break
    return global_step


def save_final_adapter(
    accelerator: Accelerator,
    models: Models,
    config: dict[str, Any],
    pairs: list[ImageCaptionPair],
    global_step: int,
    weight_dtype: torch.dtype,
) -> None:
    if not accelerator.is_main_process:
        return
    final_dir = config["output"]["checkpoints_dir"] / "final"
    unwrapped = unwrap_model(accelerator, models.transformer).to(weight_dtype)
    weights_path = save_lora(unwrapped, final_dir)
    write_json(
        final_dir / "training_manifest.json",
        {
            "model_id": config["model"]["id"],
            "model_revision": config["model"]["revision"],
            "diffusers_revision": config["diffusers_revision"],
            "lora": {
                "rank": config["lora"]["rank"],
                "alpha": config["lora"]["alpha"],
                "dropout": config["lora"]["dropout"],
                "weights": str(weights_path),
            },
            "training": {
                **{key: value for key, value in config["training"].items() if key != "text_encoder_out_layers"},
                "text_encoder_out_layers": list(config["training"]["text_encoder_out_layers"]),
            },
            "dataset_size": len(pairs),
            "steps": global_step,
        },
    )
    logger.info("Saved final LoRA weights to %s", weights_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    pairs = validate_image_caption_pairs(config["dataset"]["train_dir"])
    logger.info("Validated %s image/caption pairs in %s", len(pairs), config["dataset"]["train_dir"])
    reject_unsupported_precision(training["precision"])

    checkpoints_dir: Path = config["output"]["checkpoints_dir"]
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        mixed_precision=training["precision"] if training["precision"] != "fp32" else "no",
        project_config=ProjectConfiguration(project_dir=str(checkpoints_dir)),
    )
    set_seed(training["seed"])
    weight_dtype = torch_dtype(training["precision"])
    device = accelerator.device

    models = load_models(config, device, weight_dtype)
    lora_config = attach_lora(models, config)
    caches = cache_latents_and_prompts(
        models,
        make_dataloader(pairs, config, shuffle=False),
        config,
        device,
        weight_dtype,
        show_progress=accelerator.is_local_main_process,
    )
    optimizer, lr_scheduler = make_optimizer(models.transformer, config, accelerator)
    train_loader = make_dataloader(pairs, config, shuffle=True)
    models.transformer, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        models.transformer, optimizer, train_loader, lr_scheduler
    )
    register_checkpoint_hooks(accelerator, lora_config)
    global_step = maybe_resume(accelerator, checkpoints_dir, args.resume)
    global_step = run_training_loop(
        models,
        caches,
        train_loader,
        optimizer,
        lr_scheduler,
        accelerator,
        config,
        checkpoints_dir,
        global_step,
        device,
        weight_dtype,
    )
    accelerator.wait_for_everyone()
    save_final_adapter(accelerator, models, config, pairs, global_step, weight_dtype)


if __name__ == "__main__":
    main()
