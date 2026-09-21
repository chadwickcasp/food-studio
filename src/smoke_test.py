"""Save and reload a LoRA adapter before a full training run."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft.utils import get_peft_model_state_dict

from src.experiment import assert_lora_only_trainable, load_config, validate_image_caption_pairs, write_json
from src.train import attach_lora, load_models, pick_device, reject_unsupported_precision, reload_lora, save_lora, torch_dtype

logger = logging.getLogger("food_studio.smoke_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test FLUX.2 Klein LoRA save and reload.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    pairs = validate_image_caption_pairs(config["dataset"]["train_dir"])
    logger.info("Validated %s image/caption pairs", len(pairs))
    reject_unsupported_precision(config["training"]["precision"])

    device = pick_device()
    weight_dtype = torch_dtype(config["training"]["precision"])
    models = load_models(config, device, weight_dtype)
    lora_config = attach_lora(models, config)

    smoke_dir = config["output"]["checkpoints_dir"] / "smoke"
    saved_state = get_peft_model_state_dict(models.transformer)
    probe_key = next(iter(saved_state))
    original = saved_state[probe_key].detach().cpu().clone()
    save_lora(models.transformer, smoke_dir)
    with torch.no_grad():
        for param in models.transformer.parameters():
            if param.requires_grad:
                param.zero_()
    reload_lora(models.transformer, lora_config, smoke_dir)
    restored = get_peft_model_state_dict(models.transformer)[probe_key].detach().cpu()
    if not torch.allclose(original, restored):
        raise RuntimeError("LoRA reload did not restore the saved adapter weights.")
    trainable = assert_lora_only_trainable(
        models.transformer.named_parameters(),
        models.text_encoder.named_parameters(),
        models.vae.named_parameters(),
    )
    write_json(
        smoke_dir / "smoke_test.json",
        {
            "ok": True,
            "pairs": len(pairs),
            "trainable_parameter_count": len(trainable),
            "weights": str(smoke_dir / "pytorch_lora_weights.safetensors"),
            "diffusers_revision": config["diffusers_revision"],
            "model_revision": config["model"]["revision"],
        },
    )
    logger.info("Smoke test passed. Adapter saved and reloaded from %s", smoke_dir)


if __name__ == "__main__":
    main()
