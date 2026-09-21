"""Render development prompts for Base and every saved LoRA checkpoint."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.experiment import load_config, load_validation_prompts, write_json

logger = logging.getLogger("food_studio.development")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the development set at each configured checkpoint for visual selection."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--name", default="Rank-8 initial run")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate complete development runs instead of reusing them.",
    )
    return parser.parse_args()


def checkpoint_steps(max_steps: int, interval: int) -> list[int]:
    if max_steps <= 0 or interval <= 0:
        raise ValueError("max_steps and interval must be positive")
    steps = list(range(interval, max_steps + 1, interval))
    if not steps or steps[-1] != max_steps:
        steps.append(max_steps)
    return steps


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "development"


def run_complete(output_dir: Path, prompt_ids: list[str]) -> bool:
    return (output_dir / "manifest.json").is_file() and all(
        (output_dir / f"{prompt_id}.png").is_file() for prompt_id in prompt_ids
    )


def render_run(
    *,
    config_path: Path,
    prompts_path: Path,
    output_dir: Path,
    prompt_ids: list[str],
    lora_path: Path | None,
    force: bool,
) -> None:
    if not force and run_complete(output_dir, prompt_ids):
        logger.info("Reusing complete development run in %s", output_dir)
        return
    command = [
        sys.executable,
        "-m",
        "src.inference",
        "--config",
        str(config_path),
        "--prompts",
        str(prompts_path),
        "--output-dir",
        str(output_dir),
    ]
    if lora_path is not None:
        command.extend(["--lora", str(lora_path)])
    subprocess.run(command, check=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    config_path: Path = config["config_path"]
    prompts_path: Path = config["inference"]["development_prompts"]
    prompts = load_validation_prompts(prompts_path)
    prompt_ids = [item.prompt_id for item in prompts]
    steps = checkpoint_steps(
        config["training"]["max_train_steps"],
        config["training"]["checkpoint_steps"],
    )
    checkpoints_dir: Path = config["output"]["checkpoints_dir"]
    missing = [step for step in steps if not (checkpoints_dir / f"checkpoint-{step}").is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing configured training checkpoints: " + ", ".join(str(step) for step in missing)
        )

    experiment_dir: Path = config["output"]["development_dir"] / safe_slug(args.name)
    render_run(
        config_path=config_path,
        prompts_path=prompts_path,
        output_dir=experiment_dir / "base",
        prompt_ids=prompt_ids,
        lora_path=None,
        force=args.force,
    )
    for step in steps:
        render_run(
            config_path=config_path,
            prompts_path=prompts_path,
            output_dir=experiment_dir / str(step),
            prompt_ids=prompt_ids,
            lora_path=checkpoints_dir / f"checkpoint-{step}",
            force=args.force,
        )

    write_json(
        experiment_dir / "manifest.json",
        {
            "name": args.name,
            "createdUtc": datetime.now(timezone.utc).isoformat(),
            "modelId": config["model"]["id"],
            "modelRevision": config["model"]["revision"],
            "adapterScale": config["lora"]["adapter_scale"],
            "steps": ["base", *steps],
            "prompts": [
                {
                    "id": item.prompt_id,
                    "prompt": item.prompt,
                    "seed": item.seed,
                    "samples": {
                        "base": str((experiment_dir / "base" / f"{item.prompt_id}.png").resolve()),
                        **{
                            str(step): str((experiment_dir / str(step) / f"{item.prompt_id}.png").resolve())
                            for step in steps
                        },
                    },
                }
                for item in prompts
            ],
        },
    )
    logger.info("Development review manifest written to %s", experiment_dir / "manifest.json")


if __name__ == "__main__":
    main()
