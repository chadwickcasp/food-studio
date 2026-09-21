from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from src.experiment import (
    OFFICIAL_KLEIN_LORA_TARGET_MODULES,
    assert_lora_only_trainable,
    load_config,
    load_validation_prompts,
    validate_image_caption_pairs,
)
from src.inference import write_pair


@dataclass
class FakeParam:
    requires_grad: bool


def write_pair_files(train_dir: Path, stem: str, caption: str, suffix: str = ".jpg") -> None:
    Image.new("RGB", (32, 32), (120, 40, 20)).save(train_dir / f"{stem}{suffix}")
    (train_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")


def test_validate_pairs_accepts_matching_files(tmp_path: Path) -> None:
    write_pair_files(tmp_path, "0001", "A plated dessert with berries.")
    write_pair_files(tmp_path, "0002", "A cocktail on a marble table.", suffix=".png")
    pairs = validate_image_caption_pairs(tmp_path)
    assert [pair.image_path.stem for pair in pairs] == ["0001", "0002"]
    assert pairs[0].caption.startswith("A plated dessert")


def test_validate_pairs_rejects_missing_caption(tmp_path: Path) -> None:
    Image.new("RGB", (16, 16)).save(tmp_path / "0001.jpg")
    with pytest.raises(ValueError, match="without a matching .txt"):
        validate_image_caption_pairs(tmp_path)


def test_validate_pairs_rejects_empty_caption(tmp_path: Path) -> None:
    write_pair_files(tmp_path, "0001", "   ")
    with pytest.raises(ValueError, match="Empty captions"):
        validate_image_caption_pairs(tmp_path)


def test_validate_pairs_rejects_orphan_caption(tmp_path: Path) -> None:
    (tmp_path / "0001.txt").write_text("caption only", encoding="utf-8")
    with pytest.raises(ValueError, match="without a matching image"):
        validate_image_caption_pairs(tmp_path)


def test_validate_pairs_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No image/caption pairs"):
        validate_image_caption_pairs(tmp_path)


def test_official_target_modules_match_pinned_example() -> None:
    assert OFFICIAL_KLEIN_LORA_TARGET_MODULES[:5] == [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "to_qkv_mlp_proj",
    ]
    assert OFFICIAL_KLEIN_LORA_TARGET_MODULES[-1] == "single_transformer_blocks.23.attn.to_out"
    assert len(OFFICIAL_KLEIN_LORA_TARGET_MODULES) == 29


def test_assert_lora_only_trainable_accepts_lora_params() -> None:
    names = assert_lora_only_trainable(
        [("transformer.block.lora_A.default.weight", FakeParam(True))],
        [("text.layer.weight", FakeParam(False))],
        [("vae.encoder.weight", FakeParam(False))],
    )
    assert names == ["transformer.block.lora_A.default.weight"]


def test_assert_lora_only_trainable_rejects_base_weights() -> None:
    with pytest.raises(RuntimeError, match="original transformer"):
        assert_lora_only_trainable(
            [("transformer.block.weight", FakeParam(True))],
            [],
            [],
        )


def test_assert_lora_only_trainable_rejects_text_encoder() -> None:
    with pytest.raises(RuntimeError, match="Text encoder"):
        assert_lora_only_trainable(
            [("transformer.block.lora_B.weight", FakeParam(True))],
            [("text.layer.weight", FakeParam(True))],
            [],
        )


def test_load_config_resolves_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        Path("config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "data" / "train").mkdir(parents=True)
    loaded = load_config(config_path)
    assert loaded["dataset"]["train_dir"] == tmp_path / "data" / "train"
    assert loaded["training"]["seed"] == 42
    assert loaded["lora"]["rank"] == 8
    assert loaded["lora"]["alpha"] == 8
    assert loaded["model"]["id"] == "black-forest-labs/FLUX.2-klein-base-4B"


def test_load_config_rejects_bad_resolution(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    text = Path("config.yaml").read_text(encoding="utf-8").replace("resolution: 768", "resolution: 770")
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="divisible by 16"):
        load_config(config_path)


def test_validation_prompts_file() -> None:
    prompts = load_validation_prompts(Path("data/validation_prompts.json"))
    assert len(prompts) == 8
    assert prompts[0].prompt_id == "eval-01"
    assert len({item.seed for item in prompts}) == 8


def test_duplicate_prompt_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        '{"prompts":[{"id":"a","prompt":"one","seed":1},{"id":"a","prompt":"two","seed":2}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_validation_prompts(path)


def test_write_pair_grid(tmp_path: Path) -> None:
    left = Image.new("RGB", (64, 48), (200, 30, 30))
    right = Image.new("RGB", (64, 48), (30, 30, 200))
    out = tmp_path / "pair.png"
    grid = write_pair(left, right, "eval-01  seed=101", out)
    assert out.is_file()
    assert grid.size[0] == 64 + 64 + 8
    assert grid.size[1] == 48 + 36
