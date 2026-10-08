from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from src.development import checkpoint_steps, run_complete
from src.experiment import (
    OFFICIAL_KLEIN_LORA_TARGET_MODULES,
    ROOT,
    assert_lora_only_trainable,
    current_render_settings,
    load_config,
    load_validation_prompts,
    render_request_matches,
    repo_relative,
    validate_image_caption_pairs,
)
from src.inference import generate_samples, write_pair
from src.prepare_train import downscale_training_image, prepare_train_dir, scaled_size


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
    assert loaded["dataset"]["train_short_edge"] == 1024
    assert loaded["dataset"]["source_dir"] == tmp_path / "data"
    assert loaded["training"]["seed"] == 42
    assert loaded["lora"]["rank"] == 8
    assert loaded["lora"]["alpha"] == 8
    assert loaded["model"]["id"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert loaded["training"]["checkpoint_steps"] == 50
    assert loaded["inference"]["development_prompts"] == tmp_path / "data" / "development_prompts.json"


def test_load_config_rejects_short_edge_below_resolution(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    text = Path("config.yaml").read_text(encoding="utf-8").replace(
        "train_short_edge: 1024", "train_short_edge: 512"
    )
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="train_short_edge"):
        load_config(config_path)


def test_load_config_rejects_bad_resolution(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    text = Path("config.yaml").read_text(encoding="utf-8").replace("resolution: 768", "resolution: 770")
    config_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="divisible by 16"):
        load_config(config_path)


def test_validation_prompts_file() -> None:
    prompts = load_validation_prompts(Path("data/validation_prompts.json"))
    assert len(prompts) == 32
    assert prompts[0].prompt_id == "eval-01"
    assert prompts[0].prompt_group_id == "eval-01"
    assert prompts[0].seed == 101
    assert [item.prompt_id for item in prompts[::4]] == [f"eval-{index:02d}" for index in range(1, 9)]
    first_seed = {
        item.prompt_group_id: item.seed
        for item in prompts
        if item.prompt_id == item.prompt_group_id
    }
    assert first_seed == {
        "eval-01": 101,
        "eval-02": 202,
        "eval-03": 303,
        "eval-04": 404,
        "eval-05": 505,
        "eval-06": 606,
        "eval-07": 707,
        "eval-08": 808,
    }
    assert all(
        len([item for item in prompts if item.prompt_group_id == f"eval-{index:02d}"]) == 4
        for index in range(1, 9)
    )
    assert len({item.seed for item in prompts}) == 32


def test_development_prompts_are_separate_and_complete() -> None:
    development = load_validation_prompts(Path("data/development_prompts.json"))
    validation = load_validation_prompts(Path("data/validation_prompts.json"))
    assert len(development) == 32
    assert [item.prompt_id for item in development[::4]] == [f"dev-{index:02d}" for index in range(1, 9)]
    assert len({item.seed for item in development}) == 32
    assert all(
        len([item for item in development if item.prompt_group_id == f"dev-{index:02d}"]) == 4
        for index in range(1, 9)
    )
    assert {item.prompt_group_id for item in development}.isdisjoint(
        item.prompt_group_id for item in validation
    )
    assert {item.prompt for item in development}.isdisjoint(item.prompt for item in validation)
    assert {item.seed for item in development}.isdisjoint(item.seed for item in validation)


def test_multi_seed_prompt_preserves_first_sample_id(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        '{"prompts":[{"id":"dev-a","prompt":"one","seeds":[11,12,13,14]}]}',
        encoding="utf-8",
    )
    prompts = load_validation_prompts(path)
    assert [item.prompt_id for item in prompts] == [
        "dev-a",
        "dev-a-seed-12",
        "dev-a-seed-13",
        "dev-a-seed-14",
    ]
    assert {item.prompt_group_id for item in prompts} == {"dev-a"}


def test_prompt_rejects_seed_and_seeds_together(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        '{"prompts":[{"id":"a","prompt":"one","seed":1,"seeds":[1,2]}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="either 'seed' or 'seeds'"):
        load_validation_prompts(path)


def test_prompt_rejects_duplicate_seeds(tmp_path: Path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        '{"prompts":[{"id":"a","prompt":"one","seeds":[1,1]}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        load_validation_prompts(path)


def _sample_config() -> dict:
    return {
        "model": {"id": "model", "revision": "revision"},
        "diffusers_revision": "diffusers-revision",
        "lora": {"adapter_scale": 1.0},
        "inference": {
            "width": 16,
            "height": 16,
            "num_inference_steps": 1,
            "guidance_scale": 4.0,
            "scheduler": "scheduler",
            "precision": "bf16",
            "cpu_offload": True,
            "max_sequence_length": 32,
            "text_encoder_out_layers": [10],
        },
    }


def _write_adapter(folder: Path, payload: bytes) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    weight_file = folder / "pytorch_lora_weights.safetensors"
    weight_file.write_bytes(payload)
    return folder


def _write_prompt_file(path: Path, seed: int = 11) -> None:
    path.write_text(
        json.dumps({"prompts": [{"id": "dev-a", "prompt": "one", "seed": seed}]}),
        encoding="utf-8",
    )


def _write_previous_manifest(
    output_dir: Path,
    config: dict,
    lora_path: Path | None,
    *,
    seed: int = 11,
    legacy: bool = False,
) -> None:
    manifest = current_render_settings(config, lora_path)
    if legacy:
        manifest.pop("adapter_sha256")
        manifest.pop("cpu_offload")
    manifest["samples"] = [
        {"id": "dev-a", "prompt": "one", "seed": seed, "runtime_seconds": 1.5}
    ]
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_generate_samples_reuses_image_when_saved_sample_matches(tmp_path: Path, monkeypatch) -> None:
    prompts_path = tmp_path / "prompts.json"
    _write_prompt_file(prompts_path)
    output_dir = tmp_path / "renders"
    output_dir.mkdir()
    image_path = output_dir / "dev-a.png"
    Image.new("RGB", (16, 16), (12, 34, 56)).save(image_path)
    original = image_path.read_bytes()
    adapter = _write_adapter(tmp_path / "outputs" / "checkpoints" / "checkpoint-100", b"adapter-v1")
    config = _sample_config()
    _write_previous_manifest(output_dir, config, adapter, legacy=True)

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("The pipeline should not load when the saved sample still matches")

    monkeypatch.setattr("src.inference.load_pipeline", fail_if_loaded)
    generate_samples(
        config,
        adapter,
        prompts_path=prompts_path,
        output_dir=output_dir,
        reuse_existing=True,
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert image_path.read_bytes() == original
    assert manifest["generated_sample_count"] == 0
    assert manifest["reused_sample_count"] == 1
    assert manifest["samples"][0]["reused_existing"] is True
    assert manifest["adapter_sha256"] == current_render_settings(config, adapter)["adapter_sha256"]


def test_generate_samples_regenerates_when_image_has_no_manifest(tmp_path: Path, monkeypatch) -> None:
    prompts_path = tmp_path / "prompts.json"
    _write_prompt_file(prompts_path)
    output_dir = tmp_path / "renders"
    output_dir.mkdir()
    Image.new("RGB", (16, 16), (12, 34, 56)).save(output_dir / "dev-a.png")
    adapter = _write_adapter(tmp_path / "outputs" / "checkpoints" / "checkpoint-100", b"adapter-v1")

    def mark_loaded(*_args, **_kwargs):
        raise RuntimeError("pipeline loaded")

    monkeypatch.setattr("src.inference.load_pipeline", mark_loaded)
    with pytest.raises(RuntimeError, match="pipeline loaded"):
        generate_samples(
            _sample_config(),
            adapter,
            prompts_path=prompts_path,
            output_dir=output_dir,
            reuse_existing=True,
        )


def test_generate_samples_regenerates_when_saved_seed_differs(tmp_path: Path, monkeypatch) -> None:
    prompts_path = tmp_path / "prompts.json"
    _write_prompt_file(prompts_path, seed=11)
    output_dir = tmp_path / "renders"
    output_dir.mkdir()
    image_path = output_dir / "dev-a.png"
    Image.new("RGB", (16, 16), (12, 34, 56)).save(image_path)
    original = image_path.read_bytes()
    adapter = _write_adapter(tmp_path / "outputs" / "checkpoints" / "checkpoint-100", b"adapter-v1")
    config = _sample_config()
    _write_previous_manifest(output_dir, config, adapter, seed=99)

    def mark_loaded(*_args, **_kwargs):
        raise RuntimeError("pipeline loaded")

    monkeypatch.setattr("src.inference.load_pipeline", mark_loaded)
    with pytest.raises(RuntimeError, match="pipeline loaded"):
        generate_samples(
            config,
            adapter,
            prompts_path=prompts_path,
            output_dir=output_dir,
            reuse_existing=True,
        )
    assert image_path.read_bytes() == original


def test_render_request_rejects_replaced_adapter_file(tmp_path: Path) -> None:
    folder = _write_adapter(tmp_path / "outputs" / "checkpoints" / "checkpoint-100", b"first")
    config = _sample_config()
    previous = current_render_settings(config, folder)
    (folder / "pytorch_lora_weights.safetensors").write_bytes(b"second")
    current = current_render_settings(config, folder)
    assert render_request_matches(previous, current) is False


def test_render_request_rejects_two_existing_adapter_directories(tmp_path: Path) -> None:
    config = _sample_config()
    first = _write_adapter(tmp_path / "a" / "outputs" / "checkpoints" / "checkpoint-100", b"same")
    second = _write_adapter(tmp_path / "b" / "outputs" / "checkpoints" / "checkpoint-100", b"same")
    previous = current_render_settings(config, first)
    current = current_render_settings(config, second)
    assert previous["adapter_sha256"] == current["adapter_sha256"]
    assert render_request_matches(previous, current) is False


def test_legacy_manifest_accepts_adapter_path_from_another_machine(tmp_path: Path) -> None:
    folder = _write_adapter(tmp_path / "outputs" / "checkpoints" / "checkpoint-100", b"same")
    config = _sample_config()
    current = current_render_settings(config, folder)
    previous = dict(current)
    previous.pop("adapter_sha256")
    previous["adapter_path"] = "/home/chadcasper/food-studio/outputs/checkpoints/checkpoint-100"
    assert render_request_matches(previous, current) is True


def test_run_complete_requires_matching_manifest_and_images(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(
        json.dumps({"prompts": [{"id": "dev-a", "prompt": "one", "seeds": [11, 12]}]}),
        encoding="utf-8",
    )
    prompts = load_validation_prompts(prompt_file)
    output_dir = tmp_path / "renders"
    output_dir.mkdir()
    adapter = _write_adapter(tmp_path / "outputs" / "checkpoints" / "checkpoint-100", b"adapter-v1")
    config = _sample_config()
    for item in prompts:
        Image.new("RGB", (16, 16), (1, 2, 3)).save(output_dir / f"{item.prompt_id}.png")
    manifest = current_render_settings(config, adapter)
    manifest["samples"] = [
        {"id": item.prompt_id, "prompt": item.prompt, "seed": item.seed} for item in prompts
    ]
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert run_complete(output_dir, prompts, config, adapter) is True

    (output_dir / "dev-a-seed-12.png").unlink()
    assert run_complete(output_dir, prompts, config, adapter) is False

    Image.new("RGB", (16, 16), (1, 2, 3)).save(output_dir / "dev-a-seed-12.png")
    (adapter / "pytorch_lora_weights.safetensors").write_bytes(b"adapter-v2")
    assert run_complete(output_dir, prompts, config, adapter) is False


def test_repo_relative_strips_checkout_prefix() -> None:
    target = ROOT / "outputs" / "development" / "example.png"
    assert repo_relative(target) == "outputs/development/example.png"
    outside = Path("/tmp/example.png").resolve()
    assert repo_relative(outside) == outside.as_posix()


def test_checkpoint_steps_use_50_step_schedule() -> None:
    assert checkpoint_steps(500, 50) == list(range(50, 501, 50))


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


def test_scaled_size_caps_short_side_without_upscale() -> None:
    assert scaled_size(4000, 3000, 1024) == (1365, 1024)
    assert scaled_size(1024, 683, 1024) == (1024, 683)
    assert scaled_size(3456, 3456, 1024) == (1024, 1024)


def test_downscale_training_image_preserves_aspect(tmp_path: Path) -> None:
    image = Image.new("RGB", (4000, 2000), (10, 20, 30))
    resized = downscale_training_image(image, 1024)
    assert resized.size == (2048, 1024)


def test_prepare_train_dir_writes_downscaled_included_pairs(tmp_path: Path) -> None:
    source_dir = tmp_path / "data"
    train_dir = source_dir / "train"
    captions_dir = source_dir / "captions"
    grok_dir = captions_dir / "grok"
    grok_dir.mkdir(parents=True)
    train_dir.mkdir()
    Image.new("RGB", (4000, 3000), (200, 40, 20)).save(source_dir / "keep.jpg")
    Image.new("RGB", (800, 600), (20, 40, 200)).save(source_dir / "small.jpg")
    Image.new("RGB", (64, 64), (0, 0, 0)).save(train_dir / "stale.jpg")
    (train_dir / "stale.txt").write_text("should be removed\n", encoding="utf-8")
    (grok_dir / "keep.txt").write_text("A plated pasta with herbs.\n", encoding="utf-8")
    (captions_dir / "selections.json").write_text(
        json.dumps(
            {
                "version": 1,
                "selections": {
                    "keep": {
                        "source": "grok",
                        "customCaption": None,
                        "include": True,
                    },
                    "small": {
                        "source": "custom",
                        "customCaption": "A cocktail in a coupe glass.",
                        "include": True,
                    },
                    "skip": {
                        "source": "grok",
                        "customCaption": None,
                        "include": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    written = prepare_train_dir(
        source_dir=source_dir,
        train_dir=train_dir,
        selections_path=captions_dir / "selections.json",
        short_edge=1024,
    )
    assert sorted(path.name for path in written) == ["keep.jpg", "small.jpg"]
    keep = Image.open(train_dir / "keep.jpg")
    small = Image.open(train_dir / "small.jpg")
    assert keep.size == (1365, 1024)
    assert small.size == (800, 600)
    assert (train_dir / "keep.txt").read_text(encoding="utf-8").strip() == "A plated pasta with herbs."
    assert (train_dir / "small.txt").read_text(encoding="utf-8").strip() == "A cocktail in a coupe glass."
    assert not (train_dir / "stale.jpg").exists()
    assert not (train_dir / "stale.txt").exists()
