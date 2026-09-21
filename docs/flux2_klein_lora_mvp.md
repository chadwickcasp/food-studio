# FLUX.2 [klein] Food-Advertising LoRA MVP

## Goal

Build the smallest useful end-to-end baseline that answers one question:

> Does a LoRA trained on the curated food-advertising dataset make `black-forest-labs/FLUX.2-klein-base-4B` produce images that match the target contemporary food-advertising style better than the unmodified base model, while preserving the requested content?

The deliverable is a reproducible, same-prompt/same-seed comparison between the base model and that same model with the trained LoRA loaded.

## Scope

Use one straightforward PyTorch training path built with Hugging Face Diffusers and PEFT. Train a LoRA on the FLUX transformer only; keep the transformer base weights, text encoder, and VAE frozen.

Keep this MVP intentionally narrow:

- one model and one curated dataset;
- one single-GPU training job;
- one configuration file;
- one training entry point and one inference entry point;
- local folders for checkpoints, generated samples, and review notes.

This is a style-learning experiment, not an identity or single-concept DreamBooth project.

## Infrastructure

- Google Cloud Compute Engine Spot VM: `g2-standard-8`.
- GPU: 1 x NVIDIA L4 with 24 GB VRAM; the machine type also provides 8 vCPUs and 32 GB system memory.
- Disk: 100 GB persistent disk is sufficient for this MVP.
- Clone or sync the local repository to the VM; do not add Vertex AI or another orchestration layer.
- Spot preemption is acceptable. Save resumable training-state checkpoints periodically and retain the final LoRA weights, comparison images, and evaluation manifest on persistent storage.
- Before starting, accept any required Hugging Face model terms and authenticate on the VM without committing credentials.

Spot availability and pricing vary by region. The relevant references are the [Google Cloud G2 machine specifications](https://cloud.google.com/compute/docs/gpus) and [Spot VM pricing](https://cloud.google.com/spot-vms/pricing).

## Minimal repository shape

```text
.
├── config.yaml
├── data/
│   ├── train/
│   │   ├── 0001.jpg
│   │   ├── 0001.txt
│   │   └── ...
│   └── validation_prompts.json
├── src/
│   ├── train.py
│   └── inference.py
└── outputs/
    ├── checkpoints/
    ├── samples/
    │   ├── baseline/
    │   └── lora/
    ├── comparisons/
    └── observations.md
```

Do not create these implementation files in this planning step. This tree is the target for the next step.

## Dataset contract

- Begin with roughly 20-40 carefully selected images that consistently express the target visual style.
- Store each image beside a UTF-8 caption with the same stem: `0001.jpg` and `0001.txt`.
- Every image must have exactly one non-empty caption, and every caption must have a matching image.
- Captions should describe visible image contents only: subject, ingredients, vessels, props, setting objects, and actions. Do not caption lighting quality, camera angle, depth of field, color grade, or commercial finish. Those style signals should remain in the pixels so the LoRA can learn them as residual.
- Use natural-language captions with enough detail to separate variable subjects, props, and settings from the shared look. Do not add a synthetic style trigger token for this experiment; loading and scaling the adapter is the style control.
- Keep food subjects and compositions varied enough that the adapter learns a visual language rather than memorizing one dish, layout, or product.
- Exclude low-quality, near-duplicate, watermarked, or off-style images. Confirm that the project is permitted to use every image for training.

`train.py` should validate the pairs and fail clearly before loading the model if the dataset is malformed.

## Model and LoRA setup

- Base checkpoint: `black-forest-labs/FLUX.2-klein-base-4B`.
- Load with Diffusers and train through PEFT's supported adapter APIs.
- Freeze all original transformer weights, the text encoder, and the VAE.
- Add LoRA adapters only to the FLUX transformer.
- Use the target-module defaults from the pinned official Diffusers [FLUX.2 Klein LoRA example](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux2_klein.py). Do not design a custom target set for this MVP.
- Pin and record the Diffusers revision used. The upstream example can change, so copy its target-module default exactly from that revision rather than silently mixing versions.
- Save adapter weights in the standard Diffusers/PEFT format, preferably `pytorch_lora_weights.safetensors`, plus enough configuration and training state to reload or resume.

The implementation must assert, before optimization begins, that:

1. at least one parameter is trainable;
2. every trainable parameter belongs to a LoRA adapter; and
3. no text-encoder, VAE, or original transformer parameter is trainable.

## Initial training placeholders

These are conservative launch values, not tuned recommendations. Keep them fixed for the first successful comparison except when a change is required to resolve an out-of-memory or correctness failure. Record every such change in `outputs/observations.md`.

| Setting | Initial placeholder |
| --- | --- |
| Precision | BF16 |
| Resolution | 768 px square |
| Per-device batch size | 1 |
| Gradient accumulation | 4 steps |
| Gradient checkpointing | enabled |
| Cache VAE latents | enabled |
| Optimizer | AdamW; use 8-bit AdamW if needed for memory |
| LoRA rank / alpha | 8 / 8, pre-registered for modest additional style capacity while keeping alpha/rank at 1 |
| LoRA dropout | 0.0 |
| Learning rate | `1e-4` |
| Scheduler | constant |
| Warmup | 50 steps |
| Maximum training steps | 500 |
| Checkpoint interval | every 100 optimizer steps |
| Training seed | fixed and recorded, for example `42` |

If 24 GB is still insufficient, first enable or confirm CPU offload, 8-bit AdamW, latent caching, and gradient checkpointing; then reduce resolution to 512. Do not introduce distributed training, a different model, or a custom quantization system to rescue this MVP.

The official [Diffusers FLUX.2 training guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux2.md) is the reference for supported Klein memory controls and save/load behavior.

## Matched evaluation

### Before training

1. Write 5-10 validation prompts that cover different dishes, framing, backgrounds, lighting situations, and layouts relevant to the intended food-advertising use.
2. Give each prompt a fixed random seed and stable ID in `data/validation_prompts.json`.
3. Also fix and record the model revision, image dimensions, inference steps, guidance value, scheduler, and other generation settings.
4. Run `src/inference.py` without an adapter and save the results to `outputs/samples/baseline/`.

### After training

1. Start from the same base checkpoint and load the saved LoRA through the supported Diffusers API.
2. Regenerate every validation item with the exact same prompt, seed, and inference settings.
3. Save those images to `outputs/samples/lora/` using filenames that match the baseline images.
4. Produce a simple labeled side-by-side grid or paired folder view in `outputs/comparisons/`.

The comparison should change only one meaningful variable: LoRA absent versus LoRA loaded. Save a small manifest beside the images containing the prompt, seed, settings, base-model revision, adapter path, and adapter scale.

### Human review rubric

Score each baseline/LoRA pair from 1-5 on:

- composition and focal hierarchy;
- lighting and appetizing food presentation;
- match to the target styling and art direction;
- commercial polish;
- prompt fidelity.

Also note obvious failures such as repeated layouts, copied-looking training examples, artifacts, loss of variety, or ignored prompt details. Complicated automated metrics are not part of this MVP.

## Success criteria

The MVP is complete only when all of the following are true:

- the dataset-to-training-to-inference pipeline runs end to end on the target VM;
- periodic checkpoints and final LoRA weights save successfully;
- the LoRA can be loaded into a fresh inference process;
- the same-prompt/same-seed baseline comparison is reproducible from saved configuration and the evaluation manifest;
- reviewers see a visibly stronger match to the target contemporary food-advertising aesthetic in the LoRA images;
- prompt content remains intact, with no obvious collapse, severe memorization, or broad loss of variety.

A technically successful training run without a clear matched visual comparison is not a successful MVP.

## Recommended execution order

1. Prepare and inspect the image/caption pairs.
2. Define and freeze the validation prompts, seeds, and inference settings.
3. Generate and save the unfine-tuned baseline images.
4. Run one LoRA training job, saving periodic checkpoints.
5. Reload the final adapter in a fresh process.
6. Generate the matched post-training images.
7. Create the simple side-by-side comparison output.
8. Record rubric scores and observations; only then decide whether fine-tuning ablations are warranted.

## Not in the MVP

- custom LoRA injector;
- model-agnostic adaptation framework or generic abstractions;
- other model families or cross-model validation;
- target-layer or layer-choice ablations;
- hyperparameter sweeps;
- experiment-tracking platforms;
- Vertex AI;
- multi-GPU or distributed training;
- orchestration systems;
- TensorFlow.

## Cursor implementation rules

- Keep `train.py` and `inference.py` small, direct, and readable.
- Prefer official Diffusers and PEFT APIs and the official Klein example over hand-written adapter, checkpoint, or loading logic.
- Pin compatible dependency versions and record the exact base-model and Diffusers revisions used.
- Keep settings in `config.yaml`; do not build a framework, plugin system, registry, or deep class hierarchy.
- Add clear input checks, the trainable-parameter assertions above, and a smoke test that saves and reloads an adapter before committing to the full run.
- Make inference deterministic enough for matched review and save generation metadata with every batch.
- Do not add an experiment-tracking service for this MVP; local logs and `outputs/observations.md` are enough.
- Do not begin ablations, layer-choice experiments, or model-agnostic work until the baseline-versus-fine-tuned comparison works and has been reviewed.
