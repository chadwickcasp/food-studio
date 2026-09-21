# Food Studio Agent Guide

These instructions apply to the entire repository.

## Project purpose

Food Studio is a focused, hands-on exploration of open-weight image-generation models and LoRA fine-tuning. Food and beverage photography is the experimental domain because it exposes useful challenges in photorealism, lighting, composition, material behavior, and consistency.

Read `README.md` and `PROJECT.md` before proposing or making changes.

## Current experiment

The first experiment asks whether a LoRA trained on 18–24 rights-cleared photographs sharing one coherent photographic look improves FLUX.2 Klein outputs for unseen food and beverage subjects.

The primary controlled comparison is:

- FLUX.2 Klein Base 4B without the LoRA
- The same Base checkpoint with the trained LoRA

Prompts, seeds, resolution, scheduler, steps, guidance, and other inference settings must remain identical within each comparison. A smaller secondary experiment may test whether the selected LoRA transfers to the distilled Klein checkpoint. Train against Base; treat Distilled as a separate deployment and portability check.

## Experimental integrity

- State a falsifiable hypothesis before running an experiment.
- Change one independent variable at a time.
- Separate training data, development prompts used for checkpoint selection, and final evaluation prompts.
- Never select final examples only because they support the hypothesis.
- Record improvements, regressions, failures, and ambiguous results.
- Do not claim statistical significance from this small evaluation set.
- Treat automated similarity and prompt-alignment metrics as supporting evidence, not substitutes for photographic judgment.
- Blind and randomize human or vision-language-model A/B comparisons where practical.

Every generated result should be traceable to its configuration. Record at least:

- Model identifier and exact revision
- LoRA identifier, checkpoint, and scale, if used
- Prompt and negative prompt, if applicable
- Seed
- Width and height
- Inference steps, scheduler, and guidance
- Precision and relevant optimization settings
- Runtime, hardware, and estimated cost when available

## Data and model artifacts

- Use only imagery that is owned, licensed for the intended use, or verifiably public domain.
- Do not scrape, download, or add training data unless the user has approved the source and its terms.
- Maintain a data manifest containing source, creator when known, license or rights basis, retrieval date, and content hash.
- For this style LoRA, caption only visible variable content: subjects, ingredients, vessels, props, setting objects, and actions. Do not caption the shared photographic style, lighting quality, camera angle, composition, depth of field, color grade, or commercial finish.
- Do not add a synthetic style trigger token unless the experiment scope is explicitly changed and the change is documented. With the current trigger-free design, loading and scaling the adapter enables the learned style.
- Do not commit source datasets, model checkpoints, LoRA weights, generated image batches, credentials, or other large artifacts to Git unless explicitly requested.
- Never expose access tokens, API keys, or private URLs in source files, logs, manifests, or documentation.

## Implementation guidance

- Prefer clear Python and PyTorch/Diffusers code over notebook-only or UI-only workflows.
- Keep dependencies minimal and add them only when first needed.
- Use type hints, `pathlib`, small focused functions, and deterministic seed handling.
- Keep experiment parameters in version-controlled YAML or JSON rather than hard-coding them across scripts.
- Keep model-provider details behind narrow interfaces when doing so does not obscure model-specific capabilities.
- Validate inputs early and fail with actionable messages.
- Add lightweight tests for deterministic utilities, manifest handling, configuration validation, and metric calculations.
- Do not build a web UI unless the user asks for one; a reproducible command-line workflow and static report are sufficient for the first spike.

## Expected repository shape

Create directories only when they are needed. The intended shape is:

```text
configs/        version-controlled experiment and training configuration
data/           local datasets and manifests; image contents normally ignored by Git
eval/           held-out prompts, rubrics, and evaluation definitions
src/            generation, training support, evaluation, and reporting code
runs/           generated artifacts and machine-readable run manifests
report/         concise human-readable findings and selected contact sheets
tests/          tests for reusable deterministic code
```

## Current non-goals

- Logo or exact typography preservation
- Multi-image blending
- Task-specific edit-LoRA training
- Hyperparameter sweeps
- A polished application UI
- Production deployment

Do not expand into these areas without an explicit request or a documented change to the experiment scope.
