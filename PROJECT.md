# Project Brief

## Purpose

This project is a compact exploration of open-weight image-generation models and model adaptation. Its purpose is to understand how diffusion and flow-based image models behave before and after LoRA fine-tuning, and how dataset curation, captions, checkpoint selection, inference settings, and evaluation methods affect generated images.

Food and beverage photography provides a focused visual domain in which to study photorealism, lighting, composition, material behavior, and consistency. The project emphasizes hands-on experimentation, reproducibility, and honest analysis of both improvements and regressions.

## First experiment

The initial experiment asks:

> Does a LoRA trained on 18–24 rights-cleared photographs sharing one coherent commercial food-photography look improve FLUX.2 Klein outputs for unseen food and beverage subjects?

The photographic look will be defined before selecting the final images. Training content should vary in subject, framing, props, and background while remaining coherent in its core lighting, tonal, color, texture, and compositional treatment.

## Experimental design

1. Define a short photographic style contract.
2. Curate and caption 18–24 rights-cleared training images.
3. Train one LoRA against FLUX.2 Klein Base 4B.
4. Compare Klein Base with and without the LoRA using identical prompts, seeds, and inference settings.
5. Run a smaller transfer check on the distilled Klein checkpoint.
6. Evaluate results with blinded human comparisons, a calibrated vision-language-model rubric, and narrow automated checks.
7. Report improvements, regressions, limitations, runtime, and cost.

## In scope

- One clearly defined photographic look
- One LoRA training run
- Eight development prompts for checkpoint selection and eight held-out evaluation prompts, each with a fixed seed
- Photorealism, lighting, material behavior, composition, prompt adherence, and artifact evaluation
- Reproducible manifests recording model and generation settings
- A small Base-to-Distilled portability check
- A basic local review UI for caption curation, development-checkpoint selection, and blinded A/B scoring

## Out of scope

- Logo or exact typography preservation
- Multi-image blending
- Task-specific edit-LoRA training
- Hyperparameter searches
- Production deployment, hosted collaboration, or a general-purpose model-evaluation platform
- Claims of statistical significance from the small evaluation set

## Definition of done

The spike is complete when it has a documented dataset, a trained LoRA, paired baseline and treatment outputs, a repeatable evaluation result, and an honest written conclusion explaining what improved, what regressed, and what should be tested next.

## Principles

- Use only owned, licensed, or public-domain training imagery.
- Change one experimental variable at a time.
- Preserve prompts, seeds, settings, and model versions in a run manifest.
- Report failed or mixed results rather than selecting only favorable examples.
