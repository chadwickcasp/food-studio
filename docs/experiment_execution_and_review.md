# Initial LoRA experiment: execution and review plan

## Falsifiable hypothesis

A rank-8 LoRA trained on 18–24 rights-cleared photographs with one coherent commercial food-photography look will improve perceived style match and food presentation on unseen subjects versus the unchanged FLUX.2 Klein Base 4B checkpoint, without a material increase in artifacts or loss of prompt fidelity.

The experiment does not claim that every output will improve or that this small review is statistically significant.

## Fixed variables

- Base model and exact revision
- Rank / alpha: 8 / 8
- Training resolution, optimizer, learning rate, batch settings, and seed from `config.yaml`
- Maximum 500 optimizer steps, with checkpoints every 100 steps
- Evaluation resolution, scheduler, inference-step count, guidance, and precision
- Prompt and seed within every baseline/LoRA pair
- LoRA scale 1.0 for the primary comparison

The primary independent variable in final evaluation is only whether the selected LoRA is loaded. The distilled-model portability check happens afterward and is reported separately.

## Phase 1: curate and caption

1. Review every candidate in the Captions screen.
2. Select the most literal content-only caption: subjects, ingredients, vessels, props, setting objects, and visible actions.
3. Exclude images that are weak, near-duplicate, off-style, poorly licensed, or likely to make the visual target less coherent.
4. Stop when 18–24 included images form the strongest coherent set.
5. Audit the selected captions again for style leakage. Do not describe lighting quality, lens/camera choices, depth of field, color grade, mood, or “commercial” polish.
6. Freeze the dataset selection and its source/license manifest before training begins.

Caption decisions are stored in `data/captions/selections.json`. They must be materialized as matching image/`.txt` pairs in `data/train/` before `src/train.py` is launched.

## Phase 2: freeze prompt splits

Use disjoint prompt sets:

- Development: four representative prompts with fixed seeds, used only to choose a checkpoint.
- Final evaluation: six held-out prompts with two fixed seeds each, producing 12 matched pairs. Do not inspect these outputs before checkpoint selection is locked.

The development prompts should span at least one plated entrée, dessert, beverage, and overhead composition. The final prompts should cover subjects and arrangements not present in training while remaining relevant to the intended commercial-food use.

## Phase 3: baseline and training

1. Generate and preserve the Base outputs for the final evaluation prompts before training.
2. Train the single pre-registered rank-8 LoRA to 500 steps.
3. Save resumable training state and adapter weights at steps 100, 200, 300, 400, and 500.
4. Record runtime, hardware, cost, loss trace, and any OOM/correctness recovery changes.
5. Do not use loss alone to choose the adapter. It is a diagnostic, not the photographic objective.

## Phase 4: choose the stopping point

For each saved checkpoint, render every development prompt with identical generation settings and seeds. Include the unmodified Base output as a reference. Write one development manifest in the format documented in `ui/README.md`, then review it in the Checkpoints screen.

For every development prompt:

1. Inspect Base and steps 100–500 side by side.
2. Mark the checkpoint with the best balance of target-style strength, prompt fidelity, food realism, subject variety, and artifact control.
3. Note the first point where the adapter becomes rigid, repeats a training composition, oversaturates the style, or drops requested content.

Select the checkpoint with the most per-prompt wins. If two checkpoints are effectively tied, choose the earlier one. Reject a nominal winner if it introduces a severe prompt-fidelity or artifact regression on more than one development prompt. Lock this choice before rendering any final LoRA outputs.

## Phase 5: final blinded A/B evaluation

1. Load the selected checkpoint into a fresh inference process.
2. Generate the 12 LoRA outputs for the six held-out prompts and two fixed seeds, using the exact Base settings.
3. Confirm that both manifests contain matching IDs, prompts, seeds, dimensions, scheduler, inference steps, guidance, precision, model revision, and runtime metadata.
4. Create a blinded session in the A/B Test screen. The app randomizes A/B placement independently for each pair and withholds identities until every item is scored.
5. Score style match, prompt fidelity, food presentation, and artifact control; add notes for memorization, repeated layouts, copied-looking examples, ambiguity, and regressions.
6. Reveal identities only after all 12 items are saved.

Report win/tie counts by criterion, overall preference, prompt-level failures, and representative examples selected before identities are revealed. Treat these as descriptive results, not statistical proof.

## Decision rule

Call the experiment promising only if the selected LoRA wins more style-match and food-presentation decisions than it loses, while prompt-fidelity and artifact-control losses remain uncommon and no clear memorization pattern appears. A mixed result is still useful: document which subjects or compositions improve and where the adapter regresses.

After the Base comparison is complete, run the smaller distilled-checkpoint portability check with the selected adapter. Do not let that secondary result change the primary checkpoint choice.
