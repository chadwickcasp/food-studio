# Food Studio Review UI

This is a local, filesystem-backed review tool. It does not upload images or require a database.

## Run it

```bash
npm install
npm run dev
```

For a production build, use `npm run build` and then `npm run serve`.

## Decisions it records

- Caption choices: `data/captions/selections.json`
- Selected training checkpoint: `outputs/reviews/checkpoint-selection.json`
- Blinded A/B sessions: `outputs/reviews/ab/<session-id>/`

The A/B reveal key is stored separately from the blind response file. The UI will not reveal identities until every comparison has a saved response.

## Generation-run manifests

The A/B tool opens only after `outputs/reviews/checkpoint-selection.json` records a stopping point. It then compares two validation-prompt manifests: unmodified base, and the locked `checkpoint-<step>/pytorch_lora_weights.safetensors`. Development renders cannot be added to this test. Each prompt is shown as a shuffled pair, and identities stay hidden until every item is scored.

```json
{
  "model_id": "black-forest-labs/FLUX.2-klein-base-4B",
  "adapter_path": null,
  "samples": [
    {
      "id": "eval-01",
      "prompt": "A plated lemon tart on a pale stone surface",
      "seed": 18291,
      "image": "outputs/eval/base/eval-01.png"
    }
  ]
}
```

## Development-checkpoint manifest

The checkpoint tool discovers `outputs/development/**/manifest.json` files with `steps` and `prompts`. Each entry is one prompt–seed sample, so the same `groupId` can appear at several fixed seeds. The Checkpoints screen groups those entries into one prompt and shows each checkpoint as a contact sheet of its seeds, in seed order. Marking a step applies to every seed in the group. Each sample path may be absolute or relative to the repository.

```json
{
  "name": "rank-8 first run",
  "modelId": "black-forest-labs/FLUX.2-klein-base-4B",
  "steps": ["base", 50, 100, 150, 200],
  "prompts": [
    {
      "id": "dev-01",
      "groupId": "dev-01",
      "prompt": "A glass of iced hibiscus tea beside sliced citrus",
      "seed": 92014,
      "samples": {
        "base": "outputs/development/rank-8/base/dev-01.png",
        "50": "outputs/development/rank-8/50/dev-01.png",
        "100": "outputs/development/rank-8/100/dev-01.png"
      }
    },
    {
      "id": "dev-01-seed-92015",
      "groupId": "dev-01",
      "prompt": "A glass of iced hibiscus tea beside sliced citrus",
      "seed": 92015,
      "samples": {
        "base": "outputs/development/rank-8/base/dev-01-seed-92015.png",
        "50": "outputs/development/rank-8/50/dev-01-seed-92015.png",
        "100": "outputs/development/rank-8/100/dev-01-seed-92015.png"
      }
    }
  ]
}
```

The source prompt file may use `"seed": 92014` for a single-seed item or `"seeds": [92014, 92015, 92016, 92017]` for a multi-seed item. The first seed retains the original sample ID and filename. A later run keeps that file when the manifest still records the same prompt, seed, inference settings, and adapter weights.

Do not use final evaluation prompts in this manifest. Checkpoint selection is a development-set decision; the final A/B test should use held-out prompts.
