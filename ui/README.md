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

The A/B tool discovers every `outputs/**/manifest.json` containing a non-empty `samples` array. Two runs can be paired when their samples share the same `id`, `prompt`, and `seed`.

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

The checkpoint tool discovers `outputs/development/**/manifest.json` files with `steps` and `prompts`. Each sample path may be absolute or relative to the repository.

```json
{
  "name": "rank-8 first run",
  "modelId": "black-forest-labs/FLUX.2-klein-base-4B",
  "steps": ["base", 50, 100, 150, 200],
  "prompts": [
    {
      "id": "dev-01",
      "prompt": "A glass of iced hibiscus tea beside sliced citrus",
      "seed": 92014,
      "samples": {
        "base": "outputs/development/rank-8/base/dev-01.png",
        "50": "outputs/development/rank-8/50/dev-01.png",
        "100": "outputs/development/rank-8/100/dev-01.png"
      }
    }
  ]
}
```

Do not use final evaluation prompts in this manifest. Checkpoint selection is a development-set decision; the final A/B test should use held-out prompts.
