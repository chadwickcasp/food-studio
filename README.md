# Food Studio

Food Studio is a small, hands-on project for exploring open-weight image-generation models and LoRA fine-tuning, using food and beverage photography as the experimental domain.

The first spike will test whether a FLUX.2 Klein LoRA trained on a small, rights-cleared set of visually coherent food and beverage photographs can improve a defined studio look without harming prompt adherence, subject variety, or realism. The project will also examine how dataset design, captions, checkpoint choice, and inference settings influence the result.

See [PROJECT.md](PROJECT.md) for the hypothesis, scope, and definition of done.

The first MVP is a same-prompt/same-seed comparison of `black-forest-labs/FLUX.2-klein-base-4B` with and without a transformer-only LoRA. Settings live in [`config.yaml`](config.yaml). The training recipe follows the official Diffusers FLUX.2 Klein LoRA example pinned in that file.

## Usage

1. Copy 20–40 rights-cleared training images into `data/train/` and write a matching UTF-8 `.txt` caption for each file stem (`0001.jpg` with `0001.txt`). Captions should describe visible image contents only (subject, props, setting, actions), so the LoRA can absorb photographic style from the pixels. `src/train.py` will refuse to start if the pairs are incomplete.
2. On the training VM, accept any required Hugging Face terms, then `hf auth login`. Do not commit credentials.
3. Install PyTorch for the GPU, then `pip install -r requirements.txt`.
4. Generate baseline images before training:

   ```bash
   python -m src.inference --config config.yaml
   ```

5. Optionally check adapter save/load before the full run:

   ```bash
   python -m src.smoke_test --config config.yaml
   ```

6. Train, then generate the matched LoRA images:

   ```bash
   python -m src.train --config config.yaml
   python -m src.inference --config config.yaml --lora outputs/checkpoints/final
   ```

   Spot preemption: `python -m src.train --config config.yaml --resume latest`.

7. Score the pairs in [`outputs/observations.md`](outputs/observations.md). Change `config.yaml` only to recover from OOM or a correctness failure, and record the change there.

## Review UI

The local review app supports three experiment decisions: choosing a content-only caption per training image, comparing development renders across saved training steps, and running a randomized blinded A/B test between any two prompt-and-seed matched generation runs.

```bash
cd ui
npm install
npm run dev
```

Open `http://127.0.0.1:4173`. Decisions are written back to the repository as JSON so reviews are resumable and auditable. See [`ui/README.md`](ui/README.md) for the accepted manifest shapes and output locations.

## GCP Spot VM

The plan uses a Compute Engine Spot `g2-standard-8` (1x L4, 100 GB disk). Sync the local repo; do not add Vertex AI.

```bash
# Edit PROJECT and ZONE at the top of scripts/gcp.sh if needed.
scripts/gcp.sh create
scripts/gcp.sh sync
scripts/gcp.sh ssh
```

On the VM: `hf auth login`, install requirements, then run inference/train from `~/food-studio`. After preemption the VM stops and keeps its disk (`scripts/gcp.sh start`, then `--resume latest`). Copy review artifacts back with `scripts/gcp.sh pull`.

If 24 GB is still too small, enable `training.use_8bit_adam` (install `bitsandbytes`), confirm CPU offload and latent caching, then drop `training.resolution` to 512. Do not add distributed training or a different model for this MVP.

## Licensing

The [GNU Affero General Public License v3.0](LICENSE) applies to the source code in this repository. It does not apply to training images or other third-party image assets.

Each training image retains its own license or rights basis, as recorded in [ATTRIBUTIONS.md](ATTRIBUTIONS.md) and [data/dataset_manifest.csv](data/dataset_manifest.csv). Storing an image in this repository or using it in an experiment does not relicense that image under the repository's software license.

The current dataset also contains photographs licensed for noncommercial use only. As a conservative project policy, the dataset and any model artifacts trained with those photographs must remain noncommercial. Before any commercial use, replace those photographs and retrain from a commercial-compatible dataset, or obtain separate permission from their rights holders.
