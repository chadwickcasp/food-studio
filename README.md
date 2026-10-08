# Food Studio

Food Studio is a small, hands-on project for exploring open-weight image-generation models and LoRA fine-tuning, using food and beverage photography as the experimental domain.

The first spike will test whether a FLUX.2 Klein LoRA trained on a small, rights-cleared set of visually coherent food and beverage photographs can improve a defined studio look without harming prompt adherence, subject variety, or realism. The project will also examine how dataset design, captions, checkpoint choice, and inference settings influence the result.

See [PROJECT.md](PROJECT.md) for the hypothesis, scope, and definition of done.

The first MVP is a same-prompt/same-seed comparison of `black-forest-labs/FLUX.2-klein-base-4B` with and without a transformer-only LoRA. Settings live in [`config.yaml`](config.yaml). The training recipe follows the official Diffusers FLUX.2 Klein LoRA example pinned in that file.

## Usage

1. Review captions in the UI, then materialize the included set into `data/train/`. That command copies each selected source image from `data/`, downscales so the short side is at most `dataset.train_short_edge` (no upscale, aspect ratio kept), and writes the chosen caption. Originals in `data/` are not modified. `src/train.py` will refuse to start if the pairs are incomplete.

   ```bash
   python -m src.prepare_train --config config.yaml
   ```
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
# Set GCP_PROJECT or GCP_ZONE if needed.
scripts/gcp.sh create
scripts/gcp.sh sync
scripts/gcp.sh ssh   # first boot: hf auth login, install PyTorch + requirements
```

Training from the laptop detaches on the VM, and `scripts/gcp.sh train` stays attached until the VM stops. The wrapper first generates and preserves the held-out Base baseline if it does not already exist. After a successful training run, it renders four fixed seeds for each of the eight development prompts for Base and every 50-step checkpoint, writes the checkpoint-review manifest, and then **stops the instance** so the L4 does not keep billing. An existing development image is kept when its manifest still matches the prompt, seed, inference settings, and adapter weights. Missing seeds are generated. `src.development --force` renders every sample again. The VM also stops after a baseline, training, or development-render failure. The boot disk is kept.

```bash
scripts/gcp.sh start && scripts/gcp.sh train --resume latest
```

If Compute Engine preempts the Spot VM, or a host error stops it, before that job finishes, the same command starts the instance again and resumes from the latest checkpoint. It keeps retrying while the zone has no L4. When the job itself stops the VM, the command exits. Ctrl-C ends the local watcher and leaves the VM as it is.

Watch the log with `scripts/gcp.sh watch`, or `scripts/gcp.sh ssh` and `tail -f ~/food-studio/outputs/train.log`. Cloud Console shows **STOPPED** when the job has exited. Then `scripts/gcp.sh start`, `scripts/gcp.sh pull`, and `scripts/gcp.sh stop` (or run inference on the VM before stopping again).

After a checkpoint is saved in the review app, render the held-out eval prompts on the L4 with `scripts/gcp.sh eval`. The step defaults to `selectedStep` in `outputs/reviews/checkpoint-selection.json`; pass `--step` to choose another one. The command uses the checkpoint already on the VM, renders `data/validation_prompts.json` for Base and that checkpoint, and stays attached until the VM stops. A host preemption starts the instance again and continues the unfinished samples. `--force` renders every sample again on the first launch. A restart keeps images whose manifest still matches. Then `scripts/gcp.sh start`, `scripts/gcp.sh pull`, and `scripts/gcp.sh stop`.

`python -m src.train` over SSH does **not** stop the VM; use that only when you want the node to stay up. After a host stop the boot disk remains. `scripts/gcp.sh train --resume latest` starts the VM if needed and resumes.

If 24 GB is still too small, enable `training.use_8bit_adam` (install `bitsandbytes`), confirm CPU offload and latent caching, then drop `training.resolution` to 512. Do not add distributed training or a different model for this MVP.

### G4 inference VM

Use the `g4` selector to create a separate Spot `g4-standard-48` VM (1× RTX PRO 6000, 96 GB VRAM) in `us-central1-b`. It uses a 100 GB Hyperdisk Balanced boot disk because G4 does not support the L4 VM's `pd-balanced` disk. Every command needs the selector so it targets `food-studio-g4` instead of the existing L4 VM:

```bash
scripts/gcp.sh g4 create
scripts/gcp.sh g4 sync-inference
scripts/gcp.sh g4 push-lora outputs/checkpoints/final
scripts/gcp.sh g4 ssh
# On the VM:
cd ~/food-studio
hf auth login
pip install -r requirements.txt
python -m src.inference --config config.yaml --output-dir outputs/samples/g4-baseline
python -m src.inference --config config.yaml --lora outputs/checkpoints/final --output-dir outputs/samples/g4-lora
exit
scripts/gcp.sh g4 stop
scripts/gcp.sh g4 start  # for a later session
```

`sync-inference` copies only `src/`, `config.yaml`, `requirements.txt`, the two prompt JSON files, and `scripts/develop_then_stop.sh`. It skips training images, generated outputs, and weights. `push-lora` copies just one local adapter; pass a checkpoint directory such as `outputs/checkpoints/checkpoint-500` to try a different one. The base model downloads from Hugging Face on first use and remains cached on the G4 boot disk across VM stops and starts. Use `scripts/gcp.sh g4 pull` to retrieve renders. The G4 profile leaves `config.yaml` unchanged, including `inference.cpu_offload: true`; test no-offload as a separately recorded comparison if desired. Set `GCP_ZONE` if Spot capacity is unavailable in `us-central1-b`.

After the one-time login and `pip install`, render the current development prompts for Base and every configured checkpoint with:

```bash
scripts/gcp.sh g4 develop
```

The command reads `training.checkpoint_steps` and `training.max_train_steps` from `config.yaml` and uploads each local `outputs/checkpoints/checkpoint-<step>/pytorch_lora_weights.safetensors`. It refuses to start the VM when one of those files is missing, so pull the training checkpoints onto this machine first. On the VM it syncs the inference files, detaches `python -m src.development`, and stops the instance when the render exits. A host preemption starts the VM again and continues; an image is kept when its manifest still matches the prompt, seed, settings, and adapter. Pass `--force` to render every sample again on the first launch. A host restart keeps images whose manifest still matches. Pass `--name <run-name>` for a later fine-tune so its review manifest stays separate from the default `Rank-8 initial run`. The name starts with a letter or number and may also contain dots, underscores, and hyphens.

The same command works on the L4 profile without the `g4` selector. It uploads the local checkpoint files, including over weights already on that VM. Ctrl-C leaves the VM as it is; the remote script still stops the VM when the render exits. Then retrieve the images:

```bash
scripts/gcp.sh g4 start
scripts/gcp.sh g4 pull
scripts/gcp.sh g4 stop
```

## Licensing

The [GNU Affero General Public License v3.0](LICENSE) applies to the source code in this repository. It does not apply to training images or other third-party image assets.

Each training image retains its own license or rights basis, as recorded in [ATTRIBUTIONS.md](ATTRIBUTIONS.md) and [data/dataset_manifest.csv](data/dataset_manifest.csv). Storing an image in this repository or using it in an experiment does not relicense that image under the repository's software license.

The current dataset also contains photographs licensed for noncommercial use only. As a conservative project policy, the dataset and any model artifacts trained with those photographs must remain noncommercial. Before any commercial use, replace those photographs and retrain from a commercial-compatible dataset, or obtain separate permission from their rights holders.
