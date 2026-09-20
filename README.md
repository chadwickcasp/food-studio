# Food Studio

Food Studio is a small, hands-on project for exploring open-weight image-generation models and LoRA fine-tuning, using food and beverage photography as the experimental domain.

The first spike will test whether a FLUX.2 Klein LoRA trained on a small, rights-cleared set of visually coherent food and beverage photographs can improve a defined studio look without harming prompt adherence, subject variety, or realism. The project will also examine how dataset design, captions, checkpoint choice, and inference settings influence the result.

See [PROJECT.md](PROJECT.md) for the hypothesis, scope, and definition of done.

## Licensing

The [GNU Affero General Public License v3.0](LICENSE) applies to the source code in this repository. It does not apply to training images or other third-party image assets.

Each training image retains its own license or rights basis, as recorded in [ATTRIBUTIONS.md](ATTRIBUTIONS.md) and [data/dataset_manifest.csv](data/dataset_manifest.csv). Storing an image in this repository or using it in an experiment does not relicense that image under the repository's software license.

The current dataset also contains photographs licensed for noncommercial use only. As a conservative project policy, the dataset and any model artifacts trained with those photographs must remain noncommercial. Before any commercial use, replace those photographs and retrain from a commercial-compatible dataset, or obtain separate permission from their rights holders.
