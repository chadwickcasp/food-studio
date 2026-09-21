# Observations

Hypothesis: a LoRA trained on the curated food-advertising image/caption pairs will make `black-forest-labs/FLUX.2-klein-base-4B` match the target contemporary food-advertising look more closely than the unmodified base model, while keeping the requested dish, props, and framing intact.

## Target look

Commercial food advertising: appetizing light, clear focal hierarchy, styled ingredients, controlled color, and a finished studio surface. Subjects and layouts should stay varied.

## Run log

Record every training or inference setting change here, especially OOM recoveries (8-bit AdamW, extra offload, or 512 px).

| Date | Change | Reason |
| --- | --- | --- |
|  |  |  |

## Rubric

Score each baseline/LoRA pair from 1–5.

| ID | Composition | Lighting / presentation | Style match | Commercial polish | Prompt fidelity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| eval-01 |  |  |  |  |  |  |
| eval-02 |  |  |  |  |  |  |
| eval-03 |  |  |  |  |  |  |
| eval-04 |  |  |  |  |  |  |
| eval-05 |  |  |  |  |  |  |
| eval-06 |  |  |  |  |  |  |
| eval-07 |  |  |  |  |  |  |
| eval-08 |  |  |  |  |  |  |

Also note repeated layouts, copied-looking training images, artifacts, loss of variety, or ignored prompt details.

## Conclusion

Fill this in after the matched comparison is reviewed. Do not treat a completed training run without that comparison as a successful MVP.
