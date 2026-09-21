# Caption candidates

Caption sets in this directory are drafts for comparison, not the final training
dataset. Each caption filename matches the stem of a source image in `data/`.

- `codex/`: visually reviewed Codex captions, revised 2026-09-20
  (`codex-2026-09-20-v2`) to describe visible image contents only.
- `grok/`: draft Grok captions, revised 2026-09-20 (`grok-2026-09-20-v2`) to
  describe visible image contents only.

Keep caption generators separate until the final 18–24 training images and their
captions have been reviewed. To promote a caption, copy the selected source image
and exactly one chosen caption into `data/train/` with the same stem. Do not mix
caption wording from different generators without recording that edit.

For the style LoRA, the intended training captions name the variable content in
the picture and leave the shared lighting, camera technique, composition, color
grade, and commercial finish uncaptioned, so that look can be learned from the
images. They do not use a synthetic trigger token, infer ingredients that are
not visibly clear, or claim that every candidate belongs in the final coherent
style set. With no trigger token, loading and scaling the adapter is what enables
the learned style.
