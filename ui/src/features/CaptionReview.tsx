import { useEffect, useMemo, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Copy, PencilLine, RotateCcw } from "lucide-react";
import { api } from "../api";
import { PageHeader, StatusMessage } from "../components/Shared";
import type { CaptionItem, CaptionSelection, CaptionSource } from "../types";

const blank: CaptionSelection = { source: "codex", customCaption: null, include: true };

export function CaptionReview() {
  const [items, setItems] = useState<CaptionItem[]>([]);
  const [index, setIndex] = useState(0);
  const [draft, setDraft] = useState<CaptionSelection>(blank);
  const [state, setState] = useState<"loading" | "ready" | "saving" | "saved" | "error">("loading");
  const [message, setMessage] = useState("");
  const [copiedSource, setCopiedSource] = useState<CaptionSource | null>(null);

  useEffect(() => {
    api.captions()
      .then(({ items: loaded }) => {
        setItems(loaded);
        setState("ready");
      })
      .catch((error) => { setMessage(error.message); setState("error"); });
  }, []);

  const current = items[index];
  useEffect(() => {
    if (current) {
      setDraft(current.selection ?? blank);
      setState("ready");
    }
  }, [current]);

  const reviewed = useMemo(() => items.filter((item) => item.selection).length, [items]);

  function choose(source: CaptionSource) {
    setDraft((value) => ({ ...value, source }));
    setState("ready");
  }

  function editCopy(text: string) {
    setDraft((value) => ({ ...value, source: "custom", customCaption: text }));
    setCopiedSource(null);
    setState("ready");
  }

  async function copyCaption(source: CaptionSource, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedSource(source);
    } catch {
      setMessage("Clipboard access was unavailable. The caption text is selectable, so you can still copy it normally.");
      setState("error");
    }
  }

  async function save(move = 0) {
    if (!current) return;
    setState("saving");
    try {
      const selection = await api.saveCaption(current.id, draft);
      setItems((value) => value.map((item) => item.id === current.id ? { ...item, selection } : item));
      setState("saved");
      if (move) setIndex((value) => Math.max(0, Math.min(items.length - 1, value + move)));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not save the caption.");
      setState("error");
    }
  }

  if (state === "loading") return <StatusMessage kind="loading">Loading training images…</StatusMessage>;
  if (!current) return <StatusMessage kind="error">No training images were found in the data folder.</StatusMessage>;

  const options: Array<{ source: CaptionSource; label: string; text: string }> = [
    { source: "codex", label: "Codex", text: current.captions.codex ?? "No Codex caption found." },
    { source: "grok", label: "Grok", text: current.captions.grok ?? "No Grok caption found." },
  ];

  return (
    <div className="workflow caption-workflow">
      <PageHeader
        eyebrow="Dataset · Caption review"
        title="Choose the cleanest description"
        description="Select the caption that best describes visible content. Style language stays out so the adapter learns it from the images."
        actions={<div className="progress-chip"><strong>{reviewed}</strong> / {items.length} reviewed</div>}
      />

      <div className="caption-grid">
        <section className="hero-image-panel">
          <img src={current.imageUrl} alt={`Training image ${current.filename}`} />
          <div className="image-meta"><span>{String(index + 1).padStart(2, "0")}</span><span>{current.filename}</span></div>
        </section>

        <section className="decision-panel">
          <div className="decision-heading">
            <span>Caption candidates</span>
            <button className="icon-button" type="button" onClick={() => setDraft(current.selection ?? blank)} aria-label="Reset this caption">
              <RotateCcw size={15} />
            </button>
          </div>
          <div className="caption-options">
            {options.map((option) => (
              <article
                key={option.source}
                className={draft.source === option.source ? "caption-option selected" : "caption-option"}
              >
                <button type="button" className="caption-option-select" onClick={() => choose(option.source)} aria-label={`Use ${option.label} caption`}>
                  <span className="radio-mark">{draft.source === option.source && <Check size={12} />}</span>
                  <strong>{option.label}</strong>
                </button>
                <p className="caption-copy-text">{option.text}</p>
                <div className="caption-option-actions">
                  <button type="button" onClick={() => copyCaption(option.source, option.text)}>
                    {copiedSource === option.source ? <Check size={13} /> : <Copy size={13} />}
                    {copiedSource === option.source ? "Copied" : "Copy"}
                  </button>
                  <button type="button" onClick={() => editCopy(option.text)}><PencilLine size={13} /> Edit a copy</button>
                </div>
              </article>
            ))}
            <button
              type="button"
              className={draft.source === "custom" ? "caption-option custom-caption-option selected" : "caption-option custom-caption-option"}
              onClick={() => choose("custom")}
            >
              <span className="radio-mark">{draft.source === "custom" && <Check size={12} />}</span>
              <span><strong>Write my own</strong><span>Use a precise content-only description.</span></span>
            </button>
            {draft.source === "custom" && (
              <textarea
                className="caption-textarea"
                value={draft.customCaption ?? ""}
                onChange={(event) => setDraft((value) => ({ ...value, customCaption: event.target.value }))}
                placeholder="Describe only what is visibly present…"
                rows={5}
                autoFocus
              />
            )}
          </div>

          <label className="include-row">
            <span><strong>Include in training</strong><span>Turn off if this image weakens the shared photographic look.</span></span>
            <input type="checkbox" checked={draft.include} onChange={(event) => setDraft((value) => ({ ...value, include: event.target.checked }))} />
            <span className="switch" aria-hidden="true" />
          </label>

          {state === "error" && <StatusMessage kind="error">{message}</StatusMessage>}
          {state === "saved" && <StatusMessage kind="saved">Saved to the dataset decision record.</StatusMessage>}

          <div className="decision-actions">
            <button className="secondary-button" type="button" disabled={index === 0} onClick={() => setIndex(index - 1)}><ChevronLeft size={16} /> Previous</button>
            <button
              className="primary-button"
              type="button"
              disabled={state === "saving" || (draft.source === "custom" && !draft.customCaption?.trim())}
              onClick={() => save(1)}
            >
              {state === "saving" ? "Saving…" : index === items.length - 1 ? "Save choice" : "Save & next"}
              {index < items.length - 1 && <ChevronRight size={16} />}
            </button>
          </div>
        </section>
      </div>

      <div className="filmstrip" aria-label="Training image navigation">
        {items.map((item, itemIndex) => (
          <button
            key={item.id}
            type="button"
            className={itemIndex === index ? "active" : item.selection ? "reviewed" : ""}
            onClick={() => setIndex(itemIndex)}
            aria-label={`Open image ${itemIndex + 1}`}
          >
            <img src={item.imageUrl} alt="" loading="lazy" />
            {item.selection && <span className="filmstrip-check"><Check size={10} /></span>}
          </button>
        ))}
      </div>
    </div>
  );
}
