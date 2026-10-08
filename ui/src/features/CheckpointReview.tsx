import { useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronDown, ChevronLeft, ChevronRight, Images, Maximize2, RefreshCw, X } from "lucide-react";
import { api } from "../api";
import { EmptyState, PageHeader, StatusMessage } from "../components/Shared";
import type { CheckpointExperiment, CheckpointPrompt, CheckpointSelection } from "../types";

interface SeedSample {
  id: string;
  seed: number;
  samples: Record<string, string>;
}

interface PromptGroup {
  id: string;
  prompt: string;
  seeds: SeedSample[];
}

interface ZoomTarget {
  step: string;
  seedIndex: number;
}

interface AppliedReview {
  step: string;
  promptChoices: Record<string, string>;
  notes: string;
  selectedAt: string;
}

function stepName(step: string) {
  return step === "base" ? "Base" : `Step ${step}`;
}

function experimentSteps(experiment: CheckpointExperiment) {
  return experiment.steps.map(String);
}

function groupPrompts(prompts: CheckpointPrompt[]): PromptGroup[] {
  const groups: PromptGroup[] = [];
  const byId = new Map<string, PromptGroup>();
  for (const item of prompts) {
    const id = item.groupId ?? item.id;
    let group = byId.get(id);
    if (!group) {
      group = { id, prompt: item.prompt, seeds: [] };
      byId.set(id, group);
      groups.push(group);
    }
    group.seeds.push({ id: item.id, seed: item.seed, samples: item.samples });
  }
  return groups;
}

function formatSavedAt(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function reviewStateForExperiment(
  experiment: CheckpointExperiment,
  selection: CheckpointSelection | null,
): AppliedReview | null {
  if (!selection || selection.experimentId !== experiment.id) return null;
  const steps = new Set(experimentSteps(experiment));
  if (!steps.has(String(selection.selectedStep))) return null;
  const promptChoices: Record<string, string> = {};
  for (const [id, step] of Object.entries(selection.promptChoices ?? {})) {
    if (steps.has(String(step))) promptChoices[id] = String(step);
  }
  return {
    step: String(selection.selectedStep),
    promptChoices,
    notes: selection.notes ?? "",
    selectedAt: selection.selectedAt,
  };
}

function errorText(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function unavailableReviewMessage(experiments: CheckpointExperiment[], selection: CheckpointSelection | null) {
  if (!selection?.selectedStep) return null;
  const match = experiments.find((item) => item.id === selection.experimentId);
  if (!match) {
    return `A saved review for ${selection.experimentId} is not in this list.`;
  }
  if (!experimentSteps(match).includes(String(selection.selectedStep))) {
    const name = match.name ?? match.id;
    return `The saved review for ${name} uses step ${selection.selectedStep}, which is no longer in that experiment.`;
  }
  return null;
}

function ExperimentChooser({
  experiments,
  selection,
  selectionMessage,
  onCompare,
  onOpenReview,
}: {
  experiments: CheckpointExperiment[];
  selection: CheckpointSelection | null;
  selectionMessage: string;
  onCompare: (index: number) => void;
  onOpenReview: (index: number) => void;
}) {
  const unavailable = unavailableReviewMessage(experiments, selection);
  return (
    <div className="workflow checkpoint-workflow">
      <PageHeader
        eyebrow="Training · Checkpoint selection"
        title="Choose an experiment"
        description="Pick a development run to compare checkpoints. A saved review stays closed until you open it."
      />
      {selectionMessage && <p className="chooser-note">{selectionMessage}</p>}
      {unavailable && <p className="chooser-note">{unavailable}</p>}
      <div className="experiment-list">
        {experiments.map((experiment, index) => {
          const review = reviewStateForExperiment(experiment, selection);
          const promptCount = groupPrompts(experiment.prompts).length;
          return (
            <article className="experiment-card" key={experiment.id}>
              <div>
                <h2>{experiment.name ?? experiment.id}</h2>
                <p className="experiment-meta">{promptCount} prompts · {experiment.steps.length} steps</p>
              </div>
              {review && <p>Saved review: step {review.step} · {formatSavedAt(review.selectedAt)}</p>}
              <div className="experiment-actions">
                {review && (
                  <button className="primary-button" type="button" onClick={() => onOpenReview(index)}>Open saved review</button>
                )}
                <button className={review ? "secondary-button" : "primary-button"} type="button" onClick={() => onCompare(index)}>
                  Compare checkpoints
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

export function CheckpointReview() {
  const [experiments, setExperiments] = useState<CheckpointExperiment[]>([]);
  const [savedSelection, setSavedSelection] = useState<CheckpointSelection | null>(null);
  const [experimentIndex, setExperimentIndex] = useState<number | null>(null);
  const [promptIndex, setPromptIndex] = useState(0);
  const [selectedStep, setSelectedStep] = useState("");
  const [promptChoices, setPromptChoices] = useState<Record<string, string>>({});
  const [notes, setNotes] = useState("");
  const [reviewVisible, setReviewVisible] = useState(false);
  const [pinnedStep, setPinnedStep] = useState<string | null>(null);
  const [zoom, setZoom] = useState<ZoomTarget | null>(null);
  const [included, setIncluded] = useState<string[] | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "saving" | "saved" | "error">("loading");
  const [message, setMessage] = useState("");
  const [selectionMessage, setSelectionMessage] = useState("");
  const saveRequest = useRef(0);
  const savingRef = useRef(false);
  const viewedExperimentId = useRef<string | null>(null);

  function load() {
    setState("loading");
    setMessage("");
    Promise.allSettled([api.checkpoints(), api.checkpointSelection()]).then(([experimentsResult, selectionResult]) => {
      if (experimentsResult.status === "rejected") {
        setExperiments([]);
        setSavedSelection(null);
        setSelectionMessage("");
        setMessage(errorText(experimentsResult.reason, "Could not load development renders."));
        setState("error");
        return;
      }
      setExperiments(experimentsResult.value.experiments);
      if (selectionResult.status === "fulfilled") {
        setSavedSelection(selectionResult.value.selection);
        setSelectionMessage("");
      } else {
        setSavedSelection(null);
        setSelectionMessage(errorText(selectionResult.reason, "Could not load the saved review."));
      }
      setState("ready");
    });
  }

  useEffect(load, []);

  const experiment = experimentIndex === null ? undefined : experiments[experimentIndex];
  viewedExperimentId.current = experiment?.id ?? null;
  const groups = useMemo(() => groupPrompts(experiment?.prompts ?? []), [experiment]);
  const group = groups[promptIndex];
  const steps = useMemo(() => experiment ? experimentSteps(experiment) : [], [experiment]);
  const appliedReview = experiment ? reviewStateForExperiment(experiment, savedSelection) : null;
  const weights = useMemo(() => {
    if (experiment?.weights?.length) return experiment.weights;
    return steps.map((step) => ({
      step,
      label: step === "base" ? "base" : `checkpoint-${step}/pytorch_lora_weights.safetensors`,
    }));
  }, [experiment, steps]);
  const visibleSteps = useMemo(
    () => weights.filter((weight) => (included ?? weights.map((item) => item.step)).includes(weight.step)).map((weight) => weight.step),
    [included, weights],
  );
  const markedCount = groups.filter((item) => item.seeds.every((seed) => promptChoices[seed.id])).length;

  useEffect(() => {
    setIncluded(null);
  }, [experiment?.id]);

  useEffect(() => {
    if (experiment && !selectedStep) setSelectedStep(steps[Math.floor(steps.length / 2)] ?? "");
  }, [experiment, selectedStep, steps]);

  useEffect(() => {
    setZoom(null);
  }, [promptIndex, experiment?.id]);

  useEffect(() => {
    if (zoom && !visibleSteps.includes(zoom.step)) setZoom(null);
  }, [visibleSteps, zoom]);

  useEffect(() => {
    if (!zoom || !group) return;
    const current = zoom;
    function onKey(event: KeyboardEvent) {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Escape"].includes(event.key)) return;
      event.preventDefault();
      if (event.key === "Escape") {
        setZoom(null);
        return;
      }
      if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        const seedIndex = current.seedIndex + (event.key === "ArrowDown" ? 1 : -1);
        const seed = group.seeds[seedIndex];
        if (seed?.samples[current.step]) setZoom({ step: current.step, seedIndex });
        return;
      }
      const index = visibleSteps.indexOf(current.step);
      if (index < 0) return;
      const step = visibleSteps[index + (event.key === "ArrowRight" ? 1 : -1)];
      const seed = group.seeds[current.seedIndex];
      if (step && seed?.samples[step]) setZoom({ step, seedIndex: current.seedIndex });
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [group, visibleSteps, zoom]);

  useEffect(() => {
    if (!zoom) return;
    document.querySelector<HTMLElement>(`.checkpoint-tile[data-step="${zoom.step}"]`)?.scrollIntoView({
      inline: "nearest",
      block: "nearest",
    });
  }, [zoom]);

  useEffect(() => {
    if (!pinnedStep || !experiment) return;
    document.querySelector<HTMLElement>(`.checkpoint-tile[data-step="${pinnedStep}"]`)?.scrollIntoView({
      inline: "center",
      block: "nearest",
    });
    setPinnedStep(null);
  }, [pinnedStep, experiment]);

  function clearComparison() {
    setPromptIndex(0);
    setSelectedStep("");
    setPromptChoices({});
    setNotes("");
    setReviewVisible(false);
    setZoom(null);
    setPinnedStep(null);
    setMessage("");
  }

  function compareExperiment(index: number) {
    if (savingRef.current) return;
    setExperimentIndex(index);
    clearComparison();
    if (state !== "loading") setState("ready");
  }

  function openSavedReview(index: number) {
    if (savingRef.current) return;
    const next = experiments[index];
    const review = next ? reviewStateForExperiment(next, savedSelection) : null;
    if (!next || !review) return;
    setExperimentIndex(index);
    setPromptIndex(0);
    setSelectedStep(review.step);
    setPromptChoices(review.promptChoices);
    setNotes(review.notes);
    setReviewVisible(true);
    setZoom(null);
    setPinnedStep(review.step);
    setIncluded(null);
    setMessage("");
    if (state !== "loading") setState("ready");
  }

  function leaveExperiment() {
    if (savingRef.current) return;
    setExperimentIndex(null);
    setIncluded(null);
    clearComparison();
    if (state !== "loading") setState("ready");
  }

  async function saveSelection() {
    if (!experiment || !selectedStep || savingRef.current) return;
    const requestId = saveRequest.current + 1;
    saveRequest.current = requestId;
    const experimentId = experiment.id;
    savingRef.current = true;
    setState("saving");
    try {
      const saved = await api.saveCheckpoint({ experimentId, step: selectedStep, promptChoices, notes });
      if (saveRequest.current !== requestId) return;
      setSavedSelection(saved);
      if (viewedExperimentId.current !== experimentId) {
        setState("ready");
        return;
      }
      setNotes(saved.notes);
      setReviewVisible(true);
      setState("saved");
    } catch (error) {
      if (saveRequest.current !== requestId) return;
      if (viewedExperimentId.current !== experimentId) {
        setState("ready");
        return;
      }
      setMessage(errorText(error, "Could not save checkpoint selection."));
      setState("error");
    } finally {
      if (saveRequest.current === requestId) savingRef.current = false;
    }
  }

  function markGroup(step: string) {
    if (savingRef.current || !group) return;
    setPromptChoices((value) => {
      const next = { ...value };
      for (const seed of group.seeds) next[seed.id] = step;
      return next;
    });
  }

  if (state === "loading") return <StatusMessage kind="loading">Loading development renders…</StatusMessage>;
  if (state === "error" && experiments.length === 0) {
    return (
      <div className="workflow">
        <PageHeader eyebrow="Training · Checkpoint selection" title="Choose an experiment" description="Pick a development run to compare checkpoints." />
        <EmptyState
          icon={<Images size={26} />}
          title="Could not load development renders"
          body={message || "The development manifest could not be read."}
        >
          <button className="secondary-button" type="button" onClick={load}><RefreshCw size={16} /> Check again</button>
        </EmptyState>
      </div>
    );
  }
  if (experiments.length === 0) {
    return (
      <div className="workflow">
        <PageHeader eyebrow="Training · Checkpoint selection" title="Choose an experiment" description="Pick a development run to compare checkpoints." />
        <EmptyState
          icon={<Images size={26} />}
          title="No checkpoint comparison yet"
          body="Once a development render manifest is written under outputs/development, every prompt and checkpoint will appear here automatically."
        >
          <button className="secondary-button" type="button" onClick={load}><RefreshCw size={16} /> Check again</button>
        </EmptyState>
      </div>
    );
  }
  if (!experiment || !group) {
    return (
      <ExperimentChooser
        experiments={experiments}
        selection={savedSelection}
        selectionMessage={selectionMessage}
        onCompare={compareExperiment}
        onOpenReview={openSavedReview}
      />
    );
  }

  const saving = state === "saving";
  const zoomSeed = zoom ? group.seeds[zoom.seedIndex] : undefined;
  const zoomSrc = zoom && zoomSeed ? zoomSeed.samples[zoom.step] : undefined;

  return (
    <div className="workflow checkpoint-workflow">
      <PageHeader
        eyebrow="Training · Checkpoint selection"
        title="Find the stopping point"
        description="Each checkpoint shows that prompt’s seeds together. Mark the best step for the prompt, then lock one stopping point. The held-out A/B test stays closed until that choice is saved."
      />

      <div className="comparison-toolbar">
        <div>
          <button className="secondary-button" type="button" onClick={leaveExperiment} disabled={saving}>Experiments</button>
          {appliedReview && (
            <button className="secondary-button" type="button" onClick={() => openSavedReview(experimentIndex ?? 0)} disabled={saving}>Open saved review</button>
          )}
        </div>
        <div className="progress-chip">{promptIndex + 1} / {groups.length} prompts</div>
      </div>

      <section className="checkpoint-layout">
        <aside className="prompt-rail">
          <p className="section-label">Development prompts</p>
          {groups.map((item, index) => {
            const thumbnail = item.seeds.map((seed) => seed.samples[visibleSteps[0] ?? steps[0]]).find(Boolean);
            const marked = item.seeds.every((seed) => promptChoices[seed.id]);
            return (
              <button type="button" key={item.id} className={index === promptIndex ? "prompt-card active" : "prompt-card"} onClick={() => setPromptIndex(index)} disabled={saving}>
                {thumbnail ? <img src={thumbnail} alt="" /> : <span className="prompt-placeholder" />}
                <span><strong>{item.id}</strong><span>{item.prompt}</span></span>
                {marked && <Check size={13} className="prompt-check" />}
              </button>
            );
          })}
        </aside>

        <div className="checkpoint-main">
          <div className="prompt-copy">
            <p><span>Prompt</span>{group.prompt}</p>
            <span>{group.seeds.length === 1 ? `Seed ${group.seeds[0].seed}` : `${group.seeds.length} seeds`}</span>
          </div>

          <details className="weight-panel">
            <summary>
              <ChevronDown size={14} aria-hidden="true" />
              <span>Models</span>
              <span className="weight-count">{visibleSteps.length} of {weights.length}</span>
            </summary>
            <ul className="weight-picks">
              {weights.map((weight) => (
                <li key={weight.step}>
                  <label>
                    <input
                      type="checkbox"
                      checked={visibleSteps.includes(weight.step)}
                      disabled={saving}
                      onChange={() => setIncluded((current) => {
                        const selected = current ?? weights.map((item) => item.step);
                        return selected.includes(weight.step)
                          ? selected.filter((step) => step !== weight.step)
                          : [...selected, weight.step];
                      })}
                    />
                    <span>{weight.label}</span>
                  </label>
                </li>
              ))}
            </ul>
          </details>

          <div className="checkpoint-strip">
            {visibleSteps.length === 0 && <p className="section-label">Check at least one weights file to compare.</p>}
            {visibleSteps.map((step) => {
              const ready = group.seeds.every((seed) => seed.samples[step]);
              const chosen = group.seeds.every((seed) => promptChoices[seed.id] === step);
              return (
                <article key={step} data-step={step} className={chosen ? "checkpoint-tile chosen" : "checkpoint-tile"}>
                  <div className="seed-sheet" data-count={group.seeds.length}>
                    {group.seeds.map((seed, index) => {
                      const src = seed.samples[step];
                      return (
                        <button key={seed.id} type="button" className="seed-cell" onClick={() => src && setZoom({ step, seedIndex: index })} disabled={!src}>
                          {src ? <img src={src} alt={`${group.id} seed ${seed.seed} at ${stepName(step)}`} /> : <span className="seed-missing">No render</span>}
                          {group.seeds.length > 1 && <span className="seed-label">{seed.seed}</span>}
                          {group.seeds.length === 1 && src && <Maximize2 size={15} />}
                        </button>
                      );
                    })}
                  </div>
                  <div className="tile-meta">
                    <span className="step-label">{step === "base" ? "Base" : `Step ${step}`}</span>
                    <button type="button" className={chosen ? "mark-best active" : "mark-best"} disabled={!ready || saving} onClick={() => markGroup(step)}>
                      {chosen ? <><Check size={13} /> Best here</> : "Mark best"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>

          <div className="checkpoint-footer">
            <div className="current-choice">
              <span>Overall stopping point</span>
              <div className="step-selector">
                {steps.map((step) => (
                  <button type="button" key={step} className={selectedStep === step ? "active" : ""} onClick={() => setSelectedStep(step)} disabled={saving}>{step}</button>
                ))}
              </div>
            </div>
            {reviewVisible && appliedReview && (
              <p className="saved-review-line">Saved stopping point: step {appliedReview.step} · {formatSavedAt(appliedReview.selectedAt)}</p>
            )}
            <textarea className="checkpoint-notes" value={notes} onChange={(event) => setNotes(event.target.value)} disabled={saving} placeholder="Selection notes: where quality peaks, where overfitting begins…" />
            {state === "error" && <StatusMessage kind="error">{message}</StatusMessage>}
            {state === "saved" && <StatusMessage kind="saved">Checkpoint decision saved.</StatusMessage>}
            <div className="decision-actions">
              <button className="secondary-button" type="button" disabled={promptIndex === 0 || saving} onClick={() => setPromptIndex(promptIndex - 1)}><ChevronLeft size={16} /> Previous prompt</button>
              {promptIndex < groups.length - 1 ? (
                <button className="secondary-button" type="button" disabled={saving} onClick={() => setPromptIndex(promptIndex + 1)}>Next prompt <ChevronRight size={16} /></button>
              ) : (
                <button
                  className="primary-button"
                  type="button"
                  onClick={saveSelection}
                  disabled={!selectedStep || state === "saving" || markedCount !== groups.length}
                  title={markedCount === groups.length ? "Save the stopping point" : `Mark a best checkpoint for all ${groups.length} prompts first`}
                >
                  {state === "saving" ? "Saving…" : `Select ${selectedStep} steps`}
                </button>
              )}
            </div>
          </div>
        </div>
      </section>

      {zoom && zoomSrc && zoomSeed && (
        <div className="lightbox" role="dialog" aria-modal="true" aria-label={`${group.id} seed ${zoomSeed.seed}, ${stepName(zoom.step)}`} onClick={() => setZoom(null)}>
          <button className="lightbox-close" type="button" onClick={() => setZoom(null)} aria-label="Close image"><X /></button>
          <div className="lightbox-prompt" onClick={(event) => event.stopPropagation()}>
            <p><span>{group.id}</span>{group.prompt}</p>
            <span>Seed {zoomSeed.seed}</span>
          </div>
          <figure>
            <img src={zoomSrc} alt={`${group.id} seed ${zoomSeed.seed} at ${stepName(zoom.step)}`} />
            <figcaption>
              <strong>{stepName(zoom.step)}</strong>
              <span>{visibleSteps.indexOf(zoom.step) + 1} / {visibleSteps.length}</span>
              <span>{group.seeds.length > 1 ? "↑↓ seed  ·  ←→ step" : "Arrow keys to compare"}</span>
            </figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}
