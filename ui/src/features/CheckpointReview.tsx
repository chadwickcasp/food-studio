import { useEffect, useMemo, useState } from "react";
import { Check, ChevronDown, ChevronLeft, ChevronRight, Images, Maximize2, RefreshCw, X } from "lucide-react";
import { api } from "../api";
import { EmptyState, PageHeader, StatusMessage } from "../components/Shared";
import type { CheckpointExperiment, CheckpointPrompt } from "../types";

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

function stepName(step: string) {
  return step === "base" ? "Base" : `Step ${step}`;
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

export function CheckpointReview() {
  const [experiments, setExperiments] = useState<CheckpointExperiment[]>([]);
  const [experimentIndex, setExperimentIndex] = useState(0);
  const [promptIndex, setPromptIndex] = useState(0);
  const [selectedStep, setSelectedStep] = useState("");
  const [promptChoices, setPromptChoices] = useState<Record<string, string>>({});
  const [notes, setNotes] = useState("");
  const [zoom, setZoom] = useState<ZoomTarget | null>(null);
  const [included, setIncluded] = useState<string[] | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "saving" | "saved" | "error">("loading");
  const [message, setMessage] = useState("");

  function load() {
    setState("loading");
    api.checkpoints()
      .then(({ experiments: found }) => { setExperiments(found); setState("ready"); })
      .catch((error) => { setMessage(error.message); setState("error"); });
  }

  useEffect(load, []);

  const experiment = experiments[experimentIndex];
  const groups = useMemo(() => groupPrompts(experiment?.prompts ?? []), [experiment]);
  const group = groups[promptIndex];
  const steps = useMemo(() => experiment?.steps.map(String) ?? [], [experiment]);
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

  async function saveSelection() {
    if (!experiment || !selectedStep) return;
    setState("saving");
    try {
      await api.saveCheckpoint({ experimentId: experiment.id, step: selectedStep, promptChoices, notes });
      setState("saved");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not save checkpoint selection.");
      setState("error");
    }
  }

  function markGroup(step: string) {
    if (!group) return;
    setPromptChoices((value) => {
      const next = { ...value };
      for (const seed of group.seeds) next[seed.id] = step;
      return next;
    });
  }

  if (state === "loading") return <StatusMessage kind="loading">Loading development renders…</StatusMessage>;
  if (state === "error" && !experiment) {
    return (
      <div className="workflow">
        <PageHeader eyebrow="Training · Checkpoint selection" title="Find the stopping point" description="Compare each prompt’s seeds at every saved checkpoint." />
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
  if (!experiment || !group) {
    return (
      <div className="workflow">
        <PageHeader eyebrow="Training · Checkpoint selection" title="Find the stopping point" description="Compare each prompt’s seeds at every saved checkpoint." />
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

  const zoomSeed = zoom ? group.seeds[zoom.seedIndex] : undefined;
  const zoomSrc = zoom && zoomSeed ? zoomSeed.samples[zoom.step] : undefined;

  return (
    <div className="workflow checkpoint-workflow">
      <PageHeader
        eyebrow="Training · Checkpoint selection"
        title="Find the stopping point"
        description="Each checkpoint shows that prompt’s seeds together. Mark the best step for the prompt, then lock one stopping point. The held-out A/B test stays closed until that choice is saved."
        actions={experiments.length > 1 ? (
          <select className="select-control" value={experimentIndex} onChange={(event) => { setExperimentIndex(Number(event.target.value)); setPromptIndex(0); setSelectedStep(""); }}>
            {experiments.map((item, index) => <option value={index} key={item.id}>{item.name ?? item.id}</option>)}
          </select>
        ) : <div className="progress-chip">{promptIndex + 1} / {groups.length} prompts</div>}
      />

      <section className="checkpoint-layout">
        <aside className="prompt-rail">
          <p className="section-label">Development prompts</p>
          {groups.map((item, index) => {
            const thumbnail = item.seeds.map((seed) => seed.samples[visibleSteps[0] ?? steps[0]]).find(Boolean);
            const marked = item.seeds.every((seed) => promptChoices[seed.id]);
            return (
              <button type="button" key={item.id} className={index === promptIndex ? "prompt-card active" : "prompt-card"} onClick={() => setPromptIndex(index)}>
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
                    <button type="button" className={chosen ? "mark-best active" : "mark-best"} disabled={!ready} onClick={() => markGroup(step)}>
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
                  <button type="button" key={step} className={selectedStep === step ? "active" : ""} onClick={() => setSelectedStep(step)}>{step}</button>
                ))}
              </div>
            </div>
            <textarea value={notes} onChange={(event) => setNotes(event.target.value)} rows={2} placeholder="Selection notes: where quality peaks, where overfitting begins…" />
            {state === "error" && <StatusMessage kind="error">{message}</StatusMessage>}
            {state === "saved" && <StatusMessage kind="saved">Checkpoint decision saved.</StatusMessage>}
            <div className="decision-actions">
              <button className="secondary-button" type="button" disabled={promptIndex === 0} onClick={() => setPromptIndex(promptIndex - 1)}><ChevronLeft size={16} /> Previous prompt</button>
              {promptIndex < groups.length - 1 ? (
                <button className="secondary-button" type="button" onClick={() => setPromptIndex(promptIndex + 1)}>Next prompt <ChevronRight size={16} /></button>
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
