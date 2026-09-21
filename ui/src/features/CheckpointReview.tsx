import { useEffect, useMemo, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Images, Maximize2, RefreshCw, X } from "lucide-react";
import { api } from "../api";
import { EmptyState, PageHeader, StatusMessage } from "../components/Shared";
import type { CheckpointExperiment } from "../types";

export function CheckpointReview() {
  const [experiments, setExperiments] = useState<CheckpointExperiment[]>([]);
  const [experimentIndex, setExperimentIndex] = useState(0);
  const [promptIndex, setPromptIndex] = useState(0);
  const [selectedStep, setSelectedStep] = useState("");
  const [promptChoices, setPromptChoices] = useState<Record<string, string>>({});
  const [notes, setNotes] = useState("");
  const [zoom, setZoom] = useState<string | null>(null);
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
  const prompt = experiment?.prompts[promptIndex];
  const steps = useMemo(() => experiment?.steps.map(String) ?? [], [experiment]);
  const promptChoiceCount = experiment ? experiment.prompts.filter((item) => promptChoices[item.id]).length : 0;

  useEffect(() => {
    if (experiment && !selectedStep) setSelectedStep(steps[Math.floor(steps.length / 2)] ?? "");
  }, [experiment, selectedStep, steps]);

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

  if (state === "loading") return <StatusMessage kind="loading">Loading development renders…</StatusMessage>;
  if (!experiment || !prompt) {
    return (
      <div className="workflow">
        <PageHeader eyebrow="Training · Checkpoint selection" title="Find the stopping point" description="Compare the same development prompts at every saved checkpoint." />
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

  return (
    <div className="workflow checkpoint-workflow">
      <PageHeader
        eyebrow="Training · Checkpoint selection"
        title="Find the stopping point"
        description="Scan for improvement, then watch for memorization, rigidity, or loss of prompt fidelity. Final evaluation remains held out."
        actions={experiments.length > 1 ? (
          <select className="select-control" value={experimentIndex} onChange={(event) => { setExperimentIndex(Number(event.target.value)); setPromptIndex(0); setSelectedStep(""); }}>
            {experiments.map((item, index) => <option value={index} key={item.id}>{item.name ?? item.id}</option>)}
          </select>
        ) : <div className="progress-chip">{promptIndex + 1} / {experiment.prompts.length} prompts</div>}
      />

      <section className="checkpoint-layout">
        <aside className="prompt-rail">
          <p className="section-label">Development prompts</p>
          {experiment.prompts.map((item, index) => {
            const thumbnail = item.samples[steps[0]];
            return (
              <button type="button" key={item.id} className={index === promptIndex ? "prompt-card active" : "prompt-card"} onClick={() => setPromptIndex(index)}>
                {thumbnail ? <img src={thumbnail} alt="" /> : <span className="prompt-placeholder" />}
                <span><strong>{item.id}</strong><span>{item.prompt}</span></span>
                {promptChoices[item.id] && <Check size={13} className="prompt-check" />}
              </button>
            );
          })}
        </aside>

        <div className="checkpoint-main">
          <div className="prompt-copy">
            <p><span>Prompt</span>{prompt.prompt}</p>
            <span>Seed {prompt.seed}</span>
          </div>

          <div className="checkpoint-strip">
            {steps.map((step) => {
              const src = prompt.samples[step];
              const chosen = promptChoices[prompt.id] === step;
              return (
                <article key={step} className={chosen ? "checkpoint-tile chosen" : "checkpoint-tile"}>
                  <button type="button" className="checkpoint-image" onClick={() => src && setZoom(src)} disabled={!src}>
                    {src ? <img src={src} alt={`${prompt.id} at step ${step}`} /> : <span>No render</span>}
                    {src && <Maximize2 size={15} />}
                  </button>
                  <div>
                    <span className="step-label">{step === "base" ? "Base" : `Step ${step}`}</span>
                    <button type="button" className={chosen ? "mark-best active" : "mark-best"} disabled={!src} onClick={() => setPromptChoices((value) => ({ ...value, [prompt.id]: step }))}>
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
              {promptIndex < experiment.prompts.length - 1 ? (
                <button className="secondary-button" type="button" onClick={() => setPromptIndex(promptIndex + 1)}>Next prompt <ChevronRight size={16} /></button>
              ) : (
                <button
                  className="primary-button"
                  type="button"
                  onClick={saveSelection}
                  disabled={!selectedStep || state === "saving" || promptChoiceCount !== experiment.prompts.length}
                  title={promptChoiceCount === experiment.prompts.length ? "Save the stopping point" : `Mark a best render for all ${experiment.prompts.length} prompts first`}
                >
                  {state === "saving" ? "Saving…" : `Select ${selectedStep} steps`}
                </button>
              )}
            </div>
          </div>
        </div>
      </section>

      {zoom && (
        <div className="lightbox" role="dialog" aria-modal="true" aria-label="Checkpoint image detail" onClick={() => setZoom(null)}>
          <button className="lightbox-close" type="button" onClick={() => setZoom(null)} aria-label="Close image"><X /></button>
          <img src={zoom} alt="Checkpoint detail" />
        </div>
      )}
    </div>
  );
}
