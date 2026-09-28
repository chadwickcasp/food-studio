import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Eye, FlaskConical, LockKeyhole, Plus, RefreshCw } from "lucide-react";
import { api } from "../api";
import { EmptyState, PageHeader, StatusMessage } from "../components/Shared";
import type { AbResponse, AbReveal, AbSession, AbSessionSummary, CheckpointSelection, RunSummary } from "../types";

const blankResponse = (criteria: string[]): AbResponse => ({
  choices: Object.fromEntries(criteria.map((criterion) => [criterion, null])),
  notes: "",
});

function choiceMatches(saved: string | null | undefined, option: string) {
  if (!saved) return false;
  return saved.toLowerCase() === option.toLowerCase();
}

export function AbReview() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sessions, setSessions] = useState<AbSessionSummary[]>([]);
  const [active, setActive] = useState<AbSession | null>(null);
  const [itemIndex, setItemIndex] = useState(0);
  const [draft, setDraft] = useState<AbResponse>({ choices: {}, notes: "" });
  const [reveal, setReveal] = useState<AbReveal | null>(null);
  const [selection, setSelection] = useState<CheckpointSelection | null>(null);
  const [name, setName] = useState("");
  const [nameEdited, setNameEdited] = useState(false);
  const [state, setState] = useState<"loading" | "ready" | "saving" | "saved" | "error">("loading");
  const [message, setMessage] = useState("");

  function loadIndex() {
    setState("loading");
    Promise.all([api.runs(), api.abSessions(), api.checkpointSelection()])
      .then(([runData, sessionData, selectionData]) => {
        setRuns(runData.runs);
        setSessions(sessionData.sessions);
        setSelection(selectionData.selection);
        setState("ready");
      })
      .catch((error) => { setMessage(error.message); setState("error"); });
  }

  useEffect(loadIndex, []);

  const item = active?.items[itemIndex];
  useEffect(() => {
    if (active && item) setDraft(active.responses[item.id] ?? blankResponse(active.criteria));
  }, [active, item]);

  const completed = useMemo(() => Object.keys(active?.responses ?? {}).length, [active]);
  const isComplete = Boolean(active && completed === active.items.length);
  const letters = item?.images.map((image) => image.letter) ?? [];
  const draftComplete = Boolean(active && active.criteria.every((criterion) => draft.choices[criterion]));
  const tally = useMemo(() => {
    if (!active || !reveal) return null;
    const wins = new Map<string, number>();
    let ties = 0;
    for (const [itemId, response] of Object.entries(active.responses)) {
      const assignment = reveal.assignments[itemId];
      if (!assignment) continue;
      for (const choice of Object.values(response.choices)) {
        if (!choice || choice.toLowerCase() === "tie") {
          if (choice) ties += 1;
          continue;
        }
        const label = assignment[choice.toUpperCase()];
        if (!label) continue;
        wins.set(label, (wins.get(label) ?? 0) + 1);
      }
    }
    return { wins: [...wins.entries()].sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0])), ties };
  }, [active, reveal]);

  const heldOut = useMemo(() => {
    const validation = runs.filter((run) => run.promptSet === "validation_prompts.json");
    const step = selection?.selectedStep ?? "";
    const chosenLabel = step && step !== "base" ? `checkpoint-${step}/pytorch_lora_weights.safetensors` : "";
    return {
      base: validation.find((run) => run.weightsLabel === "base") ?? null,
      chosen: chosenLabel ? validation.find((run) => run.weightsLabel === chosenLabel) ?? null : null,
      chosenLabel,
    };
  }, [runs, selection]);

  useEffect(() => {
    if (!nameEdited && heldOut.chosenLabel) setName(`Validation · ${heldOut.chosenLabel}`);
  }, [heldOut.chosenLabel, nameEdited]);

  async function openSession(id: string) {
    setState("loading");
    try {
      const value = await api.abSession(id);
      setActive(value);
      setItemIndex(0);
      setReveal(null);
      setState("ready");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not open this review.");
      setState("error");
    }
  }

  async function createSession() {
    if (!heldOut.base || !heldOut.chosen) return;
    setState("saving");
    try {
      const { id } = await api.createAbSession({ name, runs: [heldOut.base.id, heldOut.chosen.id] });
      await openSession(id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not create this review.");
      setState("error");
    }
  }

  async function save(move = 0) {
    if (!active || !item) return;
    setState("saving");
    try {
      const value = await api.saveAbResponse(active.id, item.id, draft);
      setActive((session) => session ? {
        ...session,
        responses: { ...session.responses, [item.id]: value },
      } : session);
      setState("saved");
      if (move) setItemIndex((index) => Math.max(0, Math.min(active.items.length - 1, index + move)));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not save this comparison.");
      setState("error");
    }
  }

  async function revealModels() {
    if (!active) return;
    setState("loading");
    try {
      setReveal(await api.revealAbSession(active.id));
      setState("ready");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not reveal model identities.");
      setState("error");
    }
  }

  if (state === "loading" && !active) return <StatusMessage kind="loading">Loading model runs and reviews…</StatusMessage>;

  if (!active) {
    return (
      <div className="workflow ab-setup">
        <PageHeader eyebrow="Evaluation · Blind comparison" title="Held-out validation" description="This test opens after a development checkpoint is locked. It compares base with that one safetensors file on the validation prompts." />
        {!selection ? (
          <EmptyState
            icon={<FlaskConical size={26} />}
            title="Choose a checkpoint first"
            body="Compare development renders on the Checkpoints screen and save a stopping point. This A/B test stays closed until that choice is locked."
          >
            <button className="secondary-button" type="button" onClick={loadIndex}><RefreshCw size={16} /> Check again</button>
          </EmptyState>
        ) : selection.selectedStep === "base" ? (
          <EmptyState
            icon={<FlaskConical size={26} />}
            title="The locked choice is base"
            body="The saved stopping point is the unmodified base model, so there is no LoRA to compare against it."
          />
        ) : !heldOut.base || !heldOut.chosen ? (
          <EmptyState
            icon={<FlaskConical size={26} />}
            title="Held-out renders are not ready"
            body={`The locked weights are ${heldOut.chosenLabel}. Generate that adapter, and base, on the validation prompts with the same seeds and settings. Development renders stay out of this test.`}
          >
            <button className="secondary-button" type="button" onClick={loadIndex}><RefreshCw size={16} /> Check again</button>
          </EmptyState>
        ) : (
          <section className="setup-card">
            <div className="setup-heading"><Plus size={18} /><div><h2>Base against the locked checkpoint</h2><p>Both sides use validation_prompts.json. Letter placement is randomized per prompt.</p></div></div>
            <ul className="weight-checks">
              {[heldOut.base, heldOut.chosen].map((run) => (
                <li key={run.id}>
                  <span>
                    <strong>{run.weightsLabel}</strong>
                    <span>{run.promptSet} · {run.sampleCount} images</span>
                  </span>
                </li>
              ))}
            </ul>
            <label>Review name<input value={name} onChange={(event) => { setNameEdited(true); setName(event.target.value); }} /></label>
            {state === "error" && <StatusMessage kind="error">{message}</StatusMessage>}
            <button className="primary-button" type="button" disabled={!name.trim() || state === "saving"} onClick={createSession}>
              {state === "saving" ? "Creating…" : "Start blinded review"}
            </button>
          </section>
        )}
        {sessions.length > 0 && (
          <section className="past-reviews">
            <p className="section-label">Saved reviews</p>
            {sessions.map((session) => (
              <button type="button" key={session.id} onClick={() => openSession(session.id)}>
                <span><strong>{session.name}</strong><span>{new Date(session.createdAt).toLocaleDateString()}</span></span>
                <span>{session.completed} / {session.total}</span>
              </button>
            ))}
          </section>
        )}
      </div>
    );
  }

  if (!item) return <StatusMessage kind="error">This review contains no comparison items.</StatusMessage>;

  const options = [...letters, "tie"];

  return (
    <div className="workflow ab-workflow">
      <PageHeader
        eyebrow="Evaluation · Blind comparison"
        title={active.name}
        description="Judge the photographic result, not the model label. Choose the stronger image for each criterion, or mark a tie."
        actions={<div className="progress-chip"><strong>{completed}</strong> / {active.items.length} complete</div>}
      />

      <div className="ab-prompt"><span>Prompt</span><p>{item.prompt}</p><em>Seed {item.seed}</em></div>

      <section className="ab-images">
        {item.images.map((image) => (
          <figure key={image.letter}>
            <div className="blind-label">{image.letter}</div>
            <img src={image.url} alt={`Blind comparison ${image.letter}`} />
          </figure>
        ))}
      </section>

      {reveal ? (
        <section className="reveal-panel">
          <Eye size={18} />
          <div>
            <strong>Models revealed</strong>
            <p>
              {item.images.map((image) => (
                <span key={image.letter}>{image.letter}: {reveal.assignments[item.id]?.[image.letter] ?? "unknown"}<br /></span>
              ))}
            </p>
            {tally && (
              <p className="reveal-tally">
                Criterion wins · {tally.wins.map(([label, count]) => `${label}: ${count}`).join(" · ") || "none"} · ties: {tally.ties}
              </p>
            )}
          </div>
        </section>
      ) : (
        <section className="scoring-panel">
          <div className="scoring-title"><span>Which result is stronger?</span><span>{options.join(" · ")}</span></div>
          {active.criteria.map((criterion) => (
            <div className="score-row" key={criterion}>
              <div><strong>{criterion}</strong>{criterion === "Artifacts" && <span>fewer visible failures wins</span>}</div>
              <div className="letter-choice">
                {options.map((option) => (
                  <button
                    key={option}
                    type="button"
                    className={choiceMatches(draft.choices[criterion], option) ? "active" : ""}
                    aria-pressed={choiceMatches(draft.choices[criterion], option)}
                    onClick={() => setDraft((value) => ({ ...value, choices: { ...value.choices, [criterion]: option === "tie" ? "tie" : option } }))}
                  >
                    {option === "tie" ? "Tie" : option}
                  </button>
                ))}
              </div>
            </div>
          ))}
          <label className="notes-field">Notes<textarea rows={3} value={draft.notes} onChange={(event) => setDraft({ ...draft, notes: event.target.value })} placeholder="Optional: record why one image won, or what failed…" /></label>
        </section>
      )}

      {state === "error" && <StatusMessage kind="error">{message}</StatusMessage>}
      {state === "saved" && <StatusMessage kind="saved">This comparison is saved.</StatusMessage>}

      <div className="ab-actions">
        <button className="secondary-button" type="button" disabled={itemIndex === 0} onClick={() => setItemIndex(itemIndex - 1)}><ChevronLeft size={16} /> Previous</button>
        <button className="quiet-button" type="button" onClick={revealModels} disabled={!isComplete || Boolean(reveal)} title={isComplete ? "Reveal model identities" : "Finish every comparison first"}>
          {reveal ? <Eye size={16} /> : <LockKeyhole size={16} />} {reveal ? "Revealed" : "Reveal models"}
        </button>
        {!reveal && (
          <button className="primary-button" type="button" disabled={state === "saving" || !draftComplete} title={draftComplete ? "Save this comparison" : "Choose a letter or a tie for every criterion"} onClick={() => save(itemIndex < active.items.length - 1 ? 1 : 0)}>
            {state === "saving" ? "Saving…" : itemIndex === active.items.length - 1 ? "Save comparison" : "Save & next"}
            {itemIndex < active.items.length - 1 && <ChevronRight size={16} />}
          </button>
        )}
      </div>
    </div>
  );
}
