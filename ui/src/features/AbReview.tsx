import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Eye, FlaskConical, LockKeyhole, Plus, RefreshCw } from "lucide-react";
import { api } from "../api";
import { EmptyState, PageHeader, SegmentedChoice, StatusMessage } from "../components/Shared";
import type { AbChoice, AbResponse, AbReveal, AbSession, AbSessionSummary, RunSummary } from "../types";

const blankResponse = (criteria: string[]): AbResponse => ({
  choices: Object.fromEntries(criteria.map((criterion) => [criterion, null])),
  notes: "",
});

export function AbReview() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sessions, setSessions] = useState<AbSessionSummary[]>([]);
  const [active, setActive] = useState<AbSession | null>(null);
  const [itemIndex, setItemIndex] = useState(0);
  const [draft, setDraft] = useState<AbResponse>({ choices: {}, notes: "" });
  const [reveal, setReveal] = useState<AbReveal | null>(null);
  const [setup, setSetup] = useState({ name: "Commercial food photography", runA: "", runB: "" });
  const [state, setState] = useState<"loading" | "ready" | "saving" | "saved" | "error">("loading");
  const [message, setMessage] = useState("");

  function loadIndex() {
    setState("loading");
    Promise.all([api.runs(), api.abSessions()])
      .then(([runData, sessionData]) => {
        setRuns(runData.runs);
        setSessions(sessionData.sessions);
        setSetup((value) => ({ ...value, runA: value.runA || runData.runs[0]?.id || "", runB: value.runB || runData.runs[1]?.id || "" }));
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
  const draftComplete = Boolean(active && active.criteria.every((criterion) => draft.choices[criterion]));
  const tally = useMemo(() => {
    if (!active || !reveal) return null;
    let runA = 0;
    let runB = 0;
    let ties = 0;
    for (const [itemId, response] of Object.entries(active.responses)) {
      const assignment = reveal.assignments[itemId];
      if (!assignment) continue;
      for (const choice of Object.values(response.choices)) {
        if (choice === "tie") ties += 1;
        else if (choice === "a" && assignment.aSource === "runA") runA += 1;
        else if (choice === "b" && assignment.bSource === "runA") runA += 1;
        else if (choice === "a" || choice === "b") runB += 1;
      }
    }
    return { runA, runB, ties };
  }, [active, reveal]);

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
    setState("saving");
    try {
      const { id } = await api.createAbSession(setup);
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
        <PageHeader eyebrow="Evaluation · Blind comparison" title="Start an unbiased A/B test" description="Only prompt-and-seed matched outputs can be paired. Left and right placement is randomized per item and hidden until review is complete." />
        {runs.length < 2 ? (
          <EmptyState
            icon={<FlaskConical size={26} />}
            title="Two comparable runs are needed"
            body="Generate two runs with matching sample IDs, prompts, and seeds. Their manifests will appear here automatically."
          >
            <button className="secondary-button" type="button" onClick={loadIndex}><RefreshCw size={16} /> Check again</button>
          </EmptyState>
        ) : (
          <section className="setup-card">
            <div className="setup-heading"><Plus size={18} /><div><h2>New comparison</h2><p>Choose any two completed model runs.</p></div></div>
            <label>Review name<input value={setup.name} onChange={(event) => setSetup({ ...setup, name: event.target.value })} /></label>
            <div className="run-selectors">
              <label>Run one<select value={setup.runA} onChange={(event) => setSetup({ ...setup, runA: event.target.value })}>{runs.map((run) => <option key={run.id} value={run.id}>{run.label} · {run.sampleCount} images</option>)}</select></label>
              <span>vs</span>
              <label>Run two<select value={setup.runB} onChange={(event) => setSetup({ ...setup, runB: event.target.value })}>{runs.map((run) => <option key={run.id} value={run.id}>{run.label} · {run.sampleCount} images</option>)}</select></label>
            </div>
            {state === "error" && <StatusMessage kind="error">{message}</StatusMessage>}
            <button className="primary-button" type="button" disabled={!setup.name.trim() || setup.runA === setup.runB || state === "saving"} onClick={createSession}>{state === "saving" ? "Creating…" : "Create blinded review"}</button>
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

  return (
    <div className="workflow ab-workflow">
      <PageHeader
        eyebrow="Evaluation · Blind comparison"
        title={active.name}
        description="Judge the photographic result, not the model label. Choose the stronger image for each criterion or mark a tie."
        actions={<div className="progress-chip"><strong>{completed}</strong> / {active.items.length} complete</div>}
      />

      <div className="ab-prompt"><span>Prompt</span><p>{item.prompt}</p><em>Seed {item.seed}</em></div>

      <section className="ab-images">
        <figure><div className="blind-label">A</div><img src={item.aImageUrl} alt="Blind comparison A" /></figure>
        <figure><div className="blind-label">B</div><img src={item.bImageUrl} alt="Blind comparison B" /></figure>
      </section>

      {reveal ? (
        <section className="reveal-panel">
          <Eye size={18} />
          <div>
            <strong>Models revealed</strong>
            <p>
              Current A: {reveal.assignments[item.id]?.aSource === "runA" ? reveal.runA : reveal.runB}<br />
              Current B: {reveal.assignments[item.id]?.bSource === "runA" ? reveal.runA : reveal.runB}
            </p>
            {tally && <p className="reveal-tally">Criterion wins · {reveal.runA}: {tally.runA} · {reveal.runB}: {tally.runB} · ties: {tally.ties}</p>}
          </div>
        </section>
      ) : (
        <section className="scoring-panel">
          <div className="scoring-title"><span>Which result is stronger?</span><span>A · tie · B</span></div>
          {active.criteria.map((criterion) => (
            <div className="score-row" key={criterion}>
              <div><strong>{criterion}</strong>{criterion === "Artifacts" && <span>fewer visible failures wins</span>}</div>
              <SegmentedChoice
                value={draft.choices[criterion]}
                onChange={(choice: AbChoice) => setDraft((value) => ({ ...value, choices: { ...value.choices, [criterion]: choice } }))}
              />
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
          <button className="primary-button" type="button" disabled={state === "saving" || !draftComplete} title={draftComplete ? "Save this comparison" : "Choose A, tie, or B for every criterion"} onClick={() => save(itemIndex < active.items.length - 1 ? 1 : 0)}>
            {state === "saving" ? "Saving…" : itemIndex === active.items.length - 1 ? "Save comparison" : "Save & next"}
            {itemIndex < active.items.length - 1 && <ChevronRight size={16} />}
          </button>
        )}
      </div>
    </div>
  );
}
