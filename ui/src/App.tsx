import { useState } from "react";
import { Captions, FlaskConical, Image, SplitSquareHorizontal } from "lucide-react";
import { AbReview } from "./features/AbReview";
import { CaptionReview } from "./features/CaptionReview";
import { CheckpointReview } from "./features/CheckpointReview";
import type { ViewName } from "./types";

const views: Array<{ id: ViewName; label: string; icon: typeof Image }> = [
  { id: "captions", label: "Captions", icon: Captions },
  { id: "checkpoints", label: "Checkpoints", icon: FlaskConical },
  { id: "ab", label: "A/B Test", icon: SplitSquareHorizontal },
];

export default function App() {
  const [view, setView] = useState<ViewName>("captions");

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => setView("captions")} aria-label="Food Studio home">
          <span className="brand-mark"><Image size={17} /></span>
          <span>Food Studio</span>
        </button>
        <nav aria-label="Review workflows">
          {views.map(({ id, label, icon: Icon }) => (
            <button key={id} className={view === id ? "nav-item active" : "nav-item"} onClick={() => setView(id)}>
              <Icon size={17} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className="status-dot" />
          Local experiment data
        </div>
      </aside>
      <div className="mobile-nav" aria-label="Review workflows">
        <span className="mobile-brand">Food Studio</span>
        <div>
          {views.map(({ id, label }) => (
            <button key={id} className={view === id ? "active" : ""} onClick={() => setView(id)}>{label}</button>
          ))}
        </div>
      </div>
      <main className="main-stage">
        {view === "captions" && <CaptionReview />}
        {view === "checkpoints" && <CheckpointReview />}
        {view === "ab" && <AbReview />}
      </main>
    </div>
  );
}
