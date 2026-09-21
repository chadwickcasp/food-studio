import type { ReactNode } from "react";
import { AlertCircle, Check, LoaderCircle } from "lucide-react";

export function PageHeader({ eyebrow, title, description, actions }: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {actions && <div className="header-actions">{actions}</div>}
    </header>
  );
}

export function StatusMessage({ kind, children }: { kind: "loading" | "error" | "saved"; children: ReactNode }) {
  const Icon = kind === "loading" ? LoaderCircle : kind === "error" ? AlertCircle : Check;
  return <p className={`status-message ${kind}`}><Icon size={15} />{children}</p>;
}

export function EmptyState({ icon, title, body, children }: {
  icon: ReactNode;
  title: string;
  body: string;
  children?: ReactNode;
}) {
  return (
    <section className="empty-state">
      <div className="empty-icon">{icon}</div>
      <h2>{title}</h2>
      <p>{body}</p>
      {children}
    </section>
  );
}

export function SegmentedChoice({ value, onChange, labels = ["A", "Tie", "B"] }: {
  value: string | null | undefined;
  onChange: (value: "a" | "tie" | "b") => void;
  labels?: [string, string, string];
}) {
  return (
    <div className="segmented-choice">
      {(["a", "tie", "b"] as const).map((choice, index) => (
        <button
          key={choice}
          type="button"
          className={value === choice ? "active" : ""}
          onClick={() => onChange(choice)}
          aria-pressed={value === choice}
        >
          {labels[index]}
        </button>
      ))}
    </div>
  );
}
