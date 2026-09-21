import type {
  AbResponse,
  AbReveal,
  AbSession,
  AbSessionSummary,
  CaptionItem,
  CaptionSelection,
  CheckpointExperiment,
  RunSummary,
} from "./types";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: options?.body ? { "content-type": "application/json", ...options.headers } : options?.headers,
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error ?? `Request failed (${response.status})`);
  return value as T;
}

export const api = {
  captions: () => request<{ items: CaptionItem[]; updatedAt: string | null }>("/api/captions"),
  saveCaption: (id: string, value: CaptionSelection) =>
    request<CaptionSelection>(`/api/captions/${encodeURIComponent(id)}`, {
      method: "PUT",
      body: JSON.stringify(value),
    }),
  runs: () => request<{ runs: RunSummary[] }>("/api/runs"),
  abSessions: () => request<{ sessions: AbSessionSummary[] }>("/api/ab/sessions"),
  createAbSession: (value: { name: string; runA: string; runB: string }) =>
    request<{ id: string }>("/api/ab/sessions", { method: "POST", body: JSON.stringify(value) }),
  abSession: (id: string) => request<AbSession>(`/api/ab/sessions/${encodeURIComponent(id)}`),
  saveAbResponse: (sessionId: string, itemId: string, value: AbResponse) =>
    request<AbResponse>(
      `/api/ab/sessions/${encodeURIComponent(sessionId)}/responses/${encodeURIComponent(itemId)}`,
      { method: "PUT", body: JSON.stringify(value) },
    ),
  revealAbSession: (id: string) =>
    request<AbReveal>(`/api/ab/sessions/${encodeURIComponent(id)}/reveal`, { method: "POST" }),
  checkpoints: () => request<{ experiments: CheckpointExperiment[] }>("/api/checkpoints"),
  saveCheckpoint: (value: {
    experimentId: string;
    step: string;
    promptChoices: Record<string, string>;
    notes: string;
  }) =>
    request("/api/checkpoints/selection", { method: "PUT", body: JSON.stringify(value) }),
};
