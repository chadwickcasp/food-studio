export type ViewName = "captions" | "checkpoints" | "ab";

export type CaptionSource = "codex" | "grok" | "custom";

export interface CaptionSelection {
  source: CaptionSource;
  customCaption: string | null;
  include: boolean;
  reviewedAt?: string;
}

export interface CaptionItem {
  id: string;
  filename: string;
  imageUrl: string;
  captions: Partial<Record<Exclude<CaptionSource, "custom">, string>>;
  selection: CaptionSelection | null;
}

export interface RunSummary {
  id: string;
  label: string;
  modelId: string;
  adapterPath: string | null;
  sampleCount: number;
}

export interface AbSessionSummary {
  id: string;
  name: string;
  status: "active" | "complete";
  completed: number;
  total: number;
  createdAt: string;
}

export type AbChoice = "a" | "tie" | "b";

export interface AbResponse {
  choices: Record<string, AbChoice | null>;
  notes: string;
  reviewedAt?: string;
}

export interface AbSession {
  id: string;
  name: string;
  status: "active" | "complete";
  criteria: string[];
  responses: Record<string, AbResponse>;
  items: Array<{
    id: string;
    prompt: string;
    seed: number;
    aImageUrl: string;
    bImageUrl: string;
  }>;
}

export interface AbReveal {
  runA: string;
  runB: string;
  assignments: Record<string, { aSource: "runA" | "runB"; bSource: "runA" | "runB" }>;
}

export interface CheckpointPrompt {
  id: string;
  prompt: string;
  seed: number;
  samples: Record<string, string>;
}

export interface CheckpointExperiment {
  id: string;
  name?: string;
  modelId?: string;
  steps: Array<string | number>;
  prompts: CheckpointPrompt[];
}
