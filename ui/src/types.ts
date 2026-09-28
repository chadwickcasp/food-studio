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
  weightsLabel: string;
  promptSet: string;
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

export type AbChoice = "tie" | (string & {});

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
    images: Array<{ letter: string; url: string }>;
  }>;
}

export interface AbReveal {
  assignments: Record<string, Record<string, string>>;
}

export interface CheckpointPrompt {
  id: string;
  groupId?: string;
  prompt: string;
  seed: number;
  samples: Record<string, string>;
}

export interface CheckpointWeight {
  step: string;
  label: string;
}

export interface CheckpointExperiment {
  id: string;
  name?: string;
  modelId?: string;
  steps: Array<string | number>;
  weights?: CheckpointWeight[];
  prompts: CheckpointPrompt[];
}

export interface CheckpointSelection {
  experimentId: string;
  selectedStep: string;
  promptChoices: Record<string, string>;
  notes: string;
  selectedAt: string;
}
