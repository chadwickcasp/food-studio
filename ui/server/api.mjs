import crypto from "node:crypto";
import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(fileURLToPath(new URL("../../", import.meta.url)));
const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp"]);
const jsonHeaders = { "content-type": "application/json; charset=utf-8" };

function sendJson(res, status, value) {
  res.writeHead(status, jsonHeaders);
  res.end(JSON.stringify(value));
}

function sendError(res, status, message) {
  sendJson(res, status, { error: message });
}

function relativeToRepo(filePath) {
  return path.relative(repoRoot, filePath).split(path.sep).join("/");
}

const portableRoots = ["outputs/", "data/"];

function isInsideRepo(resolved) {
  return resolved === repoRoot || resolved.startsWith(`${repoRoot}${path.sep}`);
}

function safeRepoPath(inputPath) {
  const raw = String(inputPath);
  const direct = path.resolve(repoRoot, raw);
  if (isInsideRepo(direct)) return direct;

  // Manifests written on the training VM store absolute paths such as
  // /home/chadcasper/food-studio/outputs/... Remap the outputs/ or data/ suffix
  // onto this checkout when that file is present here.
  if (path.isAbsolute(raw)) {
    const parts = path.normalize(raw).split(path.sep).filter(Boolean);
    for (let index = 1; index < parts.length; index += 1) {
      const relative = parts.slice(index).join("/");
      if (!portableRoots.some((prefix) => relative.startsWith(prefix))) continue;
      const candidate = path.resolve(repoRoot, relative);
      if (isInsideRepo(candidate) && existsSync(candidate)) return candidate;
    }
  }
  throw new Error("Path escapes the repository.");
}

async function readJson(filePath, fallback = null) {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return fallback;
    throw error;
  }
}

async function writeJsonAtomic(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await fs.rename(temporary, filePath);
}

async function readBody(req) {
  let body = "";
  for await (const chunk of req) {
    body += chunk;
    if (body.length > 1_000_000) throw new Error("Request body is too large.");
  }
  return body ? JSON.parse(body) : {};
}

async function walk(root) {
  const files = [];
  async function visit(folder) {
    let entries;
    try {
      entries = await fs.readdir(folder, { withFileTypes: true });
    } catch (error) {
      if (error?.code === "ENOENT") return;
      throw error;
    }
    await Promise.all(
      entries.map(async (entry) => {
        const target = path.join(folder, entry.name);
        if (entry.isDirectory()) await visit(target);
        else files.push(target);
      }),
    );
  }
  await visit(root);
  return files;
}

function mediaUrl(relativePath) {
  return `/media?path=${encodeURIComponent(relativePath)}`;
}

async function captionCatalog() {
  const dataDir = path.join(repoRoot, "data");
  const entries = await fs.readdir(dataDir, { withFileTypes: true });
  const selectionsPath = path.join(dataDir, "captions", "selections.json");
  const saved = await readJson(selectionsPath, { selections: {} });
  const files = entries
    .filter((entry) => entry.isFile() && imageExtensions.has(path.extname(entry.name).toLowerCase()))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));

  const items = await Promise.all(
    files.map(async (filename) => {
      const id = path.parse(filename).name;
      const variants = {};
      for (const source of ["codex", "grok"]) {
        const captionPath = path.join(dataDir, "captions", source, `${id}.txt`);
        try {
          variants[source] = (await fs.readFile(captionPath, "utf8")).trim();
        } catch (error) {
          if (error?.code !== "ENOENT") throw error;
        }
      }
      return {
        id,
        filename,
        imageUrl: mediaUrl(`data/${filename}`),
        captions: variants,
        selection: saved.selections?.[id] ?? null,
      };
    }),
  );
  return { items, updatedAt: saved.updatedAt ?? null };
}

async function saveCaptionSelection(id, selection) {
  const catalog = await captionCatalog();
  if (!catalog.items.some((item) => item.id === id)) throw new Error("Unknown image.");
  if (!["codex", "grok", "custom"].includes(selection.source)) throw new Error("Invalid caption source.");
  if (selection.source === "custom" && !String(selection.customCaption ?? "").trim()) {
    throw new Error("A custom caption cannot be empty.");
  }
  const filePath = path.join(repoRoot, "data", "captions", "selections.json");
  const saved = await readJson(filePath, { version: 1, selections: {} });
  saved.selections ??= {};
  saved.updatedAt = new Date().toISOString();
  saved.selections[id] = {
    source: selection.source,
    customCaption: selection.source === "custom" ? String(selection.customCaption).trim() : null,
    include: Boolean(selection.include),
    reviewedAt: new Date().toISOString(),
  };
  await writeJsonAtomic(filePath, saved);
  return saved.selections[id];
}

async function discoverRuns() {
  const outputRoot = path.join(repoRoot, "outputs");
  const manifests = (await walk(outputRoot)).filter((file) => path.basename(file) === "manifest.json");
  const runs = [];
  for (const manifestPath of manifests) {
    const manifest = await readJson(manifestPath);
    if (!manifest || !Array.isArray(manifest.samples) || manifest.samples.length === 0) continue;
    const samples = manifest.samples.filter((sample) => sample.id && sample.image);
    if (!samples.length) continue;
    runs.push({
      id: relativeToRepo(manifestPath),
      label: relativeToRepo(path.dirname(manifestPath)).replace(/^outputs\//, ""),
      modelId: manifest.model_id ?? "Unknown model",
      adapterPath: manifest.adapter_path ?? null,
      weightsLabel: weightsLabel(manifest.adapter_path),
      promptSet: promptSetLabel(manifest.prompts_path),
      sampleCount: samples.length,
    });
  }
  return runs.sort((a, b) => a.label.localeCompare(b.label));
}

async function loadRun(manifestId) {
  if (!String(manifestId).startsWith("outputs/") || !String(manifestId).endsWith("manifest.json")) {
    throw new Error("Invalid run manifest.");
  }
  const manifestPath = safeRepoPath(manifestId);
  const manifest = await readJson(manifestPath);
  if (!manifest || !Array.isArray(manifest.samples)) throw new Error("Run manifest has no samples.");
  const samples = new Map(
    manifest.samples.map((sample) => [
      String(sample.id),
      {
        id: String(sample.id),
        prompt: String(sample.prompt ?? ""),
        seed: Number(sample.seed),
        image: relativeToRepo(safeRepoPath(String(sample.image))),
      },
    ]),
  );
  return { manifest, samples };
}

function safeSlug(value) {
  return String(value || "review")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48) || "review";
}

function weightsLabel(adapterPath) {
  if (!adapterPath) return "base";
  const parts = String(adapterPath).split(/[/\\]/).filter(Boolean);
  const file = parts.at(-1) ?? "adapter";
  if (file.endsWith(".safetensors")) return parts.slice(-2).join("/");
  return `${file}/pytorch_lora_weights.safetensors`;
}

function promptSetLabel(promptsPath) {
  if (!promptsPath) return "unknown prompts";
  return String(promptsPath).split(/[/\\]/).filter(Boolean).at(-1) ?? "unknown prompts";
}

function letterForIndex(index) {
  if (index < 0 || index > 25) throw new Error("A comparison can include at most 26 models.");
  return String.fromCharCode(65 + index);
}

function shuffleSources(count, sessionId, itemId) {
  const ranked = Array.from({ length: count }, (_, sourceIndex) => {
    const digest = crypto.createHash("sha256").update(`${sessionId}:${itemId}:${sourceIndex}`).digest();
    return { sourceIndex, rank: digest.readUInt32BE(0) };
  });
  ranked.sort((left, right) => left.rank - right.rank || left.sourceIndex - right.sourceIndex);
  return ranked.map((entry, position) => ({
    letter: letterForIndex(position),
    sourceIndex: entry.sourceIndex,
  }));
}

function selectedRunIds(body) {
  const requested = Array.isArray(body.runs) ? body.runs : [body.runA, body.runB];
  const runIds = requested.map((id) => String(id ?? "").trim()).filter(Boolean);
  if (runIds.length < 2) throw new Error("Choose at least two models.");
  if (new Set(runIds).size !== runIds.length) throw new Error("Each model can be included once.");
  if (runIds.length > 26) throw new Error("A comparison can include at most 26 models.");
  return runIds;
}

async function createAbSession(body) {
  const runIds = selectedRunIds(body);
  const loaded = await Promise.all(runIds.map((id) => loadRun(id)));
  assertHeldOutPair(loaded, await loadCheckpointSelection());
  const [first] = loaded;
  const compatibleIds = [...first.samples.keys()].filter((id) => {
    const reference = first.samples.get(id);
    return loaded.every((run) => {
      const sample = run.samples.get(id);
      return sample && sample.prompt === reference.prompt && sample.seed === reference.seed;
    });
  });
  if (!compatibleIds.length) throw new Error("The selected runs have no prompt-and-seed matched samples.");

  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const sessionId = `${safeSlug(body.name)}-${timestamp}`;
  const folder = path.join(repoRoot, "outputs", "reviews", "ab", sessionId);
  const assignments = {};
  const items = compatibleIds.sort().map((id) => {
    const order = shuffleSources(loaded.length, sessionId, id);
    const reference = first.samples.get(id);
    assignments[id] = Object.fromEntries(order.map((entry) => [entry.letter, entry.sourceIndex]));
    return {
      id,
      prompt: reference.prompt,
      seed: reference.seed,
      images: order.map((entry) => ({
        letter: entry.letter,
        image: loaded[entry.sourceIndex].samples.get(id).image,
      })),
    };
  });
  const criteria = Array.isArray(body.criteria) && body.criteria.length
    ? body.criteria.map(String)
    : ["Style match", "Prompt fidelity", "Food presentation", "Artifacts"];
  const session = {
    version: 2,
    id: sessionId,
    name: String(body.name || "A/B review"),
    createdAt: new Date().toISOString(),
    status: "active",
    criteria,
    items,
    responses: {},
  };
  const reveal = {
    runs: runIds.map((id, index) => ({ id, label: weightsLabel(loaded[index].manifest.adapter_path) })),
    assignments,
  };
  await writeJsonAtomic(path.join(folder, "session.json"), session);
  await writeJsonAtomic(path.join(folder, "reveal-key.json"), reveal);
  return sessionId;
}

async function listAbSessions() {
  const root = path.join(repoRoot, "outputs", "reviews", "ab");
  let entries;
  try {
    entries = await fs.readdir(root, { withFileTypes: true });
  } catch (error) {
    if (error?.code === "ENOENT") return [];
    throw error;
  }
  const sessions = await Promise.all(
    entries.filter((entry) => entry.isDirectory()).map(async (entry) => {
      const session = await readJson(path.join(root, entry.name, "session.json"));
      if (!session) return null;
      const completed = Object.keys(session.responses ?? {}).length;
      return {
        id: session.id,
        name: session.name,
        status: session.status,
        completed,
        total: session.items.length,
        createdAt: session.createdAt,
      };
    }),
  );
  return sessions.filter(Boolean).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

async function loadAbSession(sessionId) {
  const safeId = path.basename(sessionId);
  const folder = path.join(repoRoot, "outputs", "reviews", "ab", safeId);
  const session = await readJson(path.join(folder, "session.json"));
  if (!session) throw new Error("A/B session not found.");
  return { folder, session };
}

function itemLetters(item) {
  if (Array.isArray(item.images) && item.images.length) return item.images.map((image) => image.letter);
  if (item.aImage && item.bImage) return ["A", "B"];
  return [];
}

function imageForLetter(item, letter) {
  const normalized = String(letter).toUpperCase();
  if (Array.isArray(item.images)) {
    return item.images.find((image) => image.letter.toUpperCase() === normalized)?.image ?? null;
  }
  if (normalized === "A") return item.aImage ?? null;
  if (normalized === "B") return item.bImage ?? null;
  return null;
}

function publicAbSession(session) {
  return {
    id: session.id,
    name: session.name,
    status: session.status,
    criteria: session.criteria,
    responses: session.responses,
    items: session.items.map((item) => ({
      id: item.id,
      prompt: item.prompt,
      seed: item.seed,
      images: itemLetters(item).map((letter) => ({
        letter,
        url: `/api/ab/sessions/${encodeURIComponent(session.id)}/media/${encodeURIComponent(item.id)}/${encodeURIComponent(letter)}`,
      })),
    })),
  };
}

async function saveAbResponse(sessionId, itemId, body) {
  const { folder, session } = await loadAbSession(sessionId);
  if (!session.items.some((item) => item.id === itemId)) throw new Error("Unknown review item.");
  const item = session.items.find((candidate) => candidate.id === itemId);
  const allowed = new Set([...itemLetters(item).map((letter) => letter.toUpperCase()), "tie"]);
  const choices = {};
  for (const criterion of session.criteria) {
    const raw = String(body.choices?.[criterion] ?? "");
    const value = raw.toLowerCase() === "tie" ? "tie" : raw.toUpperCase();
    if (!allowed.has(value)) {
      throw new Error(`Choose one model or a tie for ${criterion}.`);
    }
    choices[criterion] = value;
  }
  session.responses[itemId] = {
    choices,
    notes: String(body.notes ?? "").trim(),
    reviewedAt: new Date().toISOString(),
  };
  if (Object.keys(session.responses).length === session.items.length) session.status = "complete";
  await writeJsonAtomic(path.join(folder, "session.json"), session);
  return session.responses[itemId];
}

async function revealAbSession(sessionId) {
  const { folder, session } = await loadAbSession(sessionId);
  if (Object.keys(session.responses ?? {}).length !== session.items.length) {
    throw new Error("Complete every comparison before revealing model identities.");
  }
  const reveal = await readJson(path.join(folder, "reveal-key.json"));
  const assignments = {};
  if (Array.isArray(reveal.runs)) {
    for (const [itemId, letters] of Object.entries(reveal.assignments ?? {})) {
      assignments[itemId] = Object.fromEntries(
        Object.entries(letters).map(([letter, sourceIndex]) => [letter, reveal.runs[sourceIndex]?.label ?? "unknown"]),
      );
    }
  } else {
    const runs = await discoverRuns();
    const runLabels = new Map(runs.map((run) => [run.id, run.weightsLabel ?? run.label]));
    const runA = runLabels.get(reveal.runA) ?? reveal.runA;
    const runB = runLabels.get(reveal.runB) ?? reveal.runB;
    for (const [itemId, pair] of Object.entries(reveal.assignments ?? {})) {
      assignments[itemId] = {
        A: pair.aSource === "runA" ? runA : runB,
        B: pair.bSource === "runA" ? runA : runB,
      };
    }
  }
  return { assignments };
}

function selectionPath() {
  return path.join(repoRoot, "outputs", "reviews", "checkpoint-selection.json");
}

async function loadCheckpointSelection() {
  const selection = await readJson(selectionPath());
  if (!selection?.selectedStep) return null;
  return selection;
}

function assertHeldOutPair(loaded, selection) {
  if (!selection) throw new Error("Choose a development checkpoint before starting the A/B test.");
  const promptSets = new Set(loaded.map((run) => promptSetLabel(run.manifest.prompts_path)));
  if (promptSets.size !== 1 || !promptSets.has("validation_prompts.json")) {
    throw new Error("The A/B test uses held-out validation prompts only.");
  }
  const step = String(selection.selectedStep);
  if (step === "base") throw new Error("The locked checkpoint is base, so there is no LoRA to compare.");
  const expected = new Set(["base", `checkpoint-${step}/pytorch_lora_weights.safetensors`]);
  const labels = loaded.map((run) => weightsLabel(run.manifest.adapter_path));
  if (labels.length !== 2 || new Set(labels).size !== 2 || labels.some((label) => !expected.has(label))) {
    throw new Error("The A/B test compares base with the checkpoint chosen on the development set.");
  }
}

async function stepWeights(experimentDir, steps) {
  const weights = [];
  for (const step of steps) {
    const manifest = await readJson(path.join(experimentDir, String(step), "manifest.json"));
    const label = manifest
      ? weightsLabel(manifest.adapter_path)
      : String(step) === "base"
        ? "base"
        : `checkpoint-${step}/pytorch_lora_weights.safetensors`;
    weights.push({ step: String(step), label });
  }
  return weights;
}

async function checkpointExperiments() {
  const root = path.join(repoRoot, "outputs", "development");
  const manifests = (await walk(root)).filter((file) => path.basename(file) === "manifest.json");
  const experiments = [];
  for (const file of manifests) {
    const value = await readJson(file);
    if (!value || !Array.isArray(value.prompts) || !Array.isArray(value.steps)) continue;
    experiments.push({
      ...value,
      id: relativeToRepo(file),
      weights: await stepWeights(path.dirname(file), value.steps),
      prompts: value.prompts.map((prompt) => ({
        ...prompt,
        samples: Object.fromEntries(
          Object.entries(prompt.samples ?? {}).map(([step, image]) => [step, mediaUrl(relativeToRepo(safeRepoPath(String(image))))]),
        ),
      })),
    });
  }
  return experiments;
}

async function saveCheckpointSelection(body) {
  if (!body.experimentId || !body.step) throw new Error("Experiment and step are required.");
  const experiments = await checkpointExperiments();
  const experiment = experiments.find((candidate) => candidate.id === body.experimentId);
  if (!experiment) throw new Error("Checkpoint experiment not found.");
  const validSteps = new Set(experiment.steps.map(String));
  if (!validSteps.has(String(body.step))) throw new Error("The selected step is not part of this experiment.");
  for (const prompt of experiment.prompts) {
    const choice = String(body.promptChoices?.[prompt.id] ?? "");
    if (!validSteps.has(choice) || !prompt.samples?.[choice]) {
      throw new Error(`Mark the best render for ${prompt.id} before selecting the stopping point.`);
    }
  }
  const output = {
    version: 1,
    experimentId: body.experimentId,
    selectedStep: String(body.step),
    promptChoices: body.promptChoices ?? {},
    notes: String(body.notes ?? "").trim(),
    selectedAt: new Date().toISOString(),
  };
  await writeJsonAtomic(selectionPath(), output);
  return output;
}

async function serveFile(res, filePath) {
  const extension = path.extname(filePath).toLowerCase();
  if (!imageExtensions.has(extension)) throw new Error("Unsupported media type.");
  const mime = extension === ".png" ? "image/png" : extension === ".webp" ? "image/webp" : "image/jpeg";
  const data = await fs.readFile(filePath);
  res.writeHead(200, { "content-type": mime, "cache-control": "no-store" });
  res.end(data);
}

export async function foodStudioApi(req, res, next) {
  const url = new URL(req.url, "http://localhost");
  try {
    if (req.method === "GET" && url.pathname === "/media") {
      const requested = url.searchParams.get("path");
      if (!requested) return sendError(res, 400, "Missing media path.");
      return await serveFile(res, safeRepoPath(requested));
    }
    if (req.method === "GET" && url.pathname === "/api/captions") {
      return sendJson(res, 200, await captionCatalog());
    }
    const captionMatch = url.pathname.match(/^\/api\/captions\/(.+)$/);
    if (req.method === "PUT" && captionMatch) {
      return sendJson(res, 200, await saveCaptionSelection(decodeURIComponent(captionMatch[1]), await readBody(req)));
    }
    if (req.method === "GET" && url.pathname === "/api/runs") {
      return sendJson(res, 200, { runs: await discoverRuns() });
    }
    if (req.method === "GET" && url.pathname === "/api/ab/sessions") {
      return sendJson(res, 200, { sessions: await listAbSessions() });
    }
    if (req.method === "POST" && url.pathname === "/api/ab/sessions") {
      return sendJson(res, 201, { id: await createAbSession(await readBody(req)) });
    }
    const abSessionMatch = url.pathname.match(/^\/api\/ab\/sessions\/([^/]+)$/);
    if (req.method === "GET" && abSessionMatch) {
      const { session } = await loadAbSession(decodeURIComponent(abSessionMatch[1]));
      return sendJson(res, 200, publicAbSession(session));
    }
    const abResponseMatch = url.pathname.match(/^\/api\/ab\/sessions\/([^/]+)\/responses\/([^/]+)$/);
    if (req.method === "PUT" && abResponseMatch) {
      const response = await saveAbResponse(
        decodeURIComponent(abResponseMatch[1]),
        decodeURIComponent(abResponseMatch[2]),
        await readBody(req),
      );
      return sendJson(res, 200, response);
    }
    const abRevealMatch = url.pathname.match(/^\/api\/ab\/sessions\/([^/]+)\/reveal$/);
    if (req.method === "POST" && abRevealMatch) {
      return sendJson(res, 200, await revealAbSession(decodeURIComponent(abRevealMatch[1])));
    }
    const abMediaMatch = url.pathname.match(/^\/api\/ab\/sessions\/([^/]+)\/media\/([^/]+)\/([A-Za-z])$/);
    if (req.method === "GET" && abMediaMatch) {
      const { session } = await loadAbSession(decodeURIComponent(abMediaMatch[1]));
      const item = session.items.find((candidate) => candidate.id === decodeURIComponent(abMediaMatch[2]));
      const image = item ? imageForLetter(item, abMediaMatch[3]) : null;
      if (!image) return sendError(res, 404, "Review item not found.");
      return await serveFile(res, safeRepoPath(image));
    }
    if (req.method === "GET" && url.pathname === "/api/checkpoints") {
      return sendJson(res, 200, { experiments: await checkpointExperiments() });
    }
    if (req.method === "GET" && url.pathname === "/api/checkpoints/selection") {
      return sendJson(res, 200, { selection: await loadCheckpointSelection() });
    }
    if (req.method === "PUT" && url.pathname === "/api/checkpoints/selection") {
      return sendJson(res, 200, await saveCheckpointSelection(await readBody(req)));
    }
    if (url.pathname.startsWith("/api/") || url.pathname === "/media") {
      return sendError(res, 404, "Not found.");
    }
    if (next) return next();
    return sendError(res, 404, "Not found.");
  } catch (error) {
    return sendError(res, 400, error?.message ?? "Unexpected error.");
  }
}
