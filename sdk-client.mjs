import { randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import { z } from "zod";
import { ULTRACODE_BASE_EFFORT, ULTRACODE_EFFORT, orchestrationPolicy, withOrchestrationPolicy } from "./orchestration.js";
import { planUsageFrom } from "./plan-usage.js";

/**
 * Only a fallback: the live catalog comes from the SDK's supportedModels(), so
 * new model families appear without a release here.
 */
/** How long one plan reading serves every asker on the host. */
const PLAN_USAGE_TTL_MS = 120_000;

const DEFAULT_MODELS = [
  { id: "opus", displayName: "Opus 5", isDefault: true, supportedReasoningEfforts: reasoningEfforts() },
  { id: "fable", displayName: "Fable 5", isDefault: false, supportedReasoningEfforts: reasoningEfforts() },
  { id: "sonnet", displayName: "Sonnet 5", isDefault: false, supportedReasoningEfforts: reasoningEfforts() },
  { id: "haiku", displayName: "Haiku 4.5", isDefault: false, supportedReasoningEfforts: reasoningEfforts() },
];

function reasoningEfforts(levels = ["low", "medium", "high", "xhigh", "max"]) {
  return levels.map(reasoningEffort => ({ reasoningEffort }));
}

/** Ultracode rides on xhigh, so it is offered only where xhigh is. */
function withUltracode(levels) {
  return levels.includes(ULTRACODE_BASE_EFFORT) ? [...levels, ULTRACODE_EFFORT] : levels;
}

function stripTag(value) {
  return String(value ?? "").replace(/\[[^\]]*\]$/, "");
}

/**
 * Anthropic's usage shape onto the harness's. Cache reads and writes stay
 * apart because the harness reports a cache-hit rate from them, and thinking
 * tokens are carried so reasoning is not billed as invisible output.
 */
export function tokenUsage(usage) {
  if (!usage) return null;
  const thinking = usage.output_tokens_details?.thinking_tokens;
  return {
    inputTokens: usage.input_tokens ?? 0,
    outputTokens: usage.output_tokens ?? 0,
    ...(Number.isSafeInteger(usage.cache_read_input_tokens) ? { cacheReadTokens: usage.cache_read_input_tokens } : {}),
    ...(Number.isSafeInteger(usage.cache_creation_input_tokens) ? { cacheWriteTokens: usage.cache_creation_input_tokens } : {}),
    ...(Number.isSafeInteger(thinking) ? { reasoningTokens: thinking } : {}),
  };
}

/**
 * Project the SDK catalog onto the runtime's model shape.
 *
 * DSH always requires an explicit model, so the "default" row is dropped — it
 * resolves to the same canonical model as a named row anyway, and its
 * recommendation is preserved by marking that row as the default. Names come
 * from `resolvedModel` (the canonical wire id) rather than the display strings,
 * and descriptions are omitted so the picker stays one line per model.
 */
function toRuntimeModels(rows) {
  const recommended = rows.find(row => row.value === "default")?.resolvedModel;
  const named = rows.filter(row => row.value !== "default");
  const seen = new Set();
  const models = [];
  for (const row of named) {
    const key = row.resolvedModel ?? row.value;
    if (seen.has(key)) continue;
    seen.add(key);
    const levels = row.supportsEffort === false ? [] : (row.supportedEffortLevels ?? []);
    models.push({
      id: row.value,
      resolvedModel: row.resolvedModel,
      displayName: modelName(row),
      isDefault: recommended !== undefined && row.resolvedModel === recommended,
      // No default effort is invented: ModelInfo reports none, so leaving it
      // unset means Claude Code applies its own. Ultracode is this plugin's own
      // row, offered only where xhigh exists because that is what it selects.
      ...(levels.length ? { supportedReasoningEfforts: reasoningEfforts(withUltracode(levels)) } : {}),
    });
  }
  if (models.length && !models.some(model => model.isDefault)) models[0].isDefault = true;
  disambiguate(models, named);
  return models;
}

/** `claude-haiku-4-5-20251001` -> `Haiku 4.5`; `claude-opus-5[1m]` -> `Opus 5`. */
function modelName(row) {
  const canonical = (row.resolvedModel ?? row.value).replace(/\[[^\]]*\]$/, "");
  const parts = canonical.replace(/^claude-/, "").replace(/-\d{8}$/, "").split("-");
  const family = parts.shift() ?? "";
  if (!family) return row.displayName ?? row.value;
  const version = parts.filter(part => /^\d+$/.test(part)).join(".");
  const name = `${family.charAt(0).toUpperCase()}${family.slice(1)}${version ? ` ${version}` : ""}`;
  return name;
}

/** Two rows of one family (e.g. `opus` and `opus[1m]`) must stay distinguishable. */
function disambiguate(models, rows) {
  const counts = new Map();
  for (const model of models) counts.set(model.displayName, (counts.get(model.displayName) ?? 0) + 1);
  for (const model of models) {
    if (counts.get(model.displayName) === 1) continue;
    const row = rows.find(candidate => candidate.value === model.id);
    const tag = /\[([^\]]+)\]$/.exec(row?.resolvedModel ?? row?.value ?? "")?.[1];
    if (tag) model.displayName = `${model.displayName} (${tag.toUpperCase()})`;
  }
}

export class ClaudeSdkClient extends EventEmitter {
  constructor({ sdk = null, pathToClaudeCodeExecutable = undefined, env = undefined, requestTimeoutMs = 30 * 60_000, interruptTimeoutMs = 5_000, clock = Date.now } = {}) {
    super();
    this.sdk = sdk;
    this.pathToClaudeCodeExecutable = pathToClaudeCodeExecutable;
    // The environment every spawned Claude Code child runs under. Unset keeps
    // the SDK's own default (`{...process.env}`); set means the caller built it
    // from the process environment and pointed ANTHROPIC_BASE_URL at a gateway.
    this.env = env;
    this.requestTimeoutMs = requestTimeoutMs;
    // A CLI that never answers the interrupt control request would otherwise leave interruptTurn
    // pending forever; see that method for why a pending interrupt leaks the child.
    this.interruptTimeoutMs = Number.isFinite(interruptTimeoutMs) && interruptTimeoutMs > 0 ? interruptTimeoutMs : 5_000;
    this.clock = clock;
    this.sessions = new Map();
    this.queries = new Map();
    this.pendingRequests = new Map();
    // Tool calls handed to the harness: the MCP handler parks here until the
    // harness executes the call and sends the result back.
    this.pendingToolCalls = new Map();
    // Context capacity per canonical model, learned from result messages.
    this.contextWindows = new Map();
    this.planUsageCache = null;
    this.planUsageInflight = null;
    this.closed = false;
  }

  async start() {
    this.sdk ??= await import("@anthropic-ai/claude-agent-sdk");
    if (typeof this.sdk.query !== "function") throw new Error("Claude Agent SDK query() is unavailable");
    this.closed = false;
  }

  /**
   * Park one tool call and announce it, so the harness can run it.
   *
   * The id is minted here rather than taken from the model's `tool_use` block:
   * the MCP handler is never told that id, and the harness only needs an id
   * that survives its own round trip. Announcing from the handler also means
   * the call is known to exist exactly when the SDK is actually waiting on it.
   */
  parkToolCall(sessionId, turnId, name, args) {
    const parkId = randomUUID();
    const parked = new Promise((resolve, reject) => {
      this.pendingToolCalls.set(parkId, { resolve, reject });
    });
    this.emit("activity", {
      method: "tool/parked",
      params: { sessionId, turnId, parkId, name, arguments: args ?? {} },
    });
    return parked;
  }

  /** Hand one harness-executed result back to the waiting tool call. */
  resolveToolCall(parkId, result) {
    const pending = this.pendingToolCalls.get(parkId);
    if (!pending) return false;
    this.pendingToolCalls.delete(parkId);
    pending.resolve(result);
    return true;
  }

  /** Fail one parked call — an aborted turn, or a harness-side execution error. */
  rejectToolCall(toolUseId, error) {
    const pending = this.pendingToolCalls.get(toolUseId);
    if (!pending) return false;
    this.pendingToolCalls.delete(toolUseId);
    pending.reject(error instanceof Error ? error : new Error(String(error)));
    return true;
  }

  /** Fail every parked call, e.g. when a turn is interrupted mid-handoff. */
  rejectAllToolCalls(error) {
    for (const parkId of [...this.pendingToolCalls.keys()]) this.rejectToolCall(parkId, error);
  }

  /**
   * Record each model's context capacity as the SDK reports it. `ModelInfo`
   * carries no context window, but every result message does, so the number is
   * exact once a turn has run and a sane default until then.
   */
  learnContextWindows(modelUsage) {
    for (const [model, usage] of Object.entries(modelUsage ?? {})) {
      if (Number.isSafeInteger(usage?.contextWindow)) this.contextWindows.set(model, usage.contextWindow);
      if (usage?.canonicalModel && Number.isSafeInteger(usage?.contextWindow)) {
        this.contextWindows.set(usage.canonicalModel, usage.contextWindow);
      }
    }
  }

  /**
   * Context capacity for one catalog row.
   *
   * @param model - the row id, e.g. `haiku` or `opus[1m]`.
   * @param resolvedModel - the canonical id that row maps to, when known.
   */
  contextWindowFor(model, resolvedModel) {
    for (const key of [resolvedModel, model].filter(Boolean)) {
      const learned = this.contextWindows.get(key) ?? this.contextWindows.get(stripTag(key));
      if (learned) return learned;
    }
    // Until a turn has run: current models (5.x, including the bare `opus`,
    // `sonnet` and `haiku` aliases) report 1M. Only untagged 4.x-and-older ids
    // are 200k; their long-context variants carry an explicit [1m] tag.
    const id = String(resolvedModel ?? model);
    if (/\[1m\]$/.test(id)) return 1_000_000;
    return /^claude-(?:haiku|sonnet|opus)-[1-4](?:\D|$)|^claude-[1-4]/.test(id) ? 200_000 : 1_000_000;
  }

/**
   * The subscription's plan-limit windows.
   *
   * Reading them costs a control session — the SDK spawns Claude Code — so the
   * answer is held briefly. The client that renders it keeps a longer floor of
   * its own; this one exists so several sessions asking at once cost one read.
   *
   * The underlying API is named for its own instability, so a failure here is
   * reported as "unsupported" rather than raised: a ring that cannot draw is a
   * missing ring, never a broken turn.
   */
  async planUsage() {
    const now = this.clock();
    if (this.planUsageCache && now - this.planUsageCache.at < PLAN_USAGE_TTL_MS) {
      return this.planUsageCache.value;
    }
    if (this.planUsageInflight) return this.planUsageInflight;

    this.planUsageInflight = this.readPlanUsage()
      .catch(error => {
        this.emit("diagnostic", `Claude plan usage unavailable: ${error?.message ?? error}`);
        return { supported: false };
      })
      .then(value => {
        this.planUsageCache = { at: this.clock(), value };
        this.planUsageInflight = null;
        return value;
      });
    return this.planUsageInflight;
  }

  async readPlanUsage() {
    if (!this.sdk) await this.start();
    if (typeof this.sdk.query !== "function") return { supported: false };
    let release;
    const idle = new Promise((resolve) => { release = resolve; });
    const prompt = (async function* () { await idle; })();
    const query = this.sdk.query({
      prompt,
      options: {
        ...(this.pathToClaudeCodeExecutable ? { pathToClaudeCodeExecutable: this.pathToClaudeCodeExecutable } : {}),
        ...(this.env ? { env: this.env } : {}),
      },
    });
    try {
      const read = query.usage_EXPERIMENTAL_MAY_CHANGE_DO_NOT_RELY_ON_THIS_API_YET;
      if (typeof read !== "function") return { supported: false };
      return planUsageFrom(await read.call(query));
    } finally {
      release();
      query.close?.();
    }
  }

  async listModels() {
    const live = await this.fetchSupportedModels().catch((error) => {
      this.emit("diagnostic", `Claude model catalog unavailable, using defaults: ${error.message}`);
      return null;
    });
    return live?.length ? live : DEFAULT_MODELS;
  }

  /**
   * Open a control-only query — streaming input that never yields, so no turn
   * runs — purely to read the catalog the installed Claude Code advertises.
   */
  async fetchSupportedModels() {
    if (!this.sdk) await this.start();
    if (typeof this.sdk.query !== "function") return null;
    let release;
    const idle = new Promise((resolve) => { release = resolve; });
    const prompt = (async function* () { await idle; })();
    const query = this.sdk.query({
      prompt,
      options: {
        ...(this.pathToClaudeCodeExecutable ? { pathToClaudeCodeExecutable: this.pathToClaudeCodeExecutable } : {}),
        ...(this.env ? { env: this.env } : {}),
      },
    });
    try {
      if (typeof query.supportedModels !== "function") return null;
      const rows = await query.supportedModels();
      return Array.isArray(rows) ? toRuntimeModels(rows) : null;
    } finally {
      release();
      query.close?.();
    }
  }

  async createSession(config = {}) {
    const id = config.sessionId ?? randomUUID();
    this.sessions.set(id, { id, cwd: config.cwd ?? process.cwd(), created: false, config: structuredClone(config) });
    return { id, cwd: config.cwd ?? process.cwd(), turns: [] };
  }

  async resumeSession(sessionId, config = {}) {
    const existing = this.sessions.get(sessionId) ?? { id: sessionId, created: true, config: {} };
    this.sessions.set(sessionId, {
      ...existing,
      cwd: config.cwd ?? existing.cwd ?? process.cwd(),
      config: { ...existing.config, ...structuredClone(config) },
    });
    return { id: sessionId, cwd: config.cwd ?? existing.cwd ?? process.cwd(), turns: [] };
  }

  async sendMessage(sessionId, message = {}) {
    const session = this.sessions.get(sessionId) ?? (await this.resumeSession(sessionId, message));
    const turnId = randomUUID();
    const abortController = new AbortController();
    const options = this.queryOptions(session, message, abortController, turnId);
    const query = this.sdk.query({ prompt: promptFor(message), options });
    this.queries.set(turnId, { query, abortController, sessionId });
    void this.consumeQuery(session, turnId, query).catch((error) => {
      this.emit("diagnostic", `Claude SDK query failed: ${error?.stack ?? error}`);
      this.completeTurn(session.id, turnId, "failed", error);
    });
    session.created = true;
    return { id: turnId, status: "inProgress", items: [] };
  }

  /**
   * Stop one in-flight turn.
   *
   * `interrupt()` is a control request the CLI answers when it can. A CLI that never answers
   * leaves the promise pending forever, so the abort and the close below would never run and the
   * CLI child would stay alive. DSH waits for the turn to become idle before it disposes the
   * agent, so a pending interrupt also stops the release in `detachAgent` from being reached.
   * Bound the wait, then abort and close regardless: the close ends the control request and the
   * child with it.
   */
  async interruptTurn(_sessionId, turnId) {
    const record = this.queries.get(turnId);
    if (!record) return;
    let interrupted;
    try {
      interrupted = record.query.interrupt?.();
    } catch {
      interrupted = undefined;
    }
    if (interrupted && typeof interrupted.then === "function") {
      let timer;
      try {
        await Promise.race([
          interrupted,
          new Promise((resolve) => { timer = setTimeout(resolve, this.interruptTimeoutMs); }),
        ]);
      } catch {
        // The interrupt itself failed. The abort and the close below still apply.
      } finally {
        clearTimeout(timer);
      }
    }
    record.abortController.abort();
    record.query.close?.();
  }

  async releaseSession(sessionId) {
    for (const [turnId, record] of this.queries) {
      if (record.sessionId === sessionId) {
        record.abortController.abort();
        record.query.close?.();
        this.queries.delete(turnId);
      }
    }
    this.sessions.delete(sessionId);
  }

  async close() {
    this.closed = true;
    for (const record of this.queries.values()) {
      record.abortController.abort();
      record.query.close?.();
    }
    this.queries.clear();
    for (const request of this.pendingRequests.values()) {
      request.resolve({ behavior: "deny", message: "Relay Claude SDK client closed" });
    }
    this.pendingRequests.clear();
  }

  resolveRequest(requestId, response = {}) {
    const request = this.pendingRequests.get(String(requestId));
    if (!request) throw new Error(`unknown pending Claude request ${requestId}`);
    this.pendingRequests.delete(String(requestId));
    request.resolve(responseForRequest(request, response));
  }

  rejectRequest(requestId, error) {
    const request = this.pendingRequests.get(String(requestId));
    if (!request) return;
    this.pendingRequests.delete(String(requestId));
    request.resolve({ behavior: "deny", message: error?.message ?? String(error) });
  }

  queryOptions(session, message, abortController, turnId) {
    const bridge = dshMcpOptions(this.sdk, message, (name, args) => this.parkToolCall(session.id, turnId, name, args));
    return {
      abortController,
      cwd: message.cwd ?? session.cwd ?? process.cwd(),
      model: message.model ?? session.config?.model,
      ...(effortOf(message, session) !== undefined ? { effort: effortOf(message, session) } : {}),
      permissionMode: sdkPermissionMode(message),
      // Ask for readable reasoning. Without `display: "summarized"` the models
      // stream thinking blocks whose text is empty — a signature and a token
      // count and nothing to render. This is the documented switch for it, and
      // it composes with `effort`, which still decides how much thinking
      // happens: Sonnet at low thinks not at all, at high it thinks and the
      // summary comes through.
      ...(message.thinkingSummaries === false ? {} : { thinking: { type: "adaptive", display: "summarized" } }),
      // Ultracode's own half: xhigh effort (selected above) plus the standing
      // orchestration posture. Claude Code's Workflow tool stays absent — the
      // harness's own workflow tool is what the prompt names, so subagent work
      // runs through the harness and lands in its trajectory.
      ...(isUltracode(message, session) ? { settings: { ultracode: true, enableWorkflows: true } } : {}),
      // DSH owns the toolset. Without this, Claude Code's built-ins stay in
      // context and win — `allowedTools` only pre-approves, it does not scope.
      tools: [],
      // Default to loading nothing. Every layer above passes [] explicitly, but
      // a caller that forgets must not silently mount the user's own Claude
      // Code config — CLAUDE.md, skills, hooks — on top of the harness prompt.
      settingSources: message.settingSources ?? session.config?.settingSources ?? [],
      systemPrompt: withOrchestrationPolicy(
        message.systemPrompt ?? session.config?.systemPrompt,
        orchestrationPolicy(message.dshTools, isUltracode(message, session)),
      ),
      pathToClaudeCodeExecutable: this.pathToClaudeCodeExecutable,
      ...(this.env ? { env: this.env } : {}),
      includePartialMessages: true,
      ...(session.created ? { resume: session.id } : { sessionId: session.id }),
      // Only ask for the permission callback when something could actually
      // reach it. Every bridged tool is pre-approved here because DSH gates and
      // executes them, so passing the callback as well makes the SDK warn that
      // it is shadowed — accurately, and on every turn.
      ...(bridge.allowedTools
        ? {}
        : { canUseTool: (toolName, input, options) => this.requestPermission(session.id, toolName, input, options) }),
      ...bridge,
    };
  }

  requestPermission(sessionId, toolName, input, options = {}) {
    const id = options.requestId ?? randomUUID();
    return new Promise((resolve) => {
      const request = {
        id,
        method: toolName === "AskUserQuestion" ? "tool/requestUserInput" : "tool/requestApproval",
        signal: options.signal,
        params: {
          sessionId,
          toolName,
          input: structuredClone(input ?? {}),
          title: options.title,
          displayName: options.displayName,
          description: options.description,
          decisionReason: options.decisionReason,
          blockedPath: options.blockedPath,
          toolUseID: options.toolUseID,
          suggestions: structuredClone(options.suggestions ?? []),
        },
      };
      this.pendingRequests.set(String(id), { request, resolve, input });
      options.signal?.addEventListener("abort", () => this.rejectRequest(id, new Error("Claude permission request was cancelled")), { once: true });
      this.emit("request", request);
    });
  }

  async consumeQuery(session, turnId, query) {
    const state = { currentMessageId: null, text: new Map(), reasoning: new Map(), activities: new Set() };
    let completed = false;
    try {
      for await (const message of query) {
        for (const event of normalizeSdkMessage(message, state)) {
          this.emit("activity", { method: event.method, params: { sessionId: session.id, turnId, ...event.params } });
        }
        if (message.type === "result") {
          completed = true;
          this.learnContextWindows(message.modelUsage);
          this.completeTurn(session.id, turnId, message.is_error ? "failed" : "completed", resultError(message), tokenUsage(message.usage));
        }
      }
      if (!completed) this.completeTurn(session.id, turnId, "completed");
    } finally {
      // Close the query when its message stream ends. On a normal turn the CLI process has
      // already exited by then, so this does nothing. A stream failure can end the loop while
      // the process is still alive, and this close is a guard for that case. It is not the fix
      // for orphaned children: that is the release in detachAgent.
      this.queries.delete(turnId);
      try {
        const closing = query.close?.();
        // close() is typed as void, but a thenable must not leave an unhandled rejection behind.
        if (typeof closing?.then === "function") {
          closing.then(undefined, error => this.emit("diagnostic", `Claude SDK query close failed: ${error?.stack ?? error}`));
        }
      } catch (error) {
        this.emit("diagnostic", `Claude SDK query close failed: ${error?.stack ?? error}`);
      }
    }
  }

  completeTurn(sessionId, turnId, status, error = null, usage = null) {
    this.emit("activity", {
      method: "turn/completed",
      params: {
        sessionId,
        turn: {
          id: turnId,
          status,
          error: error ? { message: error.message } : null,
          items: [],
          ...(usage ? { usage } : {}),
        },
      },
    });
  }
}

function dshMcpOptions(sdk, message, park) {
  // Use only the MCP servers passed here. Without this, a claude.ai login makes
  // Claude Code add every claude.ai connector (Gmail, Calendar, Drive, ...) to
  // each request: about 90k tokens DSH never sees, so its context meter and the
  // real prompt disagree. Passing mcpServers alone does not suppress connectors.
  const strict = { strictMcpConfig: true };
  const schemas = message.dshTools;
  if (!Array.isArray(schemas) || schemas.length === 0) return strict;
  if (typeof sdk.createSdkMcpServer !== "function" || typeof sdk.tool !== "function") {
    throw new Error("This Claude Agent SDK does not support in-process DSH tools");
  }
  // The handler deliberately does no work. The harness owns execution — its
  // agent loop runs the tool under its own sandbox and approval policy, and
  // records it in its own trajectory — so the call parks here until the result
  // comes back through the adapter.
  const tools = schemas.map(schema => sdk.tool(
    schema.name,
    schema.description,
    jsonSchemaShape(schema.parameters),
    async args => park(schema.name, args),
  ));
  return {
    ...strict,
    mcpServers: {
      dsh: sdk.createSdkMcpServer({ name: "dsh", version: "1.0.0", tools, alwaysLoad: true }),
    },
    // Pre-approve at the SDK layer on purpose. These are DSH's own tools, run
    // by DSH's tool runtime, which resolves "ask" decisions through its own
    // approval seam against the session's sandbox policy. Gating here as well
    // would prompt for every call regardless of that policy — workspace-write
    // included — which is not how DSH treats its own agent's calls.
    allowedTools: schemas.map(schema => `mcp__dsh__${schema.name}`),
    // DSH's prompt names its tools bare ("use the read tool"), so a model that
    // emits `Read` or `Bash` is following instructions, not hallucinating.
    // Route those names at the DSH tool of the same name instead of failing.
    toolAliases: Object.fromEntries(schemas.map(schema => [
      `${schema.name.charAt(0).toUpperCase()}${schema.name.slice(1)}`,
      `mcp__dsh__${schema.name}`,
    ])),
  };
}

function jsonSchemaShape(schema) {
  if (!schema || schema.type !== "object" || typeof schema.properties !== "object" || schema.properties === null) {
    if (schema?.type === "object" && schema.properties === undefined) return {};
    throw new Error("DSH tool parameters must use an object JSON Schema");
  }
  const required = new Set(Array.isArray(schema.required) ? schema.required : []);
  return Object.fromEntries(Object.entries(schema.properties).map(([name, property]) => {
    let field;
    try {
      field = z.fromJSONSchema(property);
    } catch {
      field = z.unknown();
    }
    return [name, required.has(name) ? field : field.optional()];
  }));
}

/**
 * The SDK prompt for one message. Text alone stays a plain string. With images, the prompt is
 * one user message: the text, then one base64 image block per image.
 */
export function promptFor(message) {
  const images = Array.isArray(message.images) ? message.images : [];
  if (images.length === 0) return message.text;
  const content = [];
  if (message.text) content.push({ type: "text", text: message.text });
  for (const image of images) {
    content.push({ type: "image", source: { type: "base64", media_type: image.mediaType, data: image.data } });
  }
  return (async function* () {
    yield { type: "user", message: { role: "user", content }, parent_tool_use_id: null };
  })();
}

export function dshToolResult(result) {
  const content = (result.content ?? []).map(block => {
    if (block?.type === "text") return { type: "text", text: String(block.text ?? "") };
    if (block?.type === "image" && typeof block.data === "string" && typeof block.mediaType === "string") {
      return { type: "image", data: block.data, mimeType: block.mediaType };
    }
    return { type: "text", text: JSON.stringify(block) };
  });
  if (content.length === 0) content.push({ type: "text", text: result.isError ? "DSH tool failed" : "DSH tool completed." });
  return { content, isError: Boolean(result.isError) };
}

function normalizeSdkMessage(message, state) {
  const events = [];
  if (message.type === "stream_event") {
    const event = message.event;
    if (event?.type === "message_start") {
      state.currentMessageId = event.message?.id ?? message.uuid ?? null;
      events.push(...usageEvents(state.currentMessageId, event.message?.usage));
      return events;
    }
    // message_delta carries the call's running output count, so it is the
    // figure the adapter needs for each model call's output tokens.
    if (event?.type === "message_delta") {
      events.push(...usageEvents(state.currentMessageId, event.usage));
    }
    if (event?.type === "content_block_start" && event.content_block?.type === "tool_use" && event.content_block.id && state.currentMessageId) {
      events.push({ method: "usage/toolUses", params: { callId: state.currentMessageId, toolUseIds: [event.content_block.id] } });
    }
    if (event?.type === "message_stop" && state.currentMessageId) {
      events.push({ method: "usage/apiCallEnded", params: { callId: state.currentMessageId } });
    }
    if (event?.type === "content_block_delta" && event.delta?.type === "text_delta") {
      const itemId = streamItemId(state, "text", event.index);
      state.text.set(itemId, `${state.text.get(itemId) ?? ""}${event.delta.text}`);
      events.push({ method: "item/agentMessage/delta", params: { itemId, delta: event.delta.text } });
    }
    if (event?.type === "content_block_delta" && event.delta?.type === "thinking_delta") {
      const itemId = streamItemId(state, "reason", event.index);
      state.reasoning.set(itemId, `${state.reasoning.get(itemId) ?? ""}${event.delta.thinking}`);
      events.push({ method: "item/reasoning/summaryTextDelta", params: { itemId, delta: event.delta.thinking } });
    }
    return events;
  }
  if (message.type === "assistant") {
    events.push(...usageEvents(message.message?.id, message.message?.usage));
    const content = message.message?.content ?? [];
    const toolUseIds = content.filter(block => block.type === "tool_use" && block.id).map(block => block.id);
    if (toolUseIds.length > 0 && message.message?.id) {
      events.push({ method: "usage/toolUses", params: { callId: message.message.id, toolUseIds } });
    }
    for (const [index, block] of content.entries()) {
      if (block.type === "text" && block.text) {
        const itemId = block.id ?? messageItemId(state.text, message, "text", content, index);
        const previous = state.text.get(itemId) ?? "";
        const delta = block.text.startsWith(previous) ? block.text.slice(previous.length) : block.text;
        state.text.set(itemId, block.text);
        if (delta) events.push({ method: "item/agentMessage/delta", params: { itemId, delta } });
      }
      if (block.type === "thinking" && block.thinking) {
        const itemId = block.id ?? messageItemId(state.reasoning, message, "reason", content, index);
        const previous = state.reasoning.get(itemId) ?? "";
        const delta = block.thinking.startsWith(previous) ? block.thinking.slice(previous.length) : block.thinking;
        state.reasoning.set(itemId, block.thinking);
        if (delta) events.push({ method: "item/reasoning/summaryTextDelta", params: { itemId, delta } });
      }
      if (block.type === "tool_use") {
        const item = { type: "toolUse", id: block.id, name: block.name, input: block.input, status: "inProgress" };
        if (!state.activities.has(item.id)) {
          state.activities.add(item.id);
          events.push({ method: "item/started", params: { item } });
        }
      }
    }
  }
  if (message.type === "user") {
    for (const block of message.message?.content ?? []) {
      if (block.type !== "tool_result") continue;
      events.push({
        method: "item/completed",
        params: {
          item: {
            type: "toolUse",
            id: block.tool_use_id,
            output: block.content,
            status: block.is_error ? "failed" : "completed",
          },
        },
      });
    }
  }
  if (message.type === "system" && message.subtype === "permission_denied") {
    events.push({
      method: "item/completed",
      params: {
        item: {
          type: "toolUse",
          id: message.tool_use_id,
          name: message.tool_name,
          output: message.message,
          status: "failed",
        },
      },
    });
  }
  return events;
}

const CALL_USAGE_FIELDS = ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"];

/**
 * One model call's usage as the SDK reports it. A call reports usage several
 * times (at message start, in the output deltas, on each streamed assistant
 * message), so each report is an activity and the adapter keeps the largest
 * value of each field.
 */
function usageEvents(callId, usage) {
  if (!callId || !usage) return [];
  const fields = {};
  for (const field of CALL_USAGE_FIELDS) {
    if (Number.isSafeInteger(usage[field])) fields[field] = usage[field];
  }
  return Object.keys(fields).length > 0
    ? [{ method: "usage/apiCall", params: { callId, usage: fields } }]
    : [];
}

function streamItemId(state, type, index) {
  return `${state.currentMessageId ?? "message"}-${type}-${index ?? 0}`;
}

function messageItemId(items, message, type, content, index) {
  const prefix = `${message.message?.id ?? message.uuid ?? "message"}-${type}-`;
  const ordinal = content.slice(0, index).filter(block => block.type === (type === "reason" ? "thinking" : type)).length;
  const existing = [...items.keys()]
    .filter(itemId => itemId.startsWith(prefix))
    .sort((left, right) => Number(left.slice(prefix.length)) - Number(right.slice(prefix.length)));
  return existing[ordinal] ?? `${prefix}${index}`;
}

function responseForRequest(pending, response) {
  if (response.action === "accept" || response.action === "allow") {
    return { behavior: "allow", updatedInput: response.updatedInput ?? pending.input };
  }
  if (response.action === "answer") {
    return { behavior: "allow", updatedInput: { ...pending.input, answers: response.answers ?? {} } };
  }
  return { behavior: "deny", message: response.message ?? "User declined this Claude tool request" };
}

function resultError(message) {
  if (!message?.is_error) return null;
  return new Error(message.errors?.join("\n") || message.subtype || "Claude SDK turn failed");
}

/**
 * An unset effort must stay unset, so Claude Code applies its own default.
 * Ultracode is this plugin's own row and is not an SDK effort: it selects
 * xhigh, and its standing-orchestration half is carried in the prompt.
 */
function effortOf(message, session) {
  const effort = message.effort ?? session.config?.effort ?? undefined;
  return effort === ULTRACODE_EFFORT ? ULTRACODE_BASE_EFFORT : effort;
}

/** Whether this turn runs under the Ultracode effort. */
function isUltracode(message, session) {
  return (message.effort ?? session.config?.effort) === ULTRACODE_EFFORT;
}

function sdkPermissionMode(message) {
  if (message.permissionMode) return message.permissionMode;
  if (message.approvalPolicy === "never") return "dontAsk";
  if (message.sandbox === "read-only") return "plan";
  return "default";
}
