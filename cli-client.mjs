import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import readline from "node:readline";

const DEFAULT_MODELS = [
  { id: "opus", displayName: "Opus 5", isDefault: true },
  { id: "fable", displayName: "Fable 5", isDefault: false },
  { id: "sonnet", displayName: "Sonnet 5", isDefault: false },
  { id: "haiku", displayName: "Haiku 4.5", isDefault: false },
];

/**
 * How long a picture turn waits for its CLI's stdout after the CLI exits. The output is already in the pipe at exit,
 * so draining it is local I/O that takes milliseconds even for a multi-megabyte line. 2 s covers a loaded host and
 * bounds the delay a grandchild holding stdout open can cause.
 */
const STDOUT_GRACE_MS = 2_000;

/** Pass --effort only when a level was chosen, so the CLI's own default stands. */
function effortArgs(message, session) {
  const effort = message.effort ?? session.config?.effort;
  return effort ? ["--effort", effort] : [];
}

export class ClaudeCliClient extends EventEmitter {
  constructor({ command = "claude", args = [], env = process.env, requestTimeoutMs = 30 * 60_000 } = {}) {
    super();
    this.command = command;
    this.args = args;
    this.env = env;
    this.requestTimeoutMs = requestTimeoutMs;
    this.sessions = new Map();
    this.processes = new Map();
    this.closed = false;
  }

  async start() {
    this.closed = false;
  }

  /**
   * The CLI fallback has no in-process tool channel, so a harness that bridges
   * its own tools cannot use it. Fail loudly rather than running a turn whose
   * tools silently do not exist.
   */
  awaitToolCall() {
    return Promise.reject(new Error("The Claude CLI fallback cannot host harness tools; the Agent SDK backend is required"));
  }

  resolveToolCall() { return false; }
  rejectToolCall() { return false; }
  rejectAllToolCalls() {}

  async listModels() {
    return DEFAULT_MODELS;
  }

  async createSession(config = {}) {
    const id = config.sessionId ?? randomUUID();
    const session = { id, cwd: config.cwd ?? process.cwd(), created: false, config: structuredClone(config) };
    this.sessions.set(id, session);
    return { id, cwd: session.cwd, turns: [] };
  }

  async resumeSession(sessionId, config = {}) {
    const existing = this.sessions.get(sessionId) ?? { id: sessionId, created: true };
    const session = { ...existing, cwd: config.cwd ?? existing.cwd ?? process.cwd(), config: { ...existing.config, ...config } };
    this.sessions.set(sessionId, session);
    return { id: sessionId, cwd: session.cwd, turns: [] };
  }

  async sendMessage(sessionId, message = {}) {
    if (Array.isArray(message.dshTools) && message.dshTools.length > 0) {
      throw new Error("The Claude CLI backend cannot expose DSH tools; use the Claude Agent SDK backend");
    }
    const session = this.sessions.get(sessionId) ?? (await this.resumeSession(sessionId, message));
    const turnId = randomUUID();
    const child = this.spawnTurn(session, turnId, message);
    this.processes.set(turnId, child);
    return { id: turnId, status: "inProgress", items: [] };
  }

  async interruptTurn(_sessionId, turnId) {
    const child = this.processes.get(turnId);
    if (!child) return;
    child.kill("SIGTERM");
  }

  async releaseSession(sessionId) {
    for (const [turnId, child] of this.processes) {
      if (child.relayClaudeSessionId === sessionId) {
        child.kill("SIGTERM");
        this.processes.delete(turnId);
      }
    }
    this.sessions.delete(sessionId);
  }

  async close() {
    this.closed = true;
    for (const child of this.processes.values()) child.kill("SIGTERM");
    this.processes.clear();
  }

  spawnTurn(session, turnId, message) {
    const settingSourceArgs = settingSourceArguments(message.settingSources ?? session.config?.settingSources);
    const systemPromptArgs = systemPromptArguments(message.systemPrompt ?? session.config?.systemPrompt);
    // Pictures cannot go on the command line, so a turn with pictures streams its message on stdin.
    const withImages = Array.isArray(message.images) && message.images.length > 0;
    const cliArgs = [
      ...this.args,
      "-p",
      ...(withImages ? ["--input-format", "stream-json"] : [message.text]),
      "--output-format",
      "stream-json",
      "--verbose",
      "--include-partial-messages",
      "--model",
      message.model ?? session.config?.model ?? "sonnet",
      ...effortArgs(message, session),
      "--permission-mode",
      permissionMode(message),
      ...settingSourceArgs,
      ...systemPromptArgs,
      ...(session.created ? ["--resume", session.id] : ["--session-id", session.id]),
    ];
    const child = spawn(this.command, cliArgs, {
      cwd: message.cwd ?? session.cwd ?? process.cwd(),
      stdio: [withImages ? "pipe" : "ignore", "pipe", "pipe"],
      // The same environment the SDK backend hands its child. It defaults to
      // this process's own environment; a caller that configured a gateway
      // endpoint passes the environment carrying that endpoint here too, so the
      // fallback cannot quietly reach Anthropic directly while the SDK path
      // routes through the gateway.
      env: this.env,
    });
    child.relayClaudeSessionId = session.id;
    // Settles once the picture message is written. It settles with an error when the write fails or the stream closes
    // first, which happens when the CLI exits before reading the picture.
    const inputWritten = withImages
      ? new Promise(resolve => {
          child.stdin.once("finish", () => resolve(null));
          child.stdin.once("close", () => resolve(new Error("the input closed before the picture was written")));
          child.stdin.on("error", error => {
            this.emit("diagnostic", `Claude CLI did not read its input: ${error.message}`);
            resolve(error);
          });
          child.stdin.end(`${imageInputLine(message)}\n`);
        })
      : Promise.resolve(null);
    session.created = true;
    const state = { textItemId: null, text: "", activities: new Set(), answered: false };
    const output = readline.createInterface({ input: child.stdout });
    output.on("line", (line) => this.handleLine(session.id, turnId, line, state));
    const outputEnded = new Promise(resolve => output.once("close", resolve));
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", chunk => this.emit("diagnostic", String(chunk)));
    let settled = false;
    // A turn completes once, even when a kill failure reports an error and the exit follows it.
    const complete = (status, error) => {
      if (settled) return;
      settled = true;
      this.completeTurn(session.id, turnId, status, error);
    };
    child.once("error", error => {
      this.emit("diagnostic", `Claude CLI failed: ${error.message}`);
      complete("failed", error);
    });
    child.once("exit", async (code, signal) => {
      this.processes.delete(turnId);
      // Node does not order exit against the stdin write, so the outcome waits for the write to settle.
      const inputError = await inputWritten;
      // Node may emit exit before the last stdout lines are read, so a picture turn waits for stdout to end.
      // The wait is capped, because a grandchild holding stdout open would otherwise hang the turn.
      if (withImages) {
        let timer;
        await Promise.race([outputEnded, new Promise(resolve => { timer = setTimeout(resolve, STDOUT_GRACE_MS); })]);
        clearTimeout(timer);
      }
      // Exit 0 completes a text turn; a picture turn completes only after its stream-json result line.
      if (signal || code) complete("failed", new Error(`claude exited (${signal ?? code})`));
      else if (inputError) complete("failed", new Error(`Claude CLI did not read its input: ${inputError.message}`));
      else if (withImages && !state.answered) complete("failed", new Error("Claude CLI exited without answering the picture"));
      else complete("completed");
    });
    return child;
  }

  handleLine(sessionId, turnId, line, state) {
    let message;
    try {
      message = JSON.parse(line);
    } catch {
      this.emitText(sessionId, turnId, state, line);
      return;
    }
    if (message.type === "result") state.answered = true;
    for (const event of normalizeClaudeStreamMessage(message, state)) {
      this.emit("activity", { method: event.method, params: { sessionId, turnId, ...event.params } });
    }
  }

  emitText(sessionId, turnId, state, text) {
    if (!text) return;
    state.textItemId ??= `answer-${turnId}`;
    this.emit("activity", {
      method: "item/agentMessage/delta",
      params: { sessionId, turnId, itemId: state.textItemId, delta: `${text}\n` },
    });
  }

  completeTurn(sessionId, turnId, status, error = null) {
    this.emit("activity", {
      method: "turn/completed",
      params: {
        sessionId,
        turn: {
          id: turnId,
          status,
          error: error ? { message: error.message } : null,
          items: [],
        },
      },
    });
  }
}

function normalizeClaudeStreamMessage(message, state) {
  const events = [];
  const content = message.message?.content ?? message.content ?? [];
  for (const block of Array.isArray(content) ? content : []) {
    if (block.type === "text" && block.text) {
      state.textItemId ??= block.id ?? `answer-${message.message?.id ?? "latest"}`;
      const delta = block.text.startsWith(state.text) ? block.text.slice(state.text.length) : block.text;
      state.text = block.text;
      if (delta) events.push({ method: "item/agentMessage/delta", params: { itemId: state.textItemId, delta } });
    }
    if ((block.type === "text_delta" || block.type === "content_block_delta") && (block.text ?? block.delta?.text)) {
      state.textItemId ??= block.id ?? `answer-${message.message?.id ?? "latest"}`;
      const delta = block.text ?? block.delta.text;
      state.text += delta;
      events.push({ method: "item/agentMessage/delta", params: { itemId: state.textItemId, delta } });
    }
    if (block.type === "thinking" && block.thinking) {
      events.push({ method: "item/reasoning/summaryTextDelta", params: { itemId: block.id ?? "reasoning", delta: block.thinking } });
    }
    if (block.type === "tool_use") {
      const item = {
        type: "toolUse",
        id: block.id ?? block.name,
        name: block.name,
        input: block.input,
        status: "inProgress",
      };
      if (!state.activities.has(item.id)) {
        state.activities.add(item.id);
        events.push({ method: "item/started", params: { item } });
      }
    }
    if (block.type === "tool_result") {
      events.push({
        method: "item/completed",
        params: {
          item: {
            type: "toolUse",
            id: block.tool_use_id ?? block.id,
            name: block.name,
            output: block.content,
            status: block.is_error ? "failed" : "completed",
          },
        },
      });
    }
  }
  if (message.type === "result" && message.result) {
    state.textItemId ??= `answer-${message.session_id ?? "latest"}`;
    const delta = String(message.result).startsWith(state.text)
      ? String(message.result).slice(state.text.length)
      : String(message.result);
    state.text = String(message.result);
    if (delta) events.push({ method: "item/agentMessage/delta", params: { itemId: state.textItemId, delta } });
  }
  return events;
}

/** One user message for the CLI's stdin: the text, then one base64 image block per picture. */
function imageInputLine(message) {
  const content = [];
  if (message.text) content.push({ type: "text", text: message.text });
  for (const image of message.images) {
    content.push({ type: "image", source: { type: "base64", media_type: image.mediaType, data: image.data } });
  }
  return JSON.stringify({ type: "user", message: { role: "user", content }, parent_tool_use_id: null });
}

function permissionMode(message) {
  if (message.permissionMode) return message.permissionMode;
  if (message.approvalPolicy === "never") return "plan";
  if (message.sandbox === "read-only") return "plan";
  return "manual";
}

function settingSourceArguments(value) {
  if (Array.isArray(value) && value.length === 0) return ["--safe-mode"];
  if (Array.isArray(value)) return ["--setting-sources", value.join(",")];
  if (typeof value === "string" && value.trim()) return ["--setting-sources", value];
  return ["--setting-sources", "user,project,local"];
}

function systemPromptArguments(value) {
  if (typeof value === "string" && value.trim()) return ["--system-prompt", value];
  return [];
}
