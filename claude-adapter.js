import { LlmAdapter } from "@deepseek-ai/dsh-llm";
import { debugLog } from "./debug.js";
import { ULTRACODE_EFFORT } from "./orchestration.js";
import { dshToolResult } from "./sdk-client.mjs";
import { scanOnlyTelemetryControl } from "./telemetry-control.js";

// Identification, not branding: the row names the thing it invokes.
export const CLAUDE_PROVIDER = "claude";

/** Appended when a parked turn continues in a new Claude session after DSH rewrote its history. */
const PARKED_TURN_NOTE = "Earlier tool calls for this request already ran. Their results, or the summary of them, are in the context above; continue from them and do not repeat those calls.";

export class ClaudeDshAdapter extends LlmAdapter {
  constructor({ runtime, ready, linkStore = null, logger = console, telemetry = scanOnlyTelemetryControl(), thinkingSummaries = true }) {
    super();
    this.telemetry = telemetry;
    this.thinkingSummaries = thinkingSummaries;
    this.runtime = runtime;
    this.ready = ready;
    this.logger = logger;
    this.linkStore = linkStore;
    this.links = new Map();
    this.settings = new Map();
    this.pendingSessions = new Map();
    this.agents = new Map();
    this.seen = new Map();
    // The DSH log position (seq) that each Claude transcript was last sent up to. A rewrite of
    // DSH history that Claude holds (up to that position, or a reply it produced since) means
    // the transcript no longer matches DSH.
    this.builtAt = new Map();
    // DSH sessions whose Claude turn is parked on tool calls DSH is executing.
    this.suspended = new Map();
    this.usageLedgers = new Map();
    // Claude sessions whose turns this adapter is running or has suspended. Only their
    // usage is recorded, so an adapter sharing a runtime never keeps another's turns.
    this.ownedSessions = new Set();
    // Claude sessions whose turn is being read by a live step right now.
    this.readers = new Set();
    // Turns that have finished; a late report for one of them is never recorded.
    this.finishedTurns = new Set();
    // Waiting tool calls are recorded for each linked Claude session, and usage for each owned turn.
    // Both are recorded for the life of the adapter, including the gap while a parked step waits
    // for DSH to run its tools, so neither is lost to that gap.
    this.waitingCalls = new Map();
    this.activityUnsubscribe = null;
    if (runtime && (typeof runtime.subscribeActivity === "function" || typeof runtime.on === "function")) {
      try {
        this.activityUnsubscribe = subscribeRuntimeActivity(runtime, message => {
          this.recordWaitingCall(message);
          this.recordUsageActivity(message);
        });
      } catch (error) {
        this.logger?.warn?.(`Claude usage and waiting-call reporting is unavailable: ${error?.message ?? error}`);
      }
    }
    for (const [sessionId, record] of linkStore?.entries() ?? []) {
      const claudeSessionId = record.claudeSessionId ?? record.sessionId ?? record.threadId;
      if (claudeSessionId) this.links.set(sessionId, claudeSessionId);
      this.settings.set(sessionId, record.config);
      if (Number.isSafeInteger(record.seen)) this.seen.set(sessionId, record.seen);
      if (Number.isSafeInteger(record.surfaceSeq)) this.builtAt.set(sessionId, record.surfaceSeq);
    }
  }

  /** Stop recording usage from the runtime, and forget every ledger and owned session. */
  dispose() {
    this.activityUnsubscribe?.();
    this.activityUnsubscribe = null;
    this.usageLedgers.clear();
    this.ownedSessions.clear();
    this.waitingCalls.clear();
    this.readers.clear();
    this.finishedTurns.clear();
  }

  providerInfo() {
    // Anthropic's Agent SDK branding guidance lists both "Claude Agent"
    // (preferred for dropdowns) and bare "Claude" (within a menu labelled
    // "Agents") as allowed. DSH's picker has no such heading, so the literal
    // reading favours the former — but "preferred" is not "required", and bare
    // "Claude" is what every BYO-subscription harness ships, T3 Code included.
    // What the guidance forbids is naming the product after Claude Code, which
    // nothing here does.
    return { id: CLAUDE_PROVIDER, name: "Claude" };
  }

  async listModels() {
    await this.ready;
    return runtimeModels(this.runtime)
      .sort((left, right) => Number(Boolean(right.isDefault)) - Number(Boolean(left.isDefault)))
      .map(model => ({
        provider: CLAUDE_PROVIDER,
        id: model.id,
        name: model.displayName ?? model.id,
        inputModalities: ["text", "image"],
      }));
  }

  async resolveModel(provider, model) {
    await this.ready;
    const info = runtimeModels(this.runtime).find(candidate => candidate.id === model);
    const contextWindow = this.runtime.contextWindowFor?.(model, info?.resolvedModel);
    return {
      provider,
      id: model,
      name: info?.displayName ?? model,
      inputModalities: ["text", "image"],
      // Without this the harness cannot show context pressure for the row.
      ...(contextWindow ? { context: { contextWindow } } : {}),
      ...(Array.isArray(info?.supportedReasoningEfforts)
        ? {
            reasoning: {
              efforts: info.supportedReasoningEfforts.map(effort => ({
                id: effort.reasoningEffort ?? effort.id ?? effort,
                name: reasoningEffortName(effort.reasoningEffort ?? effort.id ?? effort),
              })),
              defaultEffort: info.defaultReasoningEffort,
            },
          }
        : {}),
    };
  }

  attachAgent(agent) {
    this.agents.set(String(agent.id), agent);
    this.configuration(agent.id, agent.session.header.cwd);
    return true;
  }

  detachAgent(sessionId) {
    this.agents.delete(String(sessionId));
  }

  configuration(sessionId, cwd) {
    const key = String(sessionId);
    const existing = this.settings.get(key);
    if (existing) return existing;
    const models = runtimeModels(this.runtime);
    const model = models.find(candidate => candidate.isDefault) ?? models[0];
    const config = {
      model: model?.id ?? "sonnet",
      effort: model?.defaultReasoningEffort,
      sandbox: "workspace-write",
      approvalPolicy: "on-request",
      cwd: cwd ?? process.cwd(),
      // DSH's preset owns the prompt and the tool set; do not load ~/.claude
      // settings, CLAUDE.md, or the claude_code system prompt on top of it.
      settingSources: [],
      systemPrompt: undefined,
      thinkingSummaries: this.thinkingSummaries,
    };
    this.settings.set(key, config);
    return config;
  }

  configure(sessionId, patch = {}) {
    const key = String(sessionId);
    const next = { ...this.configuration(key), ...compact(patch) };
    this.settings.set(key, next);
    const claudeSessionId = this.links.get(key);
    if (claudeSessionId) {
      patchRuntimeSession(this.runtime, claudeSessionId, next);
    }
    this.persistLink(key);
    return structuredClone(next);
  }

  async ensureSession(sessionId) {
    const key = String(sessionId);
    const pending = this.pendingSessions.get(key);
    if (pending) return pending;
    const operation = this.createOrResumeSession(key).finally(() => {
      this.pendingSessions.delete(key);
    });
    this.pendingSessions.set(key, operation);
    return operation;
  }

  async createOrResumeSession(sessionId) {
    await this.ready;
    const settings = { ...this.configuration(sessionId) };
    const linked = this.links.get(sessionId);
    if (linked && hasRuntimeSession(this.runtime, linked)) return linked;
    if (linked) {
      try {
        await this.runtime.resumeSession(linked, settings);
        return linked;
      } catch (error) {
        this.logger.warn(`Relay could not resume Claude session ${linked}; creating a replacement: ${error.message}`);
        this.links.delete(sessionId);
      }
    }
    const created = await this.runtime.createSession(settings);
    this.links.set(sessionId, created.id);
    this.persistLink(sessionId);
    return created.id;
  }

  persistLink(sessionId) {
    this.linkStore?.set(sessionId, {
      claudeSessionId: this.links.get(sessionId) ?? null,
      config: this.configuration(sessionId),
      seen: this.seen.get(String(sessionId)) ?? 0,
      surfaceSeq: this.builtAt.get(String(sessionId)) ?? null,
    });
  }

  sessionFor(sessionId) {
    return this.links.get(String(sessionId)) ?? null;
  }

  dshSessionForClaudeSession(claudeSessionId) {
    for (const [sessionId, candidate] of this.links) {
      if (candidate === claudeSessionId) return sessionId;
    }
    return null;
  }

  async *stream(options) {
    await this.telemetry.enforce();
    if (options.purpose) {
      yield* this.streamAuxiliary(options);
      return;
    }
    const sessionId = String(options.sessionId ?? "");
    if (!sessionId) throw new Error("The Claude adapter requires a DSH session id");
    const agent = this.agents.get(sessionId);
    if (!agent) throw new Error(`The Claude adapter has no attached agent for ${sessionId}`);

    // DSH may have rewritten history that the Claude transcript already holds, most
    // often by compacting its context. Continuing that transcript would send the old
    // context again, so a parked turn is released and this request starts a new Claude
    // session from DSH's history.
    let suspended = this.suspended.get(sessionId);
    const rewritten = this.historyRewritten(sessionId, agent);
    const parkedTurnRewritten = rewritten && Boolean(suspended);
    if (rewritten) {
      if (suspended) await this.releaseParkedTurn(sessionId, suspended);
      suspended = undefined;
      await this.retireClaudeSession(sessionId);
    }
    debugLog("stream", {
      sessionId,
      messages: options.messages?.length ?? 0,
      tools: options.tools?.length ?? 0,
      suspended: suspended ? suspended.calls : null,
      toolResultIds: allToolResultIds(options.messages),
      rewritten,
    });
    // A turn suspended on tool calls resumes here: DSH ran them and is calling
    // back with the results, so the same Claude query continues rather than a
    // new one starting.
    if (suspended) {
      const results = toolResultsFor(options.messages, suspended.calls);
      // A DSH turn that started after the park was stopped mid-tool: DSH cut the
      // running tool short, and the user's next message opens a new turn. Resuming
      // the stopped query would hand Claude the cut-short result and drop that message.
      const superseded = turnStartedSince(agent, suspended.dshSeq);
      if (results.length && !superseded) {
        // Messages DSH received while the tool ran ride on its result: a running Claude
        // query cannot take a new user message.
        const inputs = this.undeliveredInput(sessionId, options.messages);
        debugLog("resume", { sessionId, turnId: suspended.turnId, resolving: results.map(result => result.toolCallId), inputs: inputs.length });
        this.markBuilt(sessionId, agent);
        yield* this.runTurn(agent, sessionId, suspended.claudeSessionId, options, async () => {
          for (const result of resultsWithInput(results, inputs)) {
            const delivered = this.runtime.resolveToolCall(result.toolCallId, dshToolResult(result));
            debugLog("resolve", { toolCallId: result.toolCallId, delivered });
            if (delivered) this.waitingCalls.get(suspended.claudeSessionId)?.delete(result.toolCallId);
            if (!delivered) {
              // Nothing is waiting on this id, so nothing will ever wake the
              // turn. Fail now rather than stalling for the user to notice.
              throw new Error(`No Claude tool call is waiting for result ${result.toolCallId}; the turn cannot be resumed`);
            }
          }
          this.markDelivered(sessionId, options.messages);
          return suspended.turnId;
        }, { firstEventTimeoutMs: resumeTimeoutMs(), knownTurnId: suspended.turnId });
        return;
      }
      debugLog("abandon", { sessionId, turnId: suspended.turnId, expected: suspended.calls, superseded });
      // No results for the calls we parked, or a newer DSH turn replaced the parked
      // one: release Claude before starting a new one.
      await this.abandonTurn(sessionId, suspended);
    }

    const instruction = latestUserIndex(options.messages);
    const text = instruction === -1 ? "" : messageText(options.messages[instruction]);
    if (!text) throw new Error("The Claude adapter received no user text");
    const nativePermissions = permissionConfiguration(sessionEventLog(agent.session));
    const config = this.configure(sessionId, {
      model: options.model,
      effort: options.reasoningEffort,
      systemPrompt: options.system,
      ...nativePermissions,
      cwd: agent.session.header.cwd,
    });
    const dshTools = structuredClone(options.tools ?? []);
    const claudeSessionId = await this.ensureSession(sessionId);
    const seeded = this.seedPrompt(sessionId, options.messages, instruction, text);
    const prompt = parkedTurnRewritten ? `${seeded}\n\n${PARKED_TURN_NOTE}` : seeded;
    this.markBuilt(sessionId, agent);
    yield* this.runTurn(agent, sessionId, claudeSessionId, options, async () => {
      const started = await this.runtime.sendMessage(claudeSessionId, { text: prompt, ...config, dshTools });
      this.markDelivered(sessionId, options.messages);
      return started.id;
    });
  }

  /**
   * Project one DSH step of a Claude turn.
   *
   * A step ends one of two ways: Claude finishes, or it calls tools. Tool calls
   * are handed to DSH as `tool-call` chunks with a `tool-calls` finish, so DSH's
   * own agent loop executes them — under its sandbox and approval policy, and
   * recorded in its trajectory — and then calls back into `stream()` with the
   * results, which resume the same Claude query.
   */
  async *runTurn(agent, sessionId, claudeSessionId, options, start, { firstEventTimeoutMs = 0, knownTurnId = null } = {}) {
    const queue = new ActivityQueue(options.signal, "Claude");
    const onActivity = (message) => {
      const candidate = message.params?.sessionId ?? message.params?.session?.id;
      if (candidate === claudeSessionId && !String(message.method).startsWith("usage/")) queue.push(message);
    };
    // Subscribe before starting or resuming: a runtime that emits synchronously
    // would otherwise deliver the whole turn before anyone is listening.
    const stopActivity = subscribeRuntimeActivity(this.runtime, onActivity);
    // A resumed turn already has its id, so a failure to start it still releases that turn.
    let turnId = knownTurnId;
    this.ownedSessions.add(claudeSessionId);
    this.readers.add(claudeSessionId);
    try {
      turnId = await start();
      debugLog("turn/started", { sessionId, claudeSessionId, turnId });
      const state = createStreamState(this.usageLedgerFor(turnId, claudeSessionId));
      let completedTurn = null;
      let batch = null;
      let first = true;
      while (!completedTurn && !batch) {
        // A resumed turn that never speaks again is the one failure with no
        // stack to show for it: the query is parked inside the SDK. Bound the
        // first wait so it surfaces as an error the user can act on.
        const message = await (first && firstEventTimeoutMs
          ? withTimeout(queue.next(), firstEventTimeoutMs, `Claude did not respond within ${Math.round(firstEventTimeoutMs / 1000)}s of the tool result being delivered`)
          : queue.next());
        first = false;
        const params = message.params ?? {};
        if (params.turnId && params.turnId !== turnId) continue;
        if (message.method === "turn/completed") {
          if (params.turn?.id !== turnId) continue;
          for (const item of params.turn.items ?? []) {
            for (const chunk of this.completeItem(item, state)) yield chunk;
          }
          completedTurn = params.turn;
          break;
        }
        if (isToolParked(message)) {
          batch = [message.params, ...drainParkedToolCalls(queue, turnId)];
          break;
        }
        for (const chunk of this.projectActivity(message, state)) yield chunk;
      }

      if (batch) {
        for await (const chunk of this.settleParkedStep(queue, state, batch, turnId)) yield chunk;
      }
      for (const chunk of closeOpenBlocks(state)) yield chunk;

      // The protocol wants usage before the terminal finish, once per DSH step.
      // Claude reports each model call as it runs, so a step's usage is the sum
      // of the calls it made. See stepUsageChunks for the turn-level fallback.
      for (const chunk of stepUsageChunks(state, completedTurn)) yield chunk;
      if (completedTurn && turnId) this.usageLedgers.delete(turnId);

      if (batch) {
        const calls = batch.map(parked => ({
          id: parked.parkId,
          name: parked.name,
          arguments: JSON.stringify(parked.arguments ?? {}),
        }));
        for (const call of calls) {
          const index = state.nextIndex++;
          yield { type: "block-start", index, blockType: "tool-call" };
          yield { type: "tool-call-delta", index, id: call.id, name: call.name, argumentsDelta: call.arguments };
          yield { type: "block-end", index, block: { type: "tool-call", id: call.id, name: call.name, arguments: call.arguments } };
        }
        this.suspended.set(sessionId, { claudeSessionId, turnId, calls: calls.map(call => call.id), dshSeq: logHead(agent) });
        debugLog("handoff", { sessionId, turnId, calls: calls.map(call => ({ id: call.id, name: call.name })) });
        yield { type: "finish", reason: { kind: "tool-calls" } };
        return;
      }

      this.suspended.delete(sessionId);
      if (completedTurn.status === "failed") {
        yield {
          type: "finish",
          reason: { kind: "error", failure: { message: completedTurn.error?.message ?? "Claude turn failed", code: "CLAUDE_TURN_FAILED" } },
        };
      } else {
        this.persistLink(sessionId);
        debugLog("finish", { sessionId, turnId, kind: "stop" });
        yield { type: "finish", reason: { kind: "stop" }, replayState: { claudeSessionId, turnId } };
      }
    } catch (error) {
      this.releaseTurn(claudeSessionId, turnId);
      debugLog("turn/error", { sessionId, turnId, aborted: Boolean(options.signal?.aborted), message: String(error?.message ?? error) });
      if (options.signal?.aborted) {
        if (turnId) await this.abandonTurn(sessionId, { claudeSessionId, turnId });
        yield { type: "finish", reason: { kind: "aborted", failure: { message: "Claude turn cancelled", code: "ABORTED" } } };
        return;
      }
      this.suspended.delete(sessionId);
      throw error;
    } finally {
      stopActivity();
      queue.close();
      this.readers.delete(claudeSessionId);
      if (turnId && (this.usageLedgers.get(turnId)?.completed || this.finishedTurns.has(turnId)) && !this.suspended.has(sessionId)) {
        this.usageLedgers.delete(turnId);
      }
    }
  }

  /** Remember a tool call that a linked Claude session announced, until its result is delivered or it is failed. */
  recordWaitingCall(message) {
    if (!isToolParked(message)) return;
    const { sessionId, parkId } = message.params;
    // Linked, not owned: a stale completion of an abandoned turn clears ownership while its replacement still waits.
    if (!this.dshSessionForClaudeSession(sessionId)) return;
    if (!this.waitingCalls.has(sessionId)) this.waitingCalls.set(sessionId, new Set());
    this.waitingCalls.get(sessionId).add(parkId);
  }

  /** Forget and return the tool calls a Claude session is still waiting on. */
  takeWaitingCalls(claudeSessionId) {
    const calls = [...(this.waitingCalls.get(claudeSessionId) ?? [])];
    this.waitingCalls.delete(claudeSessionId);
    return calls;
  }

  /** Forget a turn's ledger and its Claude session, once nothing more will report usage for them. */
  releaseTurn(claudeSessionId, turnId) {
    this.rememberFinished(turnId);
    for (const [id, ledger] of this.usageLedgers) {
      if (ledger.sessionId === claudeSessionId || id === turnId) this.usageLedgers.delete(id);
    }
    this.ownedSessions.delete(claudeSessionId);
    this.readers.delete(claudeSessionId);
  }

  /**
   * Release a Claude turn nobody is going to finish, and fail the tool calls its Claude
   * session is still waiting on. Only that session's calls are failed: rejectAllToolCalls
   * would also fail the calls of other DSH sessions sharing the runtime.
   */
  async abandonTurn(sessionId, { claudeSessionId, turnId }) {
    this.suspended.delete(sessionId);
    this.releaseTurn(claudeSessionId, turnId);
    const error = new Error("The DSH turn was abandoned before the tool result arrived");
    for (const call of this.takeWaitingCalls(claudeSessionId)) this.runtime.rejectToolCall?.(call, error);
    if (turnId) await this.runtime.interruptTurn(claudeSessionId, turnId).catch(() => {});
  }

  /**
   * True when DSH has rewritten history that the linked Claude transcript already
   * holds. A rewrite is a surface event that replaces an existing span, as compaction
   * and tool-result pruning do. Claude holds everything up to the position it was last
   * sent, plus the replies it produced after that, so a rewrite counts when it is logged
   * after the position and its span starts at or before the newest reply Claude holds.
   */
  historyRewritten(sessionId, agent) {
    const key = String(sessionId);
    if (!this.links.has(key)) return false;
    const log = sessionEventLog(agent.session);
    const built = this.builtAt.get(key);
    // A link saved before positions were recorded: nothing says what Claude holds,
    // so any rewrite in the log counts.
    if (built === undefined) return log.some(isReplaceEvent);
    const logged = [];
    for (let index = log.length - 1; index >= 0 && log[index].seq > built; index -= 1) logged.push(log[index]);
    const held = logged.reduce((seq, event) => (producedByClaude(event) ? Math.max(seq, event.seq) : seq), built);
    return logged.some(event => isReplaceEvent(event) && event.surfaceOp.startSeq <= held);
  }

  /**
   * Record that every DSH message so far has been handed to Claude, and persist it.
   * Called at each send, so a message DSH adds later is still undelivered.
   */
  markDelivered(sessionId, messages) {
    this.seen.set(String(sessionId), messages.length);
    this.persistLink(sessionId);
  }

  /**
   * DSH messages past the delivered position that carry text and are not tool
   * results: what the user typed, and any DSH context added, while a tool was running.
   */
  undeliveredInput(sessionId, messages) {
    const delivered = this.seen.get(String(sessionId)) ?? 0;
    return messages
      .slice(delivered)
      .filter(message => message?.role === "user" && toolResultsIn(message).length === 0 && messageText(message));
  }

  /** Record the DSH log position the Claude transcript now holds, and persist it. */
  markBuilt(sessionId, agent) {
    this.builtAt.set(String(sessionId), logHead(agent));
    this.persistLink(sessionId);
  }

  /** Forget a Claude session whose transcript no longer matches DSH's history, and release it. */
  async retireClaudeSession(sessionId) {
    const key = String(sessionId);
    const claudeSessionId = this.links.get(key) ?? null;
    this.links.delete(key);
    this.seen.delete(key);
    this.builtAt.delete(key);
    this.persistLink(key);
    if (!claudeSessionId) return;
    this.ownedSessions.delete(claudeSessionId);
    this.readers.delete(claudeSessionId);
    try {
      await this.runtime.releaseSession?.(claudeSessionId);
    } catch (error) {
      this.logger?.warn?.(`Relay could not release Claude session ${claudeSessionId}: ${error?.message ?? error}`);
    }
  }

  /**
   * Release a parked turn whose transcript DSH has rewritten. Only the calls its Claude
   * session is still waiting on are failed: rejectAllToolCalls would also fail the calls
   * of other DSH sessions sharing the runtime.
   */
  async releaseParkedTurn(sessionId, { claudeSessionId, turnId }) {
    this.suspended.delete(sessionId);
    this.releaseTurn(claudeSessionId, turnId);
    const error = new Error("DSH rewrote its history while the turn was waiting for a tool result");
    for (const call of this.takeWaitingCalls(claudeSessionId)) this.runtime.rejectToolCall?.(call, error);
    if (turnId) await this.runtime.interruptTurn(claudeSessionId, turnId).catch(() => {});
  }

  async *streamAuxiliary(options) {
    await this.telemetry.enforce();
    await this.ready;
    const text = auxiliaryInput(options.messages);
    if (!text) throw new Error(`Relay Claude adapter received no ${options.purpose} input`);
    const sessionId = String(options.sessionId ?? "");
    const agent = this.agents.get(sessionId);
    const cwd = agent?.session.header.cwd ?? this.settings.get(sessionId)?.cwd ?? process.cwd();
    const created = await this.runtime.createSession({
      model: options.model,
      effort: options.reasoningEffort,
      sandbox: "read-only",
      approvalPolicy: "never",
      cwd,
      ephemeral: true,
      settingSources: [],
      systemPrompt: options.system,
    });
    const claudeSessionId = created.id;
    const queue = new ActivityQueue(options.signal, "Claude");
    const onActivity = (message) => {
      const candidate = message.params?.sessionId ?? message.params?.session?.id;
      if (candidate === claudeSessionId) queue.push(message);
    };
    const stopActivity = subscribeRuntimeActivity(this.runtime, onActivity);
    let turnId = null;
    this.ownedSessions.add(claudeSessionId);
    try {
      const started = await this.runtime.sendMessage(claudeSessionId, {
        text,
        model: options.model,
        effort: options.reasoningEffort,
        sandbox: "read-only",
        approvalPolicy: "never",
      });
      turnId = started.id;
      const state = createStreamState();
      let completedTurn = null;
      while (!completedTurn) {
        const message = await queue.next();
        const params = message.params ?? {};
        if (params.turnId && params.turnId !== turnId) continue;
        if (message.method === "turn/completed") {
          if (params.turn?.id !== turnId) continue;
          for (const item of params.turn.items ?? []) {
            for (const chunk of completeAuxiliaryItem(state, item)) yield chunk;
          }
          completedTurn = params.turn;
          break;
        }
        for (const chunk of projectAuxiliaryActivity(message, state)) yield chunk;
      }
      for (const block of state.blocks.values()) {
        if (block.closed) continue;
        block.closed = true;
        yield { type: "block-end", index: block.index, block: { type: block.type, text: block.text } };
      }
      yield completedTurn.status === "failed"
        ? { type: "finish", reason: { kind: "error", failure: { message: completedTurn.error?.message ?? `Claude ${options.purpose} failed`, code: "CLAUDE_AUXILIARY_FAILED" } } }
        : { type: "finish", reason: { kind: "stop" } };
    } finally {
      stopActivity();
      queue.close();
      if (turnId) this.usageLedgers.delete(turnId);
      this.ownedSessions.delete(claudeSessionId);
      await this.runtime.releaseSession(claudeSessionId);
    }
  }

  /**
   * DSH owns the conversation; the Claude session only knows the turns it
   * answered. When the session is new, or when other models answered turns in
   * between, prepend those turns so a mid-session model switch does not drop
   * context.
   */
  seedPrompt(sessionId, messages, instruction, text) {
    const key = String(sessionId);
    const seen = this.seen.get(key) ?? 0;
    const context = messages
      .map((message, index) => ({ message, index }))
      .filter(entry => entry.index >= seen && entry.index !== instruction)
      .map(entry => {
        const body = seedText(entry.message);
        return body ? `${entry.message.role ?? "user"}: ${body}` : "";
      })
      .filter(Boolean)
      .join("\n\n");
    if (!context) return text;
    return [
      "<dsh-context>",
      "Context from the DeepSeek Harness session: workspace instructions, runtime snapshots, and any turns answered by another model. Background only — the request to act on follows this block.",
      context,
      "</dsh-context>",
      "",
      text,
    ].join("\n");
  }

  /**
   * The SDK starts a tool as soon as its block closes, so a parked call can
   * arrive before its model call has reported the final output count. Wait a
   * moment, bounded by the settle limit, for this step's calls to finish. While
   * waiting, the stream keeps moving: usage is recorded, another parked call
   * joins the batch, and text is passed on.
   */
  async *settleParkedStep(queue, state, batch, turnId) {
    const deadline = Date.now() + usageSettleMs();
    while (stepIsOpen(state, batch.length)) {
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      const message = await queue.nextWithin(Math.min(remaining, SETTLE_POLL_MS));
      if (!message) continue;
      if (message.params?.turnId && message.params.turnId !== turnId) continue;
      if (isToolParked(message)) {
        batch.push(message.params);
        continue;
      }
      if (message.method === "turn/completed") {
        // A parked turn cannot complete before its tool results arrive.
        queue.unshift(message);
        break;
      }
      for (const chunk of this.projectActivity(message, state)) yield chunk;
    }
    batch.push(...drainParkedToolCalls(queue, turnId));
  }

  /** Record one usage activity against its turn's ledger, whatever step is running. */
  recordUsageActivity(message) {
    const { sessionId } = message?.params ?? {};
    if (!this.ownedSessions.has(sessionId)) return;
    if (message.method === "turn/completed") {
      const turnId = message.params?.turn?.id;
      this.ownedSessions.delete(sessionId);
      if (!turnId) return;
      this.rememberFinished(turnId);
      const ledger = this.usageLedgers.get(turnId);
      if (!ledger) return;
      ledger.completed = true;
      // A live step deletes the ledger after reading it. With no live step, nothing will.
      if (!this.readers.has(ledger.sessionId)) this.usageLedgers.delete(turnId);
      return;
    }
    if (!String(message?.method).startsWith("usage/")) return;
    const turnId = message.params?.turnId;
    if (!turnId || this.finishedTurns.has(turnId)) return;
    recordUsage({ usage: this.usageLedgerFor(turnId, sessionId) }, message);
  }

  /** Remember a finished turn, bounded, so its late reports are refused. */
  rememberFinished(turnId) {
    if (!turnId) return;
    this.finishedTurns.add(turnId);
    if (this.finishedTurns.size > 256) this.finishedTurns.delete(this.finishedTurns.values().next().value);
  }

  /** The turn's usage ledger, which outlives each step while the turn is parked. */
  usageLedgerFor(turnId, sessionId) {
    let ledger = this.usageLedgers.get(turnId);
    if (!ledger) {
      ledger = newUsageLedger(sessionId);
      this.usageLedgers.set(turnId, ledger);
    }
    return ledger;
  }

  projectActivity(message, state) {
    const params = message.params ?? {};
    if (String(message.method).startsWith("usage/")) return [];
    if (message.method === "item/reasoning/summaryTextDelta" || message.method === "item/reasoning/textDelta") {
      return textDelta(state, params.itemId, "reasoning", params.delta ?? "");
    }
    if (message.method === "item/agentMessage/delta") {
      return textDelta(state, params.itemId, "text", params.delta ?? "");
    }
    if (message.method === "item/completed") return this.completeItem(params.item, state);
    return [];
  }

  completeItem(item, state) {
    if (!item?.id || state.completed.has(item.id)) return [];
    state.completed.add(item.id);
    if (item.type === "reasoning") return completeTextItem(state, item.id, "reasoning", reasoningText(item));
    if (item.type === "agentMessage") return completeTextItem(state, item.id, "text", item.text ?? "");
    // Tool items are DSH's to render: it executed them and recorded them in its
    // own trajectory.
    return [];
  }
}

class ActivityQueue {
  constructor(signal, label) {
    this.signal = signal;
    this.label = label;
    this.values = [];
    this.waiters = [];
    this.closed = false;
  }

  push(value) {
    if (this.closed) return;
    const waiter = this.waiters.shift();
    if (waiter) waiter.resolve(value);
    else this.values.push(value);
  }

  next() {
    if (this.values.length) return Promise.resolve(this.values.shift());
    if (this.closed) return Promise.reject(new Error(`${this.label} activity stream closed`));
    if (this.signal?.aborted) return Promise.reject(this.signal.reason ?? new Error("aborted"));
    return new Promise((resolve, reject) => {
      const waiter = { resolve, reject };
      this.waiters.push(waiter);
      if (this.signal) {
        const abort = () => {
          const index = this.waiters.indexOf(waiter);
          if (index >= 0) this.waiters.splice(index, 1);
          reject(this.signal.reason ?? new Error("aborted"));
        };
        this.signal.addEventListener("abort", abort, { once: true });
        waiter.resolve = (value) => {
          this.signal.removeEventListener("abort", abort);
          resolve(value);
        };
      }
    });
  }

  /**
   * Like next(), but resolves null when nothing arrives within ms, and rejects at
   * once if the turn is aborted. Every exit path detaches the waiter, so a value
   * that arrives later is never handed to a reader that has already left.
   */
  nextWithin(ms) {
    if (this.values.length) return Promise.resolve(this.values.shift());
    if (this.closed) return Promise.reject(new Error(`${this.label} activity stream closed`));
    if (this.signal?.aborted) return Promise.reject(this.signal.reason ?? new Error("aborted"));
    return new Promise((resolve, reject) => {
      const waiter = { resolve: null, reject: null };
      const detach = () => {
        clearTimeout(timer);
        this.signal?.removeEventListener("abort", onAbort);
        const index = this.waiters.indexOf(waiter);
        if (index >= 0) this.waiters.splice(index, 1);
      };
      const onAbort = () => {
        detach();
        reject(this.signal.reason ?? new Error("aborted"));
      };
      const timer = setTimeout(() => {
        detach();
        resolve(null);
      }, Math.max(0, ms));
      waiter.resolve = (value) => {
        detach();
        resolve(value);
      };
      waiter.reject = (error) => {
        detach();
        reject(error);
      };
      this.signal?.addEventListener("abort", onAbort, { once: true });
      this.waiters.push(waiter);
    });
  }

  /** Put back a value that was read but belongs to a later consumer. */
  unshift(value) {
    if (!this.closed) this.values.unshift(value);
  }

  close() {
    this.closed = true;
    for (const waiter of this.waiters.splice(0)) waiter.reject(new Error(`${this.label} activity stream closed`));
  }
}

/** A tool call parked by the SDK bridge, waiting for DSH to run it. */
function isToolParked(message) {
  return message.method === "tool/parked" && Boolean(message.params?.parkId);
}

/**
 * Claude can call several tools in one message, and the SDK reports them as
 * separate events emitted in the same tick. Take the siblings already queued so
 * the whole batch reaches DSH as one step, the way DSH's own loop handles a
 * multi-call assistant message.
 */
function drainParkedToolCalls(queue, turnId) {
  const parked = [];
  while (queue.values.length) {
    const next = queue.values[0];
    if (!isToolParked(next)) break;
    if (next.params?.turnId && next.params.turnId !== turnId) break;
    queue.values.shift();
    parked.push(next.params);
  }
  return parked;
}

/**
 * Where a harness tool result lives, across the two shapes DSH has shipped.
 *
 * DSH 0.1.x (`createToolResultMessage` in dsh-llm 0.1.5-rc.2) returns a `user`
 * message whose content holds one block of type `tool-result`, carrying the id on
 * the block. DSH 0.2.x returns a `tool` message that carries `toolCallId` on the
 * MESSAGE and has no `tool-result` block at all — its content is the plain result
 * content.
 *
 * Only the 0.2.x shape is real on the host this plugin now runs against, and
 * reading for 0.1.x alone silently found nothing: the parked call was never
 * resolved, the turn was abandoned, and the model was handed the SDK's canned
 * "the tool use was rejected" text while DSH had in fact executed the tool.
 *
 * @returns every result in `message`, as `{ toolCallId, content, isError }`.
 */
function toolResultsIn(message) {
  if (!message) return [];
  const results = [];
  // 0.2.x: the id is on the message.
  if (typeof message.toolCallId === "string") {
    results.push({
      toolCallId: message.toolCallId,
      content: message.content ?? [],
      isError: Boolean(message.isError),
    });
    return results;
  }
  // 0.1.x: the id is on a `tool-result` content block.
  for (const block of message.content ?? []) {
    if (block?.type !== "tool-result" || typeof block.toolCallId !== "string") continue;
    results.push({
      toolCallId: block.toolCallId,
      content: block.content ?? [],
      isError: Boolean(block.isError),
    });
  }
  return results;
}

/** Every tool-result id present, for tracing a mismatch against what was parked. */
function allToolResultIds(messages) {
  const ids = [];
  for (const message of messages ?? []) {
    for (const result of toolResultsIn(message)) ids.push(result.toolCallId);
  }
  return ids;
}

/** The results DSH executed for the calls this turn parked on. */
function toolResultsFor(messages, callIds) {
  const wanted = new Set(callIds);
  const results = [];
  for (const message of messages ?? []) {
    for (const result of toolResultsIn(message)) {
      if (!wanted.has(result.toolCallId)) continue;
      results.push(result);
    }
  }
  return results;
}

/** How long a resumed turn may stay silent before it is called a failure. */
function resumeTimeoutMs() {
  const configured = Number.parseInt((process.env.DSH_AGENT_BRIDGE_RESUME_TIMEOUT_MS ?? "").trim(), 10);
  return Number.isSafeInteger(configured) && configured > 0 ? configured : 300_000;
}

function withTimeout(promise, ms, message) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), ms);
    promise.then(
      value => { clearTimeout(timer); resolve(value); },
      error => { clearTimeout(timer); reject(error); },
    );
  });
}

function closeOpenBlocks(state) {
  const chunks = [];
  for (const block of state.blocks.values()) {
    if (block.closed) continue;
    block.closed = true;
    chunks.push({ type: "block-end", index: block.index, block: { type: block.type, text: block.text } });
  }
  return chunks;
}

function createStreamState(usage = newUsageLedger()) {
  return {
    nextIndex: 0,
    blocks: new Map(),
    completed: new Set(),
    activityItems: new Map(),
    startedActivities: new Set(),
    completedActivities: new Set(),
    /** The turn's usage ledger, shared by every DSH step of the turn. */
    usage,
  };
}

/** Usage bookkeeping for one Claude turn. */
function newUsageLedger(sessionId = null) {
  return { sessionId, completed: false, calls: new Map(), reported: new Map(), ended: new Set(), toolUses: new Map() };
}

/** How long a parked step waits for its model calls to finish reporting usage. */
const STEP_USAGE_SETTLE_MS = 2000;

/** How often a waiting step re-checks the ledger when nothing else arrives. */
const SETTLE_POLL_MS = 20;

/** The settle limit, overridable for tests with DSH_AGENT_BRIDGE_USAGE_SETTLE_MS. */
function usageSettleMs() {
  const raw = process.env.DSH_AGENT_BRIDGE_USAGE_SETTLE_MS;
  const ms = raw === undefined ? Number.NaN : Number(raw);
  return Number.isFinite(ms) && ms >= 0 ? ms : STEP_USAGE_SETTLE_MS;
}

/** True while a model call this step started has not finished reporting usage. */
function hasOpenCall(state) {
  return [...state.usage.calls.keys()].some(callId => !state.usage.reported.has(callId) && !state.usage.ended.has(callId));
}

/** The tool calls a model call declared, which the step waits for before reporting. */
function recordToolUses(state, callId, toolUseIds) {
  if (!callId || !Array.isArray(toolUseIds)) return;
  const ids = state.usage.toolUses.get(callId) ?? new Set();
  for (const id of toolUseIds) ids.add(id);
  state.usage.toolUses.set(callId, ids);
}

/**
 * True while this step still has something to wait for: a model call that has not
 * ended, or a declared tool call that has not parked yet. The second condition
 * covers a tool call parked after its model call has ended. If a declared tool
 * call is never parked, the wait still ends at the settle limit.
 */
function stepIsOpen(state, parked) {
  return hasOpenCall(state) || expectedParks(state) > parked;
}

function expectedParks(state) {
  let count = 0;
  for (const [callId, ids] of state.usage.toolUses) {
    if (!state.usage.reported.has(callId)) count += ids.size;
  }
  return count;
}

/** Record one usage activity against the ledger held in state.usage. */
function recordUsage(state, message) {
  const params = message.params ?? {};
  if (message.method === "usage/apiCall") recordCallUsage(state, params.callId, params.usage);
  else if (message.method === "usage/toolUses") recordToolUses(state, params.callId, params.toolUseIds);
  else if (message.method === "usage/apiCallEnded") state.usage.ended.add(params.callId);
}

/** Record the largest figure seen for each usage field of one model call. */
function recordCallUsage(state, callId, usage) {
  if (!callId) return;
  const call = state.usage.calls.get(callId) ?? {};
  for (const [field, value] of Object.entries(usage ?? {})) {
    if (Number.isSafeInteger(value)) call[field] = Math.max(call[field] ?? 0, value);
  }
  state.usage.calls.set(callId, call);
}

/**
 * The usage that arrived since the last step reported, over every model call of
 * the turn. A call's figures are cumulative, so a step reports only the growth
 * since its previous report. A figure that arrives late is counted exactly once,
 * in the step that receives it.
 */
function takeStepUsage(state) {
  const { calls, reported } = state.usage;
  let usage = null;
  for (const [callId, call] of calls) {
    const before = reported.get(callId) ?? {};
    const growth = {
      inputTokens: (call.input_tokens ?? 0) - (before.input_tokens ?? 0),
      outputTokens: (call.output_tokens ?? 0) - (before.output_tokens ?? 0),
      cacheReadTokens: (call.cache_read_input_tokens ?? 0) - (before.cache_read_input_tokens ?? 0),
      cacheWriteTokens: (call.cache_creation_input_tokens ?? 0) - (before.cache_creation_input_tokens ?? 0),
    };
    reported.set(callId, { ...call });
    if (!Object.values(growth).some(value => value !== 0)) continue;
    // DSH reads the prompt fields as the context size of one request, so they
    // must come from a single call: the largest prompt among this step's calls.
    // Summing them across calls inflated the context meter. Output still sums.
    const prompt = {
      inputTokens: call.input_tokens ?? 0,
      cacheReadTokens: call.cache_read_input_tokens ?? 0,
      cacheWriteTokens: call.cache_creation_input_tokens ?? 0,
    };
    const size = p => p.inputTokens + p.cacheReadTokens + p.cacheWriteTokens;
    const keep = usage && size(usage) >= size(prompt) ? usage : prompt;
    usage = {
      inputTokens: keep.inputTokens,
      outputTokens: (usage?.outputTokens ?? 0) + growth.outputTokens,
      cacheReadTokens: keep.cacheReadTokens,
      cacheWriteTokens: keep.cacheWriteTokens,
    };
  }
  return usage;
}

/**
 * Usage chunks for the DSH step that just ended. Each DSH step is one model call,
 * since the adapter parks at every tool request. The final step also carries the
 * turn's reasoning count, which Claude reports only at turn level. A turn in which
 * no call reported usage falls back to the turn's own total.
 */
function* stepUsageChunks(state, completedTurn) {
  const usage = takeStepUsage(state);
  if (!completedTurn) {
    if (usage) yield { type: "usage", usage };
    return;
  }
  if (state.usage.calls.size === 0) {
    if (completedTurn.usage) yield { type: "usage", usage: completedTurn.usage };
    return;
  }
  if (!usage) return;
  const reasoningTokens = completedTurn.usage?.reasoningTokens;
  yield { type: "usage", usage: reasoningTokens === undefined ? usage : { ...usage, reasoningTokens } };
}

function textDelta(state, id, type, delta) {
  if (!id || !delta) return [];
  let block = state.blocks.get(id);
  const chunks = [];
  if (!block) {
    block = { index: state.nextIndex++, type, text: "", closed: false };
    state.blocks.set(id, block);
    chunks.push({ type: "block-start", index: block.index, blockType: type });
  }
  if (block.closed) return chunks;
  block.text += delta;
  chunks.push({ type: type === "reasoning" ? "reasoning-delta" : "text-delta", index: block.index, text: delta });
  return chunks;
}

function completeTextItem(state, id, type, completeText) {
  const chunks = [];
  let block = state.blocks.get(id);
  if (!block) {
    block = { index: state.nextIndex++, type, text: "", closed: false };
    state.blocks.set(id, block);
    chunks.push({ type: "block-start", index: block.index, blockType: type });
  }
  if (completeText && completeText.startsWith(block.text) && completeText.length > block.text.length) {
    const delta = completeText.slice(block.text.length);
    block.text = completeText;
    chunks.push({ type: type === "reasoning" ? "reasoning-delta" : "text-delta", index: block.index, text: delta });
  }
  if (!block.closed) {
    block.closed = true;
    chunks.push({ type: "block-end", index: block.index, block: { type, text: block.text } });
  }
  return chunks;
}

/**
 * A session's event log, defensively.
 *
 * DSH's `Session` is an event-sourced object whose public log is `log`; there is
 * no `events` accessor. Reading the wrong name yielded `undefined` and threw
 * "events is not iterable" on the first turn, before anything reached the model.
 *
 * A missing or malformed log must not fail a turn: this only chooses the
 * permission defaults the turn runs under, and every caller already passes an
 * explicit sandbox and approval policy of its own.
 */
function sessionEventLog(session) {
  const log = session?.log;
  if (Array.isArray(log)) return log;
  return [];
}

/** The newest DSH log position: what a Claude transcript holds once it has been sent. */
function logHead(agent) {
  const seq = sessionEventLog(agent.session).at(-1)?.seq;
  return Number.isSafeInteger(seq) ? seq : -1;
}

/** True when DSH started a turn after log position `seq`, which replaces any turn parked there. */
function turnStartedSince(agent, seq) {
  if (!Number.isSafeInteger(seq)) return false;
  return sessionEventLog(agent.session).some(event => event.type === "turn/start" && event.seq > seq);
}

/** Message sources a person wrote or answered. Any other DSH message is context, not the user's input. */
const USER_SOURCE_KINDS = new Set(["user", "user-question-reply"]);

/**
 * The results a resumed Claude query receives, with the messages DSH received while the
 * tool ran appended to the last one. The user's own input is labelled as input; a DSH
 * message, such as a runtime-context snapshot, is labelled as context so Claude does not
 * read it as the user's words. Each message is delivered once, on that result.
 */
function resultsWithInput(results, inputs) {
  if (inputs.length === 0 || results.length === 0) return results;
  const typed = inputs.filter(message => USER_SOURCE_KINDS.has(message.source?.kind));
  const context = inputs.filter(message => !USER_SOURCE_KINDS.has(message.source?.kind));
  const notes = [
    typed.length > 0 && `Input received while this tool ran:\n\n${typed.map(message => messageText(message)).join("\n\n")}`,
    context.length > 0 && `Context received while this tool ran:\n\n${context.map(message => messageText(message)).join("\n\n")}`,
  ].filter(Boolean);
  const last = results.length - 1;
  return results.map((result, index) => {
    if (index !== last) return result;
    const content = Array.isArray(result.content) ? result.content : [{ type: "text", text: String(result.content ?? "") }];
    return {
      ...result,
      content: [...content, ...notes.map(text => ({ type: "text", text }))],
    };
  });
}

/** A reply the Claude route produced. DSH records the provider on the reply's message source. */
function producedByClaude(event) {
  return event.type === "assistant/message" && event.data?.message?.source?.provider === CLAUDE_PROVIDER;
}

/**
 * A surface event that replaces an existing span of DSH history, as compaction and
 * pruning do. System prompt rewrites are excluded: configure() already carries the
 * system prompt to Claude on every turn.
 */
function isReplaceEvent(event) {
  return event?.type !== "system/message"
    && typeof event?.surfaceOp === "object"
    && event.surfaceOp !== null
    && Number.isSafeInteger(event.surfaceOp.startSeq);
}

function permissionConfiguration(events) {
  let sandbox = "workspace-write";
  let approvalPolicy = "on-request";
  for (const event of events) {
    if (event.type === "sandbox/mode") sandbox = event.data.mode;
    if (event.type === "approval/policy") approvalPolicy = event.data.policy === "never" ? "never" : "on-request";
    if (event.type === "permission/preset") sandbox = event.data.preset;
  }
  return { sandbox, approvalPolicy };
}

function reasoningText(item) {
  return [...(item.summary ?? []), ...(item.content ?? [])].filter(Boolean).join("\n\n");
}

function summarizeValue(value) {
  if (value === undefined || value === null) return "";
  return firstLine(typeof value === "string" ? value : JSON.stringify(value));
}

function firstLine(value) {
  return String(value ?? "").split("\n")[0].slice(0, 240);
}

function humanize(value) {
  return String(value).replace(/([a-z])([A-Z])/g, "$1 $2").replace(/^./, letter => letter.toUpperCase());
}

function reasoningEffortName(value) {
  if (String(value) === "xhigh") return "Extra high";
  if (String(value) === ULTRACODE_EFFORT) return "Ultracode";
  return humanize(value);
}

/**
 * DSH splices its own context messages (workspace instructions, runtime
 * snapshots, the skill catalog) *after* the human's message inside the same
 * turn, so the instruction is not the last entry. Find the human's message and
 * let everything else travel as context.
 */
function latestUserIndex(messages) {
  let fallback = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role !== "user" || !messageText(message)) continue;
    if (message.source?.kind === "user") return index;
    if (fallback === -1) fallback = index;
  }
  return fallback;
}

/**
 * Render one message for the catch-up block.
 *
 * Unlike {@link messageText} this keeps the tool work, because a turn answered
 * by another model is mostly tool calls and their results — carrying only the
 * prose would hand Claude a summary of work it cannot see. Results are bounded:
 * this is context, not a transcript.
 *
 * Results go through {@link toolResultsIn} so both DSH shapes are covered. On
 * 0.2.x the whole message IS the result, so a message-shaped result is rendered
 * before its own (plain) content, which is not a `tool-result` block.
 */
function seedText(message) {
  const parts = [];
  const results = toolResultsIn(message);
  // A 0.2.x result message carries the result instead of ordinary content, so
  // render it here and skip the block loop below rather than printing its
  // content twice.
  if (typeof message?.toolCallId === "string") {
    for (const result of results) {
      const text = resultText(result);
      if (text) parts.push(`${result.isError ? "tool failed" : "tool result"}: ${bound(text, 1500)}`);
    }
    return parts.join("\n").trim();
  }
  for (const block of message?.content ?? []) {
    if (block?.type === "text" && block.text?.trim()) parts.push(block.text.trim());
    if (block?.type === "tool-call") parts.push(`called ${block.name}(${bound(block.arguments, 400)})`);
  }
  for (const result of results) {
    const text = resultText(result);
    if (text) parts.push(`${result.isError ? "tool failed" : "tool result"}: ${bound(text, 1500)}`);
  }
  return parts.join("\n").trim();
}

/** The plain text of one tool result, whatever shape its content arrived in. */
function resultText(result) {
  const content = result?.content;
  if (typeof content === "string") return content.trim();
  return (Array.isArray(content) ? content : [])
    .filter(inner => inner?.type === "text")
    .map(inner => inner.text)
    .join("\n")
    .trim();
}

function bound(value, limit) {
  const text = String(value ?? "");
  return text.length > limit ? `${text.slice(0, limit)}…` : text;
}

function messageText(message) {
  return (message?.content ?? [])
    .filter(block => block.type === "text")
    .map(block => block.text)
    .join("\n")
    .trim();
}

function auxiliaryInput(messages) {
  return messages.map((message) => {
    const text = (message?.content ?? [])
      .filter(block => block.type === "text")
      .map(block => block.text)
      .join("\n")
      .trim();
    return text ? `${message.role ?? "user"}: ${text}` : "";
  }).filter(Boolean).join("\n\n");
}

function projectAuxiliaryActivity(message, state) {
  const params = message.params ?? {};
  if (message.method === "item/reasoning/summaryTextDelta" || message.method === "item/reasoning/textDelta") {
    return textDelta(state, params.itemId, "reasoning", params.delta ?? "");
  }
  if (message.method === "item/agentMessage/delta") return textDelta(state, params.itemId, "text", params.delta ?? "");
  if (message.method === "item/completed") return completeAuxiliaryItem(state, params.item);
  return [];
}

function completeAuxiliaryItem(state, item) {
  if (!item?.id || state.completed.has(item.id)) return [];
  state.completed.add(item.id);
  if (item.type === "reasoning") return completeTextItem(state, item.id, "reasoning", reasoningText(item));
  if (item.type === "agentMessage") return completeTextItem(state, item.id, "text", item.text ?? "");
  return [];
}

function compact(value) {
  return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined && item !== null));
}

function runtimeModels(runtime) {
  return typeof runtime.listModels === "function" ? runtime.listModels() : [...runtime.models];
}

function hasRuntimeSession(runtime, sessionId) {
  return typeof runtime.hasSession === "function"
    ? runtime.hasSession(sessionId)
    : runtime.sessions.has(sessionId);
}

function patchRuntimeSession(runtime, sessionId, patch) {
  if (typeof runtime.patchSession === "function") return runtime.patchSession(sessionId, patch);
  const session = runtime.sessions.get(sessionId);
  if (session) Object.assign(session, patch);
  return Boolean(session);
}

function subscribeRuntimeActivity(runtime, listener) {
  if (typeof runtime.subscribeActivity === "function") return runtime.subscribeActivity(listener);
  runtime.on("activity", listener);
  return () => runtime.off("activity", listener);
}

