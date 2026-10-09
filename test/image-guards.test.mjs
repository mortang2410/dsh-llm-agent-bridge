import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";

import { ClaudeDshAdapter } from "../claude-adapter.js";

const SILENT = { warn() {} };

test("a stop during an image read ends the turn before anything is sent", async () => {
  const runtime = new FakeRuntime();
  const store = {
    readImageRequest(ref, target, signal) {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => resolve(imageResult(new Uint8Array([1, 2]), ref)), 200);
        signal?.addEventListener("abort", () => { clearTimeout(timer); reject(new Error("aborted")); });
      });
    },
  };
  const adapter = adapterWith(runtime, { resolveAttachments: () => store });
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 20);

  const chunks = await collect(adapter.stream(request([user([image("photo.png")])], controller.signal)));

  assert.equal(runtime.sent.length, 0, "the stopped turn is not sent to Claude");
  assert.deepEqual(chunks.at(-1).reason, { kind: "aborted", failure: { message: "Claude turn cancelled", code: "ABORTED" } });
});

test("images past the per-message budget are omitted by name, and the rest are sent", async () => {
  const runtime = new FakeRuntime();
  const bytes = new Uint8Array(4_000_000);
  const store = { readImageRequest: async ref => imageResult(bytes, ref) };
  const adapter = adapterWith(runtime, { resolveAttachments: () => store });

  await collect(adapter.stream(request([user([image("a.png"), image("b.png"), image("c.png")])])));

  const sent = runtime.sent[0].message;
  assert.equal(sent.images.length, 2, "two images fit the budget");
  assert.match(sent.text, /\[image: a\.png\]/);
  assert.match(sent.text, /\[image: b\.png\]/);
  assert.match(sent.text, /\[image omitted to fit request image limits: c\.png\]/, "the image past the budget is named as omitted");
});

function adapterWith(runtime, options = {}) {
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve(), logger: SILENT, ...options });
  adapter.attachAgent(fakeAgent());
  return adapter;
}

function imageResult(data, ref) {
  return { variantId: "v1", attachment: ref, data, mediaType: ref.mediaType, bytes: data.byteLength, width: ref.width, height: ref.height };
}

const image = name => ({ type: "image", attachment: { attachmentId: name, name, mediaType: "image/png", bytes: 1, width: 4, height: 4 } });
const user = content => ({ role: "user", source: { kind: "user" }, content });
const request = (messages, signal) => ({ provider: "claude", model: "sonnet", reasoningEffort: "high", sessionId: "dsh-1", messages, signal });

class FakeRuntime extends EventEmitter {
  constructor() {
    super();
    this.sent = [];
    this.created = 0;
    this.sessions = new Map();
    this.models = [{ id: "sonnet", displayName: "Sonnet 5", isDefault: true, supportedReasoningEfforts: [{ reasoningEffort: "low" }, { reasoningEffort: "high" }] }];
  }
  async createSession(config) {
    const session = { id: `claude-${++this.created}`, turns: [], ...config };
    this.sessions.set(session.id, session);
    return session;
  }
  async resumeSession(id, config) {
    const session = { id, turns: [], ...config };
    this.sessions.set(id, session);
    return session;
  }
  async sendMessage(sessionId, message) {
    this.sent.push({ sessionId, message });
    const turnId = `turn-${this.sent.length}`;
    queueMicrotask(() => {
      this.emit("activity", { method: "item/agentMessage/delta", params: { sessionId, turnId, itemId: "a1", delta: "ok" } });
      this.emit("activity", { method: "turn/completed", params: { sessionId, turn: { id: turnId, status: "completed", error: null, items: [], usage: { inputTokens: 1, outputTokens: 1, cacheReadTokens: 0, cacheWriteTokens: 0, reasoningTokens: 0 } } } });
    });
    return { id: turnId, status: "inProgress", items: [] };
  }
  async interruptTurn() {}
  async releaseSession() {}
  contextWindowFor() { return 200000; }
  resolveToolCall() { return false; }
  rejectAllToolCalls() {}
}

function fakeAgent() {
  return {
    id: "dsh-1",
    ctx: {},
    session: {
      header: { agentPreset: "standard", cwd: "/workspace/dsh" },
      log: [],
      append(type, data, { surfaceOp } = {}) {
        this.log.push({ seq: this.log.length + 1, type, data, ...(surfaceOp === undefined ? {} : { surfaceOp }) });
      },
    },
  };
}

async function collect(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}
