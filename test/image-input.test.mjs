import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { ClaudeDshAdapter } from "../claude-adapter.js";
import { ClaudeCliClient } from "../cli-client.mjs";
import { ClaudeSdkClient } from "../sdk-client.mjs";
import { ClaudeSessionRuntime } from "../session-runtime.mjs";

const PNG = new Uint8Array([137, 80, 78, 71, 1, 2, 3, 4]);
const IMAGE = { type: "image", attachment: { attachmentId: "att-1", name: "photo.png", mediaType: "image/png", bytes: PNG.byteLength, width: 4, height: 4 } };
const LABEL = "[image: photo.png, 4x4 image/png]";
const NOT_SENT = "[image: photo.png, 4x4 image/png; not attached to this turn]";
const SILENT = { warn() {} };
const IMAGE_BYTE_LIMIT = 3_750_000;

test("an image-only message after an answered turn is the request, and its image reaches the runtime", async () => {
  const runtime = new FakeRuntime();
  const store = attachments();
  const adapter = adapterWith(runtime, { resolveAttachments: () => store });
  const u1 = user([text("first question")]);
  const a1 = { role: "assistant", content: [text("answer one")] };
  const u2 = user([IMAGE]);
  await collect(adapter.stream(request([u1])));
  await collect(adapter.stream(request([u1, a1, u2])));

  assert.equal(runtime.sent.length, 2, "the image message starts its own turn");
  const sent = runtime.sent[1].message;
  assert.equal(sent.text.split("\n\n").at(-1), LABEL, "the image message is the request");
  assert.doesNotMatch(sent.text, /first question/, "the earlier question is not asked again");
  assert.deepEqual(sent.images, [{ mediaType: "image/png", data: Buffer.from(PNG).toString("base64") }]);
  assert.deepEqual(store.calls[0].target, { width: 4, height: 4, maxBytes: IMAGE_BYTE_LIMIT });
});

test("an image-only first message is sent with its image instead of being refused", async () => {
  const runtime = new FakeRuntime();
  const adapter = adapterWith(runtime, { resolveAttachments: () => attachments() });
  await collect(adapter.stream(request([user([IMAGE])])));

  assert.equal(runtime.sent.length, 1);
  assert.equal(runtime.sent[0].message.text, LABEL);
  assert.equal(runtime.sent[0].message.images.length, 1);
});

test("text and an image in one message both reach the runtime", async () => {
  const runtime = new FakeRuntime();
  const adapter = adapterWith(runtime, { resolveAttachments: () => attachments() });
  await collect(adapter.stream(request([user([text("look at this"), IMAGE])])));

  assert.equal(runtime.sent[0].message.text, `look at this\n${LABEL}`);
  assert.equal(runtime.sent[0].message.images.length, 1);
});

test("an image that cannot be read is named as not attached, and the turn still runs", async () => {
  const runtime = new FakeRuntime();
  const warnings = [];
  const store = { async readImageRequest() { throw new Error("blob missing"); } };
  const adapter = adapterWith(runtime, { resolveAttachments: () => store, logger: { warn: message => warnings.push(message) } });
  await collect(adapter.stream(request([user([IMAGE])])));

  assert.equal(runtime.sent[0].message.text, NOT_SENT);
  assert.deepEqual(runtime.sent[0].message.images, []);
  assert.match(warnings[0], /blob missing/, "the failure is logged for the operator");
});

test("without an attachment service the image is named as not attached", async () => {
  const runtime = new FakeRuntime();
  const adapter = adapterWith(runtime);
  await collect(adapter.stream(request([user([IMAGE])])));

  assert.equal(runtime.sent[0].message.text, NOT_SENT);
  assert.deepEqual(runtime.sent[0].message.images, []);
});

test("an image over the byte limit is not sent", async () => {
  const runtime = new FakeRuntime();
  const store = attachments(new Uint8Array(IMAGE_BYTE_LIMIT + 1));
  const adapter = adapterWith(runtime, { resolveAttachments: () => store });
  await collect(adapter.stream(request([user([IMAGE])])));

  assert.equal(runtime.sent[0].message.text, NOT_SENT);
  assert.deepEqual(runtime.sent[0].message.images, []);
});

test("an offloaded image is named and not read", async () => {
  const runtime = new FakeRuntime();
  const store = attachments();
  const adapter = adapterWith(runtime, { resolveAttachments: () => store });
  await collect(adapter.stream(request([user([{ ...IMAGE, offloaded: true }])])));

  assert.equal(store.calls.length, 0);
  assert.equal(runtime.sent[0].message.text, NOT_SENT);
});

test("the SDK client sends a turn with images as one user message, and a text turn as a string", async () => {
  const prompts = [];
  const sdk = {
    query(params) {
      prompts.push(params.prompt);
      return queryObject(async function* () {
        yield { type: "result", session_id: params.options.sessionId, uuid: "r1", subtype: "success", is_error: false, result: "ok" };
      });
    },
  };
  const client = new ClaudeSdkClient({ sdk });
  await client.start();
  const session = await client.createSession({ sessionId: "11111111-1111-4111-8111-111111111111", cwd: "/workspace/images" });
  await client.sendMessage(session.id, { text: "look", images: [{ mediaType: "image/png", data: "AAEC" }] });

  const messages = [];
  for await (const message of prompts[0]) messages.push(message);
  assert.deepEqual(messages, [{
    type: "user",
    session_id: "",
    message: { role: "user", content: [
      { type: "text", text: "look" },
      { type: "image", source: { type: "base64", media_type: "image/png", data: "AAEC" } },
    ] },
    parent_tool_use_id: null,
  }]);

  await client.sendMessage(session.id, { text: "plain" });
  assert.equal(prompts[1], "plain", "a turn without images keeps the string prompt");
});

test("the runtime forwards a turn's images to the client", async () => {
  const sent = [];
  const client = new EventEmitter();
  Object.assign(client, {
    async start() {},
    async listModels() { return [{ id: "sonnet", isDefault: true }]; },
    async createSession(config) { return { id: "c1", cwd: config.cwd ?? "/w", turns: [] }; },
    async sendMessage(sessionId, message) { sent.push(message); return { id: "t1", status: "inProgress", items: [] }; },
  });
  const runtime = new ClaudeSessionRuntime({ client, cwd: "/w" });
  await runtime.initialize();
  const session = await runtime.createSession({ model: "sonnet" });
  const images = [{ mediaType: "image/png", data: "AAEC" }];
  await runtime.sendMessage(session.id, { text: "look", images });

  assert.deepEqual(sent[0].images, images);
});

test("the Claude CLI backend refuses images instead of dropping them", async () => {
  const client = new ClaudeCliClient();
  await client.createSession({ sessionId: "44444444-4444-4444-8444-444444444444" });
  await assert.rejects(() => client.sendMessage("44444444-4444-4444-8444-444444444444", {
    text: "look",
    images: [{ mediaType: "image/png", data: "AAEC" }],
  }), /cannot send images/);
});

test("images beyond the per-message byte budget are named as not attached", async () => {
  const runtime = new FakeRuntime();
  const adapter = adapterWith(runtime, { resolveAttachments: () => attachments(new Uint8Array(3_000_000)) });
  await collect(adapter.stream(request([user(Array.from({ length: 6 }, () => ({ ...IMAGE })))])));

  assert.equal(runtime.sent[0].message.images.length, 5, "five images of 3,000,000 bytes fit the 15,000,000-byte budget");
  assert.equal(runtime.sent[0].message.text, [...Array(5).fill(LABEL), NOT_SENT].join("\n"), "the sixth image is named, not sent");
});

test("an image of exactly the byte limit is sent", async () => {
  const runtime = new FakeRuntime();
  const adapter = adapterWith(runtime, { resolveAttachments: () => attachments(new Uint8Array(IMAGE_BYTE_LIMIT)) });
  await collect(adapter.stream(request([user([IMAGE])])));

  assert.equal(runtime.sent[0].message.images.length, 1, "the limit is inclusive");
});

test("a stored image is sent as the smaller version DSH returns, and the label describes that version", async () => {
  const runtime = new FakeRuntime();
  const reencoded = { data: new Uint8Array([7, 8, 9]), mediaType: "image/jpeg", width: 2000, height: 1500 };
  const calls = [];
  const store = {
    async readImageRequest(ref, target) {
      calls.push(target);
      return { variantId: "v2", attachment: ref, bytes: reencoded.data.byteLength, ...reencoded };
    },
  };
  const stored = { ...IMAGE, attachment: { ...IMAGE.attachment, width: 4000, height: 3000 } };
  const adapter = adapterWith(runtime, { resolveAttachments: () => store });
  await collect(adapter.stream(request([user([stored])])));

  assert.equal(calls[0].maxBytes, IMAGE_BYTE_LIMIT, "DSH is asked for a version within the limit");
  assert.deepEqual(runtime.sent[0].message.images, [{ mediaType: "image/jpeg", data: Buffer.from(reencoded.data).toString("base64") }]);
  assert.equal(runtime.sent[0].message.text, "[image: photo.png, 2000x1500 image/jpeg]", "the label names the version sent, not the stored one");
});

test("a stop during the image read ends the turn before anything is sent", async () => {
  const runtime = new FakeRuntime();
  const store = {
    readImageRequest(ref, target, signal) {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => resolve({ attachment: ref, data: PNG, mediaType: "image/png", width: 4, height: 4 }), 1_000);
        signal.addEventListener("abort", () => { clearTimeout(timer); reject(new DOMException("stopped", "AbortError")); });
      });
    },
  };
  const adapter = adapterWith(runtime, { resolveAttachments: () => store });
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 10);
  const chunks = await collect(adapter.stream({ ...request([user([IMAGE])]), signal: controller.signal }));

  assert.equal(chunks.at(-1).reason.kind, "aborted");
  assert.equal(runtime.sent.length, 0, "a stopped turn is not sent to Claude");
});

test("a session title sees an image-only message as a labelled image, not as missing input", async () => {
  const runtime = new FakeRuntime();
  const adapter = adapterWith(runtime);
  await collect(adapter.stream({ provider: "claude", model: "sonnet", purpose: "session-title", sessionId: "dsh-1", messages: [user([IMAGE])] }));

  assert.match(runtime.sent[0].message.text, /\[image: photo\.png, 4x4 image\/png; not attached to this turn\]/);
});

test("the DSH plugin gives the adapter the attachment service it injects", async (t) => {
  const store = attachments();
  const runtime = new FakeRuntime();
  const { adapter } = await activatePlugin(t, runtime, store);
  await collect(adapter.stream(request([user([IMAGE])])));

  assert.equal(runtime.sent[0].message.images.length, 1);
  assert.equal(store.calls.length, 1);
});

test("the DSH plugin starts without an attachment service, and names the image", async (t) => {
  const runtime = new FakeRuntime();
  const { adapter, result } = await activatePlugin(t, runtime, null);
  assert.ok(result.capabilities["agent-bridge.dsh.v1"], "the plugin still activates");
  await collect(adapter.stream(request([user([IMAGE])])));

  assert.equal(runtime.sent[0].message.text, NOT_SENT);
  assert.deepEqual(runtime.sent[0].message.images, []);
});

/** Activate the real DSH plugin with a fake host context. The attachment service is handed over only when given. */
async function activatePlugin(t, runtime, store) {
  const { createDshClaudePlugin } = await import("../dsh-plugin.js");
  const cwd = await mkdtemp(join(tmpdir(), "image-input-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  let adapter = null;
  const agent = { id: "dsh-1", session: { header: { agentPreset: "standard", cwd }, log: [], append(type, data) { this.log.push({ type, data }); } } };
  const ctx = {
    logger: { warn() {}, error() {}, info() {}, debug() {} },
    llm: { registerAdapter(_providers, registered) { adapter = registered; return () => {}; } },
    agents: { list: () => [agent], get: () => agent },
    on: () => () => {},
    get: () => undefined,
    inject(names, callback) {
      if (names.includes("attachments") && store) callback({ get: name => (name === "attachments" ? store : undefined) });
      return { dispose() {} };
    },
  };
  const plugin = createDshClaudePlugin(ctx, { telemetry: "ignore", linkPath: join(cwd, "links.json") });
  const result = await plugin.activate({ capabilities: { require: () => runtime }, defer: () => {} });
  return { adapter, result };
}

test("an image sent while a tool runs is named in the tool result it rides on", async () => {
  const runtime = new ParkingRuntime();
  const adapter = adapterWith(runtime);
  const go = user([text("list the files")]);
  const first = await collect(adapter.stream({ ...request([go]), tools: [BASH] }));
  const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call").block;
  const result = { role: "tool", source: { kind: "tool", callId: call.id }, toolCallId: call.id, content: [text("/workspace/dsh")], isError: false };
  await collect(adapter.stream({ ...request([go, result, user([IMAGE])]), tools: [BASH] }));

  const delivered = runtime.resolved[0].result.content.map(block => block.text).join("\n");
  assert.match(delivered, /Input received while this tool ran/);
  assert.ok(delivered.includes(NOT_SENT), "the steered image is named in the tool result");
});

const BASH = { name: "bash", description: "Run a command.", parameters: { type: "object", properties: { command: { type: "string" } } } };

function adapterWith(runtime, options = {}) {
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve(), logger: SILENT, ...options });
  adapter.attachAgent(fakeAgent());
  return adapter;
}

/** A store that returns the given bytes for any image, and records each read. */
function attachments(bytes = PNG) {
  const calls = [];
  return {
    calls,
    async readImageRequest(ref, target) {
      calls.push({ ref, target });
      return { variantId: "v1", attachment: ref, data: bytes, mediaType: ref.mediaType, bytes: bytes.byteLength, width: ref.width, height: ref.height };
    },
  };
}

const request = messages => ({ provider: "claude", model: "sonnet", reasoningEffort: "high", sessionId: "dsh-1", messages });
const user = content => ({ role: "user", source: { kind: "user" }, content });
const text = value => ({ type: "text", text: value });

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
  whenReady() { return Promise.resolve(); }
  subscribeRequest() { return () => {}; }
}

/** Parks every turn that carries DSH tools until its tool result arrives, as the real runtime does. */
class ParkingRuntime extends FakeRuntime {
  constructor() {
    super();
    this.parked = null;
    this.resolved = [];
  }
  async sendMessage(sessionId, message) {
    this.sent.push({ sessionId, message });
    const turnId = `turn-${this.sent.length}`;
    queueMicrotask(() => {
      if (message.dshTools?.length) {
        this.parked = { sessionId, turnId };
        this.emit("activity", { method: "tool/parked", params: { sessionId, turnId, parkId: "park-1", name: "bash", arguments: { command: "pwd" } } });
        return;
      }
      this.emit("activity", { method: "turn/completed", params: { sessionId, turn: { id: turnId, status: "completed", error: null, items: [], usage: { inputTokens: 1, outputTokens: 1, cacheReadTokens: 0, cacheWriteTokens: 0, reasoningTokens: 0 } } } });
    });
    return { id: turnId, status: "inProgress", items: [] };
  }
  resolveToolCall(toolUseId, result) {
    this.resolved.push({ toolUseId, result });
    const parked = this.parked;
    this.parked = null;
    if (!parked) return false;
    queueMicrotask(() => this.emit("activity", { method: "turn/completed", params: { sessionId: parked.sessionId, turn: { id: parked.turnId, status: "completed", error: null, items: [], usage: { inputTokens: 1, outputTokens: 1, cacheReadTokens: 0, cacheWriteTokens: 0, reasoningTokens: 0 } } } }));
    return true;
  }
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

function queryObject(factory) {
  const iterator = factory();
  return {
    async next() { return iterator.next(); },
    async return() { return iterator.return?.() ?? { done: true }; },
    async throw(error) { return iterator.throw?.(error) ?? Promise.reject(error); },
    [Symbol.asyncIterator]() { return this; },
    async interrupt() { return { still_queued: [] }; },
    close() {},
  };
}

async function collect(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}
