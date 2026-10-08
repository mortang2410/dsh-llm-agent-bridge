import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import test from "node:test";

import { PluginHost } from "../internal/plugin-sdk.mjs";
import { createClaudeExecutionPlugin } from "../plugin.mjs";

test("Claude plugin exposes operation capabilities and closes its backend", async () => {
  const client = new FakeClaudeClient();
  const host = new PluginHost();
  await host.activate([createClaudeExecutionPlugin({ client, cwd: "/workspace" })]);
  const execution = host.capabilities.require("agent-bridge.execution.v1", "^1.0.0");

  await execution.whenReady();
  assert.deepEqual(execution.listModels().map((model) => model.id), ["claude-test"]);
  assert.equal("runtime" in execution, false);
  assert.equal("client" in execution, false);
  const requests = [];
  const stop = execution.subscribeRequest((request) => requests.push(request.id));
  client.emit("request", { id: "request-1", method: "test", params: {} });
  assert.deepEqual(requests, ["request-1"]);
  stop();
  client.emit("request", { id: "request-2", method: "test", params: {} });
  assert.deepEqual(requests, ["request-1"]);

  await host.dispose();
  assert.equal(client.closed, true);
});

class FakeClaudeClient extends EventEmitter {
  constructor() {
    super();
    this.closed = false;
  }

  async start() {}
  async listModels() { return [{ id: "claude-test", isDefault: true }]; }
  async close() { this.closed = true; }
}

test("every service the plugin dereferences at runtime is declared in inject", async () => {
  const { inject } = await import("../host-plugin.js");
  const sources = await Promise.all(["../claude-tools.js", "../dsh-plugin.js", "../claude-adapter.js"]
    .map(file => readFile(new URL(file, import.meta.url), "utf8")));

  const used = new Set();
  for (const source of sources) {
    for (const [, name] of source.matchAll(/\bctx\.([a-zA-Z][a-zA-Z0-9]*)\b/g)) used.add(name);
  }
  // cordis's own context API, not services: on/effect/inject/plugin, plus the
  // logger every context carries and llm, declared under its service name.
  for (const method of ["on", "effect", "inject", "plugin", "logger", "llm"]) used.delete(method);

  for (const name of used) {
    assert.ok(inject.includes(name), `ctx.${name} is used but not declared in inject; cordis throws only when the line runs`);
  }
});

test("inject declares nothing the plugin does not use, because a spare name parks it", async () => {
  const { inject } = await import("../host-plugin.js");
  const sources = await Promise.all(["../claude-tools.js", "../dsh-plugin.js", "../claude-adapter.js"]
    .map(file => readFile(new URL(file, import.meta.url), "utf8")));

  const used = new Set();
  for (const source of sources) {
    for (const [, name] of source.matchAll(/\bctx\.([a-zA-Z][a-zA-Z0-9]*)\b/g)) used.add(name);
  }
  for (const method of ["on", "effect", "inject", "plugin", "logger"]) used.delete(method);

  // Cordis treats every declared name as REQUIRED: the plugin stays inactive
  // until all of them exist, so `apply()` never runs. A name this plugin never
  // reads can therefore only stop it from loading — `webServer` did exactly
  // that on any profile without a web bundle. `llm` is read as `ctx.llm` and is
  // legitimately declared; `connection` is reached through the optional
  // `ctx.inject([...])` seam and must NOT be in this list.
  const spare = inject.filter(name => !used.has(name));
  assert.deepEqual(
    spare, [],
    `inject declares ${spare.join(", ")}, which this plugin never reads; cordis would park the plugin until each one exists`,
  );
});

test("the fallback client forwards every method the runtime calls on it", async () => {
  const { DELEGATED_CLIENT_METHODS } = await import("../plugin.mjs");
  const runtimeSource = await readFile(new URL("../session-runtime.mjs", import.meta.url), "utf8");

  const used = new Set();
  for (const [, name] of runtimeSource.matchAll(/this\.client\.([a-zA-Z][a-zA-Z0-9]*)/g)) used.add(name);
  // `start` is the wrapper's own decision point, and `on` comes from EventEmitter.
  used.delete("start");
  used.delete("on");

  for (const name of used) {
    assert.ok(
      DELEGATED_CLIENT_METHODS.includes(name),
      `the runtime calls client.${name}, but the fallback wrapper does not forward it — it would resolve against the inherited CLI stub instead of the active client`,
    );
  }
});

test("a parked call resolves through the fallback wrapper, not the stub it inherits", async () => {
  const { FallbackClaudeClient } = await import("../plugin.mjs");
  const { ClaudeSdkClient } = await import("../sdk-client.mjs");

  // The shape that broke live: backend "auto", so the runtime holds the wrapper
  // while the parked call lives on the SDK client behind it.
  const primary = new ClaudeSdkClient({ sdk: {} });
  const fallback = new (await import("../cli-client.mjs")).ClaudeCliClient();
  const client = new FallbackClaudeClient({ primary, fallback });
  client.active = primary;

  const parked = [];
  primary.on("activity", message => { if (message.method === "tool/parked") parked.push(message.params); });
  const pending = primary.parkToolCall("claude-1", "turn-1", "bash", { command: "pwd" });

  assert.equal(
    client.resolveToolCall(parked[0].parkId, { content: [{ type: "text", text: "ok" }] }),
    true,
    "the wrapper must reach the active client; the inherited CLI stub always returns false",
  );
  assert.deepEqual((await pending).content, [{ type: "text", text: "ok" }]);
});

test("the DSH half activates against a cordis-shaped context and serves the channel", async () => {
  const { createDshClaudePlugin, AGENT_BRIDGE_CHANNEL } = await import("../dsh-plugin.js");

  const handlers = new Map();
  const disposed = [];
  const connection = {
    rpc: {
      handle(channel, handler, options) {
        handlers.set(channel, { handler, options });
        return () => handlers.delete(channel);
      },
    },
  };
  // Shaped like cordis: inject() returns a fiber, not a disposer — the mistake
  // that took a release down, because the host asserts cleanups are callable.
  const ctx = {
    logger: { warn() {}, error() {}, info() {} },
    llm: { registerAdapter: () => () => {} },
    agents: { list: () => [], get: () => null },
    on: () => () => {},
    get: name => (name === "connection" ? connection : undefined),
    inject(_names, callback) {
      callback({ get: ctx.get, effect: (body, _label) => { disposed.push(body()); } });
      return { dispose() {} };
    },
  };

  const host = new PluginHost();
  const client = new FakeClaudeClient();
  await host.activate([
    createClaudeExecutionPlugin({ client, cwd: "/workspace" }),
    createDshClaudePlugin(ctx, {}),
  ]);

  const served = handlers.get(AGENT_BRIDGE_CHANNEL);
  assert.ok(served, "the browser half has no other way to read plan usage");
  assert.equal(served.options.authority, "loopback", "the channel answers the local UI only");

  assert.deepEqual(await served.handler("usage", {}), { ok: true, value: { supported: false } });
  const unknown = await served.handler("nope", {});
  assert.equal(unknown.ok, false);
  assert.match(unknown.error.message, /unknown .* endpoint/);

  await host.dispose();
});

test("activation completes while DSH's exporter is live, without shutting it down or disposing its fiber", async () => {
  const { createDshClaudePlugin } = await import("../dsh-plugin.js");

  // dsh 0.1.5 mounts session-telemetry-otel in FEEDBACK_ONLY by default. The
  // guard used to act on it from inside activate(), and disposing that row's
  // fiber while the cordis loader was still applying entries deadlocked the
  // boot: the loader waited on itself, the web app never printed its URL.
  const record = { shutdown: 0, disposed: 0 };
  const sessionTelemetry = {
    sharing: "feedback-only",
    provider: {},
    async shutdown() { record.shutdown += 1; },
    ctx: { fiber: { async dispose() { record.disposed += 1; } } },
  };
  const warnings = [];
  const ctx = {
    logger: { warn: message => warnings.push(message), error() {}, info() {} },
    llm: { registerAdapter: () => () => {} },
    agents: { list: () => [], get: () => null },
    on: () => () => {},
    get: name => (name === "sessionTelemetry" ? sessionTelemetry : undefined),
    inject: () => ({ dispose() {} }),
  };

  const host = new PluginHost();
  const activation = host.activate([
    createClaudeExecutionPlugin({ client: new FakeClaudeClient(), cwd: "/workspace" }),
    createDshClaudePlugin(ctx, {}),
  ]);
  const outcome = await Promise.race([
    activation.then(() => "activated"),
    new Promise(resolve => setTimeout(() => resolve("hung"), 2000)),
  ]);
  assert.equal(outcome, "activated", "activation must not block on the telemetry guard");

  assert.equal(record.disposed, 0, "another plugin's fiber is never disposed during activation");
  assert.equal(record.shutdown, 0, "activation only reads the exporter; enforcement belongs to the turn");
  assert.match(warnings.find(line => /telemetry is feedback-only/.test(line)) ?? "", /first Claude turn will shut its exporter down/);

  await host.dispose();
});

test("with telemetry: ignore, activation does not even read the exporter", async () => {
  const { createDshClaudePlugin } = await import("../dsh-plugin.js");

  const reads = [];
  const ctx = {
    logger: { warn() { throw new Error("nothing to warn about in ignore mode"); }, error() {}, info() {} },
    llm: { registerAdapter: () => () => {} },
    agents: { list: () => [], get: () => null },
    on: () => () => {},
    get: name => { reads.push(name); return undefined; },
    inject: () => ({ dispose() {} }),
  };

  const host = new PluginHost();
  await host.activate([
    createClaudeExecutionPlugin({ client: new FakeClaudeClient(), cwd: "/workspace" }),
    createDshClaudePlugin(ctx, { telemetry: "ignore" }),
  ]);
  assert.equal(reads.includes("sessionTelemetry"), false);
  await host.dispose();
});

test("a configured gateway endpoint becomes the Claude child's ANTHROPIC_BASE_URL", async (context) => {
  const { createClaudeExecutionPlugin } = await import("../plugin.mjs");
  const { mkdtemp, readFile, rm, writeFile } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");

  const directory = await mkdtemp(join(tmpdir(), "agent-bridge-env-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const envPath = join(directory, "env.json");
  const scriptPath = join(directory, "fake-claude.mjs");
  await writeFile(scriptPath, `
import { writeFileSync } from "node:fs";
writeFileSync(process.argv[2], JSON.stringify({
  baseUrl: process.env.ANTHROPIC_BASE_URL ?? null,
  apiKey: process.env.ANTHROPIC_API_KEY ?? null,
  authToken: process.env.ANTHROPIC_AUTH_TOKEN ?? null,
  nonessential: process.env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC ?? null,
  hasPath: typeof process.env.PATH === "string" && process.env.PATH.length > 0,
}));
console.log(JSON.stringify({ type: "result", result: "ok" }));
`);

  // No `client` override: the plugin must build it from the config, which is the
  // wiring under test. The CLI backend makes the resulting child environment
  // observable without spending an inference request.
  const host = new PluginHost();
  await host.activate([createClaudeExecutionPlugin({
    backend: "cli",
    command: process.execPath,
    args: [scriptPath, envPath],
    bifrostBaseUrl: "http://127.0.0.1:8090/anthropic",
    cwd: directory,
  })]);
  const capability = host.capabilities.require("agent-bridge.execution.v1", "^1.0.0");
  await capability.whenReady();

  const session = await capability.createSession({ sessionId: "s-gateway", cwd: directory });
  const completed = new Promise(resolve => {
    capability.subscribeActivity(message => { if (message.method === "turn/completed") resolve(message); });
  });
  await capability.sendMessage(session.id, { text: "hi" });
  await completed;

  const seen = JSON.parse(await readFile(envPath, "utf8"));
  assert.equal(seen.baseUrl, "http://127.0.0.1:8090/anthropic");
  assert.equal(seen.hasPath, true, "ordinary environment variables survive");
  // Pass-through, not stripping: whatever the operator's environment held for
  // these two reaches the child unchanged, and the bridge invents neither.
  assert.equal(seen.apiKey, process.env.ANTHROPIC_API_KEY ?? null, "no API key is invented");
  assert.equal(seen.authToken, process.env.ANTHROPIC_AUTH_TOKEN ?? null, "no auth token is invented");
  assert.equal(
    seen.nonessential, "1",
    "Claude Code's own title request carries its own effort, so a routed run must not emit it",
  );
  await host.dispose();
});

test("no gateway endpoint means no environment override at all", async (context) => {
  const { createClaudeExecutionPlugin } = await import("../plugin.mjs");
  const { mkdtemp, readFile, rm, writeFile } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");

  const directory = await mkdtemp(join(tmpdir(), "agent-bridge-noenv-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const envPath = join(directory, "env.json");
  const scriptPath = join(directory, "fake-claude.mjs");
  await writeFile(scriptPath, `
import { writeFileSync } from "node:fs";
writeFileSync(process.argv[2], JSON.stringify({
  baseUrl: process.env.ANTHROPIC_BASE_URL ?? null,
  nonessential: process.env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC ?? null,
}));
console.log(JSON.stringify({ type: "result", result: "ok" }));
`);

  const host = new PluginHost();
  await host.activate([createClaudeExecutionPlugin({
    backend: "cli", command: process.execPath, args: [scriptPath, envPath], cwd: directory,
  })]);
  const capability = host.capabilities.require("agent-bridge.execution.v1", "^1.0.0");
  await capability.whenReady();
  const session = await capability.createSession({ sessionId: "s-no-gateway", cwd: directory });
  const completed = new Promise(resolve => {
    capability.subscribeActivity(message => { if (message.method === "turn/completed") resolve(message); });
  });
  await capability.sendMessage(session.id, { text: "hi" });
  await completed;

  const seen = JSON.parse(await readFile(envPath, "utf8"));
  // Unconfigured means untouched, not blanked: the child inherits the parent's
  // environment exactly, including a base URL the operator exported for their
  // own reasons. What must not happen is the bridge adding its own routing.
  assert.equal(seen.baseUrl, process.env.ANTHROPIC_BASE_URL ?? null, "the bridge adds no base URL of its own");
  assert.equal(
    seen.nonessential, process.env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC ?? null,
    "the bridge adds the traffic flag only when it is routing through a gateway",
  );
  await host.dispose();
});

test("bridge initialization logs the transport endpoint once, and never a credential", async () => {
  const { createClaudeExecutionPlugin } = await import("../plugin.mjs");
  const { createDshClaudePlugin } = await import("../dsh-plugin.js");
  const { apply } = await import("../host-plugin.js");

  const infos = [];
  const ctx = {
    logger: { info: message => infos.push(message), warn() {}, error() {} },
    llm: { registerAdapter: () => () => {} },
    agents: { list: () => [], get: () => null },
    on: () => () => {},
    effect: () => () => {},
    inject: () => ({ dispose() {} }),
  };

  await apply(ctx, { claudeBackend: "sdk", bifrostBaseUrl: "http://127.0.0.1:8090/anthropic" });

  assert.deepEqual(infos, ["Claude transport endpoint: http://127.0.0.1:8090/anthropic"]);
  assert.equal(infos.some(line => /sk-ant|Bearer|authorization|AUTH_TOKEN|API_KEY/i.test(line)), false);
});
