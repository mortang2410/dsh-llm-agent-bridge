import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { PluginHost } from "../internal/plugin-sdk.mjs";
import { createClaudeExecutionPlugin } from "../plugin.mjs";
import { ClaudeSdkClient } from "../sdk-client.mjs";

/**
 * The whole DSH tool half, proved without a model and without a gateway.
 *
 * Every component on the bridge's side is production code: the plugin host, the
 * real `ClaudeSdkClient`, the real `ClaudeSessionRuntime`, and the real
 * `ClaudeDshAdapter` registered through the real DSH plugin path. Two things are
 * substituted, and neither is part of this repository:
 *
 *   - the model: a deterministic fake SDK stands in for
 *     `@anthropic-ai/claude-agent-sdk` and makes the one decision a model makes
 *     here, which is to ask for the bridged `read` tool;
 *   - DSH's agent loop, which lives in the host rather than in this repository.
 *     The driver plays it: it runs the call and appends the call and its result
 *     to the DSH agent session the bridge is attached to.
 *
 * So this proves the bridge's half of the loop end to end, and the contract the
 * host's half depends on. It does not prove the host's own loop, which this
 * repository does not contain.
 *
 * What the test drives, in order:
 *   1. Claude asks for DSH's `read` tool. The bridge parks the call and hands
 *      DSH a `tool-call` finish; the bridge executes nothing itself.
 *   2. DSH's loop runs the call and records the call and the result.
 *   3. DSH calls back with the result. The parked handler resolves, the same
 *      Claude query resumes, and its reply carries the file's contents.
 *
 * The parked query resuming rather than a new one starting is the part that
 * cannot be proved by inspecting the bridge: it is only visible as one SDK
 * query object serving both halves of the turn.
 */

test("Claude's DSH tool call runs in the harness and the same parked query resumes with the result", async () => {
  const workspace = await mkdtemp(join(tmpdir(), "agent-bridge-loop-"));
  const filePath = join(workspace, "notes.txt");
  const fileBody = "the file the harness read";
  await writeFile(filePath, `${fileBody}\n`, "utf8");

  // The fake SDK makes the model's one decision: ask for `read`. Its query
  // object is the thing whose identity proves the turn resumed rather than
  // restarted, so it is counted.
  const model = fakeModelSdk({ toolName: "read", args: { path: filePath } });

  // The real client, wrapping the fake SDK. `pathToClaudeCodeExecutable` stays
  // unset: no process is ever spawned.
  const client = new ClaudeSdkClient({ sdk: model.sdk });
  const { host, adapter, agent } = await activateBridge({ client, cwd: workspace });

  const messages = [{
    role: "user",
    source: { kind: "user" },
    content: [{ type: "text", text: "read notes.txt and tell me what it says" }],
  }];
  const request = {
    provider: "claude",
    model: "haiku",
    reasoningEffort: "low",
    sessionId: agent.id,
    messages,
    tools: [{
      name: "read",
      description: "Read a file from the workspace.",
      parameters: {
        type: "object",
        properties: { path: { type: "string" } },
        required: ["path"],
      },
    }],
  };

  // Step 1: Claude asks for the tool. DSH must be told to run it.
  const first = await collect(adapter.stream(request));
  const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");
  assert.ok(call, "the bridge must hand the tool call to DSH, not run it itself");
  assert.equal(call.block.name, "read", "DSH must see its own tool name");
  assert.deepEqual(JSON.parse(call.block.arguments), { path: filePath });
  assert.equal(first.at(-1).reason.kind, "tool-calls", "the turn ends in DSH's loop owning the execution");
  assert.equal(
    model.handlerSettled(), false,
    "the bridged handler must still be parked: the bridge is waiting on DSH, not doing the work",
  );
  assert.equal(model.queryCount, 1, "one Claude query spans both halves of the turn");

  // Step 2: DSH's agent loop runs the call and records it. That loop lives in
  // the host, not in this repository, so the driver performs it here — but it
  // records into the same DSH agent session the bridge is attached to, and the
  // id it records is the one it must hand back.
  const fileText = (await readFile(filePath, "utf8")).trim();
  agent.session.append("tool-call", { id: call.block.id, name: call.block.name, arguments: call.block.arguments });
  agent.session.append("tool-result", { toolCallId: call.block.id, content: [{ type: "text", text: fileText }], isError: false });
  const recorded = agent.session.log;
  assert.equal(recorded.length, 2, "DSH's trajectory holds the call and its result");
  assert.equal(
    recorded[0].data.id, call.block.id,
    "the id DSH records is the one the bridge parked, so the result can be matched back to it",
  );
  assert.equal(recorded[0].data.name, "read", "DSH's trajectory names the tool it ran, not the SDK's alias");

  // Step 3: DSH calls back with the result. The parked query must resume.
  const second = await collect(adapter.stream({
    ...request,
    messages: [
      ...messages,
      { role: "user", content: [{ type: "tool-result", toolCallId: recorded[1].data.toolCallId, content: recorded[1].data.content }] },
    ],
  }));

  assert.equal(model.handlerSettled(), true, "the parked handler resolved once DSH returned the result");
  assert.deepEqual(
    model.handlerResult.content,
    [{ type: "text", text: fileText }],
    "the harness result reached Claude verbatim",
  );
  assert.equal(model.queryCount, 1, "resuming must not start a second Claude query");
  const reply = second.filter(chunk => chunk.type === "text-delta").map(chunk => chunk.text).join("");
  assert.match(reply, new RegExp(fileBody), "the resumed reply must carry the file's contents");
  assert.equal(second.at(-1).reason.kind, "stop", "the resumed turn finishes normally");
  assert.equal(
    second.at(-1).replayState.claudeSessionId, adapter.sessionFor(agent.id),
    "the resumed turn replays into the same Claude session the DSH session is linked to",
  );

  await host.dispose();
});

/**
 * A deterministic stand-in for the Agent SDK, making only the decisions a model
 * makes: create the MCP server the bridge asks for, call the bridged tool, and
 * report the tool's result back in an assistant message.
 */
function fakeModelSdk({ toolName, args }) {
  const state = {
    servers: null,
    queryCount: 0,
    sessionId: "claude-session-1",
    handlerResult: null,
    settled: false,
  };
  const sdk = {
    createSdkMcpServer(options) {
      state.servers = options;
      return { type: "sdk", name: options.name };
    },
    tool(name, description, inputSchema, handler) {
      return { name, description, inputSchema, handler };
    },
    query(params) {
      // Only a turn carries an abort controller. The catalog read the client
      // opens at initialization is a control-only query and must not be
      // mistaken for a second turn.
      if (params.options.abortController) {
        state.queryCount += 1;
        state.sessionId = params.options.sessionId;
      }
      return queryObject(async function* () {
        // The model's decision: ask DSH to read the file. This is the bridged
        // handler, so it parks until the harness returns a result.
        const bridged = state.servers.tools.find(tool => tool.name === toolName);
        assert.ok(bridged, `${toolName} must be bridged into the SDK`);
        const result = await bridged.handler(args, { signal: new AbortController().signal });
        state.handlerResult = result;
        state.settled = true;

        const text = (result.content ?? []).map(block => block.text).join("\n");
        yield {
          type: "assistant",
          session_id: params.options.sessionId,
          uuid: "assistant-1",
          message: { id: "message-1", content: [{ type: "text", text: `notes.txt says: ${text}` }] },
        };
        yield {
          type: "result",
          session_id: params.options.sessionId,
          uuid: "result-1",
          subtype: "success",
          is_error: false,
          result: "done",
        };
      });
    },
  };
  return {
    sdk,
    get queryCount() { return state.queryCount; },
    get sessionId() { return state.sessionId; },
    get handlerResult() { return state.handlerResult; },
    handlerSettled: () => state.settled,
  };
}

/**
 * Activate the real bridge through the real DSH plugin path and return its
 * adapter.
 *
 * `linkPath` is pointed at a file inside the test's own temporary workspace. The
 * default resolves under the operator's `~/.dsh/plugin-data`, and the adapter
 * persists a link on the first turn — so without this, running the suite writes
 * a fake session into the real installation's link store.
 */
async function activateBridge({ client, cwd }) {
  const { createDshClaudePlugin } = await import("../dsh-plugin.js");
  let adapter = null;
  const agent = fakeAgent({ cwd });

  const ctx = {
    logger: { warn() {}, error() {}, info() {} },
    llm: { registerAdapter(_providers, registered) { adapter = registered; return () => {}; } },
    agents: { list: () => [agent], get: id => (String(id) === agent.id ? agent : null) },
    on: () => () => {},
    get: () => undefined,
    inject: () => ({ dispose() {} }),
  };

  const host = new PluginHost();
  await host.activate([
    createClaudeExecutionPlugin({ client, cwd }),
    createDshClaudePlugin(ctx, { telemetry: "ignore", linkPath: join(cwd, "agent-bridge-links.json") }),
  ]);
  assert.ok(adapter, "the DSH plugin path must register the Claude adapter");
  return { host, adapter, agent };
}

function fakeAgent({ cwd }) {
  return {
    id: "dsh-session-1",
    session: {
      header: { agentPreset: "standard", cwd },
      // DSH's Session is event-sourced: its public log is `log`, and there is no
      // `events` field. Mirroring the real name is what keeps this test honest —
      // a double that invents `events` would let the adapter's own bug through.
      log: [],
      append(type, data) { this.log.push({ type, data }); },
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
