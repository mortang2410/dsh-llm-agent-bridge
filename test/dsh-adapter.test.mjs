import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { ClaudeDshAdapter } from "../claude-adapter.js";
import { ClaudeLinkStore } from "../claude-link-store.js";
import { handleClaudeSdkRequest } from "../claude-tools.js";

test("a tool call is handed to DSH as a tool-call finish, and the answer resumes the same turn", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  const messages = [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "list the files" }] }];
  const request = {
    provider: "claude",
    model: "sonnet",
    reasoningEffort: "high",
    sessionId: agent.id,
    messages,
    tools: [{ name: "bash", description: "Run a command.", parameters: { type: "object", properties: { command: { type: "string" } } } }],
  };

  // Step one: Claude reasons, then asks for a tool. DSH must be told to run it.
  const first = await collect(adapter.stream(request));
  const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");
  assert.equal(first.find(chunk => chunk.type === "reasoning-delta").text, "Checked the workspace.");
  assert.equal(call.block.name, "bash", "DSH must see its own tool name");
  assert.deepEqual(JSON.parse(call.block.arguments), { command: "pwd" });
  assert.equal(first.at(-1).reason.kind, "tool-calls", "DSH's loop runs the tool, which is what fills its trajectory");
  assert.equal(runtime.resolved.length, 0, "nothing is resolved until DSH reports a result");

  // Step two: DSH executed it and calls back with the result.
  const second = await collect(adapter.stream({
    ...request,
    messages: [
      ...messages,
      { role: "user", content: [{ type: "tool-result", toolCallId: call.block.id, content: [{ type: "text", text: "/workspace/dsh" }] }] },
    ],
  }));

  assert.deepEqual(runtime.resolved.map(entry => entry.toolUseId), [call.block.id]);
  assert.equal(runtime.resolved[0].result.content[0].text, "/workspace/dsh", "the harness result reaches Claude verbatim");
  assert.equal(runtime.sent.length, 1, "resuming must not start a second Claude turn");
  assert.equal(second.find(chunk => chunk.type === "text-delta").text, "done");
  assert.equal(second.at(-1).reason.kind, "stop");
  assert.equal(second.at(-1).replayState.claudeSessionId, "claude-1");
  assert.equal(agent.appended.length, 0, "tool activity belongs to DSH's trajectory, not to a custom event");
});

test("a DSH 0.2.x tool result message resumes the parked turn", async () => {
  // DSH 0.2.x builds a tool result as a MESSAGE: role "tool", with the id on
  // `toolCallId` at message level and no `tool-result` content block at all
  // (dsh-llm 0.2.0-rc.2 `createToolResultMessage`). 0.1.x instead used a `user`
  // message holding one `tool-result` block. Reading only for 0.1.x silently
  // found nothing on the real host: the parked call was never resolved, the turn
  // was abandoned, and the model was told its tool call had been REJECTED even
  // though DSH had executed it. Every other test in this file uses the 0.1.x
  // shape, which is exactly why the suite passed while the real host failed.
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  const messages = [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "list the files" }] }];
  const request = {
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages,
    tools: [{ name: "bash", description: "Run a command.", parameters: { type: "object", properties: { command: { type: "string" } } } }],
  };

  const first = await collect(adapter.stream(request));
  const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");
  assert.equal(first.at(-1).reason.kind, "tool-calls");

  const second = await collect(adapter.stream({
    ...request,
    messages: [
      ...messages,
      {
        role: "tool",
        source: { kind: "tool", callId: call.block.id },
        toolCallId: call.block.id,
        content: [{ type: "text", text: "/workspace/dsh" }],
        isError: false,
      },
    ],
  }));

  assert.deepEqual(
    runtime.resolved.map(entry => entry.toolUseId), [call.block.id],
    "the 0.2.x message-shaped result must resolve the parked call",
  );
  assert.equal(runtime.resolved[0].result.content[0].text, "/workspace/dsh", "the harness result reaches Claude verbatim");
  assert.equal(runtime.sent.length, 1, "resuming must not start a second Claude turn");
  assert.equal(second.at(-1).reason.kind, "stop", "the turn resumes and finishes rather than being abandoned");
});

test("Claude models expose native reasoning effort choices", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });

  const model = await adapter.resolveModel("claude", "sonnet");

  assert.deepEqual(model.reasoning.efforts.map(effort => effort.id), ["low", "high"]);
  assert.equal(model.reasoning.defaultEffort, undefined, "an unset effort lets Claude Code apply its own default");
});

test("automatic title generation uses an isolated ephemeral Claude session", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  const [mainChunks, titleChunks] = await Promise.all([
    collect(adapter.stream({
      provider: "claude",
      model: "sonnet",
      sessionId: agent.id,
      messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "list project files" }] }],
    })),
    collect(adapter.stream({
      provider: "claude",
      model: "sonnet",
      sessionId: agent.id,
      purpose: "session-title",
      system: "Generate a concise title.",
      messages: [{
        role: "user",
        source: { kind: "plugin", plugin: "dsh-session-title-llm" },
        content: [{ type: "text", text: "Generate the session title from this JSON array: [\"list project files\"]" }],
      }],
    })),
  ]);

  const auxiliaryConfig = runtime.createdConfigs.find(config => config.ephemeral === true);
  assert.equal(adapter.sessionFor(agent.id), "claude-1");
  assert.equal(auxiliaryConfig.sandbox, "read-only");
  assert.equal(auxiliaryConfig.approvalPolicy, "never");
  assert.deepEqual(auxiliaryConfig.settingSources, [], "auxiliary calls must not load ~/.claude settings either");
  assert.deepEqual(runtime.released, ["claude-2"]);
  assert.equal(mainChunks.find(chunk => chunk.type === "text-delta").text, "done");
  assert.equal(titleChunks.find(chunk => chunk.type === "text-delta").text, "项目文件查询");
});

test("DSH-to-Claude links and configuration survive host restart", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "claude-sdk-links-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const path = join(directory, "links.json");
  const firstRuntime = new FakeRuntime();
  const first = new ClaudeDshAdapter({ runtime: firstRuntime, ready: Promise.resolve(), linkStore: new ClaudeLinkStore(path) });
  first.configure("dsh-1", { model: "sonnet", effort: "high", sandbox: "read-only" });
  const claudeSessionId = await first.ensureSession("dsh-1");

  const persisted = JSON.parse(await readFile(path, "utf8"));
  assert.equal(persisted.sessions["dsh-1"].claudeSessionId, claudeSessionId);
  const secondRuntime = new FakeRuntime();
  const second = new ClaudeDshAdapter({ runtime: secondRuntime, ready: Promise.resolve(), linkStore: new ClaudeLinkStore(path) });
  assert.equal(await second.ensureSession("dsh-1"), claudeSessionId);
  assert.equal(second.configuration("dsh-1").sandbox, "read-only");
  assert.equal(secondRuntime.created, 0);
  assert.equal(secondRuntime.resumed, 1);
});

test("concurrent first messages create exactly one Claude session", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const [left, right] = await Promise.all([adapter.ensureSession("dsh-1"), adapter.ensureSession("dsh-1")]);
  assert.equal(left, right);
  assert.equal(runtime.created, 1);
});

test("the session log's sandbox and approval events reach the runtime config", async () => {
  // The adapter reads DSH's session log to decide the sandbox and approval
  // policy it hands the child. Nothing else in this file appends those events,
  // so without this test the log could regress to always reading empty and the
  // suite would stay green while a real read-only or `never` policy was silently
  // replaced by the adapter's own defaults.
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  agent.session.append("sandbox/mode", { mode: "read-only" });
  agent.session.append("approval/policy", { policy: "never" });
  adapter.attachAgent(agent);

  const chunks = await collect(adapter.stream({
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "list the files" }] }],
  }));

  assert.equal(chunks.at(-1).reason.kind, "stop");
  const config = runtime.sent[0].message;
  assert.equal(config.sandbox, "read-only", "DSH's sandbox choice must survive into the Claude config");
  assert.equal(config.approvalPolicy, "never", "DSH's approval choice must survive into the Claude config");
});

test("without permission events the adapter applies its own documented defaults", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  await collect(adapter.stream({
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "list the files" }] }],
  }));

  const config = runtime.sent[0].message;
  assert.equal(config.sandbox, "workspace-write", "the default is the permissive-but-scoped sandbox");
  assert.equal(config.approvalPolicy, "on-request", "and the default asks before acting");
});

test("Claude SDK permission and question requests use DSH services", async () => {
  const agent = fakeAgent();
  const adapter = { dshSessionForClaudeSession: sessionId => sessionId === "claude-1" ? agent.id : null };
  const calls = { approvals: [], questions: [] };
  const ctx = {
    agents: { get: id => id === agent.id ? agent : null },
    approval: { async request(input) { calls.approvals.push(input); return "allowed-once"; } },
    userQuestions: {
      async ask(input) {
        calls.questions.push(input);
        return { answers: [{ id: "question-1", selected: ["Detailed"], custom: "with tests" }] };
      },
    },
  };
  const runtime = new InteractionRuntime();

  await handleClaudeSdkRequest(ctx, {
    adapter,
    runtime,
    request: {
      id: "approval-1",
      method: "tool/requestApproval",
      params: { sessionId: "claude-1", toolName: "Write", input: { file_path: "probe.txt" }, title: "Claude wants to write probe.txt" },
    },
  });
  assert.equal(calls.approvals[0].agent, agent);
  assert.equal(calls.approvals[0].toolName, "Claude Write");
  assert.equal(runtime.resolved.at(-1).response.action, "accept");

  await handleClaudeSdkRequest(ctx, {
    adapter,
    runtime,
    request: {
      id: "question-1",
      method: "tool/requestUserInput",
      params: { sessionId: "claude-1", input: { questions: [{
        question: "How detailed?",
        header: "Detail",
        options: [{ label: "Brief" }, { label: "Detailed" }],
      }] } },
    },
  });
  assert.equal(calls.questions[0].questions[0].question, "How detailed?");
  assert.deepEqual(runtime.resolved.at(-1).response.answers, { "How detailed?": "Detailed" });
});

test("DSH's tools travel to Claude and their results travel back untouched", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  const messages = [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "use the probe" }] }];
  const request = {
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages,
    tools: [{
      name: "cross_plugin_probe",
      description: "Probe a separately installed DSH plugin.",
      parameters: { type: "object", properties: { value: { type: "string" } }, required: ["value"] },
    }],
  };

  const first = await collect(adapter.stream(request));
  assert.deepEqual(runtime.sent[0].message.dshTools.map(tool => tool.name), ["cross_plugin_probe"]);
  assert.equal(runtime.sent[0].message.executeDshTool, undefined, "the adapter must not execute DSH's tools itself");
  const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");
  assert.equal(call.block.name, "cross_plugin_probe");

  await collect(adapter.stream({
    ...request,
    messages: [
      ...messages,
      { role: "user", content: [{ type: "tool-result", toolCallId: call.block.id, isError: true, content: [{ type: "text", text: "probe failed" }] }] },
    ],
  }));

  assert.equal(runtime.resolved[0].result.isError, true, "a failed harness tool must reach Claude as a failure");
  assert.equal(runtime.resolved[0].result.content[0].text, "probe failed");
});

test("DSH's mid-turn context splices never swallow the instruction", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  // The real shape from a DSH standard-mode turn: the human's message is first,
  // then DSH appends workspace instructions, a runtime snapshot, and the skills
  // catalogue as further "user" messages.
  for await (const chunk of adapter.stream({
    provider: "claude",
    model: "haiku",
    sessionId: agent.id,
    messages: [
      { role: "user", source: { kind: "user" }, content: [{ type: "text", text: "list the files here" }] },
      { role: "user", source: { kind: "agent-instructions" }, content: [{ type: "text", text: "<system-reminder>workspace instructions</system-reminder>" }] },
      { role: "user", source: { kind: "plugin", plugin: "@deepseek-ai/dsh-system-prompt" }, content: [{ type: "text", text: "Current runtime context." }] },
      { role: "user", source: { kind: "skill-catalog" }, content: [{ type: "text", text: "<system-reminder>skills</system-reminder>" }] },
    ],
  })) void chunk;

  const sent = runtime.sent[0].message.text;
  assert.match(sent, /list the files here$/, "the instruction must be the final thing Claude reads");
  assert.equal(sent.match(/list the files here/g).length, 1, "the instruction must not also appear inside the context block");
  assert.match(sent, /workspace instructions/, "DSH's spliced context must still reach Claude");
  assert.match(sent, /Current runtime context/);
  assert.doesNotMatch(sent, /do not act on them/i, "current-turn context must not be labelled as inert history");
});

const CHECKPOINT = "CHECKPOINT: the compacted summary of the earlier work";

function userMessage(text) {
  return { role: "user", source: { kind: "user" }, content: [{ type: "text", text }] };
}

function replyMessage(text) {
  return { role: "assistant", content: [{ type: "text", text }] };
}

/** The message DSH's compaction puts in place of the span it shadowed. */
function checkpointMessage() {
  return { role: "user", source: { kind: "compaction" }, content: [{ type: "text", text: CHECKPOINT }] };
}

/** Event data for a reply the Claude route produced: DSH records the provider on the message source. */
function claudeReply() {
  return { message: { source: { kind: "model", provider: "claude" } } };
}

/** Event data for a reply some other model produced in the same DSH session. */
function otherModelReply() {
  return { message: { source: { kind: "model", provider: "openai" } } };
}

test("after DSH compacts history Claude already holds, the next request starts a new Claude session seeded from the compacted history", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  agent.session.append("user/message", {});
  await collect(adapter.stream({ provider: "claude", model: "sonnet", sessionId: agent.id, messages: [userMessage("first question")] }));

  // DSH records Claude's reply, then compacts the span it has seen into one checkpoint.
  agent.session.append("assistant/message", claudeReply(), { surfaceOp: "append" });
  agent.session.append("compaction/summary", { shadowedSeqs: [1, 2] });
  agent.session.append("user/message", {}, { surfaceOp: { op: "replace", startSeq: 1, endSeq: 2 } });

  await collect(adapter.stream({
    provider: "claude", model: "sonnet", sessionId: agent.id,
    messages: [checkpointMessage(), userMessage("second question")],
  }));

  assert.equal(runtime.createdConfigs.length, 2, "the old transcript still holds the uncompacted history, so Claude must start over");
  assert.deepEqual(runtime.released, ["claude-1"], "the old Claude session is released, not left running");
  const sent = runtime.sent[1].message.text;
  assert.ok(sent.includes(CHECKPOINT), "the compacted history must reach Claude");
  assert.match(sent, /second question$/, "with the live question last");
});

test("a compaction that replaces only Claude's own last reply still starts a new Claude session", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  agent.session.append("user/message", {}, { surfaceOp: "append" });
  await collect(adapter.stream({ provider: "claude", model: "sonnet", sessionId: agent.id, messages: [userMessage("first question")] }));

  // DSH logs the reply Claude has just given, then compacts only that reply. Claude holds the
  // reply, so the transcript no longer matches DSH even though the span starts after the last send.
  agent.session.append("assistant/message", claudeReply(), { surfaceOp: "append" });
  agent.session.append("compaction/summary", { shadowedSeqs: [2] });
  agent.session.append("user/message", {}, { surfaceOp: { op: "replace", startSeq: 2, endSeq: 2 } });

  await collect(adapter.stream({
    provider: "claude", model: "sonnet", sessionId: agent.id,
    messages: [userMessage("first question"), checkpointMessage(), userMessage("second question")],
  }));

  assert.equal(runtime.createdConfigs.length, 2, "Claude still holds the reply that DSH replaced");
  assert.ok(runtime.sent[1].message.text.includes(CHECKPOINT), "the checkpoint must reach Claude");
});

test("a compaction while a Claude turn waits on a tool result fails only that parked call and continues in a new session", async () => {
  const runtime = new FakeRuntime();
  const rejected = [];
  runtime.rejectToolCall = (parkId) => { rejected.push(parkId); return true; };
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);
  const tools = [{ name: "bash", description: "Run a command.", parameters: { type: "object", properties: { command: { type: "string" } } } }];

  agent.session.append("user/message", {});
  const first = await collect(adapter.stream({ provider: "claude", model: "sonnet", sessionId: agent.id, messages: [userMessage("go")], tools }));
  const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");
  assert.ok(call, "the first request must park on a tool call");

  // DSH runs the tool, then compacts the span up to and including its result.
  agent.session.append("assistant/message", claudeReply(), { surfaceOp: "append" });
  agent.session.append("tool/result", {});
  agent.session.append("compaction/summary", { shadowedSeqs: [1, 2, 3] });
  agent.session.append("user/message", {}, { surfaceOp: { op: "replace", startSeq: 1, endSeq: 3 } });

  const result = { role: "user", content: [{ type: "tool-result", toolCallId: call.block.id, content: [{ type: "text", text: "/workspace/dsh" }] }] };
  const second = await collect(adapter.stream({
    provider: "claude", model: "sonnet", sessionId: agent.id,
    messages: [checkpointMessage(), result],
  }));

  assert.deepEqual(runtime.resolved, [], "the result must not be fed into the transcript Claude no longer shares with DSH");
  assert.deepEqual(rejected, ["park-1"], "only this turn's parked call is failed");
  assert.equal(runtime.createdConfigs.length, 2, "the continuation runs in a new Claude session");
  const sent = runtime.sent[1].message.text;
  assert.ok(sent.includes(CHECKPOINT), "the compacted history must reach the new session");
  assert.ok(sent.includes("tool result: /workspace/dsh"), "the tool result DSH recorded must reach the new session");
  assert.match(sent, /do not repeat/i, "the continuation must not re-run the tool call that already ran");
  assert.equal(second.at(-1).reason.kind, "stop");
});

test("a user message that replaces a parked turn fails only that turn's parked call, never another chat's", async () => {
  const runtime = new FakeRuntime();
  const rejected = [];
  let rejectedAll = 0;
  runtime.rejectToolCall = (parkId) => { rejected.push(parkId); return true; };
  runtime.rejectAllToolCalls = () => { rejectedAll += 1; };
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  const first = await collect(adapter.stream({
    provider: "claude", model: "sonnet", sessionId: agent.id,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }],
    tools: bashTool,
  }));
  assert.equal(first.at(-1).reason.kind, "tool-calls", "the first request must park on a tool call");

  const claudeSession = runtime.sent[0].sessionId;
  // Claude announces a second call after the handoff, while DSH holds the turn. It is still this turn's.
  runtime.emit("activity", notification("tool/parked", claudeSession, "turn-1", { parkId: "late-park", name: "bash", arguments: { command: "date" } }));
  // A waiting call of another chat on the same runtime must survive.
  runtime.emit("activity", notification("tool/parked", "other-claude-session", "turn-9", { parkId: "other-park", name: "bash", arguments: { command: "ls" } }));

  // The user types a new message instead of answering the parked tool call.
  await collect(adapter.stream({
    provider: "claude", model: "sonnet", sessionId: agent.id,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "never mind" }] }],
  }));

  assert.equal(rejectedAll, 0, "the broad cancel would also fail parked calls of other DSH chats");
  assert.deepEqual(rejected, ["park-1", "late-park"], "this turn's waiting calls are failed, and no other chat's");
  assert.deepEqual(runtime.interrupted, [{ sessionId: claudeSession, turnId: "turn-1" }], "the abandoned turn is interrupted");
});

test("aborting a turn while its tool call is parked fails only that turn's parked call, never another chat's", async () => {
  const runtime = new SilentCallRuntime();
  const rejected = [];
  let rejectedAll = 0;
  runtime.rejectToolCall = (parkId) => { rejected.push(parkId); return true; };
  runtime.rejectAllToolCalls = () => { rejectedAll += 1; };
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);
  const controller = new AbortController();
  // Another chat parks a call while this turn is still waiting on its usage.
  setTimeout(() => runtime.emit("activity", notification("tool/parked", "other-claude-session", "turn-9", { parkId: "other-park", name: "bash", arguments: { command: "ls" } })), 20);
  setTimeout(() => controller.abort(), 50);
  const chunks = await collect(adapter.stream({
    provider: "claude", model: "sonnet", sessionId: agent.id,
    signal: controller.signal,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }],
    tools: bashTool,
  }));

  assert.equal(chunks.at(-1).reason.kind, "aborted");
  assert.equal(rejectedAll, 0, "the broad cancel would also fail parked calls of other DSH chats");
  assert.deepEqual(rejected, ["park-1"], "only this turn's parked call is failed, not the other chat's");
  assert.deepEqual(runtime.interrupted, [{ sessionId: runtime.sent[0].sessionId, turnId: "turn-1" }], "the aborted turn is interrupted");
});

test("a plain follow-up keeps the Claude session, and rewriting only the system prompt does not reset it", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  agent.session.append("user/message", {});
  await collect(adapter.stream({ provider: "claude", model: "sonnet", sessionId: agent.id, messages: [userMessage("first question")] }));
  agent.session.append("assistant/message", claudeReply(), { surfaceOp: "append" });
  agent.session.append("system/message", {}, { surfaceOp: { op: "replace", startSeq: 1, endSeq: 1 } });
  agent.session.append("user/message", {}, { surfaceOp: "append" });
  await collect(adapter.stream({
    provider: "claude", model: "sonnet", sessionId: agent.id,
    messages: [userMessage("first question"), replyMessage("done"), userMessage("second question")],
  }));

  assert.equal(runtime.createdConfigs.length, 1, "no DSH history was compacted, so the transcript still matches");
  assert.deepEqual(runtime.released, []);
});

test("a compaction that replaces a reply from another model does not reset the Claude session", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  agent.session.append("user/message", {}, { surfaceOp: "append" });
  await collect(adapter.stream({ provider: "claude", model: "sonnet", sessionId: agent.id, messages: [userMessage("first question")] }));

  // Another model answers, DSH compacts only that answer, and Claude is selected again. Claude
  // never held the answer, so its transcript still matches DSH and the checkpoint arrives as context.
  agent.session.append("assistant/message", otherModelReply(), { surfaceOp: "append" });
  agent.session.append("compaction/summary", { shadowedSeqs: [2] });
  agent.session.append("user/message", {}, { surfaceOp: { op: "replace", startSeq: 2, endSeq: 2 } });

  await collect(adapter.stream({
    provider: "claude", model: "sonnet", sessionId: agent.id,
    messages: [userMessage("first question"), checkpointMessage(), userMessage("second question")],
  }));

  assert.equal(runtime.createdConfigs.length, 1, "the rewritten answer was never in Claude's transcript");
  assert.deepEqual(runtime.released, []);
  assert.ok(runtime.sent[1].message.text.includes(CHECKPOINT), "the checkpoint still reaches Claude as context");
});

test("rewriting a message DSH has not yet sent to Claude does not reset the session", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  agent.session.append("user/message", {}, { surfaceOp: "append" });
  await collect(adapter.stream({ provider: "claude", model: "sonnet", sessionId: agent.id, messages: [userMessage("first question")] }));

  // Claude replies, the user sends a message with an attachment, and DSH offloads that attachment
  // before the next request. Claude never held the attachment, so the transcript still matches.
  agent.session.append("assistant/message", claudeReply(), { surfaceOp: "append" });
  agent.session.append("user/message", {}, { surfaceOp: "append" });
  agent.session.append("user/message", {}, { surfaceOp: { op: "replace", startSeq: 3, endSeq: 3 } });

  await collect(adapter.stream({
    provider: "claude", model: "sonnet", sessionId: agent.id,
    messages: [userMessage("first question"), replyMessage("done"), userMessage("second question")],
  }));

  assert.equal(runtime.createdConfigs.length, 1, "nothing Claude holds was rewritten");
  assert.deepEqual(runtime.released, []);
});

test("a rewrite Claude already built on does not reset the session again after a host restart", async () => {
  const dir = await mkdtemp(join(tmpdir(), "claude-links-"));
  const path = join(dir, "links.json");
  try {
    const runtime = new FakeRuntime();
    const agent = fakeAgent();
    const before = new ClaudeDshAdapter({ runtime, ready: Promise.resolve(), linkStore: new ClaudeLinkStore(path) });
    before.attachAgent(agent);

    agent.session.append("user/message", {});
    await collect(before.stream({ provider: "claude", model: "sonnet", sessionId: agent.id, messages: [userMessage("first question")] }));
    agent.session.append("compaction/summary", { shadowedSeqs: [1] });
    agent.session.append("user/message", {}, { surfaceOp: { op: "replace", startSeq: 1, endSeq: 1 } });
    await collect(before.stream({
      provider: "claude", model: "sonnet", sessionId: agent.id,
      messages: [checkpointMessage(), userMessage("second question")],
    }));
    assert.equal(runtime.createdConfigs.length, 2);

    // The host restarts. Nothing has been rewritten since Claude was last sent a prompt.
    agent.session.append("assistant/message", claudeReply(), { surfaceOp: "append" });
    agent.session.append("user/message", {}, { surfaceOp: "append" });
    const after = new ClaudeDshAdapter({ runtime, ready: Promise.resolve(), linkStore: new ClaudeLinkStore(path) });
    after.attachAgent(agent);
    await collect(after.stream({
      provider: "claude", model: "sonnet", sessionId: agent.id,
      messages: [checkpointMessage(), userMessage("second question"), replyMessage("done"), userMessage("third question")],
    }));

    assert.equal(runtime.createdConfigs.length, 2, "the rewrite is behind Claude's last prompt, so the session must be kept");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

class FakeRuntime extends EventEmitter {
  constructor() {
    super();
    this.models = [{
      id: "sonnet",
      displayName: "Sonnet 5",
      isDefault: true,
      supportedReasoningEfforts: [{ reasoningEffort: "low" }, { reasoningEffort: "high" }],
    }];
    this.sessions = new Map();
    this.sent = [];
    this.created = 0;
    this.resumed = 0;
    this.createdConfigs = [];
    this.released = [];
    this.resolved = [];
    this.interrupted = [];
    this.parked = null;
  }

  async createSession(config) {
    await new Promise(resolve => setTimeout(resolve, 2));
    this.createdConfig = config;
    this.createdConfigs.push(structuredClone(config));
    const session = { id: `claude-${++this.created}`, turns: [], ...config };
    this.sessions.set(session.id, session);
    return session;
  }

  async resumeSession(sessionId, config) {
    this.resumed += 1;
    this.sessions.set(sessionId, { id: sessionId, turns: [], ...config });
    return this.sessions.get(sessionId);
  }

  async sendMessage(sessionId, message) {
    this.sent.push({ sessionId, message });
    const turnId = "turn-1";
    this.answer = message.text.includes("Generate the session title") ? "项目文件查询" : "done";
    queueMicrotask(() => {
      this.emit("activity", notification("item/reasoning/summaryTextDelta", sessionId, turnId, {
        itemId: "reason-1", delta: "Checked the workspace.",
      }));
      const tool = message.dshTools?.[0];
      if (tool) {
        // Park: Claude asked for a tool and is waiting on the harness to run it.
        this.parked = { sessionId, turnId };
        this.emit("activity", notification("tool/parked", sessionId, turnId, {
          parkId: "park-1", name: tool.name, arguments: { command: "pwd" },
        }));
        return;
      }
      this.finishTurn(sessionId, turnId);
    });
    return { id: turnId, status: "inProgress", items: [] };
  }

  resolveToolCall(toolUseId, result) {
    this.resolved.push({ toolUseId, result });
    const parked = this.parked;
    this.parked = null;
    if (!parked) return false;
    queueMicrotask(() => this.finishTurn(parked.sessionId, parked.turnId));
    return true;
  }

  rejectAllToolCalls() {}

  contextWindowFor() { return 200000; }

  finishTurn(sessionId, turnId) {
    this.emit("activity", notification("item/agentMessage/delta", sessionId, turnId, { itemId: "answer-1", delta: this.answer }));
    this.emit("activity", { method: "turn/completed", params: {
      sessionId,
      turn: {
        id: turnId,
        status: "completed",
        error: null,
        items: [],
        usage: { inputTokens: 20, outputTokens: 160, cacheReadTokens: 17844, cacheWriteTokens: 0, reasoningTokens: 75 },
      },
    } });
  }

  async interruptTurn(sessionId, turnId) { this.interrupted.push({ sessionId, turnId }); }

  async releaseSession(sessionId) {
    this.released.push(sessionId);
    this.sessions.delete(sessionId);
  }
}

class InteractionRuntime {
  constructor() { this.resolved = []; this.rejected = []; }
  async resolveRequest(id, response) { this.resolved.push({ id, response }); }
  rejectRequest(id, error) { this.rejected.push({ id, error }); }
}

function fakeAgent({ tools = null } = {}) {
  const appended = [];
  return {
    id: "dsh-1",
    appended,
    ctx: tools ? { tools } : {},
    session: {
      header: { agentPreset: "standard", cwd: "/workspace/dsh" },
      // DSH's Session is event-sourced and its public log is `log`; there is no
      // `events` field. A double that invents one hides a real adapter bug, so
      // append() writes the same array the adapter reads.
      log: [],
      // Each event carries the `seq`, and for a rewrite of earlier history the
      // `surfaceOp` range, as DSH's real log records them.
      append(type, data, { surfaceOp } = {}) {
        appended.push({ type, data });
        this.log.push({ seq: this.log.length + 1, type, data, ...(surfaceOp === undefined ? {} : { surfaceOp }) });
      },
    },
  };
}

function notification(method, sessionId, turnId, rest) {
  return { method, params: { sessionId, turnId, ...rest } };
}

async function collect(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}

test("a resumed turn that never speaks fails instead of stalling forever", async () => {
  const runtime = new FakeRuntime();
  runtime.resolveToolCall = function (toolUseId, result) {
    this.resolved.push({ toolUseId, result });
    return true; // accepted, but the turn never emits again
  };
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  const messages = [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }];
  const request = {
    provider: "claude",
    sessionId: agent.id,
    messages,
    tools: [{ name: "bash", description: "run", parameters: { type: "object", properties: {} } }],
  };
  const first = await collect(adapter.stream(request));
  const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");

  process.env.DSH_AGENT_BRIDGE_RESUME_TIMEOUT_MS = "80";
  try {
    await assert.rejects(
      () => collect(adapter.stream({
        ...request,
        messages: [...messages, { role: "user", content: [{ type: "tool-result", toolCallId: call.block.id, content: [{ type: "text", text: "ok" }] }] }],
      })),
      /did not respond within/,
    );
  } finally {
    delete process.env.DSH_AGENT_BRIDGE_RESUME_TIMEOUT_MS;
  }
});

test("a resume that cannot start releases the turn's usage, and nothing is left behind", async () => {
  const runtime = new FakeRuntime();
  runtime.resolveToolCall = () => false;
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  const messages = [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }];
  const request = {
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages,
    tools: [{ name: "bash", description: "run", parameters: { type: "object", properties: {} } }],
  };
  const first = await collect(adapter.stream(request));
  const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");
  await assert.rejects(() => collect(adapter.stream({
    ...request,
    messages: [...messages, { role: "user", content: [{ type: "tool-result", toolCallId: call.block.id, content: [] }] }],
  })), /No Claude tool call is waiting/);
  assert.equal(adapter.usageLedgers.size, 0, "the failed resume released its turn's ledger, even though it never started");
  assert.equal(adapter.ownedSessions.size, 0, "the failed resume gave up the session");
  assert.equal(adapter.readers.size, 0, "no step is left reading the failed resume");
});

test("a result nothing is waiting for fails the turn rather than parking it", async () => {
  const runtime = new FakeRuntime();
  runtime.resolveToolCall = () => false;
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  const messages = [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }];
  const request = {
    provider: "claude",
    sessionId: agent.id,
    messages,
    tools: [{ name: "bash", description: "run", parameters: { type: "object", properties: {} } }],
  };
  const first = await collect(adapter.stream(request));
  const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");

  await assert.rejects(() => collect(adapter.stream({
    ...request,
    messages: [...messages, { role: "user", content: [{ type: "tool-result", toolCallId: call.block.id, content: [] }] }],
  })), /No Claude tool call is waiting/);
});

test("a turn answered by another model carries its tool work, not just its prose", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  await collect(adapter.stream({
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages: [
      // What DSH's history looks like after another provider answered a turn.
      { role: "user", source: { kind: "user" }, content: [{ type: "text", text: "read package.json" }] },
      { role: "assistant", content: [{ type: "tool-call", id: "c1", name: "read", arguments: '{"file_path":"package.json"}' }] },
      { role: "user", content: [{ type: "tool-result", toolCallId: "c1", content: [{ type: "text", text: '{"name":"auth-ts","devDependencies":{"nx":"23.1.1"}}' }] }] },
      { role: "assistant", content: [{ type: "text", text: "It pins nx." }] },
      { role: "user", source: { kind: "user" }, content: [{ type: "text", text: "what version was that?" }] },
    ],
  }));

  const sent = runtime.sent[0].message.text;
  assert.match(sent, /tool result: .*23\.1\.1/, "the other model's tool output must reach Claude");
  assert.match(sent, /called read/, "and what it called to get it");
  assert.match(sent, /what version was that\?$/, "with the live question last");
});

test("the harness is given token accounting and context capacity", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  const chunks = await collect(adapter.stream({
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "hi" }] }],
  }));

  const usage = chunks.find(chunk => chunk.type === "usage");
  assert.ok(usage, "without a usage chunk the harness can report no tokens, cache rate or tok/s");
  assert.deepEqual(usage.usage, {
    inputTokens: 20, outputTokens: 160, cacheReadTokens: 17844, cacheWriteTokens: 0, reasoningTokens: 75,
  });
  // The protocol requires usage before the terminal finish, and nothing after.
  assert.equal(chunks.indexOf(usage), chunks.length - 2);
  assert.equal(chunks.at(-1).type, "finish");

  const model = await adapter.resolveModel("claude", "sonnet");
  assert.deepEqual(model.context, { contextWindow: 200000 }, "context pressure cannot render without a window");
});

/**
 * A two-step turn as the SDK reports it: the first model call asks for a tool,
 * the second answers. Each call reports usage as it runs; the turn's total
 * arrives last, and only it carries the reasoning count.
 */
class PerCallRuntime extends FakeRuntime {
  async sendMessage(sessionId, message) {
    this.sent.push({ sessionId, message });
    const turnId = "turn-1";
    queueMicrotask(() => {
      this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-a", usage: { input_tokens: 100 } }));
      this.parked = { sessionId, turnId };
      this.emit("activity", notification("tool/parked", sessionId, turnId, {
        parkId: "park-1", name: message.dshTools[0].name, arguments: { command: "pwd" },
      }));
      // The SDK can park the tool before the model call reports its output.
      setTimeout(() => {
        this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-a", usage: { output_tokens: 20 } }));
        this.emit("activity", notification("usage/apiCallEnded", sessionId, turnId, { callId: "msg-a" }));
      }, 20);
    });
    return { id: turnId, status: "inProgress", items: [] };
  }

  resolveToolCall(toolUseId, result) {
    this.resolved.push({ toolUseId, result });
    const parked = this.parked;
    this.parked = null;
    if (!parked) return false;
    queueMicrotask(() => {
      const { sessionId, turnId } = parked;
      this.emit("activity", notification("usage/apiCall", sessionId, turnId, {
        callId: "msg-b", usage: { input_tokens: 200, cache_read_input_tokens: 50, output_tokens: 30 },
      }));
      this.emit("activity", notification("usage/apiCallEnded", sessionId, turnId, { callId: "msg-b" }));
      this.emit("activity", { method: "turn/completed", params: { sessionId, turn: {
        id: turnId,
        status: "completed",
        error: null,
        items: [],
        usage: { inputTokens: 300, outputTokens: 50, cacheReadTokens: 50, cacheWriteTokens: 0, reasoningTokens: 40 },
      } } });
    });
    return true;
  }
}

test("each DSH step reports the usage of the model calls it made, and the steps add up to the turn", async () => {
  const runtime = new PerCallRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);

  const messages = [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }];
  const request = {
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages,
    tools: [{ name: "bash", description: "run", parameters: { type: "object", properties: {} } }],
  };
  const first = await collect(adapter.stream(request));
  const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");
  const second = await collect(adapter.stream({
    ...request,
    messages: [...messages, { role: "user", content: [{ type: "tool-result", toolCallId: call.block.id, content: [{ type: "text", text: "/workspace" }] }] }],
  }));

  const firstUsage = first.filter(chunk => chunk.type === "usage").map(chunk => chunk.usage);
  const secondUsage = second.filter(chunk => chunk.type === "usage").map(chunk => chunk.usage);
  assert.deepEqual(firstUsage, [{ inputTokens: 100, outputTokens: 20, cacheReadTokens: 0, cacheWriteTokens: 0 }],
    "the tool-calling step reports its own call, not the turn");
  assert.deepEqual(secondUsage, [{ inputTokens: 200, outputTokens: 30, cacheReadTokens: 50, cacheWriteTokens: 0, reasoningTokens: 40 }],
    "the final step reports its own call and the turn's reasoning count");
  assert.equal(first.at(-1).type, "finish");
  assert.equal(second.at(-1).type, "finish");
  assert.ok(first.findIndex(chunk => chunk.type === "usage") < first.length - 1, "usage precedes the terminal finish");

  // No call is dropped or counted twice: the steps sum to the turn's own total.
  const sum = [...firstUsage, ...secondUsage].reduce((total, usage) => ({
    inputTokens: total.inputTokens + usage.inputTokens,
    outputTokens: total.outputTokens + usage.outputTokens,
    cacheReadTokens: total.cacheReadTokens + usage.cacheReadTokens,
    cacheWriteTokens: total.cacheWriteTokens + usage.cacheWriteTokens,
  }), { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0 });
  assert.deepEqual(sum, { inputTokens: 300, outputTokens: 50, cacheReadTokens: 50, cacheWriteTokens: 0 });
});

/** A call whose output finishes only after its step has reported, as a late report would. */
class LateReportRuntime extends FakeRuntime {
  async sendMessage(sessionId, message) {
    this.sent.push({ sessionId, message });
    const turnId = "turn-1";
    queueMicrotask(() => {
      this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-a", usage: { input_tokens: 100, output_tokens: 1 } }));
      this.parked = { sessionId, turnId };
      this.emit("activity", notification("tool/parked", sessionId, turnId, {
        parkId: "park-1", name: message.dshTools[0].name, arguments: { command: "pwd" },
      }));
    });
    return { id: turnId, status: "inProgress", items: [] };
  }

  resolveToolCall(toolUseId, result) {
    this.resolved.push({ toolUseId, result });
    const parked = this.parked;
    this.parked = null;
    if (!parked) return false;
    queueMicrotask(() => {
      const { sessionId, turnId } = parked;
      this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-a", usage: { output_tokens: 20 } }));
      this.emit("activity", notification("usage/apiCallEnded", sessionId, turnId, { callId: "msg-a" }));
      this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-b", usage: { input_tokens: 200, output_tokens: 30 } }));
      this.emit("activity", notification("usage/apiCallEnded", sessionId, turnId, { callId: "msg-b" }));
      this.emit("activity", { method: "turn/completed", params: { sessionId, turn: {
        id: turnId,
        status: "completed",
        error: null,
        items: [],
        usage: { inputTokens: 300, outputTokens: 50, cacheReadTokens: 0, cacheWriteTokens: 0 },
      } } });
    });
    return true;
  }
}

/** Two tool calls in one assistant message, where the second park arrives during the usage wait. */
class ParallelParkRuntime extends FakeRuntime {
  async sendMessage(sessionId, message) {
    this.sent.push({ sessionId, message });
    const turnId = "turn-1";
    queueMicrotask(() => {
      this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-a", usage: { input_tokens: 100 } }));
      this.parked = { sessionId, turnId };
      this.emit("activity", notification("tool/parked", sessionId, turnId, {
        parkId: "park-1", name: message.dshTools[0].name, arguments: { command: "pwd" },
      }));
      setTimeout(() => {
        this.emit("activity", notification("tool/parked", sessionId, turnId, {
          parkId: "park-2", name: message.dshTools[0].name, arguments: { command: "date" },
        }));
        this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-a", usage: { output_tokens: 20 } }));
        this.emit("activity", notification("usage/apiCallEnded", sessionId, turnId, { callId: "msg-a" }));
      }, 10);
    });
    return { id: turnId, status: "inProgress", items: [] };
  }
}

/** A call that never reports its end, so the only thing that can stop the wait is the abort. */
class SilentCallRuntime extends FakeRuntime {
  async sendMessage(sessionId, message) {
    this.sent.push({ sessionId, message });
    const turnId = "turn-1";
    queueMicrotask(() => {
      this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-a", usage: { input_tokens: 100 } }));
      this.parked = { sessionId, turnId };
      this.emit("activity", notification("tool/parked", sessionId, turnId, {
        parkId: "park-1", name: message.dshTools[0].name, arguments: { command: "pwd" },
      }));
    });
    return { id: turnId, status: "inProgress", items: [] };
  }
}

const bashTool = [{ name: "bash", description: "run", parameters: { type: "object", properties: {} } }];

test("a late usage report is counted once, in the step that receives it", async () => {
  process.env.DSH_AGENT_BRIDGE_USAGE_SETTLE_MS = "30";
  try {
    const runtime = new LateReportRuntime();
    const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
    const agent = fakeAgent();
    adapter.attachAgent(agent);
    const messages = [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }];
    const request = { provider: "claude", model: "sonnet", sessionId: agent.id, messages, tools: bashTool };
    const first = await collect(adapter.stream(request));
    const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");
    const second = await collect(adapter.stream({
      ...request,
      messages: [...messages, { role: "user", content: [{ type: "tool-result", toolCallId: call.block.id, content: [{ type: "text", text: "/workspace" }] }] }],
    }));

    const firstUsage = first.filter(chunk => chunk.type === "usage").map(chunk => chunk.usage);
    const secondUsage = second.filter(chunk => chunk.type === "usage").map(chunk => chunk.usage);
    assert.deepEqual(firstUsage, [{ inputTokens: 100, outputTokens: 1, cacheReadTokens: 0, cacheWriteTokens: 0 }],
      "the step reports what had arrived when its wait ended");
    assert.deepEqual(secondUsage, [{ inputTokens: 200, outputTokens: 49, cacheReadTokens: 0, cacheWriteTokens: 0 }],
      "the late 19 output tokens arrive in step two, and the first call's input is not counted again");
    const total = [...firstUsage, ...secondUsage].reduce((sum, usage) => sum + usage.inputTokens, 0);
    assert.equal(total, 300, "the input across steps is the turn's input, counted once");
  } finally {
    delete process.env.DSH_AGENT_BRIDGE_USAGE_SETTLE_MS;
  }
});

test("tool calls parked while a step waits for its usage are all run", async () => {
  const runtime = new ParallelParkRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);
  const first = await collect(adapter.stream({
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }],
    tools: bashTool,
  }));
  const calls = first.filter(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call").map(chunk => chunk.block.id);
  assert.equal(calls.length, 2, "a second park that arrives during the usage wait joins the same step, instead of being dropped");
  assert.equal(first.at(-1).reason.kind, "tool-calls");
});

test("aborting a step while its usage is pending ends the turn at once", async () => {
  const runtime = new SilentCallRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 50);
  const started = Date.now();
  const chunks = await collect(adapter.stream({
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    signal: controller.signal,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }],
    tools: bashTool,
  }));
  assert.ok(Date.now() - started < 1000, "the usage wait does not hold an aborted turn for its full limit");
  assert.equal(chunks.at(-1).reason.kind, "aborted");
});

/** A step whose second tool call parks only after its model call has ended. */
class LateParkRuntime extends FakeRuntime {
  async sendMessage(sessionId, message) {
    this.sent.push({ sessionId, message });
    const turnId = "turn-1";
    queueMicrotask(() => {
      this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-a", usage: { input_tokens: 100 } }));
      this.emit("activity", notification("usage/toolUses", sessionId, turnId, { callId: "msg-a", toolUseIds: ["t1", "t2"] }));
      this.parked = { sessionId, turnId };
      this.emit("activity", notification("tool/parked", sessionId, turnId, {
        parkId: "park-1", name: message.dshTools[0].name, arguments: { command: "pwd" },
      }));
      setTimeout(() => {
        this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-a", usage: { output_tokens: 20 } }));
        this.emit("activity", notification("usage/apiCallEnded", sessionId, turnId, { callId: "msg-a" }));
      }, 10);
      setTimeout(() => {
        this.emit("activity", notification("tool/parked", sessionId, turnId, {
          parkId: "park-2", name: message.dshTools[0].name, arguments: { command: "date" },
        }));
      }, 40);
    });
    return { id: turnId, status: "inProgress", items: [] };
  }
}

test("a second tool call parked after its model call has ended still joins the step", async () => {
  const runtime = new LateParkRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);
  const first = await collect(adapter.stream({
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }],
    tools: bashTool,
  }));
  const calls = first.filter(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");
  assert.equal(calls.length, 2, "the step waits for every tool call its model call declared, not only for the model call to end");
});

/** Usage that lands while a tool runs, between the handoff and the resume. */
class GapRuntime extends FakeRuntime {
  async sendMessage(sessionId, message) {
    this.sent.push({ sessionId, message });
    const turnId = "turn-1";
    queueMicrotask(() => {
      this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-a", usage: { input_tokens: 100, output_tokens: 1 } }));
      this.parked = { sessionId, turnId };
      this.emit("activity", notification("tool/parked", sessionId, turnId, {
        parkId: "park-1", name: message.dshTools[0].name, arguments: { command: "pwd" },
      }));
    });
    return { id: turnId, status: "inProgress", items: [] };
  }

  resolveToolCall(toolUseId, result) {
    this.resolved.push({ toolUseId, result });
    const parked = this.parked;
    this.parked = null;
    if (!parked) return false;
    queueMicrotask(() => {
      const { sessionId, turnId } = parked;
      this.emit("activity", notification("usage/apiCall", sessionId, turnId, { callId: "msg-b", usage: { input_tokens: 200, output_tokens: 30 } }));
      this.emit("activity", notification("usage/apiCallEnded", sessionId, turnId, { callId: "msg-b" }));
      this.emit("activity", { method: "turn/completed", params: { sessionId, turn: {
        id: turnId,
        status: "completed",
        error: null,
        items: [],
        usage: { inputTokens: 300, outputTokens: 50, cacheReadTokens: 0, cacheWriteTokens: 0 },
      } } });
    });
    return true;
  }
}

test("usage that lands while a tool runs is counted in the step that resumes", async () => {
  process.env.DSH_AGENT_BRIDGE_USAGE_SETTLE_MS = "30";
  try {
    const runtime = new GapRuntime();
    const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
    const agent = fakeAgent();
    adapter.attachAgent(agent);
    const messages = [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "go" }] }];
    const request = { provider: "claude", model: "sonnet", sessionId: agent.id, messages, tools: bashTool };
    const first = await collect(adapter.stream(request));
    const call = first.find(chunk => chunk.type === "block-end" && chunk.block.type === "tool-call");

    // The first call finishes while DSH is running the tool, after the step has handed off.
    const { sessionId } = runtime.sent[0];
    runtime.emit("activity", notification("usage/apiCall", sessionId, "turn-1", { callId: "msg-a", usage: { output_tokens: 20 } }));
    runtime.emit("activity", notification("usage/apiCallEnded", sessionId, "turn-1", { callId: "msg-a" }));

    const second = await collect(adapter.stream({
      ...request,
      messages: [...messages, { role: "user", content: [{ type: "tool-result", toolCallId: call.block.id, content: [{ type: "text", text: "/workspace" }] }] }],
    }));
    const firstUsage = first.filter(chunk => chunk.type === "usage").map(chunk => chunk.usage);
    const secondUsage = second.filter(chunk => chunk.type === "usage").map(chunk => chunk.usage);
    assert.deepEqual(firstUsage, [{ inputTokens: 100, outputTokens: 1, cacheReadTokens: 0, cacheWriteTokens: 0 }]);
    assert.deepEqual(secondUsage, [{ inputTokens: 200, outputTokens: 49, cacheReadTokens: 0, cacheWriteTokens: 0 }],
      "the 19 output tokens that landed during the tool run are reported with the step that resumes");
  } finally {
    delete process.env.DSH_AGENT_BRIDGE_USAGE_SETTLE_MS;
  }
});

test("usage from another adapter's turns on a shared runtime is not recorded here", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  runtime.emit("activity", notification("usage/apiCall", "someone-elses-session", "their-turn", {
    callId: "their-call", usage: { input_tokens: 5 },
  }));
  assert.equal(adapter.usageLedgers.size, 0, "a session this adapter does not own leaves no ledger here");
});

test("usage that arrives after its turn has ended does not bring its ledger back", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);
  const chunks = await collect(adapter.stream({
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "hi" }] }],
  }));
  assert.equal(chunks.at(-1).type, "finish");
  assert.equal(adapter.usageLedgers.size, 0, "a finished turn leaves no ledger");

  const { sessionId } = runtime.sent[0];
  runtime.emit("activity", notification("usage/apiCall", sessionId, "turn-1", { callId: "late", usage: { input_tokens: 9 } }));
  assert.equal(adapter.usageLedgers.size, 0, "a late report for a finished turn is not recorded");
});

test("a turn whose consumer stopped early leaves no ledger once it completes", async () => {
  const runtime = new FakeRuntime();
  const adapter = new ClaudeDshAdapter({ runtime, ready: Promise.resolve() });
  const agent = fakeAgent();
  adapter.attachAgent(agent);
  for await (const chunk of adapter.stream({
    provider: "claude",
    model: "sonnet",
    sessionId: agent.id,
    messages: [{ role: "user", source: { kind: "user" }, content: [{ type: "text", text: "hi" }] }],
  })) {
    assert.ok(chunk, "the first chunk arrives");
    break;
  }
  await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal(adapter.usageLedgers.size, 0, "completion with no live reader drops the ledger");
});
