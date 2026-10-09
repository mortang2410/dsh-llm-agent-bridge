import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { ClaudeCliClient } from "../cli-client.mjs";

test("Claude CLI rejects DSH tools instead of silently omitting them", async () => {
  const client = new ClaudeCliClient();
  await client.createSession({ sessionId: "44444444-4444-4444-8444-444444444444" });
  await assert.rejects(() => client.sendMessage("44444444-4444-4444-8444-444444444444", {
    text: "use the tool",
    dshTools: [{ name: "cross_plugin_probe", description: "probe", parameters: { type: "object" } }],
  }), /cannot expose DSH tools/);
});

test("Claude CLI client uses session, settings, effort, and permission flags", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "claude-sdk-cli-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const recordPath = join(directory, "record.json");
  const scriptPath = join(directory, "fake-claude.mjs");
  await writeFile(scriptPath, `
import { writeFileSync } from "node:fs";
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => { input += chunk; });
process.stdin.on("end", () => {
  writeFileSync(process.argv[2], JSON.stringify({ args: process.argv.slice(3), input }));
  console.log(JSON.stringify({ type: "result", result: "ok" }));
});
`);

  const client = new ClaudeCliClient({ command: process.execPath, args: [scriptPath, recordPath] });
  // A child left waiting on an open stdin would otherwise keep the test process alive after a failure.
  context.after(() => client.close());
  await client.start();
  const session = await client.createSession({
    sessionId: "11111111-1111-4111-8111-111111111111",
    model: "opus",
    effort: "high",
    cwd: directory,
    settingSources: ["user", "project", "local"],
  });
  const completed = onceTurnCompleted(client);
  await client.sendMessage(session.id, { text: "hello", approvalPolicy: "on-request" });
  await completed;

  const { args, input } = JSON.parse(await readFile(recordPath, "utf8"));
  assert.equal(input, "", "a text-only turn writes nothing to stdin");
  assert.deepEqual(args.slice(0, 2), ["-p", "hello"]);
  assertFlag(args, "--output-format", "stream-json");
  assertFlag(args, "--model", "opus");
  assertFlag(args, "--effort", "high");
  assertFlag(args, "--permission-mode", "manual");
  assertFlag(args, "--setting-sources", "user,project,local");
  assertFlag(args, "--session-id", session.id);
});

test("Claude CLI client isolates explicit empty setting sources with safe mode", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "claude-sdk-cli-safe-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const argsPath = join(directory, "args.json");
  const scriptPath = join(directory, "fake-claude.mjs");
  await writeFile(scriptPath, `
import { writeFileSync } from "node:fs";
writeFileSync(process.argv[2], JSON.stringify(process.argv.slice(3)));
console.log(JSON.stringify({ type: "result", result: "ok" }));
`);

  const client = new ClaudeCliClient({ command: process.execPath, args: [scriptPath, argsPath] });
  await client.start();
  const session = await client.createSession({
    sessionId: "22222222-2222-4222-8222-222222222222",
    cwd: directory,
    settingSources: [],
    systemPrompt: "Generate a title.",
  });
  const completed = onceTurnCompleted(client);
  await client.sendMessage(session.id, { text: "title", approvalPolicy: "never", sandbox: "read-only" });
  await completed;

  const args = JSON.parse(await readFile(argsPath, "utf8"));
  assert.equal(args.includes("--safe-mode"), true);
  assert.equal(args.includes("--setting-sources"), false);
  assertFlag(args, "--system-prompt", "Generate a title.");
  assertFlag(args, "--permission-mode", "plan");
});

function onceTurnCompleted(client, timeoutMs = 1_000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("timed out waiting for turn completion")), timeoutMs);
    const onActivity = (message) => {
      if (message.method !== "turn/completed") return;
      clearTimeout(timer);
      client.off("activity", onActivity);
      resolve(message);
    };
    client.on("activity", onActivity);
  });
}

test("the CLI fallback spawns with the configured gateway environment, not process.env", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "claude-sdk-cli-env-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const envPath = join(directory, "env.json");
  const scriptPath = join(directory, "fake-claude-env.mjs");
  await writeFile(scriptPath, `
import { writeFileSync } from "node:fs";
writeFileSync(process.argv[2], JSON.stringify({
  baseUrl: process.env.ANTHROPIC_BASE_URL ?? null,
  hasPath: typeof process.env.PATH === "string" && process.env.PATH.length > 0,
}));
console.log(JSON.stringify({ type: "result", result: "ok" }));
`);

  const env = { ...process.env, ANTHROPIC_BASE_URL: "http://127.0.0.1:8090/anthropic" };
  const client = new ClaudeCliClient({ command: process.execPath, args: [scriptPath, envPath], env });
  await client.start();
  const session = await client.createSession({ sessionId: "33333333-3333-4333-8333-333333333333", cwd: directory });
  const completed = onceTurnCompleted(client);
  await client.sendMessage(session.id, { text: "hello" });
  await completed;

  const seen = JSON.parse(await readFile(envPath, "utf8"));
  assert.equal(seen.baseUrl, "http://127.0.0.1:8090/anthropic");
  assert.equal(seen.hasPath, true, "PATH survives so the child is still a working process");
});

const PNG_BASE64 = Buffer.from([137, 80, 78, 71]).toString("base64");

test("Claude CLI sends pictures on stdin as image blocks, never on the command line", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "claude-sdk-cli-image-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const recordPath = join(directory, "record.json");
  const scriptPath = join(directory, "fake-claude-image.mjs");
  await writeFile(scriptPath, `
import { writeFileSync } from "node:fs";
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => { input += chunk; });
process.stdin.on("end", () => {
  writeFileSync(process.argv[2], JSON.stringify({ args: process.argv.slice(3), input }));
  console.log(JSON.stringify({ type: "result", result: "ok" }));
});
`);

  const client = new ClaudeCliClient({ command: process.execPath, args: [scriptPath, recordPath] });
  await client.start();
  const session = await client.createSession({ sessionId: "55555555-5555-4555-8555-555555555555", cwd: directory });
  const completed = onceTurnCompleted(client);
  await client.sendMessage(session.id, {
    text: "[image: shot.png]",
    images: [{ mediaType: "image/png", data: PNG_BASE64, name: "shot.png" }],
  });
  const { params } = await completed;
  assert.equal(params.turn.status, "completed", "a picture the CLI read completes the turn");

  const { args, input } = JSON.parse(await readFile(recordPath, "utf8"));
  assert.equal(args.includes("[image: shot.png]"), false, "the text is not passed on the command line");
  assertFlag(args, "--input-format", "stream-json");
  const lines = input.trim().split("\n");
  assert.equal(lines.length, 1, "the turn is one user message");
  assert.deepEqual(JSON.parse(lines[0]), {
    type: "user",
    message: {
      role: "user",
      content: [
        { type: "text", text: "[image: shot.png]" },
        { type: "image", source: { type: "base64", media_type: "image/png", data: PNG_BASE64 } },
      ],
    },
    parent_tool_use_id: null,
  });
});

test("Claude CLI sends an image-only turn with no empty text block", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "claude-sdk-cli-image-only-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const recordPath = join(directory, "record.json");
  const scriptPath = join(directory, "fake-claude-image.mjs");
  await writeFile(scriptPath, `
import { writeFileSync } from "node:fs";
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => { input += chunk; });
process.stdin.on("end", () => {
  writeFileSync(process.argv[2], input);
  console.log(JSON.stringify({ type: "result", result: "ok" }));
});
`);

  const client = new ClaudeCliClient({ command: process.execPath, args: [scriptPath, recordPath] });
  await client.start();
  const session = await client.createSession({ sessionId: "77777777-7777-4777-8777-777777777777", cwd: directory });
  const completed = onceTurnCompleted(client);
  await client.sendMessage(session.id, {
    text: "",
    images: [{ mediaType: "image/png", data: PNG_BASE64, name: "shot.png" }],
  });
  await completed;

  const input = await readFile(recordPath, "utf8");
  assert.deepEqual(JSON.parse(input), {
    type: "user",
    message: {
      role: "user",
      content: [{ type: "image", source: { type: "base64", media_type: "image/png", data: PNG_BASE64 } }],
    },
    parent_tool_use_id: null,
  });
});

test("a CLI that exits without reading its picture fails the turn without crashing the host", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "claude-sdk-cli-epipe-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const scriptPath = join(directory, "exit-now.mjs");
  await writeFile(scriptPath, "process.exit(0);\n");

  const client = new ClaudeCliClient({ command: process.execPath, args: [scriptPath] });
  await client.start();
  const session = await client.createSession({ sessionId: "66666666-6666-4666-8666-666666666666", cwd: directory });
  const completed = onceTurnCompleted(client);
  // A picture larger than the pipe buffer, so the write is still pending when the child has gone.
  // An unhandled stdin error would throw here and fail the whole test process; the turn must still complete, as failed.
  await client.sendMessage(session.id, {
    text: "look",
    images: [{ mediaType: "image/png", data: "A".repeat(4_000_000) }],
  });
  const { params } = await completed;
  assert.equal(params.turn.status, "failed");
  assert.match(params.turn.error.message, /did not read its input/);
});

/** A fake CLI that reads its whole stdin, then answers with one result line and exits 0. */
const ANSWER_THEN_EXIT = `
process.stdin.resume();
process.stdin.on("end", () => console.log(JSON.stringify({ type: "result", result: "ok" })));
`;

/** A client whose CLI is the given fake script, run in a fresh directory that is also the session's cwd. */
async function fakeCliClient(context, source) {
  const directory = await mkdtemp(join(tmpdir(), "claude-sdk-cli-fake-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const scriptPath = join(directory, "fake-claude.mjs");
  await writeFile(scriptPath, source);
  const client = new ClaudeCliClient({ command: process.execPath, args: [scriptPath] });
  context.after(() => client.close());
  await client.start();
  const session = await client.createSession({ cwd: directory });
  return { client, session, directory };
}

/** Counts the turn/completed events for one turn, watching briefly after the first so a duplicate is seen. */
async function completionCount(client, send) {
  let count = 0;
  client.on("activity", message => { if (message.method === "turn/completed") count += 1; });
  const completed = onceTurnCompleted(client);
  await send();
  await completed;
  await new Promise(resolve => setTimeout(resolve, 200));
  return count;
}

/** Stops the grandchild a held-pipe fake left running, if it recorded its pid. */
async function killGrandchild(directory) {
  const pid = Number(await readFile(join(directory, "grandchild.pid"), "utf8").catch(() => 0));
  if (!pid) return;
  try { process.kill(pid); } catch { /* it already exited */ }
}

test("a small picture the CLI exits on without reading fails the turn", async (context) => {
  const { client, session } = await fakeCliClient(context, "process.exit(0);\n");
  const completed = onceTurnCompleted(client);
  await client.sendMessage(session.id, { text: "look", images: [{ mediaType: "image/png", data: PNG_BASE64 }] });
  const { params } = await completed;
  assert.equal(params.turn.status, "failed");
  assert.match(params.turn.error.message, /without answering/);
});

test("a small picture the CLI reads and answers completes the turn", async (context) => {
  const { client, session } = await fakeCliClient(context, ANSWER_THEN_EXIT);
  const completed = onceTurnCompleted(client);
  await client.sendMessage(session.id, { text: "look", images: [{ mediaType: "image/png", data: PNG_BASE64 }] });
  const { params } = await completed;
  assert.equal(params.turn.status, "completed");
});

test("a result line still draining when the CLI exits completes the turn", async (context) => {
  // The line is larger than the pipe buffer, so up to one buffer of it can still be unread when the CLI exits.
  const { client, session } = await fakeCliClient(context, `
process.stdin.resume();
process.stdin.on("end", () => {
  const line = JSON.stringify({ type: "result", result: "x".repeat(1_000_000) });
  process.stdout.write(line + "\\n", () => process.exit(0));
});
`);
  const completed = onceTurnCompleted(client);
  await client.sendMessage(session.id, { text: "look", images: [{ mediaType: "image/png", data: PNG_BASE64 }] });
  const { params } = await completed;
  assert.equal(params.turn.status, "completed");
});

test("a result line the CLI writes after it exits still completes a picture turn", async (context) => {
  // The CLI exits at once, but its grandchild writes the answer 300 ms later into the stdout they share.
  // Checking for the answer at exit would fail this turn, so the exit handler must wait for stdout to end.
  const { client, session } = await fakeCliClient(context, `
import { spawn } from "node:child_process";
process.stdin.resume();
process.stdin.on("end", () => {
  const answer = JSON.stringify({ type: "result", result: "ok" }) + String.fromCharCode(10);
  const later = "setTimeout(() => process.stdout.write(" + JSON.stringify(answer) + "), 300)";
  spawn(process.execPath, ["-e", later], { stdio: "inherit" });
  process.exit(0);
});
`);
  const completed = onceTurnCompleted(client, 5_000);
  await client.sendMessage(session.id, { text: "look", images: [{ mediaType: "image/png", data: PNG_BASE64 }] });
  const { params } = await completed;
  assert.equal(params.turn.status, "completed");
});

test("a picture a grandchild keeps open fails the turn within a bounded time", async (context) => {
  const { client, session, directory } = await fakeCliClient(context, `
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
const grandchild = spawn(process.execPath, ["-e", "setTimeout(() => {}, 10000)"], { stdio: "inherit" });
writeFileSync("grandchild.pid", String(grandchild.pid));
process.exit(0);
`);
  // The grandchild holds the picture's stdin and stdout open, so the turn must end without either one closing.
  try {
    const completed = onceTurnCompleted(client, 5_000);
    await client.sendMessage(session.id, { text: "look", images: [{ mediaType: "image/png", data: "A".repeat(4_000_000) }] });
    const { params } = await completed;
    assert.equal(params.turn.status, "failed");
    assert.match(params.turn.error.message, /did not read its input/);
  } finally {
    // Killed here, not in an after-hook: the directory holding the pid file is removed by an earlier after-hook.
    await killGrandchild(directory);
  }
});

test("each turn completes exactly once", async (context) => {
  const text = await fakeCliClient(context, ANSWER_THEN_EXIT);
  assert.equal(await completionCount(text.client, () => text.client.sendMessage(text.session.id, { text: "hello" })), 1, "a text turn");

  const picture = await fakeCliClient(context, ANSWER_THEN_EXIT);
  const sendPicture = () => picture.client.sendMessage(picture.session.id, { text: "look", images: [{ mediaType: "image/png", data: PNG_BASE64 }] });
  assert.equal(await completionCount(picture.client, sendPicture), 1, "a picture turn");

  const missing = new ClaudeCliClient({ command: "/nonexistent/claude-binary" });
  await missing.start();
  const missingSession = await missing.createSession({ cwd: process.cwd() });
  assert.equal(await completionCount(missing, () => missing.sendMessage(missingSession.id, { text: "hello" })), 1, "a spawn failure");

  // Node emits 'error' when a kill fails; the exit that follows must not complete the turn a second time.
  const killed = await fakeCliClient(context, ANSWER_THEN_EXIT);
  const killThenExit = async () => {
    const { id } = await killed.client.sendMessage(killed.session.id, { text: "hello" });
    killed.client.processes.get(id).emit("error", new Error("kill failed"));
  };
  assert.equal(await completionCount(killed.client, killThenExit), 1, "a kill failure, then the exit");
});

function assertFlag(args, flag, value) {
  const index = args.indexOf(flag);
  assert.notEqual(index, -1, `${flag} is present`);
  assert.equal(args[index + 1], value);
}
