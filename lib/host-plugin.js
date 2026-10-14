import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import readline from "node:readline";
import { z } from "zod";
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { LlmAdapter } from "@deepseek-ai/dsh-llm";
//#region internal/plugin-sdk.mjs
const SEMVER_PATTERN = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
const PLUGIN_ID_PATTERN = /^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$/;
const CAPABILITY_ID_PATTERN = /^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$/;
var CapabilityRegistry = class {
	#entries = /* @__PURE__ */ new Map();
	register(name, version, value, providerId) {
		assertCapabilityName(name);
		assertSemanticVersion(version, `capability ${name}`);
		if (this.#entries.has(name)) throw new Error(`capability ${name} is already available`);
		this.#entries.set(name, Object.freeze({
			name,
			version,
			value,
			providerId
		}));
	}
	unregisterProvider(providerId) {
		for (const [name, entry] of this.#entries) if (entry.providerId === providerId) this.#entries.delete(name);
	}
	require(name, range = "*") {
		const entry = this.#entries.get(name);
		if (!entry) throw new Error(`capability ${name} is not available`);
		if (!satisfiesVersion(entry.version, range)) throw new Error(`capability ${name} ${entry.version} does not satisfy ${range}`);
		return entry.value;
	}
	optional(name, range = "*") {
		if (!this.#entries.has(name)) return void 0;
		return this.require(name, range);
	}
};
var PluginHost = class {
	constructor() {
		this.capabilities = new CapabilityRegistry();
		this.active = [];
		this.disposed = false;
	}
	async activate(definitions) {
		if (this.active.length > 0) throw new Error("plugin host is already active");
		if (this.disposed) throw new Error("plugin host is disposed");
		const ordered = resolveActivationOrder(definitions);
		let current = null;
		try {
			for (const definition of ordered) {
				const access = createCapabilityAccess(definition.manifest, this.capabilities);
				const cleanups = [];
				let acceptingCleanups = true;
				const defer = (cleanup) => {
					assert.equal(typeof cleanup, "function", `plugin ${definition.manifest.id} cleanup must be a function`);
					assert.ok(acceptingCleanups, `plugin ${definition.manifest.id} cannot defer cleanup after activation`);
					cleanups.push(cleanup);
					return cleanup;
				};
				current = {
					id: definition.manifest.id,
					cleanups
				};
				let activation;
				try {
					activation = await definition.activate(Object.freeze({
						plugin: definition.manifest,
						capabilities: access,
						defer
					})) ?? {};
				} finally {
					acceptingCleanups = false;
				}
				if (typeof activation.dispose === "function") cleanups.push(activation.dispose);
				const provided = activation.capabilities ?? {};
				validateProvidedCapabilities(definition.manifest, provided);
				for (const [name, version] of Object.entries(definition.manifest.provides)) this.capabilities.register(name, version, provided[name], definition.manifest.id);
				this.active.push(current);
				current = null;
			}
		} catch (error) {
			const rollbackErrors = [];
			if (current) {
				this.capabilities.unregisterProvider(current.id);
				rollbackErrors.push(...await disposeCleanups(current.cleanups));
			}
			rollbackErrors.push(...await this.#drainActive());
			if (rollbackErrors.length > 0) throw new AggregateError([error, ...rollbackErrors], `plugin activation failed: ${error?.message ?? error}; rollback also failed`, { cause: error });
			throw error;
		}
		return this;
	}
	async dispose() {
		if (this.disposed) return;
		this.disposed = true;
		await this.#disposeActive();
	}
	async #disposeActive() {
		const errors = await this.#drainActive();
		if (errors.length === 1) throw errors[0];
		if (errors.length > 1) throw new AggregateError(errors, "multiple plugin cleanup operations failed");
	}
	async #drainActive() {
		const errors = [];
		while (this.active.length > 0) {
			const plugin = this.active.pop();
			try {
				errors.push(...await disposeCleanups(plugin.cleanups));
			} finally {
				this.capabilities.unregisterProvider(plugin.id);
			}
		}
		return errors;
	}
};
async function disposeCleanups(cleanups) {
	const errors = [];
	for (const cleanup of cleanups.reverse()) try {
		await cleanup();
	} catch (error) {
		errors.push(error);
	}
	return errors;
}
function definePlugin(definition) {
	assert.equal(typeof definition?.activate, "function", "plugin activate must be a function");
	const manifest = validateManifest(definition.manifest);
	return Object.freeze({
		manifest,
		activate: definition.activate
	});
}
function validateManifest(input) {
	assert.ok(input && typeof input === "object" && !Array.isArray(input), "plugin manifest is required");
	assert.match(input.id ?? "", PLUGIN_ID_PATTERN, "plugin id must be lowercase and stable");
	assertSemanticVersion(input.version, `plugin ${input.id}`);
	const provides = validateCapabilityMap(input.provides, "provides", { ranges: false });
	const requires = validateCapabilityMap(input.requires, "requires", { ranges: true });
	const optional = validateCapabilityMap(input.optional, "optional", { ranges: true });
	for (const name of Object.keys(requires)) assert.ok(!(name in optional), `capability ${name} cannot be both required and optional`);
	const permissions = input.permissions ?? [];
	assert.ok(Array.isArray(permissions), "plugin permissions must be an array");
	assert.ok(permissions.every((permission) => typeof permission === "string" && permission.length > 0), "plugin permissions must contain non-empty strings");
	return Object.freeze({
		id: input.id,
		version: input.version,
		provides: Object.freeze(provides),
		requires: Object.freeze(requires),
		optional: Object.freeze(optional),
		permissions: Object.freeze([...permissions])
	});
}
function satisfiesVersion(version, range) {
	const current = parseVersion(version);
	if (range === "*" || range === void 0) return true;
	if (SEMVER_PATTERN.test(range)) return compareVersions(current, parseVersion(range)) === 0;
	const majorWildcard = /^(0|[1-9]\d*)\.x$/.exec(range);
	if (majorWildcard) return current.major === Number(majorWildcard[1]);
	if (range.startsWith("^")) {
		const minimum = parseVersion(range.slice(1));
		const upper = minimum.major > 0 ? {
			major: minimum.major + 1,
			minor: 0,
			patch: 0
		} : minimum.minor > 0 ? {
			major: 0,
			minor: minimum.minor + 1,
			patch: 0
		} : {
			major: 0,
			minor: 0,
			patch: minimum.patch + 1
		};
		return compareVersions(current, minimum) >= 0 && compareVersions(current, upper) < 0;
	}
	throw new Error(`unsupported semantic version range ${range}`);
}
function resolveActivationOrder(definitions) {
	assert.ok(Array.isArray(definitions), "plugin definitions must be an array");
	const plugins = /* @__PURE__ */ new Map();
	const providers = /* @__PURE__ */ new Map();
	for (const definition of definitions) {
		assert.ok(definition?.manifest && typeof definition.activate === "function", "invalid plugin definition");
		const manifest = validateManifest(definition.manifest);
		if (plugins.has(manifest.id)) throw new Error(`duplicate plugin id ${manifest.id}`);
		plugins.set(manifest.id, definition);
		for (const [name, version] of Object.entries(manifest.provides)) {
			if (providers.has(name)) throw new Error(`capability ${name} is provided by both ${providers.get(name).id} and ${manifest.id}`);
			providers.set(name, {
				id: manifest.id,
				version
			});
		}
	}
	const dependencies = new Map([...plugins.keys()].map((id) => [id, /* @__PURE__ */ new Set()]));
	for (const definition of plugins.values()) {
		const { manifest } = definition;
		for (const [name, range] of Object.entries(manifest.requires)) {
			const provider = providers.get(name);
			if (!provider || !satisfiesVersion(provider.version, range)) {
				const found = provider ? ` (found ${provider.version})` : "";
				throw new Error(`plugin ${manifest.id} requires ${name} ${range}${found}`);
			}
			dependencies.get(manifest.id).add(provider.id);
		}
		for (const [name, range] of Object.entries(manifest.optional)) {
			const provider = providers.get(name);
			if (!provider) continue;
			if (!satisfiesVersion(provider.version, range)) throw new Error(`plugin ${manifest.id} optional capability ${name} requires ${range} (found ${provider.version})`);
			dependencies.get(manifest.id).add(provider.id);
		}
	}
	const ordered = [];
	const visiting = /* @__PURE__ */ new Set();
	const visited = /* @__PURE__ */ new Set();
	const visit = (id) => {
		if (visiting.has(id)) throw new Error(`plugin dependency cycle includes ${id}`);
		if (visited.has(id)) return;
		visiting.add(id);
		for (const dependency of dependencies.get(id)) visit(dependency);
		visiting.delete(id);
		visited.add(id);
		ordered.push(plugins.get(id));
	};
	for (const id of plugins.keys()) visit(id);
	return ordered;
}
function createCapabilityAccess(manifest, registry) {
	return Object.freeze({
		require(name) {
			const range = manifest.requires[name];
			if (!range) throw new Error(`plugin ${manifest.id} did not declare required capability ${name}`);
			return registry.require(name, range);
		},
		optional(name) {
			const range = manifest.optional[name];
			if (!range) throw new Error(`plugin ${manifest.id} did not declare optional capability ${name}`);
			return registry.optional(name, range);
		}
	});
}
function validateProvidedCapabilities(manifest, provided) {
	assert.ok(provided && typeof provided === "object" && !Array.isArray(provided), `plugin ${manifest.id} capabilities must be an object`);
	const expected = Object.keys(manifest.provides).sort();
	const actual = Object.keys(provided).sort();
	assert.deepEqual(actual, expected, `plugin ${manifest.id} provided capabilities do not match its manifest`);
	for (const name of expected) assert.notEqual(provided[name], void 0, `plugin ${manifest.id} did not provide ${name}`);
}
function validateCapabilityMap(input, label, { ranges }) {
	const map = input ?? {};
	assert.ok(map && typeof map === "object" && !Array.isArray(map), `plugin ${label} must be an object`);
	const result = {};
	for (const [name, version] of Object.entries(map)) {
		assertCapabilityName(name);
		if (ranges) satisfiesVersion("0.0.0", version);
		else assertSemanticVersion(version, `capability ${name}`);
		result[name] = version;
	}
	return result;
}
function assertCapabilityName(name) {
	assert.match(name ?? "", CAPABILITY_ID_PATTERN, "capability id must be lowercase and stable");
}
function assertSemanticVersion(version, label) {
	assert.match(version ?? "", SEMVER_PATTERN, `${label} must use a semantic version`);
}
function parseVersion(version) {
	assertSemanticVersion(version, "version");
	const [, major, minor, patch] = SEMVER_PATTERN.exec(version);
	return {
		major: Number(major),
		minor: Number(minor),
		patch: Number(patch)
	};
}
function compareVersions(left, right) {
	return left.major - right.major || left.minor - right.minor || left.patch - right.patch;
}
//#endregion
//#region cli-client.mjs
const DEFAULT_MODELS$2 = [
	{
		id: "opus",
		displayName: "Opus 5",
		isDefault: true
	},
	{
		id: "fable",
		displayName: "Fable 5",
		isDefault: false
	},
	{
		id: "sonnet",
		displayName: "Sonnet 5",
		isDefault: false
	},
	{
		id: "haiku",
		displayName: "Haiku 4.5",
		isDefault: false
	}
];
/**
* How long a picture turn waits for its CLI's stdout after the CLI exits. The output is already in the pipe at exit,
* so draining it is local I/O that takes milliseconds even for a multi-megabyte line. 2 s covers a loaded host and
* bounds the delay a grandchild holding stdout open can cause.
*/
const STDOUT_GRACE_MS = 2e3;
/** Pass --effort only when a level was chosen, so the CLI's own default stands. */
function effortArgs(message, session) {
	const effort = message.effort ?? session.config?.effort;
	return effort ? ["--effort", effort] : [];
}
var ClaudeCliClient = class extends EventEmitter {
	constructor({ command = "claude", args = [], env = process.env, requestTimeoutMs = 30 * 6e4 } = {}) {
		super();
		this.command = command;
		this.args = args;
		this.env = env;
		this.requestTimeoutMs = requestTimeoutMs;
		this.sessions = /* @__PURE__ */ new Map();
		this.processes = /* @__PURE__ */ new Map();
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
		return Promise.reject(/* @__PURE__ */ new Error("The Claude CLI fallback cannot host harness tools; the Agent SDK backend is required"));
	}
	resolveToolCall() {
		return false;
	}
	rejectToolCall() {
		return false;
	}
	rejectAllToolCalls() {}
	async listModels() {
		return DEFAULT_MODELS$2;
	}
	async createSession(config = {}) {
		const id = config.sessionId ?? randomUUID();
		const session = {
			id,
			cwd: config.cwd ?? process.cwd(),
			created: false,
			config: structuredClone(config)
		};
		this.sessions.set(id, session);
		return {
			id,
			cwd: session.cwd,
			turns: []
		};
	}
	async resumeSession(sessionId, config = {}) {
		const existing = this.sessions.get(sessionId) ?? {
			id: sessionId,
			created: true
		};
		const session = {
			...existing,
			cwd: config.cwd ?? existing.cwd ?? process.cwd(),
			config: {
				...existing.config,
				...config
			}
		};
		this.sessions.set(sessionId, session);
		return {
			id: sessionId,
			cwd: session.cwd,
			turns: []
		};
	}
	async sendMessage(sessionId, message = {}) {
		if (Array.isArray(message.dshTools) && message.dshTools.length > 0) throw new Error("The Claude CLI backend cannot expose DSH tools; use the Claude Agent SDK backend");
		const session = this.sessions.get(sessionId) ?? await this.resumeSession(sessionId, message);
		const turnId = randomUUID();
		const child = this.spawnTurn(session, turnId, message);
		this.processes.set(turnId, child);
		return {
			id: turnId,
			status: "inProgress",
			items: []
		};
	}
	async interruptTurn(_sessionId, turnId) {
		const child = this.processes.get(turnId);
		if (!child) return;
		child.kill("SIGTERM");
	}
	async releaseSession(sessionId) {
		for (const [turnId, child] of this.processes) if (child.relayClaudeSessionId === sessionId) {
			child.kill("SIGTERM");
			this.processes.delete(turnId);
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
		const withImages = Array.isArray(message.images) && message.images.length > 0;
		const cliArgs = [
			...this.args,
			"-p",
			...withImages ? ["--input-format", "stream-json"] : [message.text],
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
			...session.created ? ["--resume", session.id] : ["--session-id", session.id]
		];
		const child = spawn(this.command, cliArgs, {
			cwd: message.cwd ?? session.cwd ?? process.cwd(),
			stdio: [
				withImages ? "pipe" : "ignore",
				"pipe",
				"pipe"
			],
			env: this.env
		});
		child.relayClaudeSessionId = session.id;
		const inputWritten = withImages ? new Promise((resolve) => {
			child.stdin.once("finish", () => resolve(null));
			child.stdin.once("close", () => resolve(/* @__PURE__ */ new Error("the input closed before the picture was written")));
			child.stdin.on("error", (error) => {
				this.emit("diagnostic", `Claude CLI did not read its input: ${error.message}`);
				resolve(error);
			});
			child.stdin.end(`${imageInputLine(message)}\n`);
		}) : Promise.resolve(null);
		session.created = true;
		const state = {
			textItemId: null,
			text: "",
			activities: /* @__PURE__ */ new Set(),
			answered: false
		};
		const output = readline.createInterface({ input: child.stdout });
		output.on("line", (line) => this.handleLine(session.id, turnId, line, state));
		const outputEnded = new Promise((resolve) => output.once("close", resolve));
		child.stderr.setEncoding("utf8");
		child.stderr.on("data", (chunk) => this.emit("diagnostic", String(chunk)));
		let settled = false;
		const complete = (status, error) => {
			if (settled) return;
			settled = true;
			this.completeTurn(session.id, turnId, status, error);
		};
		child.once("error", (error) => {
			this.emit("diagnostic", `Claude CLI failed: ${error.message}`);
			complete("failed", error);
		});
		child.once("exit", async (code, signal) => {
			this.processes.delete(turnId);
			const inputError = await inputWritten;
			if (withImages) {
				let timer;
				await Promise.race([outputEnded, new Promise((resolve) => {
					timer = setTimeout(resolve, STDOUT_GRACE_MS);
				})]);
				clearTimeout(timer);
			}
			if (signal || code) complete("failed", /* @__PURE__ */ new Error(`claude exited (${signal ?? code})`));
			else if (inputError) complete("failed", /* @__PURE__ */ new Error(`Claude CLI did not read its input: ${inputError.message}`));
			else if (withImages && !state.answered) complete("failed", /* @__PURE__ */ new Error("Claude CLI exited without answering the picture"));
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
		for (const event of normalizeClaudeStreamMessage(message, state)) this.emit("activity", {
			method: event.method,
			params: {
				sessionId,
				turnId,
				...event.params
			}
		});
	}
	emitText(sessionId, turnId, state, text) {
		if (!text) return;
		state.textItemId ??= `answer-${turnId}`;
		this.emit("activity", {
			method: "item/agentMessage/delta",
			params: {
				sessionId,
				turnId,
				itemId: state.textItemId,
				delta: `${text}\n`
			}
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
					items: []
				}
			}
		});
	}
};
function normalizeClaudeStreamMessage(message, state) {
	const events = [];
	const content = message.message?.content ?? message.content ?? [];
	for (const block of Array.isArray(content) ? content : []) {
		if (block.type === "text" && block.text) {
			state.textItemId ??= block.id ?? `answer-${message.message?.id ?? "latest"}`;
			const delta = block.text.startsWith(state.text) ? block.text.slice(state.text.length) : block.text;
			state.text = block.text;
			if (delta) events.push({
				method: "item/agentMessage/delta",
				params: {
					itemId: state.textItemId,
					delta
				}
			});
		}
		if ((block.type === "text_delta" || block.type === "content_block_delta") && (block.text ?? block.delta?.text)) {
			state.textItemId ??= block.id ?? `answer-${message.message?.id ?? "latest"}`;
			const delta = block.text ?? block.delta.text;
			state.text += delta;
			events.push({
				method: "item/agentMessage/delta",
				params: {
					itemId: state.textItemId,
					delta
				}
			});
		}
		if (block.type === "thinking" && block.thinking) events.push({
			method: "item/reasoning/summaryTextDelta",
			params: {
				itemId: block.id ?? "reasoning",
				delta: block.thinking
			}
		});
		if (block.type === "tool_use") {
			const item = {
				type: "toolUse",
				id: block.id ?? block.name,
				name: block.name,
				input: block.input,
				status: "inProgress"
			};
			if (!state.activities.has(item.id)) {
				state.activities.add(item.id);
				events.push({
					method: "item/started",
					params: { item }
				});
			}
		}
		if (block.type === "tool_result") events.push({
			method: "item/completed",
			params: { item: {
				type: "toolUse",
				id: block.tool_use_id ?? block.id,
				name: block.name,
				output: block.content,
				status: block.is_error ? "failed" : "completed"
			} }
		});
	}
	if (message.type === "result" && message.result) {
		state.textItemId ??= `answer-${message.session_id ?? "latest"}`;
		const delta = String(message.result).startsWith(state.text) ? String(message.result).slice(state.text.length) : String(message.result);
		state.text = String(message.result);
		if (delta) events.push({
			method: "item/agentMessage/delta",
			params: {
				itemId: state.textItemId,
				delta
			}
		});
	}
	return events;
}
/** One user message for the CLI's stdin: the text, then one base64 image block per picture. */
function imageInputLine(message) {
	const content = [];
	if (message.text) content.push({
		type: "text",
		text: message.text
	});
	for (const image of message.images) content.push({
		type: "image",
		source: {
			type: "base64",
			media_type: image.mediaType,
			data: image.data
		}
	});
	return JSON.stringify({
		type: "user",
		message: {
			role: "user",
			content
		},
		parent_tool_use_id: null
	});
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
//#endregion
//#region orchestration.js
/**
* Orchestration policy for the harness's own workflow tool.
*
* DSH ships `workflow` — a tool that runs a script fanning work out across many
* subagents — and this plugin bridges it like any other tool. Claude Code gates
* its own equivalent behind an explicit opt-in, because a fan-out can spawn
* dozens of agents and spend accordingly. DSH's tool carries no such gate, and
* a plain xhigh turn was observed calling it twice on an ordinary audit
* request, so the gate is supplied here.
*
* The wording mirrors Claude Code's own policy for the same reason the models
* follow it there: it states the cost, defines what counts as opting in, and
* says plainly what to do instead. Two halves, matching that structure —
*
* - by default, orchestration must be asked for rather than inferred;
* - under the Ultracode effort, the opt-in is standing.
*
* Both are appended to the harness's own system prompt; neither replaces any
* part of it.
*/
/** The effort id that turns standing orchestration on. */
const ULTRACODE_EFFORT = "ultracode";
/** What the SDK is actually asked for when Ultracode is selected. */
const ULTRACODE_BASE_EFFORT = "xhigh";
/** The harness tools this policy governs. */
const WORKFLOW_TOOL = "mcp__dsh__workflow";
const SUBAGENT_TOOL = "mcp__dsh__subagent";
/** Advisory ceiling, matching the default guideline Claude Code carries. */
const SIZE_GUIDELINE = "Keep workflows under 15 agents unless the user's prompt calls for a different scale. This is a guideline, not a hard limit.";
const GATED = [
	`ONLY call ${WORKFLOW_TOOL} when the user has explicitly opted into multi-agent orchestration. Workflows can spawn dozens of agents and consume a large amount of tokens; the user must request that scale, not have it inferred. Explicit opt-in means one of:`,
	"",
	`- The user selected the Ultracode effort for this session's model.`,
	`- The user directly asked you to run a workflow or use multi-agent orchestration in their own words ("use a workflow", "fan out agents", "orchestrate this with subagents"). The ask must be in the user's words — a task that would merely benefit from a workflow does not count.`,
	`- The user invoked a skill or command whose instructions tell you to run one.`,
	"",
	`For any other task — even one that would clearly benefit from parallelism — do NOT call it. Use ${SUBAGENT_TOOL} for an individual subagent, or briefly describe what a multi-agent workflow could do and roughly what it would cost, and ask the user whether to run it. Mention that selecting the Ultracode effort opts in for the rest of the session.`,
	"",
	SIZE_GUIDELINE
].join("\n");
const STANDING = [
	`The user selected the Ultracode effort, so the opt-in to multi-agent orchestration is standing: author and run a workflow through ${WORKFLOW_TOOL} for every substantive task by default. The goal is the most exhaustive, correct answer you can produce — token cost is not a constraint.`,
	"",
	`For multi-phase work (understand → design → implement → review), that often means several workflows in sequence, one per phase, so the user stays in the loop between them. Decompose the work, fan it out across subagents, and verify findings adversarially before reporting them. Solo only on conversational turns and trivial mechanical edits.`,
	"",
	SIZE_GUIDELINE
].join("\n");
/** Whether the harness offered a workflow tool for this turn. */
function hasWorkflowTool(tools) {
	return (tools ?? []).some((tool) => tool?.name === "workflow");
}
/**
* The policy paragraph for one turn, or null when there is nothing to govern.
*
* @param tools - the harness tools offered this turn.
* @param ultracode - whether the Ultracode effort is selected.
*/
function orchestrationPolicy(tools, ultracode) {
	if (!hasWorkflowTool(tools)) return null;
	return ultracode ? STANDING : GATED;
}
/** Append the policy to the harness's own prompt, leaving it otherwise intact. */
function withOrchestrationPolicy(systemPrompt, policy) {
	if (!policy) return systemPrompt;
	if (typeof systemPrompt !== "string" || !systemPrompt.trim()) return policy;
	return `${systemPrompt}\n\n${policy}`;
}
//#endregion
//#region plan-usage.js
function windowFrom(kind, reported, scope) {
	if (!reported) return null;
	const utilization = reported.utilization;
	if (typeof utilization !== "number" || !Number.isFinite(utilization)) return null;
	const resetsAt = Date.parse(reported.resets_at ?? "");
	return {
		kind,
		...scope ? { scope } : {},
		usedPercent: Math.min(100, Math.max(0, utilization)),
		...Number.isFinite(resetsAt) ? { resetsAt } : {}
	};
}
/**
* Project the SDK's usage response onto the ring's model.
*
* @param response - what `usage_EXPERIMENTAL_MAY_CHANGE_DO_NOT_RELY_ON_THIS_API_YET` returned.
* @returns `{ supported, plan?, windows? }`; unsupported when no plan limits apply
*   (an API key, Bedrock, Vertex), which is a fact about the session, not a failure.
*/
function planUsageFrom(response) {
	if (!response || response.rate_limits_available !== true || !response.rate_limits) return { supported: false };
	const limits = response.rate_limits;
	const windows = [
		windowFrom("session", limits.five_hour),
		windowFrom("weekly", limits.seven_day),
		windowFrom("weekly", limits.seven_day_opus, "Opus"),
		windowFrom("weekly", limits.seven_day_sonnet, "Sonnet"),
		...(limits.model_scoped ?? []).map((bucket) => windowFrom("weekly", bucket, bucket?.display_name))
	].filter(Boolean);
	return {
		supported: windows.length > 0,
		...typeof response.subscription_type === "string" && response.subscription_type ? { plan: response.subscription_type } : {},
		windows
	};
}
//#endregion
//#region sdk-client.mjs
/**
* Only a fallback: the live catalog comes from the SDK's supportedModels(), so
* new model families appear without a release here.
*/
/** How long one plan reading serves every asker on the host. */
const PLAN_USAGE_TTL_MS = 12e4;
const DEFAULT_MODELS$1 = [
	{
		id: "opus",
		displayName: "Opus 5",
		isDefault: true,
		supportedReasoningEfforts: reasoningEfforts()
	},
	{
		id: "fable",
		displayName: "Fable 5",
		isDefault: false,
		supportedReasoningEfforts: reasoningEfforts()
	},
	{
		id: "sonnet",
		displayName: "Sonnet 5",
		isDefault: false,
		supportedReasoningEfforts: reasoningEfforts()
	},
	{
		id: "haiku",
		displayName: "Haiku 4.5",
		isDefault: false,
		supportedReasoningEfforts: reasoningEfforts()
	}
];
function reasoningEfforts(levels = [
	"low",
	"medium",
	"high",
	"xhigh",
	"max"
]) {
	return levels.map((reasoningEffort) => ({ reasoningEffort }));
}
/** Ultracode rides on xhigh, so it is offered only where xhigh is. */
function withUltracode(levels) {
	return levels.includes("xhigh") ? [...levels, ULTRACODE_EFFORT] : levels;
}
function stripTag(value) {
	return String(value ?? "").replace(/\[[^\]]*\]$/, "");
}
/**
* Anthropic's usage shape onto the harness's. Cache reads and writes stay
* apart because the harness reports a cache-hit rate from them, and thinking
* tokens are carried so reasoning is not billed as invisible output.
*/
function tokenUsage(usage) {
	if (!usage) return null;
	const thinking = usage.output_tokens_details?.thinking_tokens;
	return {
		inputTokens: usage.input_tokens ?? 0,
		outputTokens: usage.output_tokens ?? 0,
		...Number.isSafeInteger(usage.cache_read_input_tokens) ? { cacheReadTokens: usage.cache_read_input_tokens } : {},
		...Number.isSafeInteger(usage.cache_creation_input_tokens) ? { cacheWriteTokens: usage.cache_creation_input_tokens } : {},
		...Number.isSafeInteger(thinking) ? { reasoningTokens: thinking } : {}
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
	const recommended = rows.find((row) => row.value === "default")?.resolvedModel;
	const named = rows.filter((row) => row.value !== "default");
	const seen = /* @__PURE__ */ new Set();
	const models = [];
	for (const row of named) {
		const key = row.resolvedModel ?? row.value;
		if (seen.has(key)) continue;
		seen.add(key);
		const levels = row.supportsEffort === false ? [] : row.supportedEffortLevels ?? [];
		models.push({
			id: row.value,
			resolvedModel: row.resolvedModel,
			displayName: modelName(row),
			isDefault: recommended !== void 0 && row.resolvedModel === recommended,
			...levels.length ? { supportedReasoningEfforts: reasoningEfforts(withUltracode(levels)) } : {}
		});
	}
	if (models.length && !models.some((model) => model.isDefault)) models[0].isDefault = true;
	disambiguate(models, named);
	return models;
}
/** `claude-haiku-4-5-20251001` -> `Haiku 4.5`; `claude-opus-5[1m]` -> `Opus 5`. */
function modelName(row) {
	const parts = (row.resolvedModel ?? row.value).replace(/\[[^\]]*\]$/, "").replace(/^claude-/, "").replace(/-\d{8}$/, "").split("-");
	const family = parts.shift() ?? "";
	if (!family) return row.displayName ?? row.value;
	const version = parts.filter((part) => /^\d+$/.test(part)).join(".");
	return `${family.charAt(0).toUpperCase()}${family.slice(1)}${version ? ` ${version}` : ""}`;
}
/** Two rows of one family (e.g. `opus` and `opus[1m]`) must stay distinguishable. */
function disambiguate(models, rows) {
	const counts = /* @__PURE__ */ new Map();
	for (const model of models) counts.set(model.displayName, (counts.get(model.displayName) ?? 0) + 1);
	for (const model of models) {
		if (counts.get(model.displayName) === 1) continue;
		const row = rows.find((candidate) => candidate.value === model.id);
		const tag = /\[([^\]]+)\]$/.exec(row?.resolvedModel ?? row?.value ?? "")?.[1];
		if (tag) model.displayName = `${model.displayName} (${tag.toUpperCase()})`;
	}
}
var ClaudeSdkClient = class extends EventEmitter {
	constructor({ sdk = null, pathToClaudeCodeExecutable = void 0, env = void 0, requestTimeoutMs = 30 * 6e4, clock = Date.now } = {}) {
		super();
		this.sdk = sdk;
		this.pathToClaudeCodeExecutable = pathToClaudeCodeExecutable;
		this.env = env;
		this.requestTimeoutMs = requestTimeoutMs;
		this.clock = clock;
		this.sessions = /* @__PURE__ */ new Map();
		this.queries = /* @__PURE__ */ new Map();
		this.pendingRequests = /* @__PURE__ */ new Map();
		this.pendingToolCalls = /* @__PURE__ */ new Map();
		this.contextWindows = /* @__PURE__ */ new Map();
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
			this.pendingToolCalls.set(parkId, {
				resolve,
				reject
			});
		});
		this.emit("activity", {
			method: "tool/parked",
			params: {
				sessionId,
				turnId,
				parkId,
				name,
				arguments: args ?? {}
			}
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
			if (usage?.canonicalModel && Number.isSafeInteger(usage?.contextWindow)) this.contextWindows.set(usage.canonicalModel, usage.contextWindow);
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
		const id = String(resolvedModel ?? model);
		if (/\[1m\]$/.test(id)) return 1e6;
		return /^claude-(?:haiku|sonnet|opus)-[1-4](?:\D|$)|^claude-[1-4]/.test(id) ? 2e5 : 1e6;
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
		if (this.planUsageCache && now - this.planUsageCache.at < PLAN_USAGE_TTL_MS) return this.planUsageCache.value;
		if (this.planUsageInflight) return this.planUsageInflight;
		this.planUsageInflight = this.readPlanUsage().catch((error) => {
			this.emit("diagnostic", `Claude plan usage unavailable: ${error?.message ?? error}`);
			return { supported: false };
		}).then((value) => {
			this.planUsageCache = {
				at: this.clock(),
				value
			};
			this.planUsageInflight = null;
			return value;
		});
		return this.planUsageInflight;
	}
	async readPlanUsage() {
		if (!this.sdk) await this.start();
		if (typeof this.sdk.query !== "function") return { supported: false };
		let release;
		const idle = new Promise((resolve) => {
			release = resolve;
		});
		const prompt = (async function* () {
			await idle;
		})();
		const query = this.sdk.query({
			prompt,
			options: {
				...this.pathToClaudeCodeExecutable ? { pathToClaudeCodeExecutable: this.pathToClaudeCodeExecutable } : {},
				...this.env ? { env: this.env } : {}
			}
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
		return live?.length ? live : DEFAULT_MODELS$1;
	}
	/**
	* Open a control-only query — streaming input that never yields, so no turn
	* runs — purely to read the catalog the installed Claude Code advertises.
	*/
	async fetchSupportedModels() {
		if (!this.sdk) await this.start();
		if (typeof this.sdk.query !== "function") return null;
		let release;
		const idle = new Promise((resolve) => {
			release = resolve;
		});
		const prompt = (async function* () {
			await idle;
		})();
		const query = this.sdk.query({
			prompt,
			options: {
				...this.pathToClaudeCodeExecutable ? { pathToClaudeCodeExecutable: this.pathToClaudeCodeExecutable } : {},
				...this.env ? { env: this.env } : {}
			}
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
		this.sessions.set(id, {
			id,
			cwd: config.cwd ?? process.cwd(),
			created: false,
			config: structuredClone(config)
		});
		return {
			id,
			cwd: config.cwd ?? process.cwd(),
			turns: []
		};
	}
	async resumeSession(sessionId, config = {}) {
		const existing = this.sessions.get(sessionId) ?? {
			id: sessionId,
			created: true,
			config: {}
		};
		this.sessions.set(sessionId, {
			...existing,
			cwd: config.cwd ?? existing.cwd ?? process.cwd(),
			config: {
				...existing.config,
				...structuredClone(config)
			}
		});
		return {
			id: sessionId,
			cwd: config.cwd ?? existing.cwd ?? process.cwd(),
			turns: []
		};
	}
	async sendMessage(sessionId, message = {}) {
		const session = this.sessions.get(sessionId) ?? await this.resumeSession(sessionId, message);
		const turnId = randomUUID();
		const abortController = new AbortController();
		const options = this.queryOptions(session, message, abortController, turnId);
		const query = this.sdk.query({
			prompt: promptFor(message),
			options
		});
		this.queries.set(turnId, {
			query,
			abortController,
			sessionId
		});
		this.consumeQuery(session, turnId, query).catch((error) => {
			this.emit("diagnostic", `Claude SDK query failed: ${error?.stack ?? error}`);
			this.completeTurn(session.id, turnId, "failed", error);
		});
		session.created = true;
		return {
			id: turnId,
			status: "inProgress",
			items: []
		};
	}
	async interruptTurn(_sessionId, turnId) {
		const record = this.queries.get(turnId);
		if (!record) return;
		await record.query.interrupt?.().catch(() => {});
		record.abortController.abort();
		record.query.close?.();
	}
	async releaseSession(sessionId) {
		for (const [turnId, record] of this.queries) if (record.sessionId === sessionId) {
			record.abortController.abort();
			record.query.close?.();
			this.queries.delete(turnId);
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
		for (const request of this.pendingRequests.values()) request.resolve({
			behavior: "deny",
			message: "Relay Claude SDK client closed"
		});
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
		request.resolve({
			behavior: "deny",
			message: error?.message ?? String(error)
		});
	}
	queryOptions(session, message, abortController, turnId) {
		const bridge = dshMcpOptions(this.sdk, message, (name, args) => this.parkToolCall(session.id, turnId, name, args));
		return {
			abortController,
			cwd: message.cwd ?? session.cwd ?? process.cwd(),
			model: message.model ?? session.config?.model,
			...effortOf(message, session) !== void 0 ? { effort: effortOf(message, session) } : {},
			permissionMode: sdkPermissionMode(message),
			...message.thinkingSummaries === false ? {} : { thinking: {
				type: "adaptive",
				display: "summarized"
			} },
			...isUltracode(message, session) ? { settings: {
				ultracode: true,
				enableWorkflows: true
			} } : {},
			tools: [],
			settingSources: message.settingSources ?? session.config?.settingSources ?? [],
			systemPrompt: withOrchestrationPolicy(message.systemPrompt ?? session.config?.systemPrompt, orchestrationPolicy(message.dshTools, isUltracode(message, session))),
			pathToClaudeCodeExecutable: this.pathToClaudeCodeExecutable,
			...this.env ? { env: this.env } : {},
			includePartialMessages: true,
			...session.created ? { resume: session.id } : { sessionId: session.id },
			...bridge.allowedTools ? {} : { canUseTool: (toolName, input, options) => this.requestPermission(session.id, toolName, input, options) },
			...bridge
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
					suggestions: structuredClone(options.suggestions ?? [])
				}
			};
			this.pendingRequests.set(String(id), {
				request,
				resolve,
				input
			});
			options.signal?.addEventListener("abort", () => this.rejectRequest(id, /* @__PURE__ */ new Error("Claude permission request was cancelled")), { once: true });
			this.emit("request", request);
		});
	}
	async consumeQuery(session, turnId, query) {
		const state = {
			currentMessageId: null,
			text: /* @__PURE__ */ new Map(),
			reasoning: /* @__PURE__ */ new Map(),
			activities: /* @__PURE__ */ new Set()
		};
		let completed = false;
		try {
			for await (const message of query) {
				for (const event of normalizeSdkMessage(message, state)) this.emit("activity", {
					method: event.method,
					params: {
						sessionId: session.id,
						turnId,
						...event.params
					}
				});
				if (message.type === "result") {
					completed = true;
					this.learnContextWindows(message.modelUsage);
					this.completeTurn(session.id, turnId, message.is_error ? "failed" : "completed", resultError(message), tokenUsage(message.usage));
				}
			}
			if (!completed) this.completeTurn(session.id, turnId, "completed");
		} finally {
			this.queries.delete(turnId);
			try {
				const closing = query.close?.();
				if (typeof closing?.then === "function") closing.then(void 0, (error) => this.emit("diagnostic", `Claude SDK query close failed: ${error?.stack ?? error}`));
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
					...usage ? { usage } : {}
				}
			}
		});
	}
};
function dshMcpOptions(sdk, message, park) {
	const strict = { strictMcpConfig: true };
	const schemas = message.dshTools;
	if (!Array.isArray(schemas) || schemas.length === 0) return strict;
	if (typeof sdk.createSdkMcpServer !== "function" || typeof sdk.tool !== "function") throw new Error("This Claude Agent SDK does not support in-process DSH tools");
	const tools = schemas.map((schema) => sdk.tool(schema.name, schema.description, jsonSchemaShape(schema.parameters), async (args) => park(schema.name, args)));
	return {
		...strict,
		mcpServers: { dsh: sdk.createSdkMcpServer({
			name: "dsh",
			version: "1.0.0",
			tools,
			alwaysLoad: true
		}) },
		allowedTools: schemas.map((schema) => `mcp__dsh__${schema.name}`),
		toolAliases: Object.fromEntries(schemas.map((schema) => [`${schema.name.charAt(0).toUpperCase()}${schema.name.slice(1)}`, `mcp__dsh__${schema.name}`]))
	};
}
function jsonSchemaShape(schema) {
	if (!schema || schema.type !== "object" || typeof schema.properties !== "object" || schema.properties === null) {
		if (schema?.type === "object" && schema.properties === void 0) return {};
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
function promptFor(message) {
	const images = Array.isArray(message.images) ? message.images : [];
	if (images.length === 0) return message.text;
	const content = [];
	if (message.text) content.push({
		type: "text",
		text: message.text
	});
	for (const image of images) content.push({
		type: "image",
		source: {
			type: "base64",
			media_type: image.mediaType,
			data: image.data
		}
	});
	return (async function* () {
		yield {
			type: "user",
			message: {
				role: "user",
				content
			},
			parent_tool_use_id: null
		};
	})();
}
function dshToolResult(result) {
	const content = (result.content ?? []).map((block) => {
		if (block?.type === "text") return {
			type: "text",
			text: String(block.text ?? "")
		};
		if (block?.type === "image" && typeof block.data === "string" && typeof block.mediaType === "string") return {
			type: "image",
			data: block.data,
			mimeType: block.mediaType
		};
		return {
			type: "text",
			text: JSON.stringify(block)
		};
	});
	if (content.length === 0) content.push({
		type: "text",
		text: result.isError ? "DSH tool failed" : "DSH tool completed."
	});
	return {
		content,
		isError: Boolean(result.isError)
	};
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
		if (event?.type === "message_delta") events.push(...usageEvents(state.currentMessageId, event.usage));
		if (event?.type === "content_block_start" && event.content_block?.type === "tool_use" && event.content_block.id && state.currentMessageId) events.push({
			method: "usage/toolUses",
			params: {
				callId: state.currentMessageId,
				toolUseIds: [event.content_block.id]
			}
		});
		if (event?.type === "message_stop" && state.currentMessageId) events.push({
			method: "usage/apiCallEnded",
			params: { callId: state.currentMessageId }
		});
		if (event?.type === "content_block_delta" && event.delta?.type === "text_delta") {
			const itemId = streamItemId(state, "text", event.index);
			state.text.set(itemId, `${state.text.get(itemId) ?? ""}${event.delta.text}`);
			events.push({
				method: "item/agentMessage/delta",
				params: {
					itemId,
					delta: event.delta.text
				}
			});
		}
		if (event?.type === "content_block_delta" && event.delta?.type === "thinking_delta") {
			const itemId = streamItemId(state, "reason", event.index);
			state.reasoning.set(itemId, `${state.reasoning.get(itemId) ?? ""}${event.delta.thinking}`);
			events.push({
				method: "item/reasoning/summaryTextDelta",
				params: {
					itemId,
					delta: event.delta.thinking
				}
			});
		}
		return events;
	}
	if (message.type === "assistant") {
		events.push(...usageEvents(message.message?.id, message.message?.usage));
		const content = message.message?.content ?? [];
		const toolUseIds = content.filter((block) => block.type === "tool_use" && block.id).map((block) => block.id);
		if (toolUseIds.length > 0 && message.message?.id) events.push({
			method: "usage/toolUses",
			params: {
				callId: message.message.id,
				toolUseIds
			}
		});
		for (const [index, block] of content.entries()) {
			if (block.type === "text" && block.text) {
				const itemId = block.id ?? messageItemId(state.text, message, "text", content, index);
				const previous = state.text.get(itemId) ?? "";
				const delta = block.text.startsWith(previous) ? block.text.slice(previous.length) : block.text;
				state.text.set(itemId, block.text);
				if (delta) events.push({
					method: "item/agentMessage/delta",
					params: {
						itemId,
						delta
					}
				});
			}
			if (block.type === "thinking" && block.thinking) {
				const itemId = block.id ?? messageItemId(state.reasoning, message, "reason", content, index);
				const previous = state.reasoning.get(itemId) ?? "";
				const delta = block.thinking.startsWith(previous) ? block.thinking.slice(previous.length) : block.thinking;
				state.reasoning.set(itemId, block.thinking);
				if (delta) events.push({
					method: "item/reasoning/summaryTextDelta",
					params: {
						itemId,
						delta
					}
				});
			}
			if (block.type === "tool_use") {
				const item = {
					type: "toolUse",
					id: block.id,
					name: block.name,
					input: block.input,
					status: "inProgress"
				};
				if (!state.activities.has(item.id)) {
					state.activities.add(item.id);
					events.push({
						method: "item/started",
						params: { item }
					});
				}
			}
		}
	}
	if (message.type === "user") for (const block of message.message?.content ?? []) {
		if (block.type !== "tool_result") continue;
		events.push({
			method: "item/completed",
			params: { item: {
				type: "toolUse",
				id: block.tool_use_id,
				output: block.content,
				status: block.is_error ? "failed" : "completed"
			} }
		});
	}
	if (message.type === "system" && message.subtype === "permission_denied") events.push({
		method: "item/completed",
		params: { item: {
			type: "toolUse",
			id: message.tool_use_id,
			name: message.tool_name,
			output: message.message,
			status: "failed"
		} }
	});
	return events;
}
const CALL_USAGE_FIELDS = [
	"input_tokens",
	"output_tokens",
	"cache_read_input_tokens",
	"cache_creation_input_tokens"
];
/**
* One model call's usage as the SDK reports it. A call reports usage several
* times (at message start, in the output deltas, on each streamed assistant
* message), so each report is an activity and the adapter keeps the largest
* value of each field.
*/
function usageEvents(callId, usage) {
	if (!callId || !usage) return [];
	const fields = {};
	for (const field of CALL_USAGE_FIELDS) if (Number.isSafeInteger(usage[field])) fields[field] = usage[field];
	return Object.keys(fields).length > 0 ? [{
		method: "usage/apiCall",
		params: {
			callId,
			usage: fields
		}
	}] : [];
}
function streamItemId(state, type, index) {
	return `${state.currentMessageId ?? "message"}-${type}-${index ?? 0}`;
}
function messageItemId(items, message, type, content, index) {
	const prefix = `${message.message?.id ?? message.uuid ?? "message"}-${type}-`;
	const ordinal = content.slice(0, index).filter((block) => block.type === (type === "reason" ? "thinking" : type)).length;
	return [...items.keys()].filter((itemId) => itemId.startsWith(prefix)).sort((left, right) => Number(left.slice(prefix.length)) - Number(right.slice(prefix.length)))[ordinal] ?? `${prefix}${index}`;
}
function responseForRequest(pending, response) {
	if (response.action === "accept" || response.action === "allow") return {
		behavior: "allow",
		updatedInput: response.updatedInput ?? pending.input
	};
	if (response.action === "answer") return {
		behavior: "allow",
		updatedInput: {
			...pending.input,
			answers: response.answers ?? {}
		}
	};
	return {
		behavior: "deny",
		message: response.message ?? "User declined this Claude tool request"
	};
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
	const effort = message.effort ?? session.config?.effort ?? void 0;
	return effort === "ultracode" ? ULTRACODE_BASE_EFFORT : effort;
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
//#endregion
//#region session-runtime.mjs
const DEFAULT_MODELS = [
	{
		id: "sonnet",
		displayName: "Claude Sonnet",
		description: "Claude Code default balanced model",
		isDefault: true,
		defaultReasoningEffort: "medium",
		supportedReasoningEfforts: [
			{ reasoningEffort: "low" },
			{ reasoningEffort: "medium" },
			{ reasoningEffort: "high" }
		]
	},
	{
		id: "opus",
		displayName: "Claude Opus",
		description: "Claude Code high-capability model",
		isDefault: false,
		defaultReasoningEffort: "high",
		supportedReasoningEfforts: [{ reasoningEffort: "medium" }, { reasoningEffort: "high" }]
	},
	{
		id: "haiku",
		displayName: "Claude Haiku",
		description: "Claude Code fast model",
		isDefault: false,
		defaultReasoningEffort: "low",
		supportedReasoningEfforts: [{ reasoningEffort: "low" }, { reasoningEffort: "medium" }]
	}
];
var ClaudeSessionRuntime = class extends EventEmitter {
	constructor({ client = new ClaudeCliClient(), cwd = process.cwd() } = {}) {
		super();
		this.client = client;
		this.cwd = cwd;
		this.sessions = /* @__PURE__ */ new Map();
		this.models = DEFAULT_MODELS;
		this.selectedSessionId = null;
		this.diagnostics = [];
		this.closed = false;
		this.client.on?.("activity", (message) => this.handleActivity(message));
		this.client.on?.("request", (request) => this.emit("request", request));
		this.client.on?.("diagnostic", (message) => this.addDiagnostic(message));
		this.client.on?.("exit", (details) => {
			this.addDiagnostic(`Claude backend exited: ${JSON.stringify(details)}`);
			this.emitChange();
		});
	}
	async initialize() {
		await this.client.start?.();
		const models = await this.client.listModels?.().catch((error) => {
			this.addDiagnostic(`Claude model list failed: ${error.message}`);
			return null;
		});
		if (Array.isArray(models) && models.length > 0) this.models = models;
		this.emitChange();
		return this.snapshot();
	}
	async createSession({ model, effort, sandbox = "workspace-write", approvalPolicy = "on-request", cwd = this.cwd, ephemeral = false, settingSources = [], systemPrompt = void 0 } = {}) {
		const selectedModel = model ?? this.models.find((candidate) => candidate.isDefault)?.id ?? "sonnet";
		const selectedEffort = effort ?? this.models.find((candidate) => candidate.id === selectedModel)?.defaultReasoningEffort ?? "medium";
		const created = await this.client.createSession?.({
			model: selectedModel,
			effort: selectedEffort,
			sandbox,
			approvalPolicy,
			cwd,
			ephemeral,
			settingSources,
			systemPrompt
		});
		const session = this.upsertSession(created ?? {}, {
			id: created?.id,
			model: selectedModel,
			effort: selectedEffort,
			sandbox,
			approvalPolicy,
			cwd,
			ephemeral
		});
		if (!session.ephemeral) this.selectedSessionId = session.id;
		this.emitChange();
		return publicSession(session);
	}
	async resumeSession(sessionId, defaults = {}) {
		if (!sessionId?.trim()) throw new Error("sessionId is required");
		const resumed = await this.client.resumeSession?.(sessionId, {
			cwd: defaults.cwd ?? this.cwd,
			...defaults
		});
		const session = this.upsertSession(resumed ?? {}, {
			id: sessionId,
			...defaults
		});
		if (Array.isArray(resumed?.turns) && resumed.turns.length > 0) session.turns = structuredClone(resumed.turns);
		if (!session.ephemeral) this.selectedSessionId = sessionId;
		this.emitChange();
		return publicSession(session);
	}
	async sendMessage(sessionId, { text, images, model, effort, sandbox, approvalPolicy, cwd, systemPrompt, settingSources, dshTools, executeDshTool, permissionMode, thinkingSummaries } = {}) {
		const session = this.requireSession(sessionId);
		const hasImages = Array.isArray(images) && images.length > 0;
		if (!text?.trim() && !hasImages) throw new Error("message text is required");
		const next = {
			model: model ?? session.model,
			effort: effort ?? session.effort,
			sandbox: sandbox ?? session.sandbox,
			approvalPolicy: approvalPolicy ?? session.approvalPolicy,
			cwd: cwd ?? session.cwd
		};
		Object.assign(session, next, { updatedAt: nowSeconds() });
		const turn = await this.client.sendMessage(sessionId, {
			text,
			...hasImages ? { images } : {},
			...next,
			...systemPrompt !== void 0 ? { systemPrompt } : {},
			...settingSources !== void 0 ? { settingSources } : {},
			...permissionMode !== void 0 ? { permissionMode } : {},
			...thinkingSummaries !== void 0 ? { thinkingSummaries } : {},
			...dshTools !== void 0 ? { dshTools } : {},
			...executeDshTool !== void 0 ? { executeDshTool } : {}
		});
		this.ensureTurn(session, turn);
		this.emitChange();
		return structuredClone(turn);
	}
	/** The subscription's plan-limit windows, for the composer ring. */
	planUsage() {
		return this.client.planUsage?.() ?? Promise.resolve({ supported: false });
	}
	/** Context capacity for one catalog row, as the backend reports it. */
	contextWindowFor(model, resolvedModel) {
		return this.client.contextWindowFor?.(model, resolvedModel) ?? null;
	}
	/** Hand a harness-executed tool result back to the parked call. */
	resolveToolCall(toolUseId, result) {
		return this.client.resolveToolCall?.(toolUseId, result) ?? false;
	}
	/** Fail one parked tool call. */
	rejectToolCall(toolUseId, error) {
		return this.client.rejectToolCall?.(toolUseId, error) ?? false;
	}
	/** Fail every parked tool call, e.g. when a turn is abandoned. */
	rejectAllToolCalls(error) {
		this.client.rejectAllToolCalls?.(error);
	}
	async interruptTurn(sessionId, turnId) {
		await this.client.interruptTurn?.(sessionId, turnId);
	}
	async resolveRequest(requestId, response = {}) {
		if (typeof this.client.resolveRequest !== "function") throw new Error("Claude client does not support interactive request resolution");
		this.client.resolveRequest(requestId, response);
	}
	rejectRequest(requestId, error) {
		this.client.rejectRequest?.(requestId, error);
	}
	async releaseSession(sessionId) {
		if (!sessionId) return;
		await this.client.releaseSession?.(sessionId).catch((error) => {
			this.addDiagnostic(`Claude session release failed for ${sessionId}: ${error.message}`);
		});
		this.sessions.delete(sessionId);
		if (this.selectedSessionId === sessionId) this.selectedSessionId = null;
		this.emitChange();
	}
	getSession(sessionId) {
		const session = this.sessions.get(sessionId);
		return session ? publicSession(session) : null;
	}
	snapshot() {
		return {
			connected: !this.closed,
			selectedSessionId: this.selectedSessionId,
			cwd: this.cwd,
			models: structuredClone(this.models),
			sessions: [...this.sessions.values()].sort((left, right) => right.updatedAt - left.updatedAt).map((session) => publicSession(session)),
			diagnostics: this.diagnostics.slice(-20)
		};
	}
	async close() {
		if (this.closed) return;
		this.closed = true;
		await this.client.close?.();
	}
	handleActivity(message) {
		const params = message.params ?? {};
		const sessionId = params.sessionId ?? params.session?.id ?? null;
		const session = sessionId ? this.sessions.get(sessionId) : null;
		if (message.method === "turn/completed" && session) {
			const turn = params.turn;
			if (turn?.id) this.ensureTurn(session, turn);
		}
		this.emit("activity", message);
		this.emitChange();
	}
	upsertSession(input = {}, defaults = {}) {
		const id = input.id ?? defaults.id;
		if (!id) throw new Error("Claude session id is required");
		const existing = this.sessions.get(id) ?? {
			id,
			turns: [],
			createdAt: nowSeconds()
		};
		Object.assign(existing, {
			model: defaults.model ?? input.model ?? existing.model ?? this.models.find((candidate) => candidate.isDefault)?.id ?? "sonnet",
			effort: defaults.effort ?? input.effort ?? existing.effort ?? "medium",
			sandbox: defaults.sandbox ?? input.sandbox ?? existing.sandbox ?? "workspace-write",
			approvalPolicy: defaults.approvalPolicy ?? input.approvalPolicy ?? existing.approvalPolicy ?? "on-request",
			cwd: defaults.cwd ?? input.cwd ?? existing.cwd ?? this.cwd,
			ephemeral: Boolean(defaults.ephemeral ?? input.ephemeral ?? existing.ephemeral),
			updatedAt: input.updatedAt ?? nowSeconds()
		});
		if (Array.isArray(input.turns) && input.turns.length > 0) existing.turns = structuredClone(input.turns);
		this.sessions.set(id, existing);
		return existing;
	}
	requireSession(sessionId) {
		const session = this.sessions.get(sessionId);
		if (!session) throw new Error(`unknown Claude session ${sessionId}`);
		return session;
	}
	ensureTurn(session, turn) {
		if (!turn?.id) return;
		const existing = session.turns.findIndex((candidate) => candidate.id === turn.id);
		if (existing >= 0) {
			if (session.turns[existing].status !== "inProgress" && turn.status === "inProgress") return;
			session.turns[existing] = structuredClone(turn);
		} else session.turns.push(structuredClone(turn));
		session.updatedAt = nowSeconds();
	}
	addDiagnostic(message) {
		this.diagnostics.push(String(message));
		this.diagnostics.splice(0, Math.max(0, this.diagnostics.length - 50));
	}
	emitChange() {
		this.emit("change", this.snapshot());
	}
};
function publicSession(session) {
	return structuredClone({
		id: session.id,
		model: session.model,
		effort: session.effort,
		sandbox: session.sandbox,
		approvalPolicy: session.approvalPolicy,
		cwd: session.cwd,
		ephemeral: session.ephemeral,
		turns: session.turns ?? []
	});
}
function nowSeconds() {
	return Date.now() / 1e3;
}
//#endregion
//#region plugin.mjs
const CLAUDE_EXECUTION_CAPABILITY = "agent-bridge.execution.v1";
function createClaudeExecutionPlugin(config = {}) {
	return definePlugin({
		manifest: {
			id: "agent-bridge.execution",
			version: "1.0.0",
			provides: { [CLAUDE_EXECUTION_CAPABILITY]: "1.0.0" },
			optional: { "agent-bridge.logging.v1": "^1.0.0" },
			permissions: ["process:claude", "filesystem:workspace"]
		},
		activate({ capabilities, defer }) {
			const logger = capabilities.optional("agent-bridge.logging.v1") ?? console;
			const runtime = new ClaudeSessionRuntime({
				client: config.client ?? createClaudeClient(config),
				cwd: config.cwd ?? process.cwd()
			});
			defer(() => runtime.close());
			const ready = runtime.initialize();
			ready.catch((error) => {
				logger.error?.(`Relay Claude backend failed to initialize: ${error?.stack ?? error}`);
			});
			return { capabilities: { [CLAUDE_EXECUTION_CAPABILITY]: executionCapability(runtime, ready) } };
		}
	});
}
function executionCapability(runtime, ready) {
	return Object.freeze({
		whenReady: () => ready,
		listModels: () => structuredClone(runtime.models),
		hasSession: (sessionId) => runtime.sessions.has(sessionId),
		getSession: runtime.getSession.bind(runtime),
		patchSession(sessionId, patch) {
			const session = runtime.sessions.get(sessionId);
			if (session) Object.assign(session, structuredClone(patch));
			return Boolean(session);
		},
		createSession: runtime.createSession.bind(runtime),
		resumeSession: runtime.resumeSession.bind(runtime),
		sendMessage: runtime.sendMessage.bind(runtime),
		interruptTurn: runtime.interruptTurn.bind(runtime),
		contextWindowFor: runtime.contextWindowFor.bind(runtime),
		planUsage: runtime.planUsage.bind(runtime),
		resolveToolCall: runtime.resolveToolCall.bind(runtime),
		rejectToolCall: runtime.rejectToolCall.bind(runtime),
		rejectAllToolCalls: runtime.rejectAllToolCalls.bind(runtime),
		releaseSession: runtime.releaseSession.bind(runtime),
		resolveRequest: runtime.resolveRequest.bind(runtime),
		rejectRequest: runtime.rejectRequest.bind(runtime),
		subscribeActivity: (listener) => subscribe(runtime, "activity", listener),
		subscribeRequest: (listener) => subscribe(runtime, "request", listener)
	});
}
function createClaudeClient(config) {
	const backend = config.backend ?? "auto";
	const env = claudeEnv(config);
	if (backend === "cli") return createClaudeCliClient(config, env);
	const sdkClient = new ClaudeSdkClient({
		pathToClaudeCodeExecutable: config.codeExecutablePath,
		env,
		requestTimeoutMs: positiveInteger(config.requestTimeoutMs, 30 * 6e4)
	});
	if (backend === "sdk") return sdkClient;
	return new FallbackClaudeClient({
		primary: sdkClient,
		fallback: createClaudeCliClient(config, env)
	});
}
/**
* The environment every Claude Code child runs under.
*
* Built from this process's own environment, so PATH and everything else
* survives, with one addition when a gateway endpoint was configured. The base
* URL is the only thing this sets: authentication stays whatever Claude Code
* already owns, so the subscription credential is neither read nor replaced.
*
* Nonessential traffic is off on this path. Claude Code otherwise generates a
* session title of its own on the first turn, and that request carries its own
* effort (observed `medium`) rather than the effort the caller selected. With a
* gateway in front of the model, that is a second inference request the operator
* did not ask for, on a different effort. Turning it off leaves exactly the
* turns the harness requested.
*/
function claudeEnv(config) {
	if (!config.bifrostBaseUrl) return void 0;
	return {
		...process.env,
		ANTHROPIC_BASE_URL: config.bifrostBaseUrl,
		CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1"
	};
}
function createClaudeCliClient(config, env = void 0) {
	return new ClaudeCliClient({
		command: config.command ?? "claude",
		args: config.args ?? [],
		...env ? { env } : {},
		requestTimeoutMs: positiveInteger(config.requestTimeoutMs, 30 * 6e4)
	});
}
/**
* Every client method the runtime reaches for. The fallback wrapper forwards
* this list verbatim; a method missing from it silently resolves against the
* CLI stub it inherits, which is how a parked tool call once became
* unresolvable while its result sat in the SDK client next door.
*/
const DELEGATED_CLIENT_METHODS = [
	"listModels",
	"createSession",
	"resumeSession",
	"sendMessage",
	"interruptTurn",
	"releaseSession",
	"resolveRequest",
	"rejectRequest",
	"contextWindowFor",
	"planUsage",
	"resolveToolCall",
	"rejectToolCall",
	"rejectAllToolCalls",
	"close"
];
var FallbackClaudeClient = class extends ClaudeCliClient {
	constructor({ primary, fallback }) {
		super();
		this.primary = primary;
		this.fallback = fallback;
		this.active = primary;
		for (const event of [
			"activity",
			"request",
			"diagnostic",
			"exit"
		]) {
			primary.on(event, (...args) => this.emit(event, ...args));
			fallback.on(event, (...args) => this.emit(event, ...args));
		}
	}
	async start() {
		try {
			await this.primary.start();
			this.active = this.primary;
		} catch (error) {
			this.emit("diagnostic", `Claude Agent SDK unavailable; falling back to CLI: ${error.message}`);
			await this.fallback.start();
			this.active = this.fallback;
		}
	}
};
for (const method of DELEGATED_CLIENT_METHODS) FallbackClaudeClient.prototype[method] = function delegate(...args) {
	return this.active[method]?.(...args);
};
function subscribe(emitter, event, listener) {
	emitter.on(event, listener);
	let active = true;
	return () => {
		if (!active) return;
		active = false;
		emitter.off(event, listener);
	};
}
function positiveInteger(value, fallback) {
	return Number.isSafeInteger(value) && value > 0 ? value : fallback;
}
//#endregion
//#region debug.js
/**
* Opt-in turn trace, off unless `DSH_AGENT_BRIDGE_DEBUG` is set.
*
* The handoff spans two `stream()` calls with a live Claude query parked in
* between, so a stall has no stack to inspect — this records the decisions that
* lead to one. Set the variable to `1` to write under `$DSH_HOME/plugin-data`,
* or to a path of your own.
*/
const setting = (process.env.DSH_AGENT_BRIDGE_DEBUG ?? "").trim();
const target = setting && setting !== "0" ? setting === "1" ? join(resolve((process.env.DSH_HOME ?? "").trim() || join(homedir(), ".dsh")), "plugin-data", "agent-bridge-debug.log") : resolve(setting) : null;
function debugLog(event, data = {}) {
	if (!target) return;
	try {
		appendFileSync(target, `${JSON.stringify({
			at: (/* @__PURE__ */ new Date()).toISOString(),
			event,
			...data
		})}\n`);
	} catch {}
}
/** Row ids dsh-base uses for the exporter, as they appear in a patch or settings file. */
const TELEMETRY_KEYS = ["session-telemetry-otel", "session-telemetry"];
/**
* Read DSH's telemetry posture from the environment the host process was
* launched with, mirroring `dsh-base`'s own row semantics:
*
* - a non-empty `DSH_TELEMETRY_DISABLED` (any value, including `0` and `false`)
*   opts the process out, and the launchers patch the row off entirely;
* - `DSH_TELEMETRY_MODE` selects the mode; `DISABLED` opts out and anything
*   else (`FULL`, `FEEDBACK_ONLY`) opts in. dsh 0.1.5's row defaults to
*   `FEEDBACK_ONLY` when the variable is unset, which this scan cannot see —
*   the mounted backend is the exact source, and this scan is its fallback;
* - `DSH_TELEMETRY_OTLP_URL` overrides the production endpoint.
*
* @returns `null` when nothing is exported, otherwise the active mode and endpoint.
*/
function telemetryExport(env = process.env, readFile = readTextFile) {
	if (env.DSH_TELEMETRY_DISABLED) return null;
	const endpoint = (env.DSH_TELEMETRY_OTLP_URL ?? "").trim() || "https://harness-telemetry.deepseeksvc.com/v1/logs";
	const fromEnv = (env.DSH_TELEMETRY_MODE ?? "").trim();
	if (fromEnv && fromEnv.toUpperCase() !== "DISABLED") return {
		mode: fromEnv.toUpperCase(),
		endpoint,
		source: "DSH_TELEMETRY_MODE"
	};
	for (const path of configSources(env)) {
		const mode = telemetryModeIn(readFile(path));
		if (mode) return {
			mode,
			endpoint,
			source: path
		};
	}
	return null;
}
/** Every config layer that can carry a row override, home first then profiles. */
function configSources(env) {
	const home = resolve((env.DSH_HOME ?? "").trim() || join(homedir(), ".dsh"));
	const paths = [join(home, "settings.yaml"), join(home, "cordis.patch.yml")];
	let profiles = [];
	try {
		profiles = readdirSync(join(home, "profiles"), { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((entry) => join(home, "profiles", entry.name, "cordis.patch.yml"));
	} catch {
		profiles = [];
	}
	return [...paths, ...profiles];
}
function readTextFile(path) {
	try {
		return readFileSync(path, "utf8");
	} catch {
		return "";
	}
}
/**
* Find an enabled telemetry mode inside a YAML-ish config layer.
*
* Deliberately shallow: it walks the indented block that follows a telemetry
* row and reads `mode`/`disabled`. It cannot resolve `!!js` expressions, which
* is why the environment is checked first and separately.
*/
function telemetryModeIn(text) {
	const lines = String(text ?? "").split(/\r?\n/);
	for (const [index, line] of lines.entries()) {
		if (!TELEMETRY_KEYS.some((key) => line.includes(key))) continue;
		const indent = line.search(/\S/);
		let mode = null;
		let disabled = false;
		for (const next of lines.slice(index + 1)) {
			if (!next.trim()) continue;
			if (next.search(/\S/) <= indent) break;
			const modeMatch = /^\s*mode:\s*(.+?)\s*$/.exec(next);
			if (modeMatch && !modeMatch[1].startsWith("!!js")) mode = modeMatch[1].replace(/^['"]|['"]$/g, "").toUpperCase();
			if (/^\s*disabled:\s*true\b/.test(next)) disabled = true;
		}
		if (!disabled && mode && mode !== "DISABLED") return mode;
	}
	return null;
}
/** Operator-facing explanation, shared by the thrown error and the toast. */
function telemetryRefusal({ mode, endpoint, source }) {
	return `DSH session telemetry is enabled (mode ${mode}, from ${source === "DSH_TELEMETRY_MODE" ? "DSH_TELEMETRY_MODE" : source}), which uploads unredacted session records — including Claude's output — to ${endpoint}. Claude is not available while that is on. Turn it off there, or set DSH_TELEMETRY_MODE=DISABLED (or DSH_TELEMETRY_DISABLED=1 to opt the process out entirely) in the shell that launches DSH, then restart it.`;
}
//#endregion
//#region telemetry-control.js
/**
* What the plugin does when DSH's session exporter is live at the start of a
* Claude turn.
*
* `disable` (the default) shuts the exporter's pipeline down through the
* backend's own `shutdown()`. This plugin exists to route one vendor's model
* through another vendor's harness, and DSH's exporter uploads unredacted
* session records — including that model's output — to its own collector.
* Installing this plugin is a statement that Claude should run here; quietly
* letting its output ship elsewhere is not a reasonable default, and an
* installer who has never read this README will not know to set an environment
* variable.
*
* `refuse` fails the turn instead and leaves the exporter alone, for anyone who
* would rather the plugin never touch host configuration. `ignore` disables the
* guard entirely.
*
* The guard runs per turn, never during plugin activation. dsh 0.1.5 mounts the
* exporter by default in `FEEDBACK_ONLY`, and acting on another plugin's row
* while the cordis loader is still applying entries deadlocked the boot: the
* loader waited on a fiber this plugin had just disposed, the web app never
* printed its URL, and the operator was locked out. Activation now only reads.
*/
const TELEMETRY_MODES = [
	"disable",
	"refuse",
	"ignore"
];
/** The mounted backend, or null when the host has none. */
function liveBackend(ctx) {
	try {
		return ctx?.get?.("sessionTelemetry") ?? null;
	} catch {
		return null;
	}
}
/**
* cordis hands `ctx.get()` callers a fresh traceable proxy every time, so two
* reads of the same row are never `===`. The proxy answers this registered
* symbol with the object it wraps; that is the identity an instance keeps.
*/
const ORIGINAL = Symbol.for("cordis.original");
function identityOf(backend) {
	const original = backend?.[ORIGINAL];
	return typeof original === "object" && original !== null ? original : backend;
}
/**
* Whether records currently leave the machine.
*
* `sharing` is the backend's own three-value vocabulary ("full",
* "feedback-only", "disabled"); `provider` is the OTel pipeline, absent when
* disabled, and is the fallback for a backend that reports no sharing status.
*
* @returns the sharing mode when exporting, otherwise null.
*/
function exportingState(backend) {
	if (!backend) return null;
	if (typeof backend.sharing === "string") return backend.sharing === "disabled" ? null : backend.sharing;
	return backend.provider === void 0 ? null : "unknown";
}
/**
* Guard the turn against a live exporter.
*
* Checking the mounted backend is exact, unlike {@link telemetryExport}'s scan
* of environment and config layers — it sees the row whatever switched it on.
* The scan stays as a fallback for a host whose backend cannot be read.
*
* The guard never unmounts the row. Disposing another plugin's fiber from here
* is what deadlocked dsh 0.1.5's boot, and even at turn time it reaches into
* host structure this plugin does not own. `shutdown()` is the backend's own
* contract: once it resolves, the OTel provider drops every later record, so a
* resolved shutdown is the confirmation — and a shutdown that rejects (the
* backend's own deadline) refuses the turn rather than hoping.
*/
function createTelemetryControl(ctx, { mode = "disable", logger = console } = {}) {
	const setting = TELEMETRY_MODES.includes(mode) ? mode : "disable";
	const quiesced = /* @__PURE__ */ new WeakSet();
	let removed = null;
	function couldNotDisable(sharing, reason) {
		return /* @__PURE__ */ new Error(`DSH session telemetry is ${sharing} and could not be turned off${reason ? ` (${reason})` : ""}. Claude is not available while it is on. Set DSH_TELEMETRY_MODE=DISABLED (or DSH_TELEMETRY_DISABLED=1) in the shell that launches DSH, then restart it.`);
	}
	async function disable(backend, sharing) {
		if (typeof backend.shutdown !== "function") throw couldNotDisable(sharing, "the backend has no shutdown()");
		try {
			await backend.shutdown();
		} catch (error) {
			throw couldNotDisable(sharing, error?.message ?? String(error));
		}
		quiesced.add(identityOf(backend));
		removed = sharing;
		logger.warn?.(`agent-bridge: DSH session telemetry was ${sharing} and its exporter has been shut down for this process. It uploads unredacted session records, which would include Claude's output. Set DSH_TELEMETRY_MODE=DISABLED to keep it off at the source, or configure this plugin with telemetry: "refuse" to be failed instead.`);
	}
	return {
		/**
		* Read-only look at the host, for a log line at activation. Never awaits
		* and never acts: activation must not block on, or touch, another
		* plugin's row.
		*
		* @returns the live sharing mode, or null when nothing exports.
		*/
		inspect() {
			if (setting === "ignore") return null;
			const backend = liveBackend(ctx);
			if (backend && quiesced.has(identityOf(backend))) return null;
			return exportingState(backend);
		},
		/** @returns what the guard did, or throws when a turn must not run. */
		async enforce() {
			if (setting === "ignore") return { state: "ignored" };
			const backend = liveBackend(ctx);
			const sharing = exportingState(backend);
			if (sharing) {
				if (quiesced.has(identityOf(backend))) return {
					state: "disabled",
					sharing
				};
				if (setting === "refuse") throw new Error(telemetryRefusal(telemetryExport() ?? {
					mode: sharing.toUpperCase(),
					endpoint: "DSH's configured collector",
					source: "the running exporter"
				}));
				await disable(backend, sharing);
				return {
					state: "disabled",
					sharing
				};
			}
			if (removed) return {
				state: "disabled",
				sharing: removed
			};
			const configured = telemetryExport();
			if (configured) throw new Error(telemetryRefusal(configured));
			return { state: "off" };
		}
	};
}
/** The guard used when no host context is available, e.g. in tests. */
function scanOnlyTelemetryControl() {
	return {
		inspect() {
			return telemetryExport()?.mode?.toLowerCase() ?? null;
		},
		async enforce() {
			const configured = telemetryExport();
			if (configured) throw new Error(telemetryRefusal(configured));
			return { state: "off" };
		}
	};
}
//#endregion
//#region claude-adapter.js
const CLAUDE_PROVIDER = "claude";
/** Appended when a parked turn continues in a new Claude session after DSH rewrote its history. */
const PARKED_TURN_NOTE = "Earlier tool calls for this request already ran. Their results, or the summary of them, are in the context above; continue from them and do not repeat those calls.";
var ClaudeDshAdapter = class extends LlmAdapter {
	constructor({ runtime, ready, linkStore = null, logger = console, telemetry = scanOnlyTelemetryControl(), thinkingSummaries = true, resolveAttachments = () => void 0 }) {
		super();
		this.telemetry = telemetry;
		this.thinkingSummaries = thinkingSummaries;
		this.resolveAttachments = resolveAttachments;
		this.runtime = runtime;
		this.ready = ready;
		this.logger = logger;
		this.linkStore = linkStore;
		this.links = /* @__PURE__ */ new Map();
		this.settings = /* @__PURE__ */ new Map();
		this.pendingSessions = /* @__PURE__ */ new Map();
		this.agents = /* @__PURE__ */ new Map();
		this.seen = /* @__PURE__ */ new Map();
		this.builtAt = /* @__PURE__ */ new Map();
		this.suspended = /* @__PURE__ */ new Map();
		this.usageLedgers = /* @__PURE__ */ new Map();
		this.ownedSessions = /* @__PURE__ */ new Set();
		this.readers = /* @__PURE__ */ new Set();
		this.finishedTurns = /* @__PURE__ */ new Set();
		this.waitingCalls = /* @__PURE__ */ new Map();
		this.activityUnsubscribe = null;
		if (runtime && (typeof runtime.subscribeActivity === "function" || typeof runtime.on === "function")) try {
			this.activityUnsubscribe = subscribeRuntimeActivity(runtime, (message) => {
				this.recordWaitingCall(message);
				this.recordUsageActivity(message);
			});
		} catch (error) {
			this.logger?.warn?.(`Claude usage and waiting-call reporting is unavailable: ${error?.message ?? error}`);
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
		return {
			id: CLAUDE_PROVIDER,
			name: "Claude"
		};
	}
	async listModels() {
		await this.ready;
		return runtimeModels(this.runtime).sort((left, right) => Number(Boolean(right.isDefault)) - Number(Boolean(left.isDefault))).map((model) => ({
			provider: CLAUDE_PROVIDER,
			id: model.id,
			name: model.displayName ?? model.id,
			inputModalities: ["text", "image"]
		}));
	}
	async resolveModel(provider, model) {
		await this.ready;
		const info = runtimeModels(this.runtime).find((candidate) => candidate.id === model);
		const contextWindow = this.runtime.contextWindowFor?.(model, info?.resolvedModel);
		return {
			provider,
			id: model,
			name: info?.displayName ?? model,
			inputModalities: ["text", "image"],
			...contextWindow ? { context: { contextWindow } } : {},
			...Array.isArray(info?.supportedReasoningEfforts) ? { reasoning: {
				efforts: info.supportedReasoningEfforts.map((effort) => ({
					id: effort.reasoningEffort ?? effort.id ?? effort,
					name: reasoningEffortName(effort.reasoningEffort ?? effort.id ?? effort)
				})),
				defaultEffort: info.defaultReasoningEffort
			} } : {}
		};
	}
	attachAgent(agent) {
		this.agents.set(String(agent.id), agent);
		this.configuration(agent.id, agent.session.header.cwd);
		return true;
	}
	/**
	* DSH has disposed the agent, so no further turn will run on its Claude session. Release the
	* Claude session, which closes the CLI process it still holds. The link and its configuration
	* are kept, so reopening the same DSH session resumes the same Claude transcript.
	*/
	detachAgent(sessionId) {
		const key = String(sessionId);
		this.agents.delete(key);
		const claudeSessionId = this.links.get(key);
		if (!claudeSessionId) return;
		this.suspended.delete(key);
		this.ownedSessions.delete(claudeSessionId);
		this.readers.delete(claudeSessionId);
		const disposal = /* @__PURE__ */ new Error("The DSH agent was disposed while the turn was waiting for a tool result");
		for (const call of this.takeWaitingCalls(claudeSessionId)) try {
			this.runtime.rejectToolCall?.(call, disposal);
		} catch (error) {
			this.logger?.warn?.(`Relay could not fail tool call ${call} of Claude session ${claudeSessionId} after its DSH agent was disposed: ${error?.message ?? error}`);
		}
		if (typeof this.runtime.releaseSession !== "function") return;
		const warn = (error) => this.logger?.warn?.(`Relay could not release Claude session ${claudeSessionId} after its DSH agent was disposed: ${error?.message ?? error}`);
		try {
			Promise.resolve(this.runtime.releaseSession(claudeSessionId)).catch(warn);
		} catch (error) {
			warn(error);
		}
	}
	configuration(sessionId, cwd) {
		const key = String(sessionId);
		const existing = this.settings.get(key);
		if (existing) return existing;
		const models = runtimeModels(this.runtime);
		const model = models.find((candidate) => candidate.isDefault) ?? models[0];
		const config = {
			model: model?.id ?? "sonnet",
			effort: model?.defaultReasoningEffort,
			sandbox: "workspace-write",
			approvalPolicy: "on-request",
			cwd: cwd ?? process.cwd(),
			settingSources: [],
			systemPrompt: void 0,
			thinkingSummaries: this.thinkingSummaries
		};
		this.settings.set(key, config);
		return config;
	}
	configure(sessionId, patch = {}) {
		const key = String(sessionId);
		const next = {
			...this.configuration(key),
			...compact(patch)
		};
		this.settings.set(key, next);
		const claudeSessionId = this.links.get(key);
		if (claudeSessionId) patchRuntimeSession(this.runtime, claudeSessionId, next);
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
		if (linked) try {
			await this.runtime.resumeSession(linked, settings);
			return linked;
		} catch (error) {
			this.logger.warn(`Relay could not resume Claude session ${linked}; creating a replacement: ${error.message}`);
			this.links.delete(sessionId);
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
			surfaceSeq: this.builtAt.get(String(sessionId)) ?? null
		});
	}
	sessionFor(sessionId) {
		return this.links.get(String(sessionId)) ?? null;
	}
	dshSessionForClaudeSession(claudeSessionId) {
		for (const [sessionId, candidate] of this.links) if (candidate === claudeSessionId) return sessionId;
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
		let suspended = this.suspended.get(sessionId);
		const rewritten = this.historyRewritten(sessionId, agent);
		const parkedTurnRewritten = rewritten && Boolean(suspended);
		if (rewritten) {
			if (suspended) await this.releaseParkedTurn(sessionId, suspended);
			suspended = void 0;
			await this.retireClaudeSession(sessionId);
		}
		debugLog("stream", {
			sessionId,
			messages: options.messages?.length ?? 0,
			tools: options.tools?.length ?? 0,
			suspended: suspended ? suspended.calls : null,
			toolResultIds: allToolResultIds(options.messages),
			rewritten
		});
		if (suspended) {
			const results = toolResultsFor(options.messages, suspended.calls);
			const superseded = turnStartedSince(agent, suspended.dshSeq);
			if (results.length && !superseded) {
				const inputs = await this.prepareInputs(this.undeliveredInput(sessionId, options.messages), options.signal);
				if (options.signal?.aborted) {
					await this.abandonTurn(sessionId, suspended);
					yield {
						type: "finish",
						reason: {
							kind: "aborted",
							failure: {
								message: "Claude turn cancelled",
								code: "ABORTED"
							}
						}
					};
					return;
				}
				debugLog("resume", {
					sessionId,
					turnId: suspended.turnId,
					resolving: results.map((result) => result.toolCallId),
					inputs: inputs.length
				});
				this.markBuilt(sessionId, agent);
				const deliveredBefore = this.seen.get(String(sessionId));
				yield* this.runTurn(agent, sessionId, suspended.claudeSessionId, options, async () => {
					for (const result of resultsWithInput(results, inputs)) {
						const delivered = this.runtime.resolveToolCall(result.toolCallId, dshToolResult(result));
						debugLog("resolve", {
							toolCallId: result.toolCallId,
							delivered
						});
						if (delivered) this.waitingCalls.get(suspended.claudeSessionId)?.delete(result.toolCallId);
						if (!delivered) throw new Error(`No Claude tool call is waiting for result ${result.toolCallId}; the turn cannot be resumed`);
					}
					this.markDelivered(sessionId, options.messages);
					return suspended.turnId;
				}, {
					firstEventTimeoutMs: resumeTimeoutMs(),
					knownTurnId: suspended.turnId,
					onFailed: () => this.restoreDelivered(sessionId, deliveredBefore)
				});
				return;
			}
			debugLog("abandon", {
				sessionId,
				turnId: suspended.turnId,
				expected: suspended.calls,
				superseded
			});
			await this.abandonTurn(sessionId, suspended);
		}
		const instruction = latestUserIndex(options.messages);
		if (instruction === -1) throw new Error("The Claude adapter received no user text");
		const input = await this.prepareInput(options.messages[instruction], options.signal);
		if (options.signal?.aborted) {
			yield {
				type: "finish",
				reason: {
					kind: "aborted",
					failure: {
						message: "Claude turn cancelled",
						code: "ABORTED"
					}
				}
			};
			return;
		}
		if (!input.text && input.images.length === 0) throw new Error("The Claude adapter received no user text");
		const nativePermissions = permissionConfiguration(sessionEventLog(agent.session));
		const config = this.configure(sessionId, {
			model: options.model,
			effort: options.reasoningEffort,
			systemPrompt: options.system,
			...nativePermissions,
			cwd: agent.session.header.cwd
		});
		const dshTools = structuredClone(options.tools ?? []);
		const claudeSessionId = await this.ensureSession(sessionId);
		const seeded = this.seedPrompt(sessionId, options.messages, instruction, input.text);
		const prompt = parkedTurnRewritten ? `${seeded}\n\n${PARKED_TURN_NOTE}` : seeded;
		this.markBuilt(sessionId, agent);
		const deliveredBefore = this.seen.get(String(sessionId));
		yield* this.runTurn(agent, sessionId, claudeSessionId, options, async () => {
			const images = input.images.length > 0 ? { images: input.images } : {};
			const started = await this.runtime.sendMessage(claudeSessionId, {
				text: prompt,
				...images,
				...config,
				dshTools
			});
			this.markDelivered(sessionId, options.messages);
			return started.id;
		}, { onFailed: () => this.restoreDelivered(sessionId, deliveredBefore) });
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
	async *runTurn(agent, sessionId, claudeSessionId, options, start, { firstEventTimeoutMs = 0, knownTurnId = null, onFailed = null } = {}) {
		const queue = new ActivityQueue(options.signal, "Claude");
		const onActivity = (message) => {
			if ((message.params?.sessionId ?? message.params?.session?.id) === claudeSessionId && !String(message.method).startsWith("usage/")) queue.push(message);
		};
		const stopActivity = subscribeRuntimeActivity(this.runtime, onActivity);
		let turnId = knownTurnId;
		this.ownedSessions.add(claudeSessionId);
		this.readers.add(claudeSessionId);
		try {
			turnId = await start();
			debugLog("turn/started", {
				sessionId,
				claudeSessionId,
				turnId
			});
			const state = createStreamState(this.usageLedgerFor(turnId, claudeSessionId));
			let completedTurn = null;
			let batch = null;
			let first = true;
			while (!completedTurn && !batch) {
				const message = await (first && firstEventTimeoutMs ? withTimeout(queue.next(), firstEventTimeoutMs, `Claude did not respond within ${Math.round(firstEventTimeoutMs / 1e3)}s of the tool result being delivered`) : queue.next());
				first = false;
				const params = message.params ?? {};
				if (params.turnId && params.turnId !== turnId) continue;
				if (message.method === "turn/completed") {
					if (params.turn?.id !== turnId) continue;
					for (const item of params.turn.items ?? []) for (const chunk of this.completeItem(item, state)) yield chunk;
					completedTurn = params.turn;
					break;
				}
				if (isToolParked(message)) {
					batch = [message.params, ...drainParkedToolCalls(queue, turnId)];
					break;
				}
				for (const chunk of this.projectActivity(message, state)) yield chunk;
			}
			if (batch) for await (const chunk of this.settleParkedStep(queue, state, batch, turnId)) yield chunk;
			for (const chunk of closeOpenBlocks(state)) yield chunk;
			for (const chunk of stepUsageChunks(state, completedTurn)) yield chunk;
			if (completedTurn && turnId) this.usageLedgers.delete(turnId);
			if (batch) {
				const calls = batch.map((parked) => ({
					id: parked.parkId,
					name: parked.name,
					arguments: JSON.stringify(parked.arguments ?? {})
				}));
				for (const call of calls) {
					const index = state.nextIndex++;
					yield {
						type: "block-start",
						index,
						blockType: "tool-call"
					};
					yield {
						type: "tool-call-delta",
						index,
						id: call.id,
						name: call.name,
						argumentsDelta: call.arguments
					};
					yield {
						type: "block-end",
						index,
						block: {
							type: "tool-call",
							id: call.id,
							name: call.name,
							arguments: call.arguments
						}
					};
				}
				this.suspended.set(sessionId, {
					claudeSessionId,
					turnId,
					calls: calls.map((call) => call.id),
					dshSeq: logHead(agent)
				});
				debugLog("handoff", {
					sessionId,
					turnId,
					calls: calls.map((call) => ({
						id: call.id,
						name: call.name
					}))
				});
				yield {
					type: "finish",
					reason: { kind: "tool-calls" }
				};
				return;
			}
			this.suspended.delete(sessionId);
			if (completedTurn.status === "failed") {
				onFailed?.();
				yield {
					type: "finish",
					reason: {
						kind: "error",
						failure: {
							message: completedTurn.error?.message ?? "Claude turn failed",
							code: "CLAUDE_TURN_FAILED"
						}
					}
				};
			} else {
				this.persistLink(sessionId);
				debugLog("finish", {
					sessionId,
					turnId,
					kind: "stop"
				});
				yield {
					type: "finish",
					reason: { kind: "stop" },
					replayState: {
						claudeSessionId,
						turnId
					}
				};
			}
		} catch (error) {
			this.releaseTurn(claudeSessionId, turnId);
			debugLog("turn/error", {
				sessionId,
				turnId,
				aborted: Boolean(options.signal?.aborted),
				message: String(error?.message ?? error)
			});
			if (options.signal?.aborted) {
				if (turnId) await this.abandonTurn(sessionId, {
					claudeSessionId,
					turnId
				});
				yield {
					type: "finish",
					reason: {
						kind: "aborted",
						failure: {
							message: "Claude turn cancelled",
							code: "ABORTED"
						}
					}
				};
				return;
			}
			if (turnId) await this.abandonTurn(sessionId, {
				claudeSessionId,
				turnId
			});
			onFailed?.();
			throw error;
		} finally {
			stopActivity();
			queue.close();
			this.readers.delete(claudeSessionId);
			if (turnId && (this.usageLedgers.get(turnId)?.completed || this.finishedTurns.has(turnId)) && !this.suspended.has(sessionId)) this.usageLedgers.delete(turnId);
		}
	}
	/** Remember a tool call that a linked Claude session announced, until its result is delivered or it is failed. */
	recordWaitingCall(message) {
		if (!isToolParked(message)) return;
		const { sessionId, parkId } = message.params;
		if (!this.dshSessionForClaudeSession(sessionId)) return;
		if (!this.waitingCalls.has(sessionId)) this.waitingCalls.set(sessionId, /* @__PURE__ */ new Set());
		this.waitingCalls.get(sessionId).add(parkId);
	}
	/** Forget and return the tool calls a Claude session is still waiting on. */
	takeWaitingCalls(claudeSessionId) {
		const calls = [...this.waitingCalls.get(claudeSessionId) ?? []];
		this.waitingCalls.delete(claudeSessionId);
		return calls;
	}
	/** Forget a turn's ledger and its Claude session, once nothing more will report usage for them. */
	releaseTurn(claudeSessionId, turnId) {
		this.rememberFinished(turnId);
		for (const [id, ledger] of this.usageLedgers) if (ledger.sessionId === claudeSessionId || id === turnId) this.usageLedgers.delete(id);
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
		const error = /* @__PURE__ */ new Error("The DSH turn was abandoned before the tool result arrived");
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
		if (built === void 0) return log.some(isReplaceEvent);
		const logged = [];
		for (let index = log.length - 1; index >= 0 && log[index].seq > built; index -= 1) logged.push(log[index]);
		const held = logged.reduce((seq, event) => producedByClaude(event) ? Math.max(seq, event.seq) : seq, built);
		return logged.some((event) => isReplaceEvent(event) && event.surfaceOp.startSeq <= held);
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
	* Put back the delivered position a turn advanced before it failed, so the input it
	* carried is offered on the next turn. Used only for a turn that did not complete.
	*/
	restoreDelivered(sessionId, position) {
		const key = String(sessionId);
		if (position === void 0) this.seen.delete(key);
		else this.seen.set(key, position);
		this.persistLink(sessionId);
	}
	/**
	* DSH messages past the delivered position that carry text and are not tool
	* results: what the user typed, and any DSH context added, while a tool was running.
	*/
	undeliveredInput(sessionId, messages) {
		const delivered = this.seen.get(String(sessionId)) ?? 0;
		return messages.slice(delivered).filter((message) => [
			"user",
			"system",
			"developer"
		].includes(message?.role) && toolResultsIn(message).length === 0 && hasUserInput(message));
	}
	/**
	* One DSH message as Claude receives it: its text, with a line naming each image, and
	* the bytes of each image the attachment service could read. An image that cannot be
	* read is still named in the text, so its message is delivered and Claude knows it was sent.
	*/
	async prepareInput(message, signal) {
		const parts = [];
		const images = [];
		let encoded = 0;
		for (const block of message?.content ?? []) {
			if (block?.type === "text") parts.push(block.text);
			if (block?.type !== "image") continue;
			if (block.offloaded === true) {
				parts.push(`[image omitted to fit request image limits: ${imageName(block)}]`);
				continue;
			}
			const image = await this.readImage(block, signal);
			if (image && encoded + image.data.length > IMAGE_BYTES_PER_MESSAGE) parts.push(`[image omitted to fit request image limits: ${image.name}]`);
			else if (image) {
				encoded += image.data.length;
				images.push(image);
				parts.push(`[image: ${image.name}]`);
			} else parts.push(`[image could not be read: ${imageName(block)}]`);
		}
		return {
			source: message?.source,
			text: parts.join("\n").trim(),
			images
		};
	}
	async prepareInputs(messages, signal) {
		const prepared = [];
		for (const message of messages) prepared.push(await this.prepareInput(message, signal));
		return prepared;
	}
	/** Read one image's bytes through the host's attachment service; null when they cannot be read. */
	async readImage(block, signal) {
		const ref = block.attachment;
		try {
			const attachments = this.resolveAttachments();
			if (attachments === void 0 || typeof attachments.readImageRequest !== "function" || !ref) return null;
			const prepared = await attachments.readImageRequest(ref, imageRequestTarget(ref), signal);
			return {
				mediaType: prepared.mediaType,
				data: Buffer.from(prepared.data).toString("base64"),
				name: imageName(block)
			};
		} catch (error) {
			this.logger?.warn?.(`Could not read image ${imageName(block)} for Claude: ${error?.message ?? error}`);
			return null;
		}
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
		const error = /* @__PURE__ */ new Error("DSH rewrote its history while the turn was waiting for a tool result");
		for (const call of this.takeWaitingCalls(claudeSessionId)) this.runtime.rejectToolCall?.(call, error);
		if (turnId) await this.runtime.interruptTurn(claudeSessionId, turnId).catch(() => {});
	}
	async *streamAuxiliary(options) {
		await this.telemetry.enforce();
		await this.ready;
		const text = auxiliaryInput(options.messages);
		if (!text) throw new Error(`Relay Claude adapter received no ${options.purpose} input`);
		const sessionId = String(options.sessionId ?? "");
		const cwd = this.agents.get(sessionId)?.session.header.cwd ?? this.settings.get(sessionId)?.cwd ?? process.cwd();
		const claudeSessionId = (await this.runtime.createSession({
			model: options.model,
			effort: options.reasoningEffort,
			sandbox: "read-only",
			approvalPolicy: "never",
			cwd,
			ephemeral: true,
			settingSources: [],
			systemPrompt: options.system
		})).id;
		const queue = new ActivityQueue(options.signal, "Claude");
		const onActivity = (message) => {
			if ((message.params?.sessionId ?? message.params?.session?.id) === claudeSessionId) queue.push(message);
		};
		const stopActivity = subscribeRuntimeActivity(this.runtime, onActivity);
		let turnId = null;
		this.ownedSessions.add(claudeSessionId);
		try {
			turnId = (await this.runtime.sendMessage(claudeSessionId, {
				text,
				model: options.model,
				effort: options.reasoningEffort,
				sandbox: "read-only",
				approvalPolicy: "never"
			})).id;
			const state = createStreamState();
			let completedTurn = null;
			while (!completedTurn) {
				const message = await queue.next();
				const params = message.params ?? {};
				if (params.turnId && params.turnId !== turnId) continue;
				if (message.method === "turn/completed") {
					if (params.turn?.id !== turnId) continue;
					for (const item of params.turn.items ?? []) for (const chunk of completeAuxiliaryItem(state, item)) yield chunk;
					completedTurn = params.turn;
					break;
				}
				for (const chunk of projectAuxiliaryActivity(message, state)) yield chunk;
			}
			for (const block of state.blocks.values()) {
				if (block.closed) continue;
				block.closed = true;
				yield {
					type: "block-end",
					index: block.index,
					block: {
						type: block.type,
						text: block.text
					}
				};
			}
			yield completedTurn.status === "failed" ? {
				type: "finish",
				reason: {
					kind: "error",
					failure: {
						message: completedTurn.error?.message ?? `Claude ${options.purpose} failed`,
						code: "CLAUDE_AUXILIARY_FAILED"
					}
				}
			} : {
				type: "finish",
				reason: { kind: "stop" }
			};
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
		const context = messages.map((message, index) => ({
			message,
			index
		})).filter((entry) => entry.index >= seen && entry.index !== instruction).map((entry) => {
			const body = seedText(entry.message);
			return body ? `${entry.message.role ?? "user"}: ${body}` : "";
		}).filter(Boolean).join("\n\n");
		if (!context) return text;
		return [
			"<dsh-context>",
			"Context from the DeepSeek Harness session: workspace instructions, runtime snapshots, and any turns answered by another model. Background only — the request to act on follows this block.",
			context,
			"</dsh-context>",
			"",
			text
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
				if (message.params?.turn?.id !== turnId) continue;
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
			if (turnId && this.finishedTurns.has(turnId)) return;
			this.ownedSessions.delete(sessionId);
			if (!turnId) return;
			this.rememberFinished(turnId);
			const ledger = this.usageLedgers.get(turnId);
			if (!ledger) return;
			ledger.completed = true;
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
		if (message.method === "item/reasoning/summaryTextDelta" || message.method === "item/reasoning/textDelta") return textDelta(state, params.itemId, "reasoning", params.delta ?? "");
		if (message.method === "item/agentMessage/delta") return textDelta(state, params.itemId, "text", params.delta ?? "");
		if (message.method === "item/completed") return this.completeItem(params.item, state);
		return [];
	}
	completeItem(item, state) {
		if (!item?.id || state.completed.has(item.id)) return [];
		state.completed.add(item.id);
		if (item.type === "reasoning") return completeTextItem(state, item.id, "reasoning", reasoningText(item));
		if (item.type === "agentMessage") return completeTextItem(state, item.id, "text", item.text ?? "");
		return [];
	}
};
var ActivityQueue = class {
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
		if (this.closed) return Promise.reject(/* @__PURE__ */ new Error(`${this.label} activity stream closed`));
		if (this.signal?.aborted) return Promise.reject(this.signal.reason ?? /* @__PURE__ */ new Error("aborted"));
		return new Promise((resolve, reject) => {
			const waiter = {
				resolve,
				reject
			};
			this.waiters.push(waiter);
			if (this.signal) {
				const abort = () => {
					const index = this.waiters.indexOf(waiter);
					if (index >= 0) this.waiters.splice(index, 1);
					reject(this.signal.reason ?? /* @__PURE__ */ new Error("aborted"));
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
		if (this.closed) return Promise.reject(/* @__PURE__ */ new Error(`${this.label} activity stream closed`));
		if (this.signal?.aborted) return Promise.reject(this.signal.reason ?? /* @__PURE__ */ new Error("aborted"));
		return new Promise((resolve, reject) => {
			const waiter = {
				resolve: null,
				reject: null
			};
			const detach = () => {
				clearTimeout(timer);
				this.signal?.removeEventListener("abort", onAbort);
				const index = this.waiters.indexOf(waiter);
				if (index >= 0) this.waiters.splice(index, 1);
			};
			const onAbort = () => {
				detach();
				reject(this.signal.reason ?? /* @__PURE__ */ new Error("aborted"));
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
		for (const waiter of this.waiters.splice(0)) waiter.reject(/* @__PURE__ */ new Error(`${this.label} activity stream closed`));
	}
};
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
	if (typeof message.toolCallId === "string") {
		results.push({
			toolCallId: message.toolCallId,
			content: message.content ?? [],
			isError: Boolean(message.isError)
		});
		return results;
	}
	for (const block of message.content ?? []) {
		if (block?.type !== "tool-result" || typeof block.toolCallId !== "string") continue;
		results.push({
			toolCallId: block.toolCallId,
			content: block.content ?? [],
			isError: Boolean(block.isError)
		});
	}
	return results;
}
/** Every tool-result id present, for tracing a mismatch against what was parked. */
function allToolResultIds(messages) {
	const ids = [];
	for (const message of messages ?? []) for (const result of toolResultsIn(message)) ids.push(result.toolCallId);
	return ids;
}
/** The results DSH executed for the calls this turn parked on. */
function toolResultsFor(messages, callIds) {
	const wanted = new Set(callIds);
	const results = [];
	for (const message of messages ?? []) for (const result of toolResultsIn(message)) {
		if (!wanted.has(result.toolCallId)) continue;
		results.push(result);
	}
	return results;
}
/** How long a resumed turn may stay silent before it is called a failure. */
function resumeTimeoutMs() {
	const configured = Number.parseInt((process.env.DSH_AGENT_BRIDGE_RESUME_TIMEOUT_MS ?? "").trim(), 10);
	return Number.isSafeInteger(configured) && configured > 0 ? configured : 3e5;
}
function withTimeout(promise, ms, message) {
	return new Promise((resolve, reject) => {
		const timer = setTimeout(() => reject(new Error(message)), ms);
		promise.then((value) => {
			clearTimeout(timer);
			resolve(value);
		}, (error) => {
			clearTimeout(timer);
			reject(error);
		});
	});
}
function closeOpenBlocks(state) {
	const chunks = [];
	for (const block of state.blocks.values()) {
		if (block.closed) continue;
		block.closed = true;
		chunks.push({
			type: "block-end",
			index: block.index,
			block: {
				type: block.type,
				text: block.text
			}
		});
	}
	return chunks;
}
function createStreamState(usage = newUsageLedger()) {
	return {
		nextIndex: 0,
		blocks: /* @__PURE__ */ new Map(),
		completed: /* @__PURE__ */ new Set(),
		activityItems: /* @__PURE__ */ new Map(),
		startedActivities: /* @__PURE__ */ new Set(),
		completedActivities: /* @__PURE__ */ new Set(),
		/** The turn's usage ledger, shared by every DSH step of the turn. */
		usage
	};
}
/** Usage bookkeeping for one Claude turn. */
function newUsageLedger(sessionId = null) {
	return {
		sessionId,
		completed: false,
		calls: /* @__PURE__ */ new Map(),
		reported: /* @__PURE__ */ new Map(),
		ended: /* @__PURE__ */ new Set(),
		toolUses: /* @__PURE__ */ new Map()
	};
}
/** How long a parked step waits for its model calls to finish reporting usage. */
const STEP_USAGE_SETTLE_MS = 2e3;
/** How often a waiting step re-checks the ledger when nothing else arrives. */
const SETTLE_POLL_MS = 20;
/** The settle limit, overridable for tests with DSH_AGENT_BRIDGE_USAGE_SETTLE_MS. */
function usageSettleMs() {
	const raw = process.env.DSH_AGENT_BRIDGE_USAGE_SETTLE_MS;
	const ms = raw === void 0 ? NaN : Number(raw);
	return Number.isFinite(ms) && ms >= 0 ? ms : STEP_USAGE_SETTLE_MS;
}
/** True while a model call this step started has not finished reporting usage. */
function hasOpenCall(state) {
	return [...state.usage.calls.keys()].some((callId) => !state.usage.reported.has(callId) && !state.usage.ended.has(callId));
}
/** The tool calls a model call declared, which the step waits for before reporting. */
function recordToolUses(state, callId, toolUseIds) {
	if (!callId || !Array.isArray(toolUseIds)) return;
	const ids = state.usage.toolUses.get(callId) ?? /* @__PURE__ */ new Set();
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
	for (const [callId, ids] of state.usage.toolUses) if (!state.usage.reported.has(callId)) count += ids.size;
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
	for (const [field, value] of Object.entries(usage ?? {})) if (Number.isSafeInteger(value)) call[field] = Math.max(call[field] ?? 0, value);
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
			cacheWriteTokens: (call.cache_creation_input_tokens ?? 0) - (before.cache_creation_input_tokens ?? 0)
		};
		reported.set(callId, { ...call });
		if (!Object.values(growth).some((value) => value !== 0)) continue;
		const prompt = {
			inputTokens: call.input_tokens ?? 0,
			cacheReadTokens: call.cache_read_input_tokens ?? 0,
			cacheWriteTokens: call.cache_creation_input_tokens ?? 0
		};
		const size = (p) => p.inputTokens + p.cacheReadTokens + p.cacheWriteTokens;
		const keep = usage && size(usage) >= size(prompt) ? usage : prompt;
		usage = {
			inputTokens: keep.inputTokens,
			outputTokens: (usage?.outputTokens ?? 0) + growth.outputTokens,
			cacheReadTokens: keep.cacheReadTokens,
			cacheWriteTokens: keep.cacheWriteTokens
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
		if (usage) yield {
			type: "usage",
			usage
		};
		return;
	}
	if (state.usage.calls.size === 0) {
		if (completedTurn.usage) yield {
			type: "usage",
			usage: completedTurn.usage
		};
		return;
	}
	if (!usage) return;
	const reasoningTokens = completedTurn.usage?.reasoningTokens;
	yield {
		type: "usage",
		usage: reasoningTokens === void 0 ? usage : {
			...usage,
			reasoningTokens
		}
	};
}
function textDelta(state, id, type, delta) {
	if (!id || !delta) return [];
	let block = state.blocks.get(id);
	const chunks = [];
	if (!block) {
		block = {
			index: state.nextIndex++,
			type,
			text: "",
			closed: false
		};
		state.blocks.set(id, block);
		chunks.push({
			type: "block-start",
			index: block.index,
			blockType: type
		});
	}
	if (block.closed) return chunks;
	block.text += delta;
	chunks.push({
		type: type === "reasoning" ? "reasoning-delta" : "text-delta",
		index: block.index,
		text: delta
	});
	return chunks;
}
function completeTextItem(state, id, type, completeText) {
	const chunks = [];
	let block = state.blocks.get(id);
	if (!block) {
		block = {
			index: state.nextIndex++,
			type,
			text: "",
			closed: false
		};
		state.blocks.set(id, block);
		chunks.push({
			type: "block-start",
			index: block.index,
			blockType: type
		});
	}
	if (completeText && completeText.startsWith(block.text) && completeText.length > block.text.length) {
		const delta = completeText.slice(block.text.length);
		block.text = completeText;
		chunks.push({
			type: type === "reasoning" ? "reasoning-delta" : "text-delta",
			index: block.index,
			text: delta
		});
	}
	if (!block.closed) {
		block.closed = true;
		chunks.push({
			type: "block-end",
			index: block.index,
			block: {
				type,
				text: block.text
			}
		});
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
	return sessionEventLog(agent.session).some((event) => event.type === "turn/start" && event.seq > seq);
}
/** Message sources a person wrote or answered. Any other DSH message is context, not the user's input. */
const USER_SOURCE_KINDS = /* @__PURE__ */ new Set(["user", "user-question-reply"]);
/**
* The results a resumed Claude query receives, with the messages DSH received while the
* tool ran appended to the last one. The user's own input is labelled as input; a DSH
* message, such as a runtime-context snapshot, is labelled as context so Claude does not
* read it as the user's words. Each message is delivered once, on that result.
*/
function resultsWithInput(results, inputs) {
	if (inputs.length === 0 || results.length === 0) return results;
	const typed = inputs.filter((message) => USER_SOURCE_KINDS.has(message.source?.kind));
	const context = inputs.filter((message) => !USER_SOURCE_KINDS.has(message.source?.kind));
	const notes = [typed.length > 0 && `Input received while this tool ran:\n\n${typed.map((input) => input.text).join("\n\n")}`, context.length > 0 && `Context received while this tool ran:\n\n${context.map((input) => input.text).join("\n\n")}`].filter(Boolean);
	const images = inputs.flatMap((input) => input.images.map((image) => ({
		type: "image",
		data: image.data,
		mediaType: image.mediaType
	})));
	const last = results.length - 1;
	return results.map((result, index) => {
		if (index !== last) return result;
		const content = Array.isArray(result.content) ? result.content : [{
			type: "text",
			text: String(result.content ?? "")
		}];
		return {
			...result,
			content: [
				...content,
				...notes.map((text) => ({
					type: "text",
					text
				})),
				...images
			]
		};
	});
}
/** A reply the Claude route produced. DSH records the provider on the reply's message source. */
function producedByClaude(event) {
	return event.type === "assistant/message" && event.data?.message?.source?.provider === "claude";
}
/**
* A surface event that replaces an existing span of DSH history, as compaction and
* pruning do. System prompt rewrites are excluded: configure() already carries the
* system prompt to Claude on every turn.
*/
function isReplaceEvent(event) {
	return event?.type !== "system/message" && typeof event?.surfaceOp === "object" && event.surfaceOp !== null && Number.isSafeInteger(event.surfaceOp.startSeq);
}
function permissionConfiguration(events) {
	let sandbox = "workspace-write";
	let approvalPolicy = "on-request";
	for (const event of events) {
		if (event.type === "sandbox/mode") sandbox = event.data.mode;
		if (event.type === "approval/policy") approvalPolicy = event.data.policy === "never" ? "never" : "on-request";
		if (event.type === "permission/preset") sandbox = event.data.preset;
	}
	return {
		sandbox,
		approvalPolicy
	};
}
function reasoningText(item) {
	return [...item.summary ?? [], ...item.content ?? []].filter(Boolean).join("\n\n");
}
function humanize(value) {
	return String(value).replace(/([a-z])([A-Z])/g, "$1 $2").replace(/^./, (letter) => letter.toUpperCase());
}
function reasoningEffortName(value) {
	if (String(value) === "xhigh") return "Extra high";
	if (String(value) === "ultracode") return "Ultracode";
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
		if (message?.role !== "user" || !hasUserInput(message)) continue;
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
	if (typeof message?.toolCallId === "string") {
		for (const result of results) {
			const text = resultText(result);
			if (text) parts.push(`${result.isError ? "tool failed" : "tool result"}: ${bound(text, 1500)}`);
		}
		return parts.join("\n").trim();
	}
	for (const block of message?.content ?? []) {
		if (block?.type === "text" && block.text?.trim()) parts.push(block.text.trim());
		if (block?.type === "image") parts.push(`[image: ${imageName(block)}]`);
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
	return (Array.isArray(content) ? content : []).filter((inner) => inner?.type === "text").map((inner) => inner.text).join("\n").trim();
}
/** Longest edge, in pixels, of an image sent to Claude. Claude downsizes larger images itself. */
const IMAGE_MAX_EDGE = 4096;
/** Encoded-byte target per image. The API limits the base64 text to 5 MB, and base64 is a third larger than the bytes. */
const IMAGE_MAX_BYTES = 375e4;
/** Base64 text one message may carry in images. A request is capped at 32 MB, so images past this are omitted. */
const IMAGE_BYTES_PER_MESSAGE = 15e6;
/** The size an image is requested at: its longest edge capped, and the byte target. */
function imageRequestTarget(ref) {
	const longest = Math.max(ref.width, ref.height);
	const scale = longest > IMAGE_MAX_EDGE ? IMAGE_MAX_EDGE / longest : 1;
	return {
		width: Math.max(1, Math.round(ref.width * scale)),
		height: Math.max(1, Math.round(ref.height * scale)),
		maxBytes: IMAGE_MAX_BYTES
	};
}
/** The name an image is shown under: the one DSH gave it, else its opaque id. */
function imageName(block) {
	return block?.attachment?.name ?? block?.attachment?.attachmentId ?? "image";
}
/** True when a DSH message carries something the user typed: text, or an image. */
function hasUserInput(message) {
	return messageText(message) !== "" || (message?.content ?? []).some((block) => block?.type === "image");
}
function bound(value, limit) {
	const text = String(value ?? "");
	return text.length > limit ? `${text.slice(0, limit)}…` : text;
}
function messageText(message) {
	return (message?.content ?? []).filter((block) => block.type === "text").map((block) => block.text).join("\n").trim();
}
function auxiliaryInput(messages) {
	return messages.map((message) => {
		const text = (message?.content ?? []).filter((block) => block.type === "text").map((block) => block.text).join("\n").trim();
		return text ? `${message.role ?? "user"}: ${text}` : "";
	}).filter(Boolean).join("\n\n");
}
function projectAuxiliaryActivity(message, state) {
	const params = message.params ?? {};
	if (message.method === "item/reasoning/summaryTextDelta" || message.method === "item/reasoning/textDelta") return textDelta(state, params.itemId, "reasoning", params.delta ?? "");
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
	return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== void 0 && item !== null));
}
function runtimeModels(runtime) {
	return typeof runtime.listModels === "function" ? runtime.listModels() : [...runtime.models];
}
function hasRuntimeSession(runtime, sessionId) {
	return typeof runtime.hasSession === "function" ? runtime.hasSession(sessionId) : runtime.sessions.has(sessionId);
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
//#endregion
//#region claude-link-store.js
var ClaudeLinkStore = class {
	constructor(path) {
		this.path = path;
		this.records = loadRecords(path);
	}
	entries() {
		return [...this.records.entries()].map(([sessionId, record]) => [sessionId, structuredClone(record)]);
	}
	set(sessionId, record) {
		this.records.set(String(sessionId), structuredClone(record));
		this.persist();
	}
	delete(sessionId) {
		if (!this.records.delete(String(sessionId))) return;
		this.persist();
	}
	persist() {
		mkdirSync(dirname(this.path), { recursive: true });
		const temporary = `${this.path}.${process.pid}.tmp`;
		const value = Object.fromEntries([...this.records.entries()].sort(([left], [right]) => left.localeCompare(right)));
		writeFileSync(temporary, `${JSON.stringify({
			version: 1,
			sessions: value
		}, null, 2)}\n`, { mode: 384 });
		renameSync(temporary, this.path);
	}
};
function loadRecords(path) {
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8"));
		if (parsed?.version !== 1 || !isObject(parsed.sessions)) return /* @__PURE__ */ new Map();
		return new Map(Object.entries(parsed.sessions).filter(([, record]) => validRecord(record)));
	} catch (error) {
		if (error?.code === "ENOENT") return /* @__PURE__ */ new Map();
		throw new Error(`Unable to read Claude DSH links from ${path}: ${error.message}`, { cause: error });
	}
}
function validRecord(record) {
	return isObject(record) && (record.claudeSessionId === null || typeof record.claudeSessionId === "string") && isObject(record.config);
}
function isObject(value) {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
//#endregion
//#region claude-tools.js
async function handleClaudeSdkRequest(ctx, { adapter, runtime, request }) {
	const claudeSessionId = request.params?.sessionId;
	const dshSessionId = claudeSessionId ? adapter.dshSessionForClaudeSession(claudeSessionId) : null;
	const agent = dshSessionId ? ctx.agents.get(dshSessionId) : null;
	if (!agent) {
		runtime.rejectRequest(request.id, /* @__PURE__ */ new Error("Claude request has no owning live DSH Session"));
		return;
	}
	try {
		if (request.method === "tool/requestApproval") {
			const outcome = await ctx.approval.request({
				agent,
				toolName: approvalToolName(request),
				reason: approvalReason(request),
				signal: request.signal
			});
			await runtime.resolveRequest(request.id, {
				action: outcome === "allowed-once" ? "accept" : "decline",
				updatedInput: request.params?.input,
				message: `DSH approval returned ${outcome}.`
			});
			return;
		}
		if (request.method === "tool/requestUserInput") {
			const questions = normalizeQuestions(request.params?.input?.questions ?? []);
			const answer = await ctx.userQuestions.ask({
				agent,
				questions,
				signal: request.signal
			});
			await runtime.resolveRequest(request.id, {
				action: "answer",
				answers: normalizeAnswers(answer, questions)
			});
			return;
		}
		runtime.rejectRequest(request.id, /* @__PURE__ */ new Error(`Unsupported Claude interaction ${request.method}`));
	} catch (error) {
		runtime.rejectRequest(request.id, error);
	}
}
function approvalToolName(request) {
	const display = request.params?.displayName;
	if (typeof display === "string" && display.trim()) return `Claude ${display.trim()}`;
	const tool = request.params?.toolName;
	return tool ? `Claude ${tool}` : "Claude tool";
}
function approvalReason(request) {
	const params = request.params ?? {};
	const input = plainObject(params.input);
	if (typeof params.title === "string" && params.title.trim()) return params.title.trim();
	if (typeof params.description === "string" && params.description.trim()) return params.description.trim();
	if (typeof params.decisionReason === "string" && params.decisionReason.trim()) return params.decisionReason.trim();
	if (typeof input.command === "string" && input.command.trim()) return input.command.trim();
	if (typeof input.file_path === "string" && input.file_path.trim()) return input.file_path.trim();
	return `${params.toolName ?? "Claude"} requires permission to continue.`;
}
function normalizeQuestions(input) {
	return input.slice(0, 3).map((question, index) => ({
		id: `question-${index + 1}`,
		question: requiredString(question.question ?? question.header ?? `Question ${index + 1}`, "question"),
		header: String(question.header ?? "Claude").slice(0, 12),
		options: normalizeOptions(question.options ?? []),
		multiSelect: Boolean(question.multiSelect),
		detail: typeof question.detail === "string" ? question.detail : void 0
	}));
}
function normalizeOptions(input) {
	if (!Array.isArray(input) || input.length === 0) return [{ label: "Continue" }, { label: "Cancel" }];
	return input.slice(0, 4).map((option) => ({
		label: requiredString(option.label ?? option, "option label"),
		description: typeof option.description === "string" ? option.description : void 0
	}));
}
function normalizeAnswers(answer, questions) {
	const byId = new Map(questions.map((question) => [question.id, question]));
	return Object.fromEntries((answer.answers ?? []).flatMap((entry) => {
		const question = byId.get(entry.id);
		if (!question) return [];
		const selected = [...entry.selected ?? [], ...entry.custom ? [entry.custom] : []].filter(Boolean);
		return [[question.question, question.multiSelect ? selected : selected[0] ?? ""]];
	}));
}
function requiredString(value, label) {
	const text = String(value ?? "").trim();
	if (!text) throw new Error(`Claude ${label} is required`);
	return text;
}
function plainObject(value) {
	return typeof value === "object" && value !== null && !Array.isArray(value) ? value : {};
}
//#endregion
//#region dsh-plugin.js
/** Logical channel this plugin serves; the composer ring is its only caller today. */
const AGENT_BRIDGE_CHANNEL = "/agent-bridge";
/**
* Serve the browser half's reads.
*
* `connection` is absent in headless profiles, so this mounts through
* `ctx.inject` rather than the plugin's own dependency list — a headless run
* must not park waiting for a service it never needs. Loopback authority
* matches the other plugin channels DSH ships: this answers the local UI only.
*
* @returns a disposer. `ctx.inject` hands back a fiber, not a function, and the
*   plugin host asserts on cleanups it cannot call.
*/
function serveUsageChannel(ctx, runtime) {
	const fiber = ctx.inject(["connection"], (scoped) => {
		const connection = scoped.get("connection");
		scoped.effect(() => connection.rpc.handle(AGENT_BRIDGE_CHANNEL, async (endpoint) => {
			if (endpoint !== "usage") return {
				ok: false,
				error: { message: `unknown ${AGENT_BRIDGE_CHANNEL} endpoint "${endpoint}"` }
			};
			try {
				return {
					ok: true,
					value: await runtime.planUsage()
				};
			} catch (error) {
				return {
					ok: false,
					error: { message: String(error?.message ?? error) }
				};
			}
		}, { authority: "loopback" }), `agent-bridge: ${AGENT_BRIDGE_CHANNEL} rpc channel`);
	});
	return () => fiber?.dispose?.();
}
function createDshClaudePlugin(ctx, config = {}) {
	return definePlugin({
		manifest: {
			id: "dsh.llm.agent-bridge",
			version: "1.0.0",
			provides: { "agent-bridge.dsh.v1": "1.0.0" },
			requires: { "agent-bridge.execution.v1": "^1.0.0" },
			permissions: ["dsh:llm", "dsh:agents"]
		},
		async activate({ capabilities, defer }) {
			const runtime = capabilities.require("agent-bridge.execution.v1");
			const telemetry = createTelemetryControl(ctx, {
				mode: config.telemetry,
				logger: ctx.logger
			});
			reportTelemetryPosture(telemetry, config.telemetry, ctx.logger);
			let attachments;
			const attachmentFiber = ctx.inject(["attachments"], (scoped) => {
				attachments = scoped.get("attachments");
			});
			defer(() => attachmentFiber?.dispose?.());
			const adapter = new ClaudeDshAdapter({
				runtime,
				ready: runtime.whenReady(),
				telemetry,
				resolveAttachments: () => attachments,
				thinkingSummaries: config.thinkingSummaries,
				linkStore: new ClaudeLinkStore(resolveLinkPath(config.linkPath)),
				logger: ctx.logger
			});
			defer(ctx.llm.registerAdapter([CLAUDE_PROVIDER], adapter));
			defer(() => adapter.dispose());
			defer(runtime.subscribeRequest((request) => {
				handleClaudeSdkRequest(ctx, {
					adapter,
					runtime,
					request
				}).catch((error) => ctx.logger.error(`Relay failed to handle a Claude interaction: ${error?.stack ?? error}`));
			}));
			defer(ctx.on("agent/created", ({ agent }) => {
				adapter.attachAgent(agent);
			}));
			defer(ctx.on("agent/disposed", ({ agent }) => {
				adapter.detachAgent(agent.id);
			}));
			defer(serveUsageChannel(ctx, runtime));
			for (const agent of ctx.agents.list()) adapter.attachAgent(agent);
			return { capabilities: { "agent-bridge.dsh.v1": Object.freeze({ provider: CLAUDE_PROVIDER }) } };
		}
	});
}
/** Log what the first Claude turn will do about a live exporter; never acts, never awaits. */
function reportTelemetryPosture(telemetry, mode, logger) {
	try {
		const sharing = telemetry.inspect?.();
		if (!sharing) return;
		const consequence = mode === "refuse" ? "Claude turns will be refused while it stays on" : "the first Claude turn will shut its exporter down for this process";
		logger.warn?.(`agent-bridge: DSH session telemetry is ${sharing}; ${consequence}. Set DSH_TELEMETRY_MODE=DISABLED in the shell that launches DSH to keep it off at the source.`);
	} catch (error) {
		logger.warn?.(`agent-bridge: could not read DSH's telemetry posture: ${error?.message ?? error}`);
	}
}
function resolveLinkPath(value) {
	const configured = value ?? process.env.DSH_AGENT_BRIDGE_LINK_PATH ?? process.env.DSH_CLAUDE_SDK_LINK_PATH;
	if (configured) return resolve(configured);
	const home = resolve(process.env.DSH_HOME?.trim() || join(homedir(), ".dsh"));
	const path = join(home, "plugin-data", "agent-bridge-links.json");
	const previous = join(home, "plugin-data", "claude-agent-sdk-links.json");
	if (!existsSync(path) && existsSync(previous)) try {
		renameSync(previous, path);
	} catch {
		return previous;
	}
	return path;
}
//#endregion
//#region host-plugin.js
const name = "dsh-llm-agent-bridge";
/**
* Only what the host half actually dereferences.
*
* `approval` and `userQuestions` back the permission and question bridges in
* claude-tools.js, `agents` and `llm` are read in dsh-plugin.js. Cordis throws
* "cannot get property 'x' without inject" at the point of use, so a service
* this plugin reads must be declared here.
*
* The converse matters just as much: cordis treats every declared name as
* REQUIRED and leaves the plugin inactive until all of them exist, so `apply()`
* never runs. This list previously also named `sessions`, `sessionPersistence`,
* `tools`, `typert` and `webServer`, none of which the host half ever reads.
* `webServer` is provided only by the web bundle, so on a headless profile the
* whole plugin parked and silently did nothing. Declaring a service this plugin
* does not use can only stop it from loading.
*
* `connection` is deliberately absent: dsh-plugin.js reaches it through
* `ctx.inject(["connection"], ...)`, which is optional and lets a headless run
* start without waiting for a browser transport it will never have.
*/
const inject = [
	"agents",
	"approval",
	"llm",
	"userQuestions"
];
async function apply(ctx, config = {}) {
	const host = new PluginHost();
	const release = ctx.effect(() => () => host.dispose(), "agent-bridge()");
	try {
		if (config.bifrostBaseUrl) ctx.logger?.info?.(`Claude transport endpoint: ${config.bifrostBaseUrl}`);
		await host.activate([createClaudeExecutionPlugin({
			client: config.claude?.client,
			backend: config.claudeBackend,
			command: config.claudeCommand,
			args: config.claudeArgs,
			codeExecutablePath: config.claudeCodeExecutablePath,
			bifrostBaseUrl: config.bifrostBaseUrl,
			requestTimeoutMs: config.claudeRequestTimeoutMs,
			cwd: config.cwd
		}), createDshClaudePlugin(ctx, config)]);
	} catch (error) {
		await release();
		throw error;
	}
}
//#endregion
export { apply, inject, name };

//# sourceMappingURL=host-plugin.js.map