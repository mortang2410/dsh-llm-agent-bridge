import { existsSync, renameSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { definePlugin } from "./internal/plugin-sdk.mjs";
import { ClaudeDshAdapter, CLAUDE_PROVIDER } from "./claude-adapter.js";
import { ClaudeLinkStore } from "./claude-link-store.js";
import { createTelemetryControl } from "./telemetry-control.js";
import { handleClaudeSdkRequest } from "./claude-tools.js";

/** Logical channel this plugin serves; the composer ring is its only caller today. */
export const AGENT_BRIDGE_CHANNEL = "/agent-bridge";

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
      if (endpoint !== "usage") {
        return { ok: false, error: { message: `unknown ${AGENT_BRIDGE_CHANNEL} endpoint "${endpoint}"` } };
      }
      try {
        return { ok: true, value: await runtime.planUsage() };
      } catch (error) {
        return { ok: false, error: { message: String(error?.message ?? error) } };
      }
    }, { authority: "loopback" }), `agent-bridge: ${AGENT_BRIDGE_CHANNEL} rpc channel`);
  });
  return () => fiber?.dispose?.();
}

export function createDshClaudePlugin(ctx, config = {}) {
  return definePlugin({
    manifest: {
      id: "dsh.llm.agent-bridge", version: "1.0.0", provides: { "agent-bridge.dsh.v1": "1.0.0" },
      requires: { "agent-bridge.execution.v1": "^1.0.0" }, permissions: ["dsh:llm", "dsh:agents"],
    },
    async activate({ capabilities, defer }) {
      const runtime = capabilities.require("agent-bridge.execution.v1");
      const telemetry = createTelemetryControl(ctx, { mode: config.telemetry, logger: ctx.logger });
      // The guard runs at the start of every Claude turn (ClaudeDshAdapter), not
      // here. Activation runs inside the cordis loader's boot, and acting on
      // another plugin's row from there — dsh 0.1.5 mounts the exporter by
      // default — deadlocked the loader and locked the operator out of the UI.
      // This only reads, synchronously, so the operator sees the posture in
      // the log before the first turn is refused or shuts the exporter down.
      reportTelemetryPosture(telemetry, config.telemetry, ctx.logger);
      const adapter = new ClaudeDshAdapter({
        runtime, ready: runtime.whenReady(), telemetry,
        thinkingSummaries: config.thinkingSummaries,
        linkStore: new ClaudeLinkStore(resolveLinkPath(config.linkPath)), logger: ctx.logger,
      });
      defer(ctx.llm.registerAdapter([CLAUDE_PROVIDER], adapter));
      defer(() => adapter.dispose());
      defer(runtime.subscribeRequest(request => {
        void handleClaudeSdkRequest(ctx, { adapter, runtime, request })
          .catch(error => ctx.logger.error(`Relay failed to handle a Claude interaction: ${error?.stack ?? error}`));
      }));
      defer(ctx.on("agent/created", ({ agent }) => { adapter.attachAgent(agent); }));
      defer(ctx.on("agent/disposed", ({ agent }) => { adapter.detachAgent(agent.id); }));
      defer(serveUsageChannel(ctx, runtime));
      for (const agent of ctx.agents.list()) adapter.attachAgent(agent);
      return { capabilities: { "agent-bridge.dsh.v1": Object.freeze({ provider: CLAUDE_PROVIDER }) } };
    },
  });
}

/** Log what the first Claude turn will do about a live exporter; never acts, never awaits. */
function reportTelemetryPosture(telemetry, mode, logger) {
  try {
    const sharing = telemetry.inspect?.();
    if (!sharing) return;
    const consequence = mode === "refuse"
      ? "Claude turns will be refused while it stays on"
      : "the first Claude turn will shut its exporter down for this process";
    logger.warn?.(
      `agent-bridge: DSH session telemetry is ${sharing}; ${consequence}. `
      + "Set DSH_TELEMETRY_MODE=DISABLED in the shell that launches DSH to keep it off at the source.",
    );
  } catch (error) {
    logger.warn?.(`agent-bridge: could not read DSH's telemetry posture: ${error?.message ?? error}`);
  }
}

function resolveLinkPath(value) {
  const configured = value ?? process.env.DSH_AGENT_BRIDGE_LINK_PATH ?? process.env.DSH_CLAUDE_SDK_LINK_PATH;
  if (configured) return resolve(configured);
  const home = resolve(process.env.DSH_HOME?.trim() || join(homedir(), ".dsh"));
  const path = join(home, "plugin-data", "agent-bridge-links.json");
  // Adopt the pre-rename file so sessions keep their Claude session links.
  const previous = join(home, "plugin-data", "claude-agent-sdk-links.json");
  if (!existsSync(path) && existsSync(previous)) {
    try {
      renameSync(previous, path);
    } catch {
      return previous;
    }
  }
  return path;
}
