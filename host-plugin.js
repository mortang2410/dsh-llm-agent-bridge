import { PluginHost } from "./internal/plugin-sdk.mjs";
import { createClaudeExecutionPlugin } from "./plugin.mjs";
import { createDshClaudePlugin } from "./dsh-plugin.js";

export const name = "dsh-llm-agent-bridge";
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
export const inject = ["agents", "approval", "llm", "userQuestions"];

export async function apply(ctx, config = {}) {
  const host = new PluginHost();
  const release = ctx.effect(() => () => host.dispose(), "agent-bridge()");
  try {
    if (config.bifrostBaseUrl) {
      // One line, once, so a run's routing is provable from the log. No
      // credentials or authorization headers are ever logged.
      ctx.logger?.info?.(`Claude transport endpoint: ${config.bifrostBaseUrl}`);
    }
    await host.activate([
      createClaudeExecutionPlugin({
        client: config.claude?.client, backend: config.claudeBackend, command: config.claudeCommand,
        args: config.claudeArgs, codeExecutablePath: config.claudeCodeExecutablePath,
        bifrostBaseUrl: config.bifrostBaseUrl,
        requestTimeoutMs: config.claudeRequestTimeoutMs, cwd: config.cwd,
      }),
      createDshClaudePlugin(ctx, config),
    ]);
  } catch (error) { await release(); throw error; }
}
