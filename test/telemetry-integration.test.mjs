import assert from "node:assert/strict";
import { createServer } from "node:http";
import test from "node:test";

import { Context } from "@deepseek-ai/cordis";
import { OpenTelemetrySessionBackend } from "@deepseek-ai/dsh-session-telemetry-otel";

import { createTelemetryControl, exportingState } from "../telemetry-control.js";

/** A collector on loopback, so a guard failure is observable and still local. */
async function localCollector() {
  const requests = [];
  const server = createServer((request, response) => {
    requests.push(request.url);
    response.writeHead(200, { "content-type": "application/json" });
    response.end("{}");
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  return {
    requests,
    url: `http://127.0.0.1:${server.address().port}/v1/logs`,
    async close() { await new Promise(resolve => server.close(resolve)); },
  };
}

async function mountBackend(ctx, mode, collector) {
  ctx.provide("sessions");
  ctx.sessions = { list: () => [], get: () => undefined };
  await ctx.plugin(OpenTelemetrySessionBackend, {
    mode,
    ...(collector ? {
      exporter: { url: collector.url, timeoutMillis: 500 },
      processor: { scheduledDelayMillis: 50, maxQueueSize: 8, maxExportBatchSize: 8 },
      shutdownTimeoutMillis: 1000,
    } : {}),
  });
}

/** Push a record through the SDK pipeline the coordinator would use, and give it time to export. */
async function exportThrough(backend, body) {
  backend.provider.getLogger("test").emit({ body, severity: "info", attributes: {} });
  await new Promise(resolve => setTimeout(resolve, 250));
}

test("a real exporting backend's pipeline is shut down, while its row stays mounted", async (context) => {
  const collector = await localCollector();
  context.after(() => collector.close());

  const ctx = new Context();
  await mountBackend(ctx, "FEEDBACK_ONLY", collector);

  // The genuine article, uploading, before the guard runs.
  const backend = ctx.get("sessionTelemetry");
  assert.equal(exportingState(backend), "feedback-only");
  await exportThrough(backend, "before");
  assert.equal(collector.requests.length, 1, "the pipeline must be shown live before the guard acts, or the test proves nothing");

  // Count the SDK shutdowns the guard actually triggers. `provider` is a plain
  // property, so this reaches the real LoggerProvider behind cordis's proxy.
  let shutdowns = 0;
  const providerShutdown = backend.provider.shutdown.bind(backend.provider);
  backend.provider.shutdown = () => { shutdowns += 1; return providerShutdown(); };

  const warnings = [];
  const control = createTelemetryControl(ctx, { logger: { warn: message => warnings.push(message) } });
  assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "feedback-only" });
  assert.equal(shutdowns, 1);
  assert.match(warnings[0], /has been shut down/);

  // The row is another plugin's; it stays where the host put it, still
  // describing the mode it was configured with. Nothing here disposes its fiber.
  assert.equal(exportingState(ctx.get("sessionTelemetry")), "feedback-only", "the row must remain mounted");

  // But nothing may reach the collector afterwards: a shut-down provider drops records.
  await exportThrough(backend, "after");
  assert.equal(collector.requests.length, 1, "no records may leave after the guard has run");

  // And later turns neither refuse nor shut it down again — cordis hands out a
  // new proxy per ctx.get(), so this is the identity handling under test.
  assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "feedback-only" });
  assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "feedback-only" });
  assert.equal(shutdowns, 1, "the same instance is shut down once");
  assert.equal(warnings.length, 1);
});

test("a real disabled backend is left mounted and untouched", async () => {
  const ctx = new Context();
  await mountBackend(ctx, "DISABLED");

  const control = createTelemetryControl(ctx, { logger: { warn() {} } });

  assert.deepEqual(await control.enforce(), { state: "off" });
  assert.notEqual(ctx.get("sessionTelemetry"), undefined, "a quiet host keeps its own row");
});
