import assert from "node:assert/strict";
import test from "node:test";

import { createTelemetryControl, exportingState, scanOnlyTelemetryControl } from "../telemetry-control.js";

function hostWithTelemetry(sharing, { shutdownError = null } = {}) {
  const record = { shutdown: 0, disposed: 0 };
  let mounted = true;
  let backend = makeBackend(sharing);
  function makeBackend(next) {
    return {
      sharing: next,
      provider: next === "disabled" ? undefined : {},
      async shutdown() {
        record.shutdown += 1;
        if (shutdownError) throw shutdownError;
      },
      // The real row hands its cordis fiber out here. The guard must never
      // reach for it: disposing another plugin's fiber from inside this one
      // deadlocked dsh 0.1.5's boot.
      ctx: { fiber: { async dispose() { record.disposed += 1; mounted = false; } } },
    };
  }
  const ctx = { get: name => (name === "sessionTelemetry" && mounted ? backend : undefined) };
  return {
    ctx,
    record,
    backend: () => backend,
    unmount: () => { mounted = false; },
    /** A fresh instance, as a reload or another plugin re-mounting the row would produce. */
    remount: next => { backend = makeBackend(next); mounted = true; },
  };
}

test("the live backend decides, whatever switched it on", () => {
  assert.equal(exportingState({ sharing: "full" }), "full");
  assert.equal(exportingState({ sharing: "feedback-only" }), "feedback-only");
  assert.equal(exportingState({ sharing: "disabled" }), null);
  assert.equal(exportingState(null), null);
  // A backend that reports no sharing status still has a pipeline to read.
  assert.equal(exportingState({ provider: {} }), "unknown");
  assert.equal(exportingState({ provider: undefined }), null);
});

test("an exporting host is shut down on the first turn, and its row is never unmounted", async () => {
  const { ctx, record } = hostWithTelemetry("feedback-only");
  const warnings = [];
  const control = createTelemetryControl(ctx, { logger: { warn: message => warnings.push(message) } });

  assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "feedback-only" });
  assert.equal(record.shutdown, 1, "the pipeline must be drained and quiesced through the backend's own API");
  assert.equal(record.disposed, 0, "the row belongs to another plugin; its fiber is not ours to dispose");
  assert.match(warnings[0], /has been shut down/);
  assert.match(warnings[0], /DSH_TELEMETRY_MODE=DISABLED/, "the operator must be told how to keep it off at the source");
});

test("a host that is already quiet is left alone", async () => {
  const { ctx, record } = hostWithTelemetry("disabled");
  const control = createTelemetryControl(ctx, { logger: { warn() {} } });

  assert.deepEqual(await control.enforce(), { state: "off" });
  assert.equal(record.shutdown, 0);
  assert.equal(record.disposed, 0);
});

test("the same instance is not shut down twice, even though it still describes itself as exporting", async () => {
  // The real backend's `sharing` is configuration, fixed in its constructor; a
  // resolved shutdown() does not change it. The instance is what is remembered.
  const { ctx, record, backend } = hostWithTelemetry("full");
  const control = createTelemetryControl(ctx, { logger: { warn() {} } });

  assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "full" });
  assert.equal(backend().sharing, "full", "the row still reports the mode it was configured with");
  assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "full" });
  assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "full" });
  assert.equal(record.shutdown, 1);
});

test("later turns do not re-run the config scan against an exporter we already shut down", async () => {
  const { ctx, record, unmount } = hostWithTelemetry("full");
  const control = createTelemetryControl(ctx, { logger: { warn() {} } });
  await control.enforce();
  unmount();

  // The row is gone, so the environment still says FULL while nothing exports.
  process.env.DSH_TELEMETRY_MODE = "FULL";
  try {
    assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "full" });
    assert.equal(record.shutdown, 1, "and it is not shut down twice");
  } finally {
    delete process.env.DSH_TELEMETRY_MODE;
  }
});

test("refuse mode fails the turn and leaves the host untouched", async () => {
  const { ctx, record } = hostWithTelemetry("full");
  const control = createTelemetryControl(ctx, { mode: "refuse", logger: { warn() {} } });

  await assert.rejects(() => control.enforce(), /telemetry is enabled/);
  assert.equal(record.shutdown, 0, "refuse must not touch host configuration");
  assert.equal(record.disposed, 0);
});

test("ignore mode does nothing at all", async () => {
  const { ctx, record } = hostWithTelemetry("full");
  const control = createTelemetryControl(ctx, { mode: "ignore", logger: { warn() {} } });

  assert.deepEqual(await control.enforce(), { state: "ignored" });
  assert.equal(control.inspect(), null);
  assert.equal(record.shutdown, 0);
});

test("an unreadable host still refuses on the config scan", async () => {
  const control = createTelemetryControl({ get: () => undefined }, { logger: { warn() {} } });
  process.env.DSH_TELEMETRY_MODE = "FULL";
  try {
    await assert.rejects(() => control.enforce(), /telemetry is enabled/);
  } finally {
    delete process.env.DSH_TELEMETRY_MODE;
  }
});

test("a backend without shutdown() refuses rather than claiming success", async () => {
  const backend = { sharing: "full", provider: {} };
  const ctx = { get: name => (name === "sessionTelemetry" ? backend : undefined) };
  const control = createTelemetryControl(ctx, { logger: { warn() {} } });

  await assert.rejects(() => control.enforce(), /could not be turned off/);
});

test("the scan-only control never reaches for a host", async () => {
  assert.deepEqual(await scanOnlyTelemetryControl().enforce(), { state: "off" });
  assert.equal(scanOnlyTelemetryControl().inspect(), null);
});

test("a shutdown that rejects degrades to refusing, and is never memoized", async () => {
  // The backend's own deadline elapsed: the exporter's state cannot be confirmed.
  const { ctx, record } = hostWithTelemetry("full", { shutdownError: new Error("provider shutdown exceeded 3000ms") });
  const control = createTelemetryControl(ctx, { logger: { warn() {} } });

  await assert.rejects(() => control.enforce(), /could not be turned off[\s\S]*exceeded 3000ms/);
  // The failure must not stick: a later turn tries again rather than assuming.
  await assert.rejects(() => control.enforce(), /could not be turned off/);
  assert.equal(record.shutdown, 2);
  assert.equal(record.disposed, 0, "a failed shutdown is no licence to start unmounting rows");

  const backend = ctx.get("sessionTelemetry");
  backend.sharing = "disabled";
  backend.provider = undefined;
  assert.deepEqual(await control.enforce(), { state: "off" }, "and it recovers once the host is actually quiet");
});

test("a fresh exporter instance after a shutdown is caught, not assumed gone", async () => {
  const { ctx, record, remount } = hostWithTelemetry("full");
  const control = createTelemetryControl(ctx, { logger: { warn() {} } });

  assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "full" });
  assert.equal(record.shutdown, 1);

  // Something put a new row up — a reload, another plugin, a host that
  // restores it. The same guard must catch it rather than answering from memory.
  remount("feedback-only");
  assert.equal(control.inspect(), "feedback-only", "the new instance reads as live");
  assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "feedback-only" });
  assert.equal(record.shutdown, 2, "the new instance is shut down in its turn");
  assert.equal(record.disposed, 0);
});

test("after a shutdown, a stale environment value is not mistaken for a live exporter", async () => {
  const { ctx, unmount } = hostWithTelemetry("full");
  const control = createTelemetryControl(ctx, { logger: { warn() {} } });
  await control.enforce();
  unmount();

  process.env.DSH_TELEMETRY_MODE = "FULL";
  try {
    assert.deepEqual(await control.enforce(), { state: "disabled", sharing: "full" });
  } finally {
    delete process.env.DSH_TELEMETRY_MODE;
  }
});

test("inspect() reads the host without acting on it", async () => {
  const { ctx, record } = hostWithTelemetry("feedback-only");
  const control = createTelemetryControl(ctx, { logger: { warn() {} } });

  assert.equal(control.inspect(), "feedback-only");
  assert.equal(record.shutdown, 0);
  assert.equal(record.disposed, 0);

  await control.enforce();
  assert.equal(control.inspect(), null, "a quiesced instance no longer reads as live");
});
