import { telemetryExport, telemetryRefusal } from "./telemetry.js";

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
export const TELEMETRY_MODES = ["disable", "refuse", "ignore"];

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
export function exportingState(backend) {
  if (!backend) return null;
  if (typeof backend.sharing === "string") return backend.sharing === "disabled" ? null : backend.sharing;
  return backend.provider === undefined ? null : "unknown";
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
export function createTelemetryControl(ctx, { mode = "disable", logger = console } = {}) {
  const setting = TELEMETRY_MODES.includes(mode) ? mode : "disable";
  // Backend instances whose shutdown() has resolved. The row stays mounted and
  // keeps reporting the sharing mode it was configured with — that field is
  // configuration, not liveness — so the instance is what has to be remembered.
  // A different instance (a reload, another plugin re-mounting the row) is not
  // in the set and is treated as live.
  const quiesced = new WeakSet();
  // Records only that we shut an exporter down, so the stale environment value
  // that configured it is not mistaken for a live one once the row is gone. It
  // never short-circuits the host check itself.
  let removed = null;

  function couldNotDisable(sharing, reason) {
    return new Error(
      `DSH session telemetry is ${sharing} and could not be turned off${reason ? ` (${reason})` : ""}. `
      + "Claude is not available while it is on. Set DSH_TELEMETRY_MODE=DISABLED (or DSH_TELEMETRY_DISABLED=1) in the shell that launches DSH, then restart it.",
    );
  }

  async function disable(backend, sharing) {
    if (typeof backend.shutdown !== "function") throw couldNotDisable(sharing, "the backend has no shutdown()");
    // Drain and quiesce the pipeline through the backend's own API. Nothing is
    // unmounted: the row keeps answering ctx.get(), but its provider is shut
    // down and drops whatever the coordinator hands it afterwards.
    try {
      await backend.shutdown();
    } catch (error) {
      // The backend's own deadline elapsed or the SDK refused. Both paths leave
      // the exporter in a state nothing out here can confirm, and a guard that
      // believes itself while an exporter is still live is worse than one that
      // fails the turn. Not memoized: the next turn tries again.
      throw couldNotDisable(sharing, error?.message ?? String(error));
    }
    quiesced.add(identityOf(backend));
    removed = sharing;
    logger.warn?.(
      `agent-bridge: DSH session telemetry was ${sharing} and its exporter has been shut down for this process. `
      + "It uploads unredacted session records, which would include Claude's output. "
      + "Set DSH_TELEMETRY_MODE=DISABLED to keep it off at the source, or configure this plugin with telemetry: \"refuse\" to be failed instead.",
    );
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

      // Re-read the host every turn. ctx.get() is cheap, and a row that came
      // back after a teardown must be caught rather than assumed gone.
      const backend = liveBackend(ctx);
      const sharing = exportingState(backend);
      if (sharing) {
        // The same instance we already shut down: its provider drops records,
        // however the row still describes itself.
        if (quiesced.has(identityOf(backend))) return { state: "disabled", sharing };
        if (setting === "refuse") {
          throw new Error(telemetryRefusal(telemetryExport() ?? { mode: sharing.toUpperCase(), endpoint: "DSH's configured collector", source: "the running exporter" }));
        }
        await disable(backend, sharing);
        return { state: "disabled", sharing };
      }

      // Quiet because we made it so: the environment or config layer that
      // switched the exporter on is now stale, and must not read as live.
      if (removed) return { state: "disabled", sharing: removed };

      // No readable backend: fall back to the config scan, which can only
      // report, not disable.
      const configured = telemetryExport();
      if (configured) throw new Error(telemetryRefusal(configured));
      return { state: "off" };
    },
  };
}

/** The guard used when no host context is available, e.g. in tests. */
export function scanOnlyTelemetryControl() {
  return {
    inspect() {
      return telemetryExport()?.mode?.toLowerCase() ?? null;
    },
    async enforce() {
      const configured = telemetryExport();
      if (configured) throw new Error(telemetryRefusal(configured));
      return { state: "off" };
    },
  };
}
