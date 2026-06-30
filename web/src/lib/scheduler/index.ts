// ── Scheduler module: public API ──────────────────────────────────────────────
//
// Drop this `scheduler/` folder into the Buhera TypeScript project. Buhera:
//   1. implements `OsPort` with its real OS (module registry + audit log),
//   2. builds `LiveTask`s from the orchestrator's task graph,
//   3. constructs `new Scheduler(tasks, os)` and drives `tick()` / `run()`.
//
// The scheduler ranks tasks by measured residue and its descent, never by
// predicted running time; it reports residue and stalls; the orchestrator's
// gate sets `Sufficient`. Nothing here invents data — every decision is driven
// by real `ActResult`s returned through `OsPort`.

export * from './types';
export { trajectoryCount, nTerm, nTermOf, reclassify } from './complexity';
export { Scheduler, priority, descentOf } from './scheduler';
export type { TickOutcome } from './scheduler';
