// ── Scheduler types ───────────────────────────────────────────────────────────
//
// The vocabulary shared between Buhera OS, the orchestrator (kwasa-kwasa), and
// this scheduler, as fixed by the operations-architecture contract. Every
// interface crossing a component boundary is one of these types. The scheduler
// owns this file because, per the contract, the types are "defined once and
// appear in all three components"; Buhera implements `OsPort` and supplies
// `LiveTask`s, and imports the rest from here.
//
// Design notes
//  • The scheduler never invents data. It reads residue from the OS's
//    `ActResult` and ranks tasks by it. The numbers that drive every decision
//    come from real dispatches through `OsPort`, not from this module.
//  • `OsPort` is the single seam to Buhera. The scheduler names no concrete
//    Buhera type — only "something that can dispatch a unit of work and return
//    its result". Buhera's real OS implements it.
//  • Sufficiency is NOT computed here (No-Local-Necessity): the orchestrator's
//    gate sets `state = 'Sufficient'` and the scheduler reads it. The scheduler
//    measures; it does not judge sufficiency.

// Identifiers. 128-bit values are carried as strings (es5 target has no bigint
// literal support and these are opaque to the scheduler anyway).
export type TaskId = string;
export type ModuleId = string;
export type ActId = string;

/** The composition-inflation clock state at commit, opaque to the scheduler. */
export type TrajectoryState = string;

export type TaskState =
  | 'Pending'
  | 'Running'
  | 'Sufficient'   // set by the orchestrator's gate, read by the scheduler
  | 'Stalled'      // residue stopped descending (structure-limited)
  | 'Terminated'
  | 'Declined';

/** Categorical complexity of a task: K distinguishable trajectories over d types. */
export interface CategoricalComplexity {
  /** Distinguishable trajectories required (K >= 1). */
  k: number;
  /** Operation types (default 4). */
  d: number;
}

/** Module-specific instruction bytes; opaque to the scheduler. */
export type ModuleInstruction = unknown;

/** A unit of work the scheduler asks the OS to perform. */
export interface Dispatch {
  taskId: TaskId;
  moduleId: ModuleId;
  instruction: ModuleInstruction;
  /** Thoroughness budget for this act (how hard the module should work). */
  actBudget: number;
}

/** The result of one dispatched act, returned by the OS. */
export interface ActResult {
  taskId: TaskId;
  actId: ActId;
  trajectoryState: TrajectoryState;
  /**
   * The task's current residue AFTER this act: measured distance of the
   * partial result from a recognised solution, on [floor, U]. This is the
   * scheduler's only progress signal. Lower is closer to done.
   */
  residue: number;
  /** Cost deposited by this act (>= the floor). For accounting / audit. */
  cost: number;
  /** The module's own "I'm finished" flag (a hard completion, distinct from
   *  sufficiency, which the orchestrator decides). */
  completed: boolean;
}

/** A live task the scheduler is advancing. */
export interface LiveTask {
  id: TaskId;
  moduleId: ModuleId;
  complexity: CategoricalComplexity;
  /**
   * Sufficiency threshold supplied top-down by the orchestrator: the residue
   * at or below which this task is good enough for the parent goal. The
   * scheduler compares residue against it but never computes it.
   */
  sufficiencyThreshold: number;
  /** The residue floor for this task's resolving agent (> 0). */
  floor: number;
  /** Committed units so far (the task's monotone trajectory count). */
  committedActs: number;
  /** Most recent measured residue (from the last ActResult). */
  residue: number;
  /** Recent residue-descent rate over the descent window (>= 0). */
  descent: number;
  /** Window of recent residues, newest last, used to compute `descent`. */
  residueWindow: number[];
  state: TaskState;
  /** Next instruction to dispatch for this task (module-specific). */
  nextInstruction: ModuleInstruction;
  parent?: TaskId;
  dependencies?: TaskId[];
}

/**
 * The single seam to Buhera OS. The scheduler dispatches one unit of work and
 * awaits its result. Buhera's real OS (module registry + audit log + substrate)
 * implements this. The scheduler holds no other reference to the OS.
 */
export interface OsPort {
  dispatch(dispatch: Dispatch): Promise<ActResult>;
}

/** Why the scheduler stopped advancing a task or the whole set. */
export type DeclineReason =
  | { kind: 'complexity-overrun'; taskId: TaskId; committedActs: number; nTerm: number }
  | { kind: 'all-stalled'; taskIds: TaskId[] }
  | { kind: 'deadlock'; idleTicks: number };

/** A decision the scheduler surfaces to the orchestrator. */
export type SchedulerEvent =
  | { kind: 'dispatched'; result: ActResult }
  | { kind: 'task-sufficient'; taskId: TaskId }
  | { kind: 'task-completed'; taskId: TaskId }
  | { kind: 'task-stalled'; taskId: TaskId }
  | { kind: 'reclassify'; taskId: TaskId; oldK: number; newK: number }
  | { kind: 'decline'; reason: DeclineReason };

/** Tuning knobs for the scheduler. All have sane defaults. */
export interface SchedulerConfig {
  /** Work units a single tick may dispatch (the tick budget). */
  tickBudget: number;
  /** Length of the residue-descent window used to detect stalls. */
  descentWindow: number;
  /** Consecutive units of zero descent that mark a stall. */
  stallWindow: number;
  /** Default operation-type count when a task does not specify one. */
  defaultTypes: number;
  /** Ticks with no system progress before declaring deadlock. */
  deadlockTicks: number;
}

export const DEFAULT_CONFIG: SchedulerConfig = {
  tickBudget: 8,
  descentWindow: 4,
  stallWindow: 3,
  defaultTypes: 4,
  deadlockTicks: 4,
};
