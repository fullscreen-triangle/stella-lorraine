// ── Residue-driven scheduler ──────────────────────────────────────────────────
//
// Implements the Scheduler component of the operations-architecture contract,
// with the residue-driven priority rule (rather than the static
// remaining-units rule of the contract's first draft). The scheduler ranks
// tasks by MEASURED residue and its rate of descent, not by predicted running
// time, which is unknowable. On each tick it selects the highest-priority
// runnable task, dispatches one unit of work through the OS port, reads the
// returned residue, updates the descent estimate, and repeats until the tick
// budget is spent or every runnable task is stalled.
//
// Responsibilities (and the boundaries it respects):
//  • DOES decide which task gets the next dispatch and how much budget it gets.
//  • DOES detect stalls (descent gone flat), complexity overruns (reclassify,
//    don't fail), and deadlock (no system progress).
//  • DOES NOT compute sufficiency — it reads `state === 'Sufficient'`, set by
//    the orchestrator's gate. It reports residue; the gate judges enough.
//  • DOES NOT invent results — every number driving a decision comes from a
//    real ActResult returned by the OS port.
//
// Priority (the amended rule, Thm "Priority-rule soundness"):
//
//     P(t) = descent(t) / max( residue(t) - threshold(t), floor(t) )
//
//   reaches +Inf when residue <= threshold (finish it now), is 0 when descent
//   is 0 (stalled — do not feed it), and otherwise rewards both fast descent
//   (numerator) and closeness to the threshold (denominator). This realises:
//   feed the converging, finish the close, starve the stalled.

import type {
  LiveTask,
  OsPort,
  SchedulerConfig,
  SchedulerEvent,
  Dispatch,
  ActResult,
  DeclineReason,
} from './types';
import { DEFAULT_CONFIG } from './types';
import { nTermOf, reclassify } from './complexity';

/** Residue-descent rate over a task's window: average drop per unit, >= 0. */
export function descentOf(window: number[]): number {
  if (window.length < 2) return 0;
  let drop = 0;
  let steps = 0;
  for (let i = 1; i < window.length; i++) {
    const d = window[i - 1] - window[i];
    drop += d;
    steps++;
  }
  const rate = steps > 0 ? drop / steps : 0;
  return rate > 0 ? rate : 0;
}

/**
 * Priority of a running task. +Inf at threshold, 0 when stalled, and a positive
 * bootstrap value before enough residue has been measured to estimate descent.
 *
 * The descent estimate needs at least two residue samples, so a freshly-admitted
 * task (window shorter than 2) has no measurable descent yet. Such a task must
 * still be dispatchable — otherwise it could never take its first unit and the
 * whole set would appear stalled at start. We give it a finite positive
 * bootstrap priority (below any genuinely-descending task once measured, above
 * zero) so it gets its first units, after which the real residue-driven rule
 * takes over. A task with >= 2 samples and zero descent is a genuine stall and
 * returns 0.
 */
export function priority(task: LiveTask): number {
  if (task.state !== 'Running') return 0;
  if (task.residue <= task.sufficiencyThreshold) return Infinity;
  // Bootstrap: not enough samples to know the descent yet -> let it run.
  if (task.residueWindow.length < 2) return BOOTSTRAP_PRIORITY;
  if (task.descent <= 0) return 0; // measured flat => stalled
  const denom = Math.max(task.residue - task.sufficiencyThreshold, task.floor);
  return task.descent / denom;
}

/** Positive priority for a task with too few samples to estimate descent. */
const BOOTSTRAP_PRIORITY = 1e-6;

/** True once a window of the most recent residues shows no descent. */
function isStalled(task: LiveTask, stallWindow: number): boolean {
  const w = task.residueWindow;
  if (w.length < stallWindow + 1) return false;
  const tail = w.slice(w.length - (stallWindow + 1));
  for (let i = 1; i < tail.length; i++) {
    if (tail[i - 1] - tail[i] > 1e-12) return false; // some descent => not stalled
  }
  // flat tail AND still above the floor => structure-limited stall
  return task.residue > task.floor + 1e-12;
}

export interface TickOutcome {
  events: SchedulerEvent[];
  /** Total units dispatched this tick. */
  dispatched: number;
  /** True if no task advanced this tick (all stalled / none runnable). */
  idle: boolean;
}

export class Scheduler {
  private tasks: LiveTask[];
  private os: OsPort;
  private cfg: SchedulerConfig;
  private idleTicks = 0;

  constructor(tasks: LiveTask[], os: OsPort, cfg?: Partial<SchedulerConfig>) {
    this.os = os;
    this.cfg = Object.assign({}, DEFAULT_CONFIG, cfg || {});
    // Normalise incoming tasks: ensure window/derived fields are present.
    this.tasks = tasks.map(function (t): LiveTask {
      return {
        id: t.id,
        moduleId: t.moduleId,
        complexity: t.complexity,
        sufficiencyThreshold: t.sufficiencyThreshold,
        floor: t.floor,
        committedActs: t.committedActs || 0,
        residue: typeof t.residue === 'number' ? t.residue : Infinity,
        descent: t.descent || 0,
        residueWindow: t.residueWindow ? t.residueWindow.slice() : [],
        state: t.state || 'Running',
        nextInstruction: t.nextInstruction,
        parent: t.parent,
        dependencies: t.dependencies,
      };
    });
  }

  /** Read-only snapshot of the current task set. */
  snapshot(): LiveTask[] {
    return this.tasks.map(function (t) {
      return Object.assign({}, t, { residueWindow: t.residueWindow.slice() });
    });
  }

  /** Tasks that are still candidates for dispatch. */
  private runnable(): LiveTask[] {
    return this.tasks.filter(function (t) {
      return t.state === 'Running';
    });
  }

  /** Sum of non-negative priorities over runnable tasks (for budget shares). */
  private totalPriority(): number {
    const rs = this.runnable();
    let s = 0;
    for (let i = 0; i < rs.length; i++) {
      const p = priority(rs[i]);
      if (isFinite(p) && p > 0) s += p;
    }
    return s;
  }

  /**
   * The next runnable task to dispatch, or null if none can make progress.
   *
   * A task that has never been dispatched (empty residue window) is selected
   * first, in admission order, so every task gets at least one unit and a
   * residue reading before residue-driven ranking begins. This prevents a
   * fast-descending task from starving tasks that have not yet started. Once
   * all runnable tasks have a residue reading, selection is by priority.
   */
  private pick(): LiveTask | null {
    const rs = this.runnable();
    // Phase 1: any un-started runnable task gets a turn first.
    for (let i = 0; i < rs.length; i++) {
      if (rs[i].residueWindow.length === 0) return rs[i];
    }
    // Phase 2: highest positive priority among started tasks.
    let best: LiveTask | null = null;
    let bestP = 0;
    for (let i = 0; i < rs.length; i++) {
      const p = priority(rs[i]);
      if (p > bestP) {
        bestP = p;
        best = rs[i];
      }
    }
    return best;
  }

  /** Apply one ActResult to a task: advance count, update residue/descent/state. */
  private commit(task: LiveTask, r: ActResult): SchedulerEvent[] {
    const events: SchedulerEvent[] = [];
    task.committedActs += 1;
    task.residue = r.residue;
    task.residueWindow.push(r.residue);
    if (task.residueWindow.length > this.cfg.descentWindow) {
      task.residueWindow.shift();
    }
    task.descent = descentOf(task.residueWindow);
    events.push({ kind: 'dispatched', result: r });

    // Hard completion reported by the module.
    if (r.completed) {
      task.state = 'Terminated';
      events.push({ kind: 'task-completed', taskId: task.id });
      return events;
    }

    // Sufficiency: at or below the orchestrator-supplied threshold.
    if (task.residue <= task.sufficiencyThreshold) {
      task.state = 'Sufficient';
      events.push({ kind: 'task-sufficient', taskId: task.id });
      return events;
    }

    // Complexity overrun: reclassify (do NOT fail) if still converging; only a
    // stall turns a converging-but-overrunning task into a decline.
    const nt = nTermOf(task.complexity);
    if (task.committedActs > nt) {
      const newK = reclassify(task.committedActs, task.complexity.d);
      if (newK > task.complexity.k) {
        const oldK = task.complexity.k;
        task.complexity = { k: newK, d: task.complexity.d };
        events.push({ kind: 'reclassify', taskId: task.id, oldK: oldK, newK: newK });
      }
    }

    // Stall: descent gone flat above the floor => structure-limited.
    if (isStalled(task, this.cfg.stallWindow)) {
      task.state = 'Stalled';
      events.push({ kind: 'task-stalled', taskId: task.id });
    }
    return events;
  }

  /** One scheduler tick. Dispatches up to tickBudget units in priority order. */
  async tick(): Promise<TickOutcome> {
    const events: SchedulerEvent[] = [];
    let budget = this.cfg.tickBudget;
    let dispatched = 0;

    while (budget > 0) {
      const task = this.pick();
      if (!task) break; // no runnable task with positive priority

      const total = this.totalPriority();
      const p = priority(task);
      // Budget share proportional to priority (a +Inf-priority task takes one
      // unit and finalises). Always at least one unit, never more than budget.
      let share = 1;
      if (isFinite(p) && total > 0) {
        share = Math.max(1, Math.ceil((p / total) * budget));
      }
      const actBudget = Math.min(budget, share);

      const dispatch: Dispatch = {
        taskId: task.id,
        moduleId: task.moduleId,
        instruction: task.nextInstruction,
        actBudget: actBudget,
      };

      const result = await this.os.dispatch(dispatch);
      const evs = this.commit(task, result);
      for (let i = 0; i < evs.length; i++) events.push(evs[i]);

      budget -= actBudget;
      dispatched += 1;
    }

    const idle = dispatched === 0;
    if (idle) {
      this.idleTicks += 1;
    } else {
      this.idleTicks = 0;
    }

    // Decline conditions, evaluated once per tick.
    const decline = this.declineCheck();
    if (decline) events.push({ kind: 'decline', reason: decline });

    return { events: events, dispatched: dispatched, idle: idle };
  }

  /** Surface a decline reason if the runnable set can no longer make progress. */
  private declineCheck(): DeclineReason | null {
    const runnable = this.runnable();
    if (runnable.length === 0) return null; // nothing running: not a decline

    // All runnable tasks have zero priority => all stalled.
    let anyPositive = false;
    for (let i = 0; i < runnable.length; i++) {
      if (priority(runnable[i]) > 0) {
        anyPositive = true;
        break;
      }
    }
    if (!anyPositive) {
      const stalled = this.tasks.filter(function (t) {
        return t.state === 'Stalled' || t.state === 'Running';
      });
      return {
        kind: 'all-stalled',
        taskIds: stalled.map(function (t) {
          return t.id;
        }),
      };
    }

    // No system progress for deadlockTicks consecutive ticks.
    if (this.idleTicks >= this.cfg.deadlockTicks) {
      return { kind: 'deadlock', idleTicks: this.idleTicks };
    }
    return null;
  }

  /** True once no task is Running (all Sufficient / Terminated / Stalled / Declined). */
  isQuiescent(): boolean {
    return this.runnable().length === 0;
  }

  /**
   * Drive ticks until quiescence or a hard decline, collecting all events.
   * `maxTicks` is a safety bound, not a timeout: it caps the loop so a buggy
   * OS port cannot spin forever. Real termination is by quiescence/decline.
   */
  async run(maxTicks = 10000): Promise<SchedulerEvent[]> {
    const all: SchedulerEvent[] = [];
    for (let i = 0; i < maxTicks; i++) {
      const out = await this.tick();
      for (let j = 0; j < out.events.length; j++) all.push(out.events[j]);
      if (this.isQuiescent()) break;
      // A hard decline (all-stalled / deadlock) ends the run.
      let declined = false;
      for (let j = 0; j < out.events.length; j++) {
        if (out.events[j].kind === 'decline') {
          declined = true;
          break;
        }
      }
      if (declined) break;
    }
    return all;
  }
}
