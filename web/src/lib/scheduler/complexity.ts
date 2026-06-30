// ── Categorical complexity ────────────────────────────────────────────────────
//
// The static complexity baseline a task is scheduled against. A task of
// categorical complexity K over d operation types must realise K distinguishable
// work-trajectories to settle; the number of distinguishable trajectories of
// length n is T(n,d) = d·(d+1)^(n-1), so the least n with T(n,d) >= K is
//
//     n_term(K,d) = 1 + ceil( log_{d+1}( K / d ) ),
//
// which is O(log K). This is a PREDICTION, used only as the baseline the live
// residue is compared against. It is never treated as ground-truth remaining
// work: a task that overruns n_term is reclassified, not failed (see scheduler).

import type { CategoricalComplexity } from './types';

/** Distinguishable work-trajectories of length n over d operation types. */
export function trajectoryCount(n: number, d: number): number {
  if (n < 1) return 0;
  return d * Math.pow(d + 1, n - 1);
}

/** n_term(K,d): least committed-unit count with T(n,d) >= K. O(log K). */
export function nTerm(k: number, d: number): number {
  if (k <= d) return 1;
  return 1 + Math.ceil(Math.log(k / d) / Math.log(d + 1));
}

/** n_term for a task's CategoricalComplexity. */
export function nTermOf(c: CategoricalComplexity): number {
  return nTerm(c.k, c.d);
}

/**
 * Re-estimate categorical complexity from observed descent. If a task has
 * committed `committedActs` units and is still above its sufficiency
 * threshold while converging, its true K is at least what `committedActs`
 * trajectories realise; we round the implied K up to the next power so the
 * baseline tracks reality rather than the original (under-)estimate.
 */
export function reclassify(
  committedActs: number,
  d: number,
): number {
  // K consistent with having committed `committedActs+1` units:
  // smallest K' with n_term(K',d) > committedActs is T(committedActs+1, d).
  return Math.ceil(trajectoryCount(committedActs + 1, d));
}
