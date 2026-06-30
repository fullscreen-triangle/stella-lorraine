# A Clock and Its Specifications

**Kundai Farai Sachikonye**
Department of Bioinformatics, Technical University of Munich
`kundai.sachikonye@wzw.tum.de`

**Version:** 5.0 · **Last updated:** July 2026

---

## Summary

This repository develops a single object and several specifications of it.
The object is a **clock**: time understood operationally as a strictly
monotone count of completed work, rather than as a parameter read from an
external reference. The specifications apply that clock to distinct
problems — distributed control, federated coordination of finite agents,
a temporal programming language, and process scheduling — each as a
self-contained development with its own definitions and proofs.

The clock has three properties, established once and reused throughout:

1. **Time is a count.** For a reference oscillator of frequency $f$, the
   number of completed cycles $M$ and the elapsed time $t$ satisfy
   $t = M/f$ exactly, read as the definition of the derived continuous
   parameter $t$ from the primitive count $M$. There is no appeal to a
   universal external clock; each oscillator carries its own count, and
   counts of different oscillators are made comparable by their
   frequencies.

2. **The count is monotone.** No physical operation decrements $M$. The
   execution of any operation is itself a counted unit and therefore
   increments the count; an attempted reversal is a new forward unit, not
   a return. Consequently a process never re-occupies a prior state, and
   resemblance of configuration is never identity of state.

3. **Individuation carries a positive floor.** Telling a part from the
   rest of a non-completable whole costs a strictly positive minimum
   $\beta > 0$. The floor is the irreducible boundary content of any
   distinction; it is what makes a recognised result distinguishable from
   the candidates it cannot otherwise be told apart from, and it is the
   reason a finite agent halts at a positive residue rather than at zero.

These three properties — the time–count identity, monotonicity, and the
floor — are the whole of the foundation. Everything else is a
specification.

---

## 1. The foundation

The foundational results are developed in `chitungwiza/epistemology/`.
They are stated as theorems about finite weighted graphs, finite
oscillators, and bounded resolving agents, and use only standard
mathematics. No specification depends on any other; each foundational
paper is readable on its own.

| Result | File | Statement |
|---|---|---|
| Time–count identity | `partition-depth-propagation.tex` | $t = M/f$; $\mathrm{d}M/\mathrm{d}t = f$ exactly |
| Monotonicity and non-return | `irreducible-residue-propagation.tex`, `loschmidt-partition-count.tex` | $M$ strictly increases; no operation decrements it |
| Resolution floor | `irreducible-residue-propagation.tex` | every separation deposits content $\ge \beta > 0$ |
| Individuation by negation | `instantiation-of-weighted-finite-graphs.tex` | a part is fixed as the complement of its negation; no selector without regress |
| Forced operational structure | `computational-systems-structure.tex` | a finite agent on a finite graph admits a unique operational structure (category, residue, preorder, floor) |
| Orchestration and necessity | `semantic-categorical-orchestra.tex` | correctness is local; necessity is global and decidable only above the parts |
| Backward completion | `unconstrained-subtask-computing.tex` | a known endpoint is reached in $O(\log N)$ steps via residue navigation |

The non-completability of the whole — that no finite stage exhausts it —
is the single hypothesis beyond the finite structures themselves. It is an
order condition (no terminal stage), not a cardinality assumption, and it
is what forces the floor to be positive.

---

## 2. The clock, stated once

A bounded system executes work in discrete committed units. The count of
committed units is the system's logical time.

**Time–count identity.** For a reference oscillator of frequency $f$,
$$
t = \frac{M}{f}, \qquad \frac{\mathrm{d}M}{\mathrm{d}t} = f,
$$
exact. The continuous parameter $t$ is the $f \to \infty$ idealisation of
the integer count $M$. Different oscillators count at different rates;
their counts are made commensurable by the ratio of frequencies, with no
privileged reference.

**Monotonicity.** Each committed unit appends one entry to an append-only
record, so $M$ increases by exactly one per unit and never decreases. An
operation that purports to reverse a process is itself a committed unit
and increments $M$. Hence a state $(x, M)$ with current configuration $x$
and count $M$ is never repeated, even where $x$ recurs.

**Floor.** A finite agent that resolves a problem to a recognisable answer
cannot drive its residual distance to a solution below a strictly positive
floor $\beta$. Below $\beta$ the solution is no longer distinguishable, in
the agent's representation, from the candidates collapsed onto it; the
positive gap is the margin of recognition, not error to be removed.

---

## 3. The specifications

Each specification applies the clock to one problem. They share the three
properties of §2 and are otherwise independent.

### 3.1 Temporal programming

`chitungwiza/publications/temporal-programming/temporal-programming.tex`

A programming model in which the datum is a timing deviation
$\Delta P(k) = T_{\mathrm{ref}}(k) - t_{\mathrm{rec}}(k)$ — the gap between
when a unit was expected and when it was recorded — and cells partition
$\Delta P$-space. Because the count is monotone, a recorded timing
history cannot be replayed to a prior state; this gives the language a
structural replay-immunity that does not rest on cryptographic
assumptions. A browser-based subset of the language, with a tutorial
sandbox, is implemented under `web/`.

### 3.2 Distributed control

`chitungwiza/publications/control-system/distributed-control-system.tex`

A control specification in which the controlled quantity is the partition
count and the control law acts on the residue (distance above the floor)
rather than on a continuous error signal. Stability and the absence of a
universal time reference are treated together: each controller keeps its
own count, and coordination is expressed through the comparison of counts,
not through synchronisation to a master clock.

### 3.3 Swarm and federation

`chitungwiza/publications/swarm-federation/swarm-federation-techniques.tex`

Coordination of a federation of finite agents. Each agent is a part
individuated against the others; the federation exists as a structure
because the agents mutually maintain the boundaries between them. A
coherent collective admits a single coordinating rule on its quotient,
and the same construction recurs one level up — agent is to collective as
part is to whole. Related developments appear in
`chitungwiza/epistemology/finite-agent-coordination.tex` and
`synchronised-agent-coordination.tex`.

### 3.4 Scheduling

`chitungwiza/publications/incoherence-time-traversal/` and
`chitungwiza/scheduling-mechanism/trajectory-scheduling-mechanism.tex`

A scheduler that orders work by measured residue and its rate of descent,
rather than by predicted running time — a quantity no scheduler can know
in advance without performing the work. A task whose residue is falling is
compute-limited and is given more resources; a task whose residue has gone
flat above the floor is structure-limited and is not. A sub-task is
released as soon as it is sufficient for the parent goal, even above its
own floor — a judgment the sub-task cannot make for itself. Processes are
annotated by an orthogonal pair: a content digest for identity and a
monotone trajectory address for relatedness, since the two requirements
demand opposite locality and cannot be served by one label.

A reference implementation of the scheduler is provided as a TypeScript
module under `web/src/lib/scheduler/`. It is self-contained: a host system
implements a one-method dispatch port and supplies tasks; the scheduler
ranks them by residue, detects stalls and sufficiency, and never invents
data. The accompanying paper carries a validation suite of twelve
experiments, each comparing a theorem's prediction against an independent
measurement.

### 3.5 Irreversibility

`chitungwiza/publications/incoherence-time-traversal/loschmidt-partition-count.tex`

A self-contained treatment of Loschmidt's paradox using the monotone
count. Time-reversal invariance is a property of the solution set of the
equations of motion, not an operation on physical systems; reversing a
process is a new forward operation that increments the count. The
macroscopic arrow of time is recovered as the statement that counting is
what time is, with the nuclear spin echo as an empirical witness: the echo
recovers an observable by a forward operation on a preserved microstate
while the count advances.

---

## 4. Repository layout

```
stella-lorraine/
├── chitungwiza/
│   ├── epistemology/        # foundational papers (the clock and its proofs)
│   ├── publications/        # the specifications
│   │   ├── control-system/
│   │   ├── swarm-federation/
│   │   ├── temporal-programming/
│   │   └── incoherence-time-traversal/
│   └── scheduling-mechanism/   # scheduler paper, validation, figures
├── web/                     # Next.js app
│   └── src/lib/
│       ├── tempus/          # browser subset of the temporal language
│       └── scheduler/       # reference scheduler module (TypeScript)
├── trans_planckian/         # earlier development (superseded framing)
├── crates/                  # Rust implementations
└── docs/                    # supporting documents
```

The `web/src/lib/scheduler/` module is the most directly usable artefact.
It depends on nothing in the rest of the repository and is intended to be
copied into a host system that implements its dispatch port.

---

## 5. Scheduler reference module

A host system uses the scheduler by implementing one interface and
supplying tasks:

```ts
import { Scheduler, OsPort, LiveTask } from './scheduler';

class HostPort implements OsPort {
  async dispatch(d) {
    // execute one unit of work; return its current residue
  }
}

const scheduler = new Scheduler(tasks, new HostPort());
await scheduler.run();
```

The scheduler reads the residue returned by each dispatch, ranks tasks by
residue and descent rate, releases tasks at their sufficiency thresholds,
and reports stalls and declines. It computes no sufficiency judgment
itself — that is supplied to it as a per-task threshold — and it generates
no data; every decision follows from a real dispatch result.

---

## 6. Status and scope

The four specifications are at the level of theory and reference
implementation. The temporal language and the scheduler have running code;
the control and federation specifications are developed on paper with
worked examples. Each result is proved from stated definitions; material
that could not be proved at that standard has been omitted.

The framing is deliberately modest. The contribution is not a claim about
the limits of physical measurement but a single operational account of
time — a monotone count with a positive floor — together with several
specifications that show the account is enough to organise distinct
problems in control, coordination, programming, and scheduling.

---

## References

The foundational results cite only the standard literature
(Poincaré 1890; Liouville 1838; Boltzmann 1877; Lamport 1978;
Mac Lane 1998; Shannon 1948; and the cryptographic and scheduling
references in the respective papers). Each paper carries its own
bibliography; no result depends on an unpublished companion.

---

## Citation

```bibtex
@misc{sachikonye2026clock,
  title  = {A Clock and Its Specifications: A Monotone Partition Count
            and Its Application to Control, Federation, Temporal
            Programming, and Scheduling},
  author = {Sachikonye, Kundai Farai},
  year   = {2026},
  note   = {Technical University of Munich}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Kundai Farai Sachikonye**
Department of Bioinformatics, Technical University of Munich
`kundai.sachikonye@wzw.tum.de`
