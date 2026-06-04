// ── Token types ───────────────────────────────────────────────────────────────
export type TType =
  | 'CELL' | 'BOUNDS' | 'ACTION' | 'SYNC' | 'AT' | 'FREQ'
  | 'COMPOSE' | 'CHANNELS' | 'INTO' | 'WHEN' | 'DO'
  | 'EMIT' | 'FIRE' | 'BEGIN' | 'END' | 'WAIT'
  | 'IDENT' | 'NUMBER' | 'EQ' | 'LPAREN' | 'RPAREN'
  | 'COMMA' | 'SEMI' | 'EOF';

export interface Token { type: TType; value: string; pos: number; }

// ── Diagnostics (compile-time) ────────────────────────────────────────────────
export type Severity = 'error' | 'warning' | 'info';
export interface Diag {
  severity: Severity;
  message:  string;
  pos?:     number;   // byte offset into source, for line/col resolution
}

// ── AST ───────────────────────────────────────────────────────────────────────
export type Stmt =
  | { kind: 'emit'; name: string }
  | { kind: 'fire'; name: string; args: number[] }
  | { kind: 'wait'; duration: number }
  | { kind: 'block'; stmts: Stmt[] };

export interface CellDecl  { kind: 'cell';    name: string; lo: number; hi: number; action: number; }
export interface SyncDecl  { kind: 'sync';    name: string; freq: number; }
export interface ComposeDecl { kind: 'compose'; d: number; channels: string[]; into: string; }
export interface WhenDecl  { kind: 'when';    cell: string; stmt: Stmt; }

export type Decl = CellDecl | SyncDecl | ComposeDecl | WhenDecl;
export interface Program { decls: Decl[]; }

// ── Parsed runtime representation ────────────────────────────────────────────
export interface ParsedProgram {
  cells:   Map<string, CellDecl>;
  syncs:   Map<string, SyncDecl>;
  compose: ComposeDecl | null;
  whens:   Map<string, string[]>;   // cell name → action labels
}

// ── Simulation event record ───────────────────────────────────────────────────
export interface SimEvent {
  index:        number;
  channel:      string;
  dp:           number;
  cell:         string;   // name of matched cell, or 'anomaly'
  M:            number;   // oscillator cycle count
  phase:        'COMPILE' | 'EXECUTE';
  trajectoryId: number;
  actionFired:  string | null;  // non-null when trajectory completed
  time:         number;         // M / freq (seconds)
}
