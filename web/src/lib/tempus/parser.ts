import { Token, TType, Decl, Stmt, CellDecl, SyncDecl, ComposeDecl, WhenDecl, Program } from './types';

export class ParseError extends Error {
  constructor(msg: string, public pos: number) { super(msg); }
}

export function parse(tokens: Token[]): Program {
  let cur = 0;

  const peek  = (): Token => tokens[cur];
  const adv   = (): Token => tokens[cur++];
  const check = (t: TType) => peek().type === t;

  function expect(t: TType): Token {
    const tok = adv();
    if (tok.type !== t) throw new ParseError(`Expected ${t}, got '${tok.value}'`, tok.pos);
    return tok;
  }
  function expectIdent(): string  { return expect('IDENT').value; }
  function expectNum():   number  { return parseFloat(expect('NUMBER').value); }

  // ── Statement ──────────────────────────────────────────────────────────────
  function parseStmt(): Stmt {
    const t = peek();

    if (t.type === 'EMIT') {
      adv();
      return { kind: 'emit', name: expectIdent() };
    }
    if (t.type === 'FIRE') {
      adv();
      const name = expectIdent();
      const args: number[] = [];
      if (check('LPAREN')) {
        adv();
        while (!check('RPAREN') && !check('EOF')) {
          args.push(expectNum());
          if (check('COMMA')) adv();
        }
        expect('RPAREN');
      }
      return { kind: 'fire', name, args };
    }
    if (t.type === 'WAIT') {
      adv();
      return { kind: 'wait', duration: expectNum() };
    }
    if (t.type === 'BEGIN') {
      adv();
      const stmts: Stmt[] = [];
      while (!check('END') && !check('EOF')) {
        stmts.push(parseStmt());
        if (check('SEMI')) adv();
      }
      expect('END');
      return { kind: 'block', stmts };
    }
    throw new ParseError(`Unexpected '${t.value}' in statement`, t.pos);
  }

  // ── Declaration ────────────────────────────────────────────────────────────
  function parseDecl(): Decl | null {
    const t = peek();

    if (t.type === 'CELL') {
      adv();
      const name = expectIdent();
      expect('BOUNDS');
      expect('LPAREN');
      const lo = expectNum();
      expect('COMMA');
      const hi = expectNum();
      expect('RPAREN');
      expect('ACTION');
      const action = expectNum();
      return { kind: 'cell', name, lo, hi, action };
    }

    if (t.type === 'SYNC') {
      adv();
      const name = expectIdent();
      expect('AT');
      const freq = expectNum();
      expect('FREQ');
      return { kind: 'sync', name, freq };
    }

    if (t.type === 'COMPOSE') {
      adv();
      const dTok = adv();  // 'd'
      if (dTok.value !== 'd') throw new ParseError(`Expected 'd' after compose`, dTok.pos);
      expect('EQ');
      const d = expectNum();
      expect('CHANNELS');
      const channels: string[] = [expectIdent()];
      while (check('COMMA')) { adv(); channels.push(expectIdent()); }
      expect('INTO');
      const into = expectIdent();
      return { kind: 'compose', d, channels, into };
    }

    if (t.type === 'WHEN') {
      adv();
      const cell = expectIdent();
      expect('DO');
      const stmt = parseStmt();
      return { kind: 'when', cell, stmt };
    }

    // skip unrecognised tokens (ensemble, sync_threshold, etc.)
    adv();
    return null;
  }

  const decls: Decl[] = [];
  while (!check('EOF')) {
    try {
      const d = parseDecl();
      if (d) decls.push(d);
    } catch (e) {
      // skip to next line on parse error in a declaration
      while (!check('EOF') && peek().pos === cur) adv();
    }
  }
  return { decls };
}
