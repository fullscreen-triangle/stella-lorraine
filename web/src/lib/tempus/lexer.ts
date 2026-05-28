import { Token, TType } from './types';

const KEYWORDS: Record<string, TType> = {
  cell:     'CELL',
  bounds:   'BOUNDS',
  action:   'ACTION',
  sync:     'SYNC',
  at:       'AT',
  freq:     'FREQ',
  compose:  'COMPOSE',
  channels: 'CHANNELS',
  into:     'INTO',
  when:     'WHEN',
  do:       'DO',
  emit:     'EMIT',
  fire:     'FIRE',
  begin:    'BEGIN',
  end:      'END',
  wait:     'WAIT',
};

export function lex(src: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  while (i < src.length) {
    // whitespace
    if (/\s/.test(src[i])) { i++; continue; }

    // line comment  --
    if (src[i] === '-' && src[i + 1] === '-') {
      while (i < src.length && src[i] !== '\n') i++;
      continue;
    }

    // number  (with optional leading minus in expression position)
    const prevType = tokens.length ? tokens[tokens.length - 1].type : null;
    const numContext = prevType === null || prevType === 'LPAREN' || prevType === 'COMMA' || prevType === 'EQ';
    if (
      /[0-9]/.test(src[i]) ||
      (src[i] === '-' && numContext && /[0-9.]/.test(src[i + 1] ?? ''))
    ) {
      let j = i;
      if (src[j] === '-') j++;
      while (j < src.length && /[0-9]/.test(src[j])) j++;
      if (src[j] === '.') { j++; while (j < src.length && /[0-9]/.test(src[j])) j++; }
      if (/[eE]/.test(src[j] ?? '')) {
        j++;
        if (/[+-]/.test(src[j] ?? '')) j++;
        while (j < src.length && /[0-9]/.test(src[j])) j++;
      }
      tokens.push({ type: 'NUMBER', value: src.slice(i, j), pos: i });
      i = j;
      continue;
    }

    // identifier / keyword
    if (/[a-zA-Z_]/.test(src[i])) {
      let j = i;
      while (j < src.length && /[a-zA-Z0-9_]/.test(src[j])) j++;
      const word = src.slice(i, j);
      tokens.push({ type: KEYWORDS[word.toLowerCase()] ?? 'IDENT', value: word, pos: i });
      i = j;
      continue;
    }

    // single-char tokens
    const singles: Record<string, TType> = { '=':'EQ', '(':'LPAREN', ')':'RPAREN', ',':'COMMA', ';':'SEMI' };
    if (singles[src[i]]) {
      tokens.push({ type: singles[src[i]], value: src[i], pos: i });
    }
    i++;
  }

  tokens.push({ type: 'EOF', value: '', pos: i });
  return tokens;
}
