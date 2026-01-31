"""
CatScript Lexer
===============

Tokenizes CatScript source code into a stream of tokens.
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Iterator


class TokenType(Enum):
    # Keywords - Commands
    RESOLVE = auto()
    ENTROPY = auto()
    TEMPERATURE = auto()
    SPECTRUM = auto()
    ENHANCE = auto()
    VALIDATE = auto()
    MEASURE = auto()
    SIMULATE = auto()
    COMPUTE = auto()
    PRINT = auto()
    SET = auto()
    SHOW = auto()

    # Keywords - Nouns
    TIME = auto()
    FREQUENCY = auto()
    OSCILLATORS = auto()
    STATES = auto()
    PARTITIONS = auto()
    ORDERS = auto()
    RESOLUTION = auto()
    ENHANCEMENT = auto()

    # Keywords - Prepositions/Connectors
    AT = auto()
    OF = auto()
    WITH = auto()
    FROM = auto()
    TO = auto()
    STEPS = auto()
    USING = auto()
    FOR = auto()
    IN = auto()
    BY = auto()

    # Enhancement mechanisms
    TERNARY = auto()
    MULTIMODAL = auto()
    HARMONIC = auto()
    POINCARE = auto()
    REFINEMENT = auto()
    ALL = auto()

    # Spectroscopy types
    RAMAN = auto()
    FTIR = auto()
    IR = auto()

    # Thermodynamics
    HEAT = auto()
    DEATH = auto()
    DECAY = auto()
    ENTROPY_KW = auto()

    # Compounds
    VANILLIN = auto()
    CO = auto()

    # Units
    HZ = auto()
    KHZ = auto()
    MHZ = auto()
    GHZ = auto()
    THZ = auto()
    KELVIN = auto()
    SECONDS = auto()
    JOULES = auto()

    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    EQUALS = auto()

    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    ARROW = auto()

    # Special
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.column})"


class Lexer:
    """Tokenizes CatScript source code."""

    KEYWORDS = {
        # Commands
        'resolve': TokenType.RESOLVE,
        'entropy': TokenType.ENTROPY,
        'temperature': TokenType.TEMPERATURE,
        'spectrum': TokenType.SPECTRUM,
        'enhance': TokenType.ENHANCE,
        'validate': TokenType.VALIDATE,
        'measure': TokenType.MEASURE,
        'simulate': TokenType.SIMULATE,
        'compute': TokenType.COMPUTE,
        'print': TokenType.PRINT,
        'set': TokenType.SET,
        'show': TokenType.SHOW,

        # Nouns
        'time': TokenType.TIME,
        'frequency': TokenType.FREQUENCY,
        'oscillators': TokenType.OSCILLATORS,
        'oscillator': TokenType.OSCILLATORS,
        'states': TokenType.STATES,
        'state': TokenType.STATES,
        'partitions': TokenType.PARTITIONS,
        'partition': TokenType.PARTITIONS,
        'orders': TokenType.ORDERS,
        'resolution': TokenType.RESOLUTION,
        'enhancement': TokenType.ENHANCEMENT,

        # Prepositions
        'at': TokenType.AT,
        'of': TokenType.OF,
        'with': TokenType.WITH,
        'from': TokenType.FROM,
        'to': TokenType.TO,
        'steps': TokenType.STEPS,
        'step': TokenType.STEPS,
        'using': TokenType.USING,
        'for': TokenType.FOR,
        'in': TokenType.IN,
        'by': TokenType.BY,

        # Enhancement mechanisms
        'ternary': TokenType.TERNARY,
        'multimodal': TokenType.MULTIMODAL,
        'harmonic': TokenType.HARMONIC,
        'poincare': TokenType.POINCARE,
        'poincaré': TokenType.POINCARE,
        'refinement': TokenType.REFINEMENT,
        'all': TokenType.ALL,

        # Spectroscopy
        'raman': TokenType.RAMAN,
        'ftir': TokenType.FTIR,
        'ir': TokenType.IR,

        # Thermodynamics
        'heat': TokenType.HEAT,
        'death': TokenType.DEATH,
        'decay': TokenType.DECAY,

        # Compounds
        'vanillin': TokenType.VANILLIN,
        'co': TokenType.CO,

        # Units
        'hz': TokenType.HZ,
        'khz': TokenType.KHZ,
        'mhz': TokenType.MHZ,
        'ghz': TokenType.GHZ,
        'thz': TokenType.THZ,
        'k': TokenType.KELVIN,
        'kelvin': TokenType.KELVIN,
        's': TokenType.SECONDS,
        'sec': TokenType.SECONDS,
        'seconds': TokenType.SECONDS,
        'j': TokenType.JOULES,
        'joules': TokenType.JOULES,
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def error(self, message: str):
        raise SyntaxError(f"Lexer error at L{self.line}:{self.column}: {message}")

    def peek(self, offset: int = 0) -> Optional[str]:
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return None

    def advance(self) -> Optional[str]:
        if self.pos < len(self.source):
            char = self.source[self.pos]
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            return char
        return None

    def skip_whitespace(self):
        while self.peek() and self.peek() in ' \t\r':
            self.advance()

    def skip_comment(self):
        # Skip # comments until newline
        while self.peek() and self.peek() != '\n':
            self.advance()

    def read_number(self) -> Token:
        start_col = self.column
        value = ''

        # Handle negative numbers
        if self.peek() == '-':
            value += self.advance()

        # Integer part
        while self.peek() and self.peek().isdigit():
            value += self.advance()

        # Decimal part
        if self.peek() == '.':
            value += self.advance()
            while self.peek() and self.peek().isdigit():
                value += self.advance()

        # Scientific notation
        if self.peek() and self.peek().lower() == 'e':
            value += self.advance()
            if self.peek() in '+-':
                value += self.advance()
            while self.peek() and self.peek().isdigit():
                value += self.advance()

        return Token(TokenType.NUMBER, value, self.line, start_col)

    def read_identifier(self) -> Token:
        start_col = self.column
        value = ''

        while self.peek() and (self.peek().isalnum() or self.peek() in '_-'):
            value += self.advance()

        # Check if it's a keyword
        lower_value = value.lower()
        if lower_value in self.KEYWORDS:
            return Token(self.KEYWORDS[lower_value], value, self.line, start_col)

        return Token(TokenType.IDENTIFIER, value, self.line, start_col)

    def read_string(self) -> Token:
        start_col = self.column
        quote_char = self.advance()  # Skip opening quote
        value = ''

        while self.peek() and self.peek() != quote_char:
            if self.peek() == '\\':
                self.advance()
                if self.peek():
                    escaped = self.advance()
                    if escaped == 'n':
                        value += '\n'
                    elif escaped == 't':
                        value += '\t'
                    else:
                        value += escaped
            else:
                value += self.advance()

        if self.peek() != quote_char:
            self.error("Unterminated string")

        self.advance()  # Skip closing quote
        return Token(TokenType.STRING, value, self.line, start_col)

    def tokenize(self) -> List[Token]:
        self.tokens = []

        while self.pos < len(self.source):
            char = self.peek()

            # Skip whitespace
            if char in ' \t\r':
                self.skip_whitespace()
                continue

            # Newline
            if char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, '\n', self.line, self.column))
                self.advance()
                continue

            # Comment
            if char == '#':
                self.skip_comment()
                continue

            # Numbers (including negative)
            if char.isdigit() or (char == '-' and self.peek(1) and self.peek(1).isdigit()):
                self.tokens.append(self.read_number())
                continue

            # Identifiers and keywords
            if char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
                continue

            # Strings
            if char in '"\'':
                self.tokens.append(self.read_string())
                continue

            # Operators and punctuation
            start_col = self.column

            if char == '+':
                self.advance()
                self.tokens.append(Token(TokenType.PLUS, '+', self.line, start_col))
            elif char == '-':
                if self.peek(1) == '>':
                    self.advance()
                    self.advance()
                    self.tokens.append(Token(TokenType.ARROW, '->', self.line, start_col))
                else:
                    self.advance()
                    self.tokens.append(Token(TokenType.MINUS, '-', self.line, start_col))
            elif char == '*':
                if self.peek(1) == '*':
                    self.advance()
                    self.advance()
                    self.tokens.append(Token(TokenType.POWER, '**', self.line, start_col))
                else:
                    self.advance()
                    self.tokens.append(Token(TokenType.MULTIPLY, '*', self.line, start_col))
            elif char == '/':
                self.advance()
                self.tokens.append(Token(TokenType.DIVIDE, '/', self.line, start_col))
            elif char == '^':
                self.advance()
                self.tokens.append(Token(TokenType.POWER, '^', self.line, start_col))
            elif char == '=':
                self.advance()
                self.tokens.append(Token(TokenType.EQUALS, '=', self.line, start_col))
            elif char == '(':
                self.advance()
                self.tokens.append(Token(TokenType.LPAREN, '(', self.line, start_col))
            elif char == ')':
                self.advance()
                self.tokens.append(Token(TokenType.RPAREN, ')', self.line, start_col))
            elif char == '[':
                self.advance()
                self.tokens.append(Token(TokenType.LBRACKET, '[', self.line, start_col))
            elif char == ']':
                self.advance()
                self.tokens.append(Token(TokenType.RBRACKET, ']', self.line, start_col))
            elif char == ',':
                self.advance()
                self.tokens.append(Token(TokenType.COMMA, ',', self.line, start_col))
            elif char == ':':
                self.advance()
                self.tokens.append(Token(TokenType.COLON, ':', self.line, start_col))
            elif char == ';':
                self.advance()
                self.tokens.append(Token(TokenType.SEMICOLON, ';', self.line, start_col))
            else:
                self.error(f"Unexpected character: {char!r}")

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens

    def __iter__(self) -> Iterator[Token]:
        if not self.tokens:
            self.tokenize()
        return iter(self.tokens)


def tokenize(source: str) -> List[Token]:
    """Convenience function to tokenize source code."""
    return Lexer(source).tokenize()
