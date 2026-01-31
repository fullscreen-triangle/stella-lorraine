"""
CatScript Parser
================

Parses tokens into an Abstract Syntax Tree (AST).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Union
from enum import Enum, auto

from .lexer import Token, TokenType, Lexer


class NodeType(Enum):
    PROGRAM = auto()
    RESOLVE_TIME = auto()
    ENTROPY_CALC = auto()
    TEMPERATURE_SIM = auto()
    SPECTRUM_MEASURE = auto()
    ENHANCE_APPLY = auto()
    VALIDATE_RUN = auto()
    PRINT_STMT = auto()
    SET_VAR = auto()
    SHOW_INFO = auto()
    COMPUTE_EXPR = auto()
    NUMBER_LITERAL = auto()
    STRING_LITERAL = auto()
    IDENTIFIER = auto()
    UNIT_VALUE = auto()
    BINARY_OP = auto()


@dataclass
class ASTNode:
    """Base AST node."""
    type: NodeType
    value: Any = None
    children: List['ASTNode'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    line: int = 0
    column: int = 0

    def __repr__(self):
        return f"ASTNode({self.type.name}, {self.value}, attrs={self.attributes})"


class Parser:
    """Parses CatScript tokens into an AST."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def error(self, message: str, token: Optional[Token] = None):
        if token is None:
            token = self.current()
        raise SyntaxError(f"Parse error at L{token.line}:{token.column}: {message}")

    def current(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF

    def peek(self, offset: int = 0) -> Token:
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]

    def advance(self) -> Token:
        token = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def expect(self, *types: TokenType) -> Token:
        token = self.current()
        if token.type not in types:
            expected = ' or '.join(t.name for t in types)
            self.error(f"Expected {expected}, got {token.type.name}")
        return self.advance()

    def match(self, *types: TokenType) -> bool:
        return self.current().type in types

    def skip_newlines(self):
        while self.match(TokenType.NEWLINE):
            self.advance()

    def parse(self) -> ASTNode:
        """Parse the entire program."""
        statements = []

        while not self.match(TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.EOF):
                break

            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)

            # Expect newline or EOF after statement
            if not self.match(TokenType.EOF, TokenType.NEWLINE):
                self.error("Expected newline after statement")
            self.skip_newlines()

        return ASTNode(
            type=NodeType.PROGRAM,
            children=statements
        )

    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement."""
        token = self.current()

        if token.type == TokenType.RESOLVE:
            return self.parse_resolve()
        elif token.type == TokenType.ENTROPY:
            return self.parse_entropy()
        elif token.type == TokenType.TEMPERATURE:
            return self.parse_temperature()
        elif token.type == TokenType.SPECTRUM:
            return self.parse_spectrum()
        elif token.type == TokenType.MEASURE:
            return self.parse_measure()
        elif token.type == TokenType.ENHANCE:
            return self.parse_enhance()
        elif token.type == TokenType.VALIDATE:
            return self.parse_validate()
        elif token.type == TokenType.SIMULATE:
            return self.parse_simulate()
        elif token.type == TokenType.COMPUTE:
            return self.parse_compute()
        elif token.type == TokenType.PRINT:
            return self.parse_print()
        elif token.type == TokenType.SET:
            return self.parse_set()
        elif token.type == TokenType.SHOW:
            return self.parse_show()
        elif token.type == TokenType.IDENTIFIER:
            # Could be assignment or expression
            return self.parse_assignment_or_expr()
        else:
            self.error(f"Unexpected token: {token.type.name}")

    def parse_resolve(self) -> ASTNode:
        """Parse: resolve time at <frequency> [Hz|kHz|...]"""
        token = self.advance()  # consume 'resolve'
        node = ASTNode(
            type=NodeType.RESOLVE_TIME,
            line=token.line,
            column=token.column
        )

        # Optional 'time'
        if self.match(TokenType.TIME):
            self.advance()

        # Expect 'at'
        self.expect(TokenType.AT)

        # Parse frequency value
        freq_value = self.parse_number()
        node.attributes['frequency'] = freq_value

        # Parse optional unit
        if self.match(TokenType.HZ, TokenType.KHZ, TokenType.MHZ,
                      TokenType.GHZ, TokenType.THZ):
            unit = self.advance()
            node.attributes['unit'] = unit.value.lower()
        else:
            node.attributes['unit'] = 'hz'

        return node

    def parse_entropy(self) -> ASTNode:
        """Parse: entropy of <M> oscillators with <n> states"""
        token = self.advance()  # consume 'entropy'
        node = ASTNode(
            type=NodeType.ENTROPY_CALC,
            line=token.line,
            column=token.column
        )

        # Expect 'of'
        self.expect(TokenType.OF)

        # Parse M (number of oscillators)
        M = self.parse_number()
        node.attributes['M'] = M

        # Expect 'oscillators' or 'oscillator'
        self.expect(TokenType.OSCILLATORS)

        # Expect 'with'
        self.expect(TokenType.WITH)

        # Parse n (number of states)
        n = self.parse_number()
        node.attributes['n'] = n

        # Expect 'states' or 'state'
        self.expect(TokenType.STATES)

        return node

    def parse_temperature(self) -> ASTNode:
        """Parse: temperature from <T1> to <T2> [steps <N>]"""
        token = self.advance()  # consume 'temperature'
        node = ASTNode(
            type=NodeType.TEMPERATURE_SIM,
            line=token.line,
            column=token.column
        )

        # Expect 'from'
        self.expect(TokenType.FROM)

        # Parse initial temperature
        T1 = self.parse_temperature_value()
        node.attributes['T_initial'] = T1

        # Expect 'to'
        self.expect(TokenType.TO)

        # Parse final temperature
        T2 = self.parse_temperature_value()
        node.attributes['T_final'] = T2

        # Optional 'steps <N>'
        if self.match(TokenType.STEPS):
            self.advance()
            steps = self.parse_number()
            node.attributes['steps'] = int(steps)
        else:
            node.attributes['steps'] = 100

        return node

    def parse_spectrum(self) -> ASTNode:
        """Parse: spectrum <type> of <compound>"""
        token = self.advance()  # consume 'spectrum'
        node = ASTNode(
            type=NodeType.SPECTRUM_MEASURE,
            line=token.line,
            column=token.column
        )

        # Parse spectrum type
        if self.match(TokenType.RAMAN):
            self.advance()
            node.attributes['type'] = 'raman'
        elif self.match(TokenType.FTIR, TokenType.IR):
            self.advance()
            node.attributes['type'] = 'ftir'
        else:
            self.error("Expected 'raman' or 'ftir'")

        # Expect 'of'
        self.expect(TokenType.OF)

        # Parse compound
        if self.match(TokenType.VANILLIN):
            self.advance()
            node.attributes['compound'] = 'vanillin'
        elif self.match(TokenType.CO):
            self.advance()
            node.attributes['compound'] = 'CO'
        elif self.match(TokenType.IDENTIFIER, TokenType.STRING):
            token = self.advance()
            node.attributes['compound'] = token.value
        else:
            self.error("Expected compound name")

        return node

    def parse_measure(self) -> ASTNode:
        """Parse: measure <type> spectrum of <compound>"""
        self.advance()  # consume 'measure'

        # This is similar to spectrum, redirect
        if self.match(TokenType.RAMAN, TokenType.FTIR, TokenType.IR):
            # Push back conceptually and use parse_spectrum logic
            token = self.current()
            node = ASTNode(
                type=NodeType.SPECTRUM_MEASURE,
                line=token.line,
                column=token.column
            )

            if self.match(TokenType.RAMAN):
                self.advance()
                node.attributes['type'] = 'raman'
            else:
                self.advance()
                node.attributes['type'] = 'ftir'

            # Optional 'spectrum'
            if self.match(TokenType.SPECTRUM):
                self.advance()

            # Expect 'of'
            self.expect(TokenType.OF)

            # Parse compound
            if self.match(TokenType.VANILLIN):
                self.advance()
                node.attributes['compound'] = 'vanillin'
            elif self.match(TokenType.IDENTIFIER, TokenType.STRING):
                token = self.advance()
                node.attributes['compound'] = token.value
            else:
                self.error("Expected compound name")

            return node
        else:
            self.error("Expected measurement type (raman, ftir)")

    def parse_enhance(self) -> ASTNode:
        """Parse: enhance with <mechanism1> [mechanism2] ..."""
        token = self.advance()  # consume 'enhance'
        node = ASTNode(
            type=NodeType.ENHANCE_APPLY,
            line=token.line,
            column=token.column
        )

        # Expect 'with' or ':'
        if self.match(TokenType.WITH):
            self.advance()
        elif self.match(TokenType.COLON):
            self.advance()

        mechanisms = []

        # Check for 'all'
        if self.match(TokenType.ALL):
            self.advance()
            mechanisms = ['ternary', 'multimodal', 'harmonic', 'poincare', 'refinement']
        else:
            # Parse individual mechanisms
            mechanism_tokens = [
                TokenType.TERNARY, TokenType.MULTIMODAL, TokenType.HARMONIC,
                TokenType.POINCARE, TokenType.REFINEMENT
            ]

            while self.match(*mechanism_tokens):
                token = self.advance()
                mechanisms.append(token.value.lower())

                # Optional '+' between mechanisms
                if self.match(TokenType.PLUS):
                    self.advance()

        if not mechanisms:
            self.error("Expected at least one enhancement mechanism")

        node.attributes['mechanisms'] = mechanisms
        return node

    def parse_validate(self) -> ASTNode:
        """Parse: validate [module]"""
        token = self.advance()  # consume 'validate'
        node = ASTNode(
            type=NodeType.VALIDATE_RUN,
            line=token.line,
            column=token.column
        )

        # Optional module name
        if self.match(TokenType.ALL):
            self.advance()
            node.attributes['module'] = 'all'
        elif self.match(TokenType.IDENTIFIER):
            token = self.advance()
            node.attributes['module'] = token.value
        else:
            node.attributes['module'] = 'all'

        return node

    def parse_simulate(self) -> ASTNode:
        """Parse: simulate heat death [from <T1> to <T2>]"""
        token = self.advance()  # consume 'simulate'
        node = ASTNode(
            type=NodeType.TEMPERATURE_SIM,
            line=token.line,
            column=token.column
        )

        # Check for 'heat death'
        if self.match(TokenType.HEAT):
            self.advance()
            if self.match(TokenType.DEATH):
                self.advance()
                node.attributes['type'] = 'heat_death'
                node.attributes['T_initial'] = 300.0
                node.attributes['T_final'] = 1e-15
                node.attributes['steps'] = 200
        elif self.match(TokenType.TEMPERATURE):
            # Redirect to temperature simulation
            return self.parse_temperature()
        elif self.match(TokenType.DECAY):
            self.advance()
            node.attributes['type'] = 'decay'

        # Optional temperature range
        if self.match(TokenType.FROM):
            self.advance()
            T1 = self.parse_temperature_value()
            node.attributes['T_initial'] = T1
            self.expect(TokenType.TO)
            T2 = self.parse_temperature_value()
            node.attributes['T_final'] = T2

        return node

    def parse_compute(self) -> ASTNode:
        """Parse: compute <expression>"""
        token = self.advance()  # consume 'compute'
        node = ASTNode(
            type=NodeType.COMPUTE_EXPR,
            line=token.line,
            column=token.column
        )

        # Check for specific computations
        if self.match(TokenType.RESOLUTION):
            self.advance()
            node.attributes['compute_type'] = 'resolution'
        elif self.match(TokenType.ENHANCEMENT):
            self.advance()
            node.attributes['compute_type'] = 'enhancement'
        elif self.match(TokenType.ORDERS):
            self.advance()
            node.attributes['compute_type'] = 'orders_below_planck'
        else:
            # General expression
            expr = self.parse_expression()
            node.children.append(expr)

        return node

    def parse_print(self) -> ASTNode:
        """Parse: print <expression>"""
        token = self.advance()  # consume 'print'
        node = ASTNode(
            type=NodeType.PRINT_STMT,
            line=token.line,
            column=token.column
        )

        # Parse what to print
        if self.match(TokenType.STRING):
            str_token = self.advance()
            node.value = str_token.value
        elif self.match(TokenType.IDENTIFIER):
            id_token = self.advance()
            node.value = id_token.value
            node.attributes['is_variable'] = True
        else:
            expr = self.parse_expression()
            node.children.append(expr)

        return node

    def parse_set(self) -> ASTNode:
        """Parse: set <variable> = <value>"""
        token = self.advance()  # consume 'set'
        node = ASTNode(
            type=NodeType.SET_VAR,
            line=token.line,
            column=token.column
        )

        # Variable name
        var_token = self.expect(TokenType.IDENTIFIER)
        node.attributes['name'] = var_token.value

        # Optional '='
        if self.match(TokenType.EQUALS):
            self.advance()
        elif self.match(TokenType.TO):
            self.advance()

        # Value
        value = self.parse_expression()
        node.children.append(value)

        return node

    def parse_show(self) -> ASTNode:
        """Parse: show <info_type>"""
        token = self.advance()  # consume 'show'
        node = ASTNode(
            type=NodeType.SHOW_INFO,
            line=token.line,
            column=token.column
        )

        # What to show
        if self.match(TokenType.ENHANCEMENT):
            self.advance()
            node.attributes['info'] = 'enhancement'
        elif self.match(TokenType.RESOLUTION):
            self.advance()
            node.attributes['info'] = 'resolution'
        elif self.match(TokenType.IDENTIFIER):
            token = self.advance()
            node.attributes['info'] = token.value
        else:
            node.attributes['info'] = 'status'

        return node

    def parse_assignment_or_expr(self) -> ASTNode:
        """Parse identifier = value or expression."""
        id_token = self.advance()

        if self.match(TokenType.EQUALS):
            self.advance()
            node = ASTNode(
                type=NodeType.SET_VAR,
                line=id_token.line,
                column=id_token.column
            )
            node.attributes['name'] = id_token.value
            value = self.parse_expression()
            node.children.append(value)
            return node
        else:
            # Just an identifier reference (not common in this DSL)
            return ASTNode(
                type=NodeType.IDENTIFIER,
                value=id_token.value,
                line=id_token.line,
                column=id_token.column
            )

    def parse_expression(self) -> ASTNode:
        """Parse a mathematical expression."""
        return self.parse_additive()

    def parse_additive(self) -> ASTNode:
        """Parse addition and subtraction."""
        left = self.parse_multiplicative()

        while self.match(TokenType.PLUS, TokenType.MINUS):
            op = self.advance()
            right = self.parse_multiplicative()
            left = ASTNode(
                type=NodeType.BINARY_OP,
                value=op.value,
                children=[left, right],
                line=op.line,
                column=op.column
            )

        return left

    def parse_multiplicative(self) -> ASTNode:
        """Parse multiplication and division."""
        left = self.parse_power()

        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self.advance()
            right = self.parse_power()
            left = ASTNode(
                type=NodeType.BINARY_OP,
                value=op.value,
                children=[left, right],
                line=op.line,
                column=op.column
            )

        return left

    def parse_power(self) -> ASTNode:
        """Parse exponentiation."""
        left = self.parse_primary()

        while self.match(TokenType.POWER):
            op = self.advance()
            right = self.parse_primary()
            left = ASTNode(
                type=NodeType.BINARY_OP,
                value='^',
                children=[left, right],
                line=op.line,
                column=op.column
            )

        return left

    def parse_primary(self) -> ASTNode:
        """Parse primary expressions."""
        token = self.current()

        if token.type == TokenType.NUMBER:
            self.advance()
            return ASTNode(
                type=NodeType.NUMBER_LITERAL,
                value=float(token.value),
                line=token.line,
                column=token.column
            )
        elif token.type == TokenType.STRING:
            self.advance()
            return ASTNode(
                type=NodeType.STRING_LITERAL,
                value=token.value,
                line=token.line,
                column=token.column
            )
        elif token.type == TokenType.IDENTIFIER:
            self.advance()
            return ASTNode(
                type=NodeType.IDENTIFIER,
                value=token.value,
                line=token.line,
                column=token.column
            )
        elif token.type == TokenType.LPAREN:
            self.advance()  # consume '('
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
        else:
            self.error(f"Unexpected token in expression: {token.type.name}")

    def parse_number(self) -> float:
        """Parse and return a number value."""
        token = self.expect(TokenType.NUMBER)
        return float(token.value)

    def parse_temperature_value(self) -> float:
        """Parse a temperature value with optional 'K' unit."""
        value = self.parse_number()

        # Optional 'K' or 'kelvin'
        if self.match(TokenType.KELVIN):
            self.advance()

        return value


def parse(source: str) -> ASTNode:
    """Convenience function to parse source code."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
