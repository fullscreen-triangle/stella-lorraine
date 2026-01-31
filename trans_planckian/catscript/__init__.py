"""
CatScript - Categorical State Counting Language
================================================

A domain-specific language for trans-Planckian temporal resolution
and categorical thermodynamics calculations.

Example:
    >>> from catscript import CatScript
    >>> cs = CatScript()
    >>> cs.execute("resolve time at 5.13e13 Hz")
    >>> cs.execute("entropy of 3 oscillators with 4 states")
"""

from .interpreter import CatScript, CatScriptREPL
from .lexer import Lexer, Token, TokenType
from .parser import Parser, ASTNode
from .runtime import CatRuntime

__version__ = "1.0.0"
__all__ = ["CatScript", "CatScriptREPL", "Lexer", "Parser", "CatRuntime"]
