"""
CatScript Interpreter
=====================

Main interface for executing CatScript code.
"""

import sys
import os
from typing import Optional, Callable, List

from .lexer import Lexer, Token
from .parser import Parser, ASTNode, parse
from .runtime import CatRuntime, CatResult


class CatScript:
    """
    CatScript interpreter for categorical state counting calculations.

    Example:
        >>> cs = CatScript()
        >>> cs.execute("resolve time at 5.13e13 Hz")
        >>> cs.execute("entropy of 3 oscillators with 4 states")
        >>> cs.run_file("script.cat")
    """

    VERSION = "1.0.0"

    def __init__(self, output_callback: Optional[Callable[[str], None]] = None):
        self.runtime = CatRuntime(output_callback)
        self.history: List[str] = []
        self.errors: List[str] = []

    def execute(self, source: str) -> CatResult:
        """
        Execute CatScript source code.

        Args:
            source: CatScript source code string

        Returns:
            CatResult with execution outcome
        """
        self.history.append(source)

        try:
            # Tokenize
            lexer = Lexer(source)
            tokens = lexer.tokenize()

            # Parse
            parser = Parser(tokens)
            ast = parser.parse()

            # Execute
            result = self.runtime.execute(ast)
            return result

        except SyntaxError as e:
            error_msg = str(e)
            self.errors.append(error_msg)
            self.runtime.output(f"Syntax Error: {error_msg}")
            return CatResult(False, None, error_msg)

        except Exception as e:
            error_msg = str(e)
            self.errors.append(error_msg)
            self.runtime.output(f"Error: {error_msg}")
            return CatResult(False, None, error_msg)

    def run_file(self, filepath: str) -> CatResult:
        """
        Execute a CatScript file.

        Args:
            filepath: Path to .cat or .csc file

        Returns:
            CatResult with execution outcome
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            return self.execute(source)
        except FileNotFoundError:
            error_msg = f"File not found: {filepath}"
            self.errors.append(error_msg)
            self.runtime.output(f"Error: {error_msg}")
            return CatResult(False, None, error_msg)
        except IOError as e:
            error_msg = f"IO Error: {e}"
            self.errors.append(error_msg)
            self.runtime.output(f"Error: {error_msg}")
            return CatResult(False, None, error_msg)

    def get_variable(self, name: str):
        """Get a variable value from the runtime."""
        return self.runtime.variables.get(name)

    def set_variable(self, name: str, value):
        """Set a variable in the runtime."""
        self.runtime.variables[name] = value

    @property
    def last_result(self) -> Optional[CatResult]:
        """Get the last execution result."""
        return self.runtime.last_result


class CatScriptREPL:
    """
    Interactive Read-Eval-Print-Loop for CatScript.

    Example:
        >>> repl = CatScriptREPL()
        >>> repl.run()
    """

    BANNER = f"""
╔═══════════════════════════════════════════════════════════════════╗
║  CatScript - Categorical State Counting Language v{CatScript.VERSION}          ║
║  Trans-Planckian Temporal Resolution Framework                    ║
║                                                                   ║
║  Type 'help' for commands, 'quit' to exit                         ║
╚═══════════════════════════════════════════════════════════════════╝
"""

    HELP_TEXT = """
CatScript Commands:
═══════════════════

TIME RESOLUTION:
  resolve time at <frequency> [Hz|kHz|MHz|GHz|THz]
  Example: resolve time at 5.13e13 Hz

ENTROPY:
  entropy of <M> oscillators with <n> states
  Example: entropy of 5 oscillators with 4 states

TEMPERATURE:
  temperature from <T1> K to <T2> K [steps <N>]
  Example: temperature from 300K to 1e-15K steps 100

  simulate heat death
  Example: simulate heat death

SPECTROSCOPY:
  spectrum raman|ftir of <compound>
  Example: spectrum raman of vanillin
  Example: measure ftir spectrum of vanillin

ENHANCEMENT:
  enhance with <mechanism1> [mechanism2] ...
  Mechanisms: ternary, multimodal, harmonic, poincare, refinement
  Example: enhance with all
  Example: enhance with ternary multimodal poincare

VALIDATION:
  validate [module]
  Example: validate all

OTHER:
  show enhancement|resolution|variables
  set <variable> = <value>
  compute <expression>
  print <message>

REPL COMMANDS:
  help     - Show this help
  clear    - Clear screen
  history  - Show command history
  quit     - Exit REPL
"""

    def __init__(self):
        self.interpreter = CatScript(output_callback=print)
        self.running = False

    def run(self):
        """Start the REPL."""
        print(self.BANNER)
        self.running = True

        while self.running:
            try:
                # Get input
                line = input("cat> ").strip()

                if not line:
                    continue

                # Handle REPL commands
                if line.lower() == 'quit' or line.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif line.lower() == 'help':
                    print(self.HELP_TEXT)
                    continue
                elif line.lower() == 'clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    continue
                elif line.lower() == 'history':
                    print("\nCommand History:")
                    for i, cmd in enumerate(self.interpreter.history, 1):
                        print(f"  {i}: {cmd}")
                    print()
                    continue

                # Execute CatScript
                self.interpreter.execute(line)

            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except EOFError:
                print("\nGoodbye!")
                break

    def stop(self):
        """Stop the REPL."""
        self.running = False


def main():
    """Main entry point for CatScript."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CatScript - Categorical State Counting Language"
    )
    parser.add_argument(
        'script',
        nargs='?',
        help='CatScript file to execute'
    )
    parser.add_argument(
        '--repl', '-i',
        action='store_true',
        help='Start interactive REPL'
    )
    parser.add_argument(
        '--execute', '-e',
        help='Execute a single command'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run full validation suite'
    )
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'CatScript {CatScript.VERSION}'
    )

    args = parser.parse_args()

    cs = CatScript()

    if args.validate:
        cs.execute("validate all")
    elif args.execute:
        cs.execute(args.execute)
    elif args.script:
        cs.run_file(args.script)
    elif args.repl or (not args.script and not args.execute):
        repl = CatScriptREPL()
        repl.run()


if __name__ == '__main__':
    main()
