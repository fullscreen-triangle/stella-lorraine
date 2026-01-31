# CatScript - Categorical State Counting Language

A domain-specific language for trans-Planckian temporal resolution and categorical thermodynamics.

## Quick Start

```catscript
# Calculate temporal resolution at molecular vibration frequency
resolve time at 5.13e13 Hz

# Measure entropy for oscillator system
entropy of 5 oscillators with 4 states

# Simulate temperature evolution
temperature from 300K to 1e-15K steps 100

# Run spectroscopy validation
spectrum raman of vanillin
spectrum ftir of vanillin

# Apply enhancement chain
enhance with ternary multimodal harmonic poincare refinement
```

## Running Scripts

```bash
python -m catscript script.cat           # Run a script file
python -m catscript --repl               # Interactive mode
python -m catscript --validate           # Run full validation
```

## Language Reference

See `docs/language_reference.md` for complete syntax documentation.
