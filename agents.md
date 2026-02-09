# agents.md

## Mission
Build and maintain a pedagogical Python DFT implementation (`pydft`) and a companion book (`pydft-book`) that bridges theory and code. The pydft should be based on Rust code in dftworks

## Scope
- `pydft/`: executable, readable Python modules for a minimal Kohn-Sham DFT workflow. Based on the Rust code in dftworks
- `pydft-book/`: chaptered documentation explaining the physics, numerics, and corresponding code.

## Working Rules
1. Keep modules small and testable.
2. Prefer explicit math in code over hidden abstractions.
3. Every new module must include:
   - docstring with equations/assumptions,
   - at least one unit test,
   - one cross-reference in `pydft-book`.

## Testing Rules
- Run tests module-by-module first, then full suite.

## Documentation Rules
- Each chapter should answer:

## Quality Bar
- No silent failures.
- Clear error messages for invalid inputs.
- Numerical tolerances documented in tests.
