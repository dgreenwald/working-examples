# Repository Guidelines

## Project Structure & Module Organization
- `main.py` orchestrates solves via `create_model()` and handles calibration or solver flag changes.
- `spec.py` defines ModelBlock helpers and defaults; add blocks/parameters there before wiring them into `create_model()`.
- Caches live in `__pycache__/`. Keep large data, plots, or notebooks in private scratch space instead of the repo.

## Build, Test, and Development Commands
- `python main.py`: builds the model, calibrates, and saves the steady state (set block toggles in `main.py`).
- `python -m spec` (or `python -i main.py`): quick import validation plus REPL access to `mod.rules` and parameter dictionaries.
- `PYTHONPATH=.. python main.py`: needed when running from sibling repos so `danare` resolves.

## Coding Style & Naming Conventions
- Follow Black/PEP8 defaults: 4-space indentation, snake_case identifiers, and type hints.
- Name ModelBlock helpers with `_block` suffix and keep agent placeholders (`AGENT`, `TYPE`) so `.add_block(..., rename=...)` stays uniform.
- Store calibration variables in `mod.params` and initial guesses in `mod.steady_guess`; suffix entries with explicit agent tags (e.g., `_U`, `_C`).

## Testing Guidelines
- Deterministic solves double as regression tests. After any edit, run `python main.py` and confirm convergence.
- For block-level tweaks, drop checks under a temporary `if __name__ == "__main__"` guard in `spec.py` and remove them before committing.
- Document new shocks or parameters by adding to `get_default_params()` with a short inline comment.

## Commit & Pull Request Guidelines
- History favors imperative, descriptive titles (`adding firm structure and investment block`). Start with a verb and keep scope focused.
- Each PR should describe calibration changes, affected equations, and reproduction steps (`python main.py`). Link related research memos or issue IDs when relevant.
- Include plots or numeric summaries when the change alters dynamics; attach them as comments/files rather than committing binaries.

## Configuration & Security Notes
- The project depends on the private `danare` package; ensure your `.env` or environment module exposes it on `PYTHONPATH` and never commit credentials.
- Use deterministic seeds for shocks when sharing results so others can replicate IRF pages exactly.

## Local Data Storage (Codex Runs)
- Codex cannot write to synced paths such as the Dropbox entry in `.env`; create `local_danare_data` via `mkdir -p local_danare_data` before running.
- Point danare outputs to this folder temporarily (`DANARE_PATHS__DATA_DIR=$(pwd)/local_danare_data python main.py`) or by updating `.env` with the absolute path while Codex is active.
- Keep `local_danare_data/` out of version control (see `.gitignore`) and clear it before sharing work.
