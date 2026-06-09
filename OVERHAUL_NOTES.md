# OnePiece Overhaul — Audit & Roadmap

Branch: `fable-overhaul`. Goal: incremental overhaul keeping Streamlit —
better first-run experience, clearer UI, friendlier Python API, restructured
docs. Target users: grad students in computational catalysis.

## Done so far

- **Baseline established**: 134 tests pass, ruff clean (venv at
  `~/.venvs/onepiece`, Python 3.14).
- **NumPy 2 compatibility fixed** (`48ab149`): `np.trapz` was removed in
  NumPy 2.0 and broke 8 tests / 9 call sites; now routed through
  `onepiece._compat.trapezoid`.
- **Missing dependency declared** (`48ab149`): `onepiece.ir` imports seaborn,
  which was never in any requirements list; added to studio/dev/docs extras.
- **Dead duplicate removed** (`31413af`): `src/onepiece_studio/` was a
  byte-identical, never-packaged copy of `ui/src/onepiece_studio/` (~5k
  lines). `ui/src` is now the single source of truth.

## Audit findings

### Backend (`src/onepiece`, ~10k lines)
- `__init__.py` re-exports ~100 names from 16 modules in one flat namespace —
  overwhelming for newcomers, no guidance on what matters.
- `adsorption.py` (1051 lines) mixes formula parsing, HDF reading, reference
  assignment, and energy math.
- `workflows/registry.py` is 15 lines against a 446-line engine — the
  registry concept is vestigial.
- UI CLI hardcodes `onepiece-studio 1.0.0` instead of reading package
  metadata.

### UI (`ui/src/onepiece_studio`, ~5k lines)
- `adapters.py:53-59` retries the byte-identical `pd.read_hdf` call in its
  `except` branch, then falls back to spawning a *second Python interpreter*
  (with hardcoded `/opt/homebrew` paths) to convert HDF → pickle → re-read.
  Needs one read path with a clear error; the NumPy 1↔2 pickle-compat shim
  (`sys.modules` monkey-patching) needs a documented, tested home.
- Giant modules/functions: `workflow_builder.py` 1411 lines
  (`_render_notebook_automation` alone is 478), `controlroom.py` 869,
  `streamlit_app.py` 860.
- Pure data logic (formula parsing, element counting) lives in UI files and
  partially duplicates backend functions (`onepiece.adsorption.formula_counts`
  vs `controlroom._formula_counts`) — the README's own "thin UI" goal is not
  upheld.
- 24 distinct `st.session_state` keys managed ad hoc, no central definition.
- Single page with 7 simultaneous tabs (Workflow, Controlroom, Data
  Management, Adsorption & Barriers, Records, Visualize, Schema).
  "Controlroom" is jargon; a new user gets everything at once.
- Good bones to keep: `doctor` env check, `qa` self-test, bundled tutorial
  dataset, beginner-guidance hooks.

### Tests & tooling
- 134 tests, heaviest on UI workflow state; core science modules thinner.
  No coverage measurement, no CI workflow in the repo.

### Docs (25 pages)
- Content is rich (worked examples, `first_day_student.md`, troubleshooting)
  but flat — no audience-based entry paths.

### Dev environment
- Repo lives on an NTFS (`ntfs3`) mount: creating a venv there stalls in
  uninterruptible I/O. Venvs must live on a Linux filesystem
  (`~/.venvs/onepiece`). Document for contributors.

## Roadmap

### Phase 1 — Foundations (correctness & hygiene)
1. Rewrite `HDFSource.load`: single read path, clear actionable errors
   (file missing / wrong key / legacy pickle), isolate the NumPy pickle shim,
   delete the helper-interpreter fallback.
2. Version from `importlib.metadata` everywhere; single source in pyproject.
3. Move pure data logic out of UI into the backend; UI imports it.
4. Central `state.py` defining all session-state keys.
5. Add GitHub Actions CI: pytest + ruff, Python 3.10–3.14.

### Phase 2 — First-run experience
1. `onepiece-studio` with no args → welcome page: open tutorial dataset,
   pick a file, recent files list — instead of requiring subcommands.
2. Friendly failure screens (bad HDF, wrong key, missing optional deps)
   with the fix spelled out.
3. Evaluate making PyTables optional (parquet path doesn't need it) to
   slim the default install.
4. Contributor setup guide (incl. NTFS pitfall).

### Phase 3 — UI redesign (within Streamlit)
1. Multi-page app (`st.navigation`) replacing the 7-tab wall:
   Data → Filter → Analyze → Visualize → Export, with a guided flow.
2. Rename jargon ("Controlroom" → "Filter"), progressive disclosure for
   advanced tools (workflow builder, schema inspector).
3. Split the giant modules one-page-per-module; add `st.cache_data` on loads.

### Phase 4 — Python API
1. Curated top-level namespace (~15 core functions), documented submodules,
   deprecation aliases for everything moved.
2. Split `adsorption.py`; docstrings with examples on the core API.

### Phase 5 — Docs & tutorials
1. Three entry paths: "I have a dataset, show me" (UI), "I write notebooks"
   (API), "I'm starting my thesis" (concepts).
2. Quickstart rewritten against the new welcome flow; troubleshooting
   consolidated.

### Phase 6 — Test depth
1. Coverage measurement in CI; raise coverage of energy/reference math;
   golden-file tests for the worked examples.
