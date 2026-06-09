# OnePiece Overhaul — Audit Notes (work in progress)

Branch: `fable-overhaul`. Goal: incremental overhaul — first-run experience,
UI design, Python API ergonomics, docs — keeping Streamlit.

## Findings so far

### Repo structure
- `src/onepiece_studio/` is a byte-identical, never-packaged copy of
  `ui/src/onepiece_studio/` (~5k lines). Root `pyproject.toml` only packages
  `onepiece*`; `MANIFEST.in` explicitly prunes it. → delete, update README and
  `docs/source/onepiece_studio_architecture.md` paths.
- Two distributions in one repo: `onepiece` (backend) and `onepiece-studio`
  (`ui/pyproject.toml`). Keep, but document the layout clearly.

### Backend (`src/onepiece`, ~10k lines)
- `__init__.py` is 331 lines re-exporting everything from 16 modules — a flat
  ~100-name API. Hard for newcomers; needs a curated top-level API plus
  documented submodules.
- `adsorption.py` (1051 lines) mixes formula parsing, HDF reading, reference
  assignment, and energy math — split candidates.
- `workflows/registry.py` is only 15 lines; `engine.py` 446 — registry concept
  is underdeveloped.
- Version is hardcoded in `ui` CLI (`onepiece-studio 1.0.0`) instead of read
  from package metadata.

### UI (`ui/src/onepiece_studio`, ~5k lines)
- Huge modules: `workflow_builder.py` 1411, `controlroom.py` 869,
  `streamlit_app.py` 860.
- Pure data logic lives inside UI files (`_formula_counts`,
  `_anonymous_formula`, element parsing in `controlroom.py`) and partially
  duplicates backend functions (`onepiece.adsorption.formula_counts`). →
  move logic to backend, make UI thin (the README itself states this goal).
- Single page, 7 tabs at once (Workflow, Controlroom, Data Management,
  Adsorption & Barriers, Records, Visualize, Schema). "Controlroom" is jargon;
  newcomers get everything at once with no guided path.
- Good bones already present: `onepiece-studio doctor` env check, `qa`
  self-test, bundled tutorial dataset, session onboarding hook.

### Concrete code smells found
- `ui/src/onepiece_studio/adapters.py:53-59`: the `except` branch retries the
  *byte-identical* `pd.read_hdf` call, then falls back to
  `_read_hdf_with_helper_python` — spawns a second Python interpreter (with
  hardcoded `/opt/homebrew` paths on every platform) to convert HDF → pickle →
  re-read. Replace with one read + a clear error message; the numpy 1↔2
  pickle-compat shim (`sys.modules` monkey-patching) needs a documented,
  tested home or removal.
- `workflow_builder.py:_render_notebook_automation` is a single 478-line
  function; `_render_add_derived_column` is 223 lines.
- 24 distinct `st.session_state` keys managed ad hoc across UI modules — no
  central state definition.

### Dev environment
- Repo lives on an NTFS (`ntfs3`) drive: venv creation/pip install on it
  stalls in uninterruptible I/O. Dev venv must live on a Linux filesystem
  (now at `~/.venvs/onepiece`). Worth a note in the contributor docs.
- Python 3.14 works for the install (wheels available as of 2026).

### Tests
- 16 files, ~3.3k lines, biggest is UI state tests. Coverage unknown until
  baseline run completes (venv build in progress — PyTables compiles from
  source, which itself is a first-run-experience pain point worth fixing:
  consider making `tables` optional or documenting wheels).

### Docs
- 25 markdown pages incl. `first_day_student.md`, worked examples,
  troubleshooting — content exists but needs an information architecture pass
  (audience-based entry points: student / notebook user / UI user).

## Next steps
1. Finish baseline (pytest + ruff) once env build completes.
2. Delete `src/onepiece_studio`, fix doc references.
3. Full roadmap with prioritized phases.
