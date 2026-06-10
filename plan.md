# Port plan: `new_eos_update` EOS features

## Goal

Port the EOS features from `origin/new_eos_update` onto the current `durca_0526`
branch:

- richer metamodel beta-equilibrium/proton-fraction physics, including exact
  electron/muon treatment and optional Direct Urca diagnostics;
- Skyrme EOS support;
- standalone `metamodel_only` and `skyrme_only` EOS modes that keep crust
  matching but do not attach CSE or peakCSE;
- validation using the branch notebooks:
  `docs/examples/eos_tov/eos_tov_beta-test.ipynb` and
  `docs/examples/eos_tov/eos_tov_skryme.ipynb`.

## Source Context

Relevant `origin/new_eos_update` commits, oldest to newest:

- `b6a5c5b`: exact proton-fraction calculation, muon effects, examples.
- `05496cc`: Skyrme EOS with exact proton/muon treatment and examples.
- `8d6cecc`: Skyrme example updates.
- `0d603c6`: API compatibility fixes for example files.
- `bac05c7`: Skyrme inference integration.
- `88d2e0b`: data API adjustments.
- `c832813`: large `nbreak` testing.

The branch was based on older main history, so do not bulk-merge it. Port the
targeted EOS changes manually.

## Conflict Rules

- For metamodel and crust implementation disputes, prefer current `main` unless
  the branch contains richer physics.
- Treat exact beta equilibrium with muons and Direct Urca diagnostics as richer
  physics and port them unless they break current APIs.
- Keep current `main` fixes around log-spaced metamodel grids, peakCSE support,
  current config-schema structure, current crust-name validation, and existing
  CSE/peakCSE behavior unless a branch change is clearly required for the new
  physics.
- If a conflict changes physics semantics and cannot be resolved from code,
  notebook behavior, or commit history, stop and ask.

## Implementation Checklist

1. Inspect current `main` metamodel, CSE, peakCSE, config, transform, and tests.
2. Diff `origin/new_eos_update` versions of:
   - `jesterTOV/eos/metamodel/base.py`
   - `jesterTOV/eos/skyrme/*`
   - `jesterTOV/eos/__init__.py`
   - `jesterTOV/inference/config/schemas/eos.py`
   - `jesterTOV/inference/transforms/transform.py`
   - `tests/test_eos.py`
   - the two requested notebooks
3. Port metamodel physics:
   - add `proton_fraction` modes: fixed float, `"approx"`, `"exact"`, and
     default exact with muons;
   - add exact beta-equilibrium solver and muon contributions;
   - add optional Direct Urca threshold diagnostics;
   - preserve current crust loading and log-spaced metamodel density grid.
4. Add Skyrme EOS:
   - add `jesterTOV/eos/skyrme/` with standalone, CSE, and peakCSE classes as
     needed by branch code;
   - update exports;
   - adapt style/API to current `main` rather than copying older config or data
     API assumptions.
5. Add standalone-only EOS modes:
   - expose `metamodel_only` as an alias/config path for crust + metamodel
     without CSE/peakCSE;
   - expose `skyrme_only` as crust + Skyrme without CSE/peakCSE;
   - keep existing `metamodel`, `metamodel_cse`, and `metamodel_peak_cse`
     behavior working for backward compatibility.
6. Update inference config and transform factory:
   - add Skyrme config models;
   - add `calculate_durca` and `proton_fraction` fields where needed;
   - add discriminated-union entries for `metamodel_only` and `skyrme_only`;
   - pass config fields into EOS constructors.
7. Port focused tests:
   - metamodel exact/approx/fixed proton fraction bounds;
   - muon-aware energy/pressure sanity checks;
   - Skyrme constructor and EOS construction;
   - config parsing for new EOS types;
   - transform factory creation for standalone and CSE variants.
8. Port or copy the two example notebooks from `origin/new_eos_update`, then
   update imports/config names if needed for the new standalone aliases.
9. Validate:
   - run focused EOS/config tests first;
   - execute or smoke-test `eos_tov_beta-test.ipynb`;
   - execute or smoke-test `eos_tov_skryme.ipynb`;
   - run broader tests only if focused tests expose cross-module risk.

## Open Decisions

- Branch code uses `skyrme` naming while the user requested `skryme_only`.
  Implement canonical `skyrme_only` and, if low-risk, accept `skryme_only` as a
  compatibility alias.
- Decide whether `metamodel_only` should be a new public type distinct from
  existing `metamodel`, or whether `metamodel` remains the implementation and
  `metamodel_only` is just an alias. Prefer aliasing to avoid duplicating code.

## Status

- Implemented `metamodel_only` as an alias of the existing standalone metamodel
  config path.
- Implemented canonical `skyrme_only` plus compatibility alias `skryme_only`.
- Ported Skyrme standalone, Skyrme+CSE, and Skyrme+peakCSE modules.
- Ported exact beta-equilibrium with muons and optional Direct Urca diagnostics
  into the current metamodel implementation while preserving current crust/grid
  behavior.
- Restored `MetaModel_with_CSE_EOS_model.construct_eos(..., ngrids, cs2grids,
  return_extra=True)` compatibility used by the source notebook.
- Added focused EOS/config tests and smoke-tested the two requested notebooks'
  EOS construction paths.

## Files Changed

- `jesterTOV/eos/metamodel/base.py`: exact beta equilibrium, muons, Direct Urca
  diagnostics, and lepton-aware sound speed while keeping current metamodel
  crust/grid behavior.
- `jesterTOV/eos/metamodel/metamodel_CSE.py`: backward-compatible notebook API
  for explicit CSE grids and `return_extra=True`.
- `jesterTOV/eos/skyrme/`: new Skyrme standalone, CSE, and peakCSE EOS models.
- `jesterTOV/inference/config/schemas/eos.py`: Skyrme config models,
  standalone aliases, `proton_fraction`, and `calculate_durca`.
- `jesterTOV/inference/transforms/transform.py`: factory wiring for Skyrme and
  the new standalone aliases.
- `jesterTOV/inference/run_inference.py`: CSE/peakCSE handling generalized for
  Skyrme-based EOS configs.
- `jesterTOV/utils.py`: muon mass constant and curve-intersection helper for
  Direct Urca threshold calculation.
- `docs/examples/eos_tov/eos_tov_beta-test.ipynb` and
  `docs/examples/eos_tov/eos_tov_skryme.ipynb`: copied from
  `origin/new_eos_update`.
- `tests/test_eos.py` and `tests/test_inference/test_config.py`: focused
  coverage for Skyrme construction and new config aliases.
- `pyproject.toml` / `uv.lock`: explicit `optimistix>=0.0.11` dependency.

## Validation Performed

- `python -m compileall jesterTOV/eos jesterTOV/inference/config
  jesterTOV/inference/transforms jesterTOV/utils.py`
- `uv run ruff check jesterTOV/eos/metamodel/base.py jesterTOV/eos/skyrme
  jesterTOV/inference/config jesterTOV/inference/transforms/transform.py
  jesterTOV/inference/run_inference.py jesterTOV/utils.py tests/test_eos.py
  tests/test_inference/test_config.py`
- `python -m pytest -o addopts='-ra -q --strict-markers --strict-config'
  tests/test_eos.py::TestSkyrmeEOSModel
  tests/test_inference/test_config.py::TestEOSConfig
  tests/test_eos.py::TestMetaModelEOSModel::test_metamodel_construct_eos -q`
- Notebook-derived smoke tests for the EOS construction calls in
  `eos_tov_beta-test.ipynb` and `eos_tov_skryme.ipynb`, using reduced grid sizes.

## Remaining Notes

- The notebooks were not executed end-to-end because they include plotting and
  exploratory batch cells with undefined variables such as `global_nan_mask`.
  The core EOS construction paths used by the notebooks were smoke-tested.
- JAX logs a CUDA plugin warning on this machine because no CUDA device is
  available; the CPU execution path completed successfully.
- Existing untracked `examples/ST_runs/` was intentionally left untouched.
