# AIFS v1.1: IFS 49r1 versus 50r1

This repository investigates how the IFS 49r1 → 50r1 analysis-cycle change
affects AIFS Single v1.1, then tests fixed input adapters that do not require
model fine-tuning.

## Main HTML results

- `outputs/open_data_input_comparison_report.html` — input-distribution
  comparison using public IFS step-zero fields.
- `outputs/inferred_v11_50r1_vs_49r1_scorecard.html` — diagnostic inference
  from the two public ECMWF scorecards.
- `outputs/aifs_v11_same_season_forecast_scorecard.html` — direct forecast
  comparison for the same May dates in different years.
- `outputs/aifs_v11_cutover_forecast_scorecard.html` — direct forecast
  comparison immediately before and after the 2026 cutover.
- `outputs/aifs_v11_frozen_adapter_comparison.html` — five-candidate adapter
  ablation.
- `outputs/aifs_v11_corrected_vs_uncorrected_forecast_scorecard.html` —
  primary paired result for the selected correction.

## Final correction result

The primary comparison runs AIFS twice from each of the same sixteen held-out
50r1 initializations from 17–20 May 2026:

1. uncorrected 50r1 inputs;
2. corrected 50r1 inputs using the selected frozen adapter.

Both forecasts use the same AIFS v1.1 checkpoint and ERA5T verification.
Uncertainty uses 4,000 paired-bootstrap draws.

The selected adapter combines:

- additive ERA5T-anchored residual alignment for `q100` and `q150`;
- robust 50r1 → 49r1 median-and-spread mapping for `q50` outside
  20°S–20°N;
- no initial `q50` change in the tropics;
- no changes to the other 87 dynamic inputs or four static inputs.

Its early global paired RMSE changes are −37.4% for `q50`, −15.9% for the
mean of `q100` and `q150`, and −1.1% across the operational guardrails.
Negative values indicate improvement.

## What the five candidates mean

The candidates are ablations of two input transformations, not separate
forecast models:

| Candidate | Plain-language definition | Outcome |
|---|---|---|
| Broad residual | Correct every dynamic field except `q50` | Rejected: harms guardrails |
| Half broad residual | Apply half of the broad correction | Safer, but still weak |
| Targeted residual | Correct only `q100` and `q150` | Conservative fallback |
| All-latitude hybrid | Targeted residual plus median/spread matching for `q50` everywhere | Rejected: large tropical `q50` harm |
| Regional hybrid | Apply the `q50` mapping only outside the tropics | Selected |

See `forecast_adapter_methods.md` for the exact split, formulas, clipping
rules, uncertainty calculation, and limitations.

## Forecast experiment

The direct experiment includes all 00/06/12/18 UTC initializations and the
preceding six-hour state required by AIFS. Forecasts run to ten days and are
verified every 24 hours over all 96 instantaneous outputs and six domains.
May 2025 uses final ERA5; May 2026 uses ERA5T.

On the GPU host:

```bash
.venv-gpu/bin/python src/aifs_forecast_experiment.py audit
.venv-gpu/bin/python src/aifs_forecast_experiment.py cache-inputs
.venv-gpu/bin/python src/aifs_forecast_experiment.py cache-references
.venv-gpu/bin/python src/aifs_forecast_experiment.py run-batch
.venv-gpu/bin/python src/aifs_forecast_experiment.py combine

.venv-gpu/bin/python src/aifs_adapter_experiment.py audit-split
.venv-gpu/bin/python src/aifs_adapter_experiment.py build-profiles
.venv-gpu/bin/python src/aifs_adapter_experiment.py run-batch \
  hybrid_q50_affine_extratropics_q100_q150
.venv-gpu/bin/python src/aifs_adapter_experiment.py combine \
  hybrid_q50_affine_extratropics_q100_q150
```

Render the baseline and adapter reports:

```bash
uv run python src/build_forecast_scorecards.py
uv run python src/build_adapter_report.py \
  --adapters residual_all_no_q50 residual_all_no_q50_half \
  residual_q100_q150 hybrid_q50_affine_q100_q150 \
  hybrid_q50_affine_extratropics_q100_q150 \
  --best hybrid_q50_affine_extratropics_q100_q150
```

## Supporting analyses

The open-data report is regenerated with:

```bash
uv run python src/analyze_open_data_inputs.py
uv run python src/plot_open_data_comparison.py
```

The inferred public-scorecard analysis is regenerated with:

```bash
uv run python src/build_inferred_scorecard.py
```

The retained methods documents are:

- `public_input_comparison_methods.md`;
- `forecast_experiment_methods.md`;
- `forecast_adapter_methods.md`.

## Environment and tests

Recreate the local environment with `uv sync`, then run:

```bash
uv run python -m unittest discover -s tests -v
```
