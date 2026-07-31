# Frozen IFS 50r1 input-adapter experiment

## Decision question

Can a fixed preprocessing adapter improve AIFS Single v1.1 forecasts
initialized from IFS Cycle 50r1 without fine-tuning the model?

Adapter forecasts are compared directly with unmodified forecasts from the
same 50r1 initializations and against the same ERA5T valid-time fields. This
paired comparison isolates the effect of the adapter from differences in
weather cases.

## Chronological calibration and evaluation

The experiment reuses the downloaded within-2026 cohorts:

- 49r1 calibration: all 32 initializations from 4–11 May 2026;
- 50r1 calibration: 15 initializations from 13 May 00 UTC through
  16 May 12 UTC;
- 50r1 evaluation: all 16 initializations from 17–20 May 2026.

The six-hour lag of the first evaluation initialization is 16 May 18 UTC,
after the calibration cutoff. Therefore neither member of any evaluation
`(t−6 h, t0)` input pair occurs in calibration.

ERA5T is used only to estimate the frozen profiles. Once estimated, the
additive adapters require only the incoming IFS fields at runtime.

## Frozen residual-bias profile

For every calibration state, field and spatial stratum, the retained
statistic is the spatial median of:

`IFS step-0 − ERA5T`.

The frozen additive correction is:

`d(s) = median(r50 | s) − median(r49 | s)`

`x50_corrected = x50 − α d(s)`.

Profiles use 10-degree latitude bands but pool all four UTC hours. Pressure
fields use latitude only. Atmospheric surface fields additionally separate
land and ocean; soil fields are changed over common land only. Each
coefficient therefore uses 32 49r1 states and 15 50r1 states. This produces
1,756 coefficients, compared with 56,192 rows in the original detailed
leave-day-out profile.

## Adapter candidates

These are controlled ablations of two simple operations, not five different
forecast models. Every candidate uses the same frozen AIFS checkpoint.

- **Residual alignment** subtracts the estimated systematic cycle-dependent
  offset. It changes the centre of a field's distribution but not its spread.
- **Robust affine mapping** changes both the centre and spread of `q50`,
  separately in each latitude band, while preserving ranks within the band.

| Candidate | What is changed | Purpose |
|---|---|---|
| `residual_all_no_q50` | Full residual alignment for all 90 dynamic fields except `q50` | Test whether broad distribution alignment is helpful |
| `residual_all_no_q50_half` | The same broad correction at half strength | Test whether regularization removes broad-correction harms |
| `residual_q100_q150` | Residual alignment for `q100` and `q150` only | Conservative targeted fallback |
| `hybrid_q50_affine_q100_q150` | Targeted `q100`/`q150` residual alignment plus affine `q50` mapping at all latitudes | Test whether matching the large `q50` scale shift helps |
| `hybrid_q50_affine_extratropics_q100_q150` | The same hybrid, but affine `q50` mapping only outside 20°S–20°N | Retain extratropical gains without the tropical failure |

The hybrid estimates, separately by latitude band, the median of state
spatial medians and the median of state spatial interquartile ranges. It maps
the 50r1 `q50` location and scale to the 49r1 values while preserving ranks:

`q50' = location49 + (q50 − location50) × IQR49 / IQR50`.

Scale ratios are clipped to `[0.25, 4]`. This is deliberately treated as an
experimental candidate: unlike the residual-bias correction, marginal
location/scale matching can also attenuate genuine weather anomalies.

Specific humidity is clipped at zero after a non-identity transformation.
Soil moisture is clipped to `[0,1]`. Static fields are unchanged.

## Selection result

The all-latitude hybrid revealed an important failure mode: early tropical
`q50` RMSE increased by 77.4%. Restricting the affine `q50` operation to the
extratropics changed the tropical result to a 5.5% improvement. The regional
hybrid was therefore selected.

On the 16 held-out cases, its early global RMSE changes are:

- `q50`: −37.4%;
- mean of `q100` and `q150`: −15.9%;
- operational guardrail mean: −1.1%.

The `q100`/`q150`-only candidate remains the conservative fallback: it gives
a smaller humidity improvement but changes no other input fields.

## Forecast verification and paired uncertainty

Every candidate uses the same checkpoint, N320 preprocessing, ten-day
integration, ERA5T references, 24-hour leads, 96 instantaneous forecast
fields, six domains, area weights and sufficient-statistic scoring as the
baseline experiment.

For each variable, region and lead, baseline and adapter per-case errors are
joined by initialization ID. The reported treatment effect is:

`100 × (RMSEadapter / RMSEbaseline − 1)`.

Negative values indicate improvement. Ninety-five percent intervals use
4,000 paired case-bootstrap draws: one set of initialization indices is
applied to both baseline and adapter errors in each draw.

The operational guardrails are Z500, T850, 2 m temperature, 10 m wind
components and mean sea-level pressure. Upper-atmosphere humidity
(`q50`, `q100`, `q150`) is reported as the primary shift diagnostic.

## Limitations

- The correction is calibrated on short adjacent weather periods rather
  than parallel 49r1/50r1 analyses.
- ERA5T is a common anchor, but IFS-minus-ERA5T residuals can still depend on
  the sampled weather regime.
- The 16-case chronological test supports a fast patch decision, not a
  production climatology.
- The robust `q50` marginal map is not guaranteed to preserve multivariate
  or vertical physical relationships and should be rejected if it harms the
  forecast guardrails.
