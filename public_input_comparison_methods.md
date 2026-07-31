# Public AIFS v1.1 input-distribution comparison

## Question

Do public IFS step-zero fields drawn from the operational 49r1 and 50r1
periods show a distribution shift in the variables consumed by AIFS v1.1?

This is a descriptive, unpaired cohort comparison. It is not a causal
same-weather cycle experiment.

## Cohorts

| Comparison | 49r1 cohort | 50r1 cohort | t0 samples |
| --- | --- | --- | ---: |
| Same season | 13–20 May 2025 | 13–20 May 2026 | 32 + 32 |
| Cutover | 4–11 May 2026 | 13–20 May 2026 | 32 + 32 |

All 00/06/12/18 UTC initializations are included. Each cohort includes one
additional state six hours before its first t0. The 50r1 cohort is reused by
both comparisons, leaving 99 unique downloaded states.

There is no 12 May 2026 t0 sample. The 12 May 18 UTC state is retained only as
t−6 for 13 May 00 UTC; both states in that pair are 50r1.

## Source and identity

- Anonymous AWS mirror: `https://ecmwf-forecasts.s3.amazonaws.com/`
- Model: IFS control forecast
- Resolution: regular latitude/longitude 0.25 degrees
- Type and step: `fc`, step 0
- 49r1 streams: `oper` at 00/12 UTC and `scda` at 06/18 UTC
- 50r1 stream: `oper` at all four base times

The operational cycle is established by timestamp and ECMWF's documented
cutover at 12 May 2026 06 UTC. `expver=0001` is shared across operational
cycles and does not identify the IFS cycle by itself.

The public fields are the source used by the official community inference
notebook. They are not bit-identical to the native O1280 operational control
analyses used by ECMWF internally.

## Fields

- Pressure levels: `gh,t,u,v,w,q` at
  1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100 and 50 hPa
- Surface: `10u,10v,2d,2t,msl,skt,sp,tcw`
- Soil: `vsw,sot`, layers 1 and 2
- Static, once per cycle: `lsm,z,slor,sdor`

This is 90 dynamic messages per state and four static messages per cycle.
Remote JSON-lines indexes determine exact byte ranges. Nearby selected ranges
are coalesced to reduce request count; only selected GRIB messages are retained.
All raw data and library interpolation assets are stored under this assignment
directory; no experiment cache is written to the user's home cache.

## Preprocessing

1. Validate timestamp, stream, step, parameter, level, units, missing values
   and the 1440 × 721 regular grid.
2. Roll longitude exactly as in the official notebook.
3. Interpolate 0.25-degree values to N320 with `earthkit-regrid`.
4. Rename soil variables to `swvl1/2` and `stl1/2`.
5. Convert pressure-level height with `z = gh × 9.80665`.
6. Apply the variable-specific normalization defined by the published AIFS
   v1.1 checkpoint.

The normalization statistics and N320 coordinates are read from small metadata
members inside the remote checkpoint ZIP. Model weights are neither downloaded
nor unpickled.

## Diagnostics

For each comparison, field, level, region, land/ocean class and UTC:

- area-weighted mean, standard deviation and variance;
- 1st, 5th, 25th, 50th, 75th, 95th and 99th percentiles;
- mean shift in raw and AIFS-normalized units;
- variance ratio `variance(50r1) / variance(49r1)`;
- one-dimensional Wasserstein distance in raw and normalized units.

The same diagnostics are calculated for six-hour tendencies
`x(t0) − x(t−6)`. Quantiles and Wasserstein distances use a deterministic
area-weighted sample of N320 points. Means and variances use all points with
Gaussian quadrature area weights.

The four static forcings are compared directly between the two cycles and
written to a separate table and map bundle; they are not mixed into the
time-varying cohort summaries. Because every displayed static comparison is
zero, the redundant static scorecard is omitted from the HTML presentation.

Regions are global, tropics (20°S–20°N) and extratropics (absolute latitude
greater than 20°). Land and ocean results use the intersection of the two
cycle land-sea masks; changed coastal cells appear only in the all-surface
result.

## Interpretation

The two comparisons are reported separately. Shifts with the same direction
in both comparisons receive the greatest weight. Conflicting shifts are
labelled as likely sensitive to weather, seasonal progression or interannual
variability. No claim of perfect causal isolation is made.

## Presentation order

All complete scorecard matrices use the same 90-field x-axis. The order
extends `OFFICIAL_FIELD_ORDER` from the inferred scorecard:

1. Group by parameter.
2. Order pressure levels 50, 100, 150, 200, 250, 300, 400, 500, 600, 700,
   850, 925 and 1000 hPa.
3. Place related single-level fields after the corresponding pressure block.
4. Finish with the remaining surface and soil fields.

Rows are global, tropical, extratropical, global land and global ocean,
followed by global 00/06/12/18 UTC. The same order is used for state and
six-hour-tendency mean shifts, log2 variance ratios and normalized
Wasserstein distances. Signed scorecards use a symmetric-log colour scale;
the underlying values remain linear and are available in the comparison
tables.

## Official references

- [ECMWF Open Data documentation](https://confluence.ecmwf.int/spaces/DAC/pages/272310539/ECMWF+open+data+real-time+forecasts+from+IFS+and+AIFS)
- [Published AIFS v1.1 checkpoint](https://huggingface.co/ecmwf/aifs-single-1.1)
- [ECMWF AIFS inference guidance](https://confluence.ecmwf.int/spaces/UDOC/pages/599165906/AIFS+How+To+Generate+a+forecast+with+the+AIFS)
