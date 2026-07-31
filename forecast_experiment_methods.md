# AIFS Single v1.1 forecast verification experiment

## Pre-run data and design audit

This audit was completed before forecast inference or verification scope was
changed.

### Cohorts

The two comparisons remain separate:

| Scorecard | 49r1 initializations | 50r1 initializations | Cases per cohort |
| --- | --- | --- | ---: |
| Same season, different year | 13–20 May 2025 | 13–20 May 2026 | 32 |
| Within-2026 cutover | 4–11 May 2026 | 13–20 May 2026 | 32 |

All date bounds are inclusive and all 00/06/12/18 UTC initializations are
included. The shared 13–20 May 2026 cohort is inferred once and reused, so the
experiment contains 96 unique 10-day forecasts rather than 128.

The retained initialization archive has 99 unique states. It includes
12 May 2025 18 UTC, 3 May 2026 18 UTC and 12 May 2026 18 UTC, which provide
the required six-hour lag for the first initialization of each period. Every
other lag is the preceding member of its six-hourly sequence.

### Initialization source

The input archive contains ECMWF public IFS control-forecast fields at step
zero on a 0.25-degree regular latitude/longitude grid. These are the fields
used by ECMWF's public AIFS inference notebook. They are not bit-identical to
the native O1280 operational control analyses used internally by ECMWF.

Each input state contains the complete 90-field dynamic AIFS input inventory:

- `z,t,u,v,w,q` on 13 pressure levels from 50 to 1000 hPa;
- `10u,10v,2d,2t,msl,skt,sp,tcw`;
- soil temperature and soil moisture in layers 1 and 2.

The four static fields (`lsm,z,slor,sdor`) are present once for each IFS cycle
and are repeated across the two input times as required by AIFS. Pressure-level
geopotential height is converted with `z = gh × 9.80665`.

### Model identity

The model is the public `ecmwf/aifs-single-1.1` checkpoint
`aifs-single-mse-1.1.ckpt`. The Hugging Face repository revision observed
before the run is `049b9ab1ccac3382b6332870ae550fd20a432faf`; the checkpoint
object is 993,937,386 bytes. The downloaded object hash is recorded in the run
manifest before inference.

### Common verification reference

Verification uses the public Google ARCO ERA5 archive. Its metadata observed
on 30 July 2026 reported:

- final ERA5 through 30 April 2026;
- preliminary ERA5T through 24 July 2026;
- archive update time 30 July 2026 03:04 UTC.

Consequently, the May 2025 reference is final ERA5 and the May 2026 reference
is ERA5T. Every cohort uses the same archive interface, preprocessing and
grid. Both cutover cohorts use ERA5T; the same-season comparison necessarily
compares final ERA5 in 2025 with ERA5T in 2026. The scorecards label this
explicitly.

The raw ARCO files are daily NetCDF3 files on a 0.25-degree regular grid.
Only the required timestamp slices are retrieved with HTTP byte ranges.
Source URL, object generation, ETag, byte range and decoded units are retained
in the reference manifest.

### Variable matching

The primary scorecards cover 96 instantaneous forecast fields:

- all 78 pressure-level outputs (`z,t,u,v,w,q` at all 13 model levels);
- all 12 prognostic surface/soil outputs also used as dynamic inputs;
- `100u,100v,hcc,mcc,lcc,tcc`.

The six accumulated diagnostic outputs (`cp,tp,ro,sf,ssrd,strd`) are not mixed
into the instantaneous scorecard. AIFS exposes them as forecast accumulations,
whereas ERA5 stores short-forecast accumulations reorganized by valid time.
They require an interval-score definition rather than a point-valid-time
state comparison. Omitting them avoids silently comparing incompatible
accumulation windows. They are listed as an ERA5-variable mismatch in the run
manifest and can be evaluated later as a separate interval scorecard.

ERA5 names, units and transformations are checked before scoring. In
particular, ERA5 pressure-level geopotential is already in `m² s⁻²` and is not
multiplied by gravity. Forecast and reference units must agree after
normalization of spelling only; a mismatch stops the run.

### Regridding and masks

Both inputs and verification fields are interpolated with the same
`earthkit-regrid` matrix from a 0.25-degree regular grid to the checkpoint's
542,080-point N320 reduced Gaussian grid. Initialization fields are rolled
from the public IFS `[-180°,180°)` order to `[0°,360°)` before interpolation.
ARCO ERA5 longitude coordinates are inspected and reordered to that same
source convention before interpolation.

N320 point ordering is checked against the latitude and longitude arrays
embedded in the checkpoint. Area weights are exact Gaussian quadrature
latitude weights divided equally among the points in each latitude row.

Reported domains are global, tropics (20°S–20°N), northern extratropics,
southern extratropics, global land and global ocean. Land/ocean masks use the
intersection of the 49r1 and 50r1 initialization land-sea masks so that a
cycle-dependent coastal mask cannot change the evaluated point set.

### Metrics, samples and uncertainty

For every initialization, variable, domain and 24-hour lead from 24 to 240
hours, the pipeline retains weighted error sum, weighted squared-error sum,
weight sum and valid-point count. Cohort bias is the pooled weighted mean
error. Cohort RMSE is the square root of pooled weighted mean squared error.

Each cohort has 32 forecast cases at every complete lead. Ninety-five percent
confidence intervals are case-bootstrap percentile intervals with a fixed
seed. Differences between cohorts are bootstrapped independently because the
weather cases are not paired causal experiments. The machine-readable output
retains case counts and valid grid-point counts for every cell.

### Computational storage limitation

Persisting every 96-variable N320 field for 96 forecasts at ten retained leads
would require about 200 GB before metadata and compression. The primary run
therefore streams forecast fields into per-case sufficient statistics and
retains full forecast-minus-ERA5 fields only for the highlighted `q50` and
`q100` diagnostics. Checkpoints, preprocessed inputs, reference caches,
per-case metrics, manifests, logs and rendered scorecards are retained in the
RDS experiment directory. This changes storage, not forecast integration or
metric calculation.
