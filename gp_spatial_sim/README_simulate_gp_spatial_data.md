# simulate_gp_spatial_data.py

Synthetic spatial dataset generator following the simulation protocol from:

> Sigrist F. (2022). IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2).

Part of the fakeitright collection of simulation scripts for benchmarking statistical and spatial models.
Claude was used to format and generate part of the code and documentation.
---

## Purpose

Benchmarking spatial models requires knowing the ground truth. Real datasets do not provide this: you observe a response, but you cannot verify whether a model correctly separated the covariate effect from the spatial effect.

This script solves that problem by generating data where both components are known by construction. The response depends on a non-linear covariate function **F(X)** and a spatially structured effect **b(s)**, both with variance ≈ 1. Since the true decomposition is stored in the output, any fitted model can be evaluated not only on predictive accuracy but on whether it correctly recovers each component.

Two response types are supported: **binary classification** (default) and **positive-valued regression**.

---

## Data generating process

### Linear predictor (shared across tasks)

```
m(i) = F(X_i) + b(s_i)
```

Both F and b are calibrated to variance ≈ 1, so neither dominates by construction.

### Response — binary (default)

```
y ~ Bernoulli(Φ(m))
```

Probit link. `y` is 0 or 1. Use this for classification benchmarks and spatial logistic regression.

### Response — regression

```
y = exp(m + ε),   ε ~ N(0, noise_std²)
```

Log-normal response. `y` is always positive. Equivalently, `log(y) ~ N(m, noise_std²)`.

The log-normal is a natural choice for positive-valued outcomes such as insurance claim amounts, precipitation totals, or environmental concentrations. It uses the same linear predictor `m` as the binary case, so results are directly comparable across tasks. The `--noise_std` parameter controls how much additional variance is added on top of the signal.

### Covariate function

```
F_raw(x) = 2x₁ + x₂² + 4·1{x₃ > 0} + 2·log|x₁|·x₃
```

Nine covariates `x₁, …, x₉` are drawn i.i.d. from N(0, 1). Only `x₁`, `x₂`, `x₃` enter `F_raw`. Covariates `x₄` through `x₉` are pure noise and should be discarded by any well-calibrated model.

`F_raw` is centred and scaled empirically on a large calibration draw (n = 50,000 by default) so that `F_scaled = C2 · (F_raw − C1)` has mean ≈ 0 and variance ≈ 1. Constants `C1` and `C2` are reported in the summary file.

The function combines four structures deliberately:
- a linear term (`2x₁`)
- a quadratic term (`x₂²`)
- a discontinuous step (`4·1{x₃ > 0}`)
- a non-linear interaction with log-transform (`2·log|x₁|·x₃`)

This makes it non-trivial for tree-based or linear models to recover F(X) exactly.

### Spatial effect

```
b(s) ~ GP(0, σ²·exp(−‖s − s'‖ / ρ))
```

The exponential (Matérn 1/2) covariance produces a **rough, short-range** spatial field. With the default `ρ = 0.1`, spatial correlation decays rapidly: two points 0.3 units apart have correlation ≈ 0.05. This is a deliberately challenging benchmark for spatial smoothers, which tend to over-smooth such fields.

The GP is drawn jointly across all locations (training, interpolation, extrapolation) via Cholesky decomposition, ensuring spatial consistency between splits.

### Spatial layout

Locations are drawn uniformly in `[0, 1]²` with the following split:

| Split | Region | Role |
|---|---|---|
| Training | `[0,1]² \ [0.5,1]²` | Model fitting |
| Test (interpolation) | `[0,1]² \ [0.5,1]²` | Generalisation within observed region |
| Test (extrapolation) | `[0.5, 1]²` | Generalisation to unseen spatial zone |

The top-right quadrant is entirely held out from training. This tests whether a model can extrapolate the spatial field beyond its observed support, which is much harder than interpolation and more realistic for applications where new locations must be scored without historical data.

---

## Output files

All files are written to the output folder (default: `./simulation_output/`).

| File | Description |
|---|---|
| `train_df.csv` | Training dataset |
| `test_interp_df.csv` | Interpolation test dataset |
| `test_extrap_df.csv` | Extrapolation test dataset |
| `spatial_locations.png` | Scatter plot of the train/test spatial split |
| `simulation_summary.txt` | Parameters, calibration constants, summary statistics |

### Column descriptions

| Column | Description |
|---|---|
| `s1`, `s2` | Spatial coordinates in `[0, 1]²` |
| `x1` … `x9` | Covariates, i.i.d. N(0,1). `x1`–`x3` are active, `x4`–`x9` are noise |
| `F_effect` | Scaled covariate function `F(X)`, variance ≈ 1 |
| `spatial_effect` | GP realisation `b(s)`, variance ≈ 1 |
| `m` | Linear predictor: `F_effect + spatial_effect` |
| `y` | Response: `Bernoulli(Φ(m))` for binary, `exp(m + ε)` for regression |

The ground-truth columns `F_effect` and `spatial_effect` are included so that model estimates can be compared directly to the true decomposition using metrics such as Pearson correlation or RMSE per component.

---

## Usage

**Default run** — binary, 500 training points, output to `./simulation_output/`:

```bash
python simulate_gp_spatial_data.py
```

**Regression mode** — log-normal positive response:

```bash
python simulate_gp_spatial_data.py --task regression
```

**Custom parameters:**

```bash
# Binary with larger sample and smoother GP
python simulate_gp_spatial_data.py \
    --task    binary \
    --n_train 1000   \
    --n_test  500    \
    --rho     0.3    \
    --seed    0      \
    --out     my_output

# Regression with low noise
python simulate_gp_spatial_data.py \
    --task      regression \
    --noise_std 0.2        \
    --rho       0.1        \
    --out       my_output
```

### All parameters

| Argument | Default | Description |
|---|---|---|
| `--task` | `binary` | Response type: `binary` (probit) or `regression` (log-normal) |
| `--seed` | `42` | Random seed for full reproducibility |
| `--n_train` | `500` | Number of training observations |
| `--n_test` | `500` | Number of observations per test split (interp and extrap) |
| `--sigma2` | `1.0` | GP marginal variance σ² |
| `--rho` | `0.1` | GP range parameter ρ. Smaller = rougher, shorter-range field |
| `--noise_std` | `0.5` | Std of log-normal noise (regression task only) |
| `--n_calib` | `50000` | Sample size used to calibrate F to mean=0, Var=1 |
| `--out` | `simulation_output` | Output folder (created if it does not exist) |

### Effect of `--rho` on the spatial field

| `rho` | Field character | Typical use |
|---|---|---|
| `0.05` | Very rough, near-nugget | Stress test for spatial smoothers |
| `0.1` | Rough, short-range | **Default** |
| `0.25` | Smooth, medium-range | Easier benchmark, closer to real environmental data |
| `0.5` | Very smooth, long-range | Risk of confounding with covariate effects |

### Effect of `--noise_std` on the regression response

| `noise_std` | Signal-to-noise ratio | Interpretation |
|---|---|---|
| `0.1` | High | Clean log-normal; log(y) ≈ m |
| `0.5` | Medium | **Default** |
| `1.0` | Low | Heavy additional noise; harder to recover F and b |
| `2.0` | Very low | Extreme noise; mostly a stress test |

---

## Dependencies

```
numpy
pandas
scipy
matplotlib
```

Install with:

```bash
pip install numpy pandas scipy matplotlib
```

---

## Initialisation

- `--task binary`, `--n_train 500`, `--n_test 500`
- GP with exponential covariance: `--sigma2 1.0`, `--rho 0.1`
- Covariate function: `F_raw = 2x₁ + x₂² + 4·1{x₃>0} + 2·log|x₁|·x₃`
- Probit link: `y ~ Bernoulli(Φ(F(X) + b(s)))`
- Spatial split: training and interpolation in `[0,1]² \ [0.5,1]²`, extrapolation in `[0.5,1]²`

Results will differ across runs in terms of specific GP realisations.

---