# simulate_river_spatial_data.py

Synthetic spatial dataset with a deterministic river-corridor effect, designed to test whether spatial models can recover a sharp discontinuous boundary.

Part of the [fakeitright](https://github.com/YOUR_USERNAME/fakeitright) collection of simulation scripts for benchmarking statistical and spatial models.

---

## Purpose

Most spatial benchmarks use smooth Gaussian Process fields. Real-world risk, however, often follows discontinuous geographic structures: flood plains, urban zones, infrastructure corridors. This script simulates one such structure — a river corridor where risk drops abruptly to zero at the edge.

The key question: does your spatial model correctly identify the corridor boundary, or does regularisation smooth the effect beyond its true support?

---

## Data generating process

### Linear predictor

```
m(i) = F(X_i) + b(s_i)
```

Both F and b are scaled to variance ≈ 1, so neither dominates the response.

### Response

| Task | Model | Output |
|---|---|---|
| `binary` (default) | `y ~ Bernoulli(Φ(m))` | 0 or 1 |
| `regression` | `y = exp(m + ε), ε ~ N(0, σ²)` | positive float |

### River corridor spatial effect

A straight river runs from `(0, 0.35)` to `(1, 0.65)` across the unit square. For each point `s`, let `d` be its perpendicular distance to the river centre line:

```
b(s) = A · (1 − d/D)^alpha    if d ≤ D
b(s) = 0                       if d > D
```

| Parameter | Default | Role |
|---|---|---|
| `A` (`--amplitude`) | 1.5 | Peak effect at the river centre line |
| `D` (`--d_cutoff`) | 0.18 | Corridor half-width — hard zero beyond this |
| `alpha` (`--alpha`) | 0.3 | Decay shape inside the corridor |

With `alpha = 0.3` (< 1), the decay is concave: the effect remains high across most of the corridor and drops sharply only near the edge. This flat profile makes the corridor look almost uniform from the inside, which challenges models that rely on the gradient of the effect to locate the boundary.

The discontinuity at `d = D` is the defining challenge: SPDE-based and GP-based models apply smoothness regularisation that tends to bleed the effect beyond the true boundary. A good model should concentrate the spatial field inside the corridor without over-smoothing at the edge.

**Perpendicular distance formula:**

```
d(s) = |(s − p1) × (p2 − p1)| / ‖p2 − p1‖
```

where × denotes the 2D cross product.

### Covariate function

```
F_raw(x) = 1.5x₁ + x₂² + 3·1{x₃ > 0} + x₄·x₅
```

8 covariates: `x₁`–`x₅` are active, `x₆`–`x₈` are pure noise. The river appears only in `b(s)`, not in `F(X)` — there is no confounding between the spatial effect and the covariates. This is the simpler dissociability case (contrast with `simulate_hotspot_spatial_data.py` where the river also enters as a covariate).

### Spatial layout

| Split | Region | Role |
|---|---|---|
| Training | `[0,1]² \ [0.5,1]²` | Model fitting |
| Test (interpolation) | `[0,1]² \ [0.5,1]²` | Generalisation within observed region |
| Test (extrapolation) | `[0.5, 1]²` | Generalisation to unseen spatial zone |

Note that the river crosses both the training zone and the extrapolation zone — the model must extrapolate the corridor effect into the unseen quadrant.

---

## Output files

| File | Description |
|---|---|
| `train_df.csv` | Training dataset |
| `test_interp_df.csv` | Interpolation test dataset |
| `test_extrap_df.csv` | Extrapolation test dataset |
| `plot_locations.png` | Train/test split with river corridor overlaid |
| `plot_covariate_effect.png` | Training scatter coloured by F(X) |
| `plot_spatial_effect.png` | Training scatter coloured by b(s) |
| `simulation_summary.txt` | Parameters and summary statistics |

### Column descriptions

| Column | Description |
|---|---|
| `s1`, `s2` | Spatial coordinates in `[0, 1]²` |
| `x1` … `x8` | Covariates: `x1`–`x5` active, `x6`–`x8` noise, all i.i.d. N(0,1) |
| `F_effect` | Scaled covariate function, variance ≈ 1 |
| `spatial_effect` | River corridor effect `b(s)`, always ≥ 0 |
| `m` | Linear predictor: `F_effect + spatial_effect` |
| `y` | Response: `Bernoulli(Φ(m))` or `exp(m + ε)` |

---

## Usage

**Default run:**

```bash
python simulate_river_spatial_data.py
```

**Regression mode:**

```bash
python simulate_river_spatial_data.py --task regression
```

**Custom corridor geometry:**

```bash
python simulate_river_spatial_data.py \
    --amplitude 2.0 \
    --d_cutoff  0.25 \
    --alpha     1.0   # linear decay instead of concave
```

### All parameters

| Argument | Default | Description |
|---|---|---|
| `--task` | `binary` | Response type: `binary` or `regression` |
| `--seed` | `42` | Random seed |
| `--n_train` | `500` | Training observations |
| `--n_test` | `500` | Observations per test split |
| `--amplitude` | `1.5` | River effect peak value A |
| `--d_cutoff` | `0.18` | Corridor half-width D (hard zero beyond) |
| `--alpha` | `0.3` | Decay exponent. `<1` = slow/concave, `1` = linear, `>1` = convex |
| `--noise_std` | `0.5` | Log-normal noise std (regression only) |
| `--n_calib` | `50000` | Calibration draws for F scaling |
| `--out` | `simulation_output` | Output folder |

### Effect of `--alpha` on the corridor profile

| `alpha` | Decay shape | Implication for the model |
|---|---|---|
| `0.1` | Very flat — near-uniform plateau | Easy inside, hard boundary |
| `0.3` | Concave — **default** | Realistic corridor profile |
| `1.0` | Linear | Easier gradient to follow |
| `2.0` | Convex | Effect drops quickly from centre line |

---

## Dependencies

```bash
pip install numpy pandas scipy matplotlib
```

---

## Related scripts in this repository

| Script | Spatial DGP | Key challenge |
|---|---|---|
| `simulate_gp_spatial_data.py` | Exponential GP (ρ = 0.1) | Short-range field recovery |
| `simulate_river_spatial_data.py` | Deterministic river corridor | **Hard boundary recovery** |
| `simulate_hotspot_spatial_data.py` | Urban Gaussian hotspots | Dissociability under confounding |
