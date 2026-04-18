"""
simulate_gp_spatial_data.py
─────────────────────────────────────────────────────────────────────────────
This script reproduces the spatial simulation protocol from:
  Sigrist F. (2022) IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2).

Claude was used to format and generate part of the code and documentation.

What this script does
─────────────────────
Generates a synthetic dataset where the response depends on BOTH a
non-linear covariate function F(X) AND a spatially structured effect b(s)
drawn from a Gaussian Process.

Two response types are supported via --task:

  binary     (default)
    y ~ Bernoulli(Φ(m))           probit link
    Use for: classification benchmarks, spatial logistic regression.

  regression
    y = exp(m + ε),  ε ~ N(0, noise_std²)    log-normal response
    Use for: positive-valued outcomes such as claim amounts, counts,
             environmental measurements. The log-normal keeps the same
             linear predictor m so results are directly comparable
             across tasks.

The idea is to benchmark methods that claim to disentangle covariate
effects from spatial effects. Since both F_effect and spatial_effect are
stored in the output, any fitted model can be evaluated not only on
predictive accuracy but on whether it recovers each component correctly.

The spatial setup
─────────────────
  - Training + interpolation: uniform in [0,1]² \ [0.5,1]²  (≈ 75%)
  - Extrapolation:            uniform in [0.5,1]²  (held-out quadrant)

Output columns (in each CSV)
─────────────────────────────
  s1, s2           : spatial coordinates in [0,1]²
  x1 … x9         : i.i.d. N(0,1) covariates (x1–x3 active, x4–x9 noise)
  F_effect         : scaled covariate function F(X), variance ≈ 1
  spatial_effect   : GP realisation b(s), variance ≈ 1
  m                : linear predictor  m = F_effect + spatial_effect
  y                : response (see --task above)

Usage
─────
  python simulate_gp_spatial_data.py                         # binary, defaults
  python simulate_gp_spatial_data.py --task regression       # log-normal response
  python simulate_gp_spatial_data.py --n_train 1000 --rho 0.3 --seed 0
  python simulate_gp_spatial_data.py --task regression --noise_std 0.3 --out my_folder
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PARAMETERS  (all overridable via CLI)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    task       = "binary",        # "binary" or "regression"
    seed       = 42,
    n_train    = 500,             # training sample size
    n_test     = 500,             # size of each test set (interp & extrap)
    sigma2     = 1.0,             # GP marginal variance σ²
    rho        = 0.1,             # GP range parameter ρ
    noise_std  = 0.5,             # log-normal noise std (regression only)
    n_calib    = 50_000,          # draws used to calibrate F to mean=0, Var=1
    out        = "simulation_output",
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOCATION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def sample_lower_left(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Sample n points uniformly in [0,1]² excluding the top-right quadrant.

    The top-right quadrant [0.5,1]² is reserved for extrapolation testing.
    Rejection sampling is used: roughly 75% of draws are accepted on average.

    Returns: (n, 2) array of (s1, s2) coordinates.
    """
    pts = []
    while len(pts) < n:
        candidates = rng.uniform(0, 1, size=(n * 2, 2))
        in_extrap  = (candidates[:, 0] >= 0.5) & (candidates[:, 1] >= 0.5)
        pts.extend(candidates[~in_extrap].tolist())
    return np.array(pts[:n])


def sample_top_right(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Sample n points uniformly in [0.5,1]².
    This is the extrapolation region: no training points ever fall here.

    Returns: (n, 2) array of (s1, s2) coordinates.
    """
    return rng.uniform(0.5, 1.0, size=(n, 2))


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAUSSIAN PROCESS WITH EXPONENTIAL COVARIANCE
# ─────────────────────────────────────────────────────────────────────────────

def exponential_cov(locs_a: np.ndarray,
                    locs_b: np.ndarray,
                    sigma2: float,
                    rho: float) -> np.ndarray:
    """
    Compute the exponential (Matérn 1/2) covariance matrix.

      c(s, s') = σ² · exp(−‖s − s'‖ / ρ)

    Short range (ρ = 0.1) produces a rough, rapidly varying spatial field,
    which is a challenging benchmark for spatial models.

    Args:
        locs_a: (n, 2) array of spatial locations.
        locs_b: (m, 2) array of spatial locations.
        sigma2: marginal variance σ².
        rho:    range parameter ρ controlling spatial correlation decay.

    Returns: (n, m) covariance matrix.
    """
    diff  = locs_a[:, None, :] - locs_b[None, :, :]   # (n, m, 2)
    dists = np.sqrt((diff ** 2).sum(axis=-1))          # (n, m) Euclidean distances
    return sigma2 * np.exp(-dists / rho)


def draw_gp_joint(rng: np.random.Generator,
                  train_locs: np.ndarray,
                  interp_locs: np.ndarray,
                  extrap_locs: np.ndarray,
                  sigma2: float,
                  rho: float) -> tuple:
    """
    Draw one GP realisation jointly over all locations via Cholesky decomposition.

    Drawing jointly ensures spatial consistency: the GP values at test locations
    are correlated with those at training locations, as they would be in reality.
    A small nugget (1e-8) is added to the diagonal for numerical stability.

    Returns: (b_train, b_interp, b_extrap) — one array per split.
    """
    all_locs = np.vstack([train_locs, interp_locs, extrap_locs])
    n_total  = len(all_locs)

    K  = exponential_cov(all_locs, all_locs, sigma2, rho)
    K += 1e-8 * np.eye(n_total)                  # nugget for numerical stability
    L  = np.linalg.cholesky(K)                   # Cholesky factor L: K = L Lᵀ
    b  = L @ rng.standard_normal(n_total)        # GP draw: b = L·z, z ~ N(0, I)

    n_tr = len(train_locs)
    n_ti = len(interp_locs)
    return b[:n_tr], b[n_tr:n_tr + n_ti], b[n_tr + n_ti:]


# ─────────────────────────────────────────────────────────────────────────────
# 3. COVARIATE FUNCTION F(X)
# ─────────────────────────────────────────────────────────────────────────────

def F_raw(X: np.ndarray) -> np.ndarray:
    """
    Non-linear covariate function.

      F_raw(x) = 2x₁ + x₂² + 4·1{x₃>0} + 2·log|x₁|·x₃

    Only x₁, x₂, x₃ are active. Columns x₄–x₉ are noise covariates
    included to test whether models correctly discard irrelevant features.

    The function combines four structures deliberately:
      - a linear term                (2x₁)
      - a quadratic term             (x₂²)
      - a step function              (4·1{x₃>0})
      - a log-transform interaction  (2·log|x₁|·x₃)

    Args:
        X: (n, 9) array of covariates, all drawn i.i.d. from N(0, 1).

    Returns: (n,) array of raw covariate effects.
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (2 * x1
            + x2 ** 2
            + 4 * (x3 > 0).astype(float)
            + 2 * np.log(np.abs(x1)) * x3)


def calibrate_F(rng: np.random.Generator, n_calib: int) -> tuple:
    """
    Estimate centering (C1) and scaling (C2) constants from a large i.i.d. draw.

    Goal: F_scaled = C2 · (F_raw − C1) has mean ≈ 0 and Var ≈ 1.
    This ensures F and b(s) contribute equally to the linear predictor m,
    regardless of the natural scale of F_raw.

    Returns: (C1, C2) as floats.
    """
    X_cal = rng.standard_normal((n_calib, 9))
    f     = F_raw(X_cal)
    c1    = float(np.mean(f))
    c2    = float(1.0 / np.std(f))
    return c1, c2


def F_scaled(X: np.ndarray, c1: float, c2: float) -> np.ndarray:
    """Apply centering and scaling: F = C2 · (F_raw(X) − C1)."""
    return c2 * (F_raw(X) - c1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. RESPONSE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_response(rng: np.random.Generator,
                  m: np.ndarray,
                  task: str,
                  noise_std: float = 0.5) -> np.ndarray:
    """
    Generate the response variable from the linear predictor m.

    Two modes:

      binary     — probit classification
        y ~ Bernoulli(Φ(m))
        Returns integer array of 0s and 1s.

      regression — log-normal positive response
        y = exp(m + ε),  ε ~ N(0, noise_std²)
        Equivalently: log(y) ~ N(m, noise_std²)
        Returns positive float array.
        The log-normal is a natural choice for positive outcomes (claim costs,
        precipitation amounts, etc.) and keeps m as the conditional mean of
        log(y), making the two tasks directly comparable.

    Args:
        rng:       random generator for reproducibility.
        m:         (n,) array of linear predictors.
        task:      "binary" or "regression".
        noise_std: standard deviation of the log-normal noise (regression only).

    Returns: (n,) response array.
    """
    if task == "binary":
        probs = norm.cdf(m)
        return rng.binomial(1, probs).astype(int)

    elif task == "regression":
        # log(y) = m + ε,  ε ~ N(0, noise_std²)  →  y = exp(m + ε) > 0 always
        epsilon = rng.normal(0, noise_std, size=len(m))
        return np.exp(m + epsilon)

    else:
        raise ValueError(f"Unknown task '{task}'. Choose 'binary' or 'regression'.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ASSEMBLE DATAFRAME
# ─────────────────────────────────────────────────────────────────────────────

def build_df(rng: np.random.Generator,
             locs: np.ndarray,
             X: np.ndarray,
             F_eff: np.ndarray,
             b: np.ndarray,
             task: str,
             noise_std: float) -> pd.DataFrame:
    """
    Combine locations, covariates, effects and response into a single DataFrame.

    F_effect, spatial_effect and m are stored explicitly so that model estimates
    can be compared directly to the true decomposition (e.g. via Pearson r or RMSE
    per component).
    """
    m = F_eff + b
    y = make_response(rng, m, task, noise_std)

    df = pd.DataFrame(locs, columns=["s1", "s2"])
    for j in range(X.shape[1]):
        df[f"x{j+1}"] = X[:, j]
    df["F_effect"]       = F_eff
    df["spatial_effect"] = b
    df["m"]              = m
    df["y"]              = y
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_locations(train_locs: np.ndarray,
                   interp_locs: np.ndarray,
                   extrap_locs: np.ndarray,
                   out_path: Path) -> None:
    """
    Scatter plot of training and test locations.
    The extrapolation quadrant is shaded to make the spatial split visible.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(train_locs[:, 0],  train_locs[:, 1],
               s=8, alpha=0.5, color="#1f77b4", label="Train")
    ax.scatter(interp_locs[:, 0], interp_locs[:, 1],
               s=8, alpha=0.5, color="#ff7f0e", marker="^", label="Test (interp)")
    ax.scatter(extrap_locs[:, 0], extrap_locs[:, 1],
               s=8, alpha=0.5, color="#2ca02c", marker="D", label="Test (extrap)")

    ax.fill_between([0.5, 1.0], 0.5, 1.0,
                    alpha=0.08, color="#2ca02c", label="Extrap region")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("s₁"); ax.set_ylabel("s₂")
    ax.set_title("Spatial train/test split\n")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → location plot saved to {out_path}")


def plot_effect(locs: np.ndarray,
                values: np.ndarray,
                title: str,
                cbar_label: str,
                out_path: Path,
                cmap: str = "RdBu_r") -> None:
    """
    Scatter plot of training locations coloured by a continuous effect value.

    Used to visualise either the covariate effect F(X) or the spatial effect
    b(s) on the training set, so the user can visually verify that the two
    components look structurally different (one is driven by covariates, the
    other by location).

    The colormap is centred on zero (vmin = -|max|, vmax = +|max|) so that
    positive and negative values are always symmetrically represented.

    Args:
        locs:       (n, 2) array of spatial coordinates.
        values:     (n,) array of effect values to colour by.
        title:      plot title.
        cbar_label: label for the colourbar.
        out_path:   file path for the saved figure.
        cmap:       matplotlib colourmap (default RdBu_r: blue=negative, red=positive).
    """
    # centre the colormap symmetrically around zero
    abs_max = np.abs(values).max()

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(
        locs[:, 0], locs[:, 1],
        c=values, cmap=cmap,
        vmin=-abs_max, vmax=abs_max,
        s=12, alpha=0.85
    )
    plt.colorbar(sc, ax=ax, label=cbar_label)

    # shade the extrapolation quadrant for reference
    ax.fill_between([0.5, 1.0], 0.5, 1.0,
                    alpha=0.06, color="gray", label="Extrap region (no train data)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("s₁"); ax.set_ylabel("s₂")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → effect plot saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame, name: str, task: str) -> str:
    """Return a formatted string of key statistics for one split."""
    lines = [
        f"\n{'═'*50}",
        f"  {name}  (n={len(df)})",
        f"{'─'*50}",
        f"  F_effect   mean  : {df['F_effect'].mean():.4f}",
        f"  F_effect   std   : {df['F_effect'].std():.4f}",
        f"  spatial    mean  : {df['spatial_effect'].mean():.4f}",
        f"  spatial    std   : {df['spatial_effect'].std():.4f}",
        f"  m          mean  : {df['m'].mean():.4f}",
        f"  m          std   : {df['m'].std():.4f}",
    ]
    if task == "binary":
        lines.append(f"  y=1 frequency    : {df['y'].mean():.4f}")
    else:
        lines += [
            f"  y (log-normal) mean   : {df['y'].mean():.4f}",
            f"  y (log-normal) median : {df['y'].median():.4f}",
            f"  log(y)         mean   : {np.log(df['y']).mean():.4f}",
            f"  log(y)         std    : {np.log(df['y']).std():.4f}",
        ]
    lines.append(f"{'═'*50}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 50)
    print("  Spatial Simulation")
    print(f"  Task: {args.task}")
    print("=" * 50)

    # Step 1 — calibrate F
    print(f"\n[1] Calibrating F(x) on {args.n_calib:,} draws ...")
    c1, c2 = calibrate_F(rng, args.n_calib)
    print(f"    C1 (centering) = {c1:.4f}")
    print(f"    C2 (scaling)   = {c2:.4f}")

    # Step 2 — sample locations
    print("\n[2] Sampling spatial locations ...")
    train_locs  = sample_lower_left(rng, args.n_train)
    interp_locs = sample_lower_left(rng, args.n_test)
    extrap_locs = sample_top_right( rng, args.n_test)
    print(f"    Train  : {len(train_locs)} pts in [0,1]² \\ [0.5,1]²")
    print(f"    Interp : {len(interp_locs)} pts in [0,1]² \\ [0.5,1]²")
    print(f"    Extrap : {len(extrap_locs)} pts in [0.5,1]²")

    # Step 3 — draw GP
    print(f"\n[3] Drawing GP realisation (σ²={args.sigma2}, ρ={args.rho}) ...")
    b_train, b_interp, b_extrap = draw_gp_joint(
        rng, train_locs, interp_locs, extrap_locs, args.sigma2, args.rho
    )
    print(f"    b_train — mean: {b_train.mean():.3f}, std: {b_train.std():.3f}")

    # Step 4 — draw covariates
    print("\n[4] Drawing covariates X ~ N(0, I₉) ...")
    X_train  = rng.standard_normal((args.n_train, 9))
    X_interp = rng.standard_normal((args.n_test,  9))
    X_extrap = rng.standard_normal((args.n_test,  9))

    # Step 5 — compute F(X)
    print("\n[5] Computing F(X) ...")
    F_train  = F_scaled(X_train,  c1, c2)
    F_interp = F_scaled(X_interp, c1, c2)
    F_extrap = F_scaled(X_extrap, c1, c2)
    print(f"    F_train — mean: {F_train.mean():.3f}, std: {F_train.std():.3f}")

    # Step 6 — assemble DataFrames
    if args.task == "regression":
        print(f"\n[6] Generating y = exp(m + ε), ε ~ N(0, {args.noise_std}²) ...")
    else:
        print("\n[6] Generating y ~ Bernoulli(Φ(m)) ...")

    train_df  = build_df(rng, train_locs,  X_train,  F_train,  b_train,
                         args.task, args.noise_std)
    interp_df = build_df(rng, interp_locs, X_interp, F_interp, b_interp,
                         args.task, args.noise_std)
    extrap_df = build_df(rng, extrap_locs, X_extrap, F_extrap, b_extrap,
                         args.task, args.noise_std)

    # Step 7 — save CSVs
    print("\n[7] Saving CSVs ...")
    for df, name in [(train_df,  "train_df.csv"),
                     (interp_df, "test_interp_df.csv"),
                     (extrap_df, "test_extrap_df.csv")]:
        path = out_dir / name
        df.to_csv(path, index=False)
        print(f"    → {path}")

    # Step 8 — plots
    print("\n[8] Generating diagnostic plots ...")
    plot_locations(
        train_locs, interp_locs, extrap_locs,
        out_dir / "plot_locations.png"
    )
    plot_effect(
        train_locs, F_train,
        title="Covariate effect F(X) — training set",
        cbar_label="F(X)",
        out_path=out_dir / "plot_covariate_effect.png",
    )
    plot_effect(
        train_locs, b_train,
        title="Spatial effect b(s) — training set\n(GP, exponential kernel)",
        cbar_label="b(s)",
        out_path=out_dir / "plot_spatial_effect.png",
        cmap="RdYlBu_r",
    )

    # Step 9 — summary
    noise_line = (f"  log-normal noise: noise_std={args.noise_std}\n"
                  if args.task == "regression" else "")
    summary = (
        f"\nSimulation parameters\n"
        f"  task={args.task}, seed={args.seed}\n"
        f"  n_train={args.n_train}, n_test={args.n_test}\n"
        f"  GP: σ²={args.sigma2}, ρ={args.rho}\n"
        f"{noise_line}"
        f"  F calibration: C1={c1:.6f}, C2={c2:.6f}\n"
        + summary_stats(train_df,  "TRAIN",               args.task)
        + summary_stats(interp_df, "TEST — interpolation", args.task)
        + summary_stats(extrap_df, "TEST — extrapolation", args.task)
        + "\n\nColumn descriptions\n"
          "  s1, s2         : spatial coordinates\n"
          "  x1 … x9       : covariates (x1–x3 active, x4–x9 noise)\n"
          "  F_effect       : scaled covariate function, Var ≈ 1\n"
          "  spatial_effect : GP realisation b(s), Var ≈ 1\n"
          "  m              : F_effect + spatial_effect\n"
        + ("  y              : Bernoulli(Φ(m))  — 0 or 1\n"
           if args.task == "binary"
           else "  y              : exp(m + ε), ε ~ N(0, noise_std²)  — positive float\n")
    )
    print(summary)
    (out_dir / "simulation_summary.txt").write_text(summary, encoding="utf-8")

    # Step 10 — sanity checks
    print("\n[9] Sanity checks ...")
    assert np.all((train_locs[:, 0] < 0.5) | (train_locs[:, 1] < 0.5)), \
        "Training point found in extrapolation quadrant!"
    assert np.all((extrap_locs[:, 0] >= 0.5) & (extrap_locs[:, 1] >= 0.5)), \
        "Extrapolation point outside [0.5,1]²!"
    if args.task == "regression":
        assert np.all(train_df["y"] > 0), "Non-positive value in regression response!"
    print(f"    F_effect  Var ≈ {train_df['F_effect'].var():.3f}  (target ≈ 1.0)")
    print(f"    spatial   Var ≈ {train_df['spatial_effect'].var():.3f}  (target ≈ 1.0)")
    print("    All checks passed.")

    print(f"\nDone. Output written to {out_dir}/")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spatial simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task",      type=str,   default=DEFAULTS["task"],
                        choices=["binary", "regression"],
                        help="Response type: probit binary or log-normal regression.")
    parser.add_argument("--seed",      type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--n_train",   type=int,   default=DEFAULTS["n_train"])
    parser.add_argument("--n_test",    type=int,   default=DEFAULTS["n_test"])
    parser.add_argument("--sigma2",    type=float, default=DEFAULTS["sigma2"],
                        help="GP marginal variance σ².")
    parser.add_argument("--rho",       type=float, default=DEFAULTS["rho"],
                        help="GP range parameter ρ. Smaller = rougher field.")
    parser.add_argument("--noise_std", type=float, default=DEFAULTS["noise_std"],
                        help="Std of log-normal noise (regression task only).")
    parser.add_argument("--n_calib",   type=int,   default=DEFAULTS["n_calib"],
                        help="Sample size used to calibrate F to mean=0, Var=1.")
    parser.add_argument("--out",       type=str,   default=DEFAULTS["out"],
                        help="Output folder (created if it does not exist).")
    main(parser.parse_args())
