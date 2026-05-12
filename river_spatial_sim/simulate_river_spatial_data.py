"""
simulate_river_spatial_data.py
─────────────────────────────────────────────────────────────────────────────
Spatial simulation with a deterministic river-corridor spatial effect.

What this script does
─────────────────────
Generates a synthetic dataset where the response depends on a non-linear
covariate function F(X) and a deterministic spatial effect b(s) that
concentrates risk along a river corridor crossing the unit square.

Unlike the GP-based simulation (simulate_gp_spatial_data.py), the spatial
field here is not random: it is fully determined by the geometry of the
river. This tests whether a spatial model can recover a sharp, discontinuous
boundary — the corridor edge where b(s) drops abruptly to zero.

Two response types are supported via --task:

  binary     (default)
    y ~ Bernoulli(Φ(m))           probit link
    Use for: classification benchmarks, spatial logistic regression.

  regression
    y = exp(m + ε),  ε ~ N(0, noise_std²)    log-normal response
    Use for: positive-valued outcomes such as claim amounts or flood losses.

River corridor spatial effect
──────────────────────────────
  A straight river runs from (0, 0.35) to (1, 0.65) across the unit square.
  For each point s = (s1, s2), let d = perpendicular distance to the river:

    b(s) = A · (1 − d/D)^alpha    if d ≤ D   (inside corridor)
    b(s) = 0                       if d > D   (outside corridor)

  with A = 1.5 (amplitude), D = 0.18 (corridor half-width), alpha = 0.3
  (very slow decay — the effect stays high across most of the corridor).

  The hard zero at d = D is the key modelling challenge: SPDE/GP smoothers
  tend to bleed the effect beyond the true boundary.

Covariate function
──────────────────
  F_raw(x) = 1.5x₁ + x₂² + 3·1{x₃>0} + x₄·x₅

  8 covariates total: x₁–x₅ are active, x₆–x₈ are pure noise.
  F is centred and scaled to variance ≈ 1 before combining with b(s).

Response model
──────────────
  m(i) = F(X_i) + b(s_i)
  y_i  ~ Bernoulli(Φ(m_i))     [binary]
  y_i  = exp(m_i + εᵢ)         [regression, εᵢ ~ N(0, noise_std²)]

Output columns (in each CSV)
─────────────────────────────
  s1, s2           : spatial coordinates in [0,1]²
  x1 … x8         : covariates (x1–x5 active, x6–x8 noise)
  F_effect         : scaled covariate function F(X), variance ≈ 1
  spatial_effect   : river corridor effect b(s), ≥ 0
  m                : linear predictor  m = F_effect + spatial_effect
  y                : response (see --task)

Usage
─────
  python simulate_river_spatial_data.py
  python simulate_river_spatial_data.py --task regression
  python simulate_river_spatial_data.py --amplitude 2.0 --d_cutoff 0.25
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
    task       = "regression",  # "binary" or "regression"
    seed       = 42,
    n_train    = 500,
    n_test     = 500,
    amplitude  = 1.5,      # river effect amplitude A
    d_cutoff   = 0.18,     # corridor half-width D
    alpha      = 0.3,      # decay exponent (small = slow decay inside corridor)
    noise_std  = 0.5,      # log-normal noise std (regression only)
    n_calib    = 50_000,
    out        = "simulation_output",
)

# River geometry: fixed entry and exit points
RIVER_P1 = np.array([0.0, 0.35])
RIVER_P2 = np.array([1.0, 0.65])


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOCATION GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def sample_lower_left(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Sample n points uniformly in [0,1]² excluding the top-right quadrant.

    The top-right quadrant [0.5,1]² is reserved for extrapolation testing.
    Uses rejection sampling: ~75% of draws are accepted on average.

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
    Sample n points uniformly in [0.5,1]² (extrapolation region).

    Returns: (n, 2) array of (s1, s2) coordinates.
    """
    return rng.uniform(0.5, 1.0, size=(n, 2))


# ─────────────────────────────────────────────────────────────────────────────
# 2. RIVER CORRIDOR SPATIAL EFFECT
# ─────────────────────────────────────────────────────────────────────────────

def river_distance(locs: np.ndarray) -> np.ndarray:
    """
    Compute the perpendicular distance from each point to the river centre line.

    Uses the 2D cross-product formula:
      d = ||(s − p1) × (p2 − p1)|| / ||p2 − p1||

    Args:
        locs: (n, 2) array of spatial coordinates.

    Returns: (n,) array of non-negative distances.
    """
    line_vec  = RIVER_P2 - RIVER_P1                          # direction vector
    line_len  = np.linalg.norm(line_vec)
    point_vec = locs - RIVER_P1                              # (n, 2)
    # 2D cross product: v1[0]*v2[1] − v1[1]*v2[0]
    cross     = (point_vec[:, 0] * line_vec[1]
                 - point_vec[:, 1] * line_vec[0])
    return np.abs(cross) / line_len


def river_effect(locs: np.ndarray,
                 amplitude: float,
                 d_cutoff: float,
                 alpha: float) -> np.ndarray:
    """
    Deterministic river-corridor spatial effect.

      b(s) = A · (1 − d/D)^alpha    if d ≤ D
      b(s) = 0                       if d > D

    The power alpha controls the decay shape inside the corridor:
      alpha < 1  : concave decay — effect stays high, drops near the edge
      alpha = 1  : linear decay
      alpha > 1  : convex decay — effect drops quickly near the centre line

    With the default alpha = 0.3, the profile is very flat inside the
    corridor, testing whether a model can recover the hard cutoff at d = D
    without over-smoothing it.

    Args:
        locs:      (n, 2) array of spatial coordinates.
        amplitude: maximum effect value A (at the river centre line).
        d_cutoff:  corridor half-width D; points beyond this have b(s) = 0.
        alpha:     decay exponent.

    Returns: (n,) array of non-negative effect values.
    """
    d     = river_distance(locs)
    ratio = np.clip(1.0 - d / d_cutoff, 0.0, None)   # clamp float noise
    return np.where(d <= d_cutoff, amplitude * ratio ** alpha, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. COVARIATE FUNCTION F(X)
# ─────────────────────────────────────────────────────────────────────────────

def F_raw(X: np.ndarray) -> np.ndarray:
    """
    Non-linear covariate function.

      F_raw(x) = 1.5x₁ + x₂² + 3·1{x₃>0} + x₄·x₅

    Only x₁–x₅ are active. Columns x₆–x₈ are noise covariates that should
    be discarded by any well-calibrated model.

    Combines:
      - a linear term          (1.5x₁)
      - a quadratic term       (x₂²)
      - a step function        (3·1{x₃>0})
      - a bilinear interaction (x₄·x₅)

    Args:
        X: (n, 8) array of covariates drawn i.i.d. from N(0, 1).

    Returns: (n,) array of raw covariate effects.
    """
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    x4, x5     = X[:, 3], X[:, 4]
    return (1.5 * x1
            + x2 ** 2
            + 3.0 * (x3 > 0).astype(float)
            + x4 * x5)


def calibrate_F(rng: np.random.Generator, n_calib: int) -> tuple:
    """
    Estimate centering (C1) and scaling (C2) from a large i.i.d. draw.

    Goal: F_scaled = C2 · (F_raw − C1) has mean ≈ 0 and Var ≈ 1,
    so F and b(s) contribute equally to m.

    Returns: (C1, C2) as floats.
    """
    X_cal = rng.standard_normal((n_calib, 8))
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

      binary:     y ~ Bernoulli(Φ(m))          — 0 or 1
      regression: y = exp(m + ε), ε~N(0,σ²)   — positive float

    Args:
        rng:       random generator.
        m:         (n,) linear predictor.
        task:      "binary" or "regression".
        noise_std: log-normal noise std (regression only).

    Returns: (n,) response array.
    """
    if task == "binary":
        return rng.binomial(1, norm.cdf(m)).astype(int)
    elif task == "regression":
        return np.exp(m + rng.normal(0, noise_std, size=len(m)))
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
    Combine locations, covariates, effects and response into a DataFrame.

    F_effect, spatial_effect and m are stored so that model estimates can
    be compared to the true decomposition (e.g. Pearson r per component).
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
                   d_cutoff: float,
                   out_path: Path) -> None:
    """
    Scatter plot of training and test locations with the river corridor overlaid.
    The extrapolation quadrant and corridor boundaries are shown for reference.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(train_locs[:, 0],  train_locs[:, 1],
               s=8, alpha=0.5, color="#1f77b4", label="Train")
    ax.scatter(interp_locs[:, 0], interp_locs[:, 1],
               s=8, alpha=0.5, color="#ff7f0e", marker="^", label="Test (interp)")
    ax.scatter(extrap_locs[:, 0], extrap_locs[:, 1],
               s=8, alpha=0.5, color="#2ca02c", marker="D", label="Test (extrap)")

    ax.fill_between([0.5, 1.0], 0.5, 1.0,
                    alpha=0.06, color="#2ca02c", label="Extrap region")

    # river centre line and corridor boundaries
    ax.plot([RIVER_P1[0], RIVER_P2[0]], [RIVER_P1[1], RIVER_P2[1]],
            color="steelblue", lw=2, label="River centre line")
    line_vec = RIVER_P2 - RIVER_P1
    perp     = np.array([-line_vec[1], line_vec[0]]) / np.linalg.norm(line_vec)
    for sign in [+1, -1]:
        offset = sign * d_cutoff * perp
        ax.plot([RIVER_P1[0] + offset[0], RIVER_P2[0] + offset[0]],
                [RIVER_P1[1] + offset[1], RIVER_P2[1] + offset[1]],
                color="steelblue", lw=1, linestyle="--", alpha=0.6)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("s₁"); ax.set_ylabel("s₂")
    ax.set_title("Spatial train/test split — river corridor")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → location plot saved to {out_path}")


def plot_effect(locs: np.ndarray,
                values: np.ndarray,
                title: str,
                cbar_label: str,
                out_path: Path,
                cmap: str = "RdBu_r",
                centre_zero: bool = True) -> None:
    """
    Scatter plot of training locations coloured by a continuous effect value.

    Used to visualise F(X) or b(s) on the training set so the user can verify
    that the two components look structurally different.

    When centre_zero=True the colormap is centred symmetrically on zero.
    For b(s) in the river case (all non-negative), centre_zero=False is used
    so the full dynamic range of the colormap is used.

    Args:
        locs:        (n, 2) training coordinates.
        values:      (n,) effect values.
        title:       plot title.
        cbar_label:  colourbar label.
        out_path:    output file path.
        cmap:        matplotlib colormap name.
        centre_zero: if True, set vmin/vmax symmetrically around zero.
    """
    if centre_zero:
        abs_max = np.abs(values).max()
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = values.min(), values.max()

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(locs[:, 0], locs[:, 1],
                    c=values, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    s=12, alpha=0.85)
    plt.colorbar(sc, ax=ax, label=cbar_label)

    ax.fill_between([0.5, 1.0], 0.5, 1.0,
                    alpha=0.06, color="gray", label="Extrap region")
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
    """Return formatted summary statistics for one split."""
    pct_corridor = (df["spatial_effect"] > 0).mean()
    lines = [
        f"\n{'═'*52}",
        f"  {name}  (n={len(df)})",
        f"{'─'*52}",
        f"  F_effect   mean  : {df['F_effect'].mean():.4f}",
        f"  F_effect   std   : {df['F_effect'].std():.4f}",
        f"  spatial    mean  : {df['spatial_effect'].mean():.4f}",
        f"  spatial    std   : {df['spatial_effect'].std():.4f}",
        f"  spatial    max   : {df['spatial_effect'].max():.4f}",
        f"  pct inside corridor : {pct_corridor:.4f}",
        f"  m          mean  : {df['m'].mean():.4f}",
        f"  m          std   : {df['m'].std():.4f}",
    ]
    if task == "binary":
        lines.append(f"  y=1 frequency    : {df['y'].mean():.4f}")
    else:
        lines += [
            f"  y (log-normal) mean   : {df['y'].mean():.4f}",
            f"  y (log-normal) median : {df['y'].median():.4f}",
            f"  log(y)         std    : {np.log(df['y']).std():.4f}",
        ]
    lines.append(f"{'═'*52}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 52)
    print("  River Corridor Spatial Simulation")
    print(f"  Task: {args.task}")
    print("=" * 52)

    # Step 1 — calibrate F
    print(f"\n[1] Calibrating F(x) on {args.n_calib:,} draws ...")
    c1, c2 = calibrate_F(rng, args.n_calib)
    print(f"    C1 = {c1:.4f},  C2 = {c2:.4f}")

    # Step 2 — sample locations
    print("\n[2] Sampling spatial locations ...")
    train_locs  = sample_lower_left(rng, args.n_train)
    interp_locs = sample_lower_left(rng, args.n_test)
    extrap_locs = sample_top_right( rng, args.n_test)
    print(f"    Train  : {len(train_locs)} pts in [0,1]² \\ [0.5,1]²")
    print(f"    Interp : {len(interp_locs)} pts")
    print(f"    Extrap : {len(extrap_locs)} pts in [0.5,1]²")

    # Step 3 — river spatial effect (deterministic, no random draw)
    print(f"\n[3] Computing river corridor spatial effect ...")
    print(f"    River: ({RIVER_P1}) → ({RIVER_P2})")
    print(f"    Corridor: D={args.d_cutoff}, alpha={args.alpha}, A={args.amplitude}")
    b_train  = river_effect(train_locs,  args.amplitude, args.d_cutoff, args.alpha)
    b_interp = river_effect(interp_locs, args.amplitude, args.d_cutoff, args.alpha)
    b_extrap = river_effect(extrap_locs, args.amplitude, args.d_cutoff, args.alpha)
    pct_in   = (b_train > 0).mean()
    print(f"    b_train — mean: {b_train.mean():.3f}, std: {b_train.std():.3f}")
    print(f"    Pct of training points inside corridor: {pct_in:.2%}")

    # Step 4 — draw covariates
    print("\n[4] Drawing covariates X ~ N(0, I₈) ...")
    print("    x1–x5 active, x6–x8 noise")
    X_train  = rng.standard_normal((args.n_train, 8))
    X_interp = rng.standard_normal((args.n_test,  8))
    X_extrap = rng.standard_normal((args.n_test,  8))

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
        args.d_cutoff,
        out_dir / "plot_locations.png"
    )
    plot_effect(
        train_locs, F_train,
        title="Covariate effect F(X) — training set",
        cbar_label="F(X)",
        out_path=out_dir / "plot_covariate_effect.png",
        cmap="RdBu_r",
        centre_zero=True,
    )
    plot_effect(
        train_locs, b_train,
        title="Spatial effect b(s) — river corridor",
        cbar_label="b(s)",
        out_path=out_dir / "plot_spatial_effect.png",
        cmap="YlOrRd",
        centre_zero=False,   # b(s) ≥ 0 always — no need to centre on zero
    )

    # Step 9 — summary
    noise_line = (f"  log-normal noise: noise_std={args.noise_std}\n"
                  if args.task == "regression" else "")
    summary = (
        f"\nRiver Corridor Simulation\n"
        f"  task={args.task}, seed={args.seed}\n"
        f"  n_train={args.n_train}, n_test={args.n_test}\n"
        f"  River: ({RIVER_P1}) → ({RIVER_P2})\n"
        f"  Corridor: D={args.d_cutoff}, alpha={args.alpha}, A={args.amplitude}\n"
        f"  F: 1.5x₁ + x₂² + 3·1{{x₃>0}} + x₄·x₅\n"
        f"  Noise covariates: x6, x7, x8\n"
        f"{noise_line}"
        f"  F calibration: C1={c1:.6f}, C2={c2:.6f}\n"
        + summary_stats(train_df,  "TRAIN",               args.task)
        + summary_stats(interp_df, "TEST — interpolation", args.task)
        + summary_stats(extrap_df, "TEST — extrapolation", args.task)
        + "\n\nColumn descriptions\n"
          "  s1, s2         : spatial coordinates\n"
          "  x1 … x8       : covariates (x1–x5 active, x6–x8 noise)\n"
          "  F_effect       : scaled covariate function, Var ≈ 1\n"
          "  spatial_effect : river corridor effect b(s), ≥ 0\n"
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
    assert np.all(b_train >= 0), \
        "Negative spatial effect detected!"
    assert np.all(b_train[river_distance(train_locs) > args.d_cutoff] == 0.0), \
        "Non-zero effect found outside corridor!"
    if args.task == "regression":
        assert np.all(train_df["y"] > 0), "Non-positive value in regression response!"
    print(f"    F_effect  Var ≈ {train_df['F_effect'].var():.3f}  (target ≈ 1.0)")
    print(f"    spatial   max ≈ {b_train.max():.3f}  (should be ≤ {args.amplitude})")
    print("    All checks passed.")

    print(f"\nDone. Output written to {out_dir}/")
    print("=" * 52)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="River corridor spatial simulation for benchmarking spatial models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task",      type=str,   default=DEFAULTS["task"],
                        choices=["binary", "regression"])
    parser.add_argument("--seed",      type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--n_train",   type=int,   default=DEFAULTS["n_train"])
    parser.add_argument("--n_test",    type=int,   default=DEFAULTS["n_test"])
    parser.add_argument("--amplitude", type=float, default=DEFAULTS["amplitude"],
                        help="River effect amplitude A.")
    parser.add_argument("--d_cutoff",  type=float, default=DEFAULTS["d_cutoff"],
                        help="Corridor half-width D.")
    parser.add_argument("--alpha",     type=float, default=DEFAULTS["alpha"],
                        help="Decay exponent inside corridor. <1 = slow decay.")
    parser.add_argument("--noise_std", type=float, default=DEFAULTS["noise_std"],
                        help="Log-normal noise std (regression only).")
    parser.add_argument("--n_calib",   type=int,   default=DEFAULTS["n_calib"])
    parser.add_argument("--out",       type=str,   default=DEFAULTS["out"])
    main(parser.parse_args())
