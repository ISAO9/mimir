"""
mimir.baselines.eikonal_tomography
==================================

Classical iterative travel-time tomography (Aster, Borchers, Thurber 2018,
chapters 6 and 10):

    repeat:
        1. Solve the forward problem from each source via FMM (eikonal)
        2. Predict travel times at each receiver
        3. Form residuals d_T = t_obs - t_pred
        4. Reconstruct curved rays by back-tracing along -∇T from each receiver
        5. Build the sparse Jacobian G where G[i, j] = path length of ray i
           through cell j
        6. Solve the regularized linear system

               min  ||G ds - d_T||^2  +  lambda_d^2 ||ds||^2
                                         +  lambda_s^2 ||L ds||^2

           via LSMR (Fong & Saunders 2011), where L is a 2D Laplacian
           smoothness operator
        7. Update slowness:  s ← s + alpha * ds, with v = 1/s clipped to a
           plausible range
    until residual converges or max_iter

This is the textbook standard against which MIMIR must be compared. We
deliberately do NOT use straight rays (those are weaker than curved rays
and would be an unfair handicap) and we DO regularize (un-regularized
tomography is universally known to fail on ill-posed problems).

Implementation notes
--------------------
* FMM forward uses the same skfmm engine as the synthetic data generator,
  so there is no forward-model mismatch artificially helping or hurting.
* Ray back-tracing is vectorized over rays sharing the same source
  (≈ 12 rays per source in our default geometry), which keeps the cost
  per outer iteration to seconds even on CPU.
* Jacobian construction uses dense ray sub-sampling (1024 points per ray)
  binned into the velocity grid via scipy.sparse COO with duplicate
  summation — accurate and fast.
* The 2D Laplacian uses a 5-point stencil with Neumann boundary conditions
  (∇v · n = 0 at the edges), the standard choice for closed-domain
  tomography.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import skfmm
from skimage.metrics import structural_similarity as ski_ssim

from mimir.data.benchmarks import BenchmarkSpec


# ---------------------------------------------------------------------------
# config + state
# ---------------------------------------------------------------------------


@dataclass
class EikonalTomographyConfig:
    """Hyperparameters for ClassicalEikonalTomography."""

    base_velocity: float = 3.0
    velocity_min: float = 2.0
    velocity_max: float = 5.5

    max_iter: int = 30
    step_size: float = 0.5             # damped Newton step (1.0 = full Gauss-Newton)
    rmse_tol: float = 1e-5             # early stop if data RMSE drops below this
    plateau_iters: int = 5             # early stop if val RMSE plateaus this long

    damping: float = 1e-3              # Tikhonov on slowness perturbation ||ds||
    smoothing: float = 1.0             # 2D-Laplacian smoothness on ds

    ray_max_steps: int = 4000
    ray_step_frac: float = 0.4         # step size as fraction of min(dx, dz)
    n_path_samples: int = 1024         # samples per ray for Jacobian construction
    lsmr_atol: float = 1e-6
    lsmr_btol: float = 1e-6
    lsmr_maxiter: int = 400

    log_every: int = 1


@dataclass
class EikonalTomographyState:
    """History container."""

    iter: int = 0
    best_val_rmse: float = math.inf
    best_iter: int = -1
    best_velocity: Optional[np.ndarray] = None
    history: dict = field(default_factory=lambda: {
        "iter": [], "data_rmse": [], "val_rmse": [], "val_ssim": [], "val_pearson": [],
        "ds_norm": [], "step_actual": [],
    })


# ---------------------------------------------------------------------------
# helpers — coordinate <-> grid index
# ---------------------------------------------------------------------------


def _world_to_grid_xy(coord: np.ndarray, spec: BenchmarkSpec) -> tuple[int, int]:
    x, z = float(coord[0]), float(coord[1])
    col = int(round((x - spec.domain_x[0]) / (spec.domain_x[1] - spec.domain_x[0]) * (spec.nx - 1)))
    row = int(round((z - spec.domain_z[0]) / (spec.domain_z[1] - spec.domain_z[0]) * (spec.nz - 1)))
    col = int(np.clip(col, 0, spec.nx - 1))
    row = int(np.clip(row, 0, spec.nz - 1))
    return col, row


def _world_to_grid_batch(xy: np.ndarray, spec: BenchmarkSpec) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized world-to-grid for an (N, 2) array. Returns (cols, rows)."""
    x = xy[:, 0]
    z = xy[:, 1]
    col = ((x - spec.domain_x[0]) / (spec.domain_x[1] - spec.domain_x[0]) * (spec.nx - 1)).round().astype(int)
    row = ((z - spec.domain_z[0]) / (spec.domain_z[1] - spec.domain_z[0]) * (spec.nz - 1)).round().astype(int)
    col = np.clip(col, 0, spec.nx - 1)
    row = np.clip(row, 0, spec.nz - 1)
    return col, row


def _bilinear_interp(field: np.ndarray, x: np.ndarray, z: np.ndarray, spec: BenchmarkSpec) -> np.ndarray:
    """Vectorized bilinear interpolation of `field` at world coordinates (x, z)."""
    nx, nz = spec.nx, spec.nz
    fx = (x - spec.domain_x[0]) / (spec.domain_x[1] - spec.domain_x[0]) * (nx - 1)
    fz = (z - spec.domain_z[0]) / (spec.domain_z[1] - spec.domain_z[0]) * (nz - 1)
    fx = np.clip(fx, 0.0, nx - 1.0)
    fz = np.clip(fz, 0.0, nz - 1.0)

    j0 = np.floor(fx).astype(int)
    j1 = np.clip(j0 + 1, 0, nx - 1)
    i0 = np.floor(fz).astype(int)
    i1 = np.clip(i0 + 1, 0, nz - 1)

    tx = fx - j0
    tz = fz - i0

    f00 = field[i0, j0]
    f01 = field[i0, j1]
    f10 = field[i1, j0]
    f11 = field[i1, j1]

    return (
        (1 - tz) * ((1 - tx) * f00 + tx * f01)
        + tz * ((1 - tx) * f10 + tx * f11)
    )


# ---------------------------------------------------------------------------
# forward solver: FMM
# ---------------------------------------------------------------------------


def _fmm_travel_time_field(
    velocity_grid: np.ndarray, source_xy: np.ndarray, spec: BenchmarkSpec,
) -> np.ndarray:
    """Solve the eikonal equation |∇T| = 1/v from a single source by FMM."""
    phi = np.ones_like(velocity_grid, dtype=np.float64)
    col, row = _world_to_grid_xy(source_xy, spec)
    phi[row, col] = -1.0

    dx = (spec.domain_x[1] - spec.domain_x[0]) / (spec.nx - 1)
    dz = (spec.domain_z[1] - spec.domain_z[0]) / (spec.nz - 1)
    if not np.isclose(dx, dz):
        raise ValueError(f"FMM requires square cells; got dx={dx}, dz={dz}.")

    return skfmm.travel_time(phi, velocity_grid.astype(np.float64), dx=dx)


# ---------------------------------------------------------------------------
# back-tracing curved rays from -∇T
# ---------------------------------------------------------------------------


def _back_trace_rays_batched(
    tt_field: np.ndarray,
    source_xy: np.ndarray,
    receivers_xy: np.ndarray,
    spec: BenchmarkSpec,
    max_steps: int,
    step_frac: float,
) -> list[np.ndarray]:
    """
    Back-trace several rays from `receivers_xy` toward `source_xy` simultaneously
    by stepping in the -∇T direction.

    Returns a list (length = n_receivers) of (n_path_pts, 2) arrays containing
    the ray paths in physical units, ordered from receiver to source.
    """
    dx = (spec.domain_x[1] - spec.domain_x[0]) / (spec.nx - 1)
    dz = (spec.domain_z[1] - spec.domain_z[0]) / (spec.nz - 1)

    # Pre-compute travel-time gradient field
    Tz, Tx = np.gradient(tt_field, dz, dx)

    n_rays = len(receivers_xy)
    step = step_frac * min(dx, dz)
    src = np.asarray(source_xy, dtype=np.float64)
    cur = np.asarray(receivers_xy, dtype=np.float64).copy()

    # Buffers — list of (n_rays, 2) one entry per step
    trail = [cur.copy()]
    active = np.ones(n_rays, dtype=bool)
    stop_radius = 1.5 * max(dx, dz)

    for _ in range(max_steps):
        if not active.any():
            break

        # Distance to source — freeze rays close enough
        dist_to_src = np.linalg.norm(cur - src, axis=1)
        active &= dist_to_src > stop_radius
        if not active.any():
            break

        # Vectorized gradient sampling at all current points
        gx = _bilinear_interp(Tx, cur[:, 0], cur[:, 1], spec)
        gz = _bilinear_interp(Tz, cur[:, 0], cur[:, 1], spec)
        gnorm = np.sqrt(gx * gx + gz * gz)

        # Freeze rays where the gradient vanishes (numerical)
        active &= gnorm > 1e-9
        if not active.any():
            break

        # Step in -∇T direction (toward source)
        nxt = cur.copy()
        nxt[active, 0] -= step * gx[active] / gnorm[active]
        nxt[active, 1] -= step * gz[active] / gnorm[active]

        # Keep within domain
        nxt[:, 0] = np.clip(nxt[:, 0], spec.domain_x[0], spec.domain_x[1])
        nxt[:, 1] = np.clip(nxt[:, 1], spec.domain_z[0], spec.domain_z[1])

        # Frozen rays stay where they were
        nxt[~active] = cur[~active]
        cur = nxt
        trail.append(cur.copy())

    # Append source as final point of every ray (clean termination)
    final = np.tile(src, (n_rays, 1))
    trail.append(final)

    # Re-format: trail is a list of (n_rays, 2). Per-ray we want (n_pts, 2).
    trail_arr = np.stack(trail, axis=1)   # (n_rays, n_pts, 2)
    return [trail_arr[i] for i in range(n_rays)]


# ---------------------------------------------------------------------------
# Jacobian builder
# ---------------------------------------------------------------------------


def _build_jacobian(
    ray_paths: list[np.ndarray], spec: BenchmarkSpec, n_samples: int,
) -> sp.csr_matrix:
    """
    Build sparse G of shape (n_rays, nz*nx) where G[i, j] is the path length
    of ray i through cell j (flat index = row * nx + col).

    Method: arc-length-uniform sampling along each ray with `n_samples` points,
    each contributing `total_length / n_samples` to the cell containing it.
    Duplicate cells along the ray are summed (sum_duplicates on COO).
    """
    n_rays = len(ray_paths)
    n_cells = spec.nz * spec.nx
    rows_all: list[np.ndarray] = []
    cols_all: list[np.ndarray] = []
    data_all: list[np.ndarray] = []

    for ray_idx, ray in enumerate(ray_paths):
        if ray.shape[0] < 2:
            continue
        diffs = np.diff(ray, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        total_len = float(seg_len.sum())
        if total_len < 1e-9:
            continue
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        s = np.linspace(0.0, total_len, n_samples)
        x_samp = np.interp(s, cum, ray[:, 0])
        z_samp = np.interp(s, cum, ray[:, 1])
        col, row = _world_to_grid_batch(np.column_stack([x_samp, z_samp]), spec)
        flat = row * spec.nx + col
        per_sample = total_len / n_samples
        rows_all.append(np.full(flat.size, ray_idx, dtype=np.int64))
        cols_all.append(flat.astype(np.int64))
        data_all.append(np.full(flat.size, per_sample, dtype=np.float64))

    if not rows_all:
        return sp.csr_matrix((n_rays, n_cells), dtype=np.float64)

    rows_arr = np.concatenate(rows_all)
    cols_arr = np.concatenate(cols_all)
    data_arr = np.concatenate(data_all)
    G = sp.coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(n_rays, n_cells))
    G.sum_duplicates()
    return G.tocsr()


# ---------------------------------------------------------------------------
# 2D Laplacian (5-point stencil, Neumann BC) — slowness-side regularizer
# ---------------------------------------------------------------------------


def _build_2d_laplacian(spec: BenchmarkSpec) -> sp.csr_matrix:
    """
    Sparse 2D Laplacian operator for an nz-by-nx grid. Neumann (zero-flux)
    boundary conditions: edge cells couple only to their interior neighbors.

    The returned matrix `L` has shape (n_cells, n_cells) where applying `L @ s`
    (with s flattened row-major) returns the 5-point Laplacian of s.
    """
    nz, nx = spec.nz, spec.nx
    n = nz * nx
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(nz):
        for j in range(nx):
            idx = i * nx + j
            count = 0
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < nz and 0 <= nj < nx:
                    rows.append(idx)
                    cols.append(ni * nx + nj)
                    data.append(-1.0)
                    count += 1
            rows.append(idx)
            cols.append(idx)
            data.append(float(count))
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


# ---------------------------------------------------------------------------
# regularized least-squares solve
# ---------------------------------------------------------------------------


def _solve_regularized_lsmr(
    G: sp.csr_matrix,
    d_T: np.ndarray,
    L: sp.csr_matrix,
    damping: float,
    smoothing: float,
    atol: float, btol: float, maxiter: int,
) -> np.ndarray:
    """
    Solve  argmin_ds  ||G ds - dT||^2 + damping^2 ||ds||^2 + smoothing^2 ||L ds||^2

    Stack the equations:
        [        G        ]       [ dT ]
        [ damping  * I    ] ds  ≈ [ 0  ]
        [ smoothing * L   ]       [ 0  ]
    and solve with LSMR.
    """
    n = G.shape[1]
    blocks = [G]
    rhs_blocks = [d_T]
    if damping > 0:
        blocks.append(damping * sp.eye(n, format="csr"))
        rhs_blocks.append(np.zeros(n))
    if smoothing > 0:
        blocks.append(smoothing * L)
        rhs_blocks.append(np.zeros(L.shape[0]))
    A = sp.vstack(blocks).tocsr()
    b = np.concatenate(rhs_blocks)
    result = spla.lsmr(A, b, atol=atol, btol=btol, maxiter=maxiter)
    return result[0]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    rng = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if rng <= 0:
        return float("nan")
    return float(ski_ssim(a, b, data_range=rng))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    av = a.flatten() - a.mean()
    bv = b.flatten() - b.mean()
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 0:
        return float("nan")
    return float((av * bv).sum() / denom)


# ---------------------------------------------------------------------------
# the algorithm
# ---------------------------------------------------------------------------


class ClassicalEikonalTomography:
    """
    Iterative regularized travel-time tomography using FMM forward, curved
    ray back-tracing, and LSMR-based regularized least-squares updates.

    See module docstring for algorithm overview.
    """

    def __init__(self, spec: BenchmarkSpec, cfg: EikonalTomographyConfig | None = None) -> None:
        self.spec = spec
        self.cfg = cfg or EikonalTomographyConfig()
        # Pre-compute Laplacian once (it depends only on the grid)
        self._L = _build_2d_laplacian(spec)

    def fit(
        self,
        sources: np.ndarray,        # (R, 2)
        receivers: np.ndarray,      # (R, 2)
        observed_tt: np.ndarray,    # (R,)
        truth_grid: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> tuple[np.ndarray, EikonalTomographyState]:
        """
        Run the inversion. Returns the *best* (lowest val-RMSE) reconstructed
        velocity grid and the full training state.

        If `truth_grid` is None, validation metrics will not be computed and
        the method will return the final-iteration velocity instead of a
        best-by-validation pick.
        """
        cfg = self.cfg
        spec = self.spec
        nz, nx = spec.nz, spec.nx
        n_rays = len(observed_tt)

        # Initialize: uniform velocity at base
        v = np.full((nz, nx), cfg.base_velocity, dtype=np.float64)

        # Pre-group rays by source so we run FMM once per unique source
        ray_diffs = np.any(np.diff(sources, axis=0) != 0, axis=1)
        starts = np.concatenate([[0], np.where(ray_diffs)[0] + 1])
        ends = np.concatenate([starts[1:], [n_rays]])
        unique_sources = sources[starts]
        n_unique_sources = unique_sources.shape[0]

        state = EikonalTomographyState()

        plateau_count = 0
        prev_val_rmse = math.inf

        for it in range(cfg.max_iter):
            # --- 1. Forward + ray back-tracing ---
            tt_pred_flat = np.empty(n_rays, dtype=np.float64)
            ray_paths: list[np.ndarray] = [None] * n_rays  # type: ignore[list-item]
            for s_idx in range(n_unique_sources):
                src = unique_sources[s_idx]
                lo, hi = starts[s_idx], ends[s_idx]
                recs = receivers[lo:hi]

                tt_field = _fmm_travel_time_field(v, src, spec)

                # predicted tt at each receiver (simple bilinear lookup)
                tt_pred = _bilinear_interp(tt_field, recs[:, 0], recs[:, 1], spec)
                tt_pred_flat[lo:hi] = tt_pred

                # back-trace this batch of rays
                paths = _back_trace_rays_batched(
                    tt_field, src, recs, spec,
                    max_steps=cfg.ray_max_steps,
                    step_frac=cfg.ray_step_frac,
                )
                for k, p in enumerate(paths):
                    ray_paths[lo + k] = p

            # --- 2. Residuals + Jacobian ---
            d_T = observed_tt.astype(np.float64) - tt_pred_flat
            data_rmse = float(np.sqrt(np.mean(d_T ** 2)))

            G = _build_jacobian(ray_paths, spec, n_samples=cfg.n_path_samples)

            # --- 3. Regularized solve ---
            ds_flat = _solve_regularized_lsmr(
                G, d_T, self._L, cfg.damping, cfg.smoothing,
                atol=cfg.lsmr_atol, btol=cfg.lsmr_btol, maxiter=cfg.lsmr_maxiter,
            )
            ds = ds_flat.reshape(nz, nx)

            # --- 4. Update slowness with bounded step ---
            s_curr = 1.0 / v
            s_new = s_curr + cfg.step_size * ds
            # Enforce velocity bounds via slowness clipping
            s_min = 1.0 / cfg.velocity_max
            s_max = 1.0 / cfg.velocity_min
            s_new = np.clip(s_new, s_min, s_max)
            v = 1.0 / s_new

            # --- 5. Validation metrics ---
            if truth_grid is not None:
                val_rmse = float(np.sqrt(np.mean((v - truth_grid) ** 2)))
                val_ssim = _ssim(truth_grid, v)
                val_pear = _pearson(truth_grid, v)
            else:
                val_rmse = float("nan")
                val_ssim = float("nan")
                val_pear = float("nan")

            # --- 6. Log + best-checkpoint ---
            state.history["iter"].append(it)
            state.history["data_rmse"].append(data_rmse)
            state.history["val_rmse"].append(val_rmse)
            state.history["val_ssim"].append(val_ssim)
            state.history["val_pearson"].append(val_pear)
            state.history["ds_norm"].append(float(np.linalg.norm(ds_flat)))
            state.history["step_actual"].append(cfg.step_size)
            state.iter = it + 1

            if truth_grid is not None and val_rmse < state.best_val_rmse:
                state.best_val_rmse = val_rmse
                state.best_iter = it
                state.best_velocity = v.copy()

            if verbose and (it % cfg.log_every == 0 or it == cfg.max_iter - 1):
                print(f"  [classical] iter={it:3d}  "
                      f"data_rmse={data_rmse:.4f}  val_rmse={val_rmse:.4f}  "
                      f"val_ssim={val_ssim:.3f}  ||ds||={state.history['ds_norm'][-1]:.3f}")

            # --- 7. Convergence checks ---
            if data_rmse < cfg.rmse_tol:
                if verbose:
                    print(f"  [classical] data RMSE {data_rmse:.2e} < tol {cfg.rmse_tol:.2e}; stop")
                break
            if truth_grid is not None:
                if val_rmse >= prev_val_rmse - 1e-5:
                    plateau_count += 1
                else:
                    plateau_count = 0
                if plateau_count >= cfg.plateau_iters:
                    if verbose:
                        print(f"  [classical] val RMSE plateaued for {cfg.plateau_iters} iters; stop")
                    break
                prev_val_rmse = val_rmse

        # Return best (or final if no truth provided)
        if state.best_velocity is not None:
            return state.best_velocity, state
        return v, state
