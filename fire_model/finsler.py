from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

from fire_model.ca import FireEnv, FireState
from fire_model.boundary import (
    FireBoundary,
    extract_fire_boundary,
    between_boundaries_mask,
    plot_fire_boundary,
)

# ============================================================
# Utility functions
# ============================================================

def _unit_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + eps)


def _dx_m_from_env(env: FireEnv) -> float:
    nx, _ = env.grid_size
    return float(env.domain_km) / max(nx, 1) * 1000.0


def _retardant_decay_factor(env: FireEnv, dt_s: float) -> float:
    """Continuous-time retardant decay factor after dt_s seconds."""
    hl = float(getattr(env, "retardant_half_life_s", 0.0))
    if hl <= 0.0:
        return 1.0
    return float(np.exp(-np.log(2.0) * dt_s / hl))


def _wind_at_time(env: FireEnv, t_s: float) -> np.ndarray:
    """
    Wind field at absolute time t_s (seconds).

    Supports:
      - constant wind: (nx, ny, 2)
      - time-varying wind: (T, nx, ny, 2) indexed using env.dt_s
    """
    wind = getattr(env, "wind", None)
    if wind is None:
        return np.zeros((*env.grid_size, 2), dtype=float)

    w = np.asarray(wind, dtype=float)
    if w.ndim == 3:
        return w
    if w.ndim == 4:
        dt_s = float(getattr(env, "dt_s", 1.0))
        if dt_s <= 0:
            raise ValueError("env.dt_s must be > 0 for time-varying wind.")
        idx = int(np.clip(np.floor(t_s / dt_s), 0, w.shape[0] - 1))
        return w[idx]

    raise ValueError("env.wind must have shape (nx,ny,2) or (T,nx,ny,2)")


def _slope_factor(slope_vec: np.ndarray | None, move_dir: np.ndarray, k_slope: float) -> float:
    """Multiplicative slope bias; >1 uphill, <1 downhill depending on k_slope."""
    if slope_vec is None or k_slope == 0.0:
        return 1.0
    slope_mag = float(np.linalg.norm(slope_vec))
    if slope_mag < 1e-12:
        return 1.0
    align = float(np.dot(_unit_vec(slope_vec), _unit_vec(move_dir)))
    return float(np.exp(k_slope * slope_mag * align))


def _directional_rate(
    local_ros: float,
    wind_vec: np.ndarray,
    move_dir: np.ndarray,
    *,
    k_wind: float,
    w_ref: float,
    clamp: tuple[float, float],
    slope_vec: np.ndarray | None = None,
    k_slope: float = 0.0,
) -> float:
    """Directional rate of spread (m/s) at a cell."""
    wmag = float(np.linalg.norm(wind_vec))
    if wmag < 1e-12:
        align = 0.0
    else:
        align = float(np.dot(_unit_vec(wind_vec), _unit_vec(move_dir)))

    wind_factor = np.exp(k_wind * (wmag / (w_ref + 1e-12)) * align)
    slope_bias = _slope_factor(slope_vec, move_dir, k_slope)
    rate = local_ros * wind_factor * slope_bias

    lo, hi = clamp
    return float(np.clip(rate, lo * local_ros, hi * local_ros))


# ============================================================
# Core deterministic solver
# ============================================================

def anisotropic_arrival_times(
    env: FireEnv,
    ignition_mask: np.ndarray,
    *,
    diag: bool = True,
    k_wind: float = 0.25,
    w_ref: float = 5.0,
    clamp: tuple[float, float] = (0.05, 5.0),
    k_slope: float = 0.0,
    retardant0: np.ndarray | None = None,
    start_time_s: float = 0.0,
    retardant_applied_time_s: float = 0.0,
) -> np.ndarray:
    """
    Deterministic anisotropic front propagation via Dijkstra on a 4/8-neighbour grid.

    Key accuracy features:
      - wind evaluated at evolving absolute time (start_time_s + travel_time)
      - retardant decays with time since application (abs_time - retardant_applied_time_s)
    """
    grid_size = env.grid_size

    fuel = getattr(env, "fuel", None)
    if fuel is None:
        fuel = np.ones(grid_size, dtype=float)
    else:
        fuel = np.asarray(fuel, dtype=float)
        if fuel.shape != grid_size:
            raise ValueError(f"env.fuel must have shape {grid_size}, got {fuel.shape}")

    ignition_mask = np.asarray(ignition_mask, dtype=bool)
    if ignition_mask.shape != grid_size:
        raise ValueError(f"ignition_mask must have shape {grid_size}, got {ignition_mask.shape}")

    dx_m = _dx_m_from_env(env)
    base_ros = float(env.ros_mps)
    slope_field = getattr(env, "slope", None)
    if slope_field is not None:
        slope_field = np.asarray(slope_field, dtype=float)
        expected_shape = (*grid_size, 2)
        if slope_field.shape != expected_shape:
            raise ValueError(f"env.slope must have shape {expected_shape}, got {slope_field.shape}")
    k_ret = float(getattr(env, "retardant_k", 1.0))

    if retardant0 is not None:
        r0 = np.asarray(retardant0, dtype=float)
        if r0.shape != grid_size:
            raise ValueError(f"retardant0 must have shape {grid_size}, got {r0.shape}")
    else:
        r0 = None

    moves: list[tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        moves += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    dist = np.full(grid_size, np.inf, dtype=float)
    pq: list[tuple[float, int, int]] = []

    xs, ys = np.where(ignition_mask)
    for x, y in zip(xs, ys):
        dist[int(x), int(y)] = 0.0
        pq.append((0.0, int(x), int(y)))
    heapq.heapify(pq)

    while pq:
        t_curr, x, y = heapq.heappop(pq)
        if t_curr != dist[x, y]:
            continue

        abs_time_s = float(start_time_s) + float(t_curr)

        wind_field = _wind_at_time(env, abs_time_s)
        wind_vec = wind_field[x, y]
        slope_vec = None if slope_field is None else slope_field[x, y]

        if r0 is None:
            attn = 1.0
        else:
            time_since_drop = max(0.0, abs_time_s - float(retardant_applied_time_s))
            r_eff = r0[x, y] * _retardant_decay_factor(env, time_since_drop)
            attn = float(np.exp(-k_ret * max(r_eff, 0.0)))

        local_ros_xy = base_ros * float(fuel[x, y]) * attn

        for dx_i, dy_i in moves:
            xn, yn = x + dx_i, y + dy_i
            if not (0 <= xn < grid_size[0] and 0 <= yn < grid_size[1]):
                continue

            step_vec = np.array([dx_i, dy_i], dtype=float)
            step_len = dx_m * float(np.linalg.norm(step_vec))

            rate = _directional_rate(
                local_ros=local_ros_xy,
                wind_vec=wind_vec,
                move_dir=step_vec,
                k_wind=float(k_wind),
                w_ref=float(w_ref),
                clamp=clamp,
                slope_vec=slope_vec,
                k_slope=float(k_slope),
            )

            dt = step_len / max(rate, 1e-9)
            cand = float(t_curr) + float(dt)
            if cand < dist[xn, yn]:
                dist[xn, yn] = cand
                heapq.heappush(pq, (cand, xn, yn))

    return dist


# ============================================================
# High-level model
# ============================================================

@dataclass
class FinslerResult:
    arrival_s: np.ndarray
    ignition_mask: np.ndarray
    retardant: np.ndarray


class FinslerFireModel:
    """Deterministic, time-aware Finsler fire-spread model compatible with FireState."""

    def __init__(
        self,
        env: FireEnv,
        *,
        diag: bool | None = None,
        k_wind: float = 0.25,
        w_ref: float = 5.0,
        clamp: tuple[float, float] = (0.05, 5.0),
        k_slope: float = 0.0,
    ):
        self.env = env
        self.diag = bool(env.diag if diag is None else diag)
        self.k_wind = float(k_wind)
        self.w_ref = float(w_ref)
        self.clamp = clamp
        self.k_slope = float(k_slope)

        self._retardant = np.zeros(env.grid_size, dtype=float)
        self._arrival: np.ndarray | None = None
        self._ignition_mask: np.ndarray | None = None
        self._start_time_s = 0.0
        self._retardant_t0_s = 0.0

    @property
    def arrival(self) -> np.ndarray:
        if self._arrival is None:
            raise RuntimeError("Call init_state(...) or simulate_* before accessing arrival.")
        return self._arrival

    def reset_retardant(self):
        self._retardant.fill(0.0)

    def apply_retardant_cartesian(
        self,
        drone_params: np.ndarray | None,
        *,
        drop_w_km: float | None = None,
        drop_h_km: float | None = None,
        amount: float | None = None,
        cell_cap: float | None = None,
    ):
        if drone_params is None:
            return
        drone_params = np.asarray(drone_params, dtype=float)
        if drone_params.size == 0:
            return
        if drone_params.ndim != 2 or drone_params.shape[1] != 3:
            raise ValueError(f"drone_params must have shape (D,3); got {drone_params.shape}")

        env = self.env
        drop_w_km = float(drop_w_km if drop_w_km is not None else env.drop_w_km)
        drop_h_km = float(drop_h_km if drop_h_km is not None else env.drop_h_km)
        amount = float(amount if amount is not None else env.drop_amount)
        cell_cap = cell_cap if cell_cap is not None else env.retardant_cell_cap

        nx, ny = env.grid_size
        cell_km = float(env.domain_km) / max(nx, 1)
        half_w = 0.5 * (drop_w_km / cell_km)
        half_h = 0.5 * (drop_h_km / cell_km)

        X = np.arange(nx)[:, None]
        Y = np.arange(ny)[None, :]

        for x0, y0, phi in drone_params:
            xp = X - x0
            yp = Y - y0
            c = np.cos(phi)
            s = np.sin(phi)
            xr = c * xp + s * yp
            yr = -s * xp + c * yp
            mask = (np.abs(xr) <= half_w) & (np.abs(yr) <= half_h)
            self._retardant[mask] += amount
            if cell_cap is not None:
                capped = np.minimum(self._retardant[mask], cell_cap)
                self._retardant[mask] = capped

    def init_state(self, *, center, radius_km: float):
        cx, cy = center
        nx, ny = self.env.grid_size
        dx_km = float(self.env.domain_km) / max(nx, 1)
        radius_cells = max(int(np.ceil(radius_km / dx_km)), 1)

        X = np.arange(nx)[:, None]
        Y = np.arange(ny)[None, :]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius_cells**2

        self._ignition_mask = mask
        self._arrival = anisotropic_arrival_times(
            self.env,
            mask,
            diag=self.diag,
            k_wind=self.k_wind,
            w_ref=self.w_ref,
            clamp=self.clamp,
            k_slope=self.k_slope,
            retardant0=self._retardant,
            start_time_s=self._start_time_s,
            retardant_applied_time_s=self._retardant_t0_s,
        )

    def firestate_at_time(self, t_abs_s: float) -> FireState:
        """
        Produce a FireState at absolute time t_abs_s (seconds).
        We return boolean burning/burned fields (more natural for deterministic fronts),
        but FireState also tolerates floats.
        """
        if self._arrival is None:
            raise RuntimeError("Call init_state(...) or simulate_* before firestate_at_time().")

        t_abs_s = float(t_abs_s)
        arrival = self._arrival

        affected = arrival <= t_abs_s
        burned = arrival <= max(0.0, t_abs_s - float(self.env.burn_time_s0))
        burning = affected & (~burned)

        burn_remaining = np.maximum(0.0, arrival + float(self.env.burn_time_s0) - t_abs_s)

        dt_ret = max(0.0, t_abs_s - float(self._retardant_t0_s))
        retardant_eff = self._retardant * _retardant_decay_factor(self.env, dt_ret)

        return FireState(
            burning=burning[None, :, :].astype(bool),
            burned=burned[None, :, :].astype(bool),
            burn_remaining_s=burn_remaining[None, :, :].astype(float),
            retardant=retardant_eff[None, :, :].astype(float),
            t=int(t_abs_s / max(float(self.env.dt_s), 1e-6)),
        )

    def simulate_from_ignition(
        self,
        *,
        T: float,
        center,
        radius_km: float,
        drone_params: np.ndarray | None = None,
        start_time_s: float = 0.0,
    ) -> FireState:
        self.reset_retardant()
        self._start_time_s = float(start_time_s)
        self._retardant_t0_s = float(start_time_s)

        self.apply_retardant_cartesian(drone_params)
        self.init_state(center=center, radius_km=radius_km)
        return self.firestate_at_time(self._start_time_s + float(T))

    def simulate_from_firestate(
        self,
        init_firestate: FireState,
        *,
        T: float,
        drone_params: np.ndarray | None = None,
        n_sims: int | None = None,
        ros_mps: float | None = None,
        wind_coeff: float | None = None,
        diag: bool | None = None,
        seed: int | None = None,
        avoid_burning_drop: bool = True, #added for compatibility
        burning_prob_threshold: float = 0.25, #added for compatibility
    ) -> FireState:
        start_t = float(getattr(init_firestate, "time_s", float(getattr(init_firestate, "t", 0)) * float(self.env.dt_s)))
        target_t = start_t + float(T)

        burning = np.asarray(init_firestate.burning[0], dtype=bool)
        front_mask = burning.copy()
        self._ignition_mask = np.asarray(front_mask, dtype=bool)

        self.reset_retardant()
        self._start_time_s = start_t
        self._retardant_t0_s = start_t

        self.apply_retardant_cartesian(drone_params)

        if np.any(self._ignition_mask):
            self._arrival = anisotropic_arrival_times(
                self.env,
                self._ignition_mask,
                diag=self.diag,
                k_wind=self.k_wind,
                w_ref=self.w_ref,
                clamp=self.clamp,
                k_slope=self.k_slope,
                retardant0=self._retardant,
                start_time_s=self._start_time_s,
                retardant_applied_time_s=self._retardant_t0_s,
            )
        elif self._arrival is None:
            self._arrival = np.full(self.env.grid_size, np.inf, dtype=float)

        return self.firestate_at_time(target_t)

    def plot_firestate(
        self,
        state: FireState,
        *,
        sim_idx: int = 0,
        kind: str = "auto",
        title: str | None = None,
        extent_km: float | None = None,
    ):
        burning = state.burning[sim_idx]
        burned = state.burned[sim_idx]
        brs = state.burn_remaining_s[sim_idx]

        is_prob = np.issubdtype(burning.dtype, np.floating) or np.issubdtype(burned.dtype, np.floating)
        if kind == "auto":
            kind = "p_affected" if is_prob else "discrete"

        if extent_km is not None:
            extent = [0, extent_km, 0, extent_km]
            xlabel, ylabel = "x (km)", "y (km)"
        else:
            extent = None
            xlabel, ylabel = "x cell", "y cell"

        plt.figure(figsize=(6, 5))
        if kind == "discrete":
            s = np.zeros_like(burning, dtype=np.int8)
            s[burning.astype(bool)] = 1
            s[burned.astype(bool)] = 2
            im = plt.imshow(s.T, origin="lower", aspect="equal", extent=extent)
            plt.colorbar(im, ticks=[0, 1, 2], label="State (0=unburned, 1=burning, 2=burned)")
        elif kind == "p_burning":
            im = plt.imshow(np.clip(burning, 0, 1).T, origin="lower", vmin=0, vmax=1, aspect="equal", extent=extent)
            plt.colorbar(im, label="P(burning)")
        elif kind == "p_burned":
            im = plt.imshow(np.clip(burned, 0, 1).T, origin="lower", vmin=0, vmax=1, aspect="equal", extent=extent)
            plt.colorbar(im, label="P(burned)")
        elif kind == "p_affected":
            affected = (burning | burned) if (burning.dtype == bool and burned.dtype == bool) else np.clip(burning + burned, 0, 1)
            im = plt.imshow(affected.T, origin="lower", vmin=0, vmax=1, aspect="equal", extent=extent)
            plt.colorbar(im, label="P(burning or burned)" if is_prob else "Affected (burning or burned)")
        elif kind == "burn_remaining":
            im = plt.imshow(brs.T, origin="lower", aspect="equal", extent=extent)
            plt.colorbar(im, label="Burn remaining (s)")
        elif kind == "retardant":  # note: overlays value map
            r = state.retardant[sim_idx]
            v = np.asarray(self.env.value, dtype=float)

            plt.imshow(v.T, origin="lower", aspect="equal", extent=extent, cmap="viridis", interpolation="nearest")

            r_max = float(np.max(r)) if np.size(r) else 0.0
            alpha = np.clip(r / r_max, 0.0, 1.0).T if r_max > 0.0 else 0.0

            im = plt.imshow(
                r.T,
                origin="lower",
                aspect="equal",
                extent=extent,
                cmap="Reds",
                interpolation="nearest",
                alpha=alpha,
                vmin=0.0,
                vmax=max(r_max, 1e-12),
            )
            plt.colorbar(im, label="Retardant load")
        else:
            raise ValueError(f"Unknown kind={kind}")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.show()

    # ---- Boundary utilities ----

    def extract_fire_boundary(
        self,
        firestate,
        *,
        K: int,
        p_boundary: float = 0.5,
        field: str = "affected",
        anchor: str = "max_x",
        ccw: bool = True,
    ) -> FireBoundary:
        return extract_fire_boundary(
            firestate,
            K=K,
            p_boundary=p_boundary,
            field=field,
            anchor=anchor,
            ccw=ccw,
        )

    def plot_fire_boundary(
        self,
        firestate,
        boundary: FireBoundary,
        *,
        field: str = "affected",
        title: str | None = None,
        show_points: bool = True,
    ):
        return plot_fire_boundary(
            firestate,
            boundary,
            field=field,
            title=title,
            show_points=show_points,
        )

    def plot_search_domain(self, mask: np.ndarray, title: str | None = None):
        plt.figure(figsize=(6, 5))
        im = plt.imshow(mask.T, origin="lower", aspect="equal")
        plt.colorbar(im, label="Search region mask")
        plt.xlabel("x cell")
        plt.ylabel("y cell")
        plt.title(title if title is not None else "Region between Finsler boundaries")
        plt.tight_layout()
        plt.show()

    def generate_search_domain(
        self,
        T: float,
        n_sims: int = 1,
        *,
        init_firestate: FireState,
        ros_mps: float | None = None,
        wind_coeff: float | None = None,
        diag: bool | None = None,
        seed: int | None = None,
        p_boundary: float = 0.5,
        K: int = 200,
        boundary_field: str = "affected",
        return_boundaries: bool = False,
    ) -> np.ndarray:
        """
        Deterministic search domain between two fronts.
        Works for bool or float FireState fields. When `return_boundaries=True` the
        initial/final boundaries plus the evolved state are also returned (for BO tooling).
        """
        final_state = self.simulate_from_firestate(init_firestate, T=T, drone_params=None)

        inner_mask = np.logical_or(
            np.asarray(init_firestate.burning[0], dtype=float) > 0.5,
            np.asarray(init_firestate.burned[0], dtype=float) > 0.5,
        )
        outer_mask = np.logical_or(
            np.asarray(final_state.burning[0], dtype=float) > 0.5,
            np.asarray(final_state.burned[0], dtype=float) > 0.5,
        )

        ring_mask = outer_mask & (~inner_mask)

        init_boundary = None
        final_boundary = None

        try:
            init_boundary = self.extract_fire_boundary(
                init_firestate,
                K=K,
                p_boundary=p_boundary,
                field=boundary_field,
                anchor="max_x",
                ccw=True,
            )
            final_boundary = self.extract_fire_boundary(
                final_state,
                K=K,
                p_boundary=p_boundary,
                field=boundary_field,
                anchor="max_x",
                ccw=True,
            )
            boundary_mask = between_boundaries_mask(init_boundary.xy, final_boundary.xy, self.env.grid_size)
        except Exception:
            boundary_mask = np.zeros(self.env.grid_size, dtype=bool)

        if ring_mask.any():
            mask = ring_mask
        elif boundary_mask.any():
            mask = boundary_mask
        else:
            mask = outer_mask

        if return_boundaries:
            return mask, init_boundary, final_boundary, final_state
        return mask


__all__ = ["FinslerFireModel", "anisotropic_arrival_times", "FinslerResult"]
