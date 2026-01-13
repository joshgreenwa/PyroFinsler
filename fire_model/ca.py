import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from numpy.random import default_rng

from fire_model.boundary import (
    FireBoundary,
    between_boundaries_mask,
    extract_fire_boundary,
    plot_fire_boundary,
)


@dataclass(frozen=True)
class FireEnv:
    grid_size: tuple[int, int]
    domain_km: float
    fuel: np.ndarray  # (nx, ny)
    value: np.ndarray  # (nx, ny)
    wind: np.ndarray  # (T, nx, ny, 2) or (nx, ny, 2) if constant
    dt_s: float  # seconds per CA step
    burn_time_s0: float = 600.0
    retardant_half_life_s: float = 1800.0
    retardant_k: float = 1.0
    retardant_cell_cap: float | None = None  # max retardant stored per cell (None disables cap)
    drop_w_km: float = 0.2
    drop_h_km: float = 1.0
    drop_amount: float = 1.0
    ros_mps: float = 0.5
    wind_coeff: float = 0.6
    slope: np.ndarray | None = None  # optional (nx, ny, 2) slope vectors
    diag: bool = True
    avoid_burning_drop: bool = True  # whether to avoid dropping retardant on burning cells
    avoid_drop_p_threshold: float = 0.25  # threshold for considering a cell as burning when avoid_burning_drop is True
    ros_future_jitter_frac: float = 0.0  # fractional stddev for ROS uncertainty after drop (MC-specific)
    wind_coeff_future_jitter_frac: float = 0.0  # fractional stddev for wind-coeff uncertainty after drop
    control_suitability: np.ndarray | None = None  # optional (nx, ny) suitability map for POD/control heuristics


@dataclass
class FireState:
    burning: np.ndarray  # (..., nx, ny) bool or float
    burned: np.ndarray  # (..., nx, ny) bool or float
    burn_remaining_s: np.ndarray  # (..., nx, ny) float
    retardant: np.ndarray  # (n_sims, nx, ny) float
    t: int = 0


class CAFireModel:
    def __init__(self, env: FireEnv, seed: int | None = None):
        self.env = env
        self.base_seed = seed

        nx, _ = env.grid_size
        self.dx = self.env.domain_km / nx
        self.dx_m = self.dx * 1000.0

    def init_state_batch(self, n_sims: int, center, radius_km: float) -> FireState:
        nx, ny = self.env.grid_size
        radius_cells = int(radius_km / self.dx)

        x = np.arange(nx)[:, None]
        y = np.arange(ny)[None, :]
        cx, cy = center
        mask2d = (x - cx) ** 2 + (y - cy) ** 2 <= radius_cells ** 2

        burning = np.zeros((n_sims, nx, ny), dtype=bool)
        burning[:, mask2d] = True

        burned = np.zeros((n_sims, nx, ny), dtype=bool)
        burn_remaining_s = np.broadcast_to(self.env.burn_time_s0, (n_sims, nx, ny)).copy()
        retardant = np.zeros((n_sims, nx, ny), dtype=float)
        return FireState(burning=burning, burned=burned, burn_remaining_s=burn_remaining_s, t=0, retardant=retardant)

    @staticmethod
    def _shift_no_wrap(a: np.ndarray, sx: int, sy: int) -> np.ndarray:
        n_sims, nx, ny = a.shape
        out = np.zeros_like(a)

        x_from0 = max(0, -sx)
        x_from1 = min(nx, nx - sx)
        y_from0 = max(0, -sy)
        y_from1 = min(ny, ny - sy)

        x_to0 = max(0, sx)
        x_to1 = min(nx, nx + sx)
        y_to0 = max(0, sy)
        y_to1 = min(ny, ny + sy)

        if x_from0 < x_from1 and y_from0 < y_from1:
            out[:, x_to0:x_to1, y_to0:y_to1] = a[:, x_from0:x_from1, y_from0:y_from1]
        return out

    def step_batch(
        self,
        state: FireState,
        *,
        ros_mps: float | np.ndarray = 0.5,
        wind_coeff: float | np.ndarray = 0.5,
        diag: bool = True,
    ):
        env = self.env
        dt_s = float(env.dt_s)
        dx_m = float(self.dx_m)
        nx, ny = env.grid_size

        burning = state.burning
        burned = state.burned

        seed = None if self.base_seed is None else (self.base_seed + state.t)
        rng = np.random.default_rng(seed)

        hl = float(env.retardant_half_life_s)
        if hl > 0.0:
            decay = np.exp(-np.log(2.0) * dt_s / hl)
            state.retardant *= decay

        state.burn_remaining_s[burning] = np.maximum(0.0, state.burn_remaining_s[burning] - dt_s)
        newly_burned = burning & (state.burn_remaining_s <= 0.0)
        burned[newly_burned] = True
        burning[newly_burned] = False

        if not np.any(burning):
            state.t += 1
            return

        unburned = ~(burning | burned)
        if env.wind.ndim == 4:
            wt = int(np.clip(int(state.t), 0, env.wind.shape[0] - 1))
            w = env.wind[wt]
        else:
            w = env.wind
        wx = w[..., 0][None, :, :]
        wy = w[..., 1][None, :, :]
        fuel_mul = env.fuel[None, :, :]
        ros_arr = np.asarray(ros_mps, dtype=float)
        n_sims = state.burning.shape[0]
        if ros_arr.ndim == 0:
            lambda0 = float(ros_arr) / dx_m
        else:
            if ros_arr.shape[0] != n_sims:
                raise ValueError("ros_mps array must have length equal to n_sims.")
            lambda0 = ros_arr.reshape(n_sims, 1, 1) / dx_m

        if diag:
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        else:
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        prob_no = np.ones_like(state.burn_remaining_s, dtype=float)
        k = float(env.retardant_k)
        retardant_attn = np.exp(-k * np.maximum(state.retardant, 0.0))

        wind_arr = np.asarray(wind_coeff, dtype=float)
        for sx, sy in dirs:
            src = self._shift_no_wrap(burning, sx, sy)
            if not np.any(src):
                continue

            dist = float(np.hypot(sx, sy))
            ux, uy = sx / dist, sy / dist

            align = wx * ux + wy * uy
            if wind_arr.ndim == 0:
                bias = 1.0 + float(wind_arr) * np.maximum(0.0, align)
            else:
                if wind_arr.shape[0] != n_sims:
                    raise ValueError("wind_coeff array must have length equal to n_sims.")
                bias = 1.0 + wind_arr.reshape(n_sims, 1, 1) * np.maximum(0.0, align)

            lambda_dir = (lambda0 / dist) * fuel_mul * bias
            if env.slope is not None:
                slope_grad = np.asarray(env.slope, dtype=float)
                if slope_grad.shape != (n_sims, nx, ny, 2) and slope_grad.shape != env.slope.shape:
                    slope = np.asarray(env.slope, dtype=float)
                else:
                    slope = slope_grad
                if slope.ndim == 3:  # same for all sims
                    slope_x = slope[..., 0]
                    slope_y = slope[..., 1]
                else:
                    slope_x = slope[..., 0]
                    slope_y = slope[..., 1]
                grade_along = slope_x * ux + slope_y * uy
                theta = np.arctan(grade_along) * 180.0 / np.pi
                theta = np.clip(theta, -30.0, 30.0)
                slope_factor = np.power(2.0, theta / 10.0)
                bias *= slope_factor

            lambda_dir = (lambda0 / dist) * fuel_mul * bias
            lambda_dir = lambda_dir * retardant_attn

            p_dir = 1.0 - np.exp(-lambda_dir * dt_s)
            p_dir = np.clip(p_dir, 0.0, 1.0)

            prob_no *= np.where(src, (1.0 - p_dir), 1.0)

        ignite_prob = np.clip(1.0 - prob_no, 0.0, 1.0)
        u = rng.random(size=ignite_prob.shape)
        newly_ignited = unburned & (u < ignite_prob)

        if np.any(newly_ignited):
            burning[newly_ignited] = True
            state.burn_remaining_s[newly_ignited] = float(env.burn_time_s0)

        state.t += 1

    def simulate_burned_probability(
        self,
        T: int,
        n_sims: int,
        center,
        radius_km: float,
        *,
        ros_mps: float = 0.5,
        wind_coeff: float = 0.50,
        diag: bool = True,
    ) -> np.ndarray:
        state = self.init_state_batch(n_sims=n_sims, center=center, radius_km=radius_km)
        num_steps = int(T / self.env.dt_s)
        for _ in range(num_steps):
            self.step_batch(state, ros_mps=ros_mps, wind_coeff=wind_coeff, diag=diag)
        p_affected = (state.burned | state.burning).mean(axis=0)
        return p_affected

    def aggregate_mc_to_state(self, batch_state: FireState) -> FireState:
        burning_bool = batch_state.burning
        burned_bool = batch_state.burned

        p_burning = burning_bool.mean(axis=0)
        p_burned = burned_bool.mean(axis=0)

        burn_sum = (batch_state.burn_remaining_s * burning_bool).sum(axis=0)
        burn_cnt = burning_bool.sum(axis=0).astype(float)
        burn_mean_cond = np.divide(
            burn_sum,
            burn_cnt,
            out=np.zeros_like(burn_sum, dtype=float),
            where=(burn_cnt > 0),
        )

        r_mean = batch_state.retardant.mean(axis=0)

        return FireState(
            burning=p_burning[None, :, :],
            burned=p_burned[None, :, :],
            burn_remaining_s=burn_mean_cond[None, :, :],
            retardant=r_mean[None, :, :],
            t=batch_state.t,
        )

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

    def discretise_between_boundaries(
        self,
        init_fire_boundary: FireBoundary,
        final_fire_boundary: FireBoundary,
    ) -> np.ndarray:
        return between_boundaries_mask(init_fire_boundary.xy, final_fire_boundary.xy, self.env.grid_size)

    def plot_search_domain(self, discrete_grid_between_boundaries: np.ndarray, title: str = "Region Between Fire Boundaries"):
        plt.figure(figsize=(6, 5))
        im = plt.imshow(discrete_grid_between_boundaries.T, origin="lower", aspect="equal")
        plt.colorbar(im, label="Search Domain (Between Fire Boundaries)")
        plt.xlabel("x cell")
        plt.ylabel("y cell")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def apply_retardant_cartesian(
        self,
        state: FireState,
        drone_params: np.ndarray | None,
        *,
        drop_w_km: float,
        drop_h_km: float,
        amount: float = 1.0,
        avoid_burning: bool = False, # whether to avoid dropping retardant on burning cells
        forbid_burning_overlap: bool = False,  # whether to raise value error if drop overlaps burning cells
        burning_prob_threshold: float = 0.25,  # threshold for considering a cell as burning when avoid_burning is True
        cell_cap: float | None = None,  # optional cap on accumulated retardant per cell
    ):
        if drone_params is None:
            print("No drone_params")
            return
        drone_params = np.asarray(drone_params, dtype=float)
        if drone_params.size == 0:
            print("Empty drone_params")
        if drone_params.ndim != 2 or drone_params.shape[1] != 3:
            raise ValueError(f"drone_params must have shape (D,3); got {drone_params.shape}")

        n_sims, nx, ny = state.retardant.shape
        
        half_w = max(0.5, 0.5 * (drop_w_km / self.dx))
        half_h = max(0.5, 0.5 * (drop_h_km / self.dx))

        print(f"Applying retardant drop: D={drone_params}, amount={amount}, drop_w_km={drop_w_km}, drop_h_km={drop_h_km}")
        X = np.arange(nx)[:, None]
        Y = np.arange(ny)[None, :]

        burning_union = None
        if avoid_burning or forbid_burning_overlap:
            thr = float(burning_prob_threshold)
            if not (0.0 <= thr <= 1.0):
                raise ValueError("burning_prob_threshold must be in [0,1].")
            burning = np.asarray(state.burning)
            if burning.ndim == 2:
                burning2d = burning
            elif burning.ndim == 3:
                burning2d = burning.any(axis=0)
            else:
                raise ValueError(f"state.burning must have shape (nx,ny) or (n_sims,nx,ny); got {burning.shape}")
            if np.issubdtype(burning2d.dtype, np.floating):
                burning_union = burning2d > thr
            else:
                burning_union = burning2d.astype(bool, copy=False)

        for x0, y0, phi in drone_params:
            xp = X - x0
            yp = Y - y0

            c = np.cos(phi)
            s = np.sin(phi)

            xr = c * xp + s * yp
            yr = -s * xp + c * yp

            mask = (np.abs(xr) <= half_w) & (np.abs(yr) <= half_h)
            if burning_union is not None:
                if forbid_burning_overlap and np.any(mask & burning_union):
                    raise ValueError("Retardant drop overlaps actively burning cells.")
                if avoid_burning:
                    mask = mask & (~burning_union)
            state.retardant[:, mask] += amount
            if cell_cap is not None:
                capped = np.minimum(state.retardant[:, mask], cell_cap)
                state.retardant[:, mask] = capped

    def simulate_from_ignition(
        self,
        T: int,
        n_sims: int,
        center,
        radius_km: float,
        *,
        drone_params: np.ndarray | None = None,
        ros_mps: float = 0.5,
        wind_coeff: float = 0.50,
        diag: bool = True,
        avoid_burning_drop: bool = True, # whether to avoid dropping retardant on burning cells
        forbid_burning_overlap: bool = False, # whether to raise value error if drop overlaps burning cells
        burning_prob_threshold: float = 0.25, # used if burning map is probabilistic
    ) -> np.ndarray:
        state = self.init_state_batch(n_sims=n_sims, center=center, radius_km=radius_km)
        self.apply_retardant_cartesian(
            state,
            drone_params,
            drop_w_km=self.env.drop_w_km,
            drop_h_km=self.env.drop_h_km,
            amount=self.env.drop_amount,
            avoid_burning=avoid_burning_drop,
            forbid_burning_overlap=forbid_burning_overlap,
            burning_prob_threshold=burning_prob_threshold,
            cell_cap=self.env.retardant_cell_cap,
        )

        if getattr(self, "base_seed", None) is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.base_seed)

        ros_samples, wind_coeff_samples = self._sample_future_spread_params(
            n_sims,
            base_ros_mps=ros_mps,
            base_wind_coeff=wind_coeff,
            rng=rng,
        )

        num_steps = int(T / self.env.dt_s)
        for _ in range(num_steps):
            self.step_batch(state, ros_mps=ros_samples, wind_coeff=wind_coeff_samples, diag=diag)

        updated_firestate = self.aggregate_mc_to_state(state)
        return updated_firestate

    def _sample_future_spread_params(
        self,
        n_sims: int,
        *,
        base_ros_mps: float,
        base_wind_coeff: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        ros = np.full(int(n_sims), float(base_ros_mps), dtype=float)
        wind_coeff = np.full(int(n_sims), float(base_wind_coeff), dtype=float)

        frac_ros = max(float(self.env.ros_future_jitter_frac), 0.0)
        if frac_ros > 0.0:
            noise = rng.normal(0.0, frac_ros, size=n_sims)
            ros = np.clip(ros * (1.0 + noise), 1e-6, None)

        frac_wind = max(float(self.env.wind_coeff_future_jitter_frac), 0.0)
        if frac_wind > 0.0:
            noise = rng.normal(0.0, frac_wind, size=n_sims)
            wind_coeff = np.clip(wind_coeff * (1.0 + noise), 0.0, None)

        return ros, wind_coeff

    def simulate_from_firestate(
        self,
        init_firestate: FireState,
        T: float,
        n_sims: int,
        *,
        drone_params: np.ndarray | None = None,
        ros_mps: float = 0.5,
        wind_coeff: float = 0.50,
        diag: bool = True,
        seed: int | None = None,
        avoid_burning_drop: bool = True, # whether to avoid dropping retardant on burning cells
        forbid_burning_overlap: bool = False, # whether to raise value error if drop overlaps burning cells
        burning_prob_threshold: float = 0.25, # if avoid burning is True, threshold for considering a cell as burning
    ) -> FireState:
        nx, ny = self.env.grid_size
        dt_s = float(self.env.dt_s)
        num_steps = int(np.ceil(float(T) / dt_s))

        if seed is None:
            seed = None if getattr(self, "base_seed", None) is None else (self.base_seed + int(init_firestate.t))
        rng = np.random.default_rng(seed)

        def _ensure_batched(arr, dtype=None):
            a = np.asarray(arr)
            if dtype is not None:
                a = a.astype(dtype, copy=False)
            if a.ndim == 2:
                a = a[None, :, :]
            if a.shape[1:] != (nx, ny):
                raise ValueError(f"Expected shape (*,{nx},{ny}), got {a.shape}")
            return a

        burning0 = _ensure_batched(init_firestate.burning)
        burned0 = _ensure_batched(init_firestate.burned)
        br0 = _ensure_batched(init_firestate.burn_remaining_s, dtype=float)
        ret0 = _ensure_batched(init_firestate.retardant, dtype=float)
        t0 = int(init_firestate.t)

        if burning0.dtype == bool and burned0.dtype == bool:
            m = burning0.shape[0]
            if m == n_sims:
                burning = burning0.copy()
                burned = burned0.copy()
                burn_remaining_s = br0.copy()
            else:
                reps = int(np.ceil(n_sims / m))
                burning = np.tile(burning0, (reps, 1, 1))[:n_sims].copy()
                burned = np.tile(burned0, (reps, 1, 1))[:n_sims].copy()
                burn_remaining_s = np.tile(br0, (reps, 1, 1))[:n_sims].copy()

            burning &= ~burned
            burn_remaining_s[burned] = 0.0

            retardant = ret0.copy() if m == n_sims else np.tile(ret0, (reps, 1, 1))[:n_sims].copy()
            state = FireState(burning=burning, burned=burned, burn_remaining_s=burn_remaining_s, retardant=retardant, t=t0)
        else:
            p_burning = burning0[0].astype(float, copy=False)
            p_burned = burned0[0].astype(float, copy=False)

            p_burned = np.clip(p_burned, 0.0, 1.0)
            p_burning = np.clip(p_burning, 0.0, 1.0)
            p_burning = np.minimum(p_burning, 1.0 - p_burned)

            u = rng.random((n_sims, nx, ny))
            burned = u < p_burned[None, :, :]
            burning = (u >= p_burned[None, :, :]) & (u < (p_burned + p_burning)[None, :, :])

            burn_remaining_s = np.zeros((n_sims, nx, ny), dtype=float)
            br_map = br0[0]
            burn_remaining_s[burning] = np.broadcast_to(br_map, (n_sims, nx, ny))[burning]
            burn_remaining_s[burned] = 0.0

            ret_map = ret0[0]
            retardant = np.broadcast_to(ret_map, (n_sims, nx, ny)).copy()
            state = FireState(burning=burning, burned=burned, burn_remaining_s=burn_remaining_s, retardant=retardant, t=t0)

        self.apply_retardant_cartesian(
            state,
            drone_params,
            drop_w_km=self.env.drop_w_km,
            drop_h_km=self.env.drop_h_km,
            amount=self.env.drop_amount,
            avoid_burning=avoid_burning_drop,
            forbid_burning_overlap=forbid_burning_overlap,
            burning_prob_threshold=burning_prob_threshold,
            cell_cap=self.env.retardant_cell_cap,
        )

        ros_samples, wind_coeff_samples = self._sample_future_spread_params(
            n_sims,
            base_ros_mps=ros_mps,
            base_wind_coeff=wind_coeff,
            rng=rng,
        )

        for _ in range(num_steps):
            self.step_batch(state, ros_mps=ros_samples, wind_coeff=wind_coeff_samples, diag=diag)

        return self.aggregate_mc_to_state(state)

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
        elif kind == "retardant": #note: overlays value map
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

    def state_int_single(self, state: FireState, sim_idx: int = 0) -> np.ndarray:
        s = np.zeros(state.burning.shape[1:], dtype=np.int8)
        s[state.burning[sim_idx]] = 1
        s[state.burned[sim_idx]] = 2
        return s

    def plot_single_run(self, state: FireState, sim_idx: int = 0, title: str | None = None):
        s = self.state_int_single(state, sim_idx=sim_idx)
        plt.figure(figsize=(6, 5))
        plt.imshow(s.T, origin="lower", aspect="equal")
        plt.colorbar(ticks=[0, 1, 2], label="State (0=unburned, 1=burning, 2=burned)")
        plt.xlabel("x cell")
        plt.ylabel("y cell")
        plt.title(title if title is not None else f"Fire state at t={state.t}")
        plt.tight_layout()
        plt.show()

    def plot_probability_map(
        self,
        p_map: np.ndarray,
        *,
        title: str | None = None,
        cbar_label: str = "P(burned by T)",
        extent_km: bool = True,
    ):
        extent = [0, self.env.domain_km, 0, self.env.domain_km] if extent_km else None
        plt.figure(figsize=(6, 5))
        im = plt.imshow(p_map.T, origin="lower", vmin=0.0, vmax=1.0, aspect="equal", extent=extent)
        plt.colorbar(im, label=cbar_label)
        plt.xlabel("x (km)" if extent_km else "x cell")
        plt.ylabel("y (km)" if extent_km else "y cell")
        plt.title(title if title is not None else "Monte Carlo burned probability map")
        plt.tight_layout()
        plt.show()

    def generate_search_domain(
        self,
        T: float,
        n_sims: int,
        *,
        init_firestate: FireState = None,
        ros_mps: float = 0.5,
        wind_coeff: float = 0.50,
        diag: bool = True,
        seed: int | None = None,
        p_boundary: float = 0.25,
        K: int = 200,
        boundary_field: str = "affected",
        return_boundaries: bool = False,
    ):
        if init_firestate is None:
            init_firestate = self.init_state_batch(n_sims=n_sims, center=(50, 50), radius_km=0.2)

        final_firestate = self.simulate_from_firestate(
            init_firestate,
            T=T,
            n_sims=n_sims,
            drone_params=None,
            ros_mps=ros_mps,
            wind_coeff=wind_coeff,
            diag=diag,
            seed=seed,
        )

        init_boundary = self.extract_fire_boundary(
            init_firestate,
            K=K,
            p_boundary=p_boundary,
            field=boundary_field,
            anchor="max_x",
            ccw=True,
        )

        final_boundary = self.extract_fire_boundary(
            final_firestate,
            K=K,
            p_boundary=p_boundary,
            field=boundary_field,
            anchor="max_x",
            ccw=True,
        )

        search_domain_mask = self.discretise_between_boundaries(init_boundary, final_boundary)
        if return_boundaries:
            return search_domain_mask, init_boundary, final_boundary, final_firestate
        return search_domain_mask


__all__ = ["FireEnv", "FireState", "CAFireModel"]
