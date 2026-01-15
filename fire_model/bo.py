import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter, Matern, WhiteKernel, Kernel
from scipy.spatial import cKDTree

from fire_model.ca import CAFireModel, FireState
from fire_model.boundary import FireBoundary


class SearchGridProjector:
    """Project 2D points onto a search grid defined by a boolean mask of valid cells."""

    def __init__(self, mask: np.ndarray, coords: np.ndarray | None = None):
        mask = np.asarray(mask, dtype=bool)
        xs, ys = np.where(mask)
        coords = coords if coords is not None else np.stack([xs, ys], axis=1)
        if coords.size == 0:
            raise ValueError("Search grid is empty; adjust boundary parameters.")
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        self.coords = coords[order].astype(float)
        self.mask = mask
        self.shape = mask.shape
        self.tree = cKDTree(self.coords)

    def snap(self, x: float, y: float) -> np.ndarray:
        _, idx = self.tree.query([x, y], k=1)
        return self.coords[idx]

    def evenly_spaced(self, n: int) -> np.ndarray:
        idx = np.linspace(0, len(self.coords) - 1, num=max(n, 1), dtype=int)
        return self.coords[idx]

    def random_coords(self, rng, n: int) -> np.ndarray:
        idx = rng.integers(0, len(self.coords), size=n)
        return self.coords[idx]


class TiedXYFiMatern(Kernel):
    """Matérn kernel on inputs with repeating 4D blocks per drone: [x_norm, y_norm, sin(phi), cos(phi)]."""

    def __init__(
        self,
        lx: float = 0.2,
        ly: float = 0.2,
        lphi: float = 0.5,
        nu: float = 2.5,
        length_scale_bounds=(1e-3, 1e3),
        fd_eps: float = 1e-6,
    ):
        self.lx = float(lx)
        self.ly = float(ly)
        self.lphi = float(lphi)
        self.nu = nu
        self.length_scale_bounds = length_scale_bounds
        self.fd_eps = float(fd_eps)
        self._base = Matern(length_scale=1.0, nu=nu)

    @property
    def hyperparameter_lx(self):
        return Hyperparameter("lx", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_ly(self):
        return Hyperparameter("ly", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_lphi(self):
        return Hyperparameter("lphi", "numeric", self.length_scale_bounds)

    @property
    def theta(self):
        return np.log([self.lx, self.ly, self.lphi])

    @theta.setter
    def theta(self, theta):
        lx, ly, lphi = np.exp(theta)
        self.lx, self.ly, self.lphi = float(lx), float(ly), float(lphi)

    @property
    def bounds(self):
        b = np.log(np.array([self.length_scale_bounds] * 3, dtype=float))
        return b

    def _scale(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        d = X.shape[1]
        if d % 4 != 0:
            raise ValueError(f"Expected feature dim multiple of 4, got {d}. "
                             "Make sure you're using [x,y,sin,cos] per drone.")

        Xs = X.copy()
        idx = np.arange(d)
        x_idx = (idx % 4) == 0
        y_idx = (idx % 4) == 1
        p_idx = (idx % 4) >= 2

        Xs[:, x_idx] /= self.lx
        Xs[:, y_idx] /= self.ly
        Xs[:, p_idx] /= self.lphi
        return Xs

    def __call__(self, X, Y=None, eval_gradient=False):
        Xs = self._scale(X)
        Ys = self._scale(Y) if Y is not None else None

        K = self._base(Xs, Ys, eval_gradient=False)
        if not eval_gradient:
            return K

        theta0 = self.theta.copy()
        eps = self.fd_eps

        grad = np.empty(K.shape + (3,), dtype=float)
        for i in range(3):
            th = theta0.copy()
            th[i] += eps
            k_tmp = self.clone_with_theta(th)
            Kp = k_tmp(X, Y, eval_gradient=False)
            grad[..., i] = (Kp - K) / eps

        return K, grad

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}(lx={self.lx:.3g}, ly={self.ly:.3g}, lphi={self.lphi:.3g}, nu={self.nu})"


def expected_improvement(X_candidates, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.clip(sigma, 1e-9, None)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 0.0] = 0.0
    return ei


class RetardantDropBayesOpt:
    """Bayesian optimisation loop for choosing retardant drop parameters to minimise expected burned area."""

    def __init__(
        self,
        fire_model: CAFireModel,
        init_firestate: FireState,
        n_drones: int,
        evolution_time_s: float,
        n_sims: int,
        fire_boundary_probability: float = 0.25,
        rng=None,
        search_grid_evolution_time_s: float | None = None,
    ):
        self.fire_model = fire_model
        self.init_firestate = init_firestate
        self.n_drones = n_drones
        self.dim = 3 * n_drones
        self.evolution_time_s = evolution_time_s
        self.search_grid_evolution_time_s = search_grid_evolution_time_s
        self.n_sims = n_sims
        self.p_boundary = fire_boundary_probability
        self.rng = default_rng() if rng is None else rng

        self.search_domain_mask = None
        self.projector = None
        self.shape = None
        self.init_boundary: FireBoundary | None = None
        self.final_boundary: FireBoundary | None = None
        self.final_search_firestate: FireState | None = None
        self.sr_grid: np.ndarray | None = None
        self.sr_r_targets: np.ndarray | None = None
        self.sr_phi_grid: np.ndarray | None = None
        self.sr_valid_mask: np.ndarray | None = None
        self.sr_index_tree: cKDTree | None = None
        self.sr_valid_indices: np.ndarray | None = None

    @staticmethod
    def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        n = float(np.linalg.norm(v))
        return v / (n + eps)

    @staticmethod
    def _wrap_angle(phi: float) -> float:
        return float(np.mod(phi, 2.0 * np.pi))

    @classmethod
    def _phi_from_long_axis_angle(cls, long_axis_angle: float) -> float:
        """
        Convert a desired long-axis direction angle to the model's `phi`.

        In `apply_retardant_cartesian`, the long axis of the retardant rectangle is aligned
        with the rotated y'-axis, i.e. long-axis direction is `[sin(phi), cos(phi)]`.
        """
        return cls._wrap_angle(0.5 * np.pi - float(long_axis_angle))

    def _estimate_mean_wind(self) -> np.ndarray:
        env = self.fire_model.env
        wind = getattr(env, "wind", None)
        if wind is None:
            return np.zeros(2, dtype=float)

        w = np.asarray(wind, dtype=float)
        if w.ndim == 4:
            idx = int(np.clip(int(self.init_firestate.t), 0, w.shape[0] - 1))
            w0 = w[idx]
        else:
            w0 = w
        if w0.ndim != 3 or w0.shape[-1] != 2:
            return np.zeros(2, dtype=float)

        burning = np.asarray(self.init_firestate.burning)
        if burning.ndim == 3:
            burning = burning[0]
        if np.issubdtype(burning.dtype, np.floating):
            burning = burning > 0.5

        if burning.any():
            return np.mean(w0[burning], axis=0)
        return np.mean(w0.reshape(-1, 2), axis=0)

    def generate_search_grid(self, K=500, boundary_field="affected"):
        if self.search_grid_evolution_time_s is not None:
            T = self.search_grid_evolution_time_s
        else:
            T = self.evolution_time_s
        search = self.fire_model.generate_search_domain(
            T=T,
            n_sims=self.n_sims,
            init_firestate=self.init_firestate,
            ros_mps=self.fire_model.env.ros_mps,
            wind_coeff=self.fire_model.env.wind_coeff,
            diag=self.fire_model.env.diag,
            seed=None,
            p_boundary=self.p_boundary,
            K=K,
            boundary_field=boundary_field,
            return_boundaries=True,
        )
        search_domain_mask, init_boundary, final_boundary, final_state = search
        self.init_boundary = init_boundary
        self.final_boundary = final_boundary
        self.final_search_firestate = final_state

        xs, ys = np.where(search_domain_mask)
        coords = np.stack([xs.astype(float), ys.astype(float)], axis=1)
        return search_domain_mask, coords

    def setup_search_grid(self, K=500, boundary_field="affected"):
        mask, coords = self.generate_search_grid(K=K, boundary_field=boundary_field)
        self.search_domain_mask = mask
        self.projector = SearchGridProjector(mask=mask, coords=coords)
        self.shape = mask.shape
        return mask, coords

    @staticmethod
    def _build_linear_grid(boundary0_xy: np.ndarray, boundary1_xy: np.ndarray, r_targets: np.ndarray) -> np.ndarray:
        r_targets = np.asarray(r_targets, dtype=float)
        grid = np.zeros((boundary0_xy.shape[0], r_targets.size, 2), dtype=float)
        for j, r in enumerate(r_targets):
            grid[:, j, :] = (1.0 - r) * boundary0_xy + r * boundary1_xy
        return grid

    @staticmethod
    def _smooth_grid_laplace(grid: np.ndarray, *, n_iters: int = 300, omega: float = 1.0) -> np.ndarray:
        g = np.asarray(grid, dtype=float).copy()
        for _ in range(int(n_iters)):
            avg = 0.25 * (
                np.roll(g, 1, axis=0)
                + np.roll(g, -1, axis=0)
                + np.roll(g, 1, axis=1)
                + np.roll(g, -1, axis=1)
            )
            g[:, 1:-1, :] = (1.0 - omega) * g[:, 1:-1, :] + omega * avg[:, 1:-1, :]
        return g

    @staticmethod
    def _grid_angle_along_s(grid: np.ndarray) -> np.ndarray:
        d = np.roll(grid, -1, axis=0) - np.roll(grid, 1, axis=0)
        return np.arctan2(d[..., 1], d[..., 0])

    def setup_search_grid_polar(
        self,
        *,
        K: int = 500,
        boundary_field: str = "affected",
        n_r: int = 160,
        smooth_iters: int = 350,
        omega: float = 1.0,
    ):
        self.setup_search_grid(K=K, boundary_field=boundary_field)
        if self.init_boundary is None or self.final_boundary is None:
            raise RuntimeError("Search grid not initialised; boundaries are missing.")

        inner_xy = np.asarray(self.init_boundary.xy, dtype=float)
        outer_xy = np.asarray(self.final_boundary.xy, dtype=float)
        if inner_xy.shape != outer_xy.shape:
            raise ValueError("Boundary point counts do not match; rerun setup_search_grid.")

        r_targets = np.linspace(0.0, 1.0, int(n_r))
        grid_linear = self._build_linear_grid(inner_xy, outer_xy, r_targets)
        grid = self._smooth_grid_laplace(grid_linear, n_iters=smooth_iters, omega=omega)

        valid = np.isfinite(grid[..., 0]) & np.isfinite(grid[..., 1])
        idx = np.column_stack(np.where(valid))
        if idx.size == 0:
            raise RuntimeError("Polar grid has no valid points; check boundaries or parameters.")

        self.sr_grid = grid
        self.sr_r_targets = r_targets
        self.sr_phi_grid = self._grid_angle_along_s(grid)
        self.sr_valid_mask = valid
        self.sr_valid_indices = idx.astype(float)
        self.sr_index_tree = cKDTree(self.sr_valid_indices)
        return grid

    def _sr_lookup(self, s: float, r: float) -> tuple[np.ndarray, float]:
        if self.sr_grid is None or self.sr_phi_grid is None or self.sr_index_tree is None:
            raise RuntimeError("Polar grid not initialised; call setup_search_grid_polar(...).")

        grid = self.sr_grid
        K, R = grid.shape[0], grid.shape[1]
        s = float(np.clip(s, 0.0, 1.0))
        r = float(np.clip(r, 0.0, 1.0))
        i_f = s * (K - 1)
        j_f = r * (R - 1)
        i = int(np.round(i_f))
        j = int(np.round(j_f))

        if not self.sr_valid_mask[i, j]:
            _, idx = self.sr_index_tree.query([i_f, j_f], k=1)
            i = int(self.sr_valid_indices[idx, 0])
            j = int(self.sr_valid_indices[idx, 1])

        xy = grid[i, j]
        phi = float(self.sr_phi_grid[i, j])
        return xy, phi

    def plot_polar_domain(
        self,
        *,
        n_s_lines: int = 12,
        n_r_lines: int = 8,
        show_fields: bool = True,
        show_diagnostics: bool = True,
        hist_bins: int = 40,
    ):
        if self.sr_grid is None or self.sr_r_targets is None:
            raise RuntimeError("Polar grid not initialised; call setup_search_grid_polar(...).")
        if self.search_domain_mask is None:
            raise RuntimeError("Search mask missing; call setup_search_grid_polar(...).")

        grid = self.sr_grid
        mask = self.search_domain_mask

        fig, axes = plt.subplots(1, 3 if show_fields else 1, figsize=(14, 4))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        ax0 = axes[0]
        ax0.imshow(mask.T, origin="lower", aspect="equal", cmap="Greys", alpha=0.35)
        for j in range(0, grid.shape[1], max(1, grid.shape[1] // n_r_lines)):
            curve = grid[:, j]
            ax0.plot(curve[:, 0], curve[:, 1], color="tab:blue", linewidth=1)
        for i in range(0, grid.shape[0], max(1, grid.shape[0] // n_s_lines)):
            curve = grid[i]
            ax0.plot(curve[:, 0], curve[:, 1], color="tab:orange", linewidth=0.7, alpha=0.7)
        if self.init_boundary is not None:
            ax0.plot(self.init_boundary.xy[:, 0], self.init_boundary.xy[:, 1], color="tab:red", linewidth=2)
        if self.final_boundary is not None:
            ax0.plot(self.final_boundary.xy[:, 0], self.final_boundary.xy[:, 1], color="tab:green", linewidth=2)
        ax0.set_title("Polar grid (s,r) on mask")
        ax0.set_xticks([])
        ax0.set_yticks([])

        if not show_fields:
            plt.tight_layout()
            plt.show()
            return

        valid = np.isfinite(grid[..., 0]) & np.isfinite(grid[..., 1])
        pts = grid[valid]
        idxs = np.column_stack(np.where(valid))
        tree = cKDTree(pts)
        coords = np.column_stack(np.where(mask)).astype(float)
        _, nn = tree.query(coords, k=1)

        s_idx = idxs[nn, 0]
        r_idx = idxs[nn, 1]
        s_field = np.full(mask.shape, np.nan)
        r_field = np.full(mask.shape, np.nan)
        s_field[coords[:, 0].astype(int), coords[:, 1].astype(int)] = s_idx / float(grid.shape[0])
        r_field[coords[:, 0].astype(int), coords[:, 1].astype(int)] = self.sr_r_targets[r_idx]

        ax1 = axes[1]
        im1 = ax1.imshow(s_field.T, origin="lower", aspect="equal", cmap="viridis")
        ax1.set_title("s field")
        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = axes[2]
        im2 = ax2.imshow(r_field.T, origin="lower", aspect="equal", cmap="viridis")
        ax2.set_title("r field")
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        plt.tight_layout()
        plt.show()

        if not show_diagnostics:
            return

        dist, nn = tree.query(coords, k=1)
        s_idx = idxs[nn, 0]
        r_idx = idxs[nn, 1]

        counts = np.zeros(grid.shape[:2], dtype=int)
        np.add.at(counts, (s_idx, r_idx), 1)

        density_field = np.full(mask.shape, np.nan)
        density_field[coords[:, 0].astype(int), coords[:, 1].astype(int)] = counts[s_idx, r_idx]

        dist_field = np.full(mask.shape, np.nan)
        dist_field[coords[:, 0].astype(int), coords[:, 1].astype(int)] = dist

        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
        axd = axes2[0]
        imd = axd.imshow(density_field.T, origin="lower", aspect="equal", cmap="viridis")
        axd.set_title("Mask points per grid point")
        axd.set_xticks([])
        axd.set_yticks([])
        plt.colorbar(imd, ax=axd, fraction=0.046)

        axdist = axes2[1]
        imdist = axdist.imshow(dist_field.T, origin="lower", aspect="equal", cmap="magma")
        axdist.set_title("Distance to nearest grid point")
        axdist.set_xticks([])
        axdist.set_yticks([])
        plt.colorbar(imdist, ax=axdist, fraction=0.046)

        axh = axes2[2]
        axh.hist(dist, bins=max(int(hist_bins), 5), color="tab:blue", alpha=0.8)
        axh.set_title("Distance histogram")
        axh.set_xlabel("Distance (cells)")
        axh.set_ylabel("Count")

        plt.tight_layout()
        plt.show()

    def sample_random_theta(self, n: int = 1):
        if n == 1:
            return self.rng.random(self.dim)
        return self.rng.random((n, self.dim))

    def sample_qmc_theta(self, n: int, *, method: str = "sobol") -> np.ndarray:
        """
        Low-discrepancy samples in [0,1]^dim (better global coverage than pure random).

        `method` options: 'sobol', 'halton', 'lhs'.
        """
        n = int(n)
        if n <= 0:
            raise ValueError("n must be >= 1")

        method = str(method).lower().strip()
        seed = int(self.rng.integers(0, 2**32 - 1))

        if method == "sobol":
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=seed)
            X = sampler.random(n)
        elif method == "halton":
            sampler = qmc.Halton(d=self.dim, scramble=True, seed=seed)
            X = sampler.random(n)
        elif method in {"lhs", "latin", "latin-hypercube"}:
            sampler = qmc.LatinHypercube(d=self.dim, seed=seed)
            X = sampler.random(n)
        else:
            raise ValueError("Unknown QMC method. Use 'sobol', 'halton', or 'lhs'.")

        return np.asarray(X, dtype=float)

    def sample_random_theta_on_mask(self, n: int = 1):
        """
        Random initialisation that is uniform over valid search cells (instead of uniform over [0,1]^dim
        plus snapping), with random orientations.
        """
        if self.projector is None or self.shape is None:
            raise RuntimeError("Call setup_search_grid(...) before sampling on mask.")

        nx, ny = self.shape
        thetas = np.empty((max(n, 1), self.dim), dtype=float)

        for i in range(max(n, 1)):
            coords = self.projector.random_coords(self.rng, self.n_drones)
            phis = self.rng.random(self.n_drones) * (2.0 * np.pi)
            params = np.column_stack([coords, phis])

            theta = np.empty(self.dim, dtype=float)
            for d, (xg, yg, phi) in enumerate(params):
                theta[3 * d + 0] = float(xg) / max(nx - 1, 1)
                theta[3 * d + 1] = float(yg) / max(ny - 1, 1)
                theta[3 * d + 2] = self._wrap_angle(float(phi)) / (2.0 * np.pi)
            thetas[i] = np.clip(theta, 0.0, 1.0)

        return thetas[0] if n == 1 else thetas

    def _random_params_on_mask(self, count: int) -> np.ndarray:
        """
        Utility that mirrors `sample_random_theta_on_mask` but returns snapped (x,y,phi) tuples.
        """
        count = int(max(count, 0))
        if count == 0:
            return np.empty((0, 3), dtype=float)
        if self.projector is None:
            raise RuntimeError("Call setup_search_grid(...) before sampling.")
        coords = self.projector.random_coords(self.rng, count)
        phis = self.rng.random(count) * (2.0 * np.pi)
        params = np.column_stack([coords, phis])
        return params

    def _normalize_field(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmax <= vmin + 1e-12:
            return np.zeros_like(values, dtype=float)
        return (values - vmin) / (vmax - vmin)

    def _sample_map_along(
        self,
        field_map: np.ndarray | None,
        inner_xy: np.ndarray,
        directions: np.ndarray,
        *,
        alpha: float,
    ) -> np.ndarray:
        if field_map is None:
            return np.zeros(inner_xy.shape[0], dtype=float)

        fmap = np.asarray(field_map, dtype=float)
        if fmap.shape != self.shape:
            raise ValueError(f"Field map shape {fmap.shape} does not match grid {self.shape}.")

        alpha = float(alpha)
        pts = inner_xy + alpha * directions
        nx, ny = self.shape
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, nx - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, ny - 1)
        vals = fmap[xi, yi]
        return self._normalize_field(vals)

    def _build_heuristic_context(
        self,
        *,
        control_map: np.ndarray | None = None,
        score_alpha: float = 0.6,
    ) -> dict:
        # Build context with precomputed quantities for heuristic sampling.
        # Used internally by various heuristic methods.
        if any(x is None for x in (self.projector, self.shape, self.init_boundary, self.final_boundary)):
            raise RuntimeError("Search grid not initialised; call setup_search_grid(...) first.")

        inner_xy = np.asarray(self.init_boundary.xy, dtype=float)
        outer_xy = np.asarray(self.final_boundary.xy, dtype=float)
        if inner_xy.shape != outer_xy.shape:
            raise ValueError("Boundary point counts do not match; rerun setup_search_grid.")

        v_out = outer_xy - inner_xy
        tangents = np.roll(inner_xy, -1, axis=0) - np.roll(inner_xy, 1, axis=0)
        tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9
        tangents_unit = tangents / tangent_norm
        v_norm = np.linalg.norm(v_out, axis=1, keepdims=True) + 1e-9
        v_unit = v_out / v_norm
        normals = np.stack([-tangents_unit[:, 1], tangents_unit[:, 0]], axis=1)

        w_mean = self._estimate_mean_wind()
        w_mag = float(np.linalg.norm(w_mean))
        wind_unit = self._unit(w_mean)

        env = self.fire_model.env
        value_map = np.asarray(env.value, dtype=float)
        ctrl_map = control_map
        if ctrl_map is None:
            ctrl_map = getattr(env, "control_suitability", None)
        ctrl_map = None if ctrl_map is None else np.asarray(ctrl_map, dtype=float)
        if ctrl_map is not None and ctrl_map.shape != self.shape:
            raise ValueError("control_suitability map must match grid size.")

        score_alpha = float(np.clip(score_alpha, 0.0, 1.0))
        value_scores = self._sample_map_along(value_map, inner_xy, v_out, alpha=score_alpha)
        control_scores = self._sample_map_along(ctrl_map, inner_xy, v_out, alpha=score_alpha) if ctrl_map is not None else np.zeros_like(value_scores)

        init_burning = np.asarray(self.init_firestate.burning, dtype=float)
        init_burned = np.asarray(self.init_firestate.burned, dtype=float)
        if init_burning.ndim == 3:
            init_burning = init_burning[0]
        if init_burned.ndim == 3:
            init_burned = init_burned[0]
        p_init = np.clip(init_burning + init_burned, 0.0, 1.0)

        final_state = self.final_search_firestate
        if final_state is not None:
            p_fin = np.asarray(final_state.burning[0], dtype=float) + np.asarray(final_state.burned[0], dtype=float)
            p_final = np.clip(p_fin, 0.0, 1.0)
        else:
            p_final = np.zeros(self.shape, dtype=float)

        coords = np.indices(self.shape)
        total_w = p_init.sum()
        if total_w > 1e-9:
            cx = float((coords[0] * p_init).sum() / total_w)
            cy = float((coords[1] * p_init).sum() / total_w)
        else:
            cx = float(self.shape[0] / 2.0)
            cy = float(self.shape[1] / 2.0)

        boundary_tree = cKDTree(outer_xy) if outer_xy.size > 0 else None

        context = {
            "inner_xy": inner_xy,
            "outer_xy": outer_xy,
            "v_out": v_out,
            "v_unit": v_unit,
            "tangents": tangents_unit,
            "normals": normals,
            "wind_unit": wind_unit,
            "wind_mag": w_mag,
            "value_map": value_map,
            "value_scores": value_scores,
            "control_scores": control_scores,
            "control_map": ctrl_map,
            "p_init": p_init,
            "p_final": p_final,
            "init_centroid": np.array([cx, cy], dtype=float),
            "boundary_tree": boundary_tree,
            "search_mask": self.search_domain_mask,
            "score_alpha": score_alpha,
        }
        return context

    def _resolve_mode_counts(self, total: int, modes: list[str], allocations: dict | None) -> dict[str, int]:
        total = int(max(total, 0))
        if total == 0 or not modes:
            return {m: 0 for m in modes}
        weights = []
        if allocations:
            for m in modes:
                weights.append(max(float(allocations.get(m, 0.0)), 0.0))
        if not allocations or sum(weights) <= 0.0:
            weights = [1.0] * len(modes)
        total_w = float(sum(weights))
        desired = [w / total_w * total for w in weights]
        counts = {modes[i]: int(np.floor(desired[i])) for i in range(len(modes))}
        assigned = int(sum(counts.values()))
        remainders = sorted(
            ((desired[i] - np.floor(desired[i]), i) for i in range(len(modes))),
            reverse=True,
        )
        idx = 0
        while assigned < total and idx < len(remainders):
            _, i = remainders[idx]
            counts[modes[i]] += 1
            assigned += 1
            idx += 1
        i = 0
        while assigned < total and i < len(modes):
            counts[modes[i]] += 1
            assigned += 1
            i += 1
        return counts

    def _min_arc_for_count(self, K: int, count: int, min_arc_sep_frac: float) -> int:
        K = int(max(K, 1))
        count = int(max(count, 1))
        per = float(K) / float(count)
        min_arc = int(np.floor(float(min_arc_sep_frac) * per))
        return int(np.clip(min_arc, 1, max(K // 2, 1)))

    def _select_spaced_by_score(
        self,
        count: int,
        scores: np.ndarray,
        *,
        min_arc: int,
        idx_pool: np.ndarray | None = None,
    ) -> list[int]:
        if count <= 0:
            return []
        scores = np.asarray(scores, dtype=float)
        K = scores.shape[0]
        idx_pool = np.arange(K, dtype=int) if idx_pool is None else np.asarray(idx_pool, dtype=int)
        if idx_pool.size == 0:
            return []
        order = idx_pool[np.argsort(scores[idx_pool])[::-1]]
        taken = np.zeros(K, dtype=bool)
        chosen: list[int] = []
        for idx in order:
            if len(chosen) >= count:
                break
            if taken[idx]:
                continue
            chosen.append(int(idx))
            for off in range(-min_arc, min_arc + 1):
                taken[(idx + off) % K] = True
        return chosen

    def _select_evenly_spaced(self, count: int, idx_pool: np.ndarray) -> list[int]:
        count = int(max(count, 0))
        if count <= 0:
            return []
        pool = np.asarray(idx_pool, dtype=int)
        if pool.size == 0:
            return []
        pool = np.sort(pool)
        if count >= pool.size:
            return pool.tolist()
        offset = int(self.rng.integers(0, pool.size))
        base = np.linspace(0, pool.size - 1, num=count, dtype=int)
        sel = (base + offset) % pool.size
        return pool[sel].tolist()

    def _placement_downwind(
        self,
        idx: int,
        ctx: dict,
        *,
        offset_cells: float,
        phi_jitter_rad: float,
        orientation: str = "perp",
    ) -> list[float] | None:
        if ctx["wind_mag"] <= 1e-9:
            return None
        inner_xy = ctx["inner_xy"]
        wind_unit = ctx["wind_unit"]
        pt = inner_xy[idx] + wind_unit * float(offset_cells)
        xg, yg = self.projector.snap(float(pt[0]), float(pt[1]))
        orient = str(orientation).lower().strip()
        if orient in ("perp", "perpendicular", "cross", "crosswind"):
            u_long = self._unit(np.array([-wind_unit[1], wind_unit[0]], dtype=float))
        elif orient in ("parallel", "wind", "downwind"):
            u_long = wind_unit
        else:
            u_long = self._unit(np.array([-wind_unit[1], wind_unit[0]], dtype=float))
        long_angle = float(np.arctan2(u_long[1], u_long[0]))
        phi = self._phi_from_long_axis_angle(long_angle)
        if phi_jitter_rad > 0.0:
            phi = self._wrap_angle(phi + float(self.rng.normal(0.0, phi_jitter_rad)))
        return [xg, yg, phi]

    def _placement_tangent(
        self,
        idx: int,
        ctx: dict,
        *,
        offset_cells: float,
        phi_jitter_rad: float,
        orientation: str = "normal",
    ) -> list[float]:
        inner_xy = ctx["inner_xy"]
        v_unit = ctx["v_unit"]
        tangents = ctx["tangents"]
        pt = inner_xy[idx] + v_unit[idx] * float(offset_cells)
        xg, yg = self.projector.snap(float(pt[0]), float(pt[1]))
        orient = str(orientation).lower().strip()
        if orient in ("normal", "perp", "perpendicular"):
            u_long = v_unit[idx]
            if float(np.linalg.norm(u_long)) <= 1e-9:
                u_long = ctx["normals"][idx]
        elif orient in ("tangent", "parallel"):
            u_long = tangents[idx]
        else:
            u_long = tangents[idx]
        long_angle = float(np.arctan2(u_long[1], u_long[0]))
        phi = self._phi_from_long_axis_angle(long_angle)
        if phi_jitter_rad > 0.0:
            phi = self._wrap_angle(phi + float(self.rng.normal(0.0, phi_jitter_rad)))
        return [xg, yg, phi]

    def _heuristic_fire_asset_blocking(
        self,
        count: int,
        ctx: dict,
        *,
        asset_value_quantile: float,
        asset_alpha_range: tuple[float, float],
        asset_distance_scale_cells: float | None,
        asset_burning_threshold: float,
        min_arc_sep_frac: float,
        downwind_offset_cells: float,
        tangent_offset_cells: float,
        tangent_wind_scale: float,
        phi_jitter_rad: float,
    ) -> list[list[float]]:
        if count <= 0:
            return []
        value_map = ctx["value_map"]
        flat = value_map.ravel()
        if float(np.max(flat)) <= 0.0:
            return []
        quantile = float(np.clip(asset_value_quantile, 0.0, 1.0))
        thresh = np.quantile(flat, quantile) if quantile > 0.0 else np.min(flat)
        asset_mask = value_map >= thresh
        if not np.any(asset_mask):
            return []
        asset_coords = np.column_stack(np.where(asset_mask)).astype(float)
        asset_vals = value_map[asset_mask].astype(float)
        vmin = float(np.min(asset_vals))
        vmax = float(np.max(asset_vals))
        if vmax <= vmin + 1e-9:
            asset_scores = np.ones_like(asset_vals, dtype=float)
        else:
            asset_scores = (asset_vals - vmin) / (vmax - vmin)

        tree = cKDTree(asset_coords)
        inner_xy = ctx["inner_xy"]
        dists, nearest = tree.query(inner_xy)
        scale = asset_distance_scale_cells
        if scale is None:
            scale = max(float(np.median(dists)), 1.0)
        score = np.exp(-dists / float(max(scale, 1e-6))) * (0.5 + 0.5 * asset_scores[nearest])

        K = inner_xy.shape[0]
        min_arc = self._min_arc_for_count(K, count, min_arc_sep_frac)
        chosen = self._select_spaced_by_score(count, score, min_arc=min_arc)
        if not chosen:
            return []

        lo, hi = asset_alpha_range
        lo = float(np.clip(lo, 0.0, 1.0))
        hi = float(np.clip(hi, 0.0, 1.0))
        if hi < lo:
            lo, hi = hi, lo

        p_init = ctx["p_init"]
        placements: list[list[float]] = []
        for idx in chosen:
            asset_pos = asset_coords[int(nearest[idx])]
            ax = int(np.clip(round(asset_pos[0]), 0, self.shape[0] - 1))
            ay = int(np.clip(round(asset_pos[1]), 0, self.shape[1] - 1))
            asset_burning = p_init[ax, ay] >= float(asset_burning_threshold)
            if asset_burning:
                downwind = self._placement_downwind(
                    idx,
                    ctx,
                    offset_cells=downwind_offset_cells,
                    phi_jitter_rad=phi_jitter_rad,
                )
                if downwind is not None:
                    placements.append(downwind)
                else:
                    offset = float(tangent_offset_cells) + float(tangent_wind_scale) * float(ctx["wind_mag"])
                    placements.append(
                        self._placement_tangent(
                            idx,
                            ctx,
                            offset_cells=offset,
                            phi_jitter_rad=phi_jitter_rad,
                        )
                    )
                continue

            vec = asset_pos - inner_xy[idx]
            if float(np.linalg.norm(vec)) <= 1e-9:
                offset = float(tangent_offset_cells) + float(tangent_wind_scale) * float(ctx["wind_mag"])
                placements.append(
                    self._placement_tangent(
                        idx,
                        ctx,
                        offset_cells=offset,
                        phi_jitter_rad=phi_jitter_rad,
                    )
                )
                continue

            alpha = float(self.rng.uniform(lo, hi))
            pt = inner_xy[idx] + alpha * vec
            xg, yg = self.projector.snap(float(pt[0]), float(pt[1]))
            u = self._unit(vec)
            u_perp = self._unit(np.array([-u[1], u[0]], dtype=float))
            long_angle = float(np.arctan2(u_perp[1], u_perp[0]))
            phi = self._phi_from_long_axis_angle(long_angle)
            if phi_jitter_rad > 0.0:
                phi = self._wrap_angle(phi + float(self.rng.normal(0.0, phi_jitter_rad)))
            placements.append([xg, yg, phi])

        return placements

    def _heuristic_downwind_blocking(
        self,
        count: int,
        ctx: dict,
        *,
        offset_cells: float,
        head_align_threshold: float,
        phi_jitter_rad: float,
        min_arc_sep_frac: float,
        orientation: str,
        layer_spacing_cells: float,
        max_layers: int,
    ) -> list[list[float]]:
        if count <= 0 or ctx["wind_mag"] <= 1e-9:
            return []
        v_unit = ctx["v_unit"]
        wind_unit = ctx["wind_unit"]
        align = v_unit @ wind_unit
        head_idx = np.where(align >= float(head_align_threshold))[0]
        if head_idx.size == 0:
            return []
        K = v_unit.shape[0]
        min_arc = self._min_arc_for_count(K, count, min_arc_sep_frac)
        scores = np.zeros(K, dtype=float)
        scores[head_idx] = 1.0
        chosen = self._select_spaced_by_score(count, scores, min_arc=min_arc, idx_pool=head_idx)
        if not chosen:
            return []
        chosen = np.asarray(chosen, dtype=int)
        layer_spacing_cells = float(max(layer_spacing_cells, 0.0))
        max_layers = int(max(max_layers, 1))
        placements: list[list[float]] = []
        if count <= len(chosen):
            chosen = chosen[:count]
            for idx in chosen:
                downwind = self._placement_downwind(
                    int(idx),
                    ctx,
                    offset_cells=offset_cells,
                    phi_jitter_rad=phi_jitter_rad,
                    orientation=orientation,
                )
                if downwind is not None:
                    placements.append(downwind)
            return placements

        layers = int(np.ceil(float(count) / max(len(chosen), 1)))
        layers = min(layers, max_layers)
        base = chosen.tolist()
        for layer in range(layers):
            needed = count - len(placements)
            if needed <= 0:
                break
            offset = float(offset_cells) + float(layer) * layer_spacing_cells
            if layer > 0 and len(base) > 1:
                shift = int(self.rng.integers(0, len(base)))
                layer_idx = base[shift:] + base[:shift]
            else:
                layer_idx = base
            for idx in layer_idx[:needed]:
                downwind = self._placement_downwind(
                    int(idx),
                    ctx,
                    offset_cells=offset,
                    phi_jitter_rad=phi_jitter_rad,
                    orientation=orientation,
                )
                if downwind is not None:
                    placements.append(downwind)
        return placements

    def _heuristic_tangent_blocking(
        self,
        count: int,
        ctx: dict,
        *,
        offset_cells: float,
        wind_offset_scale: float,
        shoulder_align_range: tuple[float, float],
        phi_jitter_rad: float,
        orientation: str,
    ) -> list[list[float]]:
        if count <= 0:
            return []
        v_unit = ctx["v_unit"]
        wind_unit = ctx["wind_unit"]
        align = v_unit @ wind_unit
        lo, hi = shoulder_align_range
        lo = float(np.clip(lo, 0.0, 1.0))
        hi = float(np.clip(hi, 0.0, 1.0))
        if hi < lo:
            lo, hi = hi, lo
        if ctx["wind_mag"] <= 1e-9:
            shoulder_idx = np.arange(v_unit.shape[0], dtype=int)
        else:
            shoulder_idx = np.where((np.abs(align) >= lo) & (np.abs(align) <= hi))[0]
        if shoulder_idx.size == 0:
            return []
        chosen = self._select_evenly_spaced(count, shoulder_idx)
        offset = float(offset_cells) + float(wind_offset_scale) * float(ctx["wind_mag"])
        placements: list[list[float]] = []
        for idx in chosen:
            placements.append(
                self._placement_tangent(
                    int(idx),
                    ctx,
                    offset_cells=offset,
                    phi_jitter_rad=phi_jitter_rad,
                    orientation=orientation,
                )
            )
        return placements

    def _heuristic_connected_line(
        self,
        count: int,
        ctx: dict,
        *,
        offset_cells: float | None,
        overlap_frac: float,
        wind_bias: float,
        value_bias: float,
        control_bias: float,
        phi_jitter_rad: float,
        orientation: str,
        centered: bool,
    ) -> list[list[float]]:
        if count <= 0:
            return []
        inner_xy = ctx["inner_xy"]
        v_out = ctx["v_out"]
        v_unit = ctx["v_unit"]
        tangents = ctx["tangents"]
        normals = ctx["normals"]
        K = inner_xy.shape[0]
        if K == 0:
            return []

        dx = float(getattr(self.fire_model, "dx", 1.0))
        env = self.fire_model.env
        drop_h_cells = float(getattr(env, "drop_h_km", 0.0)) / max(dx, 1e-9)
        drop_w_cells = float(getattr(env, "drop_w_km", 0.0)) / max(dx, 1e-9)
        drop_h_cells = max(drop_h_cells, 1.0)
        drop_w_cells = max(drop_w_cells, 1.0)

        if offset_cells is None:
            offset_cells = 0.6 * drop_w_cells
        offset_cells = float(max(offset_cells, 0.0))

        overlap_frac = float(np.clip(overlap_frac, 0.0, 0.9))
        spacing_cells = max(0.5, drop_h_cells * (1.0 - overlap_frac))

        seg = np.linalg.norm(np.roll(inner_xy, -1, axis=0) - inner_xy, axis=1)
        mean_seg = float(np.mean(seg)) if seg.size else 1.0
        if mean_seg <= 1e-6:
            mean_seg = 1.0
        step = max(1, int(round(spacing_cells / mean_seg)))
        max_step = max(int(K / max(count, 1)), 1)
        step = min(step, max_step)

        scores = np.ones(K, dtype=float)
        if ctx["wind_mag"] > 1e-9 and float(wind_bias) != 0.0:
            align = v_unit @ ctx["wind_unit"]
            scores *= np.exp(np.clip(float(wind_bias) * np.clip(align, -1.0, 1.0), -20.0, 20.0))
        if float(value_bias) != 0.0:
            scores *= np.exp(np.clip(float(value_bias) * ctx["value_scores"], -20.0, 20.0))
        if float(control_bias) != 0.0:
            scores *= np.exp(np.clip(float(control_bias) * ctx["control_scores"], -20.0, 20.0))

        if centered:
            half = count // 2
            if count % 2 == 1:
                offsets = np.arange(-half, half + 1, dtype=int)
            else:
                offsets = np.arange(-half, half, dtype=int)
        else:
            offsets = np.arange(0, count, dtype=int)

        min_off = int(np.min(offsets)) * step
        max_off = int(np.max(offsets)) * step
        idx_pool = np.arange(K, dtype=int)
        valid = idx_pool[(idx_pool + min_off >= 0) & (idx_pool + max_off < K)]
        wrap = False
        if valid.size == 0:
            wrap = True
            valid = idx_pool

        w = scores.copy()
        w = w[valid]
        if float(w.sum()) <= 0.0:
            start_idx = int(self.rng.choice(valid))
        else:
            p = w / float(w.sum())
            start_idx = int(self.rng.choice(valid, p=p))

        placements: list[list[float]] = []
        orient = str(orientation).lower().strip()
        for off in offsets:
            idx = int(start_idx + int(off) * step)
            if wrap:
                idx %= K
            p0 = inner_xy[idx]
            v = v_out[idx]
            v_len = float(np.linalg.norm(v))
            if v_len > 1e-9 and offset_cells > 0.0:
                alpha = min(1.0, offset_cells / v_len)
                pt = p0 + v * alpha
            else:
                pt = p0
            xg, yg = self.projector.snap(float(pt[0]), float(pt[1]))

            if orient in ("normal", "perp", "perpendicular"):
                u_long = v_unit[idx]
                if float(np.linalg.norm(u_long)) <= 1e-9:
                    u_long = normals[idx]
            elif orient in ("wind", "downwind") and ctx["wind_mag"] > 1e-9:
                u_long = ctx["wind_unit"]
            else:
                u_long = tangents[idx]

            long_angle = float(np.arctan2(u_long[1], u_long[0]))
            phi = self._phi_from_long_axis_angle(long_angle)
            if phi_jitter_rad > 0.0:
                phi = self._wrap_angle(phi + float(self.rng.normal(0.0, phi_jitter_rad)))
            placements.append([xg, yg, phi])

        return placements

    def _heuristic_boundary(
        self,
        count: int,
        ctx: dict,
        *,
        alpha_range: tuple[float, float],
        wind_bias: float,
        value_bias: float,
        control_bias: float = 0.0,
        min_arc_sep_frac: float,
        wind_long_axis_blend: float,
        phi_jitter_rad: float,
        allowed_idx: np.ndarray | None = None,
    ) -> list[list[float]]:
        # Boundary-following heuristic sampling. 
        if count <= 0:
            return []
        inner_xy = ctx["inner_xy"]
        v_out = ctx["v_out"]
        tangents = ctx["tangents"]
        value_scores = ctx["value_scores"]
        control_scores = ctx["control_scores"]
        K = inner_xy.shape[0]
        idx_pool = np.arange(K, dtype=int) if allowed_idx is None else np.asarray(allowed_idx, dtype=int)
        if idx_pool.size == 0:
            return []

        scores = np.zeros(K, dtype=float)
        scores[idx_pool] = 1.0

        w_unit = ctx["wind_unit"]
        if ctx["wind_mag"] > 1e-9 and float(wind_bias) != 0.0:
            align = ctx["v_unit"] @ w_unit
            s = np.exp(np.clip(float(wind_bias) * np.clip(align, -1.0, 1.0), -20.0, 20.0))
            scores *= s

        if float(value_bias) != 0.0:
            s = np.exp(np.clip(float(value_bias) * value_scores, -20.0, 20.0))
            scores *= s

        if float(control_bias) != 0.0:
            s = np.exp(np.clip(float(control_bias) * control_scores, -20.0, 20.0))
            scores *= s

        min_arc = int(np.clip(np.floor(float(min_arc_sep_frac) * (K / max(self.n_drones, 1))), 1, max(K // 2, 1)))
        lo, hi = alpha_range
        lo = float(np.clip(lo, 0.0, 1.0))
        hi = float(np.clip(hi, 0.0, 1.0))
        if hi < lo:
            lo, hi = hi, lo

        placements: list[list[float]] = []
        available = scores > 0.0
        for _ in range(count):
            w = np.where(available, scores, 0.0)
            if float(w.sum()) <= 0.0:
                w = available.astype(float)
            if float(w.sum()) <= 0.0:
                idx = int(self.rng.choice(idx_pool))
            else:
                p = w / float(w.sum())
                idx = int(self.rng.choice(K, p=p))
            available[idx] = False
            for off in range(-min_arc, min_arc + 1):
                available[(idx + off) % K] = False

            p0 = inner_xy[idx]
            v = v_out[idx]
            if float(np.linalg.norm(v)) <= 1e-9:
                v = np.array([1.0, 0.0], dtype=float)
            alpha = float(self.rng.uniform(lo, hi))
            cand = p0 + alpha * v
            xg, yg = self.projector.snap(float(cand[0]), float(cand[1]))

            t = tangents[idx]
            u_long = self._unit(t)
            if wind_long_axis_blend > 0.0 and ctx["wind_mag"] > 1e-9:
                u_wperp = self._unit(np.array([-w_unit[1], w_unit[0]], dtype=float))
                u_mix = (1.0 - wind_long_axis_blend) * u_long + wind_long_axis_blend * u_wperp
                if float(np.linalg.norm(u_mix)) > 1e-9:
                    u_long = self._unit(u_mix)

            long_angle = float(np.arctan2(u_long[1], u_long[0]))
            phi = self._phi_from_long_axis_angle(long_angle)
            if phi_jitter_rad > 0.0:
                phi = self._wrap_angle(phi + float(self.rng.normal(0.0, phi_jitter_rad)))

            placements.append([xg, yg, phi])

        return placements

    def _heuristic_point_protection(
        self,
        count: int,
        ctx: dict,
        *,
        offset_cells: float,
        exclusion_cells: float,
        asset_wind_blend: float,
        min_value_quantile: float,
        pos_jitter_cells: float = 0.0,
        phi_jitter_rad: float = 0.0,
    ) -> list[list[float]]:
        if count <= 0:
            return []
        value_map = ctx["value_map"]
        nx, ny = value_map.shape
        flat = value_map.ravel()
        if float(np.max(flat)) <= 0.0:
            return []
        quantile = float(np.clip(min_value_quantile, 0.0, 1.0))
        thresh = np.quantile(flat, quantile) if quantile > 0.0 else np.min(flat)
        order = np.argsort(flat)[::-1]
        centers: list[np.ndarray] = []
        placements: list[list[float]] = []
        fire_center = ctx["init_centroid"]
        wind_unit = ctx["wind_unit"]
        asset_wind_blend = float(np.clip(asset_wind_blend, 0.0, 1.0))
        offset_cells = float(max(offset_cells, 0.0))
        pos_jitter_cells = float(max(pos_jitter_cells, 0.0))
        phi_jitter_rad = float(max(phi_jitter_rad, 0.0))

        for idx in order:
            if len(placements) >= count:
                break
            val = flat[idx]
            if val < thresh:
                break
            xi, yi = np.unravel_index(int(idx), (nx, ny))
            pos = np.array([float(xi), float(yi)], dtype=float)
            if any(np.linalg.norm(pos - c) < exclusion_cells for c in centers):
                continue

            fire_vec = self._unit(pos - fire_center)
            approach = self._unit(asset_wind_blend * wind_unit + (1.0 - asset_wind_blend) * fire_vec)
            if float(np.linalg.norm(approach)) <= 1e-9:
                approach = np.array([0.0, 1.0], dtype=float)
            target = pos - approach * offset_cells
            if pos_jitter_cells > 0.0:
                target = target + self.rng.normal(0.0, pos_jitter_cells, size=2)
            xg, yg = self.projector.snap(float(target[0]), float(target[1]))
            long_axis = np.array([-approach[1], approach[0]], dtype=float)
            phi = self._phi_from_long_axis_angle(float(np.arctan2(long_axis[1], long_axis[0])))
            if phi_jitter_rad > 0.0:
                phi = self._wrap_angle(phi + float(self.rng.normal(0.0, phi_jitter_rad)))
            placements.append([xg, yg, phi])
            centers.append(pos)

        return placements

    def _heuristic_head_flank(
        self,
        count: int,
        ctx: dict,
        *,
        head_frac: float,
        flank_frac: float,
        back_frac: float,
        alpha_range: tuple[float, float],
        **boundary_kwargs,
    ) -> list[list[float]]:
        if count <= 0:
            return []
        wind_unit = ctx["wind_unit"]
        v_unit = ctx["v_unit"]
        align = v_unit @ wind_unit
        head_idx = np.where(align >= 0.5)[0]
        back_idx = np.where(align <= -0.3)[0]
        flank_idx = np.where((np.abs(align) < 0.5))[0]
        modes = ["head", "flank", "back"]
        allocs = {"head": head_frac, "flank": flank_frac, "back": back_frac}
        counts = self._resolve_mode_counts(count, modes, allocs)

        placements: list[list[float]] = []
        pools = {"head": head_idx, "flank": flank_idx, "back": back_idx}
        surplus = 0
        for mode in modes:
            if pools[mode].size == 0:
                surplus += counts.get(mode, 0)
                counts[mode] = 0
        for mode in modes:
            if surplus <= 0:
                break
            if pools[mode].size > 0:
                counts[mode] += surplus
                surplus = 0
        for mode in modes:
            c = counts.get(mode, 0)
            if c <= 0:
                continue
            allowed_idx = pools[mode]
            placements.extend(
                self._heuristic_boundary(
                    c,
                    ctx,
                    alpha_range=alpha_range,
                    allowed_idx=allowed_idx,
                    **boundary_kwargs,
                )
            )
        return placements

    def _heuristic_confine(
        self,
        count: int,
        ctx: dict,
        *,
        offset_cells: float,
        pos_jitter_cells: float = 0.0,
        phi_jitter_rad: float = 0.0,
    ) -> list[list[float]]:
        if count <= 0:
            return []
        outer = ctx["outer_xy"]
        if outer.shape[0] < 4:
            return []
        center = np.mean(outer, axis=0)
        centered = outer - center
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)
        long_vec = self._unit(evecs[:, order[-1]])
        short_vec = self._unit(evecs[:, order[0]])
        scale = float(np.sqrt(max(evals[order[-1]], 1e-6)))
        t_vals = np.linspace(-1.0, 1.0, num=max(count, 1))
        placements: list[list[float]] = []
        offset_cells = float(offset_cells)
        pos_jitter_cells = float(max(pos_jitter_cells, 0.0))
        phi_jitter_rad = float(max(phi_jitter_rad, 0.0))
        for i, t in enumerate(t_vals[:count]):
            base = center + long_vec * (t * scale)
            direction = short_vec * ((-1) ** i) * offset_cells
            pt = base + direction
            if pos_jitter_cells > 0.0:
                pt = pt + self.rng.normal(0.0, pos_jitter_cells, size=2)
            xg, yg = self.projector.snap(float(pt[0]), float(pt[1]))
            phi = self._phi_from_long_axis_angle(float(np.arctan2(short_vec[1], short_vec[0])))
            if phi_jitter_rad > 0.0:
                phi = self._wrap_angle(phi + float(self.rng.normal(0.0, phi_jitter_rad)))
            placements.append([xg, yg, phi])
        return placements

    def _apply_effective_filter(
        self,
        params: np.ndarray,
        ctx: dict,
        *,
        min_final_prob: float,
        max_initial_prob: float,
        max_boundary_dist: float | None,
    ) -> np.ndarray:
        if params.size == 0:
            return params
        p_final = ctx["p_final"]
        p_init = ctx["p_init"]
        tree = ctx["boundary_tree"]
        keep = []
        nx, ny = self.shape
        for p in params:
            xi = int(np.clip(round(p[0]), 0, nx - 1))
            yi = int(np.clip(round(p[1]), 0, ny - 1))
            if p_final[xi, yi] < min_final_prob:
                continue
            if p_init[xi, yi] > max_initial_prob:
                continue
            if tree is not None and max_boundary_dist is not None:
                dist, _ = tree.query([p[0], p[1]])
                if dist > max_boundary_dist:
                    continue
            keep.append(p)
        if not keep:
            return np.empty((0, 3), dtype=float)
        return np.asarray(keep, dtype=float)

    def sample_local_theta_on_mask(
        self,
        anchors_theta: np.ndarray,
        n: int,
        *,
        sigma_cells: float = 3.0,
        sigma_phi_rad: float = np.deg2rad(15.0),
        resample_phi_prob: float = 0.05,
        resample_xy_prob: float = 0.0,
    ) -> np.ndarray:
        """
        Local refinement candidates around one or more anchors.

        Perturbs in *cell space* then snaps back to the valid mask (more effective than tiny
        perturbations in [0,1] when `decode_theta` causes discretisation/plateaus).
        """
        if self.projector is None or self.shape is None:
            raise RuntimeError("Call setup_search_grid(...) before local sampling.")

        anchors_theta = np.atleast_2d(np.asarray(anchors_theta, dtype=float))
        if anchors_theta.shape[0] < 1 or anchors_theta.shape[1] != self.dim:
            raise ValueError(f"anchors_theta must have shape (*,{self.dim}); got {anchors_theta.shape}")

        nx, ny = self.shape
        sigma_cells = float(max(sigma_cells, 0.0))
        sigma_phi_rad = float(max(sigma_phi_rad, 0.0))
        resample_phi_prob = float(np.clip(resample_phi_prob, 0.0, 1.0))
        resample_xy_prob = float(np.clip(resample_xy_prob, 0.0, 1.0))

        out = np.empty((max(int(n), 1), self.dim), dtype=float)

        for i in range(out.shape[0]):
            base = anchors_theta[int(self.rng.integers(0, anchors_theta.shape[0]))]
            params = self.decode_theta(base)  # (D,3) snapped, sorted

            theta = np.empty(self.dim, dtype=float)
            for d in range(self.n_drones):
                x0, y0, phi0 = params[d]

                if float(self.rng.random()) < resample_xy_prob:
                    xg, yg = self.projector.random_coords(self.rng, 1)[0]
                else:
                    x_cont = float(x0) + float(self.rng.normal(0.0, sigma_cells))
                    y_cont = float(y0) + float(self.rng.normal(0.0, sigma_cells))
                    xg, yg = self.projector.snap(x_cont, y_cont)

                if float(self.rng.random()) < resample_phi_prob:
                    phi = float(self.rng.random()) * (2.0 * np.pi)
                else:
                    phi = self._wrap_angle(float(phi0) + float(self.rng.normal(0.0, sigma_phi_rad)))

                theta[3 * d + 0] = float(xg) / max(nx - 1, 1)
                theta[3 * d + 1] = float(yg) / max(ny - 1, 1)
                theta[3 * d + 2] = float(phi) / (2.0 * np.pi)

            out[i] = np.clip(theta, 0.0, 1.0)

        return out[0] if n == 1 else out

    def sample_heuristic_theta(
        self,
        n: int = 1,
        *,
        alpha_range: tuple[float, float] = (0.35, 0.70),
        wind_bias: float = 2.0,
        value_bias: float = 0.0,
        wind_long_axis_blend: float = 0.25,
        phi_jitter_rad: float = np.deg2rad(10.0),
        min_arc_sep_frac: float = 0.25,
        heuristic_modes: list[str] | tuple[str, ...] | None = None,
        heuristic_allocations: dict | None = None,
        control_map: np.ndarray | None = None,
        control_bias: float = 2.0,
        asset_value_quantile: float = 0.90,
        asset_alpha_range: tuple[float, float] = (0.45, 0.70),
        asset_distance_scale_cells: float | None = None,
        asset_burning_threshold: float = 0.3,
        downwind_offset_cells: float = 6.0,
        downwind_head_align: float = 0.4,
        downwind_orientation: str = "perp",
        downwind_min_arc_sep_frac: float | None = None,
        downwind_layer_spacing_cells: float = 3.0,
        downwind_max_layers: int = 3,
        tangent_offset_cells: float = 3.0,
        tangent_wind_scale: float = 2.0,
        tangent_shoulder_align: tuple[float, float] = (0.1, 0.6),
        tangent_orientation: str = "normal",
        line_offset_cells: float | None = None,
        line_overlap_frac: float = 0.2,
        line_wind_bias: float = 2.0,
        line_value_bias: float = 0.0,
        line_control_bias: float = 0.0,
        line_orientation: str = "tangent",
        line_centered: bool = True,
        contingency_alpha_range: tuple[float, float] = (0.65, 0.95),
        point_offset_cells: float = 4.0,
        point_exclusion_cells: float = 6.0,
        point_asset_wind_blend: float = 0.5,
        point_value_quantile: float = 0.85,
        point_jitter_cells: float = 0.5,
        head_frac: float = 0.4,
        flank_frac: float = 0.4,
        back_frac: float = 0.2,
        confine_offset_cells: float = 3.0,
        confine_jitter_cells: float = 0.5,
        effective_min_final_prob: float = 0.2,
        effective_max_init_prob: float = 0.7,
        effective_max_boundary_dist: float | None = 6.0,
        score_alpha: float = 0.6,
    ):
        """
        Rich heuristic initialisation that can combine multiple strategies:
          - fire-asset blocking between the fire front and high-value cells
          - downwind blocking ahead of the head fire
          - tangent/shoulder blocking offset from the fire front
          - connected retardant line placements to build continuous barriers
          - boundary-based (wind-aware / value-aware, original behaviour)
          - POD/control-feature tie-in (control map biasing)
          - contingency-line allocation
          - point/zone protection from the value map
          - head/flank/back region allocation using wind alignment
          - confine-future-footprint placements using PCA of the outer boundary
          - effective-interaction filtering to reject implausible drops

        `heuristic_modes` controls which strategies are active (names are case-insensitive).
        """
        ctx = self._build_heuristic_context(control_map=control_map, score_alpha=score_alpha)
        default_modes = [
            "fire_asset_blocking",
            "downwind_blocking",
            "tangent_blocking",
        ]
        modes = heuristic_modes if heuristic_modes is not None else default_modes
        modes = [str(m).lower().strip() for m in modes if str(m).strip()]
        if not modes:
            modes = ["boundary"]
        alias_map = {
            "connected_barrier": "connected_line",
            "line_barrier": "connected_line",
            "retardant_line": "connected_line",
        }
        modes = [alias_map.get(m, m) for m in modes]

        effective_enabled = "effective_interaction" in modes
        generator_modes = [m for m in modes if m != "effective_interaction"]
        if not generator_modes:
            generator_modes = ["boundary"]
        counts = self._resolve_mode_counts(self.n_drones, list(generator_modes), heuristic_allocations or {})
        assigned = sum(counts.values())
        if assigned < self.n_drones:
            counts["boundary"] = counts.get("boundary", 0) + (self.n_drones - assigned)
            if "boundary" not in generator_modes:
                generator_modes.append("boundary")

        base_boundary_kwargs = dict(
            alpha_range=alpha_range,
            wind_bias=wind_bias,
            value_bias=value_bias,
            min_arc_sep_frac=min_arc_sep_frac,
            wind_long_axis_blend=float(np.clip(wind_long_axis_blend, 0.0, 1.0)),
            phi_jitter_rad=float(max(phi_jitter_rad, 0.0)),
        )

        n = max(int(n), 1)
        thetas = np.empty((n, self.dim), dtype=float)

        for i in range(n):
            placements: list[list[float]] = []
            for mode in generator_modes:
                cnt = counts.get(mode, 0)
                if cnt <= 0:
                    continue
                if mode == "boundary":
                    kwargs = dict(base_boundary_kwargs)
                    placements.extend(self._heuristic_boundary(cnt, ctx, control_bias=0.0, **kwargs))
                elif mode == "control_tie_in":
                    placements.extend(
                        self._heuristic_boundary(
                            cnt,
                            ctx,
                            control_bias=control_bias,
                            **dict(base_boundary_kwargs),
                        )
                    )
                elif mode == "contingency":
                    placements.extend(
                        self._heuristic_boundary(
                            cnt,
                            ctx,
                            control_bias=control_bias,
                            **dict(base_boundary_kwargs, alpha_range=contingency_alpha_range),
                        )
                    )
                elif mode == "point_protection":
                    placements.extend(
                        self._heuristic_point_protection(
                            cnt,
                            ctx,
                            offset_cells=point_offset_cells,
                            exclusion_cells=point_exclusion_cells,
                            asset_wind_blend=point_asset_wind_blend,
                            min_value_quantile=point_value_quantile,
                            pos_jitter_cells=point_jitter_cells,
                            phi_jitter_rad=phi_jitter_rad,
                        )
                    )
                elif mode == "head_flank":
                    head_kwargs = dict(base_boundary_kwargs)
                    alpha_h = head_kwargs.pop("alpha_range", alpha_range)
                    placements.extend(
                        self._heuristic_head_flank(
                            cnt,
                            ctx,
                            head_frac=head_frac,
                            flank_frac=flank_frac,
                            back_frac=back_frac,
                            alpha_range=alpha_h,
                            **head_kwargs,
                        )
                    )
                elif mode == "confine":
                    placements.extend(
                        self._heuristic_confine(
                            cnt,
                            ctx,
                            offset_cells=confine_offset_cells,
                            pos_jitter_cells=confine_jitter_cells,
                            phi_jitter_rad=phi_jitter_rad,
                        )
                    )
                elif mode == "fire_asset_blocking":
                    placements.extend(
                        self._heuristic_fire_asset_blocking(
                            cnt,
                            ctx,
                            asset_value_quantile=asset_value_quantile,
                            asset_alpha_range=asset_alpha_range,
                            asset_distance_scale_cells=asset_distance_scale_cells,
                            asset_burning_threshold=asset_burning_threshold,
                            min_arc_sep_frac=min_arc_sep_frac,
                            downwind_offset_cells=downwind_offset_cells,
                            tangent_offset_cells=tangent_offset_cells,
                            tangent_wind_scale=tangent_wind_scale,
                            phi_jitter_rad=phi_jitter_rad,
                        )
                    )
                elif mode == "downwind_blocking":
                    dw_min_arc = min_arc_sep_frac if downwind_min_arc_sep_frac is None else downwind_min_arc_sep_frac
                    placements.extend(
                        self._heuristic_downwind_blocking(
                            cnt,
                            ctx,
                            offset_cells=downwind_offset_cells,
                            head_align_threshold=downwind_head_align,
                            phi_jitter_rad=phi_jitter_rad,
                            min_arc_sep_frac=dw_min_arc,
                            orientation=downwind_orientation,
                            layer_spacing_cells=downwind_layer_spacing_cells,
                            max_layers=downwind_max_layers,
                        )
                    )
                elif mode == "tangent_blocking":
                    placements.extend(
                        self._heuristic_tangent_blocking(
                            cnt,
                            ctx,
                            offset_cells=tangent_offset_cells,
                            wind_offset_scale=tangent_wind_scale,
                            shoulder_align_range=tangent_shoulder_align,
                            phi_jitter_rad=phi_jitter_rad,
                            orientation=tangent_orientation,
                        )
                    )
                elif mode == "connected_line":
                    placements.extend(
                        self._heuristic_connected_line(
                            cnt,
                            ctx,
                            offset_cells=line_offset_cells,
                            overlap_frac=line_overlap_frac,
                            wind_bias=line_wind_bias,
                            value_bias=line_value_bias,
                            control_bias=line_control_bias,
                            phi_jitter_rad=phi_jitter_rad,
                            orientation=line_orientation,
                            centered=line_centered,
                        )
                    )
                else:
                    # Unknown mode fallback: boundary drop.
                    placements.extend(self._heuristic_boundary(cnt, ctx, control_bias=0.0, **dict(base_boundary_kwargs)))

            if len(placements) < self.n_drones:
                fallback = self._heuristic_boundary(
                    self.n_drones - len(placements), ctx, control_bias=0.0, **dict(base_boundary_kwargs)
                )
                placements.extend(fallback)

            if not placements:
                params = self._random_params_on_mask(self.n_drones)
            else:
                params = np.asarray(placements, dtype=float)

            if params.shape[0] > self.n_drones:
                idx = self.rng.permutation(params.shape[0])[: self.n_drones]
                params = params[idx]
            elif params.shape[0] < self.n_drones:
                extra = self._heuristic_boundary(
                    self.n_drones - params.shape[0], ctx, control_bias=0.0, **dict(base_boundary_kwargs)
                )
                params = np.vstack([params, np.asarray(extra, dtype=float)]) if extra else params

            if effective_enabled:
                filtered = self._apply_effective_filter(
                    params,
                    ctx,
                    min_final_prob=effective_min_final_prob,
                    max_initial_prob=effective_max_init_prob,
                    max_boundary_dist=effective_max_boundary_dist,
                )
                params = filtered
                attempts = 0
                while params.shape[0] < self.n_drones and attempts < 3:
                    need = (self.n_drones - params.shape[0]) * 2
                    extra = self._heuristic_boundary(need, ctx, control_bias=0.0, **dict(base_boundary_kwargs))
                    if not extra:
                        break
                    params = np.vstack([params, np.asarray(extra, dtype=float)])
                    params = self._apply_effective_filter(
                        params,
                        ctx,
                        min_final_prob=effective_min_final_prob,
                        max_initial_prob=effective_max_init_prob,
                        max_boundary_dist=effective_max_boundary_dist,
                    )
                    attempts += 1
                if params.shape[0] < self.n_drones:
                    extra = self._random_params_on_mask(self.n_drones - params.shape[0])
                    params = np.vstack([params, extra])

            if params.shape[0] > self.n_drones:
                idx = self.rng.permutation(params.shape[0])[: self.n_drones]
                params = params[idx]
            elif params.shape[0] < self.n_drones:
                extra = self._random_params_on_mask(self.n_drones - params.shape[0])
                params = np.vstack([params, extra])

            thetas[i] = self._encode_params(params)

        return thetas[0] if n == 1 else thetas

    def sample_initial_thetas(
        self,
        n_init: int,
        *,
        strategy: str = "random",
        heuristic_random_frac: float = 0.2,
        heuristic_kwargs: dict | None = None,
    ) -> np.ndarray:
        """
        Strategy for choosing initial points before BO:
          - `random`: existing behaviour (uniform over [0,1]^dim, snapped to mask)
          - `random_mask`: uniform over valid cells + random phi
          - `heuristic`: boundary-aware placement with optional random_mask mixing
        """
        strategy = str(strategy).lower().strip()
        if n_init <= 0:
            raise ValueError("n_init must be >= 1")

        if strategy == "random":
            return self.sample_random_theta(n_init)
        if strategy == "random_mask":
            return self.sample_random_theta_on_mask(n_init)
        if strategy != "heuristic":
            raise ValueError("Unknown init strategy. Use 'random', 'random_mask', or 'heuristic'.")

        heuristic_kwargs = {} if heuristic_kwargs is None else dict(heuristic_kwargs)
        heuristic_random_frac = float(np.clip(heuristic_random_frac, 0.0, 1.0))

        n_rand = int(np.round(n_init * heuristic_random_frac))
        n_heur = int(n_init - n_rand)
        if n_heur <= 0:
            return self.sample_random_theta_on_mask(n_init)

        A = self.sample_heuristic_theta(n_heur, **heuristic_kwargs)
        A = np.atleast_2d(A)
        if n_rand > 0:
            B = self.sample_random_theta_on_mask(n_rand)
            B = np.atleast_2d(B)
            X = np.vstack([A, B])
        else:
            X = A

        order = self.rng.permutation(X.shape[0])
        return X[order]

    def sample_random_theta_polar(self, n: int = 1):
        if n == 1:
            return self.rng.random(self.dim)
        return self.rng.random((n, self.dim))

    def sample_local_theta_polar(
        self,
        anchors_theta: np.ndarray,
        n: int,
        *,
        sigma_s: float = 0.05,
        sigma_r: float = 0.05,
        sigma_delta_rad: float = np.deg2rad(15.0),
        resample_delta_prob: float = 0.05,
    ) -> np.ndarray:
        anchors_theta = np.atleast_2d(np.asarray(anchors_theta, dtype=float))
        if anchors_theta.shape[0] < 1 or anchors_theta.shape[1] != self.dim:
            raise ValueError(f"anchors_theta must have shape (*,{self.dim}); got {anchors_theta.shape}")

        sigma_s = float(max(sigma_s, 0.0))
        sigma_r = float(max(sigma_r, 0.0))
        sigma_delta_rad = float(max(sigma_delta_rad, 0.0))
        resample_delta_prob = float(np.clip(resample_delta_prob, 0.0, 1.0))

        out = np.empty((max(int(n), 1), self.dim), dtype=float)
        for i in range(out.shape[0]):
            base = anchors_theta[int(self.rng.integers(0, anchors_theta.shape[0]))]
            theta = np.empty(self.dim, dtype=float)
            for d in range(self.n_drones):
                s0 = float(base[3 * d + 0]) + float(self.rng.normal(0.0, sigma_s))
                r0 = float(base[3 * d + 1]) + float(self.rng.normal(0.0, sigma_r))

                if float(self.rng.random()) < resample_delta_prob:
                    delta = float(self.rng.random()) * (2.0 * np.pi)
                else:
                    delta0 = float(base[3 * d + 2]) * (2.0 * np.pi)
                    delta = self._wrap_angle(delta0 + float(self.rng.normal(0.0, sigma_delta_rad)))

                theta[3 * d + 0] = np.clip(s0, 0.0, 1.0)
                theta[3 * d + 1] = np.clip(r0, 0.0, 1.0)
                theta[3 * d + 2] = delta / (2.0 * np.pi)
            out[i] = np.clip(theta, 0.0, 1.0)

        return out[0] if n == 1 else out

    def _encode_sr_params(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        if params.ndim != 2 or params.shape[1] != 3:
            raise ValueError(f"Expected SR params with shape (D,3); got {params.shape}")
        theta = np.empty(self.dim, dtype=float)
        for d, (s, r, delta) in enumerate(params):
            theta[3 * d + 0] = float(np.clip(s, 0.0, 1.0))
            theta[3 * d + 1] = float(np.clip(r, 0.0, 1.0))
            theta[3 * d + 2] = self._wrap_angle(float(delta)) / (2.0 * np.pi)
        return np.clip(theta, 0.0, 1.0)

    def decode_theta_polar(self, theta: np.ndarray) -> np.ndarray:
        if self.sr_grid is None:
            raise RuntimeError("Call setup_search_grid_polar(...) before decode_theta_polar.")

        theta = np.asarray(theta, dtype=float)
        params = []
        for d in range(self.n_drones):
            s = float(theta[3 * d + 0])
            r = float(theta[3 * d + 1])
            delta = float(theta[3 * d + 2]) * (2.0 * np.pi)

            xy, phi_s = self._sr_lookup(s, r)
            phi_long = phi_s + delta
            phi = self._phi_from_long_axis_angle(phi_long)
            params.append((float(xy[0]), float(xy[1]), float(phi)))

        params = np.array(params, dtype=float)
        order = np.lexsort((params[:, 2], params[:, 1], params[:, 0]))
        params = params[order]
        return params

    def theta_to_gp_features_polar(self, theta: np.ndarray) -> np.ndarray:
        if self.sr_grid is None:
            raise RuntimeError("Call setup_search_grid_polar(...) before theta_to_gp_features_polar.")
        params = self.decode_theta_polar(theta)
        nx, ny = self.shape if self.shape is not None else self.fire_model.env.grid_size
        x = params[:, 0] / max(nx - 1, 1)
        y = params[:, 1] / max(ny - 1, 1)
        phi = params[:, 2]
        feats = np.stack([x, y, np.sin(phi), np.cos(phi)], axis=1)
        return feats.reshape(-1)

    def _simulate_firestate_with_params(
        self,
        drone_params: np.ndarray,
        *,
        n_sims: int | None = None,
        seed: int | None = None,
    ) -> FireState:
        n_sims = self.n_sims if n_sims is None else int(n_sims)
        return self.fire_model.simulate_from_firestate(
            self.init_firestate,
            T=self.evolution_time_s,
            n_sims=n_sims,
            drone_params=drone_params,
            ros_mps=self.fire_model.env.ros_mps,
            wind_coeff=self.fire_model.env.wind_coeff,
            diag=self.fire_model.env.diag,
            seed=seed,
            avoid_burning_drop=self.fire_model.env.avoid_burning_drop,
            burning_prob_threshold=self.fire_model.env.avoid_drop_p_threshold,
        )

    def _expected_value_from_firestate(self, evolved_firestate: FireState) -> float:
        p_burning = evolved_firestate.burning[0].astype(float, copy=False)
        p_burned = evolved_firestate.burned[0].astype(float, copy=False)
        p_affected = np.clip(p_burning + p_burned, 0.0, 1.0)

        nx, _ = self.fire_model.env.grid_size
        dx = self.fire_model.env.domain_km / nx
        expected_value_burned = np.sum(p_affected * self.fire_model.env.value) * (dx ** 2)
        return float(expected_value_burned)

    def expected_value_burned_area_polar(self, theta: np.ndarray) -> float:
        drone_params = self.decode_theta_polar(theta)
        evolved_firestate = self._simulate_firestate_with_params(drone_params)
        return self._expected_value_from_firestate(evolved_firestate)

    @staticmethod
    def _sr_arc_length(boundary_xy: np.ndarray) -> np.ndarray:
        seg = np.diff(np.vstack([boundary_xy, boundary_xy[0]]), axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        return np.concatenate([[0.0], np.cumsum(seg_len[:-1])])

    @staticmethod
    def _sr_restrict_by_compactness(scores: np.ndarray, arc_len: np.ndarray, compactness: float) -> np.ndarray:
        compactness = float(np.clip(compactness, 0.0, 1.0))
        if compactness >= 1.0:
            return scores
        best = int(np.nanargmax(scores))
        total = float(arc_len[-1])
        if total <= 0.0:
            return scores
        span = compactness * total
        out = np.full_like(scores, -np.inf, dtype=float)
        for i in range(len(scores)):
            d = abs(arc_len[i] - arc_len[best])
            d = min(d, total - d)
            if d <= span:
                out[i] = scores[i]
        return out

    @staticmethod
    def _sr_greedy_select_indices(
        scores: np.ndarray,
        arc_len: np.ndarray,
        min_spacing: float,
        n_keep: int,
    ) -> list[int]:
        order = np.argsort(scores)[::-1]
        selected: list[int] = []
        total = float(arc_len[-1])
        if total <= 0.0:
            return [int(i) for i in order[: max(int(n_keep), 0)]]
        for idx in order:
            if not np.isfinite(scores[idx]):
                continue
            ok = True
            for sidx in selected:
                d = abs(arc_len[idx] - arc_len[sidx])
                d = min(d, total - d)
                if d < min_spacing:
                    ok = False
                    break
            if ok:
                selected.append(int(idx))
            if len(selected) >= n_keep:
                break
        return selected

    @staticmethod
    def _sr_indices_by_arclength(arc_len: np.ndarray, n_keep: int) -> list[int]:
        total = float(arc_len[-1])
        if total <= 0.0:
            return [0 for _ in range(max(int(n_keep), 0))]
        targets = np.linspace(0.0, total, int(n_keep), endpoint=False)
        idx = np.searchsorted(arc_len, targets)
        idx = np.clip(idx, 0, len(arc_len) - 1)
        return [int(i) for i in idx]

    def _sr_compact_scores(self, scores: np.ndarray, compactness: float) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        compactness = float(np.clip(compactness, 0.0, 1.0))
        if compactness >= 1.0 or scores.size == 0:
            return scores
        best = int(np.nanargmax(scores))
        span = int(max(1, np.floor(compactness * scores.size)))
        idx = np.arange(scores.size)
        dist = np.minimum((idx - best) % scores.size, (best - idx) % scores.size)
        out = np.full_like(scores, -np.inf, dtype=float)
        out[dist <= span] = scores[dist <= span]
        return out

    def _sr_select_indices(self, scores: np.ndarray, count: int, min_arc_sep_frac: float) -> list[int]:
        count = int(max(count, 0))
        if count == 0:
            return []
        min_arc = self._min_arc_for_count(len(scores), count, min_arc_sep_frac)
        return self._select_spaced_by_score(count, scores, min_arc=min_arc)

    def _sr_value_blocking(
        self,
        *,
        r_value: float,
        count: int,
        compactness: float,
        min_spacing: float | None = None,
        min_arc_sep_frac: float | None = None,
    ) -> list[tuple[float, float, float]]:
        if self.sr_grid is None or self.sr_r_targets is None or self.init_boundary is None:
            raise RuntimeError("Call setup_search_grid_polar(...) before SR heuristics.")

        r_idx = int(round(float(r_value) * (self.sr_grid.shape[1] - 1)))
        scores = np.zeros(self.sr_grid.shape[0], dtype=float)
        for i in range(self.sr_grid.shape[0]):
            p = self.sr_grid[i, r_idx]
            if not np.isfinite(p[0]):
                scores[i] = -np.inf
                continue
            xi = int(np.clip(round(p[0]), 0, self.shape[0] - 1))
            yi = int(np.clip(round(p[1]), 0, self.shape[1] - 1))
            scores[i] = float(self.fire_model.env.value[xi, yi])

        arc_len = self._sr_arc_length(np.asarray(self.init_boundary.xy, dtype=float))
        if min_spacing is None:
            frac = 0.0 if min_arc_sep_frac is None else float(min_arc_sep_frac)
            total = float(arc_len[-1])
            per = total / max(count, 1) if total > 0.0 else 0.0
            min_spacing = frac * per
        scores = self._sr_restrict_by_compactness(scores, arc_len, compactness)
        idx = self._sr_greedy_select_indices(scores, arc_len, float(min_spacing), int(count))
        return [(i / float(self.sr_grid.shape[0]), float(r_value), 0.0) for i in idx]

    def _sr_downwind_blocking(
        self,
        *,
        r_value: float,
        count: int,
        compactness: float,
        min_spacing: float | None = None,
        min_arc_sep_frac: float | None = None,
    ) -> list[tuple[float, float, float]]:
        if self.init_boundary is None or self.sr_grid is None:
            raise RuntimeError("Call setup_search_grid_polar(...) before SR heuristics.")
        wind = self._estimate_mean_wind()
        wind_unit = self._unit(wind)
        inner_xy = np.asarray(self.init_boundary.xy, dtype=float)
        scores = inner_xy @ wind_unit
        arc_len = self._sr_arc_length(inner_xy)
        if min_spacing is None:
            frac = 0.0 if min_arc_sep_frac is None else float(min_arc_sep_frac)
            total = float(arc_len[-1])
            per = total / max(count, 1) if total > 0.0 else 0.0
            min_spacing = frac * per
        scores = self._sr_restrict_by_compactness(scores, arc_len, compactness)
        idx = self._sr_greedy_select_indices(scores, arc_len, float(min_spacing), int(count))
        return [(i / float(self.sr_grid.shape[0]), float(r_value), 0.0) for i in idx]

    def _sr_surrounding(
        self,
        *,
        r_value: float,
        count: int,
        grouping: int,
        group_span_frac: float,
    ) -> list[tuple[float, float, float]]:
        if self.init_boundary is None or self.sr_grid is None:
            raise RuntimeError("Call setup_search_grid_polar(...) before SR heuristics.")
        grouping = int(max(grouping, 1))
        K = self.sr_grid.shape[0]
        arc_len = self._sr_arc_length(np.asarray(self.init_boundary.xy, dtype=float))

        if grouping == 1:
            idx = self._sr_indices_by_arclength(arc_len, max(count, 1))
            return [(int(i) / float(K), float(r_value), 0.0) for i in idx]

        centers = self._sr_indices_by_arclength(arc_len, grouping)
        counts = [count // grouping] * grouping
        for i in range(count % grouping):
            counts[i] += 1

        out: list[tuple[float, float, float]] = []
        total = float(arc_len[-1])
        span = float(group_span_frac) * total if total > 0.0 else 0.0
        for center, ccount in zip(centers, counts):
            if ccount <= 0:
                continue
            targets = np.linspace(-span / 2.0, span / 2.0, num=ccount, endpoint=False)
            for t in targets:
                if total <= 0.0:
                    idx = 0
                else:
                    target_len = (arc_len[center] + t) % total
                    idx = int(np.searchsorted(arc_len, target_len))
                    idx = min(idx, len(arc_len) - 1)
                out.append((idx / float(K), float(r_value), 0.0))
        return out

    def sample_initial_thetas_polar(
        self,
        n_init: int,
        *,
        heuristic_random_frac: float = 0.2,
        heuristic_allocations: dict | None = None,
        r_value: float = 0.6,
        r_downwind: float = 0.8,
        r_surround: float = 0.35,
        compact_value: float = 0.4,
        compact_downwind: float = 0.4,
        min_arc_sep_frac: float = 0.25,
        min_spacing: float | None = None,
        surrounding_grouping: int = 2,
        surrounding_group_span: float = 0.2,
        mix_prob: float = 0.35,
        single_strategy_probs: dict | None = None,
        random_mix_allocations: bool = True,
        plot_runs: bool = False,
        plot_n_sims: int | None = None,
        plot_every: int = 1,
        plot_title_prefix: str | None = None,
    ) -> np.ndarray:
        if n_init <= 0:
            raise ValueError("n_init must be >= 1")
        if self.sr_grid is None:
            raise RuntimeError("Call setup_search_grid_polar(...) before polar sampling.")

        heuristic_random_frac = float(np.clip(heuristic_random_frac, 0.0, 1.0))
        n_rand = int(np.round(n_init * heuristic_random_frac))
        n_heur = int(n_init - n_rand)

        modes = ["value_blocking", "downwind_blocking", "surrounding"]
        mix_prob = float(np.clip(mix_prob, 0.0, 1.0))
        if single_strategy_probs is None:
            probs = np.full(len(modes), 1.0 / len(modes), dtype=float)
        else:
            probs = np.array([float(single_strategy_probs.get(m, 0.0)) for m in modes], dtype=float)
            total = float(np.sum(probs))
            probs = probs / total if total > 0.0 else np.full(len(modes), 1.0 / len(modes), dtype=float)

        thetas = []
        for _ in range(max(n_heur, 0)):
            if float(self.rng.random()) < mix_prob:
                if heuristic_allocations is None and random_mix_allocations:
                    weights = self.rng.random(len(modes))
                    allocations = {m: float(w) for m, w in zip(modes, weights)}
                else:
                    allocations = heuristic_allocations or {}
                counts = self._resolve_mode_counts(self.n_drones, modes, allocations)
            else:
                chosen = modes[int(self.rng.choice(len(modes), p=probs))]
                counts = self._resolve_mode_counts(self.n_drones, modes, {chosen: 1.0})

            placements: list[tuple[float, float, float]] = []
            placements.extend(
                self._sr_value_blocking(
                    r_value=r_value,
                    count=counts.get("value_blocking", 0),
                    compactness=compact_value,
                    min_arc_sep_frac=min_arc_sep_frac,
                    min_spacing=min_spacing,
                )
            )
            placements.extend(
                self._sr_downwind_blocking(
                    r_value=r_downwind,
                    count=counts.get("downwind_blocking", 0),
                    compactness=compact_downwind,
                    min_arc_sep_frac=min_arc_sep_frac,
                    min_spacing=min_spacing,
                )
            )
            placements.extend(
                self._sr_surrounding(
                    r_value=r_surround,
                    count=counts.get("surrounding", 0),
                    grouping=surrounding_grouping,
                    group_span_frac=surrounding_group_span,
                )
            )

            if not placements:
                theta = self.sample_random_theta_polar(1)
            else:
                params = np.asarray(placements, dtype=float)
                if params.shape[0] > self.n_drones:
                    idx = self.rng.permutation(params.shape[0])[: self.n_drones]
                    params = params[idx]
                elif params.shape[0] < self.n_drones:
                    need = self.n_drones - params.shape[0]
                    extra = self.rng.random((need, 3))
                    extra[:, 2] *= 2.0 * np.pi
                    params = np.vstack([params, extra])
                theta = self._encode_sr_params(params)
            thetas.append(theta)

        if n_rand > 0:
            rand = np.atleast_2d(self.sample_random_theta_polar(n_rand))
            if thetas:
                X = np.vstack([np.vstack(thetas), rand])
            else:
                X = rand
        else:
            X = np.vstack(thetas) if thetas else np.atleast_2d(self.sample_random_theta_polar(1))

        order = self.rng.permutation(X.shape[0])
        X = X[order]

        if plot_runs:
            total = X.shape[0]
            plot_every = max(int(plot_every), 1)
            for i, theta in enumerate(X, start=1):
                if (i % plot_every) != 0:
                    continue
                objective = self.plot_evolved_firestate_polar(
                    theta,
                    n_sims=plot_n_sims,
                    title_prefix=plot_title_prefix,
                    run_idx=i,
                    run_total=total,
                )
                print(f"[Init polar] run {i:03d}/{total} | objective={objective:.6g}")

        return X

    def decode_theta(self, theta):
        if self.projector is None or self.shape is None:
            raise RuntimeError("Call setup_search_grid(...) before decode_theta / optimisation.")

        theta = np.asarray(theta, dtype=float)
        nx, ny = self.shape

        params = []
        for d in range(self.n_drones):
            tx = theta[3 * d + 0]
            ty = theta[3 * d + 1]
            tphi = theta[3 * d + 2]

            x_cont = tx * (nx - 1)
            y_cont = ty * (ny - 1)
            xg, yg = self.projector.snap(x_cont, y_cont)
            phi = tphi * (2.0 * np.pi)
            params.append((xg, yg, phi))

        params = np.array(params, dtype=float)
        order = np.lexsort((params[:, 2], params[:, 1], params[:, 0]))
        params = params[order]
        return params

    def _encode_params(self, params: np.ndarray) -> np.ndarray:
        """
        Helper to convert snapped (x,y,phi) tuples back into theta space.
        """
        if self.shape is None:
            raise RuntimeError("Call setup_search_grid(...) before encoding parameters.")
        params = np.asarray(params, dtype=float)
        if params.ndim != 2 or params.shape[1] != 3:
            raise ValueError(f"Expected params with shape (D,3); got {params.shape}")

        nx, ny = self.shape
        theta = np.empty(self.dim, dtype=float)
        for d, (xg, yg, phi) in enumerate(params):
            theta[3 * d + 0] = float(xg) / max(nx - 1, 1)
            theta[3 * d + 1] = float(yg) / max(ny - 1, 1)
            theta[3 * d + 2] = self._wrap_angle(float(phi)) / (2.0 * np.pi)
        return np.clip(theta, 0.0, 1.0)

    def theta_to_gp_features(self, theta: np.ndarray) -> np.ndarray:
        if self.shape is None:
            raise RuntimeError("Call setup_search_grid(...) before theta_to_gp_features.")

        nx, ny = self.shape
        params = self.decode_theta(theta)

        x = params[:, 0] / max(nx - 1, 1)
        y = params[:, 1] / max(ny - 1, 1)
        phi = params[:, 2]

        feats = np.stack([x, y, np.sin(phi), np.cos(phi)], axis=1)
        return feats.reshape(-1)

    def expected_value_burned_area(self, theta: np.ndarray) -> float:
        drone_params = self.decode_theta(theta)
        print("drone_params:", drone_params)
        evolved_firestate = self._simulate_firestate_with_params(drone_params)
        return self._expected_value_from_firestate(evolved_firestate)

    def run_bayes_opt(
        self,
        n_init: int = 10,
        n_iters: int = 30,
        n_candidates: int = 5000,
        xi: float = 0.01,
        K_grid: int = 500,
        boundary_field: str = "affected",
        verbose: bool = True,
        print_every: int = 1,
        use_ard_kernel: bool = False,
        init_strategy: str = "random",  # "random_mask", "heuristic"
        init_heuristic_random_frac: float = 0.2, # fraction of heuristic init points to random
        init_heuristic_kwargs: dict | None = None, # passed to sample_heuristic_theta 
        candidate_strategy: str = "random", # "random_mask", "qmc", "mixed"
        candidate_qmc: str = "sobol", # "sobol", "halton", "lhs"
        candidate_global_masked: bool = False, # only for "mixed" strategy (if True: global half is random_mask - cell uniform, else: global half is QMC in [0,1]^dim)
        candidate_local_frac: float = 0.5, # only for "mixed" strategy (fraction of candidates allocated to local refinment. Higher = more exploitative)
        candidate_local_top_k: int = 3, # only for "mixed" strategy (number of best points to use as anchors for local refinement)
        candidate_local_sigma_cells: float = 3.0, # only for "mixed" strategy (stddev of local (x,y) refinements in cell space before snapping to mask)
        candidate_local_sigma_phi_rad: float = np.deg2rad(15.0), # only for "mixed" strategy (stddev of local phi refinements in radians)
        candidate_local_resample_phi_prob: float = 0.05, # only for "mixed" strategy (probability of resampling phi locally)
    ):
        if self.projector is None:
            self.setup_search_grid(K=K_grid, boundary_field=boundary_field)
            if verbose:
                print(f"[BO] Search grid set up with {len(self.projector.coords)} valid cells in grid.")
                self.fire_model.plot_search_domain(self.search_domain_mask, title="Current Search Domain:")

        X_theta = self.sample_initial_thetas(
            n_init=n_init,
            strategy=init_strategy,
            heuristic_random_frac=init_heuristic_random_frac,
            heuristic_kwargs=init_heuristic_kwargs,
        )
        X_theta = np.atleast_2d(X_theta)
        y = np.array([self.expected_value_burned_area(th) for th in X_theta], dtype=float)
        X = np.vstack([self.theta_to_gp_features(th) for th in X_theta])

        if verbose:
            best0 = float(np.min(y))
            print(f"[BO] init: n_init={n_init}, dim={self.dim}, n_cells={len(self.projector.coords)}")
            print(f"[BO] init: best_y={best0:.6g}, mean_y={float(np.mean(y)):.6g}, std_y={float(np.std(y)):.6g}")

        if use_ard_kernel:
            base_kernel = Matern(
                length_scale=np.ones(X.shape[1], dtype=float),
                nu=2.5,
                length_scale_bounds=(1e-3, 1e3),
            )
        else:
            base_kernel = TiedXYFiMatern(lx=0.2, ly=0.2, lphi=0.5, nu=2.5, length_scale_bounds=(1e-3, 1e3))

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * base_kernel
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2))
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=None,
        )

        y_nexts = []
        y_bests = []

        for it in range(1, n_iters + 1):
            gp.fit(X, y)

            cstrat = str(candidate_strategy).lower().strip()
            if cstrat == "mixed":
                local_frac = float(np.clip(candidate_local_frac, 0.0, 1.0))
                n_local = int(np.round(n_candidates * local_frac))
                n_global = int(n_candidates - n_local)

                globals_ = (
                    self.sample_random_theta_on_mask(n_global)
                    if candidate_global_masked
                    else self.sample_qmc_theta(n_global, method=candidate_qmc)
                )

                top_k = int(np.clip(int(candidate_local_top_k), 1, len(y)))
                anchor_idx = np.argsort(y)[:top_k]
                anchors = X_theta[anchor_idx]
                locals_ = self.sample_local_theta_on_mask(
                    anchors,
                    n_local,
                    sigma_cells=candidate_local_sigma_cells,
                    sigma_phi_rad=candidate_local_sigma_phi_rad,
                    resample_phi_prob=candidate_local_resample_phi_prob,
                )

                Xcand_theta = np.vstack([np.atleast_2d(globals_), np.atleast_2d(locals_)])
                Xcand_theta = Xcand_theta[self.rng.permutation(Xcand_theta.shape[0])]
            elif cstrat == "random_mask":
                Xcand_theta = self.sample_random_theta_on_mask(n_candidates)
            elif cstrat == "qmc":
                Xcand_theta = self.sample_qmc_theta(n_candidates, method=candidate_qmc)
            else:
                Xcand_theta = self.sample_random_theta(n_candidates)
            Xcand = np.vstack([self.theta_to_gp_features(th) for th in Xcand_theta])

            y_best = float(np.min(y))
            ei = expected_improvement(Xcand, gp, y_best=y_best, xi=xi)

            best_ei_idx = int(np.argmax(ei))
            theta_next = Xcand_theta[best_ei_idx]
            x_next = Xcand[best_ei_idx]

            mu_next, std_next = gp.predict(x_next[None, :], return_std=True)
            mu_next = float(mu_next[0])
            std_next = float(std_next[0])

            y_next = float(self.expected_value_burned_area(theta_next))

            X_theta = np.vstack([X_theta, theta_next])
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)

            if verbose and (it % max(print_every, 1) == 0 or it == 1 or it == n_iters):
                improved = y_next < y_best
                ei_max = float(ei[best_ei_idx])
                print(
                    f"[BO] iter {it:03d}/{n_iters} | "
                    f"y_next={y_next:.6g} | best_y={float(np.min(y)):.6g} "
                    f"({'improved' if improved else 'no-improve'}) | "
                    f"EI_max={ei_max:.3g} | mu={mu_next:.6g} | std={std_next:.3g}"
                )
                params = self.decode_theta(theta_next)
                print(f"      proposed (x,y,phi) per drone:\n      {params}")
                print(f"      gp.kernel_ = {gp.kernel_}")

            y_nexts.append(y_next)
            y_bests.append(float(np.min(y)))

        best_idx = int(np.argmin(y))
        best_theta = X_theta[best_idx]
        best_params = self.decode_theta(best_theta)
        best_y = float(y[best_idx])

        if verbose:
            print(f"[BO] done: best_y={best_y:.6g}")
            print(f"[BO] best params:\n{best_params}")

        return best_theta, best_params, best_y, (X, y), y_nexts, y_bests

    def run_bayes_opt_polar(
        self,
        n_init: int = 10,
        n_iters: int = 30,
        n_candidates: int = 5000,
        xi: float = 0.01,
        K_grid: int = 500,
        boundary_field: str = "affected",
        n_r: int = 160,
        smooth_iters: int = 350,
        omega: float = 1.0,
        verbose: bool = True,
        print_every: int = 1,
        use_ard_kernel: bool = False,
        init_strategy: str = "heuristic",  # "random", "heuristic"
        init_heuristic_random_frac: float = 0.2,
        init_heuristic_kwargs: dict | None = None,
        candidate_strategy: str = "random",  # "random", "qmc", "mixed"
        candidate_qmc: str = "sobol",
        candidate_local_frac: float = 0.5,
        candidate_local_top_k: int = 3,
        candidate_local_sigma_s: float = 0.05,
        candidate_local_sigma_r: float = 0.05,
        candidate_local_sigma_delta_rad: float = np.deg2rad(15.0),
        candidate_local_resample_delta_prob: float = 0.05,
    ):
        if self.sr_grid is None:
            self.setup_search_grid_polar(
                K=K_grid,
                boundary_field=boundary_field,
                n_r=n_r,
                smooth_iters=smooth_iters,
                omega=omega,
            )
            if verbose:
                print(f"[BO polar] Search grid set up with {self.sr_grid.shape[0]}x{self.sr_grid.shape[1]} points.")
                self.plot_polar_domain()

        init_strategy = str(init_strategy).lower().strip()
        if init_strategy == "random":
            X_theta = self.sample_random_theta_polar(n_init)
        elif init_strategy == "heuristic":
            heuristic_kwargs = {} if init_heuristic_kwargs is None else dict(init_heuristic_kwargs)
            X_theta = self.sample_initial_thetas_polar(
                n_init=n_init,
                heuristic_random_frac=init_heuristic_random_frac,
                **heuristic_kwargs,
            )
        else:
            raise ValueError("Unknown init strategy. Use 'random' or 'heuristic'.")

        X_theta = np.atleast_2d(X_theta)
        y = np.array([self.expected_value_burned_area_polar(th) for th in X_theta], dtype=float)
        X = np.vstack([self.theta_to_gp_features_polar(th) for th in X_theta])

        if verbose:
            best0 = float(np.min(y))
            print(f"[BO polar] init: n_init={n_init}, dim={self.dim}")
            print(f"[BO polar] init: best_y={best0:.6g}, mean_y={float(np.mean(y)):.6g}, std_y={float(np.std(y)):.6g}")

        if use_ard_kernel:
            base_kernel = Matern(
                length_scale=np.ones(X.shape[1], dtype=float),
                nu=2.5,
                length_scale_bounds=(1e-3, 1e3),
            )
        else:
            base_kernel = TiedXYFiMatern(lx=0.2, ly=0.2, lphi=0.5, nu=2.5, length_scale_bounds=(1e-3, 1e3))

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * base_kernel
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2))
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=None,
        )

        y_nexts = []
        y_bests = []

        for it in range(1, n_iters + 1):
            gp.fit(X, y)

            cstrat = str(candidate_strategy).lower().strip()
            if cstrat == "mixed":
                local_frac = float(np.clip(candidate_local_frac, 0.0, 1.0))
                n_local = int(np.round(n_candidates * local_frac))
                n_global = int(n_candidates - n_local)
                globals_ = self.sample_qmc_theta(n_global, method=candidate_qmc)

                if n_local > 0:
                    top_k = int(max(candidate_local_top_k, 1))
                    best_idx = np.argsort(y)[:top_k]
                    anchors = X_theta[best_idx]
                    locals_ = self.sample_local_theta_polar(
                        anchors,
                        n_local,
                        sigma_s=candidate_local_sigma_s,
                        sigma_r=candidate_local_sigma_r,
                        sigma_delta_rad=candidate_local_sigma_delta_rad,
                        resample_delta_prob=candidate_local_resample_delta_prob,
                    )
                    Xcand_theta = np.vstack([np.atleast_2d(globals_), np.atleast_2d(locals_)])
                else:
                    Xcand_theta = np.atleast_2d(globals_)
            elif cstrat == "qmc":
                Xcand_theta = self.sample_qmc_theta(n_candidates, method=candidate_qmc)
            elif cstrat == "random":
                Xcand_theta = self.sample_random_theta_polar(n_candidates)
            else:
                raise ValueError("Unknown candidate_strategy for polar. Use 'random', 'qmc', or 'mixed'.")

            Xcand_theta = np.atleast_2d(Xcand_theta)
            Xcand = np.vstack([self.theta_to_gp_features_polar(th) for th in Xcand_theta])
            y_best = float(np.min(y))
            ei = expected_improvement(Xcand, gp, y_best=y_best, xi=xi)
            best_ei_idx = int(np.argmax(ei))
            theta_next = Xcand_theta[best_ei_idx]
            x_next = Xcand[best_ei_idx]

            mu_next, std_next = gp.predict(x_next[None, :], return_std=True)
            mu_next = float(mu_next[0])
            std_next = float(std_next[0])

            y_next = float(self.expected_value_burned_area_polar(theta_next))

            X_theta = np.vstack([X_theta, theta_next])
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)

            if verbose and (it % max(print_every, 1) == 0 or it == 1 or it == n_iters):
                improved = y_next < y_best
                ei_max = float(ei[best_ei_idx])
                print(
                    f"[BO polar] iter {it:03d}/{n_iters} | "
                    f"y_next={y_next:.6g} | best_y={float(np.min(y)):.6g} "
                    f"({'improved' if improved else 'no-improve'}) | "
                    f"EI_max={ei_max:.3g} | mu={mu_next:.6g} | std={std_next:.3g}"
                )
                print(f"      objective (expected value burned) = {y_next:.6g}")
                params = self.decode_theta_polar(theta_next)
                print(f"      proposed (x,y,phi) per drone:\n      {params}")
                print(f"      gp.kernel_ = {gp.kernel_}")

            y_nexts.append(y_next)
            y_bests.append(float(np.min(y)))

        best_idx = int(np.argmin(y))
        best_theta = X_theta[best_idx]
        best_params = self.decode_theta_polar(best_theta)
        best_y = float(y[best_idx])

        if verbose:
            print(f"[BO polar] done: best_y={best_y:.6g}")
            print(f"[BO polar] best params:\n{best_params}")

        return best_theta, best_params, best_y, (X, y), y_nexts, y_bests

    def run_random_search(
        self,
        n_evals: int = 50,
        K_grid: int = 500,
        boundary_field: str = "affected",
        verbose: bool = True,
        print_every: int = 1,
    ):
        """Simple random search baseline over [0,1]^dim."""
        if self.projector is None:
            self.setup_search_grid(K=K_grid, boundary_field=boundary_field)
            if verbose:
                print(f"[Random] Search grid set up with {len(self.projector.coords)} valid cells in grid.")
                self.fire_model.plot_search_domain(self.search_domain_mask, title="Current Search Domain:")

        thetas = self.sample_random_theta(n_evals)
        y_vals: list[float] = []
        y_bests: list[float] = []

        best_theta = None
        best_y = float("inf")

        for i, theta in enumerate(thetas, start=1):
            y_val = float(self.expected_value_burned_area(theta))
            y_vals.append(y_val)

            if y_val < best_y:
                best_y = y_val
                best_theta = theta

            y_bests.append(best_y)

            if verbose and (i % max(print_every, 1) == 0 or i == 1 or i == n_evals):
                params = self.decode_theta(theta)
                print(
                    f"[Random] eval {i:03d}/{n_evals} | y={y_val:.6g} | best={best_y:.6g}\n"
                    f"        (x,y,phi) per drone:\n        {params}"
                )

        best_params = self.decode_theta(best_theta)
        X_feats = np.vstack([self.theta_to_gp_features(th) for th in thetas])
        y_arr = np.array(y_vals, dtype=float)
        y_nexts = list(y_vals)
        y_bests_arr = np.array(y_bests, dtype=float)

        if verbose:
            print(f"[Random] done: best_y={best_y:.6g}")
            print(f"[Random] best params:\n{best_params}")

        return best_theta, best_params, best_y, (X_feats, y_arr), y_nexts, y_bests_arr

    def run_heuristic_search(
        self,
        n_evals: int = 50,
        K_grid: int = 500,
        boundary_field: str = "affected",
        *,
        heuristic_random_frac: float = 0.0,
        heuristic_kwargs: dict | None = None,
        verbose: bool = True,
        print_every: int = 1,
    ):
        """
        Heuristic search baseline (repeatedly sample `sample_heuristic_theta` and keep the best).

        Use this to compare "what the heuristic alone can do" against BO for a fixed evaluation budget.
        """
        if self.projector is None:
            self.setup_search_grid(K=K_grid, boundary_field=boundary_field)
            if verbose:
                print(f"[Heuristic] Search grid set up with {len(self.projector.coords)} valid cells in grid.")
                self.fire_model.plot_search_domain(self.search_domain_mask, title="Current Search Domain:")

        heuristic_kwargs = {} if heuristic_kwargs is None else dict(heuristic_kwargs)
        thetas = self.sample_initial_thetas(
            n_init=n_evals,
            strategy="heuristic",
            heuristic_random_frac=heuristic_random_frac,
            heuristic_kwargs=heuristic_kwargs,
        )

        y_vals: list[float] = []
        y_bests: list[float] = []

        best_theta = None
        best_y = float("inf")

        for i, theta in enumerate(thetas, start=1):
            y_val = float(self.expected_value_burned_area(theta))
            y_vals.append(y_val)

            if y_val < best_y:
                best_y = y_val
                best_theta = theta

            y_bests.append(best_y)

            if verbose and (i % max(print_every, 1) == 0 or i == 1 or i == n_evals):
                params = self.decode_theta(theta)
                print(
                    f"[Heuristic] eval {i:03d}/{n_evals} | y={y_val:.6g} | best={best_y:.6g}\n"
                    f"           (x,y,phi) per drone:\n           {params}"
                )

        best_params = self.decode_theta(best_theta)
        X_feats = np.vstack([self.theta_to_gp_features(th) for th in thetas])
        y_arr = np.array(y_vals, dtype=float)
        y_nexts = list(y_vals)
        y_bests_arr = np.array(y_bests, dtype=float)

        if verbose:
            print(f"[Heuristic] done: best_y={best_y:.6g}")
            print(f"[Heuristic] best params:\n{best_params}")

        return best_theta, best_params, best_y, (X_feats, y_arr), y_nexts, y_bests_arr

    def plot_evolved_firestate(
        self,
        theta: np.ndarray,
        n_sims: int = 10,
        title: str | None = None,
    ):
        drone_params = self.decode_theta(theta)

        evolved_firestate = self.fire_model.simulate_from_firestate(
            self.init_firestate,
            T=self.evolution_time_s,
            n_sims=n_sims,
            drone_params=drone_params,
            ros_mps=self.fire_model.env.ros_mps,
            wind_coeff=self.fire_model.env.wind_coeff,
            diag=self.fire_model.env.diag,
            seed=None,
        )

        self.fire_model.plot_firestate(
            self.init_firestate,
            kind="p_affected",
            title=title if title is not None else "Initial FireState before Retardant Drop",
        )

        self.fire_model.plot_firestate(
            evolved_firestate,
            kind="p_affected",
            title=title if title is not None else "Evolved FireState after Retardant Drop",
        )

        self.fire_model.plot_firestate(
            evolved_firestate,
            kind="retardant",
            title=title if title is not None else "Corresponding Retardant Locations",
        )

    def plot_evolved_firestate_polar(
        self,
        theta: np.ndarray,
        *,
        n_sims: int | None = None,
        title_prefix: str | None = None,
        run_idx: int | None = None,
        run_total: int | None = None,
    ) -> float:
        drone_params = self.decode_theta_polar(theta)
        evolved_firestate = self._simulate_firestate_with_params(drone_params, n_sims=n_sims)
        objective = self._expected_value_from_firestate(evolved_firestate)

        prefix = title_prefix if title_prefix is not None else "Polar placement"
        if run_idx is not None and run_total is not None:
            prefix = f"{prefix} {run_idx}/{run_total}"
        base_title = f"{prefix} | objective={objective:.6g}"

        self.fire_model.plot_firestate(
            self.init_firestate,
            kind="p_affected",
            title=f"{base_title} | initial",
        )

        self.fire_model.plot_firestate(
            evolved_firestate,
            kind="p_affected",
            title=f"{base_title} | evolved",
        )

        self.fire_model.plot_firestate(
            evolved_firestate,
            kind="retardant",
            title=f"{base_title} | retardant",
        )
        return objective


__all__ = [
    "SearchGridProjector",
    "TiedXYFiMatern",
    "expected_improvement",
    "RetardantDropBayesOpt",
]
