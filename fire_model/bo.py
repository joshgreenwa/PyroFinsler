import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
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
        search_domain_mask, init_boundary, final_boundary, _final_firestate = search
        self.init_boundary = init_boundary
        self.final_boundary = final_boundary

        xs, ys = np.where(search_domain_mask)
        coords = np.stack([xs.astype(float), ys.astype(float)], axis=1)
        return search_domain_mask, coords

    def setup_search_grid(self, K=500, boundary_field="affected"):
        mask, coords = self.generate_search_grid(K=K, boundary_field=boundary_field)
        self.search_domain_mask = mask
        self.projector = SearchGridProjector(mask=mask, coords=coords)
        self.shape = mask.shape
        return mask, coords

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
    ):
        """
        Heuristic initialisation for (x,y,phi) per drone:
          - choose points along the *current* fire boundary, biased toward the downwind-facing front
          - place drops ahead of the front inside the search ring using nearest-point mapping to the outer boundary
          - orient the retardant line approximately tangent to the boundary (optionally blended with cross-wind)
        """
        if self.projector is None or self.shape is None:
            raise RuntimeError("Call setup_search_grid(...) before heuristic sampling.")
        if self.init_boundary is None or self.final_boundary is None:
            raise RuntimeError("Search boundaries not available; call setup_search_grid(...) first.")

        nx, ny = self.shape
        inner_xy = np.asarray(self.init_boundary.xy, dtype=float)
        outer_xy = np.asarray(self.final_boundary.xy, dtype=float)
        if inner_xy.ndim != 2 or inner_xy.shape[1] != 2:
            raise ValueError(f"init_boundary.xy must have shape (K,2); got {inner_xy.shape}")
        if outer_xy.ndim != 2 or outer_xy.shape[1] != 2:
            raise ValueError(f"final_boundary.xy must have shape (K,2); got {outer_xy.shape}")

        K = inner_xy.shape[0]
        if K < max(8, 2 * self.n_drones):
            return self.sample_random_theta_on_mask(n=n)

        outer_tree = cKDTree(outer_xy)
        _, nn_idx = outer_tree.query(inner_xy, k=1)
        v_out = outer_xy[nn_idx] - inner_xy

        inner_prev = np.roll(inner_xy, 1, axis=0)
        inner_next = np.roll(inner_xy, -1, axis=0)
        tangents = inner_next - inner_prev

        w_mean = self._estimate_mean_wind()
        wmag = float(np.linalg.norm(w_mean))
        w_unit = self._unit(w_mean) if wmag > 1e-9 else np.zeros(2, dtype=float)

        v_unit = np.array([self._unit(v) for v in v_out])
        align = v_unit @ w_unit if wmag > 1e-9 else np.zeros(K, dtype=float)
        scores = np.exp(float(wind_bias) * np.clip(align, -1.0, 1.0))

        value_bias = float(value_bias)
        if value_bias != 0.0:
            val_map = np.asarray(self.fire_model.env.value, dtype=float)
            alpha_mid = 0.6
            mid = inner_xy + alpha_mid * v_out
            xi = np.clip(np.round(mid[:, 0]).astype(int), 0, nx - 1)
            yi = np.clip(np.round(mid[:, 1]).astype(int), 0, ny - 1)
            vals = val_map[xi, yi]
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmax > vmin + 1e-12:
                vals = (vals - vmin) / (vmax - vmin)
            else:
                vals = np.zeros_like(vals, dtype=float)
            scores = scores * np.exp(value_bias * vals)

        arc_sep = int(np.clip(np.floor(float(min_arc_sep_frac) * (K / max(self.n_drones, 1))), 1, max(K // 2, 1)))
        alpha_lo, alpha_hi = alpha_range
        alpha_lo, alpha_hi = float(alpha_lo), float(alpha_hi)
        if not (0.0 <= alpha_lo <= alpha_hi <= 1.0):
            raise ValueError("alpha_range must satisfy 0 <= lo <= hi <= 1.")

        wind_long_axis_blend = float(np.clip(wind_long_axis_blend, 0.0, 1.0))
        phi_jitter_rad = float(max(phi_jitter_rad, 0.0))

        thetas = np.empty((max(n, 1), self.dim), dtype=float)

        for i in range(max(n, 1)):
            available = np.ones(K, dtype=bool)
            chosen: list[int] = []

            for _d in range(self.n_drones):
                w = np.where(available, scores, 0.0)
                if float(w.sum()) <= 0.0:
                    w = available.astype(float)
                if float(w.sum()) <= 0.0:
                    chosen.append(int(self.rng.integers(0, K)))
                else:
                    p = w / float(w.sum())
                    chosen.append(int(self.rng.choice(K, p=p)))

                idx0 = chosen[-1]
                for off in range(-arc_sep, arc_sep + 1):
                    available[(idx0 + off) % K] = False
                if (not available.any()) and (len(chosen) < self.n_drones):
                    available[:] = True

            theta = np.empty(self.dim, dtype=float)
            for d, idx in enumerate(chosen):
                p0 = inner_xy[idx]
                v = v_out[idx]
                if float(np.linalg.norm(v)) <= 1e-9:
                    v = np.array([1.0, 0.0], dtype=float)

                alpha = float(self.rng.uniform(alpha_lo, alpha_hi))
                cand = p0 + alpha * v
                xg, yg = self.projector.snap(float(cand[0]), float(cand[1]))

                t = tangents[idx]
                u_tan = self._unit(t) if float(np.linalg.norm(t)) > 1e-9 else np.array([0.0, 1.0], dtype=float)
                u_long = u_tan
                if wmag > 1e-9 and wind_long_axis_blend > 0.0:
                    u_wperp = self._unit(np.array([-w_unit[1], w_unit[0]], dtype=float))
                    u_mix = (1.0 - wind_long_axis_blend) * u_tan + wind_long_axis_blend * u_wperp
                    if float(np.linalg.norm(u_mix)) > 1e-9:
                        u_long = self._unit(u_mix)

                long_angle = float(np.arctan2(u_long[1], u_long[0]))
                phi = self._phi_from_long_axis_angle(long_angle)
                if phi_jitter_rad > 0.0:
                    phi = self._wrap_angle(phi + float(self.rng.normal(0.0, phi_jitter_rad)))

                theta[3 * d + 0] = float(xg) / max(nx - 1, 1)
                theta[3 * d + 1] = float(yg) / max(ny - 1, 1)
                theta[3 * d + 2] = float(phi) / (2.0 * np.pi)

            thetas[i] = np.clip(theta, 0.0, 1.0)

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
        evolved_firestate = self.fire_model.simulate_from_firestate(
            self.init_firestate,
            T=self.evolution_time_s,
            n_sims=self.n_sims,
            drone_params=drone_params,
            ros_mps=self.fire_model.env.ros_mps,
            wind_coeff=self.fire_model.env.wind_coeff,
            diag=self.fire_model.env.diag,
            seed=None,
            avoid_burning_drop=self.fire_model.env.avoid_burning_drop,
            burning_prob_threshold=self.fire_model.env.avoid_drop_p_threshold,
        )

        p_burning = evolved_firestate.burning[0].astype(float, copy=False)
        p_burned = evolved_firestate.burned[0].astype(float, copy=False)
        p_affected = np.clip(p_burning + p_burned, 0.0, 1.0)

        nx, _ = self.fire_model.env.grid_size
        dx = self.fire_model.env.domain_km / nx

        expected_value_burned = np.sum(p_affected * self.fire_model.env.value) * (dx ** 2)
        return float(expected_value_burned)

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


__all__ = [
    "SearchGridProjector",
    "TiedXYFiMatern",
    "expected_improvement",
    "RetardantDropBayesOpt",
]
