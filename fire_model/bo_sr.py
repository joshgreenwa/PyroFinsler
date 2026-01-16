import numpy as np
from numpy.random import default_rng
from scipy.stats import norm, qmc
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter, Matern, WhiteKernel, Kernel
from scipy.spatial import cKDTree

from fire_model.ca import CAFireModel, FireState


def expected_improvement(X_candidates, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.clip(sigma, 1e-9, None)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 0.0] = 0.0
    return ei


class TiedSRDeltaMatern(Kernel):
    """Matérn kernel on inputs with repeating 4D blocks per drone: [s, r, sin(delta), cos(delta)]."""

    def __init__(
        self,
        ls: float = 0.2,
        lr: float = 0.2,
        ldelta: float = 0.5,
        nu: float = 2.5,
        length_scale_bounds=(1e-3, 1e3),
        fd_eps: float = 1e-6,
    ):
        self.ls = float(ls)
        self.lr = float(lr)
        self.ldelta = float(ldelta)
        self.nu = nu
        self.length_scale_bounds = length_scale_bounds
        self.fd_eps = float(fd_eps)
        self._base = Matern(length_scale=1.0, nu=nu)

    @property
    def hyperparameter_ls(self):
        return Hyperparameter("ls", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_lr(self):
        return Hyperparameter("lr", "numeric", self.length_scale_bounds)

    @property
    def hyperparameter_ldelta(self):
        return Hyperparameter("ldelta", "numeric", self.length_scale_bounds)

    @property
    def theta(self):
        return np.log([self.ls, self.lr, self.ldelta])

    @theta.setter
    def theta(self, theta):
        ls, lr, ldelta = np.exp(theta)
        self.ls, self.lr, self.ldelta = float(ls), float(lr), float(ldelta)

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
            raise ValueError(
                f"Expected feature dim multiple of 4, got {d}. "
                "Make sure you're using [s,r,sin,cos] per drone."
            )

        Xs = X.copy()
        idx = np.arange(d)
        s_idx = (idx % 4) == 0
        r_idx = (idx % 4) == 1
        p_idx = (idx % 4) >= 2

        Xs[:, s_idx] /= self.ls
        Xs[:, r_idx] /= self.lr
        Xs[:, p_idx] /= self.ldelta
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
        return (
            f"{self.__class__.__name__}(ls={self.ls:.3g}, lr={self.lr:.3g}, "
            f"ldelta={self.ldelta:.3g}, nu={self.nu})"
        )


class RetardantDropBayesOptSR:
    """Bayesian optimisation over (s,r,delta) for retardant drop placement."""

    def __init__(
        self,
        fire_model: CAFireModel,
        init_firestate: FireState,
        n_drones: int,
        evolution_time_s: float,
        n_sims: int,
        fire_boundary_probability: float = 0.25,
        search_grid_evolution_time_s: float | None = None,
        rng=None,
    ):
        self.fire_model = fire_model
        self.init_firestate = init_firestate
        self.n_drones = int(n_drones)
        self.dim = 3 * self.n_drones

        self.evolution_time_s = float(evolution_time_s)
        self.n_sims = int(n_sims)
        self.p_boundary = float(fire_boundary_probability)
        self.search_grid_evolution_time_s = search_grid_evolution_time_s

        self.rng = default_rng() if rng is None else rng

        self.search_domain_mask: np.ndarray | None = None
        self.shape: tuple[int, int] | None = None
        self.init_boundary = None
        self.final_boundary = None
        self.final_search_firestate = None

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

    def generate_search_grid(self, K: int = 500, boundary_field: str = "affected"):
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
        self.search_domain_mask = search_domain_mask
        self.init_boundary = init_boundary
        self.final_boundary = final_boundary
        self.final_search_firestate = final_state
        self.shape = search_domain_mask.shape
        return search_domain_mask, init_boundary, final_boundary

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

    def setup_search_grid_sr(
        self,
        *,
        K: int = 500,
        boundary_field: str = "affected",
        n_r: int = 160,
        smooth_iters: int = 350,
        omega: float = 1.0,
    ):
        self.generate_search_grid(K=K, boundary_field=boundary_field)
        if self.init_boundary is None or self.final_boundary is None:
            raise RuntimeError("Search grid not initialised; boundaries are missing.")

        inner_xy = np.asarray(self.init_boundary.xy, dtype=float)
        outer_xy = np.asarray(self.final_boundary.xy, dtype=float)
        if inner_xy.shape != outer_xy.shape:
            raise ValueError("Boundary point counts do not match; rerun setup_search_grid_sr.")

        r_targets = np.linspace(0.0, 1.0, int(n_r))
        grid_linear = self._build_linear_grid(inner_xy, outer_xy, r_targets)
        grid = self._smooth_grid_laplace(grid_linear, n_iters=smooth_iters, omega=omega)

        valid = self.sr_valid_mask if self.sr_valid_mask is not None else np.isfinite(grid[..., 0]) & np.isfinite(grid[..., 1])
        if self.search_domain_mask is not None:
            xi = np.clip(np.round(grid[..., 0]).astype(int), 0, self.search_domain_mask.shape[0] - 1)
            yi = np.clip(np.round(grid[..., 1]).astype(int), 0, self.search_domain_mask.shape[1] - 1)
            valid &= self.search_domain_mask[xi, yi]
        idx = np.column_stack(np.where(valid))
        if idx.size == 0:
            raise RuntimeError("SR grid has no valid points; check boundaries or parameters.")

        self.sr_grid = grid
        self.sr_r_targets = r_targets
        self.sr_phi_grid = self._grid_angle_along_s(grid)
        self.sr_valid_mask = valid
        self.sr_valid_indices = idx.astype(float)
        self.sr_index_tree = cKDTree(self.sr_valid_indices)
        return grid

    def _sr_lookup(self, s: float, r: float) -> tuple[np.ndarray, float]:
        if self.sr_grid is None or self.sr_phi_grid is None or self.sr_index_tree is None:
            raise RuntimeError("SR grid not initialised; call setup_search_grid_sr(...).")

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

    def decode_theta_sr(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        params = []
        for d in range(self.n_drones):
            s = float(theta[3 * d + 0])
            r = float(theta[3 * d + 1])
            delta = float(theta[3 * d + 2]) * (2.0 * np.pi)
            params.append((s, r, delta))
        params = np.array(params, dtype=float)
        order = np.lexsort((params[:, 2], params[:, 1], params[:, 0]))
        return params[order]

    def decode_theta(self, theta: np.ndarray) -> np.ndarray:
        if self.sr_grid is None:
            raise RuntimeError("Call setup_search_grid_sr(...) before decode_theta.")

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
        return params[order]

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

    def theta_to_gp_features(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        feats = []
        for d in range(self.n_drones):
            s = float(theta[3 * d + 0])
            r = float(theta[3 * d + 1])
            delta = float(theta[3 * d + 2]) * (2.0 * np.pi)
            feats.extend([s, r, np.sin(delta), np.cos(delta)])
        return np.asarray(feats, dtype=float)

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

    def _sr_value_grid(self) -> np.ndarray:
        if self.sr_grid is None:
            raise RuntimeError("Call setup_search_grid_sr(...) before SR value scoring.")
        values = np.asarray(self.fire_model.env.value, dtype=float)
        grid = self.sr_grid
        valid = self.sr_valid_mask if self.sr_valid_mask is not None else np.isfinite(grid[..., 0]) & np.isfinite(grid[..., 1])

        xi = np.clip(np.round(grid[..., 0]).astype(int), 0, values.shape[0] - 1)
        yi = np.clip(np.round(grid[..., 1]).astype(int), 0, values.shape[1] - 1)

        value_grid = np.full(grid.shape[:2], np.nan, dtype=float)
        value_grid[valid] = values[xi[valid], yi[valid]]
        return value_grid

    def _sr_value_blocking(
        self,
        *,
        r_value: float | None = None,
        r_offset: float = 0.0,
        count: int,
        compactness: float,
        min_spacing: float | None = None,
        min_arc_sep_frac: float | None = None,
        value_power: float = 1.0,
        value_offset: float | None = None,
        selection: str = "weighted",
    ) -> list[tuple[float, float, float]]:
        if self.sr_grid is None or self.sr_r_targets is None or self.init_boundary is None:
            raise RuntimeError("Call setup_search_grid_sr(...) before SR heuristics.")

        value_grid = self._sr_value_grid()
        valid = np.isfinite(value_grid)
        if not np.any(valid):
            return []

        background = float(np.nanmin(value_grid)) if value_offset is None else float(value_offset)
        value_rel = np.where(valid, value_grid - background, np.nan)
        weights_grid = np.where(value_rel > 0.0, value_rel, 0.0)
        weights_grid = np.power(weights_grid, float(value_power))

        if np.any(weights_grid > 0.0):
            r_idx = np.argmax(weights_grid, axis=1)
            scores = weights_grid[np.arange(weights_grid.shape[0]), r_idx]
            scores = np.where(scores > 0.0, scores, -np.inf)
        else:
            value_filled = np.where(valid, value_grid, -np.inf)
            r_idx = np.argmax(value_filled, axis=1)
            scores = value_filled[np.arange(value_filled.shape[0]), r_idx]
            scores = np.where(np.isfinite(scores), scores, 0.0)

        arc_len = self._sr_arc_length(np.asarray(self.init_boundary.xy, dtype=float))
        if min_spacing is None:
            frac = 0.0 if min_arc_sep_frac is None else float(min_arc_sep_frac)
            total = float(arc_len[-1])
            per = total / max(count, 1) if total > 0.0 else 0.0
            min_spacing = frac * per

        scores = self._sr_restrict_by_compactness(scores, arc_len, compactness)

        selection = str(selection).lower().strip()
        if selection not in {"weighted", "greedy"}:
            raise ValueError("selection must be 'weighted' or 'greedy'")

        if selection == "greedy":
            idx = self._sr_greedy_select_indices(scores, arc_len, float(min_spacing), int(count))
        else:
            idx = []
            scores_work = scores.copy()
            total = float(arc_len[-1])
            min_spacing = float(min_spacing)
            K = scores_work.size
            for _ in range(int(count)):
                weights = np.clip(scores_work, 0.0, None)
                wsum = float(np.sum(weights))
                if wsum <= 0.0 or not np.isfinite(wsum):
                    valid = np.isfinite(scores_work)
                    weights = np.zeros_like(scores_work)
                    weights[valid] = 1.0
                    wsum = float(np.sum(weights))
                    if wsum <= 0.0:
                        break
                probs = weights / wsum
                choice = int(self.rng.choice(K, p=probs))
                idx.append(choice)

                if total <= 0.0:
                    scores_work[:] = 0.0
                    continue
                for j in range(K):
                    d = abs(arc_len[j] - arc_len[choice])
                    d = min(d, total - d)
                    if d < min_spacing:
                        scores_work[j] = 0.0

        r_base = (
            self.sr_r_targets[np.clip(r_idx, 0, len(self.sr_r_targets) - 1)]
            if r_value is None
            else np.full(value_grid.shape[0], float(r_value), dtype=float)
        )
        out = []
        for i in idx:
            r = float(np.clip(float(r_base[i]) + float(r_offset), 0.0, 1.0))
            out.append((i / float(self.sr_grid.shape[0]), r, 0.0))
        return out

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

    def _random_params_on_mask(self, count: int) -> np.ndarray:
        if self.sr_valid_indices is None or self.sr_r_targets is None or self.sr_grid is None:
            raise RuntimeError("Call setup_search_grid_sr(...) before sampling on mask.")
        count = int(max(count, 1))
        idx = self.rng.integers(0, len(self.sr_valid_indices), size=count)
        sr_idx = self.sr_valid_indices[idx].astype(int)
        s = sr_idx[:, 0] / float(self.sr_grid.shape[0])
        r = self.sr_r_targets[sr_idx[:, 1]]
        delta = self.rng.random(count) * (2.0 * np.pi)
        return np.column_stack([s, r, delta])

    def sample_random_theta_on_mask(self, n: int = 1):
        params = self._random_params_on_mask(self.n_drones * max(int(n), 1))
        params = params.reshape(max(int(n), 1), self.n_drones, 3)
        out = np.empty((params.shape[0], self.dim), dtype=float)
        for i in range(params.shape[0]):
            out[i] = self._encode_sr_params(params[i])
        return out[0] if n == 1 else out

    def sample_local_theta(
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

    def sample_heuristic_theta(
        self,
        n: int,
        *,
        r_value: float | None = 0.6,
        r_offset: float = 0.0,
        n_keep: int | None = None,
        compactness: float = 0.25,
        min_spacing: float | None = None,
        min_arc_sep_frac: float = 0.25,
        value_power: float = 1.0,
        value_offset: float | None = None,
        selection: str = "weighted",
    ) -> np.ndarray:
        n = int(max(n, 1))
        n_keep = self.n_drones if n_keep is None else int(max(n_keep, 1))

        thetas = np.empty((n, self.dim), dtype=float)
        placements = self._sr_value_blocking(
            r_value=r_value,
            r_offset=r_offset,
            count=n_keep,
            compactness=compactness,
            min_spacing=min_spacing,
            min_arc_sep_frac=min_arc_sep_frac,
            value_power=value_power,
            value_offset=value_offset,
            selection=selection,
        )

        params = np.asarray(placements, dtype=float)
        if params.shape[0] > self.n_drones:
            idx = self.rng.permutation(params.shape[0])[: self.n_drones]
            params = params[idx]

        for i in range(n):
            params_i = params
            if params_i.shape[0] < self.n_drones:
                need = self.n_drones - params_i.shape[0]
                extra = self._random_params_on_mask(need)
                params_i = np.vstack([params_i, extra])
            thetas[i] = self._encode_sr_params(params_i)

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
          - `random`: uniform over [0,1]^dim
          - `random_mask`: uniform over valid SR grid points + random delta
          - `heuristic`: value-blocking placement with optional random_mask mixing
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

    def expected_value_burned_area(self, theta: np.ndarray) -> float:
        drone_params = self.decode_theta(theta)
        evolved_firestate = self._simulate_firestate_with_params(drone_params)
        return self._expected_value_from_firestate(evolved_firestate)

    def plot_sr_domain(
        self,
        *,
        n_s_lines: int = 12,
        n_r_lines: int = 8,
        show_fields: bool = True,
        show_diagnostics: bool = True,
        hist_bins: int = 40,
    ):
        if self.sr_grid is None or self.sr_r_targets is None:
            raise RuntimeError("SR grid not initialised; call setup_search_grid_sr(...).")
        if self.search_domain_mask is None:
            raise RuntimeError("Search mask missing; call setup_search_grid_sr(...).")

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
        ax0.set_title("SR grid (s,r) on mask")
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
        dist, nn = tree.query(coords, k=1)

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

    def plot_evolved_firestate(
        self,
        theta: np.ndarray,
        *,
        n_sims: int | None = None,
        title_prefix: str | None = None,
    ) -> float:
        drone_params = self.decode_theta(theta)
        evolved_firestate = self._simulate_firestate_with_params(drone_params, n_sims=n_sims)
        objective = self._expected_value_from_firestate(evolved_firestate)

        prefix = title_prefix if title_prefix is not None else "SR placement"
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

    def run_bayes_opt(
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
        init_strategy: str = "random",  # "random", "random_mask", "heuristic"
        init_heuristic_random_frac: float = 0.2,
        init_heuristic_kwargs: dict | None = None,
        candidate_strategy: str = "random",  # "random", "random_mask", "qmc", "mixed"
        candidate_qmc: str = "sobol",
        candidate_global_masked: bool = False,
        candidate_local_frac: float = 0.5,
        candidate_local_top_k: int = 3,
        candidate_local_sigma_s: float = 0.05,
        candidate_local_sigma_r: float = 0.05,
        candidate_local_sigma_delta_rad: float = np.deg2rad(15.0),
        candidate_local_resample_delta_prob: float = 0.05,
    ):
        if self.sr_grid is None:
            self.setup_search_grid_sr(
                K=K_grid,
                boundary_field=boundary_field,
                n_r=n_r,
                smooth_iters=smooth_iters,
                omega=omega,
            )
            if verbose:
                print(f"[BO SR] Search grid set up with {self.sr_grid.shape[0]}x{self.sr_grid.shape[1]} points.")
                self.fire_model.plot_search_domain(self.search_domain_mask, title="Search Domain (Between Boundaries)")
                self.plot_sr_domain()

        init_strategy = str(init_strategy).lower().strip()
        if init_strategy == "random":
            X_theta = self.sample_random_theta(n_init)
        elif init_strategy == "random_mask":
            X_theta = self.sample_random_theta_on_mask(n_init)
        elif init_strategy == "heuristic":
            heuristic_kwargs = {} if init_heuristic_kwargs is None else dict(init_heuristic_kwargs)
            X_theta = self.sample_initial_thetas(
                n_init=n_init,
                strategy="heuristic",
                heuristic_random_frac=init_heuristic_random_frac,
                heuristic_kwargs=heuristic_kwargs,
            )
        else:
            raise ValueError("Unknown init strategy. Use 'random', 'random_mask', or 'heuristic'.")

        X_theta = np.atleast_2d(X_theta)
        y = np.array([self.expected_value_burned_area(th) for th in X_theta], dtype=float)
        X = np.vstack([self.theta_to_gp_features(th) for th in X_theta])

        if verbose:
            best0 = float(np.min(y))
            print(f"[BO SR] init: n_init={n_init}, dim={self.dim}")
            print(f"[BO SR] init: best_y={best0:.6g}, mean_y={float(np.mean(y)):.6g}, std_y={float(np.std(y)):.6g}")

        if use_ard_kernel:
            base_kernel = Matern(
                length_scale=np.ones(X.shape[1], dtype=float),
                nu=2.5,
                length_scale_bounds=(1e-3, 1e3),
            )
        else:
            base_kernel = TiedSRDeltaMatern(ls=0.2, lr=0.2, ldelta=0.5, nu=2.5, length_scale_bounds=(1e-3, 1e3))

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
                if candidate_global_masked:
                    globals_ = self.sample_random_theta_on_mask(n_global)
                else:
                    globals_ = self.sample_qmc_theta(n_global, method=candidate_qmc)

                if n_local > 0:
                    top_k = int(max(candidate_local_top_k, 1))
                    best_idx = np.argsort(y)[:top_k]
                    anchors = X_theta[best_idx]
                    locals_ = self.sample_local_theta(
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
                Xcand_theta = self.sample_random_theta(n_candidates)
            elif cstrat == "random_mask":
                Xcand_theta = self.sample_random_theta_on_mask(n_candidates)
            else:
                raise ValueError("Unknown candidate_strategy. Use 'random', 'random_mask', 'qmc', or 'mixed'.")

            Xcand_theta = np.atleast_2d(Xcand_theta)
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
                    f"[BO SR] iter {it:03d}/{n_iters} | "
                    f"y_next={y_next:.6g} | best_y={float(np.min(y)):.6g} "
                    f"({'improved' if improved else 'no-improve'}) | "
                    f"EI_max={ei_max:.3g} | mu={mu_next:.6g} | std={std_next:.3g}"
                )
                sr_params = self.decode_theta_sr(theta_next)
                print(f"      proposed (s,r,delta) per drone:\n      {sr_params}")
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
            print(f"[BO SR] done: best_y={best_y:.6g}")
            print(f"[BO SR] best params:\n{best_params}")

        return best_theta, best_params, best_y, (X, y), y_nexts, y_bests

    def run_heuristic_search(
        self,
        n_evals: int = 50,
        K_grid: int = 500,
        boundary_field: str = "affected",
        n_r: int = 160,
        smooth_iters: int = 350,
        omega: float = 1.0,
        *,
        heuristic_random_frac: float = 0.0,
        heuristic_kwargs: dict | None = None,
        verbose: bool = True,
        print_every: int = 1,
    ):
        """
        Heuristic search baseline (value-blocking only, repeated sampling).
        """
        if self.sr_grid is None:
            self.setup_search_grid_sr(
                K=K_grid,
                boundary_field=boundary_field,
                n_r=n_r,
                smooth_iters=smooth_iters,
                omega=omega,
            )
            if verbose:
                print(f"[Heuristic SR] Search grid set up with {self.sr_grid.shape[0]}x{self.sr_grid.shape[1]} points.")
                self.fire_model.plot_search_domain(self.search_domain_mask, title="Search Domain (Between Boundaries)")
                self.plot_sr_domain()

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

        for i, theta in enumerate(np.atleast_2d(thetas), start=1):
            y_val = float(self.expected_value_burned_area(theta))
            y_vals.append(y_val)

            if y_val < best_y:
                best_y = y_val
                best_theta = theta

            y_bests.append(best_y)

            if verbose and (i % max(print_every, 1) == 0 or i == 1 or i == n_evals):
                sr_params = self.decode_theta_sr(theta)
                params = self.decode_theta(theta)
                print(
                    f"[Heuristic SR] eval {i:03d}/{n_evals} | y={y_val:.6g} | best={best_y:.6g}\n"
                    f"              (s,r,delta) per drone:\n              {sr_params}\n"
                    f"              (x,y,phi) per drone:\n              {params}"
                )

        best_params = self.decode_theta(best_theta)
        X_feats = np.vstack([self.theta_to_gp_features(th) for th in np.atleast_2d(thetas)])
        y_arr = np.array(y_vals, dtype=float)
        y_nexts = list(y_vals)
        y_bests_arr = np.array(y_bests, dtype=float)

        if verbose:
            print(f"[Heuristic SR] done: best_y={best_y:.6g}")
            print(f"[Heuristic SR] best params:\n{best_params}")

        return best_theta, best_params, best_y, (X_feats, y_arr), y_nexts, y_bests_arr


__all__ = [
    "TiedSRDeltaMatern",
    "RetardantDropBayesOptSR",
    "expected_improvement",
]
