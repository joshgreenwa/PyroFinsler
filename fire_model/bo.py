import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter, Matern, WhiteKernel, Kernel
from scipy.spatial import cKDTree

from fire_model.ca import CAFireModel, FireState


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
    ):
        self.fire_model = fire_model
        self.init_firestate = init_firestate
        self.n_drones = n_drones
        self.dim = 3 * n_drones
        self.evolution_time_s = evolution_time_s
        self.n_sims = n_sims
        self.p_boundary = fire_boundary_probability
        self.rng = default_rng() if rng is None else rng

        self.search_domain_mask = None
        self.projector = None
        self.shape = None

    def generate_search_grid(self, K=500, boundary_field="affected"):
        search_domain_mask = self.fire_model.generate_search_domain(
            T=self.evolution_time_s,
            n_sims=self.n_sims,
            init_firestate=self.init_firestate,
            ros_mps=self.fire_model.env.ros_mps,
            wind_coeff=self.fire_model.env.wind_coeff,
            diag=self.fire_model.env.diag,
            seed=None,
            p_boundary=self.p_boundary,
            K=K,
            boundary_field=boundary_field,
        )

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
    ):
        if self.projector is None:
            self.setup_search_grid(K=K_grid, boundary_field=boundary_field)
            if verbose:
                print(f"[BO] Search grid set up with {len(self.projector.coords)} valid cells in grid.")
                self.fire_model.plot_search_domain(self.search_domain_mask, title="Current Search Domain:")

        X_theta = self.sample_random_theta(n_init)
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
                params = self.decode_theta(x_next)
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
