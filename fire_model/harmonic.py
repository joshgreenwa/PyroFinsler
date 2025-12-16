import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.path import Path


def _polygon_inside_mask(poly_xy: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    nx, ny = grid_size
    X = np.arange(nx)[:, None]
    Y = np.arange(ny)[None, :]
    pts = np.stack([np.broadcast_to(X, (nx, ny)).ravel(), np.broadcast_to(Y, (nx, ny)).ravel()], axis=1)
    path = Path(poly_xy, closed=True)
    return path.contains_points(pts).reshape(nx, ny)


def _rasterize_polyline_cells(poly_xy: np.ndarray, grid_size: tuple[int, int], step: float = 0.25) -> np.ndarray:
    nx, ny = grid_size
    mask = np.zeros((nx, ny), dtype=bool)
    K = poly_xy.shape[0]
    for i in range(K):
        a = poly_xy[i]
        b = poly_xy[(i + 1) % K]
        seg_len = float(np.hypot(*(b - a)))
        n = max(2, int(np.ceil(seg_len / step)))
        ts = np.linspace(0.0, 1.0, n)
        pts = a[None, :] * (1 - ts)[:, None] + b[None, :] * ts[:, None]
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, nx - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, ny - 1)
        mask[xi, yi] = True
    return mask


def _solve_laplace_dirichlet_sor(
    region_mask: np.ndarray,
    bc0_mask: np.ndarray,
    bc0_val: float,
    bc1_mask: np.ndarray,
    bc1_val: float,
    *,
    omega: float = 1.9,
    max_iters: int = 8000,
    tol: float = 1e-5,
) -> np.ndarray:
    nx, ny = region_mask.shape
    psi = np.zeros((nx, ny), dtype=float)
    psi[region_mask] = 0.5
    psi[bc0_mask] = bc0_val
    psi[bc1_mask] = bc1_val

    unknown = region_mask & (~bc0_mask) & (~bc1_mask)
    if not np.any(unknown):
        return psi

    xs, ys = np.where(unknown)
    for _ in range(max_iters):
        maxdiff = 0.0
        for x, y in zip(xs, ys):
            if x == 0 or x == nx - 1 or y == 0 or y == ny - 1:
                continue
            new = 0.25 * (psi[x + 1, y] + psi[x - 1, y] + psi[x, y + 1] + psi[x, y - 1])
            upd = (1 - omega) * psi[x, y] + omega * new
            diff = abs(upd - psi[x, y])
            if diff > maxdiff:
                maxdiff = diff
            psi[x, y] = upd

        psi[bc0_mask] = bc0_val
        psi[bc1_mask] = bc1_val
        if maxdiff < tol:
            break

    return psi


def _bilinear_sample(F: np.ndarray, x: float, y: float) -> float:
    nx, ny = F.shape
    x = float(np.clip(x, 0.0, nx - 1.001))
    y = float(np.clip(y, 0.0, ny - 1.001))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, nx - 1)
    y1 = min(y0 + 1, ny - 1)
    tx = x - x0
    ty = y - y0
    f00 = F[x0, y0]
    f10 = F[x1, y0]
    f01 = F[x0, y1]
    f11 = F[x1, y1]
    return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f10 + (1 - tx) * ty * f01 + tx * ty * f11


def _bilinear_sample_vec(Fx: np.ndarray, Fy: np.ndarray, x: float, y: float) -> np.ndarray:
    return np.array([_bilinear_sample(Fx, x, y), _bilinear_sample(Fy, x, y)], dtype=float)


def _resample_polyline_equal_flux(p: np.ndarray, w: np.ndarray, K: int) -> np.ndarray:
    N = p.shape[0]
    p_next = p[(np.arange(N) + 1) % N]
    seg = np.linalg.norm(p_next - p, axis=1)
    w_next = w[(np.arange(N) + 1) % N]
    w_avg = 0.5 * (w + w_next)

    q = w_avg * seg
    Q = np.concatenate([[0.0], np.cumsum(q)])
    Qtot = Q[-1]
    if Qtot <= 1e-12:
        s = np.linspace(0.0, 1.0, K, endpoint=False)
        idx = (s * N).astype(int)
        return p[idx].copy()

    targets = np.linspace(0.0, Qtot, K, endpoint=False)
    out = np.zeros((K, 2), dtype=float)

    j = 0
    for i, T in enumerate(targets):
        while j < N and Q[j + 1] < T:
            j += 1
        if j >= N:
            j = N - 1
        if Q[j + 1] <= Q[j] + 1e-12:
            alpha = 0.0
        else:
            alpha = (T - Q[j]) / (Q[j + 1] - Q[j])
        out[i] = (1 - alpha) * p[j] + alpha * p_next[j]
    return out


def _trace_streamline(psi, gx, gy, x0, y0, *, step=0.5, max_steps=4000, grad_eps=1e-8):
    pts = [(float(x0), float(y0))]
    psi_curr = _bilinear_sample(psi, x0, y0)

    x, y = float(x0), float(y0)
    for _ in range(max_steps):
        if psi_curr >= 1.0 - 1e-3:
            break
        g = _bilinear_sample_vec(gx, gy, x, y)
        nrm = float(np.hypot(g[0], g[1]))
        if nrm < grad_eps:
            break
        v = g / nrm
        x2 = x + step * v[0]
        y2 = y + step * v[1]
        psi2 = _bilinear_sample(psi, x2, y2)
        if psi2 < psi_curr:
            x2 = x + 0.25 * step * v[0]
            y2 = y + 0.25 * step * v[1]
            psi2 = _bilinear_sample(psi, x2, y2)
        pts.append((x2, y2))
        x, y, psi_curr = x2, y2, psi2

    return np.asarray(pts, dtype=float)


def _resample_polyline_by_arclength(poly: np.ndarray, M: int) -> np.ndarray:
    d = np.diff(poly, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    L = float(np.sum(seg))
    if L <= 1e-12:
        return np.broadcast_to(poly[-1], (M, 2)).copy()
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, L, M)
    x = np.interp(targets, s, poly[:, 0])
    y = np.interp(targets, s, poly[:, 1])
    return np.stack([x, y], axis=1)


@dataclass(frozen=True)
class HarmonicStripMap:
    t0: int
    t1: int
    p_boundary: float
    s_grid: np.ndarray  # (K,) in [0,1)
    d_grid: np.ndarray  # (M,) in [0,1]
    xy: np.ndarray  # (K,M,2)
    theta: np.ndarray  # (K,M), 0=vertical
    psi: np.ndarray  # (nx,ny)


def build_harmonic_strip_map_uniform(
    boundary0,
    boundary1,
    *,
    grid_size: tuple[int, int],
    M: int = 60,
    streamline_step: float = 0.5,
    seed_mode: str = "equal_flux",
    omega: float = 1.9,
    laplace_max_iters: int = 8000,
    laplace_tol: float = 1e-5,
    grad_eps: float = 1e-8,
) -> HarmonicStripMap:
    p0 = np.asarray(boundary0.xy, dtype=float)
    p1 = np.asarray(boundary1.xy, dtype=float)
    if p0.shape != p1.shape:
        raise ValueError(f"Boundaries must have same shape (same K). Got {p0.shape} vs {p1.shape}")
    K = p0.shape[0]

    inside0 = _polygon_inside_mask(p0, grid_size)
    inside1 = _polygon_inside_mask(p1, grid_size)
    if inside0.sum() >= inside1.sum():
        outer, inner = inside0, inside1
        bc_outer, bc_inner = p0, p1
    else:
        outer, inner = inside1, inside0
        bc_outer, bc_inner = p1, p0

    region = outer & (~inner)
    bc0_mask = _rasterize_polyline_cells(p0, grid_size, step=0.25)
    bc1_mask = _rasterize_polyline_cells(p1, grid_size, step=0.25)

    psi = _solve_laplace_dirichlet_sor(
        region_mask=region,
        bc0_mask=bc0_mask,
        bc0_val=0.0,
        bc1_mask=bc1_mask,
        bc1_val=1.0,
        omega=omega,
        max_iters=laplace_max_iters,
        tol=laplace_tol,
    )

    gx, gy = np.gradient(psi)

    if seed_mode == "equal_flux":
        w = np.array([np.hypot(*_bilinear_sample_vec(gx, gy, p0[i, 0], p0[i, 1])) for i in range(K)], dtype=float)
        w = np.maximum(w, 1e-6)
        seeds = _resample_polyline_equal_flux(p0, w, K)
        s_grid = np.linspace(0.0, 1.0, K, endpoint=False)
    elif seed_mode == "arclength":
        seeds = p0.copy()
        s_grid = np.linspace(0.0, 1.0, K, endpoint=False)
    else:
        raise ValueError("seed_mode must be 'equal_flux' or 'arclength'")

    d_grid = np.linspace(0.0, 1.0, M)
    xy = np.zeros((K, M, 2), dtype=float)
    for i in range(K):
        poly = _trace_streamline(
            psi,
            gx,
            gy,
            seeds[i, 0],
            seeds[i, 1],
            step=streamline_step,
            max_steps=4000,
            grad_eps=grad_eps,
        )
        xy[i] = _resample_polyline_by_arclength(poly, M)

    dxy = np.diff(xy, axis=1, prepend=xy[:, :1, :])
    theta = np.arctan2(dxy[:, :, 0], dxy[:, :, 1] + 1e-12)

    return HarmonicStripMap(
        t0=int(boundary0.t),
        t1=int(boundary1.t),
        p_boundary=float(boundary0.p_boundary),
        s_grid=s_grid,
        d_grid=d_grid,
        xy=xy,
        theta=theta,
        psi=psi,
    )


def plot_strip_map(
    boundary0,
    boundary1,
    strip: HarmonicStripMap,
    *,
    stride_s: int = 10,
    show_psi: bool = False,
    title: str | None = None,
):
    p0 = np.asarray(boundary0.xy, dtype=float)
    p1 = np.asarray(boundary1.xy, dtype=float)

    plt.figure(figsize=(7, 7))

    if show_psi:
        im = plt.imshow(strip.psi.T, origin="lower", aspect="equal", vmin=0.0, vmax=1.0)
        plt.colorbar(im, label="ψ (harmonic)")

    plt.plot(p0[:, 0], p0[:, 1], linewidth=2, label="Boundary 0")
    plt.plot(p1[:, 0], p1[:, 1], linewidth=2, label="Boundary 1")

    K = strip.xy.shape[0]
    for i in range(0, K, max(1, int(stride_s))):
        pts = strip.xy[i]
        plt.plot(pts[:, 0], pts[:, 1], linewidth=1)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x cell")
    plt.ylabel("y cell")
    plt.title(title if title is not None else f"Streamlines between boundaries (stride={stride_s})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def sd_to_xy_theta(strip: HarmonicStripMap, s, d, *, delta: float | None = None):
    def _sd_to_xy(strip_obj, s_val, d_val):
        s_arr = np.asarray(s_val, dtype=float)
        d_arr = np.asarray(d_val, dtype=float)

        K, M = strip_obj.xy.shape[0], strip_obj.xy.shape[1]
        s_arr = np.mod(s_arr, 1.0)
        d_arr = np.clip(d_arr, 0.0, 1.0)

        us = s_arr * K
        i0 = np.floor(us).astype(int)
        a = us - i0
        i1 = (i0 + 1) % K

        ud = d_arr * (M - 1)
        j0 = np.floor(ud).astype(int)
        b = ud - j0
        j1 = np.clip(j0 + 1, 0, M - 1)

        p00 = strip_obj.xy[i0, j0]
        p10 = strip_obj.xy[i1, j0]
        p01 = strip_obj.xy[i0, j1]
        p11 = strip_obj.xy[i1, j1]

        a_ = a[..., None]
        b_ = b[..., None]
        return (1 - a_) * (1 - b_) * p00 + a_ * (1 - b_) * p10 + (1 - a_) * b_ * p01 + a_ * b_ * p11

    xy = _sd_to_xy(strip, s, d)
    M = strip.xy.shape[1]
    if delta is None:
        delta = 0.5 / (M - 1)

    d0 = np.clip(np.asarray(d, dtype=float) - delta, 0.0, 1.0)
    d1 = np.clip(np.asarray(d, dtype=float) + delta, 0.0, 1.0)

    xy0 = _sd_to_xy(strip, s, d0)
    xy1 = _sd_to_xy(strip, s, d1)

    t = xy1 - xy0
    theta = np.arctan2(t[..., 0], t[..., 1])
    return xy, theta


__all__ = [
    "HarmonicStripMap",
    "build_harmonic_strip_map_uniform",
    "plot_strip_map",
    "sd_to_xy_theta",
    "BoundaryMap",
    "build_boundary_map",
    "plot_boundary_correspondence",
]


class BoundaryMap:
    """Simple linear correspondence between two boundaries of equal length."""

    def __init__(self, boundary0, boundary1):
        p0 = np.asarray(boundary0.xy, dtype=float)
        p1 = np.asarray(boundary1.xy, dtype=float)
        if p0.shape != p1.shape:
            raise ValueError(f"Boundaries must have same shape. Got {p0.shape} vs {p1.shape}")
        self.p0 = p0
        self.p1 = p1
        self.K = p0.shape[0]
        self.s_grid = np.linspace(0.0, 1.0, self.K, endpoint=False)
        self.theta_i = np.arctan2(p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1] + 1e-12)

    def _interp_boundary(self, s_query, p):
        s = np.mod(s_query, 1.0)
        u = s * self.K
        i0 = np.floor(u).astype(int)
        a = u - i0
        i1 = (i0 + 1) % self.K
        p00 = p[i0]
        p10 = p[i1]
        a_ = a[..., None]
        return (1 - a_) * p00 + a_ * p10

    def xy(self, s_query: float, d_query: float):
        p0q = self._interp_boundary(s_query, self.p0)
        p1q = self._interp_boundary(s_query, self.p1)
        d = np.clip(d_query, 0.0, 1.0)
        d_ = np.asarray(d)[..., None]
        return (1 - d_) * p0q + d_ * p1q

    def theta(self, s_query: float):
        p0q = self._interp_boundary(s_query, self.p0)
        p1q = self._interp_boundary(s_query, self.p1)
        v = p1q - p0q
        return np.arctan2(v[..., 0], v[..., 1] + 1e-12)


def build_boundary_map(boundary0, boundary1):
    return BoundaryMap(boundary0, boundary1)


def plot_boundary_correspondence(
    boundary0,
    boundary1,
    *,
    stride: int = 5,
    show_points: bool = True,
    title: str | None = None,
):
    bm = BoundaryMap(boundary0, boundary1)
    p0, p1 = bm.p0, bm.p1

    plt.figure(figsize=(7, 7))
    plt.plot(p0[:, 0], p0[:, 1], label="Boundary 0", linewidth=2)
    plt.plot(p1[:, 0], p1[:, 1], label="Boundary 1", linewidth=2)

    for i in range(0, bm.K, max(1, stride)):
        plt.plot([p0[i, 0], p1[i, 0]], [p0[i, 1], p1[i, 1]], color="gray", linewidth=0.8, alpha=0.6)

    if show_points:
        plt.scatter(p0[:, 0], p0[:, 1], s=10, color="tab:blue")
        plt.scatter(p1[:, 0], p1[:, 1], s=10, color="tab:orange")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x cell")
    plt.ylabel("y cell")
    plt.title(title if title is not None else "Boundary correspondence")
    plt.legend()
    plt.tight_layout()
    plt.show()
