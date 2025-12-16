import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.path import Path


@dataclass(frozen=True)
class FireBoundary:
    t: int
    p_boundary: float
    s: np.ndarray  # (K,) uniform in [0,1)
    xy: np.ndarray  # (K,2) points (x_cell, y_cell)
    closed: bool = True


def _polyline_length(xy: np.ndarray) -> float:
    d = np.diff(xy, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _signed_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _resample_closed_polyline(xy: np.ndarray, K: int) -> np.ndarray:
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])

    d = np.diff(xy, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    L = float(np.sum(seg))
    if L <= 1e-12:
        raise ValueError("Boundary contour has near-zero length.")

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s_targets = np.linspace(0.0, L, K, endpoint=False)

    x = np.interp(s_targets, cum, xy[:, 0])
    y = np.interp(s_targets, cum, xy[:, 1])
    return np.stack([x, y], axis=1)


def extract_fire_boundary(
    firestate,
    *,
    K: int,
    p_boundary: float = 0.5,
    field: str = "affected",
    anchor: str = "max_x",
    ccw: bool = True,
) -> FireBoundary:
    burning = firestate.burning[0]
    burned = firestate.burned[0]

    is_prob = np.issubdtype(burning.dtype, np.floating) or np.issubdtype(burned.dtype, np.floating)

    if field == "burning":
        p = burning.astype(float) if not is_prob else np.clip(burning, 0.0, 1.0)
    elif field == "burned":
        p = burned.astype(float) if not is_prob else np.clip(burned, 0.0, 1.0)
    elif field == "affected":
        p = (burning | burned).astype(float) if not is_prob else np.clip(burning + burned, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown field={field}")

    fig = plt.figure()
    try:
        CS = plt.contour(p.T, levels=[p_boundary])
        if hasattr(CS, "allsegs") and len(CS.allsegs) > 0:
            segs = CS.allsegs[0]
        elif hasattr(CS, "collections") and CS.collections:
            segs = [path.vertices for path in CS.collections[0].get_paths()]
        else:
            segs = []
    finally:
        plt.close(fig)

    segs = [s for s in segs if isinstance(s, np.ndarray) and s.shape[0] >= 3]
    if not segs:
        raise ValueError(f"No boundary found for level p_boundary={p_boundary}. Try a different level.")

    xy0 = max(segs, key=_polyline_length)

    if not np.allclose(xy0[0], xy0[-1]):
        xy0 = np.vstack([xy0, xy0[0]])

    if ccw and _signed_area(xy0) < 0:
        xy0 = xy0[::-1].copy()
    if (not ccw) and _signed_area(xy0) > 0:
        xy0 = xy0[::-1].copy()

    xyK = _resample_closed_polyline(xy0, K)

    if anchor == "max_x":
        i0 = int(np.argmax(xyK[:, 0]))
    elif anchor == "min_x":
        i0 = int(np.argmin(xyK[:, 0]))
    elif anchor == "max_y":
        i0 = int(np.argmax(xyK[:, 1]))
    elif anchor == "min_y":
        i0 = int(np.argmin(xyK[:, 1]))
    else:
        raise ValueError(f"Unknown anchor={anchor}")

    xyK = np.roll(xyK, -i0, axis=0)
    s = np.linspace(0.0, 1.0, K, endpoint=False)

    return FireBoundary(t=int(firestate.t), p_boundary=float(p_boundary), s=s, xy=xyK, closed=True)


def plot_fire_boundary(
    firestate,
    boundary: FireBoundary,
    *,
    field: str = "affected",
    title: str | None = None,
    show_points: bool = True,
):
    burning = firestate.burning[0]
    burned = firestate.burned[0]

    is_prob = np.issubdtype(burning.dtype, np.floating) or np.issubdtype(burned.dtype, np.floating)

    if field == "burning":
        p = burning.astype(float) if not is_prob else np.clip(burning, 0.0, 1.0)
        cbar = "P(burning)"
    elif field == "burned":
        p = burned.astype(float) if not is_prob else np.clip(burned, 0.0, 1.0)
        cbar = "P(burned)"
    elif field == "affected":
        p = (burning | burned).astype(float) if not is_prob else np.clip(burning + burned, 0.0, 1.0)
        cbar = "P(affected)"
    else:
        raise ValueError(f"Unknown field={field}")

    plt.figure(figsize=(6, 5))
    im = plt.imshow(p.T, origin="lower", vmin=0.0, vmax=1.0, aspect="equal")
    plt.colorbar(im, label=cbar)

    xy = boundary.xy
    plt.plot(xy[:, 0], xy[:, 1], linewidth=2, label=f"p={boundary.p_boundary:g} contour", color="tab:red")

    if show_points:
        plt.scatter(xy[:, 0], xy[:, 1], s=10)

    plt.xlabel("x cell")
    plt.ylabel("y cell")
    plt.title(title if title is not None else f"Fire boundary at t={boundary.t}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def between_boundaries_mask(boundary0_xy, boundary1_xy, grid_size):
    nx, ny = grid_size

    X = np.arange(nx)[:, None]
    Y = np.arange(ny)[None, :]
    pts = np.stack([np.broadcast_to(X, (nx, ny)).ravel(), np.broadcast_to(Y, (nx, ny)).ravel()], axis=1)

    path0 = Path(boundary0_xy, closed=True)
    path1 = Path(boundary1_xy, closed=True)

    inside0 = path0.contains_points(pts).reshape(nx, ny)
    inside1 = path1.contains_points(pts).reshape(nx, ny)

    if inside0.sum() >= inside1.sum():
        outer, inner = inside0, inside1
    else:
        outer, inner = inside1, inside0

    return outer & (~inner)


def candidates_from_mask(mask, *, stride=1):
    xs, ys = np.where(mask)
    if stride > 1:
        keep = (xs % stride == 0) & (ys % stride == 0)
        xs, ys = xs[keep], ys[keep]
    return np.stack([xs.astype(float), ys.astype(float)], axis=1)


__all__ = [
    "FireBoundary",
    "extract_fire_boundary",
    "plot_fire_boundary",
    "between_boundaries_mask",
    "candidates_from_mask",
]
