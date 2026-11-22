"""An empirical violin restricted to the support of the observed draws."""
import numpy as np


def violin(ax, values, position, color, *, side=0, halfwidth=.33, bins=26,
           interval=True, horizontal=False, point=None):
    """Draw empirical density and an equal-tailed interval on data axes.

    Each shape has its own maximum width. The interval is calculated from the
    original draws. Repeated boundary estimates stay in their occupied bin.
    """
    data = np.asarray(values).ravel()
    data = data[np.isfinite(data)]
    if not data.size:
        return
    def line(x, y, **kw):
        return ax.plot(y, x, **kw) if horizontal else ax.plot(x, y, **kw)
    if np.ptp(data) > 0:
        from scipy.ndimage import gaussian_filter1d
        counts, edges = np.histogram(data, bins=bins)
        # Retain a histogram when a substantial point mass (e.g. a MAP
        # boundary) makes a smoothed density misleading.
        _, frequencies = np.unique(data.round(7), return_counts=True)
        discrete = frequencies.max() >= max(3, .08 * data.size)
        if discrete:
            widths = halfwidth * counts / counts.max()
            grid = np.repeat(edges, 2)[1:-1]
            widths = np.repeat(widths, 2)
        else:
            centers = .5 * (edges[:-1] + edges[1:])
            fine = np.linspace(data.min(), data.max(), 151)
            smooth = gaussian_filter1d(counts.astype(float), 1.35, mode="constant")
            density = np.interp(fine, np.r_[edges[0], centers, edges[-1]],
                                 np.r_[0, smooth, 0])
            widths = halfwidth * density / density.max()
            grid = fine
        left = position - (widths if side <= 0 else np.zeros_like(widths))
        right = position + (widths if side >= 0 else np.zeros_like(widths))
        fill = ax.fill_between if horizontal else ax.fill_betweenx
        fill(grid, left, right, facecolor=color, edgecolor=color, alpha=.55, linewidth=.5)
    else:
        end = [position - halfwidth * (side <= 0), position + halfwidth * (side >= 0)]
        line(end, [data[0], data[0]], color=color, lw=2)
    if interval:
        q = np.quantile(data, [.025, .5, .975])
        x = position + side * halfwidth * .18
        line([x, x], [q[0], q[2]], color=color, lw=2.1, solid_capstyle="round")
        line(x, q[1], color="white", marker="o", mec=color, ms=3.6, mew=.75)
    if point is not None and np.isfinite(point):
        line(position, point, color="#26303a", marker="D", ms=3)
