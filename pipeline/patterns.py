import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from pipeline.core import (
    make_grid,
    dirichlet_boundary_mask,
    homogeneous_state,
    precompute_diffusion,
    step_reaction_diffusion
)

cmap_u = LinearSegmentedColormap.from_list(
    "yellow_to_darkblue",
    ["#fde456", "#08306b"]
)
cmap_v = LinearSegmentedColormap.from_list(
    "yellow_to_darkgreen",
    ["#fde456", "#00441b"]
)

# -------------------
# Initial conditions
# --------------------
def initial_conditions(nx, ny, a, m, boundary_mask, noise=1e-3):
    """
    Creates initial conditions around the homogeneous stationary state
    with a small random perturbation.

    Parameters
    nx, ny : int
        Numbers of grid points.
    a, m : float
        Model parameters.
    boundary_mask : np.ndarray
        Boolean mask of boundary points.
    noise : float
        Amplitude of the random perturbation.

    Returns
    tuple[np.ndarray, np.ndarray]
        Initial vectors ``u`` and ``v``.
    """
    u_star, v_star = homogeneous_state(a, m)

    u = u_star * np.ones(nx * ny)
    v = v_star * np.ones(nx * ny)

    rng = np.random.default_rng()

    u += noise * rng.standard_normal(nx * ny)
    v += noise * rng.standard_normal(nx * ny)
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)

    u[boundary_mask] = 0
    v[boundary_mask] = 0

    return u, v

# -------------------
# Pattern simulation
# -------------------
def simulate_patterns(
    a: float,
    m: float,
    d1: float,
    d2: float,
    lx: float,
    ly: float,
    nx: int,
    ny: int,
    T: int,
    ht: float = 0.025,
    noise: float = 1e-2,
    return_matrices: bool = False, # old 'do-modelu'
    check_every: int = 200,
    mean_tol: float = 1e-5,
    max_tol: float = 1e-5,
    var_tol: float = 1e-5,
    early_stop: bool =True,
    verbose: bool = False,
    var_threshold: float = 1e-4,
    mean_threshold: float = 1e-3,
    back_steps: int = 50,
):
    """
    Simulates the reaction-diffusion system for a prescribed number
    of time steps.

    Logic
    1. If NaN or inf appears and a previously saved patterned state exists,
       that state is returned.
    2. If no patterned state has been detected yet, the function returns
       a previously saved valid state.
    3. If the stored history is shorter than expected, the oldest available
       valid state is returned.

    Parameters
    a, m, d1, d2 : float
        Model parameters.
    lx, ly : float
        Domain sizes.
    nx, ny : int
        Number of grid points.
    T : int
        Number of time steps.
    ht : float
        Time step.
    noise : float
        Amplitude of the random perturbation in the initial condition.
    return_matrices : bool
        If ``True``, returns final 2D matrices ``u`` and ``v``.
    check_every : int
        Frequency of computing statistics.
    tol_mean, tol_max,  tol_var: float
        Tolerance for the mean, max and variance value.
    early_stop : bool
        Whether to stop the simulation early after stabilization.
    verbose : bool
        Whether to print diagnostic information.
    var_threshold, mean_threshold : float
        Variance and mean thresholds for pattern detection.
    back_steps : int
        Number of steps used as fallback when instability is detected.

    Returns
    dict or tuple[np.ndarray, np.ndarray]
        Either a dictionary with simulation data or final matrices
        ``u`` and ``v`` if ``return_matrices=True``.
    """
    x, y, X, Y, h = make_grid(lx, ly, nx, ny)
    boundary_mask = dirichlet_boundary_mask(X, Y, lx, ly)

    lu_Au, lu_Av = precompute_diffusion(nx, ny, h, ht, d1, d2)
    u_0, v_0 = initial_conditions(nx, ny, a, m, boundary_mask, noise=noise)

    u_curr, v_curr = u_0.copy(), v_0.copy()

    stats_hist = []
    stopped_early = False
    nan_detected = False
    patterns_detected  = False

    last_step = 0

    # Last valid state
    last_valid_u = u_curr.copy()
    last_valid_v = v_curr.copy()
    last_valid_step = 0

    # Last checkpoint with patterns
    last_pattern_u  = None
    last_pattern_v = None
    last_pattern_step = None

    # History of valid states for fallback
    history = []

    for t in range(T):
        u_new, v_new = step_reaction_diffusion(
            u_curr, v_curr,
            a, m,
            ht,
            lu_Au, lu_Av,
            boundary_mask
        )

        # Stop if NaN or inf is detected
        if (not np.all(np.isfinite(u_curr))) or (not np.all(np.isfinite(v_curr))):
            nan_detected = True

            if last_pattern_u is not None:
                u_curr = last_pattern_u.copy()
                v_curr = last_pattern_v.copy()
                last_step = last_pattern_step

                if verbose:
                    print(
                        f"NaN/inf detected at step {step + 1} - returning the last "
                        f"patterned state from step {last_pattern_step}"
                    )
            elif len(history) > 0:
                step_saved, u_saved, v_saved = history[0]
                u_curr = u_saved.copy()
                v_curr = v_saved.copy()
                last_step = step_saved

                if verbose:
                    print(
                        f"NaN/inf detected at step {step + 1} - no patterned state "
                        f"available, returning saved state from step {last_step}"
                    )

            else:
                u_curr = last_valid_u.copy()
                v_curr = last_valid_v.copy()
                last_step = last_valid_step

                if verbose:
                    print(
                        f"NaN/inf detected at step {step + 1} - no patterned state "
                        f"stored, returning last valid state from step {last_valid_step}"
                    )
            break

        # Accept the computed step
        u_curr, v_curr = u_new, v_new
        last_step = t + 1

        last_valid_u = u_curr.copy()
        last_valid_v = v_curr.copy()
        last_valid_step = last_step

        history.append((last_step, u_curr.copy(), v_curr.copy()))
        if len(history) > back_steps:
            history.pop(0)

        # Compute statistics every check_every steps
        if (t + 1) % check_every == 0:
            v_inside = v_curr[~boundary_mask]
            mean_v = np.mean(v_inside)
            max_v = np.max(v_inside)
            var_v = np.var(v_inside)

            patterns_now = (var_v > var_threshold) and (mean_v > mean_threshold)

            if patterns_now:
                patterns_detected = True
                last_pattern_u = u_curr.copy()
                last_pattern_v = v_curr.copy()
                last_pattern_step = t + 1

            stats_hist.append({
                "step": t + 1,
                "mean_v": mean_v,
                "max_v": max_v,
                "var_v": var_v,
                "patterns_detected": patterns_now,
            })

            if verbose:
                print(
                    f"step={step + 1}, mean(v)={mean_v:.6e}, "
                    f"max(v)={max_v:.6e}, var(v)={var_v:.6e}, "
                    f"patterns_detected={patterns_now}"
                )

            if early_stop and len(stats_hist) >= 2:
                prev = stats_hist[-2]
                curr = stats_hist[-1]

                d_mean = abs(curr["mean_v"] - prev["mean_v"])
                d_max = abs(curr["max_v"] - prev["max_v"])
                d_var = abs(curr["var_v"] - prev["var_v"])

                if d_mean < mean_tol and d_max < max_tol and d_var < var_tol:
                    stopped_early = True

                    if last_pattern_u is not None:
                        u_curr = last_pattern_u.copy()
                        v_curr = last_pattern_v.copy()
                        last_step = last_pattern_step

                    if verbose:
                        print(f"Stopped early after stabilization at step {step + 1}")
                    break

    if return_matrices is True:
        return u_curr.reshape(ny, nx), v_curr.reshape(ny, nx)

    return {
        "X": X,
        "Y": Y,
        "u0": u_0,
        "v0": v_0,
        "uT": u_curr,
        "vT": v_curr,
        "stats_hist": stats_hist,
        "stopped_early": stopped_early,
        "nan_detected": nan_detected,
        "patterns_detected": patterns_detected,
        "last_step": last_step,
        "last_pattern_step": last_pattern_step,
    }

# -----------------
# Simulation plots
# -----------------
def plot_patterns(sim_data: dict, plot: str = "uv"):
    """
    Plots the initial and final simulation states
    for variables ``u`` and ``v``.

    Parameters
    sim_data : dict
         Dictionary returned by ``simulate_patterns``.
    plot : str
        Specifies which plots to show:
        - ``"u"``  : water only,
        - ``"v"``  : biomass only,
        - ``"uv"`` : both.

    Returns
    None
        The function only displays plots.
    """
    X, Y = sim_data["X"], sim_data["Y"]
    ny, nx = X.shape

    states = {
        "0": (sim_data["u0"], sim_data["v0"]),
        "T": (sim_data["uT"], sim_data["vT"])
    }
    
    if "u" in wykres:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        data = [states[t][0] for t in states]
        levels = np.linspace(min(d.min() for d in data), max(d.max() for d in data), 50)

        for ax, (time_label, (u, _)) in zip(axs.flat, states.items()):
            im = ax.contourf(X, Y, u.reshape(ny, nx), levels=levels, cmap=cmap_u)
            ax.set_title(f"Water u(t={time_label})")

        fig.colorbar(im, ax=axs)
        plt.show()

    if "v" in wykres:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        data = [states[t][1] for t in states]
        levels = np.linspace(min(d.min() for d in data), max(d.max() for d in data), 50)

        for ax, (time_label, (_, v)) in zip(axs.flat, states.items()):
            im = ax.contourf(X, Y, v.reshape(ny, nx), levels=levels, cmap=cmap_v)
            ax.set_title(f"Biomass v(t={time_label})")

        fig.colorbar(im, ax=axs)
        plt.show()

# -------------------
# Single matrix plot
# -------------------
def plot_matrix(
    matrix,
    plot_title: str = "Plot",
    show: bool = True,
    cmap=None,
):
    """
    Plots a 2D matrix using a contour plot.

    Parameters
    matrix : array-like
        Matrix to be plotted.
    plot_title : str, default="Plot"
        Plot title.
    show : bool, default=True
        Whether to display the plot immediately.
    cmap : matplotlib colormap, optional
        Colormap used for visualization. If ``None``, ``cmap_v`` is used.

    Returns
    None
        The function only draws the plot.
    """
    if cmap is None:
        cmap = cmap_v

    ny, nx = matrix.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    vmin = matrix.min()
    vmax = matrix.max()
    if vmin == vmax:
        vmax = vmin + 1e-8

    levels = np.linspace(vmin, vmax, 50)

    plt.figure(figsize=(5, 4))
    im = plt.contourf(X, Y, matrix, levels=levels, cmap=cmap)

    plt.colorbar(im)
    plt.title(plot_title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    if show is True:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(0.1)
