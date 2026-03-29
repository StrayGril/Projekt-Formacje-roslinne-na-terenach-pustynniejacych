import numpy as np
import matplotlib.pyplot as plt
from pipeline.core import (
    v_steady,
    u_steady,
    make_grid,
    dirichlet_boundary_mask,
    precompute_diffusion,
    simulate_to_steady
)

# ---------------------------------------------------
# Numerical continuation with respect to parameter a
# ---------------------------------------------------
def continuation_sweep(
    a_values: np.ndarray,
    u_init: np.ndarray,
    v_init: np.ndarray,
    m: float,
    ht: float,
    lu_Au,
    lu_Av,
    boundary_mask: np.ndarray,
    max_steps: int = 500,
    eps: float = 1e-8,
    store_states: bool = True,
):
    """
    Performs numerical continuation with respect to parameter ``a``.

    For each parameter value in ``a_values``:
        1. Solves the system until a stationary state is reached.
        2. Uses the computed solution as the initial condition
           for the next parameter value.
        3. Stores selected measures of the final state.

    Recorded measures:
        - ``avg`` : spatial average of biomass,
        - ``max`` : spatial maximum of biomass.

    Parameters
    a_values : np.ndarray
        Values of parameter ``a`` in the prescribed continuation order.
    u_init, v_init : np.ndarray
        Initial condition for variables ``u``, ``v`` at the first value of ``a``.
    m : float
        Model parameter.
    ht : float
        Time step.
    lu_Au, lu_Av :
        LU factorizations of the diffusion matrices.
    boundary_mask : np.ndarray
        Mask corresponding to Dirichlet boundary conditions.
    max_steps : int, default=500
        Maximum number of iterations in the steady-state solver.
    eps : float, default=1e-8
        Convergence tolerance.
    store_states : bool, default=True
        Whether to store full states ``(u, v)`` for each parameter value.

    Returns
    dict
        Dictionary containing:
        - ``"avg"``    : array of average biomass values,
        - ``"max"``    : array of maximum biomass values,
        - ``"states"`` : list of pairs ``(u, v)`` if ``store_states=True``.
    """
    if a_values.size == 0:
        raise ValueError("a_values cannot be empty")
    if ht <= 0:
        raise ValueError("ht must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    u = u_init.copy()
    v = v_init.copy()

    avg_values = []
    max_values = []
    states = [] if store_states else None

    for a in a_values:
        u, v, _ = simulate_to_steady(
            u, v, a, m, ht, lu_Au, lu_Av,
            boundary_mask=boundary_mask, max_steps=max_steps, eps=eps
        )

        avg_values.append(float(np.mean(v)))
        max_values.append(float(np.max(v)))

        if store_states:
            states.append((u.copy(), v.copy()))

    result = {
        "avg": np.array(avg_values),
        "max": np.array(max_values)
    }

    if store_states:
        result["states"] = states

    return result

# --------------------------
# Tipping point estimation
# --------------------------
def estimate_tipping_point(
        a_values_desc: np.ndarray,
        max_series: np.ndarray
):
    """
    Estimates the critical point (tipping point) based on
    the largest jump in the biomass maximum.

    Algorithm:
        1. Computes differences between consecutive values of ``max(v)``.
        2. Selects the index of the largest absolute change.
        3. Takes the critical point as the midpoint of the interval
           between two neighboring parameter values.

    Parameters
    a_values_desc : np.ndarray
        Decreasing values of parameter ``a``.
    max_series : np.ndarray
        Biomass maxima corresponding to consecutive parameter values.

    Returns
    tuple[float, int]
        Pair ``(tp, idx)``, where:
        - ``tp``  : approximate tipping point,
        - ``idx`` : index corresponding to the largest jump.
    """
    if len(a_values_desc) != len(max_series):
        raise ValueError("a_values_desc and max_series must have the same length")
    if len(max_series) < 2:
        raise ValueError("At least 2 points are required to estimate a tipping point")

    delta_max = np.abs(np.diff(max_series))
    idx = int(np.argmax(delta_max))
    tipping_point = 0.5 * (a_values_desc[idx] + a_values_desc[idx + 1])

    return float(tipping_point), idx

# --------------------------
# Full bifurcation analysis
# --------------------------
def run_bifurcation(
    m: float,
    d1: float,
    d2: float,
    lx: float = 10,
    ly: float = 10,
    nx: int = 30,
    ny: int = 30,
    ht: float = 0.025,
    max_steps: int = 500,
    eps: float = 1e-8,
    ha: float = 5e-4,
    amax_factor: float = 4,
    a_max: float | None = None,
    store_down_states: bool = True,
):
    """
    Performs full bifurcation analysis with respect to parameter ``a``.

    Algorithm:
        1. Generates the spatial grid and Dirichlet boundary mask.
        2. Precomputes LU factorizations for the diffusion part.
        3. Performs continuation for decreasing values of ``a``.
        4. Identifies the tipping point based on the largest jump in ``max(v)``.
        5. Performs continuation for increasing values of ``a``,
           starting from a state located just above the tipping point.

    Parameters
    m, d1, d2 : float
        Model parameters.
    lx, ly : float
        Dimensions of the spatial domain.
    nx, ny : int
        Number of grid points in each dimension.
    ht : float
        Time step.
    max_steps : int
        Maximum number of iterations in the steady-state solver.
    eps : float
        Convergence tolerance.
    ha : float
        Step of parameter ``a`` in the continuation procedure.
    amax_factor : float
        Factor determining the maximal parameter value:
        ``a_max = amax_factor * m``.
    a_max : float | None = None,
        If ``None``, the value ``amax_factor * m`` is used.
        Otherwise, the maximal value of the parameter is prescribed explicitly.
    store_down_states : bool, default=True
        Whether to store full states for the decreasing branch.

    Returns
    dict
        Dictionary containing:
        - parameter series ``a_down`` and ``a_up``,
        - corresponding biomass averages and maxima,
        - approximate tipping point ``tp``,
        - index of the jump ``tp_idx``,
        - reference value ``a = 2m``,
        - grid and numerical parameter information.
    """
    if nx <= 1 or ny <= 1:
        raise ValueError("nx and ny must be greater than 1")
    if ht <= 0:
        raise ValueError("ht must be positive")
    if ha <= 0:
        raise ValueError("ha must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    if a_max is None:
        amax = amax_factor * m
    else:
        if a_max <= 0:
            raise ValueError("a_max must be positive")
        amax = a_max

    # Grid construction and Dirichlet boundary mask
    _, _, X, Y, h = make_grid(lx, ly, nx, ny)
    boundary_mask = dirichlet_boundary_mask(X, Y, lx, ly)

    # LU factorizations for the diffusion part
    lu_Au, lu_Av = precompute_diffusion(nx, ny, h, ht, d1, d2)

    # Parameter values for the decreasing branch
    a_down = np.arange(amax, 0, -ha)

    # Start from the homogeneous ODE equilibrium for the largest a
    v0 = v_steady(a_down[0], m)
    u0 = u_steady(v0, m)
    
    u_init = np.full(nx * ny, u0)
    v_init = np.full(nx * ny, v0)
    u_init[boundary_mask] = 0
    v_init[boundary_mask] = 0

    down = continuation_sweep(
        a_values=a_down,
        u_init=u_init,
        v_init=v_init,
        m=m,
        ht=ht,
        lu_Au=lu_Au,
        lu_Av=lu_Av,
        boundary_mask=boundary_mask,
        max_steps=max_steps,
        eps=eps,
        store_states=store_down_states,
    )

    tipping_point, tp_idx = estimate_tipping_point(a_down, down["max"])

    # Initial state for the increasing branch: state just above tp
    idx_start = np.where(a_down > tipping_point)[0][-1]
    u_start, v_start = down["states"][idx_start]

    # Parameter values for the increasing branch
    a_up = np.arange(tipping_point, amax, ha)

    up = continuation_sweep(
        a_values=a_up,
        u_init=u_start,
        v_init=v_start,
        m=m,
        ht=ht,
        lu_Au=lu_Au,
        lu_Av=lu_Av,
        boundary_mask=boundary_mask,
        max_steps=max_steps,
        eps=eps,
        store_states=store_down_states,
    )

    return {
        "a_down": a_down,
        "down_avg": down["avg"],
        "down_max": down["max"],
        "a_up": a_up,
        "up_avg": up["avg"],
        "up_max": up["max"],
        "tp": tipping_point,
        "tp_idx": tp_idx,
        "a_2m": 2.0 * m,
        "brzeg": boundary_mask,
        "grid": {"lx": lx, "ly": ly, "nx": nx, "ny": ny, "h": h},
        "params": {"m": m, "d1": d1, "d2": d2, "ht": ht, "max_steps": max_steps, "eps": eps, "ha": ha},
    }

# -------------------------------------
# Plot of the full bifurcation diagram
# -------------------------------------
def plot_bifurcation(
        result: dict,
        title: str | None = None,
        show: bool = True,
        ax=None,
):
    """
    Plots the bifurcation diagram with respect to parameter ``a``.

    For the decreasing branch (``a`` decreasing) and increasing branch
    (``a`` increasing), the following quantities are shown:
        - spatial average of biomass,
        - spatial maximum of biomass,
        - reference value ``a = 2m``,
        - approximate tipping point.


    Parameters
    result : dict
        Dictionary returned by ``run_bifurcation``.
    title : str | None, default=None
        Plot title.
    show : bool, default=True
        Whether to call ``plt.show()``.
    ax : matplotlib.axes.Axes | None, default=None
        Axis on which the plot should be drawn.

    Returns
    matplotlib.axes.Axes
        Axis with the plotted bifurcation diagram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    a_down = result["a_down"]
    a_up = result["a_up"]

    ax.scatter(a_down, result["down_avg"], s=8, color="cyan", label=r"$v_{avg}$ for a$\downarrow$")
    ax.scatter(a_down, result["down_max"], s=8, color="blue", label=r"$v_{\max}$ for a$\downarrow$")

    ax.scatter(a_up, result["up_avg"], s=8, color="red", label=r"$v_{avg}$ for a$\uparrow$")
    ax.scatter(a_up, result["up_max"], s=8, color="orange", label=r"$v_{\max}$ for a$\uparrow$")

    ax.axvline(x=result["a_2m"], color="black", linestyle=":", label="a = 2m")
    ax.axvline(x=result["tp"], color="purple", linestyle=":", label=rf"$tp \approx {result['tp']:.4f}$")

    ax.set_xlabel(r"$a$")
    ax.set_ylabel("Biomass in the stationary state")

    if title is None:
        title = rf"Bifurcation diagram for $d_1 = {result['params']['d1']:.2f}$"

    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if show:
        plt.show()

    return ax

# -------------------------------------------
# Bifurcation analysis for decreasing a only
# -------------------------------------------
def run_bifurcation_down(
    m: float,
    d1: float,
    d2: float,
    lx: float = 10,
    ly: float = 10,
    nx: int = 30,
    ny: int = 30,
    ht: float = 0.025,
    max_steps: int = 200,
    eps: float = 1e-5,
    ha: float = 5e-4,
    amax_factor: float = 4,
    a_max: float | None = None,
    store_down_states: bool = False,
):
    """
    Performs bifurcation analysis with respect to parameter ``a``
    only for decreasing values.

    Algorithm:
        1. Generates the spatial grid and Dirichlet boundary mask.
        2. Precomputes LU factorizations for the diffusion part.
        3. Performs continuation for decreasing values of ``a``.
        4. Identifies the tipping point based on the largest jump in ``max(v)``.
        5. Detects discrete local maxima of the ``down_max`` series.

    Parameters
    m, d1, d2 : float
        Model parameters.
    lx, ly : float
        Dimensions of the spatial domain.
    nx, ny : int
        Number of grid points in each dimension.
    ht : float
        Time step.
    krok_max : int
        Maximum number of iterations in the steady-state solver.
    eps : float
        Convergence tolerance
    ha : float
        Step of parameter ``a`` in the continuation procedure.
    amax_factor : float
        Factor determining the maximal parameter value:
        ``a_max = amax_factor * m``.
    a_max : float | None = None,
        If ``None``, the value ``amax_factor * m`` is used.
        Otherwise, the maximal value of the parameter is prescribed explicitly.
    store_down_states : bool
        Whether to store full states for the decreasing branch.

    Zwraca
    dict
        Dictionary containing:
        - parameter series ``a_down``,
        - corresponding biomass averages and maxima,
        - approximate tipping point ``tp``,
        - jump index ``tp_idx``,
        - indices of local maxima ``peak_idx``,
        - logical mask ``peak_mask``,
        - optionally stored states,
        - reference value ``a = 2m``,
        - grid and numerical parameter information.
    """
    if nx <= 1 or ny <= 1:
        raise ValueError("nx and ny must be greater than 1")
    if ht <= 0:
        raise ValueError("ht must be positive")
    if ha <= 0:
        raise ValueError("ha must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")
    
    if a_max is None:
        amax = amax_factor * m
    else:
        if a_max <= 0:
            raise ValueError("a_max must be positive")
        amax = a_max

    # Grid construction and Dirichlet boundary mask
    _, _, X, Y, h = make_grid(lx, ly, nx, ny)
    boundary_mask = dirichlet_boundary_mask(X, Y, lx, ly)

    # LU factorizations for the diffusion part
    lu_Au, lu_Av = precompute_diffusion(nx, ny, h, ht, d1, d2)

    # Parameter values for the decreasing branch
    a_down = np.arange(amax, 0, -ha)

    # Start from the homogeneous ODE equilibrium for the largest a
    v0 = v_steady(a_down[0], m)
    u0 = u_steady(v0, m)

    u_init = np.full(nx * ny, u0)
    v_init = np.full(nx * ny, v0)
    u_init[boundary_mask] = 0
    v_init[boundary_mask] = 0

    down = continuation_sweep(
        a_values=a_down,
        u_init=u_init,
        v_init=v_init,
        m=m,
        ht=ht,
        lu_Au=lu_Au,
        lu_Av=lu_Av,
        boundary_mask=boundary_mask,
        #boundary_mask=boundary_mask,
        eps=eps,
        store_states=store_down_states,
    )

    tipping_point, tp_idx = estimate_tipping_point(a_down, down["max"])

    max_series = down["max"]
    peak_idx = np.where(
        (max_series[1:-1] > max_series[:-2]) &
        (max_series[1:-1] > max_series[2:])
    )[0] + 1

    peak_mask = np.zeros(len(max_series), dtype=bool)
    peak_mask[peak_idx] = True

    return {
        "a_down": a_down,
        "down_avg": down["avg"],
        "down_max": down["max"],
        "tp": tipping_point,
        "tp_idx": tp_idx,
        "peak_idx": peak_idx,
        "peak_mask": peak_mask,
        "down_states": down["states"] if store_down_states else None,
        "a_2m": 2.0 * m,
        "boundary_mask": boundary_mask,
        "grid": {"lx": lx, "ly": ly, "nx": nx, "ny": ny, "h": h},
        "params": {"m": m, "d1": d1, "d2": d2, "ht": ht, "krok_max": max_steps, "eps": eps, "ha": ha},
    }

# -------------------------------------------------
# Plot of the bifurcation diagram for decreasing a
# -------------------------------------------------
def plot_bifurcation_down(
    result: dict,
    title: str | None = None,
    show: bool = True,
    ax=None,
    show_peaks: bool = True,
):
    """
    Plots the bifurcation diagram only for the branch
    corresponding to decreasing values of ``a``.

    Parameters
    result : dict
        Dictionary returned by ``run_bifurcation_down``.
    title : str | None
        Plot title.
    show : bool
        Whether to call ``plt.show()``.
    ax : matplotlib.axes.Axes | None
        Axis on which the plot should be drawn.
    show_peaks : bool, default=True
        Whether to mark local maxima of the ``down_max`` series.

    Returns
    matplotlib.axes.Axes
        Axis with the plotted bifurcation diagram.
    """
    created_ax = ax is None

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    else:
        fig = ax.figure

    a_down = result["a_down"]
    avg_values = result["down_avg"]
    max_values = result["down_max"]
    params = result["params"]

    ax.scatter(a_down, avg_values, s=8, color="black", label=r"$v_{\mathrm{avg}}$")
    ax.scatter(a_down, max_values, s=8, color="green", label=r"$v_{max}$")

    # Local maxima of max(v)
    if show_peaks and "peak_mask" in result:
        peak_mask = result["peak_mask"]
        ax.scatter(
            a_down[peak_mask],
            max_values[peak_mask],
            s=20,
            color="red",
            zorder=3,
            label="local maxima"
        )

    # Reference lines
    ax.axvline(result["a_2m"], linestyle=":", color="black", label=r"$a = 2m$")

    if result.get("tp") is not None:
        ax.axvline(result["tp"], linestyle=":", color="purple",
                   label=rf"$tp \approx {result['tp']:.4f}$")

    ax.set_xlabel("a")
    ax.set_ylabel("Biomass in the stationary state")

    if title is None:
        title = rf"$d_1={params['d1']:.6f},\ d_2={params['d2']:.6f}$"

    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if show and created_ax:
        plt.show()

    return ax