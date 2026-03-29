import numpy as np
import matplotlib.pyplot as plt

from pipeline.core import (
    homogeneous_state,
    check_ode_stability
)

# --------------------
# Dispersion relation
# --------------------
def dispersion_relation(
    jacobian_matrix: np.ndarray,
    d1: float,
    d2: float,
    k_min: float = 0,
    k_max: float = 5,
    n_k: int = 1000,
):
    """
    Computes the dispersion relation ``lambda_max(k)``.

    Algorithm:
    For each value of ``k``, the eigenvalues of the matrix
        J - k^2 D
    are computed, and the largest real part is selected.

    Parameters
    jacobian_matrix : np.ndarray
        Reaction Jacobian matrix of shape ``(2, 2)``.
    d1, d2 : float
        Diffusion coefficients.
    k_min, k_max : float, default=0
        Bounds of the analyzed wavenumber range.
    n_k : int
        Number of uniformly spaced wavenumber values.

    Returns
    tuple[np.ndarray, np.ndarray]
        Pair ``(k_values, lambda_max)``, where:
        - ``k_values`` is the array of wavenumbers,
        - ``lambda_max`` is the array of maximal real parts of eigenvalues.
    """
    k_values = np.linspace(k_min, k_max, n_k)
    lambda_max = np.zeros_like(k_values)

    # macierz dyfuzji
    diffusion_matrix = np.diag([d1, d2])

    for i, k in enumerate(k_values):
        matrix = jacobian_matrix - (k**2) * diffusion_matrix
        eigenvalues = np.linalg.eigvals(matrix)
        lambda_max[i] = np.max(np.real(eigenvalues))

    return k_values, lambda_max

# ------------
# Turing band
# ------------
def turing_band(k_values: np.ndarray, lambda_max: np.ndarray):
    """
    Determines the unstable wavenumbers forming the Turing band.

    Parameters
    k_values : np.ndarray
        Array of analyzed wavenumbers.
    lambda_max : np.ndarray
        Maximal real part of the dispersion relation.

    Returns
    dict or None
        If instability exists, returns a dictionary with:
        - ``"k_min"`` : left endpoint of the unstable band,
        - ``"k_max"`` : right endpoint of the unstable band,
        - ``"k_dom"`` : wavenumber corresponding to the global maximum
          of the dispersion relation.

        Otherwise returns ``None``.
    """
    unstable_k = k_values[lambda_max > 0]

    if unstable_k.size == 0:
        return None

    return {
        "k_min": unstable_k.min(),
        "k_max": unstable_k.max(),
        "k_dom": k_values[np.argmax(lambda_max)]
    }

# ----------------
# Turing analysis
# ----------------
def turing_analysis(
    a: float,
    m: float,
    d1: float,
    d2: float,
    k_min: float = 0,
    k_max: float = 5,
    n_k: int = 1000,
):
    """
    Performs Turing instability analysis.

    Algorithm
    1. Computes the homogeneous stationary state.
    2. Checks ODE stability.
    3. Computes the dispersion relation ``lambda(k)``.
    4. Determines the unstable Turing band.

    Parameters
    a, m, d1, d2 : float
        Model parameters.
    k_min, k_max : float
        Bounds of the analyzed wavenumber range.
    n_k : int
        Number of analyzed wavenumber values.

    Returns
    dict
        Dictionary containing:
        - ``"J"``      : Jacobian matrix,
        - ``"k"``      : array of wavenumbers,
        - ``"lambda"`` : maximal real part of eigenvalues,
        - ``"band"``   : unstable Turing band or ``None``.
    """
    stable, jacobian_matrix = check_ode_stability(a, m)

    if not stable:
        raise ValueError("Reaction system is unstable — no Turing instability")

    k_values, lambda_max = dispersion_relation(
        jacobian_matrix,
        d1, d2,
        k_min, k_max,
        n_k
    )

    band = turing_band(k_values, lambda_max)

    return {
        "J": jacobian_matrix,
        "k": k_values,
        "lambda": lambda_max,
        "band": band,
    }

# ----------------
# Dispersion plot
# ----------------
def plot_dispersion(k_values: np.ndarray, lambda_max: np.ndarray, band: dict):
    """
    Plots the dispersion relation and marks the characteristic points
    of the Turing instability band.

    Parameters
    k_values : np.ndarray
        Array of analyzed wavenumbers.
    lambda_max : np.ndarray
        Maximal real part of the dispersion relation.
    band : dict
        Dictionary describing the Turing band.

    Returns
    None
        The function does not return any value. It only displays the plot.
    """
    plt.figure(figsize=(8,5))

    plt.plot(k_values, lambda_max, color = "navy")
    plt.axhline(0, color = "black")

    k_left = band["k_min"]
    lambda_left = np.interp(k_left, k_values, lambda_max)

    k_right = band["k_max"]
    lambda_right = np.interp(k_right, k_values, lambda_max) #tu było kmax

    k_dom = band["k_dom"]
    lambda_dom = np.interp(k_dom, k_values, lambda_max)

    plt.axvline(k_left, linestyle = ":", color = "gray", label = "k_min")
    plt.axvline(k_right, linestyle = ":", color = "gray", label = "k_max")
    plt.axvline(k_dom, linestyle = "--", color = "gray", label = "k_dom")

    plt.scatter([k_left, k_right, k_dom], [lambda_left, lambda_right, lambda_dom], zorder=5)

    plt.xlabel("k")
    plt.ylabel(r"max Re($\lambda(k)$)")
    plt.title("Dispersion relation – Turing analysis")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------------
# Turing scan over the (a, m) plane
# ----------------------------------
def scan_turing_am(
    d1: float,
    d2: float,
    m_values: np.ndarray,
    a_values: np.ndarray,
    k_min: float = 0,
    k_max: float = 20,
    n_k: int = 4000,
):
    """
    Scans the parameter plane ``(a, m)`` for fixed diffusion coefficients
    and checks where Turing instability may occur.

    Parameters
    d1, d2 : float
        Diffusion coefficients.
    m_values, a_values : array-like
        Array of tested values of parameters ``m`` and ``a``.
    k_min, k_max : float
        Bounds of the analyzed wavenumber range.
    n_k : int, default=4000
        Number of wavenumber values used in the dispersion analysis.

    Returns
    list[dict]
        List of dictionaries. Each dictionary contains information
        about one tested pair ``(a, m)``, including the homogeneous
        state, existence of Turing instability, maximal growth rate,
        and characteristic wavenumbers.
    """
    results = []

    for m in m_values:
        for a in a_values:
            try:
                u_star, v_star = homogeneous_state(a, m)
            except:
                u_star, v_star = np.nan, np.nan

            # Skip cases without a meaningful positive state
            if not np.isfinite(u_star) or not np.isfinite(v_star) or v_star <= 1:
                results.append({
                    "a": a,
                    "m": m,
                    "u_star": np.nan,
                    "v_star": np.nan,
                    "has_state": 0,
                    "has_turing": 0,
                    "lambda_max": np.nan,
                    "k_left": np.nan,
                    "k_right": np.nan,
                    "k_dom": np.nan,
                })
                continue

            try:
                res = turing_analysis(a, m, d1, d2, k_min=k_min, k_max=k_max, n_k=n_k)
            except ValueError:
                results.append({
                    "a": a,
                    "m": m,
                    "u_star": u_star,
                    "v_star": v_star,
                    "has_state": 1,
                    "has_turing": 0,
                    "lambda_max": np.nan,
                    "k_left": np.nan,
                    "k_right": np.nan,
                    "k_dom": np.nan
                })
                continue

            lambda_values = res.get("lambda", None)
            band = res.get("band", None)

            lambda_max = np.nan
            if lambda_values is not None:
                lambda_max = np.max(np.real(lambda_values))

            k_left = np.nan
            k_right = np.nan
            k_dom = np.nan
            has_turing = 0

            if band is not None:
                k_left = band.get("k_min", np.nan)
                k_right = band.get("k_max", np.nan)
                k_dom = band.get("k_dom", np.nan)

                if np.isfinite(k_left) and np.isfinite(k_right) and (k_right > k_left):
                    has_turing = 1

            results.append({
                "a": a,
                "m": m,
                "u_star": u_star,
                "v_star": v_star,
                "has_state": 1,
                "has_turing": has_turing,
                "lambda_max": lambda_max,
                "k_left": k_left,
                "k_right": k_right,
                "k_dom": k_dom
            })

    return results

def unpack_scan_results(results: list[dict]):
    """
    Converts the list of dictionaries returned by ``scan_turing_am``
    into NumPy arrays.

    Parameters
    results : list[dict]
        Output returned by ``scan_turing_am``.

    Returns
    tuple
        Tuple of NumPy arrays in the following order:
        ``(a_array, m_array, has_state_array, has_turing_array,
        lambda_max_array, v_star_array, k_left_array,
        k_right_array, k_dom_array)``.
    """
    a_array = np.array([r["a"] for r in results])
    m_array = np.array([r["m"] for r in results])
    has_state_array = np.array([r["has_state"] for r in results])
    has_turing_array = np.array([r["has_turing"] for r in results])
    lambda_max_array = np.array([r["lambda_max"] for r in results])
    v_star_array = np.array([r["v_star"] for r in results])
    k_left_array = np.array([r["k_left"] for r in results])
    k_right_array = np.array([r["k_right"] for r in results])
    k_dom_array = np.array([r["k_dom"] for r in results])

    return (
        a_array,
        m_array,
        has_state_array,
        has_turing_array,
        lambda_max_array,
        v_star_array,
        k_left_array,
        k_right_array,
        k_dom_array,
    )

def plot_lambda_map(results: list[dict], ax=None):
    """
    Plots the map of the maximal growth rate ``max Re(lambda)``
    in the parameter plane ``(a, m)``.

    Parameters
    results : list[dict]
        Output returned by ``scan_turing_am``.
    ax : matplotlib.axes.Axes | None, default=None
        Axis on which the plot is drawn. If ``None``, a new figure
        and axis are created.

    Returns
    None
        The function does not return any value. It only draws the plot.
    """
    (
        a_array,
        m_array,
        has_state_array,
        has_turing_array,
        lambda_max_array,
        v_star_array,
        k_left_array,
        k_right_array,
        k_dom_array,
    ) = unpack_scan_results(results)

    mask = np.isfinite(lambda_max_array)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(a_array[mask], m_array[mask], c=lambda_max_array[mask], s=18)
    plt.colorbar(scatter, ax=ax, label=r"maxRe($\lambda$)")
    ax.set_xlabel("a")
    ax.set_ylabel("m")
    ax.set_title(r"Map of max Re($\lambda$)")
    ax.grid(True, alpha=0.3)

def plot_turing_regions(results: list[dict], ax = None):
    """
    Plots the classification of points in the parameter plane ``(a, m)``
    into no state, stable state, and Turing region.

    Parameters
    results : list[dict]
        Output returned by ``scan_turing_am``.
    ax : matplotlib.axes.Axes | None, default=None
        Axis on which the plot is drawn. If ``None``, a new figure
        and axis are created.

    Returns
    None
        The function does not return any value. It only draws the plot.
    """
    (
        a_array,
        m_array,
        has_state_array,
        has_turing_array,
        lambda_max_array,
        v_star_array,
        k_left_array,
        k_right_array,
        k_dom_array,
    ) = unpack_scan_results(results)

    mask_no_state = (has_state_array == 0)
    mask_state_no_turing = (has_state_array  == 1) & (has_turing_array == 0)
    mask_turing = (has_turing_array == 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(a_array[mask_no_state], m_array[mask_no_state], s=8, alpha=0.35, label="brak stanu")
    ax.scatter(a_array[mask_state_no_turing], m_array[mask_state_no_turing], s=8, alpha=0.50, label="stan stabilny")
    ax.scatter(a_array[mask_turing], m_array[mask_turing], s=12, alpha=0.9, label="obszar Turinga")

    ax.set_xlabel("a")
    ax.set_ylabel("m")
    ax.set_title("State map in the (a, m) plane")
    ax.legend()
    ax.grid(True, alpha=0.3)

def a_m_pairs(results: list[dict], m_values: np.ndarray,):
    """
    Extracts characteristic values of parameter ``a`` for each selected
    value of ``m`` inside the Turing region.

    Parameters
    results : list[dict]
        Output returned by ``scan_turing_am``.
    m_values : array-like
        Array of selected values of parameter ``m``.

    Returns
    list[dict]
        List of dictionaries. For each value of ``m``, the output contains
        the point with the largest ``lambda_max`` and a selected band
        of strong Turing points.
        """
    out = []

    for m in m_values:
        data_for_m = []

        # zbieramy tylko punkty z tym m i has_turing = 1
        for r in results:
            if (
                    r["m"] == m
                    and r["has_turing"] == 1
                    and np.isfinite(r["lambda_max"])
                    and r["lambda_max"] > 0
            ):
                data_for_m.append(r)

        # jeśli brak punktów Turinga dla tego m
        if len(data_for_m) == 0:
            out.append(
                {
                "m": m,
                "a_max": np.nan,
                "lambda_max": np.nan,
                "a_mean": np.nan,
                "lambda_mean_like": np.nan
                }
            )
            continue

        best_result = data_for_m[0]
        for result in data_for_m:
            if result["lambda_max"] > best_result["lambda_max"]:
                best_result = result

        lambda_max_val = best_result["lambda_max"]
        strong_data = []

        for result in data_for_m:
            if result["lambda_max"] >= 0.25 * lambda_max_val:
                strong_data.append(result)

        if len(strong_data) == 0:
            strong_data = data_for_m

        strong_data = sorted(strong_data, key=lambda result: result["a"])
        a_band = [result["a"] for result in strong_data]
        lambda_band = [result["lambda_max"] for result in strong_data]

        out.append(
            {
            "m": m,
            "a_max": best_result["a"],
            "lambda_max": best_result["lambda_max"],
            "a_band": a_band,
            "lambda_band": lambda_band,
            }
        )

    return out