import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

# ============================================================
# DIMENSIONAL TO DIMENSIONLESS CONVERSION
# ============================================================
def dimensional_to_dimensionless(A, L, R, DW, J, M, DN, lx):
    """
    Converts dimensional model parameters into dimensionless parameters.

    Parameters
    A : float
        Water input.
    L : float
        Water loss coefficient.
    R : float
        Biomass growth-related coefficient.
    DW : float
        Water diffusion coefficient.
    J : float
        Water uptake coefficient by biomass.
    M : float
        Biomass mortality coefficient.
    DN : float
        Biomass diffusion coefficient.
    lx : float
        Side length of the spatial domain.

    Returns
    tuple[float, float, float, float]
        Dimensionless parameters:
        - ``a``  : dimensionless input,
        - ``m``  : dimensionless biomass mortality,
        - ``d1`` : dimensionless water diffusion,
        - ``d2`` : dimensionless biomass diffusion.

    Raises
    ValueError
        If any parameter is non-positive or if the discriminant
        associated with the positive stationary state is negative.

    Notes
    The function uses the scaling:
    N0 = sqrt(L / R),
    W0 = sqrt(L / (J^2 * R)),
    T0 = 1 / R,
    X0 = lx.

    Then the dimensionless parameters are computed as:
    a = A / (L * W0),
    m = M / L,
    d1 = DW / (L * X0^2),
    d2 = DN / (L * X0^2).
    """
    if any(x <= 0 for x in (A, L, R, DW, J, M, DN, lx)):
        raise ValueError("All parameters must be positive")

    if A**2 - ((4 * L * M**2) / (J**2 * R)) < 0:
        raise ValueError("Negative discriminant for the positive stationary state")

    N0 = np.sqrt(L / R)
    W0 = np.sqrt(L / (J**2 * R))
    T0 = 1 / R
    X0 = lx

    a = A / (L * W0)
    m = M / L
    d1 = DW / (L * X0**2)
    d2 = DN / (L * X0**2)

    return a, m, d1, d2

# ============================================================
# STATIONARY STATES AND REACTION PART OF THE MODEL
# ============================================================
def v_steady(a: float, m: float, mode: int = 1, add_delta: bool = True):
    """
    Computes the positive homogeneous stationary value ``v*`` of the
    reaction system from the corresponding quadratic equation.

    Parameters
    a : float
        Control parameter.
    m : float
        Linear loss parameter.
    mode : int
        Behavior for negative discriminant:
        - ``1`` returns ``0``,
        - otherwise raises ``ValueError``.
    add_delta : bool
        If ``True``, uses the ``+sqrt(delta)`` branch.
        Otherwise, uses the ``-sqrt(delta)`` branch.

    Returns
    float
        Stationary value ``v*`` if it exists.
    """
    delta = a * a - 4 * m * m
    if delta < 0:
        if mode == 1:
            return 0
        raise ValueError("Negative discriminant")

    if add_delta:
        return (a + np.sqrt(delta)) / (2 * m)
    return (a - np.sqrt(delta)) / (2 * m)

def u_steady(v: float, m: float):
    """
    Computes the corresponding stationary value ``u*`` for a given ``v*``.

    Parameters
    v : float
        Stationary value ``v*``.
    m : float
        Model parameter.

    Returns
    float
        Stationary value ``u*``.
        If ``v <= 0``, returns ``0``.
    """
    return 0.0 if v <= 0 else m / v

def homogeneous_state(a: float, m: float):
    """
    Returns the homogeneous stationary state ``(u*, v*)``.

    Returns
    a : float
        Water input parameter.
    m : float
        Mortality parameter.

    Zwraca
    tuple[float, float]
        Homogeneous equilibrium state of the reaction system.
    """
    v_star = v_steady(a, m, mode=1, add_delta=True)
    u_star = u_steady(v_star, m)
    return u_star, v_star

# Explicitly computed derivatives
def reaction(u: np.ndarray, v: np.ndarray, a: float, m: float):
    """
    Computes the reaction part of the system without diffusion.

    The reaction terms are:
        du/dt = a - u - u v²
        dv/dt = u v² - m v

    Parameters
    u, v : np.ndarray
        Current values of variables ``u`` and ``v``.
    a, m : float
        Model parameters.

    Returns
    tuple[np.ndarray, np.ndarray]
        Pair ``(du, dv)`` of reaction time derivatives.
    """
    vv = v * v
    du = a - u - u * vv
    dv = u * vv - m * v
    return du, dv

def jacobian(u_steady_value: float, v_steady_value: float, m: float):
    """
    Computes the Jacobian matrix of the reaction part at a stationary state.

    Parameters
    u_steady_value : float
        Stationary value ``u*``.
    v_steady_value : float
        Stationary value ``v*``.
    m : float
        Model parameter.

    Returns
    np.ndarray
        Jacobian matrix.
        """
    return np.array([
        [-1 - v_steady_value ** 2, -2 * u_steady_value * v_steady_value],
        [v_steady_value ** 2, 2 * u_steady_value * v_steady_value - m],
    ])

def check_ode_stability(a: float, m: float):
    """
    Checks the stability of the reaction system without diffusion.

    Parameters
    a, m : float
        Model parameters.

    tuple[bool, np.ndarray]
        Pair ``(stable, J)``, where:
        - ``stable`` indicates whether the homogeneous state is stable,
        - ``J`` is the Jacobian matrix at ``(u*, v*)``.
    """
    u_star, v_star = homogeneous_state(a, m)
    J = jacobian(u_star, v_star, m)

    trace_J = np.trace(J)
    det_J = np.linalg.det(J)

    stable = (trace_J < 0) and (det_J > 0)
    return stable, J

# ==============================================
# SPATIAL DISCRETIZATION AND LAPLACIAN OPERATOR
# ==============================================
def D2(n: int) -> np.ndarray:
    """
    Constructs the sparse 1D second-derivative matrix
    using the standard tridiagonal finite-difference stencil.

    Parameters
    n : int
        Matrix size.

    Zwraca
    scipy.sparse.csr_matrix
        Sparse ``n x n`` matrix approximating ``d²/dx²``.
    """
    diagonals = [
        np.ones(n - 1),
        -2.0 * np.ones(n),
        np.ones(n - 1),
    ]
    offsets = [-1, 0, 1]
    return diags(diagonals, offsets, shape=(n, n), format="csr")

def laplacian2D(nx: int, ny: int, h: float):
    """
    Constructs the sparse discrete 2D Laplacian matrix
    on a rectangular grid with spacing ``h``.

    Parameters
    nx, ny : int
        Number of grid points in the dimensions.
    h : float
        Grid spacing, assuming ``hx = hy``.

    Zwraca
    scipy.sparse.csr_matrix
        Sparse matrix of shape ``(nx * ny, nx * ny)``
        representing the discrete 2D Laplacian.
    """
    d2x = D2(nx)
    d2y = D2(ny)
    ix = eye(nx, format="csr")
    iy = eye(ny, format="csr")
    
    laplacian = (kron(iy, d2x) + kron(d2y, ix)) / (h * h)
    return laplacian.tocsr()

def make_grid(lx: float, ly: float, nx: int, ny: int):
    """
    Generates a uniform rectangular grid on the domain
    ``[0, lx] x [0, ly]`` together with the spatial step size.

    Parameters
    lx, ly : float
        Domain's lengths.
    nx, ny : int
        Numbers of grid points.

    Returns
    tuple
        ``(x, y, X, Y, h)``, where:
        - ``x, y`` are coordinate vectors,
        - ``X, Y`` are meshgrid arrays,
        - ``h`` is the spatial step size, assuming ``hx = hy``.

    Raises
    ValueError
        If the grid spacing in the ``x`` and ``y`` directions is not equal.
    """
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    X, Y = np.meshgrid(x, y)

    hx = x[1] - x[0]
    hy = y[1] - y[0]

    if hx != hy:
        raise ValueError("The grid requires hx = hy")
    h = hx

    return x, y, X, Y, h


def dirichlet_boundary_mask(X: np.ndarray, Y: np.ndarray, lx: float, ly: float):
    """
    Computes the logical mask for Dirichlet boundary conditions
    on a rectangular domain.

    Parameters
    X, Y : np.ndarray
        Grid coordinates.
    lx, ly : float
        Domain lengths.

    Returns
    np.ndarray
        Flattened boolean mask indicating boundary points.
    """
    Xf = X.flatten()
    Yf = Y.flatten()

    boundary = (Xf == 0.0) | (Xf == lx) | (Yf == 0.0) | (Yf == ly)
    return boundary

# ===========
# PDE SOLVER
# ===========
# Implicit diffusion
def precompute_diffusion(
        nx: int,
        ny: int,
        h: float,
        ht: float,
        d1: float,
        d2: float
):
    """
    Precomputes LU factorizations of matrices corresponding to the
    implicit time-stepping scheme for the diffusion part.

    At each time step, the following systems are solved:
        (I - dt * d1 * L) u^{n+1} = u^n + dt * f_u(u^n, v^n)
        (I - dt * d2 * L) v^{n+1} = v^n + dt * f_v(u^n, v^n)

    where:
        - ``L`` is the discrete Laplacian,
        - ``dt`` is the time step,
        - ``d1, d2`` are diffusion coefficients,
        - ``f_u, f_v`` are the reaction terms.

    Parameters
    nx, ny : int
        Numbers of grid points.
    h : float
        Spatial step size, assuming ``hx = hy``.
    ht : float
        Time step.
    d1, d2 : float
        Diffusion coefficients.

    Returns
    tuple
        Pair ``(lu_Au, lu_Av)``, where each element is an LU factorization
        object for the corresponding sparse matrix.
    """
    L = laplacian2D(nx, ny, h)
    I = eye(nx * ny, format="csc")

    Au = (I - ht * d1 * L).tocsc()
    Av = (I - ht * d2 * L).tocsc()

    lu_Au = splu(Au)
    lu_Av = splu(Av)

    return lu_Au, lu_Av

def step_reaction_diffusion(
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    m: float,
    ht: float,
    lu_Au,
    lu_Av,
    boundary_mask: np.ndarray,
):
    """
    Performs one time step using:
        - explicit Euler for the reaction part,
        - implicit solve for the diffusion part.

    The numerical scheme is:
        (I - dt d1 L) u^{n+1} = u^n + dt f_u(u^n, v^n)
        (I - dt d2 L) v^{n+1} = v^n + dt f_v(u^n, v^n)

    Parameters
    u, v : np.ndarray
        Values at time ``t^n`` for variables ``u``, ``v``.
    a, m : float
        Model parameters.
    ht : float
        Time step.
    lu_Au, lu_Av :
        LU factorizations for the diffusion part.
    boundary_mask : np.ndarray
        Boolean mask for Dirichlet boundary conditions.

    Returns
    tuple[np.ndarray, np.ndarray]
        Pair ``(u_new, v_new)`` with values at time ``t^{n+1}``.
    """
    du, dv = reaction(u, v, a, m)

    ru = u + ht * du
    rv = v + ht * dv

    ru[boundary_mask] = 0
    rv[boundary_mask] = 0

    u_new = lu_Au.solve(ru)
    v_new = lu_Av.solve(rv)

    u_new = np.maximum(u_new, 0)
    v_new = np.maximum(v_new, 0)

    u_new[boundary_mask] = 0
    v_new[boundary_mask] = 0

    return u_new, v_new

def simulate_to_steady(
    u0: np.ndarray,
    v0: np.ndarray,
    a: float,
    m: float,
    ht: float,
    lu_Au,
    lu_Av,
    boundary_mask: np.ndarray,
    max_steps: int = 500,
    eps: float = 1e-6,
    check_every=10,
):
    """
    Iterates the time-stepping scheme until a stationary state is reached.

    The stopping criterion is:
        ||v^{n+1} - v^n|| < tol

    evaluated every ``check_every`` steps on interior points only.

    Parameters
    u0, v0 : np.ndarray
        Initial conditions.
    a, m : float
        Model parameters.
    ht : float
        Time step.
    lu_Au, lu_Av :
        LU factorizations of the diffusion matrices.
    boundary_mask : np.ndarray
        Boolean mask for Dirichlet boundary conditions.
    max_steps : int
        Maximum number of iterations.
    eps : float
        Convergence tolerance.
    check_every : int
        Frequency of the convergence check.

    Returns
    tuple[np.ndarray, np.ndarray, int]
        Triple ``(u, v, n_steps)``, where:
        - ``u, v`` are the final states,
        - ``n_steps`` is the number of iterations performed.
    """
    u = u0.copy()
    v = v0.copy()

    for i in range(max_steps):
        v_prev = v.copy()
        u, v = step_reaction_diffusion(u, v, a, m, ht, lu_Au, lu_Av, boundary_mask=boundary_mask)

        # test zbieżności
        if (i + 1) % check_every == 0:
            err = np.max(np.abs(v[~boundary_mask] - v_prev[~boundary_mask]))
            if err < eps:
                return u, v, i + 1

    return u, v, max_steps

