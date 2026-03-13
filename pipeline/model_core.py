import numpy as np
from scipy.linalg import lu_factor, lu_solve

# ============================================================
# STANY STACJONARNE I CZĘŚĆ REAKCYJNA MODELU
# ============================================================

# Jednorodne stany stacjonarne (część reakcyjna)
def v_stac(a: float, m: float, mode: int = 1, add_delta = True) -> float:
    """
    Oblicza dodatni jednorodny punkt stacjonarny v* układu reakcyjnego
    wynikający z rozwiązania równania kwadratowego.

    Parametry
    a : float
        Parametr sterujący (np. dopływ/zasób).
    m : float
        Parametr liniowej utraty.
    mode : float
        1 podmienia delte na 0, 2 pozwala na ujemną delte.
    add_delta : bool
        Czy dodajemy delte? False zwraca odejmowanie

    Zwraca
    float
        Rozwiązanie v*, jeśli istnieje.
    """
    delta = a * a - 4 * m * m
    if delta < 0:
        if mode == 1:
            return 0
        raise ValueError("Ujemna delta.")

    if add_delta:
        return (a + np.sqrt(delta)) / (2 * m)
    return (a - np.sqrt(delta)) / (2 * m)


def u_stac(v: float, m: float) -> float:
    """
    Oblicza odpowiadający punkt stacjonarny u* dla zadanego v*,
    korzystając z relacji między zmiennymi.

    Parametry
    v : float
        Wartość stacjonarna v*.
    m : float
        Parametr modelu.

    Zwraca
    float
        Wartość u*. Jeśli v <= 0, zwraca 0.
    """
    return 0 if v <= 0 else m / v

def homogeneous_state(a: float, m: float):
    """
    Zwraca jednorodny punkt stacjonarny (u*, v*)

    Parametry
    a : float
        Parametr zasobu wody.
    m : float
        Parametr śmiertelności.

    Zwraca
        Jednorodny stan równowagi modelu reakcji (u*, v*).
    """
    vs = v_stac(a, m, mode  = 1, add_delta = True)
    us = u_stac(vs, m)
    return us, vs

# Pochodne obliczone jawnie
def reaction(u: np.ndarray, v: np.ndarray, a: float, m: float):
    """
    Oblicza część reakcyjną układu (bez dyfuzji).

    Równania reakcyjne mają postać:
        du/dt = a - u - u v²
        dv/dt = u v² - m v

    Parametry
    u, v : np.ndarray
        Aktualne wartości zmiennych w chwili t_n.
    a : float
        Parametr wilgotności.
    m : float
        Parametr śmiertelności.

    Zwraca
        (du, dv) — wartości pochodnych czasowych części reakcyjnej.
    """
    vv = v * v
    du = a - u - u * vv
    dv = u * vv - m * v
    return du, dv

# Jacobian
def jacobian(u_stac: float, v_stac: float, m: float) -> np.ndarray:
    return np.array([[-1 - v_stac**2, -2 * u_stac * v_stac],
                     [v_stac**2, 2 * u_stac * v_stac - m]])

# Stabilność bez dyfuzji (ODE)
def check_ode_stability(a: float, m: float):
    """
    Sprawdza stabilność układu reakcyjnego (bez dyfuzji).

    Parametry
    a : float
        Parametr zasobu wody.
    m : float
        Parametr śmiertelności.

    Zwraca
        stable : bool
            Czy punkt jednorodny jest stabilny.
        J : ndarray (2x2)
            Jacobian w punkcie (u*, v*).
    """
    us, vs = homogeneous_state(a, m)
    J = jacobian(us, vs, m)

    trJ = np.trace(J)
    detJ = np.linalg.det(J)

    stable = (trJ < 0 and detJ > 0)

    return stable, J

# ============================================================
# DYSKRETYZACJA PRZESTRZENI I OPERATOR LAPLACE'A
# ============================================================

# Macierz drugich pochodnych
def D2(N: int) -> np.ndarray:
    """
    Konstruuje macierz dyskretnej drugiej pochodnej w 1D
    przy użyciu standardowego schematu trójdiagonalnego.

    Parametry
    N : int
        Wymiar macierzy.

    Zwraca
    np.ndarray
        Macierz NxN aproksymująca operator d²/dx²
        bez narzuconych warunków brzegowych.
    """
    return -2.0 * np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)

# Laplasjan
def laplacian2D(Nx: int, Ny: int, h: float) -> np.ndarray:
    """
    Konstruuje macierz dyskretnego operatora Laplace’a w 2D
    na prostokątnej siatce o kroku h, wykorzystując iloczyn
    Kroneckera macierzy jednowymiarowych.

    Parametry
    Nx : int
        Liczba kolumn macierzy dla wartości x.
    Ny : int
        Liczba wierszy macierzy dla wartości y.
    h : float
        Krok siatki (zakładamy hx = hy).

    Zwraca
    np.ndarray
        Macierz (Nx*Ny) x (Nx*Ny) będąca
        dyskretnym operatorem Laplace’a.
    """
    D2x = D2(Nx)
    D2y = D2(Ny)
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    return (np.kron(Iy, D2x) + np.kron(D2y, Ix)) / (h * h)

# Dyskretyzacja przestrzeni
def make_grid(Lx: float, Ly: float, Nx: int, Ny: int):
    """
    Generuje jednorodną siatkę prostokątną na obszarze
    [0, Lx] x [0, Ly] wraz z krokiem przestrzennym.

    Parametry
    Lx : float
        Długość domeny w kierunku x.
    Ly : float
        Długość domeny w kierunku y.
    Nx : int
        Liczba wartości dla x.
    Ny : int
        Liczba wartości dla y.

    Zwraca
        (x, y, X, Y, h), gdzie:
        x, y  – wektory współrzędnych,
        X, Y  – macierze siatki (meshgrid),
        h     – krok przestrzenny (hx = hy).

    Wyjątki
    ValueError
        Gdy kroki w kierunku x i y nie są równe.
    """
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    h = x[1] - x[0]
    if x[1] - x[0] != y[1] - y[0]:
        raise ValueError("")
    return x, y, X, Y, h

# Warunki brzegowe Dirichleta
def dirichlet_boundary_mask(X: np.ndarray, Y: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """
    Wyznacza maskę logiczną dla warunków brzegowych
    Dirichleta na prostokątnej domenie.

    Parametry
    X, Y : np.ndarray
        Macierze współrzędnych siatki.
    Lx, Ly : float
        Wymiary domeny.

    Zwraca
    np.ndarray
        Spłaszczona maska typu bool wskazująca punkty
        należące do brzegu obszaru.
    """
    Xf = X.flatten()
    Yf = Y.flatten()
    boundary = (Xf == 0.0) | (Xf == Lx) | (Yf == 0.0) | (Yf == Ly)
    return boundary

# ============================================================
# SOLVER PDE
# ============================================================

# Niejawna dyfuzja
def precompute_diffusion(Nx: int, Ny: int, h: float, ht: float, d1: float, d2: float):
    """
    Przygotowuje rozkład formy LU macierzy odpowiadających
    niejawnemu schematowi czasowemu dla części dyfuzyjnej.

    W każdym kroku czasowym rozwiązywany jest układ:
        (I - ht * d1 * L) u^{n+1} = u^n + ht * f_u(u^n, v^n)
        (I - ht * d2 * L) v^{n+1} = v^n + ht * f_v(u^n, v^n),
        gdzie:
            L  – dyskretny operator Laplace’a,
            ht – krok czasowy,
            d1, d2 – współczynniki dyfuzji,
            f_u, f_v – część reakcyjna.

    Parametry
    Nx, Ny : int
        Liczba wartości dla x i y.
    h : float
        Krok przestrzenny (zakładamy hx = hy).
    ht : float
        Krok czasowy.
    d1 : float
        Współczynnik dyfuzji pierwszej zmiennej.
    d2 : float
        Współczynnik dyfuzji drugiej zmiennej.

    Zwraca
        (lu_Au, lu_Av), gdzie każdy element reprezentuje rozkład macierzy:
            Au = I - ht * d1 * L,
            Av = I - ht * d2 * L.
    """
    L = laplacian2D(Nx, Ny, h)
    I = np.eye(Nx * Ny)

    Au = I - ht * d1 * L
    Av = I - ht * d2 * L

    return lu_factor(Au), lu_factor(Av)


# Krok czasowy
def step_reaction_diffusion(
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    m: float,
    ht: float,
    lu_Au,
    lu_Av,
    brzeg: np.ndarray,
):
    """
    Wykonuje jeden krok czasowy metodą:
        reakcja jawnie (Euler),
        dyfuzja niejawnie (rozwiązanie układu liniowego).

    Schemat numeryczny ma postać:
        (I - ht d1 L) u^{n+1} = u^n + ht f_u(u^n, v^n)
        (I - ht d2 L) v^{n+1} = v^n + ht f_v(u^n, v^n)
    gdzie:
        L  – dyskretny operator Laplace’a,
        ht – krok czasowy,
        d1, d2 – współczynniki dyfuzji,
        f_u, f_v – część reakcyjna.

    Parametry
    u, v : np.ndarray
        Wartości w chwili t^n.
    a, m : float
        Parametry modelu.
    ht : float
        Krok czasowy.
    lu_Au, lu_Av :
        Rozkłady LU macierzy (I - ht d L).
    brzeg : np.ndarray
        Maska warunków brzegowych Dirichleta.
    clip_nonnegative : bool
        Czy wymuszać nieujemność rozwiązania.

    Zwraca
        (u_new, v_new) — wartości w chwili t_{n+1}.
    """
    du, dv = reaction(u, v, a, m)

    ru = u + ht * du
    rv = v + ht * dv

    ru[brzeg] = 0
    rv[brzeg] = 0

    u_new = lu_solve(lu_Au, ru)
    v_new = lu_solve(lu_Av, rv)

    u_new = np.maximum(u_new, 0)
    v_new = np.maximum(v_new, 0)

    u_new[brzeg] = 0
    v_new[brzeg] = 0

    return u_new, v_new


# Symulacja do punktu stacjonarnego
def simulate_to_steady(
    u0: np.ndarray,
    v0: np.ndarray,
    a: float,
    m: float,
    ht: float,
    lu_Au,
    lu_Av,
    brzeg: np.ndarray,
    krok_max: int = 500,
    eps: float = 1e-8,
):
    """
    Iteruje schemat czasowy aż do osiągnięcia stanu stacjonarnego, gdzie stosowane kryterium to:
        || v^{n+1} - v^n || < eps

    Parametry
    u0, v0 : np.ndarray
        Warunki początkowe.
    a, m : float
        Parametry modelu.
    ht : float
        Krok czasowy.
    lu_Au, lu_Av :
        Rozkłady LU macierzy dyfuzyjnych.
    brzeg : np.ndarray
        Maska warunków Dirichleta.
    krok_max : int
        Maksymalna liczba iteracji.
    eps : float
        Tolerancja zbieżności.

    Zwraca
        (u, v, i) — rozwiązanie stacjonarne oraz liczba wykonanych iteracji.
    """
    u = u0.copy()
    v = v0.copy()

    for i in range(krok_max):
        v_prev = v.copy()
        u, v = step_reaction_diffusion(u, v, a, m, ht, lu_Au, lu_Av, brzeg=brzeg)

        # test zbieżności
        if np.linalg.norm((v - v_prev)[~brzeg]) < eps:
            return u, v, i + 1

    return u, v, krok_max

