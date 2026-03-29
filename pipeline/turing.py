import numpy as np
import matplotlib.pyplot as plt
from pipeline.core import (
    homogeneous_state,
    check_ode_stability
)

# --------------------------------------------------
# Relacja dyspersji
# --------------------------------------------------
def dispersion(J: np.ndarray, d1: float, d2: float,
            k_min: float = 0, k_max: float = 5, n_k: int = 1000):
    """
    Oblicza relację dyspersji λ_max(k).

    Algorytm:
        Dla każdego k liczone są wartości własne macierzy J - k^2 D
        i wybierana jest największa część rzeczywista.

    Parametry
    J : ndarray (2x2)
        Jacobian reakcji.
    d1, d2 : float
        Współczynniki dyfuzji.
    k_min, k_max : float
        Zakres analizowanych k.
    n_k : int
        Liczba równomiernie rozłożonych punktów k.

    Zwraca
        k_vals : ndarray
            Wartości k.
        lambda_max : ndarray
            Największa część rzeczywista λ(k).
    """
    k_vals = np.linspace(k_min, k_max, n_k)
    lambda_max = np.zeros_like(k_vals)

    # macierz dyfuzji
    D = np.diag([d1, d2])

    for i, k in enumerate(k_vals):
        M = J - (k**2) * D
        # wartosci własne
        eig_vals = np.linalg.eigvals(M)
        # maksymalna część rzeczywista
        lambda_max[i] = np.max(np.real(eig_vals))

    return k_vals, lambda_max

# --------------------------------------------------
# Pasmo Turinga
# --------------------------------------------------
def turing_band(k_vals: np.ndarray, lambda_max: np.ndarray):
    """
    Wyznacza punkty niestabilne k (pasmo Turinga).

    Parametry
    k_vals : ndarray
    lambda_max : ndarray
        Największa część rzeczywista λ(k).

    Zwraca
        Jeśli istnieje niestabilność:
            {
                "k_min" : początek pasma,
                "k_max" : koniec pasma,
                "k_dom" : globalne maksimum relacji dyspersji
            }
        W przeciwnym razie None.
    """
    unstable = k_vals[lambda_max > 0]

    if unstable.size == 0:
        return None

    return {
        "k_min": unstable.min(),
        "k_max": unstable.max(),
        "k_dom": k_vals[np.argmax(lambda_max)]
    }

# --------------------------------------------------
# Analiza Turinga
# --------------------------------------------------
def turing_analysis(a: float, m: float, d1: float, d2: float,
                    k_min: float = 0, k_max: float = 5, n_k: int = 1000):
    """
     Analiza niestabilności Turinga.

    Algorytm:
        1. Wyznaczenie punktu jednorodnego.
        2. Sprawdzenie stabilności ODE.
        3. Obliczenie relacji dyspersji λ(k) (stabilny -> niestabilny po dodaniu dyfuzji).
        4. Wyznaczenie pasma niestabilnych k.

    Parametry
    a : float
        Parametr zasobu wody.
    m : float
        Parametr śmiertelności.
    d1, d2 : float
        Współczynniki dyfuzji.
    k_min, k_max : float
        Zakres analizowanych k.
    n_k : int
        Liczba punktów k.

    Zwraca
        - J : Jacobian
        - k : wartości k
        - lambda : λ_max(k)
        - band : zakres pasma niestabilności (lub None)
    """

    # stabilność ODE
    stable, J = check_ode_stability(a, m)

    if not stable:
        raise ValueError("Układ reakcyjny niestabilny — brak Turinga")

    # relacja dyspersji
    k_vals, lambda_max = dispersion(J, d1, d2, k_min, k_max, n_k)

    # pasmo Turinga
    band = turing_band(k_vals, lambda_max)

    return {
        "J": J,
        "k": k_vals,
        "lambda": lambda_max,
        "band": band
    }

# --------------------------------------------------
# Wykres
# --------------------------------------------------
def plot_dispersion(k_vals: np.ndarray, lambda_max: np.ndarray, band: dict):
    """
    Rysuje relację dyspersji λ(k) z zaznaczonym pasmem Turinga.

    Parametry
    k_vals : ndarray
    lambda_max : ndarray
        Największa część rzeczywista λ(k).
    """
    plt.figure(figsize=(8,5))
    # krzywa dyspersji
    plt.plot(k_vals, lambda_max, color = "navy")
    plt.axhline(0, color = "black")

    # Zaznaczanie punktów kluczowych dla pasma Turinga w postaci:
    # punkty szczególne k na osi x
    # wysokość wykresu dokładnie w n_k miejscu na osi y (interpolacja liniowa)
    k_min = band["k_min"]
    l_min = np.interp(k_min, k_vals, lambda_max)

    k_max = band["k_max"]
    l_max = np.interp(k_max, k_vals, lambda_max)

    k_dom = band["k_dom"]
    l_dom = np.interp(k_dom, k_vals, lambda_max)

    plt.axvline(k_min, linestyle = ":", color = "gray", label = "k_min")
    plt.axvline(k_max, linestyle = ":", color = "gray", label = "k_max")
    plt.axvline(k_dom, linestyle = "--", color = "gray", label = "k_dom")

    plt.scatter([k_min, k_max, k_dom], [l_min, l_max, l_dom], zorder=5)

    plt.xlabel("k")
    plt.ylabel("max Re(λ(k))")
    plt.title("Relacja dyspersji – analiza Turinga")
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Analiza Turinga
# --------------------------------------------------
def scan_turing_am(
    d1,
    d2,
    m_values,
    a_values,
    k_min=0,
    k_max=20,
    n_k=4000,
):
    """
    Skanuje płaszczyznę (a, m) dla ustalonych dyfuzji d1, d2.

    Zwraca listę słowników z informacją o:
    - istnieniu stanu jednorodnego,
    - max Re(lambda),
    - pasmie Turinga.
    """
    results = []

    for m in m_values:
        for a in a_values:
            try:
                u_star, v_star = homogeneous_state(a, m)
            except:
                u_star, v_star = np.nan, np.nan

            # pomijamy przypadki bez sensownego dodatniego stanu
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

            lam = res.get("lambda", None)
            band = res.get("band", None)

            lambda_max = np.nan
            if lam is not None:
                lambda_max = np.max(np.real(lam))

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

def unpack_scan_results(results):
    """
    Zamienia listę słowników na tablice numpy. (uwaga A != wymiarowe A)
    """
    A = np.array([r["a"] for r in results])
    M = np.array([r["m"] for r in results])
    S = np.array([r["has_state"] for r in results])
    T = np.array([r["has_turing"] for r in results])
    L = np.array([r["lambda_max"] for r in results])
    V = np.array([r["v_star"] for r in results])
    K1 = np.array([r["k_left"] for r in results])
    K2 = np.array([r["k_right"] for r in results])
    KD = np.array([r["k_dom"] for r in results])

    return A, M, S, T, L, V, K1, K2, KD

def plot_lambda_map(results, ax=None):
    """
    Rysuje mapę max Re(lambda) w płaszczyźnie (a, m).
    """
    A, M, S, T, L, V, K1, K2, KD = unpack_scan_results(results)

    mask = np.isfinite(L)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sc = ax.scatter(A[mask], M[mask], c=L[mask], s=18)
    plt.colorbar(sc, ax=ax, label=r"maxRe($\lambda$)")
    ax.set_xlabel("a")
    ax.set_ylabel("m")
    ax.set_title(r"Mapa wartości maxRe($\lambda$)")
    ax.grid(True, alpha=0.3)

def plot_turing_regions(results, ax = None):
    """
    Rysuje mapę klas:
    - brak stanu,
    - stan stabilny,
    - obszar Turinga.
    """
    A, M, S, T, L, V, K1, K2, KD = unpack_scan_results(results)

    mask_no_state = (S == 0)
    mask_state_no_turing = (S == 1) & (T == 0)
    mask_turing = (T == 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(A[mask_no_state], M[mask_no_state], s=8, alpha=0.35, label="brak stanu")
    ax.scatter(A[mask_state_no_turing], M[mask_state_no_turing], s=8, alpha=0.50, label="stan stabilny")
    ax.scatter(A[mask_turing], M[mask_turing], s=12, alpha=0.9, label="obszar Turinga")

    ax.set_xlabel("a")
    ax.set_ylabel("m")
    ax.set_title("Mapa stanów w płaszczyźnie (a, m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

def a_m_pairs(results, m_values):
    out = []

    for m in m_values:
        dane_m = []

        # zbieramy tylko punkty z tym m i has_turing = 1
        for r in results:
            if (
                    r["m"] == m
                    and r["has_turing"] == 1
                    and np.isfinite(r["lambda_max"])
                    and r["lambda_max"] > 0
            ):
                dane_m.append(r)

        # jeśli brak punktów Turinga dla tego m
        if len(dane_m) == 0:
            out.append({
                "m": m,
                "a_max": np.nan,
                "lambda_max": np.nan,
                "a_mean": np.nan,
                "lambda_mean_like": np.nan
            })
            continue

        # maksimum lambda
        best = dane_m[0]
        for r in dane_m:
            if r["lambda_max"] > best["lambda_max"]:
                best = r


        # CZĘŚĆ Z WYBOREM PUNKTÓW
        lambda_max_val = best["lambda_max"]
        dane_strong = []

        for r in dane_m:
            if r["lambda_max"] >= 0.25 * lambda_max_val:
                dane_strong.append(r)

        if len(dane_strong) == 0:
            dane_strong = dane_m

        dane_strong = sorted(dane_strong, key=lambda r: r["a"])
        a_band = [r["a"] for r in dane_strong]
        lambda_band = [r["lambda_max"] for r in dane_strong]

        out.append({
            "m": m,
            "a_max": best["a"],
            "lambda_max": best["lambda_max"],
            "a_band": a_band,
            "lambda_band": lambda_band,
        })

    return out