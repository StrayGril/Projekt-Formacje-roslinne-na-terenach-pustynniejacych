import numpy as np
import matplotlib.pyplot as plt
from pipeline.turing_instability import turing_analysis
from pipeline.model_core import homogeneous_state

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

def a_m_pairs(results, m_values, n_selected=5, ):
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
                "a_band": [],
                "lambda_band": []
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

        n_take = min(n_selected, len(dane_strong))
        idx = np.linspace(0, len(dane_strong) - 1, n_take, dtype=int)
        idx = np.unique(idx)

        selected = []
        for i in idx:
            selected.append(dane_strong[i])

        a_band = [round(r["a"], 4) for r in selected] # !!!przybliżenie
        lambda_band = [r["lambda_max"] for r in selected]

        out.append({
            "m": m,
            "a_max": best["a"],
            "lambda_max": best["lambda_max"],
            "a_band": a_band,
            "lambda_band": lambda_band,
        })

    return out