import numpy as np
import matplotlib.pyplot as plt
from pipeline.core import (
    v_stac,
    u_stac,
    make_grid,
    dirichlet_boundary_mask,
    precompute_diffusion,
    step_reaction_diffusion,
    simulate_to_steady
)

# ---------------------------------------
# Kontynuacja symulacji
# ---------------------------------------
def continuation_sweep(
    a_values: np.ndarray,
    u_init: np.ndarray,
    v_init: np.ndarray,
    m: float,
    ht: float,
    lu_Au,
    lu_Av,
    brzeg: np.ndarray,
    krok_max: int = 500,
    eps: float = 1e-8,
    store_states: bool = True,
):
    """
    Wykonuje kontynuację numeryczną względem parametru a.

    Dla każdej wartości a z a_values:
        1. Rozwiązuje układ do stanu stacjonarnego.
        2. Używa otrzymanego rozwiązania jako warunku początkowego
           dla kolejnej wartości parametru.
        3. Zapisuje otrzymane wyniki.

    Rejestrowane miary:
        μ_v  = średnia biomasa,
        max_v = maksimum biomasy.

    Parametry
    ----------
    a_values : np.ndarray
        Wartości parametru a (w ustalonej kolejności).
    u_init, v_init : np.ndarray
        Warunki początkowe dla pierwszej wartości parametru.
    m : float
        Parametr modelu.
    ht : float
        Krok czasowy.
    lu_Au, lu_Av :
        Rozkłady LU macierzy dyfuzyjnych.
    brzeg : np.ndarray
        Maska warunków Dirichleta.
    krok_max : int
        Maksymalna liczba iteracji w solverze steady-state.
    eps : float
        Tolerancja zbieżności.
    store_states : bool
        Czy zapisywać pełne stany (u, v) dla każdej wartości a.

    Zwraca
        (avg, max, states)
    """
    u = u_init.copy()
    v = v_init.copy()

    avgs = []
    maxs = []
    states = [] if store_states else None

    for a in a_values:
        u, v, _ = simulate_to_steady(
            u, v, a, m, ht, lu_Au, lu_Av,
            brzeg=brzeg, krok_max=krok_max, eps=eps
        )

        avgs.append(float(np.mean(v)))
        maxs.append(float(np.max(v)))

        if store_states:
            states.append((u.copy(), v.copy()))

    out = {"avg": np.array(avgs), "max": np.array(maxs)}
    if store_states:
        out["states"] = states
    return out

# ---------------------------------------
# Estymacja tipping point
# ---------------------------------------
def estimate_tipping_point(a_values_desc: np.ndarray, max_series: np.ndarray) -> tuple[float, int]:
    """
    Szacuje punkt krytyczny (tipping point) na podstawie
    największego skoku maksimum biomasy.

    Algorytm:
        1. Oblicza różnice kolejnych wartości max(v).
        2. Wybiera indeks największej bezwzględnej zmiany.
        3. Przyjmuje punkt krytyczny jako środek przedziału
           pomiędzy dwiema sąsiednimi wartościami parametru.

    Parametry
    a_values_desc : np.ndarray
        Malejące wartości parametru a.
    max_series : np.ndarray
        Maksimum biomasy dla kolejnych wartości a.

    Zwraca
        (tp, idx), gdzie:
            tp  – przybliżony punkt krytyczny,
            idx – indeks odpowiadający największemu skokowi.
    """
    dv = np.abs(np.diff(max_series))
    idx = int(np.argmax(dv))
    tp = 0.5 * (a_values_desc[idx] + a_values_desc[idx + 1])
    return float(tp), idx

# ---------------------------------------
# Pełna symulacja bifurkacyjna
# ---------------------------------------
def run_bifurcation(
    m: float,
    d1: float,
    d2: float,
    Lx: float = 10,
    Ly: float = 10,
    Nx: int = 30,
    Ny: int = 30,
    ht: float = 0.025,
    krok_max: int = 500,
    eps: float = 1e-8,
    ha: float = 5e-4,
    amax_factor: float = 4,
    a_max: float | None = None,
    store_down_states: bool = True,
):
    """
    Wykonuje pełną analizę bifurkacyjną względem parametru a.

    Algorytm:
        1. Generacja siatki przestrzennej oraz maski Dirichleta.
        2. Faktoryzacja macierzy dyfuzji (schemat niejawny).
        3. Kontynuacja dla malejących wartości a (przejście „w dół”).
        4. Identyfikacja punktu krytycznego (tipping point)
           na podstawie największego skoku max(v).
        5. Kontynuacja dla rosnących wartości a
           (przejście „w górę” z okolic punktu krytycznego).

    Parametry
    m : float
        Parametr śmiertelności biomasy.
    d1, d2 : float
        Współczynniki dyfuzji.
    Lx, Ly : float
        Wymiary domeny przestrzennej.
    Nx, Ny : int
        Liczba wartości dla x i y.
    ht : float
        Krok czasowy.
    krok_max : int
        Maksymalna liczba iteracji w solverze stanu stacjonarnego.
    eps : float
        Tolerancja zbieżności.
    ha : float
        Krok parametru a w procedurze kontynuacji.
    amax_factor : float
        Współczynnik wyznaczający maksymalną wartość parametru:
            a_max = amax_factor * m.
    a_max : float | None = None,
        Jeżeli none, stosujemy amax_factor. W przeciwnym przypadku, dobieramy je jawnie

    Zwraca
        - serie parametrów (a_down, a_up),
        - odpowiadające średnie i maksima biomasy,
        - przybliżony punkt krytyczny tp,
        - indeks skoku tp_idx,
        - indeksy lokalnych maksimów peak_idx dla serii down_max,
        - wartość referencyjną a = 2m,
        - informacje o siatce i parametrach numerycznych.
    """
    if a_max is None:
        amax = amax_factor * m
    else:
        if a_max <= 0:
            raise ValueError("a_max musi być dodatnie")
        amax = a_max

    # siatka i brzeg
    _, _, X, Y, h = make_grid(Lx, Ly, Nx, Ny)
    brzeg = dirichlet_boundary_mask(X, Y, Lx, Ly)

    # LU dla dyfuzji
    lu_Au, lu_Av = precompute_diffusion(Nx, Ny, h, ht, d1, d2)

    # a malejące
    a_down = np.arange(amax, 0, -ha)

    # start na równowadze ODE dla amax
    v0 = v_stac(a_down[0], m)
    u0 = u_stac(v0, m)
    u_init = np.full(Nx * Ny, u0)
    v_init = np.full(Nx * Ny, v0)
    u_init[brzeg] = 0
    v_init[brzeg] = 0

    down = continuation_sweep(
        a_values=a_down,
        u_init=u_init,
        v_init=v_init,
        m=m,
        ht=ht,
        lu_Au=lu_Au,
        lu_Av=lu_Av,
        brzeg=brzeg,
        krok_max=krok_max,
        eps=eps,
        store_states=store_down_states,
    )

    tp, idx = estimate_tipping_point(a_down, down["max"])

    # stan startowy "tuż nad" tp
    idx_c = np.where(a_down > tp)[0][-1]
    u_start, v_start = down["states"][idx_c]

    a_up = np.arange(tp, amax, ha)

    up = continuation_sweep(
        a_values=a_up,
        u_init=u_start,
        v_init=v_start,
        m=m,
        ht=ht,
        lu_Au=lu_Au,
        lu_Av=lu_Av,
        brzeg=brzeg,
        krok_max=krok_max,
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
        "tp": tp,
        "tp_idx": idx,
        "a_2m": 2.0 * m,
        "brzeg": brzeg,
        "grid": {"Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny, "h": h},
        "params": {"m": m, "d1": d1, "d2": d2, "ht": ht, "krok_max": krok_max, "eps": eps, "ha": ha},
    }

# ---------------------------------------
# Wykres
# ---------------------------------------
def plot_bifurcation(result: dict, title: str | None = None, show: bool = True, ax=None):
    """
    Wizualizuje diagram bifurkacyjny względem parametru a.

    Na wykresie dla przejścia w dół (a malejące) oraz w górę (a rosnące) przedstawiane są:
        - średnia przestrzenna biomasy μ_v,
        - maksimum przestrzenne biomasy max(v),
        - wartość referencyjna a = 2m,
        - przybliżony punkt krytyczny (tipping point).

    Parametry
    result : dict
        parametry zwrócone przez funkcję run_bifurcation.
    title : str | None
        Tytuł wykresu.
    show : bool
        Czy wywołać plt.show().
    ax : matplotlib.axes.Axes | None
        Oś, na której ma zostać narysowany wykres.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    a_down = result["a_down"]
    a_up = result["a_up"]

    ax.scatter(a_down, result["down_avg"], s=8, color="cyan", label=r"$v_{avg}$ dla a$\downarrow$")
    ax.scatter(a_down, result["down_max"], s=8, color="blue", label=r"$v_{\max}$ dla a$\downarrow$")

    ax.scatter(a_up, result["up_avg"], s=8, color="red", label=r"$v_{avg}$ dla a$\uparrow$")
    ax.scatter(a_up, result["up_max"], s=8, color="orange", label=r"$v_{\max}$ dla a$\uparrow$")

    ax.axvline(x=result["a_2m"], color="black", linestyle=":", linewidth=2, label="a = 2m")
    ax.axvline(x=result["tp"], color="purple", linestyle=":", linewidth=2, label=rf"$tp \approx {result['tp']:.4f}$")

    ax.set_xlabel(r"$a$")
    ax.set_ylabel("Biomasa w stanie stacjonarnym")

    if title is None:
        title = rf"Diagram bifurkacyjny dla $d_1 = {result['params']['d1']:.2f}$"
    ax.set_title(title)

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if show:
        plt.show()

    return ax

# ---------------------------------------
# Symulacja bifurkacyjna z malejącym a
# ---------------------------------------
def run_bifurcation_down(
    m: float,
    d1: float,
    d2: float,
    Lx: float = 10,
    Ly: float = 10,
    Nx: int = 30,
    Ny: int = 30,
    ht: float = 0.025,
    krok_max: int = 200,
    eps: float = 1e-5,
    ha: float = 5e-4,
    amax_factor: float = 4,
    a_max: float | None = None,
    store_down_states: bool = False,
):
    """
    Wykonuje analizę bifurkacyjną względem parametru a
    wyłącznie dla malejących wartości a.

    Algorytm:
        1. Generacja siatki przestrzennej oraz maski Dirichleta.
        2. Faktoryzacja macierzy dyfuzji (schemat niejawny).
        3. Kontynuacja dla malejących wartości a.
        4. Identyfikacja punktu krytycznego (tipping point)
           na podstawie największego skoku max(v).
        5. Wyznaczenie lokalnych maksimów serii down_max.

    Parametry
    m : float
        Parametr śmiertelności biomasy.
    d1, d2 : float
        Współczynniki dyfuzji.
    Lx, Ly : float
        Wymiary domeny przestrzennej.
    Nx, Ny : int
        Liczba wartości dla x i y.
    ht : float
        Krok czasowy.
    krok_max : int
        Maksymalna liczba iteracji w solverze stanu stacjonarnego.
    eps : float
        Tolerancja zbieżności.
    ha : float
        Krok parametru a w procedurze kontynuacji.
    amax_factor : float
        Współczynnik wyznaczający maksymalną wartość parametru:
            a_max = amax_factor * m.
    a_max : float | None = None,
        Jeżeli none, stosujemy amax_factor. W przeciwnym przypadku, dobieramy je jawnie
    store_down_states : bool
        Czy zapisywać pełne stany dla gałęzi malejącej.

    Zwraca
        - serie parametrów a_down,
        - odpowiadające średnie i maksima biomasy,
        - przybliżony punkt krytyczny tp,
        - indeks skoku tp_idx,
        - indeksy lokalnych maksimów peak_idx dla serii down_max,
        - wartość referencyjną a = 2m,
        - informacje o siatce i parametrach numerycznych.
    """
    if a_max is None:
        amax = amax_factor * m
    else:
        if a_max <= 0:
            raise ValueError("a_max musi być dodatnie")
        amax = a_max

    # siatka i brzeg
    _, _, X, Y, h = make_grid(Lx, Ly, Nx, Ny)
    brzeg = dirichlet_boundary_mask(X, Y, Lx, Ly)

    # LU dla dyfuzji
    lu_Au, lu_Av = precompute_diffusion(Nx, Ny, h, ht, d1, d2)

    # a malejące
    a_down = np.arange(amax, 0, -ha)

    # start na równowadze ODE dla amax
    v0 = v_stac(a_down[0], m)
    u0 = u_stac(v0, m)
    u_init = np.full(Nx * Ny, u0)
    v_init = np.full(Nx * Ny, v0)
    u_init[brzeg] = 0
    v_init[brzeg] = 0

    down = continuation_sweep(
        a_values=a_down,
        u_init=u_init,
        v_init=v_init,
        m=m,
        ht=ht,
        lu_Au=lu_Au,
        lu_Av=lu_Av,
        brzeg=brzeg,
        krok_max=krok_max,
        eps=eps,
        store_states=store_down_states,
    )

    tp, idx = estimate_tipping_point(a_down, down["max"])

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
        "tp": tp,
        "tp_idx": idx,
        "peak_idx": peak_idx,
        "peak_mask": peak_mask,
        "down_states": down["states"] if store_down_states else None,
        "a_2m": 2.0 * m,
        "brzeg": brzeg,
        "grid": {"Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny, "h": h},
        "params": {"m": m, "d1": d1, "d2": d2, "ht": ht, "krok_max": krok_max, "eps": eps, "ha": ha},
    }


# --------------------------------------------------
# Bifurkacja dla symulacji z malejącym a
# --------------------------------------------------
def plot_bifurcation_down(
    result: dict,
    title: str | None = None,
    show: bool = True,
    ax=None,
    show_peaks: bool = True,
):
    """
    Rysuje diagram bifurkacyjny tylko dla przejścia z dużego a w dół.

    Parametry
    result : dict
        parametry zwrócone przez funkcję run_bifurcation.
    title : str | None
        Tytuł wykresu.
    show : bool
        Czy wywołać plt.show().
    ax : matplotlib.axes.Axes | None
        Oś, na której ma zostać narysowany wykres.
    """
    created_ax = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    else:
        fig = ax.figure

    a_down = result["a_down"]
    avg = result["down_avg"]
    maxv = result["down_max"]
    params = result["params"]

    ax.scatter(a_down, avg, s=8, color="black", label=r"$v_{avg}$")
    ax.scatter(a_down, maxv, s=8, color="green", label=r"$v_{max}$")

    # peaki (potencjalnie ciekawe punkty)
    if show_peaks and "peak_mask" in result:
        mask = result["peak_mask"]
        ax.scatter(
            a_down[mask],
            maxv[mask],
            s=20,
            color="red",
            zorder=3,
            label="peaki"
        )

    # linie referencyjne
    ax.axvline(result["a_2m"], linestyle=":", color="black", label="a = 2m")

    if result.get("tp") is not None:
        ax.axvline(result["tp"], linestyle=":", color="purple",
                   label=rf"$tp \approx {result['tp']:.4f}$")

    ax.set_xlabel("a")
    ax.set_ylabel("Biomasa w stanie stacjonarnym")

    if title is None:
        title = rf"$d_1={params['d1']:.6f},\ d_2={params['d2']:.6f}$"

    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if show and created_ax:
        plt.show()
    return ax

