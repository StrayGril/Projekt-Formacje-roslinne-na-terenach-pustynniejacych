import numpy as np
import matplotlib.pyplot as plt
from pipeline.model_core import (
    make_grid,
    dirichlet_boundary_mask,
    homogeneous_state,
    precompute_diffusion,
    step_reaction_diffusion
)
from matplotlib.colors import LinearSegmentedColormap
cmap_u = LinearSegmentedColormap.from_list("yellow_to_darkblue", ["#fde456", "#08306b"])
cmap_v = LinearSegmentedColormap.from_list("yellow_to_darkgreen", ["#fde456", "#00441b"])


# --------------------------------------------------
# Warunki początkowe
# --------------------------------------------------
def initial_conditions(Nx, Ny, a, m, brzeg, noise=1e-3):
    """
    Tworzy warunki początkowe wokół jednorodnego stanu stacjonarnego
    z dodanym małym szumem losowym.

    Parametry
    Nx : int
        Liczba punktów siatki w kierunku x.
    Ny : int
        Liczba punktów siatki w kierunku y.
    a : float
        Parametr sterujący.
    m : float
        Parametr liniowej utraty.
    brzeg : array-like
        Maska punktów brzegowych.
    noise : float
        Amplituda szumu losowego.

    Zwraca
    tuple
        Wektory początkowe u i v.
    """
    u_star, v_star = homogeneous_state(a, m)

    u = u_star * np.ones(Nx * Ny)
    v = v_star * np.ones(Nx * Ny)

    rng = np.random.default_rng()

    u += noise * rng.standard_normal(Nx * Ny)
    v += noise * rng.standard_normal(Nx * Ny)
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)

    u[brzeg] = 0
    v[brzeg] = 0

    return u, v

# --------------------------------------------------
# Symulacja wzorów
# --------------------------------------------------
def simulate_patterns(a, m, d1, d2, Lx, Ly, Nx, Ny, T, ht = 0.025, noise=1e-2, do_modelu=False):
    """
    Przeprowadza symulację układu reakcji-dyfuzji przez T kroków czasowych.

    Parametry
    a : float
        Parametr sterujący (przyrost zasobów wody).
    m : float
        Parametr liniowej utraty biomasy.
    d1 : float
        Współczynnik dyfuzji zmiennej u.
    d2 : float
        Współczynnik dyfuzji zmiennej v.
    Lx : float
        Długość domeny w kierunku x.
    Ly : float
        Długość domeny w kierunku y.
    Nx : int
        Liczba punktów siatki w kierunku x.
    Ny : int
        Liczba punktów siatki w kierunku y.
    T : int
        Liczba kroków czasowych symulacji.
    do_modelu : bool
        Czy zwracać końcowe macierze u i v.

    Zwraca
    dict lub tuple
        Słownik danych do wykresów albo końcowe macierze u i v.
    """
    x, y, X, Y, h = make_grid(Lx, Ly, Nx, Ny)
    brzeg = dirichlet_boundary_mask(X, Y, Lx, Ly)

    lu_Au, lu_Av = precompute_diffusion(Nx, Ny, h, ht, d1, d2)
    u_0, v_0 = initial_conditions(Nx, Ny, a, m, brzeg, noise)
    u_curr, v_curr = u_0.copy(), v_0.copy()

    for t in range(T):
        u_curr, v_curr = step_reaction_diffusion(u_curr, v_curr, a, m, ht, lu_Au, lu_Av, brzeg)

    if do_modelu is True:
        return u_curr.reshape(Ny, Nx), v_curr.reshape(Ny, Nx)

    return {
        "X": X,
        "Y": Y,
        "u0": u_0,
        "v0": v_0,
        "uT": u_curr,
        "vT": v_curr,
    }

# --------------------------------------------------
# Wykresy z symulacji
# --------------------------------------------------
def plot_patterns(sim_data, wykres="uv"):
    """
    Rysuje wykresy stanów początkowych i końcowych symulacji
    dla zmiennych u i v.

    Parametry
    sim_data : dict
        Słownik zwrócony przez funkcję simulate_patterns.
    wykres : str
        Określa, które wykresy pokazać: "u" (wody), "v" (biomasy) lub "uv" (oba).

    Zwraca
    None
        Funkcja nie zwraca wartości, tylko wyświetla wykresy.
    """
    X, Y = sim_data["X"], sim_data["Y"]
    Ny, Nx = X.shape

    states = {
        "0": (sim_data["u0"], sim_data["v0"]),
        "T": (sim_data["uT"], sim_data["vT"])
    }
    
    if "u" in wykres:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        dane = [states[t][0] for t in states]
        levels = np.linspace(min(d.min() for d in dane), max(d.max() for d in dane), 50)

        for ax, (t, (u, _)) in zip(axs.flat, states.items()):
            im = ax.contourf(X, Y, u.reshape(Ny, Nx), levels=levels, cmap=cmap_u)
            ax.set_title(f"Woda u(t={t})")

        fig.colorbar(im, ax=axs)
        plt.show()

    if "v" in wykres:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        dane = [states[t][1] for t in states]
        levels = np.linspace(min(d.min() for d in dane), max(d.max() for d in dane), 50)

        for ax, (t, (_, v)) in zip(axs.flat, states.items()):
            im = ax.contourf(X, Y, v.reshape(Ny, Nx), levels=levels, cmap=cmap_v)
            ax.set_title(f"Biomasa v(t={t})")

        fig.colorbar(im, ax=axs)
        plt.show()

# --------------------------------------------------
# Wykresy ???
# --------------------------------------------------
def plot_matrix(M, plot_title="Wykres", show=True):
    """
    Rysuje wykres niespłaszczonej macierzy.

    Parametry
    M : array-like
        Macierz do narysowania.
    plot_title : str
        Tytuł wykresu.
    show : bool
        Czy wyświetlić wykres od razu.

    Zwraca
    None
        Funkcja nie zwraca wartości, tylko rysuje wykres.
    """
    Ny, Nx = M.shape
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)

    levels = np.linspace(M.min(), M.max(), 50)

    plt.figure(figsize=(5, 4))
    im = plt.contourf(X, Y, M, levels=levels, cmap="viridis")

    plt.colorbar(im)
    plt.title(plot_title)

    if show is True:
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.show()
