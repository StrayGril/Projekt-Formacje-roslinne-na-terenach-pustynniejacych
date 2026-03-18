import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Dodanie katalogu głównego repo do ścieżki
sys.path.append(os.path.abspath(".."))


from pipeline.model_core import (
    make_grid,
    dirichlet_boundary_mask,
    homogeneous_state,
    precompute_diffusion,
    step_reaction_diffusion
)


# --------------------------------------------------
# Warunki początkowe
# --------------------------------------------------
def initial_conditions(Nx, Ny, a, m, brzeg, noise=1e-3):

    u_star, v_star = homogeneous_state(a, m)

    u = u_star * np.ones(Nx * Ny)
    v = v_star * np.ones(Nx * Ny)

    rng = np.random.default_rng()

    u += noise * rng.standard_normal(Nx * Ny)
    v += noise * rng.standard_normal(Nx * Ny)

    u[brzeg] = 0
    v[brzeg] = 0

    return u, v


# --------------------------------------------------
# Symulacja wzorów
# --------------------------------------------------
def simulate_patterns(a, m, d1, d2, Lx, Ly, Nx, Ny, T, do_modelu=False):
    """
        Przeprowadza symulację o podanych parametrach na T powtórzeń.
        Parametry:
            ...
        Zwraca słownik danych do wykresu, najważniejsze to stany końcowe symulacji: macierze uT i vT.
    """
    x, y, X, Y, h = make_grid(Lx, Ly, Nx, Ny)
    brzeg = dirichlet_boundary_mask(X, Y, Lx, Ly)

    ht = 0.025
    lu_Au, lu_Av = precompute_diffusion(Nx, Ny, h, ht, d1, d2)

    u_0, v_0 = initial_conditions(Nx, Ny, a, m, brzeg, noise=1e-2) #zmienilam 3 na 2

    u_curr, v_curr = u_0.copy(), v_0.copy()

    for t in range(T):
        u_curr, v_curr = step_reaction_diffusion(u_curr, v_curr, a, m, ht, lu_Au, lu_Av, brzeg)

    if do_modelu is True:
        return u_curr.reshape(Ny, Nx), v_curr.reshape(Ny, Nx)

    return {"X": X,
        "Y": Y,
        "u0": u_0,
        "v0": v_0,
        "uT": u_curr,
        "vT": v_curr}



# --------------------------------------------------
# Wykresy z symulacji
# --------------------------------------------------
def plot_patterns(sim_data, wykres="uv"):
    """
        Rysuje wykresy symulacji w chwilach: 0, T na podstawie słownika wymiarów i spłaszczonych macierzy z symulacji.

        Parametry:
        sim_data - wynik funkcji simulate_patterns
        wykres : string "u", "v", "uv", które wykresy pokazać
    """

    X, Y = sim_data["X"], sim_data["Y"]
    Ny, Nx = X.shape

    states = {
        "0": (sim_data["u0"], sim_data["v0"]),
        "T": (sim_data["uT"], sim_data["vT"])}

    if "u" in wykres: # wykres wody

        fig, axs = plt.subplots(1,2, figsize=(10,4))
        dane = [states[t][0] for t in states]
        levels = np.linspace(min(d.min() for d in dane), max(d.max() for d in dane), 50)

        for ax, (t, (u, _)) in zip(axs.flat, states.items()):
            im = ax.contourf(X, Y, u.reshape(Ny, Nx), levels=levels, cmap="Spectral")
            ax.set_title(f"Woda u(t={t})")

        fig.colorbar(im, ax=axs)
        plt.show()


    if "v" in wykres: # wykres biomasy

        fig, axs = plt.subplots(1,2, figsize=(10,4))
        dane = [states[t][1] for t in states]
        levels = np.linspace(min(d.min() for d in dane), max(d.max() for d in dane), 50)

        for ax, (t, (_, v)) in zip(axs.flat, states.items()):
            im = ax.contourf(X, Y, v.reshape(Ny, Nx), levels=levels, cmap="RdYlGn")
            ax.set_title(f"Biomasa v(t={t})")

        fig.colorbar(im, ax=axs)
        plt.show()





def plot_matrix(M, plot_title="Wykres", show=True):
    """
    Rysuje wykres kwadratowej (niespłaszczonej) macierzy (np. u_T lub v_T)
        plot_title="Wykres": mozna tu nadac wlasna nazwe
        show=True: mozna False zeby zcustowizowac wyswietlanie samemu
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

