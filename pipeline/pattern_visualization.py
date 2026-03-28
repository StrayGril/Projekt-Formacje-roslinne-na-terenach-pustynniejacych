import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from ipykernel.eventloops import loop_qt

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

def simulate_patterns(
        a,
        m,
        d1,
        d2,
        Lx,
        Ly,
        Nx,
        Ny,
        T,
        ht=0.025,
        noise=1e-2,
        do_modelu=False,
        check_every=200,
        tol_mean=1e-5,
        tol_max=1e-5,
        tol_var=1e-5,
        early_stop=True,
        verbose=False,
):
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
    check_every : int
        Co ile kroków porównywać mean/max/var.
    tol_mean : float
        Tolerancja dla średniej.
    tol_max : float
        Tolerancja dla maksimum.
    tol_var : float
        Tolerancja dla wariancji.
    early_stop : bool
        Czy zatrzymać symulację wcześniej po stabilizacji.
    verbose : bool
        Czy wypisywać informacje diagnostyczne.

    Zwraca
    dict lub tuple
        Słownik danych do wykresów albo końcowe macierze u i v.
    """

    x, y, X, Y, h = make_grid(Lx, Ly, Nx, Ny)
    brzeg = dirichlet_boundary_mask(X, Y, Lx, Ly)

    lu_Au, lu_Av = precompute_diffusion(Nx, Ny, h, ht, d1, d2)
    u_0, v_0 = initial_conditions(Nx, Ny, a, m, brzeg, noise)
    u_curr, v_curr = u_0.copy(), v_0.copy()
    u_hist, v_hist = u_0.copy(), v_0.copy()

    stats_hist = []
    stopped_early = False
    nan_detected = False
    last_step = T - 1

    for t in range(T):
        u_curr, v_curr = step_reaction_diffusion(
            u_curr, v_curr, a, m, ht, lu_Au, lu_Av, brzeg
        )

        # przerwanie jeśli pojawi się NaN albo inf
        if (not np.all(np.isfinite(u_curr))) or (not np.all(np.isfinite(v_curr))):
            nan_detected = True
            last_step = t
            u_curr, v_curr = u_hist, v_hist
            if verbose:
                print(f"Przerwano: NaN/inf w kroku {t}. Zwracam poprzedni udany zapis.")
            break

        # co pewien czas sprawdzamy stabilizację rozwiązania
        if (t + 1) % check_every == 0:
            u_hist, v_hist = u_curr.copy(), v_curr.copy()

            v_inside = v_curr[~brzeg]
            mean_v = np.mean(v_inside)
            var_v = np.var(v_inside)
            max_v = np.max(v_inside)

            stats_hist.append((mean_v, max_v, var_v))

            if verbose:
                print(
                    f"krok={t + 1}, mean(v)={mean_v:.6e}, "
                    f"max(v)={max_v:.6e}, var(v)={var_v:.6e}"
                )

            if early_stop and len(stats_hist) >= 2:
                prev_mean, prev_max, prev_var = stats_hist[-2]
                curr_mean, curr_max, curr_var = stats_hist[-1]

                d_mean = abs(curr_mean - prev_mean)
                d_max = abs(curr_max - prev_max)
                d_var = abs(curr_var - prev_var)

                if (
                        d_mean < tol_mean
                        and d_max < tol_max
                        and d_var < tol_var
                ):
                    stopped_early = True
                    last_step = t
                    if verbose:
                        print(f"Przerwano wcześniej po stabilizacji w kroku {t + 1}")
                    break
        last_step = t

    #print(f"powyższe to {round(a,4)} ostatni krok t {last_step}")

    if do_modelu is True:
        return u_curr.reshape(Ny, Nx), v_curr.reshape(Ny, Nx)
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
        "last_step": last_step + 1,
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
def plot_matrix(M, plot_title="Wykres", show=True, cmap=None):
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
    if cmap is None:
        cmap = cmap_v

    Ny, Nx = M.shape
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y)

    vmin = M.min()
    vmax = M.max()
    if vmin == vmax:
        vmax = vmin + 1e-8

    levels = np.linspace(vmin, vmax, 50)

    plt.figure(figsize=(5, 4))
    im = plt.contourf(X, Y, M, levels=levels, cmap=cmap)

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


# Generowanie i zapisywanie macierzy
def save_as_npz(file_name, a_v, m_v, d1_v, d2_v, Lx=20, Ly=20, Nx=100, Ny=100, T=8000, ht=0.01,
                folder_zapisu="wykresy_bez_etykiet"):
    """
    Zapisuje macierze i dane w zewnętrznym pliku.

    """
    length = len(a_v)

    if (len(m_v) != length) or (len(d1_v) != length) or (len(d2_v) != length):
        raise ValueError("Vectors' lengths unequal")

    U = []
    V = []
    a_ok = []
    m_ok = []
    d1_ok = []
    d2_ok = []

    old_settings = np.seterr(over='warn', invalid='warn', divide='warn')

    try:
        for i in range(length):  # symulacja dla kolejnych parametrow
            try:
                u, v = simulate_patterns(
                    a_v[i], m_v[i], d1_v[i], d2_v[i], Lx=Lx, Ly=Ly,
                    Nx=Nx, Ny=Ny, T=T, ht=ht, do_modelu=True, verbose=True) #@zmiana

                U.append(u)
                V.append(v)
                a_ok.append(a_v[i])
                m_ok.append(m_v[i])
                d1_ok.append(d1_v[i])
                d2_ok.append(d2_v[i])


            except Exception as e:
                print(f"Symulacja dla a={a_v[i]} nie powiodła się: {e}")
                continue
    finally:
        np.seterr(**old_settings)

    sciezka = os.path.join(folder_zapisu, f"{file_name}.npz")

    np.savez_compressed(sciezka,
                        U=np.array(U),
                        V=np.array(V),
                        a=np.array(a_ok),
                        m=np.array(m_ok),
                        d1=np.array(d1_ok),
                        d2=np.array(d2_ok),
                        patterns=np.full(len(a_ok), -1, dtype=int)
                        )  # lub tu można też dać już wymiarowe parametry

    print("Koniec zapisu.")


# ogladamy obrazki i dopisujemy etykiety
def define_patterns(file_name, folder="wykresy_etykiety", folder_stary="wykresy_bez_etykiet", cmap=None):
    sciezka = os.path.join(folder_stary, f"{file_name}.npz")

    with np.load(sciezka, allow_pickle=True) as loader:
        dane = dict(loader)

    if cmap is None:
        cmap = cmap_v

    ile_macierzy = len(dane["V"])
    patterns = dane["patterns"].copy()

    for i in range(ile_macierzy):
        # pomijamy te, które już mają etykiete
        if patterns[i] != -1:
            continue

        title = f"{file_name}, i={i}"
        plot_matrix(dane["V"][i], plot_title=title, show=False, cmap=cmap)

        odp = input("0. nic, 1. cętki, 2. pasy, 3. labirynty, 4. dziury, 5. coś (q=wyjdź): ")

        if odp.lower() == 'q':
            plt.close()
            break

        try:
            patterns[i] = int(odp)
        except ValueError:
            print("Pominięto (niepoprawny znak).")

        plt.close()

    dane["patterns"] = np.array(patterns)
    os.makedirs(folder, exist_ok=True)

    output_path = os.path.join(folder, f"{file_name}.npz")

    np.savez_compressed(output_path, **dane)
    print(f"Koniec. Etykiety ma {sum(1 for x in patterns if x != -1)}/{ile_macierzy} macierzy.")
    print(f"Plik zapisano do: {output_path}")


# konwersja do csv z pominieciem macierzy
def convert_to_csv(npz_file_name):
    dane = np.load(f"{npz_file_name}.npz", allow_pickle=True)

    # Tworzymy słownik tylko z tych danych, które chcemy w tabeli
    tabela = {
        'a': dane['a'],
        'm': dane['m'],
        'd1': dane['d1'],
        'd2': dane['d2'],
        'pattern': dane['patterns']
    }

    df = pd.DataFrame(tabela)

    output_name = f"{npz_file_name}.csv"
    df.to_csv(output_name, index=False)

    print(f"Tabela została zapisana do pliku: {output_name}.")
    return df
