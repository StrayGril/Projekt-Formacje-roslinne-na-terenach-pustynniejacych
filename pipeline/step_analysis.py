import numpy as np
import matplotlib.pyplot as plt

import tqdm


from pipeline.core import (
    make_grid,
    dirichlet_boundary_mask,
    precompute_diffusion,
    step_reaction_diffusion
)
from pipeline.patterns import initial_conditions



def test_ht(T, kroki, a, m, d1, d2, Lx, Ly, Nx, Ny):
    """
    Test podwajania kroku czasowego MAE (w czasie i przestrzeni)

    Parametry
    T : int
        Liczba kroków czasowych symulacji.
    kroki : wektor
        Wektor kroków czasowych, czyli dziedziny wykresu.
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

    Zwraca
    tuple
       Wektory błędów u i v do wykresu (oraz ustawia wszystko do wykresu, wystarczy dopisać plt.show()).
    """

    # przestrzen
    x, y, X, Y, h = make_grid(Lx, Ly, Nx, Ny)
    brzeg = dirichlet_boundary_mask(X, Y, Lx, Ly)


    u_0, v_0 = initial_conditions(Nx, Ny, a, m, brzeg)

    bledy_u =[]
    bledy_v = []

    for ht in tqdm.tqdm(kroki):
        # zwykla i podwojonego u
        u_1 = u_0.copy()
        u_2 = u_0.copy()
        # zwykla i podwojonego v
        v_1 = v_0.copy()
        v_2 = v_0.copy()

        # macierze do zwyklego i podwojonego bledu
        lu_Au1, lu_Av1 = precompute_diffusion(Nx, Ny, h, ht, d1, d2)
        lu_Au2, lu_Av2 = precompute_diffusion(Nx, Ny, h, ht/2, d1, d2)

        t = int(T / ht)

        blad_ht_u = 0
        blad_ht_v = 0

        for _ in range(t):
            # krok symulacji ht
            u_1, v_1 = step_reaction_diffusion(u_1, v_1, a, m, ht, lu_Au1, lu_Av1, brzeg)
            for i in range(2): # dla ht/2 dwa obroty pętli
                u_2, v_2 = step_reaction_diffusion(u_2, v_2, a, m, ht/2, lu_Au2, lu_Av2, brzeg)

            blad_ht_u += np.mean(np.abs(u_1 - u_2))
            blad_ht_v += np.mean(np.abs(v_1 - v_2))

        bledy_u.append(blad_ht_u / t)
        bledy_v.append(blad_ht_v / t)


    plt.plot(kroki, np.array(bledy_u), label=f"water", c="blue")
    plt.plot(kroki, np.array(bledy_v), label="biomass", c="green")

    plt.title(f"Double step error plot for T={T}")
    plt.xlabel("$h_t$")
    plt.ylabel("log(MAE)")
    #plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    #plt.show()
    return bledy_u, bledy_v
