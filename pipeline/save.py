import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipeline.patterns import simulate_patterns, plot_matrix, cmap_v


# Generowanie i zapisywanie macierzy
def save_as_npz(
        file_name,
        a_vector,
        m_vector,
        d1_vector,
        d2_vector,
        Lx=60,
        Ly=60,
        Nx=100,
        Ny=100,
        T=10000,
        ht=0.025,
        folder="wykresy_bez_etykiet",
        verbose_show=False,
):
    """
    Zapisuje macierze i dane w zewnętrznym pliku.

    """
    length = len(a_vector)

    if (
            (len(m_vector) != length)
            or (len(d1_vector) != length)
            or (len(d2_vector) != length)
    ):
        raise ValueError("Vectors' lengths unequal")

    U = []
    V = []
    a_ok = []
    m_ok = []
    d1_ok = []
    d2_ok = []

    os.makedirs(folder, exist_ok=True)

    old_settings = np.seterr(over='raise', invalid='raise', divide='raise')

    try:
        for i in range(length):  # symulacja dla kolejnych parametrow
            try:
                u, v = simulate_patterns(
                    a_vector[i],
                    m_vector[i],
                    d1_vector[i],
                    d2_vector[i],
                    Lx=Lx,
                    Ly=Ly,
                    Nx=Nx,
                    Ny=Ny,
                    T=T,
                    ht=ht,
                    do_modelu=True
                )

                if (not np.all(np.isfinite(u))) or (not np.all(np.isfinite(v))):
                    if verbose_show:
                        print(f"Pomijam i={i}: wynik ma NaN/inf")
                    continue

                # chcemy macierze czy wektory? obie czy v?
                U.append(u)
                V.append(v)
                a_ok.append(a_vector[i])
                m_ok.append(m_vector[i])
                d1_ok.append(d1_vector[i])
                d2_ok.append(d2_vector[i])

            except Exception:
                if verbose_show:
                    print(
                        f"Błąd dla i={i}, a={a_vector[i]}, m={m_vector[i]}, "
                        f"d1={d1_vector[i]}, d2={d2_vector[i]}: {Exception}"
                    )
                continue
    finally:
        np.seterr(**old_settings)

    sciezka = os.path.join(folder, f"{file_name}.npz")

    np.savez_compressed(
        sciezka,
        U=np.array(U),
        V=np.array(V),
        a=np.array(a_ok),
        m=np.array(m_ok),
        d1=np.array(d1_ok),
        d2=np.array(d2_ok),
        patterns=np.full(len(a_ok), -1, dtype=int)
    )

    if verbose_show:
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