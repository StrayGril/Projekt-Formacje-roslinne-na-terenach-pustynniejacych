import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipeline.patterns import simulate_patterns, plot_matrix, cmap_v

# Main folder for labelling plots
DATA_DIR = "data"

# Subfolders
RAW_PATTERNS_DIR = os.path.join(DATA_DIR, "wykresy_bez_etykiet")
LABELED_PATTERNS_DIR = os.path.join(DATA_DIR, "wykresy_etykiety")
CSV_PATTERNS_DIR = os.path.join(DATA_DIR, "wykresy_etykiety_csv")

# --------------------------------------
# Generating and saving matrices as .npz
# --------------------------------------
def save_as_npz(
        file_name,
        a_vector,
        m_vector,
        d1_vector,
        d2_vector,
        lx=60,
        ly=60,
        nx=100,
        ny=100,
        T=10000,
        ht=0.025,
        folder=RAW_PATTERNS_DIR,
        verbose=False,
):
    """
    Runs multiple simulations for given parameter sets and saves results to a .npz file.
    If a simulation becomes unstable or blows up, the last stable state is returned.
    Saves a compressed .npz file containing matrices (U, V) and corresponding parameter sets (a, m, d1, d2) and an empty set for patterns.

    Parameters
    file_name : str
        Name of the output file (without .npz extension).
    a, m, d1, d2 : array-like
        Vectors of model parameters (must be equal length).
    lx, ly : float, optional
        Domain sizes. Default: 60.
    nx, ny : int, optional
        Number of grid points. Default: 100.
    T : int, optional
        Maximum number of time steps. Default: 10000.
    ht : float, optional
        Time step. Default: 0.025
    folder : str, optional
        Target directory for the saved .npz file. Default: "wykresy_bez_etykiet".
    verbose : bool, optional
        If True, prints diagnostic information about simulation progress and stability.

    Returns: None
    """

    length = len(a_vector)

    if ((len(m_vector) != length)
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
    T_end_ok = []
    mean_v_0_ok = []
    var_v_0_ok = []
    max_v_0_ok = []
    mean_v_T_ok = []
    var_v_T_ok = []
    max_v_T_ok = []

    os.makedirs(folder, exist_ok=True)

    old_settings = np.seterr(over='ignore', invalid='ignore', divide='ignore') #ignore

    try:
        for i in range(length):
            try:
                sim_data = simulate_patterns(
                    a_vector[i],
                    m_vector[i],
                    d1_vector[i],
                    d2_vector[i],
                    lx=lx,
                    ly=ly,
                    nx=nx,
                    ny=ny,
                    T=T,
                    ht=ht,
                    return_matrices=False
                )
                    
                u = sim_data["uT"].reshape(ny, nx)
                v = sim_data["vT"].reshape(ny, nx)
                T_end = sim_data["last_step"]
                    
                if (not np.all(np.isfinite(u))) or (not np.all(np.isfinite(v))):
                    if verbose:
                        print(f"Skipping i={i}: result has NaN/inf")
                    continue

                U.append(u)
                V.append(v)
                a_ok.append(a_vector[i])
                m_ok.append(m_vector[i])
                d1_ok.append(d1_vector[i])
                d2_ok.append(d2_vector[i])

                T_end_ok.append(sim_data["last_step"])

                mean_v_0_ok.append(sim_data["mean_v_0"])
                var_v_0_ok.append(sim_data["var_v_0"])
                max_v_0_ok.append(sim_data["max_v_0"])

                mean_v_T_ok.append(sim_data["mean_v_T"])
                var_v_T_ok.append(sim_data["var_v_T"])
                max_v_T_ok.append(sim_data["max_v_T"])

            except Exception as e:
                if verbose:
                    print(
                        f"Error for i={i}, a={a_vector[i]}, m={m_vector[i]}, "
                        f"d1={d1_vector[i]}, d2={d2_vector[i]}: {e}"
                    )
                continue
    finally:
        np.seterr(**old_settings)

    path = os.path.join(folder, f"{file_name}.npz")

    np.savez_compressed(
        path,
        U=np.array(U),
        V=np.array(V),
        a=np.array(a_ok),
        m=np.array(m_ok),
        d1=np.array(d1_ok),
        d2=np.array(d2_ok),
        patterns=np.full(len(a_ok), -1, dtype=int)
    )

    if verbose:
        print("Saving complete.")

# --------------------------------------------
# Manually defining the patterns from npz file
# --------------------------------------------
def define_patterns(
        file_name, 
        folder=LABELED_PATTERNS_DIR, 
        folder_old=RAW_PATTERNS_DIR, 
        cmap=None
):
    """
    Interactively labels Turing patterns and saves back to an .npz file.

    Logic
        1. Displays each matrix
        2. User manually assigns a category (0. nothing, 1. spots, 2. stripes, 3. labyrinths, 4. gaps, 5. something else) or quits (q)

    It allows for resuming previous work by skipping already labeled matrices.

    Parameters
    file_name : str
        The name of the .npz file to load (without extension).
    folder : str, optional
        The destination directory. Default: "wykresy_etykiety".
    folder_old : str, optional
        The source directory. Default is "wykresy_bez_etykiet".
    cmap : matplotlib.colors.Colormap, optional
        Colormap used for displaying the matrices. Defaults to the globally defined 'cmap_v'.

    Returns: None

    Progress is saved only after the loop finishes or is interrupted by 'q'.
    """
    our_path = os.path.join(folder_old, f"{file_name}.npz")

    with np.load(our_path, allow_pickle=True) as loader:
        data = dict(loader)

    if cmap is None:
        cmap = cmap_v

    length = len(data["V"])
    patterns = data["patterns"].copy()

    for i in range(length):
        # skipping ones with label
        if patterns[i] != -1:
            continue

        title = f"{file_name}, i={i}"
        plot_matrix(data["V"][i], plot_title=title, show=False, cmap=cmap)

        ans = input("0. nothing, 1. spots, 2. stripes, 3. labyrinths, 4. gaps, 5. something else (d=delete)(q=quit): ")

        if ans.lower() == 'q':
            plt.close()
            break
        if ans.lower() == 'd':
            patterns[i] = 99
            plt.close()
            continue

        try:
            patterns[i] = int(ans)
        except ValueError:
            print("Skipped (unknown input).")

        plt.close()

    patterns = np.array(patterns)
    keep_mask = patterns < 99

    for key in ["U", "V", "a", "m", "d1", "d2"]:
        data[key] = data[key][keep_mask]

    data["patterns"] = patterns[keep_mask]

    print(f"Deleted {np.sum(~keep_mask)}/{length} matrices.")


    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, f"{file_name}.npz")
    np.savez_compressed(output_path, **data)

    length = len(data["patterns"])
    stayed = sum(1 for x in data["patterns"] if x != -1)
    print(f"End of file. Patterns are defined on {stayed}/{length} matrices.")
    print(f"File saved to: {output_path}")

# --------------------------------------------
# Conversion of parameters and patterns to csv
# --------------------------------------------
def convert_to_csv(
        npz_file_name,
        input_folder=LABELED_PATTERNS_DIR,
        output_folder=CSV_PATTERNS_DIR
):
    """
        Loads a labeled .npz file from input_folder and creates its reduced .csv
        version in output_folder. The original .npz file is left unchanged.

        Parameters
        npz_file_name : str
            File name without extension.
        input_folder : str, optional
            Folder containing the source .npz file.
            Default: "wykresy_etykiety".
        output_folder : str, optional
            Folder where the .csv file will be saved.
            Default: "csv_etykiety".

        Returns
        pd.DataFrame
            DataFrame containing only parameters and pattern labels.
        """
    input_path = os.path.join(input_folder, f"{npz_file_name}.npz")
    output_path = os.path.join(output_folder, f"{npz_file_name}.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"There is no file: {input_path}")

    os.makedirs(output_folder, exist_ok=True)

    with np.load(input_path, allow_pickle=True) as data:
        table = {
            'a': data['a'],
            'm': data['m'],
            'd1': data['d1'],
            'd2': data['d2'],
            'pattern': data['patterns']
        }

    df = pd.DataFrame(table)
    df.to_csv(output_path, index=False)

    print(f"CSV saved to: {output_path}")
    return df
