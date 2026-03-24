import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# ============================================================
# KROK 1: WCZYTANIE DANYCH (dostosuj do swojego formatu)
# ============================================================

def load_simulation_data(csv_file):
    """
    Wczytuje dane z symulacji.
    Zakładam, że plik CSV zawiera:
    - parametry: m, d1, d2
    - wyniki: mean_biomass, max_biomass
    """
    df = pd.read_csv(csv_file)
    return df

# Przykładowe dane (gdy nie masz jeszcze pliku - do testów)
def generate_sample_data(n_samples=1000):
    """Generuje przykładowe dane do testowania kodu"""
    np.random.seed(42)
    
    m = np.random.uniform(0.1, 1.0, n_samples)
    d1 = np.random.uniform(0.1, 2.0, n_samples)
    d2 = np.random.uniform(0.01, 0.5, n_samples)
    
    # Sztuczna zależność (do testów)
    mean_biomass = (1.5 - m) * np.exp(-0.5 * d1) * (1 + 0.3 * np.sin(10 * d2))
    max_biomass = mean_biomass * (1.5 + 0.5 * np.random.random(n_samples))
    
    df = pd.DataFrame({
        'm': m,
        'd1': d1,
        'd2': d2,
        'mean_biomass': mean_biomass,
        'max_biomass': max_biomass
    })
    return df

# ============================================================
# KROK 2: ANALIZA JEDNOWYMIAROWA (każdy parametr osobno)
# ============================================================

def analyze_univariate(df):
    """Analiza wpływu każdego parametru osobno"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Wpływ parametrów na biomasę - analiza jednowymiarowa', fontsize=16)
    
    parametry = ['m', 'd1', 'd2']
    kolory = ['red', 'blue', 'green']
    
    for i, param in enumerate(parametry):
        # Średnia biomasa
        axes[0, i].scatter(df[param], df['mean_biomass'], alpha=0.3, c=kolory[i])
        axes[0, i].set_xlabel(f'{param}')
        axes[0, i].set_ylabel('Średnia biomasa')
        axes[0, i].set_title(f'Średnia biomasa vs {param}')
        
        # Dodaj linię trendu
        z = np.polyfit(df[param], df['mean_biomass'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df[param].min(), df[param].max(), 100)
        axes[0, i].plot(x_trend, p(x_trend), 'k--', alpha=0.8, label=f'trend: {z[0]:.3f}')
        axes[0, i].legend()
        
        # Maksymalna biomasa
        axes[1, i].scatter(df[param], df['max_biomass'], alpha=0.3, c=kolory[i])
        axes[1, i].set_xlabel(f'{param}')
        axes[1, i].set_ylabel('Maksymalna biomasa')
        axes[1, i].set_title(f'Maksymalna biomasa vs {param}')
        
        # Dodaj linię trendu
        z = np.polyfit(df[param], df['max_biomass'], 1)
        p = np.poly1d(z)
        axes[1, i].plot(x_trend, p(x_trend), 'k--', alpha=0.8, label=f'trend: {z[0]:.3f}')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.show()

# ============================================================
# KROK 3: ANALIZA DWUWYMIAROWA (heatmapy)
# ============================================================

def analyze_bivariate(df):
    """Analiza interakcji między parametrami"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Interakcje między parametrami - heatmapy', fontsize=16)
    
    # Pary parametrów do analizy
    pairs = [('m', 'd1'), ('m', 'd2'), ('d1', 'd2')]
    
    for idx, (x_param, y_param) in enumerate(pairs):
        # Średnia biomasa
        heatmap_data = df.pivot_table(
            values='mean_biomass', 
            index=pd.cut(df[y_param], 20), 
            columns=pd.cut(df[x_param], 20),
            aggfunc='mean'
        )
        
        im1 = axes[0, idx].imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis')
        axes[0, idx].set_xlabel(x_param)
        axes[0, idx].set_ylabel(y_param)
        axes[0, idx].set_title(f'Średnia biomasa: {x_param} vs {y_param}')
        plt.colorbar(im1, ax=axes[0, idx])
        
        # Maksymalna biomasa
        heatmap_data = df.pivot_table(
            values='max_biomass', 
            index=pd.cut(df[y_param], 20), 
            columns=pd.cut(df[x_param], 20),
            aggfunc='mean'
        )
        
        im2 = axes[1, idx].imshow(heatmap_data, aspect='auto', origin='lower', cmap='plasma')
        axes[1, idx].set_xlabel(x_param)
        axes[1, idx].set_ylabel(y_param)
        axes[1, idx].set_title(f'Maksymalna biomasa: {x_param} vs {y_param}')
        plt.colorbar(im2, ax=axes[1, idx])
    
    plt.tight_layout()
    plt.show()

# ============================================================
# KROK 4: WIZUALIZACJA 3D
# ============================================================

def analyze_3d(df):
    """Wizualizacja 3D zależności"""
    
    fig = plt.figure(figsize=(18, 6))
    
    # m vs d1 -> mean_biomass
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(df['m'], df['d1'], df['mean_biomass'], 
                          c=df['mean_biomass'], cmap='viridis', alpha=0.6)
    ax1.set_xlabel('m (śmiertelność)')
    ax1.set_ylabel('d1 (dyfuzja wody)')
    ax1.set_zlabel('Średnia biomasa')
    ax1.set_title('m i d1 vs średnia biomasa')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    # m vs d2 -> mean_biomass
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(df['m'], df['d2'], df['mean_biomass'], 
                          c=df['mean_biomass'], cmap='plasma', alpha=0.6)
    ax2.set_xlabel('m (śmiertelność)')
    ax2.set_ylabel('d2 (dyfuzja biomasy)')
    ax2.set_zlabel('Średnia biomasa')
    ax2.set_title('m i d2 vs średnia biomasa')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    # d1 vs d2 -> max_biomass
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(df['d1'], df['d2'], df['max_biomass'], 
                          c=df['max_biomass'], cmap='magma', alpha=0.6)
    ax3.set_xlabel('d1 (dyfuzja wody)')
    ax3.set_ylabel('d2 (dyfuzja biomasy)')
    ax3.set_zlabel('Maksymalna biomasa')
    ax3.set_title('d1 i d2 vs maksymalna biomasa')
    plt.colorbar(scatter3, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    plt.show()

# ============================================================
# KROK 5: STATYSTYKI I KORELACJE
# ============================================================

def calculate_statistics(df):
    """Oblicza podstawowe statystyki i korelacje"""
    
    print("=" * 60)
    print("PODSTAWOWE STATYSTYKI")
    print("=" * 60)
    
    # Statystyki opisowe
    print("\nStatystyki opisowe:")
    print(df[['m', 'd1', 'd2', 'mean_biomass', 'max_biomass']].describe())
    
    # Korelacje
    print("\n" + "=" * 60)
    print("MACIERZ KORELACJI")
    print("=" * 60)
    
    corr_matrix = df[['m', 'd1', 'd2', 'mean_biomass', 'max_biomass']].corr()
    print(corr_matrix)
    
    # Wizualizacja macierzy korelacji
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1)
    plt.title('Macierz korelacji między parametrami a biomasą')
    plt.tight_layout()
    plt.show()
    
    # Najsilniejsze korelacje z biomasą
    print("\n" + "=" * 60)
    print("NAJSILNIEJSZE KORELACJE Z BIOMASĄ:")
    print("=" * 60)
    
    for biomass_type in ['mean_biomass', 'max_biomass']:
        print(f"\n{biomass_type.upper()}:")
        correlations = corr_matrix[biomass_type].drop([biomass_type])
        for param, corr in correlations.sort_values(ascending=False).items():
            print(f"  {param}: {corr:.3f}")

# ============================================================
# KROK 6: ANALIZA DLA STAŁYCH WARTOŚCI
# ============================================================

def analyze_fixed_parameters(df):
    """
    Analizuje jak zmienia się biomasa gdy jeden parametr jest stały
    """
    
    # Podziel dane na kwartyle dla m
    df['m_quartile'] = pd.qcut(df['m'], 4, labels=['Q1 (niskie)', 'Q2', 'Q3', 'Q4 (wysokie)'])
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Wpływ d1 i d2 dla różnych poziomów śmiertelności (m)', fontsize=16)
    
    for idx, (m_level, group) in enumerate(df.groupby('m_quartile')):
        # Średnia biomasa
        scatter1 = axes[0, idx].scatter(group['d1'], group['d2'], 
                                        c=group['mean_biomass'], 
                                        cmap='viridis', s=50, alpha=0.6)
        axes[0, idx].set_xlabel('d1')
        axes[0, idx].set_ylabel('d2')
        axes[0, idx].set_title(f'm = {m_level}\nŚrednia biomasa')
        plt.colorbar(scatter1, ax=axes[0, idx])
        
        # Maksymalna biomasa
        scatter2 = axes[1, idx].scatter(group['d1'], group['d2'], 
                                        c=group['max_biomass'], 
                                        cmap='plasma', s=50, alpha=0.6)
        axes[1, idx].set_xlabel('d1')
        axes[1, idx].set_ylabel('d2')
        axes[1, idx].set_title(f'm = {m_level}\nMaksymalna biomasa')
        plt.colorbar(scatter2, ax=axes[1, idx])
    
    plt.tight_layout()
    plt.show()

# ============================================================
# KROK 7: IDENTYFIKACJA OPTYMALNYCH PARAMETRÓW
# ============================================================

def find_optimal_parameters(df):
    """
    Znajduje parametry dające najwyższą biomasę
    """
    
    print("=" * 60)
    print("OPTYMALNE PARAMETRY (NAJWYŻSZA BIOMASA)")
    print("=" * 60)
    
    # Dla średniej biomasy
    idx_max_mean = df['mean_biomass'].idxmax()
    print("\nMAX ŚREDNIA BIOMASA:")
    print(f"  Wartość: {df.loc[idx_max_mean, 'mean_biomass']:.3f}")
    print(f"  Parametry: m={df.loc[idx_max_mean, 'm']:.3f}, "
          f"d1={df.loc[idx_max_mean, 'd1']:.3f}, "
          f"d2={df.loc[idx_max_mean, 'd2']:.3f}")
    
    # Dla maksymalnej biomasy
    idx_max_max = df['max_biomass'].idxmax()
    print("\nMAX MAKSYMALNA BIOMASA:")
    print(f"  Wartość: {df.loc[idx_max_max, 'max_biomass']:.3f}")
    print(f"  Parametry: m={df.loc[idx_max_max, 'm']:.3f}, "
          f"d1={df.loc[idx_max_max, 'd1']:.3f}, "
          f"d2={df.loc[idx_max_max, 'd2']:.3f}")
    
    # Średnie dla top 10%
    top_10_pct = df.nlargest(int(len(df)*0.1), 'mean_biomass')
    print("\nŚREDNIE PARAMETRY DLA TOP 10% (najwyższa średnia biomasa):")
    print(f"  m: {top_10_pct['m'].mean():.3f} ± {top_10_pct['m'].std():.3f}")
    print(f"  d1: {top_10_pct['d1'].mean():.3f} ± {top_10_pct['d1'].std():.3f}")
    print(f"  d2: {top_10_pct['d2'].mean():.3f} ± {top_10_pct['d2'].std():.3f}")

# ============================================================
# KROK 8: GŁÓWNY PROGRAM
# ============================================================

def main():
    """Główna funkcja analizy"""
    
    print("=" * 60)
    print("ANALIZA WPŁYWU PARAMETRÓW NA BIOMASĘ")
    print("=" * 60)
    
    # Wczytaj dane (tu przykładowe, zastąp swoimi)
    print("\nWczytywanie danych...")
    # df = load_simulation_data('twoje_dane.csv')
    df = generate_sample_data(n_samples=2000)  # tymczasowo generujemy przykładowe
    
    print(f"Wczytano {len(df)} próbek")
    print(f"Parametry: m, d1, d2")
    print(f"Wyniki: średnia biomasa, maksymalna biomasa")
    
    # 1. Podstawowe statystyki
    calculate_statistics(df)
    
    # 2. Analiza jednowymiarowa
    print("\nGenerowanie wykresów jednowymiarowych...")
    analyze_univariate(df)
    
    # 3. Analiza dwuwymiarowa (heatmapy)
    print("Generowanie heatmap...")
    analyze_bivariate(df)
    
    # 4. Wizualizacja 3D
    print("Generowanie wykresów 3D...")
    analyze_3d(df)
    
    # 5. Analiza dla stałych wartości m
    print("Analiza dla różnych poziomów śmiertelności...")
    analyze_fixed_parameters(df)
    
    # 6. Optymalne parametry
    find_optimal_parameters(df)
    
    # 7. Dodatkowe: histogramy rozkładów
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Rozkłady parametrów i wyników', fontsize=16)
    
    for idx, col in enumerate(['m', 'd1', 'd2']):
        axes[0, idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[0, idx].set_xlabel(col)
        axes[0, idx].set_ylabel('Częstość')
        axes[0, idx].set_title(f'Rozkład {col}')
    
    axes[1, 0].hist(df['mean_biomass'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Średnia biomasa')
    axes[1, 0].set_ylabel('Częstość')
    axes[1, 0].set_title('Rozkład średniej biomasy')
    
    axes[1, 1].hist(df['max_biomass'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Maksymalna biomasa')
    axes[1, 1].set_ylabel('Częstość')
    axes[1, 1].set_title('Rozkład maksymalnej biomasy')
    
    axes[1, 2].scatter(df['mean_biomass'], df['max_biomass'], alpha=0.3)
    axes[1, 2].set_xlabel('Średnia biomasa')
    axes[1, 2].set_ylabel('Maksymalna biomasa')
    axes[1, 2].set_title('Korelacja: średnia vs maksymalna')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()