import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============================================================
# KROK 1: TWORZENIE SYNTETYCZNEGO DATASETU (imituje Wasze dane)
# ============================================================

def generate_synthetic_data(n_samples=1000):
    """
    Tworzy przykładowy dataset:
    X: parametry modelu [a, m, d1, d2]
    y: etykieta wzoru (0-pustynia, 1-pasy, 2-plamy, 3-labirynt)
    """
    np.random.seed(42)
    
    # Losuj parametry w sensownych zakresach (dostosujcie do Waszego modelu)
    a = np.random.uniform(0, 4, n_samples)      # opady
    m = np.random.uniform(0.1, 1.0, n_samples)  # śmiertelność
    d1 = np.random.uniform(0.1, 2.0, n_samples) # dyfuzja wody
    d2 = np.random.uniform(0.01, 0.5, n_samples) # dyfuzja biomasy
    
    X = np.column_stack([a, m, d1, d2])
    
    # Tworzymy etykiety na podstawie parametrów (to zastąpią Wasze rzeczywiste symulacje!)
    # To jest TYLKO przykład - wy musicie wczytać rzeczywiste wyniki symulacji
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Prosta reguła do generowania etykiet (TYLKO PRZYKŁAD!)
        if a[i] < 1.0:
            y[i] = 0  # pustynia
        elif a[i] < 2.0:
            if d2[i] < 0.1:
                y[i] = 1  # pasy
            else:
                y[i] = 2  # plamy
        else:
            if m[i] < 0.5:
                y[i] = 3  # labirynt
            else:
                y[i] = 1  # pasy
    
    return X, y

# ============================================================
# KROK 2: WCZYTANIE RZECZYWISTYCH DANYCH (to użyjecie!)
# ============================================================

def load_your_simulation_data(csv_file):
    """
    Wczytuje Wasze dane z symulacji.
    Zakładam, że macie plik CSV z kolumnami:
    a, m, d1, d2, pattern_type
    
    pattern_type to etykieta: 'pustynia', 'pasy', 'plamy', 'labirynt', itd.
    """
    df = pd.read_csv(csv_file)
    
    # Kolumny z parametrami
    X = df[['a', 'm', 'd1', 'd2']].values
    
    # Mapowanie nazw kategorii na liczby
    pattern_types = df['pattern_type'].unique()
    type_to_int = {name: i for i, name in enumerate(pattern_types)}
    y = df['pattern_type'].map(type_to_int).values
    
    return X, y, pattern_types

# ============================================================
# KROK 3: GŁÓWNY PIPELINE MODELU 1
# ============================================================

def train_classification_model(X, y, class_names, model_type='logistic'):
    """
    Trenuje model klasyfikacji, który z parametrów przewiduje typ wzoru
    """
    
    # Podział na treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Skalowanie cech (ważne dla regresji logistycznej!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Wybór modelu
    if model_type == 'logistic':
        # TWOJA PROPOZYCJA: Regresja logistyczna wielomianowa
        model = LogisticRegression(
            multi_class='multinomial',      # <-- to daje wektor prawdopodobieństw
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
            class_weight='balanced'         # jeśli klasy niezbalansowane
        )
    elif model_type == 'random_forest':
        # Alternatywa: Random Forest też daje prawdopodobieństwa
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
    else:
        raise ValueError("Nieznany typ modelu")
    
    # Trenowanie
    print(f"\nTrenowanie modelu: {model_type}")
    model.fit(X_train_scaled, y_train)
    
    # Predykcje na testowym
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Ewaluacja
    accuracy = np.mean(y_pred == y_test)
    print(f"Dokładność: {accuracy:.3f}")
    
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return model, scaler, X_test_scaled, y_test, y_pred_proba

# ============================================================
# KROK 4: PRZYKŁAD UŻYCIA
# ============================================================

# Opcja A: Użycie syntetycznych danych (dla testu)
print("=== MODEL 1: KLASYFIKATOR PARAMETRÓW ===\n")
X, y = generate_synthetic_data(n_samples=1000)
class_names = np.array(['pustynia', 'pasy', 'plamy', 'labirynt'])

# Opcja B: Wczytanie rzeczywistych danych (ODKOMENTUJ)
# X, y, class_names = load_your_simulation_data('wasze_dane.csv')

print(f"Dane: {X.shape[0]} próbek, {X.shape[1]} parametry")
print(f"Klasy: {class_names}")
print(f"Rozkład klas: {np.bincount(y)}")

# Trenuj model regresji logistycznej (Twój wybór!)
model, scaler, X_test, y_test, y_pred_proba = train_classification_model(
    X, y, class_names, model_type='logistic'
)

# ============================================================
# KROK 5: ANALIZA WYNIKÓW
# ============================================================

# 1. Pokaż przykład predykcji z prawdopodobieństwami
print("\n=== PRZYKŁADOWE PREDYKCJE ===")
for i in range(5):
    print(f"\nPróbka {i+1}:")
    print(f"  Prawdziwa klasa: {class_names[y_test[i]]}")
    print(f"  Przewidywana klasa: {class_names[model.predict(X_test[i:i+1])[0]]}")
    print("  Prawdopodobieństwa:")
    probs = model.predict_proba(X_test[i:i+1])[0]
    for name, prob in zip(class_names, probs):
        print(f"    {name}: {prob:.3f}")

# 2. Macierz pomyłek
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title('Macierz pomyłek - Model 1')
plt.ylabel('Prawdziwa klasa')
plt.xlabel('Przewidywana klasa')
plt.tight_layout()
plt.show()

# 3. Dla regresji logistycznej: współczynniki (które parametry są ważne)
if isinstance(model, LogisticRegression):
    plt.figure(figsize=(10, 6))
    coef = model.coef_
    for i, class_name in enumerate(class_names):
        plt.plot(coef[i], 'o-', label=class_name, markersize=8)
    plt.xticks(range(4), ['a (opady)', 'm (śmiertelność)', 'd1 (dyfuzja wody)', 'd2 (dyfuzja biomasy)'])
    plt.xlabel('Parametr')
    plt.ylabel('Wartość współczynnika')
    plt.title('Współczynniki regresji logistycznej dla każdej klasy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n=== INTERPRETACJA WSPÓŁCZYNNIKÓW ===")
    for i, class_name in enumerate(class_names):
        print(f"\nKlasa {class_name}:")
        for j, param in enumerate(['a', 'm', 'd1', 'd2']):
            print(f"  {param}: {coef[i][j]:+.3f}")

# ============================================================
# KROK 6: ZAPIS MODELU DO UŻYCIA W MODELU 2
# ============================================================

# Zapisz model i skaler
joblib.dump(model, 'model1_klasyfikator.pkl')
joblib.dump(scaler, 'model1_scaler.pkl')
np.save('model1_class_names.npy', class_names)

print("\n=== MODEL ZAPISANY ===")
print("Pliki:")
print("  - model1_klasyfikator.pkl")
print("  - model1_scaler.pkl")
print("  - model1_class_names.npy")

# ============================================================
# KROK 7: PRZYKŁAD UŻYCIA ZAPISANEGO MODELU
# ============================================================

def predict_pattern(parameters, model, scaler, class_names):
    """
    Dla nowych parametrów przewiduj typ wzoru
    parameters: [a, m, d1, d2]
    """
    # Skaluj parametry
    params_scaled = scaler.transform([parameters])
    
    # Przewiduj prawdopodobieństwa
    probs = model.predict_proba(params_scaled)[0]
    
    return probs

# Przykład: nowe parametry do sprawdzenia
nowe_parametry = [2.5, 0.45, 1.5, 0.02]  # a, m, d1, d2
probs = predict_pattern(nowe_parametry, model, scaler, class_names)

print("\n=== PRZEWIDYWANIE DLA NOWYCH PARAMETRÓW ===")
print(f"Parametry: a={nowe_parametry[0]}, m={nowe_parametry[1]}, d1={nowe_parametry[2]}, d2={nowe_parametry[3]}")
print("Przewidywany rozkład wzorów:")
for name, prob in zip(class_names, probs):
    print(f"  {name}: {prob:.3f}")

# Wizualizacja
plt.figure(figsize=(8, 4))
plt.bar(class_names, probs, color=['red', 'blue', 'green', 'orange'])
plt.title('Przewidywane prawdopodobieństwa wzorów')
plt.ylabel('Prawdopodobieństwo')
plt.ylim(0, 1)
for i, v in enumerate(probs):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.show()