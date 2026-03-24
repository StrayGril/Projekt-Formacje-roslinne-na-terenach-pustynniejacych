import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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

def train_classification_model(X, y, class_names, model_type='random_forest', use_smote=True, 
                                verbose=True, scale_data=True):
    """
    Trenuje model klasyfikacji z ogromną liczbą dostępnych modeli!
    
    PARAMETRY:
    ----------
    X : numpy array - cechy (parametry)
    y : numpy array - etykiety (wzory)
    class_names : list - nazwy klas
    model_type : str - typ modelu (lista dostępnych poniżej)
    use_smote : bool - czy użyć SMOTE do balansowania
    verbose : bool - czy wypisywać szczegóły
    scale_data : bool - czy skalować dane (dla modeli liniowych)
    
    DOSTĘPNE MODELE:
    ----------------
    'logistic'           - Regresja logistyczna
    'random_forest'      - Random Forest (domyślny)
    'xgboost'            - XGBoost (bardzo popularny)
    'lightgbm'           - LightGBM (szybki)
    'catboost'           - CatBoost (dobry dla kategorycznych)
    'gradient_boosting'  - Gradient Boosting
    'svm'                - Support Vector Machine
    'knn'                - K-Nearest Neighbors
    'decision_tree'      - Drzewo decyzyjne
    'naive_bayes'        - Naiwny Bayes
    'neural_network'     - Sieć neuronowa (MLP)
    'lda'                - Liniowa Analiza Dyskryminacyjna
    'qda'                - Kwadratowa Analiza Dyskryminacyjna
    'adaboost'           - AdaBoost
    'extra_trees'        - Extremely Randomized Trees
    'one_vs_rest_rf'     - OneVsRest z Random Forest
    'one_vs_rest_svm'    - OneVsRest z SVM
    """
    
    # ============================================================
    # PODZIAŁ DANYCH
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # ============================================================
    # SMOTE - BALANSOWANIE DANYCH
    # ============================================================
    if use_smote:
        if verbose:
            print("\n--- Stosuję SMOTE do balansowania klas ---")
            print(f"Rozkład przed SMOTE: {Counter(y_train)}")
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        if verbose:
            print(f"Rozkład po SMOTE: {Counter(y_train_balanced)}")
            print(f"Liczba próbek przed SMOTE: {len(X_train)}")
            print(f"Liczba próbek po SMOTE: {len(X_train_balanced)}")
        
        X_train_to_use = X_train_balanced
        y_train_to_use = y_train_balanced
    else:
        X_train_to_use = X_train
        y_train_to_use = y_train
    
    # ============================================================
    # SKALOWANIE (dla modeli wrażliwych na skalę)
    # ============================================================
    scaler = StandardScaler()
    
    # Modele, które NIE potrzebują skalowania
    no_scaling_models = [
        'random_forest', 'xgboost', 'lightgbm', 'catboost', 
        'gradient_boosting', 'decision_tree', 'extra_trees',
        'adaboost', 'naive_bayes'
    ]
    
    if scale_data and model_type not in no_scaling_models:
        X_train_scaled = scaler.fit_transform(X_train_to_use)
        X_test_scaled = scaler.transform(X_test)
        if verbose:
            print("\n--- Dane zostały przeskalowane ---")
    else:
        X_train_scaled = X_train_to_use
        X_test_scaled = X_test
        # WAŻNE: Trenujemy scaler, żeby działał dla nowych danych!
        scaler.fit(X_train_to_use)  # <--- DODAJ TĘ JEDNĄ LINIĘ!
        if verbose and model_type in no_scaling_models:
            print("\n--- Skalowanie pominięte (model odporny na skalę) ---")
    
    # ============================================================
    # WYBÓR MODELU
    # ============================================================
    
    if verbose:
        print(f"\n--- Tworzę model: {model_type} ---")
    
    # ---- 1. REGRESJA LOGISTYCZNA ----
    if model_type == 'logistic':
        if use_smote:
            model = LogisticRegression(
                solver='lbfgs', max_iter=2000, random_state=42,
                class_weight=None
            )
        else:
            model = LogisticRegression(
                solver='lbfgs', max_iter=2000, random_state=42,
                class_weight='balanced'
            )
    
    # ---- 2. RANDOM FOREST ----
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
            class_weight='balanced' if not use_smote else None
        )
    
    # ---- 3. XGBOOST (bardzo popularny, często najlepszy) ----
    elif model_type == 'xgboost':
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='mlogloss',
            n_jobs=-1
        )
    
    # ---- 4. LIGHTGBM (szybszy od XGBoost) ----
    elif model_type == 'lightgbm':
        model = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
    
    # ---- 5. CATBOOST (dobry dla danych kategorycznych) ----
    elif model_type == 'catboost':
        try:
            model = CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1,
                random_state=42, verbose=False
            )
        except:
            print("CatBoost nie jest zainstalowany. Użyj: pip install catboost")
            print("Zamiast tego używam Random Forest")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # ---- 6. GRADIENT BOOSTING ----
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=2, random_state=42
        )
    
    # ---- 7. SVM (Support Vector Machine) ----
    elif model_type == 'svm':
        model = SVC(
            kernel='rbf', C=1.0, gamma='scale', probability=True,
            class_weight='balanced' if not use_smote else None,
            random_state=42
        )
    
    # ---- 8. KNN (K-Nearest Neighbors) ----
    elif model_type == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=5, weights='distance', metric='minkowski'
        )
    
    # ---- 9. DRZEWO DECYZYJNE ----
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, min_samples_leaf=2,
            class_weight='balanced' if not use_smote else None,
            random_state=42
        )
    
    # ---- 10. NAIWNY BAYES ----
    elif model_type == 'naive_bayes':
        model = GaussianNB()
    
    # ---- 11. SIEĆ NEURONOWA (MLP) ----
    elif model_type == 'neural_network':
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50), activation='relu',
            solver='adam', max_iter=1000, random_state=42,
            early_stopping=True
        )
    
    # ---- 12. LDA (Liniowa Analiza Dyskryminacyjna) ----
    elif model_type == 'lda':
        model = LinearDiscriminantAnalysis()
    
    # ---- 13. QDA (Kwadratowa Analiza Dyskryminacyjna) ----
    elif model_type == 'qda':
        model = QuadraticDiscriminantAnalysis()
    
    # ---- 14. ADABOOST ----
    elif model_type == 'adaboost':
        model = AdaBoostClassifier(
            n_estimators=200, learning_rate=0.1, random_state=42
        )
    
    # ---- 15. EXTRA TREES (Extremely Randomized Trees) ----
    elif model_type == 'extra_trees':
        model = ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
    
    # ---- 16. ONE VS REST z RANDOM FOREST ----
    elif model_type == 'one_vs_rest_rf':
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        model = OneVsRestClassifier(base_model)
    
    # ---- 17. ONE VS REST z SVM ----
    elif model_type == 'one_vs_rest_svm':
        base_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        model = OneVsRestClassifier(base_model)
    
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}\n"
                         f"Dostępne: logistic, random_forest, xgboost, lightgbm, catboost, "
                         f"gradient_boosting, svm, knn, decision_tree, naive_bayes, "
                         f"neural_network, lda, qda, adaboost, extra_trees, "
                         f"one_vs_rest_rf, one_vs_rest_svm")
    
    # ============================================================
    # TRENOWANIE
    # ============================================================
    if verbose:
        print(f"\n--- Trenuję model ---")
    
    try:
        model.fit(X_train_scaled, y_train_to_use)
    except Exception as e:
        print(f"Błąd podczas trenowania {model_type}: {e}")
        print("Spróbuj innego modelu lub sprawdź dane")
        return None, None, None, None, None
    
    # ============================================================
    # EWALUACJA
    # ============================================================
    y_pred = model.predict(X_test_scaled)
    
    # Sprawdź czy model ma predict_proba
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)
    else:
        y_pred_proba = None
        if verbose:
            print("Uwaga: Model nie obsługuje predict_proba")
    
    accuracy = np.mean(y_pred == y_test)
    
    if verbose:
        print(f"\n--- WYNIKI ---")
        print(f"Dokładność: {accuracy:.4f}")
        print("\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred, target_names=class_names))
    
    return model, scaler, X_test_scaled, y_test, y_pred_proba

# Funkcja do testowania wszystkich modeli
def test_all_models(X, y, class_names, use_smote=True):
    """
    Testuje wszystkie dostępne modele i pokazuje wyniki
    """
    models_to_test = [
        'logistic',
        'random_forest',
        'xgboost',
        'lightgbm',
        'gradient_boosting',
        'svm',
        'knn',
        'decision_tree',
        'naive_bayes',
        'neural_network',
        'lda',
        'qda',
        'adaboost',
        'extra_trees'
    ]
    
    results = []
    
    print("=" * 80)
    print("TESTOWANIE WSZYSTKICH MODELI")
    print("=" * 80)
    
    for model_type in models_to_test:
        print("\n" + "-" * 60)
        print(f"Testuję: {model_type.upper()}")
        print("-" * 60)
        
        try:
            model, scaler, X_test, y_test, y_pred_proba = train_classification_model(
                X, y, class_names, 
                model_type=model_type, 
                use_smote=use_smote,
                verbose=True  # pokazuje wyniki
            )
            
            # Zapisz wynik
            if model is not None:
                y_pred = model.predict(X_test)
                accuracy = np.mean(y_pred == y_test)
                results.append({
                    'model': model_type,
                    'accuracy': accuracy,
                    'model_obj': model
                })
        
        except Exception as e:
            print(f"Błąd dla {model_type}: {e}")
            continue
    
    # Podsumowanie
    print("\n" + "=" * 80)
    print("PODSUMOWANIE WSZYSTKICH MODELI")
    print("=" * 80)
    print(f"{'Model':<20} {'Dokładność':<10}")
    print("-" * 30)
    
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{r['model']:<20} {r['accuracy']:.4f}")
    
    # Najlepszy model
    if results:
        best = max(results, key=lambda x: x['accuracy'])
        print("\n" + "=" * 80)
        print(f"🏆 NAJLEPSZY MODEL: {best['model']} z dokładnością {best['accuracy']:.4f}")
        print("=" * 80)
        return best['model_obj'], best['model']
    
    return None, None

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

"""
# Trenuj model regresji logistycznej (Twój wybór!)
model, scaler, X_test, y_test, y_pred_proba = train_classification_model(
    X, y, class_names, model_type='logistic', use_smote = True
)

# Trenuj model random_forest (Twój wybór!)
model, scaler, X_test, y_test, y_pred_proba = train_classification_model(
    X, y, class_names, model_type='random_forest', use_smote = True
)
"""

# Przetestuj wszystkie modele
best_model, best_name = test_all_models(X, y, class_names, use_smote=True)

# Wytrenuj najlepszy model, żeby mieć dane testowe
model, scaler, X_test, y_test, y_pred_proba = train_classification_model(
    X, y, class_names, model_type=best_name, use_smote=True, verbose=True
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