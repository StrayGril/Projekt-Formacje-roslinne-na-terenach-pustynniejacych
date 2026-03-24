import joblib
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# WCZYTANIE ZAPISANEGO MODELU
# ============================================================

print("=== WCZYTYWANIE MODELU ===\n")

# Wczytaj model
model = joblib.load('model1_klasyfikator.pkl')
scaler = joblib.load('model1_scaler.pkl')
class_names = np.load('model1_class_names.npy', allow_pickle=True)

print(f"✅ Model wczytany pomyślnie!")
print(f"📊 Klasy: {class_names}")
print(f"🔢 Typ modelu: {type(model).__name__}")

# ============================================================
# FUNKCJA DO PRZEWIDYWANIA (POPRAWIONA!)
# ============================================================

def przewiduj_wzor(a, m, d1, d2):
    """
    Przewiduje wzór dla podanych parametrów
    UWAGA: Dla XGBoost NIE SKALUJEMY danych!
    """
    parametry = np.array([[a, m, d1, d2]])
    
    # WAŻNE: XGBoost nie potrzebuje skalowania!
    # Używamy surowych parametrów, tak jak podczas trenowania
    params_use = parametry
    
    probs = model.predict_proba(params_use)[0]
    pred = model.predict(params_use)[0]
    
    return class_names[pred], probs

# ============================================================
# TESTOWANIE
# ============================================================

print("\n" + "="*60)
print("TESTOWANIE MODELU")
print("="*60)

# Test 1: Według reguły (powinien dać labirynt)
print("\n1. Parametry zgodne z regułą (powinny dać labirynt):")
wzor, probs = przewiduj_wzor(2.5, 0.45, 1.5, 0.15)
print(f"   a=2.5, m=0.45, d1=1.5, d2=0.15")
print(f"   → Przewidywany wzór: {wzor}")
print(f"   Prawdopodobieństwa:")
for name, prob in zip(class_names, probs):
    print(f"     {name}: {prob:.3f}")

# Test 2: Niskie opady (powinna być pustynia)
print("\n2. Niskie opady (powinna być pustynia):")
wzor, probs = przewiduj_wzor(0.5, 0.5, 1.0, 0.1)
print(f"   a=0.5, m=0.5, d1=1.0, d2=0.1")
print(f"   → Przewidywany wzór: {wzor}")
print(f"   Prawdopodobieństwa:")
for name, prob in zip(class_names, probs):
    print(f"     {name}: {prob:.3f}")

# Test 3: Średnie opady, niska dyfuzja (pasy)
print("\n3. Średnie opady, niska dyfuzja biomasy (pasy):")
wzor, probs = przewiduj_wzor(1.5, 0.5, 1.0, 0.05)
print(f"   a=1.5, m=0.5, d1=1.0, d2=0.05")
print(f"   → Przewidywany wzór: {wzor}")
print(f"   Prawdopodobieństwa:")
for name, prob in zip(class_names, probs):
    print(f"     {name}: {prob:.3f}")

# Test 4: Średnie opady, wysoka dyfuzja (plamy)
print("\n4. Średnie opady, wysoka dyfuzja biomasy (plamy):")
wzor, probs = przewiduj_wzor(1.5, 0.5, 1.0, 0.3)
print(f"   a=1.5, m=0.5, d1=1.0, d2=0.3")
print(f"   → Przewidywany wzór: {wzor}")
print(f"   Prawdopodobieństwa:")
for name, prob in zip(class_names, probs):
    print(f"     {name}: {prob:.3f}")

# Test 5: Wysokie opady, niska śmiertelność (labirynt)
print("\n5. Wysokie opady, niska śmiertelność (labirynt):")
wzor, probs = przewiduj_wzor(3.0, 0.3, 1.5, 0.2)
print(f"   a=3.0, m=0.3, d1=1.5, d2=0.2")
print(f"   → Przewidywany wzór: {wzor}")
print(f"   Prawdopodobieństwa:")
for name, prob in zip(class_names, probs):
    print(f"     {name}: {prob:.3f}")

# ============================================================
# WIZUALIZACJA
# ============================================================

print("\n" + "="*60)
print("WIZUALIZACJA DLA PRZYKŁADOWYCH PARAMETRÓW")
print("="*60)

a_test = 2.5
m_test = 0.45
d1_test = 1.5
d2_test = 0.15

wzor, probs = przewiduj_wzor(a_test, m_test, d1_test, d2_test)

print(f"\nParametry: a={a_test}, m={m_test}, d1={d1_test}, d2={d2_test}")
print(f"Przewidywany wzór: {wzor}")

plt.figure(figsize=(8, 4))
colors = ['red', 'blue', 'green', 'orange']
bars = plt.bar(class_names, probs, color=colors)
plt.title(f'Przewidywane prawdopodobieństwa wzorów\n(a={a_test}, m={m_test}, d1={d1_test}, d2={d2_test})')
plt.ylabel('Prawdopodobieństwo')
plt.ylim(0, 1)
for i, v in enumerate(probs):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
plt.tight_layout()
plt.show()

print("\n✅ Test zakończony!")