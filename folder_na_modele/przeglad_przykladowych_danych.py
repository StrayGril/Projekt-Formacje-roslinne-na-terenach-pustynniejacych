import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Dodanie katalogu głównego repo do ścieżki
sys.path.append(os.path.abspath(""))
from pipeline.pattern_visualization import plot_matrix

# plik z zapisanym słownikiem, w którym pod nazwami ui, vi (i z zakresu [0,21]) są kolejne pary macierzy z symulacji o parametrach:
# m, d1, d2 =0.45, 1, 0.02,
# L, N = 20, 60
# T = 8000
# a to po kolei:
# [0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4, 1.5, 1.5]
# każde a powtarza się 2krotnie bo dzięki temu np. wiemy że v0 i v1 powinny mieć takie same wzory
# skupiamy się na v bo to biomasa

# wgranie pliku:

data = np.load("symulacje_bezwymiarowy.npz")

u0 = data["u9"]
v0 = data["v9"]

print(v0.shape)
print(u0)
#plot_matrix(u0, "u")

# v po kolei
# przy czym wzory wychodza w 8 do 15 bo nie przemyslalam wartosci a
for i in range(22):
    v = data[f"v{i}"]
    plot_matrix(v, "v")


