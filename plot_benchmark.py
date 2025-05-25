# plot_benchmark.py

import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv("benchmarks/benchmark_all.csv")

# --- Graphe 1 : Temps d'exécution brut ---
plt.figure(figsize=(10, 6))
plt.plot(df["n"], df["time_v0"], marker="o", label="v0 - numpy")
plt.plot(df["n"], df["time_v1"], marker="s", label="v1 - OpenMP")
plt.plot(df["n"], df["time_v2"], marker="^", label="v2 - blocage mémoire")
plt.plot(df["n"], df["time_v3"], marker="x", label="v3 - AVX2 QKᵀ")
plt.plot(df["n"], df["time_v4"], marker="d", label="v4 - pipeline fusionné")

plt.xlabel("Taille n")
plt.ylabel("Temps moyen (s)")
plt.yscale("log")  # Échelle logarithmique pour comparer les ordres de grandeur
plt.title("Temps d'exécution de l'attention pour différentes versions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/benchmark_temps.png")
plt.show()

# --- Graphe 2 : Speedup relatif à v0 ---
plt.figure(figsize=(10, 6))
plt.plot(df["n"], df["speedup_v1"], marker="s", label="v1 - OpenMP")
plt.plot(df["n"], df["speedup_v2"], marker="^", label="v2 - blocage mémoire")
plt.plot(df["n"], df["speedup_v3"], marker="x", label="v3 - AVX2 QKᵀ")
plt.plot(df["n"], df["speedup_v4"], marker="d", label="v4 - fusion complète")

plt.xlabel("Taille n")
plt.ylabel("Accélération (speedup vs v0)")
plt.title("Accélération relative à NumPy (v0)")
plt.axhline(1.0, color='gray', linestyle='--')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/benchmark_speedup.png")
plt.show()
