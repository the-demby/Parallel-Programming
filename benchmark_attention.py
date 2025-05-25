# benchmark_attention.py

import numpy as np
import pandas as pd
import time
from attention_numpy import attention_numpy
import attention


def run_version(version: int, sizes, repeat=90, warmup=2):
    data = []
    for n in sizes:
        d, dk = 64, 64
        Q = np.random.rand(n, dk).astype(np.float32)
        K = np.random.rand(n, dk).astype(np.float32)
        V = np.random.rand(n, d).astype(np.float32)

        # Vérification : doit correspondre à v0
        ref = attention_numpy(Q, K, V)
        if version == 0:
            out = attention_numpy(Q, K, V)
        else:
            out = attention.attention(Q, K, V, version=version)

        np.testing.assert_allclose(ref, out, atol=1e-4)

        # Temps moyen (avec warmup)
        times = []
        for i in range(repeat + warmup):
            start = time.perf_counter()
            if version == 0:
                _ = attention_numpy(Q, K, V)
            else:
                _ = attention.attention(Q, K, V, version=version)
            end = time.perf_counter()
            if i >= warmup:
                times.append(end - start)

        mean_time = np.mean(times)
        print(f"[v{version}] n={n} → {mean_time:.5f} sec")
        data.append(dict(n=n, time=mean_time, version=version))

    return pd.DataFrame(data)


def benchmark_all():
    sizes = [64, 128, 256, 512, 1024]
    repeat = 90

    dfs = []
    for v in [0, 1, 2, 3, 4]:
        print(f"\n🔍 Benchmark version {v}")
        dfv = run_version(v, sizes, repeat)
        dfs.append(dfv.rename(columns={"time": f"time_v{v}"}).drop(columns=["version"]))

    df = dfs[0]
    for dfv in dfs[1:]:
        df = df.merge(dfv, on="n")

    # Ajout des speedups relatifs à v0
    for v in [1, 2, 3, 4]:
        df[f"speedup_v{v}"] = df["time_v0"] / df[f"time_v{v}"]

    print("\n📊 Résumé final :\n", df)
    df.to_csv("benchmarks/benchmark_all.csv", index=False)


if __name__ == "__main__":
    benchmark_all()
