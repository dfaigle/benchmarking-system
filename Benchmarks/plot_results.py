import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/benchmark_results.csv")

for benchmark in df["benchmark"].unique():

    subset = df[df["benchmark"] == benchmark]

    plt.plot(
        subset["total_gates"],
        subset["avg"],
        marker="o",
        label=benchmark
    )

plt.xlabel("Total Gates")
plt.ylabel("Average Runtime (s)")
plt.title("PennyLane Benchmark")
plt.legend()
plt.grid(True)

plt.show()