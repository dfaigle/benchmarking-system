import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/benchmark_results_V3.csv")

fig, ax = plt.subplots()

lines = []
labels = []

for benchmark in df["benchmark"].unique():
    for nq in sorted(df["num_qubits"].unique()):
        subset = df[(df["benchmark"] == benchmark) & (df["num_qubits"] == nq)]
        subset = subset.sort_values("total_gates")

        line, = ax.plot(
            subset["total_gates"],
            subset["avg"],
            marker="o",
            label=f"{benchmark}, {nq} qubits"
        )
        lines.append(line)
        labels.append(line.get_label())

ax.set_xlabel("Total Gates")
ax.set_ylabel("Average Runtime (s)")
ax.set_title("PennyLane Benchmark")
ax.grid(True)

leg = ax.legend()
leg_lines = leg.get_lines()

# Legenden-Linien klickbar machen
for leg_line in leg_lines:
    leg_line.set_picker(True)
    leg_line.set_pickradius(5)

# Mapping Legende -> echte Linie
legend_to_line = {leg_line: line for leg_line, line in zip(leg_lines, lines)}

def on_pick(event):
    leg_line = event.artist
    orig_line = legend_to_line[leg_line]
    visible = not orig_line.get_visible()
    orig_line.set_visible(visible)
    # Legenden-Transparenz anpassen
    leg_line.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()

fig.canvas.mpl_connect("pick_event", on_pick)

plt.show()