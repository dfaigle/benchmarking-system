"""Vergleichsplot Qiskit (roh) vs. Executor (Qiskit-Backend) über mehrere Qubit-Zahlen — Speicher.

Speicher-Gegenstück zu ``plot_qiskit_vs_executor_multi_qubit.py`` (dort Zeit):
ein gemeinsames Diagramm für alle drei Benchmark-Modi (creation/execution/
gradient) und mehrere Qubit-Zahlen, gleiches Encoding wie die Zeit-Variante:
Farbe = Modus, Linienstil = Framework (roh durchgezogen, Executor strich-
punktiert), Qubit-Zahl = Transparenz + Liniendicke (blass/dünn = wenige
Qubits, kräftig/dick = viele Qubits). Jede Modus-Gruppe wird zusätzlich
direkt an der Linie beschriftet statt nur über die Legende.

Gemessen wird der Peak-Speicher (tracemalloc, MiB) statt der Laufzeit —
Spalten ``qc_mem_avg`` / ``qnc_mem_avg`` statt ``qc_avg`` / ``qnc_avg``.

Nimmt je Qubit-Zahl einen Qiskit-Roh-Lauf-Ordner und einen zugehörigen
Executor(Qiskit-Backend)-Lauf-Ordner entgegen (s. RUNS unten) und liest
daraus die drei Modus-CSVs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

ROOT = Path(__file__).parent.parent.parent  # benchmarks/plot_forms/ -> Projekt-Wurzel

# Qubit-Zahl -> (Qiskit-Roh-Lauf-Ordner, Executor-Qiskit-Lauf-Ordner)
RUNS = {
    5:  ("raw_results/Qiskit/2026-07-27_10-29-55_qubits-5",  "raw_results/Executor/2026-07-27_10-37-20_qubits-5_qiskit"),
    8:  ("raw_results/Qiskit/2026-07-27_10-30-25_qubits-8",  "raw_results/Executor/2026-07-27_10-37-53_qubits-8_qiskit"),
    10: ("raw_results/Qiskit/2026-07-27_10-31-09_qubits-10", "raw_results/Executor/2026-07-27_10-38-19_qubits-10_qiskit"),
    12: ("raw_results/Qiskit/2026-07-27_10-31-33_qubits-12", "raw_results/Executor/2026-07-27_10-38-45_qubits-12_qiskit"),
}

GATE_SET = "clifford_plus_non_clifford"
MODES = ["creation", "execution", "gradient"]

MODE_COLOR = {
    "creation":  "tab:blue",
    "execution": "tab:orange",
    "gradient":  "mediumseagreen",
}
MODE_LABEL = {
    "creation":  "Erstellung",
    "execution": "Ausführung",
    "gradient":  "Gradient (Parameter-Shift)",
}
FRAMEWORK_STYLE = {
    "Qiskit (roh)":              "-",
    "Executor (Qiskit-Backend)": "-.",
}
# Qubit-Zahl -> (Transparenz, Liniendicke): blass/dünn = wenige, kräftig/dick = viele
QUBIT_STYLE = {
    5:  {"alpha": 0.35, "linewidth": 1.2},
    8:  {"alpha": 0.55, "linewidth": 1.6},
    10: {"alpha": 0.75, "linewidth": 2.0},
    12: {"alpha": 1.00, "linewidth": 2.4},
}

BG   = "#fcfcfb"
GRID = "#dcdcd8"


def load(run_dir: str, file_prefix: str, mode: str) -> pd.DataFrame:
    f = ROOT / run_dir / f"{file_prefix}_{GATE_SET}_{mode}.csv"
    return pd.read_csv(f).sort_values("total_gates")


def resolve_label_collisions(ax, end_points: dict, min_px_gap: float = 26.0) -> dict:
    """Verschiebt Label-y-Positionen in Pixel-Raum auseinander, wenn sie zu nah beieinander
    lägen (z. B. wenn Erstellung/Ausführung beim Speicherverbrauch am rechten Rand
    zusammenlaufen). Reine Text-Labels ohne Verbindungslinie zum Datenpunkt, daher
    unschädlich für die Lesbarkeit, den Datenpunkt selbst leicht zu verfehlen."""
    ax.figure.canvas.draw()  # Transforms müssen final sein, bevor wir in Pixel rechnen
    items = sorted(end_points.items(), key=lambda kv: kv[1][1])  # nach y aufsteigend
    px = [(mode, x, ax.transData.transform((x, y))[1]) for mode, (x, y) in items]

    for i in range(1, len(px)):
        mode, x, y = px[i]
        _, _, y_prev = px[i - 1]
        if y - y_prev < min_px_gap:
            px[i] = (mode, x, y_prev + min_px_gap)

    resolved = {}
    for mode, x, y_px in px:
        x_disp = ax.transData.transform((x, end_points[mode][1]))[0]
        y_data = ax.transData.inverted().transform((x_disp, y_px))[1]
        resolved[mode] = (x, y_data)
    return resolved


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Für die Direkt-Beschriftung je Modus: Endpunkt der Linie mit dem größten x
    end_points = {mode: (0.0, 0.0) for mode in MODES}

    for q, (qiskit_dir, exec_dir) in RUNS.items():
        style = QUBIT_STYLE[q]
        for mode in MODES:
            df_q = load(qiskit_dir, "qiskit", mode)
            ax.plot(
                df_q["total_gates"], df_q["qc_mem_avg"],
                linestyle=FRAMEWORK_STYLE["Qiskit (roh)"], color=MODE_COLOR[mode],
                **style,
            )
            x_last, y_last = df_q["total_gates"].iloc[-1], df_q["qc_mem_avg"].iloc[-1]
            if x_last > end_points[mode][0]:
                end_points[mode] = (x_last, y_last)

            df_e = load(exec_dir, "benchmark_executor_qiskit", mode)
            ax.plot(
                df_e["total_gates"], df_e["qnc_mem_avg"],
                linestyle=FRAMEWORK_STYLE["Executor (Qiskit-Backend)"], color=MODE_COLOR[mode],
                **style,
            )
            x_last, y_last = df_e["total_gates"].iloc[-1], df_e["qnc_mem_avg"].iloc[-1]
            if x_last > end_points[mode][0]:
                end_points[mode] = (x_last, y_last)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Anzahl Gatter", fontsize=12)
    ax.set_ylabel("Peak-Speicher (MiB)", fontsize=12)
    ax.set_title(
        "Qiskit (roh) vs. Executor (Qiskit-Backend) – Speicherverbrauch (Peak) aller Modi "
        f"und Qubit-Zahlen  |  Gate-Set: {GATE_SET}",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.grid(True, which="major", color=GRID, linewidth=0.8, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    # Direkt-Beschriftung je Modus, am rechten Ende der am weitesten reichenden Linie
    # (Kollisionsvermeidung: bei Speicherverbrauch laufen Modi am rechten Rand teils
    # eng zusammen, anders als bei der Laufzeit)
    end_points = resolve_label_collisions(ax, end_points)
    for mode, (x, y) in end_points.items():
        ax.annotate(
            MODE_LABEL[mode], xy=(x, y), xytext=(10, 0), textcoords="offset points",
            color=MODE_COLOR[mode], fontsize=13, fontweight="bold", va="center",
        )

    # Legende 1: Framework (Linienstil)
    framework_handles = [
        plt.Line2D([0], [0], color="#555555", linestyle=ls, linewidth=2, label=label)
        for label, ls in FRAMEWORK_STYLE.items()
    ]
    legend1 = ax.legend(
        handles=framework_handles, title="Framework  (Farbe = Modus)",
        loc="upper left", fontsize=10, title_fontsize=10, frameon=True,
    )
    ax.add_artist(legend1)

    # Legende 2: Qubitzahl (Transparenz + Liniendicke, neutrales Grau)
    qubit_handles = [
        plt.Line2D([0], [0], color="#555555", linestyle="-", label=f"{q} Qubits", **style)
        for q, style in QUBIT_STYLE.items()
    ]
    ax.legend(
        handles=qubit_handles, title="Qubitzahl",
        loc="lower left", fontsize=10, title_fontsize=10, frameon=True,
    )

    fig.tight_layout()

    out_dir = ROOT / "final_plotted_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qiskit_vs_executor_qubits_mem.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
