"""Vergleichsplot Qiskit (roh) vs. Executor (Qiskit-Backend) über mehrere Qubit-Zahlen.

Ein gemeinsames Diagramm für alle drei Benchmark-Modi (creation/execution/
gradient) und mehrere Qubit-Zahlen, im Stil von
``Results/FrameworkVergleich_2-5q_500/qiskit_vs_executor_qubits_time.png``:
Farbe = Modus, Linienstil = Framework (roh durchgezogen, Executor strich-
punktiert), Qubit-Zahl = Transparenz + Liniendicke (blass/dünn = wenige
Qubits, kräftig/dick = viele Qubits) statt Marker-Form — bei 24 Linien in
3 überlappenden Farbclustern sind unterschiedliche Marker-Formen kaum zu
unterscheiden, eine Helligkeits-/Dicken-Rampe für eine ordinale Größe wie
die Qubit-Zahl liest sich sauberer. Jede Modus-Gruppe wird zusätzlich direkt
an der Linie beschriftet statt nur über die Legende.

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
                df_q["total_gates"], df_q["qc_avg"],
                linestyle=FRAMEWORK_STYLE["Qiskit (roh)"], color=MODE_COLOR[mode],
                **style,
            )
            x_last, y_last = df_q["total_gates"].iloc[-1], df_q["qc_avg"].iloc[-1]
            if x_last > end_points[mode][0]:
                end_points[mode] = (x_last, y_last)

            df_e = load(exec_dir, "benchmark_executor_qiskit", mode)
            ax.plot(
                df_e["total_gates"], df_e["qnc_avg"],
                linestyle=FRAMEWORK_STYLE["Executor (Qiskit-Backend)"], color=MODE_COLOR[mode],
                **style,
            )
            x_last, y_last = df_e["total_gates"].iloc[-1], df_e["qnc_avg"].iloc[-1]
            if x_last > end_points[mode][0]:
                end_points[mode] = (x_last, y_last)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Anzahl Gatter", fontsize=12)
    ax.set_ylabel("Zeit (s)", fontsize=12)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    # Direkt-Beschriftung je Modus, am rechten Ende der am weitesten reichenden Linie
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
        loc="lower right", fontsize=10, title_fontsize=10, frameon=True,
    )

    fig.tight_layout()

    out_dir = ROOT / "final_plotted_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qiskit_vs_executor_qubits_time.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
