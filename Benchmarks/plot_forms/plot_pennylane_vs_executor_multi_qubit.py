"""Vergleichsplot PennyLane (roh) vs. Executor (PennyLane-Backend) über mehrere Qubit-Zahlen.

Ein gemeinsames Diagramm für alle drei Benchmark-Modi (creation/execution/
gradient) und mehrere Qubit-Zahlen, im Stil von
``Results/FrameworkVergleich_2-5q_500/pennylane_vs_executor_qubits_time.png``:
Farbe = Modus, Linienstil = Framework (roh durchgezogen, Executor strich-
punktiert), Qubit-Zahl = Transparenz + Liniendicke (blass/dünn = wenige
Qubits, kräftig/dick = viele Qubits) statt Marker-Form — bei 24 Linien in
3 überlappenden Farbclustern sind unterschiedliche Marker-Formen kaum zu
unterscheiden, eine Helligkeits-/Dicken-Rampe für eine ordinale Größe wie
die Qubit-Zahl liest sich sauberer. Jede Modus-Gruppe wird zusätzlich direkt
an der Linie beschriftet statt nur über die Legende.

Anders als beim Qiskit-Vergleich liegen PennyLane-roh und Executor-Läufe
nicht sauber in je einem Ordner pro Qubit-Zahl: Der Executor(PennyLane)-
Gradientenmodus wurde separat nachgezogen (eigener Lauf-Ordner, s.
``executor_dirs["gradient"]`` in RUNS) und bricht bei allen vier Qubit-
Zahlen einheitlich bei 23357 Gattern ab (manueller Stopp) statt bei 100000
wie creation/execution und die PennyLane-Roh-Seite.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

ROOT = Path(__file__).parent.parent  # Benchmarks/ -> Projekt-Wurzel

# Qubit-Zahl -> {PennyLane-Roh-Lauf-Ordner, Executor(PennyLane)-Lauf-Ordner je Modus}
RUNS = {
    5: {
        "pennylane_dir": "Results/Pennylane/2026-07-27_10-26-32_qubits-5",
        "executor_dirs": {
            "creation":  "Results/Executor/2026-07-27_10-34-54_qubits-5_pennylane",
            "execution": "Results/Executor/2026-07-27_10-34-54_qubits-5_pennylane",
            "gradient":  "Results/Executor/2026-07-31_20-59-17_qubits-5_pennylane",
        },
    },
    8: {
        "pennylane_dir": "Results/Pennylane/2026-07-27_10-27-07_qubits-8",
        "executor_dirs": {
            "creation":  "Results/Executor/2026-07-27_10-35-32_qubits-8_pennylane",
            "execution": "Results/Executor/2026-07-27_10-35-32_qubits-8_pennylane",
            "gradient":  "Results/Executor/2026-07-31_20-59-25_qubits-8_pennylane",
        },
    },
    10: {
        "pennylane_dir": "Results/Pennylane/2026-07-27_10-27-40_qubits-10",
        "executor_dirs": {
            "creation":  "Results/Executor/2026-07-27_10-36-03_qubits-10_pennylane",
            "execution": "Results/Executor/2026-07-27_10-36-03_qubits-10_pennylane",
            "gradient":  "Results/Executor/2026-07-31_20-59-32_qubits-10_pennylane",
        },
    },
    12: {
        "pennylane_dir": "Results/Pennylane/2026-07-27_10-28-12_qubits-12",
        "executor_dirs": {
            "creation":  "Results/Executor/2026-07-27_10-36-28_qubits-12_pennylane",
            "execution": "Results/Executor/2026-07-27_10-36-28_qubits-12_pennylane",
            "gradient":  "Results/Executor/2026-07-31_20-59-39_qubits-12_pennylane",
        },
    },
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
    "gradient":  "Gradient (Backpropagation)",
}
FRAMEWORK_STYLE = {
    "PennyLane (roh)":              "-",
    "Executor (PennyLane-Backend)": "-.",
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

    for q, run_cfg in RUNS.items():
        style = QUBIT_STYLE[q]
        for mode in MODES:
            df_pl = load(run_cfg["pennylane_dir"], "benchmark", mode)
            ax.plot(
                df_pl["total_gates"], df_pl["qnc_avg"],
                linestyle=FRAMEWORK_STYLE["PennyLane (roh)"], color=MODE_COLOR[mode],
                **style,
            )
            x_last, y_last = df_pl["total_gates"].iloc[-1], df_pl["qnc_avg"].iloc[-1]
            if x_last > end_points[mode][0]:
                end_points[mode] = (x_last, y_last)

            df_e = load(run_cfg["executor_dirs"][mode], "benchmark_executor_pennylane", mode)
            ax.plot(
                df_e["total_gates"], df_e["qnc_avg"],
                linestyle=FRAMEWORK_STYLE["Executor (PennyLane-Backend)"], color=MODE_COLOR[mode],
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

    out_dir = ROOT / "Results" / "FrameworkVergleich_5-12q_pennylane"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pennylane_vs_executor_qubits_time_new.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
