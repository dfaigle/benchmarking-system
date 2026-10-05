"""Vergleichsplot PennyLane (roh) vs. Executor (PennyLane-Backend) über mehrere Qubit-Zahlen — Speicher.

Speicher-Gegenstück zu ``plot_pennylane_vs_executor_multi_qubit.py`` (dort
Zeit): ein gemeinsames Diagramm für alle drei Benchmark-Modi (creation/
execution/gradient) und mehrere Qubit-Zahlen, gleiches Encoding wie die
Zeit-Variante: Farbe = Modus, Linienstil = Framework (roh durchgezogen,
Executor strichpunktiert), Qubit-Zahl = Transparenz + Liniendicke (blass/
dünn = wenige Qubits, kräftig/dick = viele Qubits). Jede Modus-Gruppe wird
zusätzlich direkt an der Linie beschriftet statt nur über die Legende.

Gemessen wird der Peak-Speicher (tracemalloc, MiB) statt der Laufzeit —
Spalte ``qnc_mem_avg`` statt ``qnc_avg`` (beide Seiten nutzen denselben
Spaltenpräfix ``qnc``, s. Zeit-Skript).

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

ROOT = Path(__file__).parent.parent.parent  # benchmarks/plot_forms/ -> Projekt-Wurzel

# Qubit-Zahl -> {PennyLane-Roh-Lauf-Ordner, Executor(PennyLane)-Lauf-Ordner je Modus}
RUNS = {
    5: {
        "pennylane_dir": "raw_results/Pennylane/2026-07-27_10-26-32_qubits-5",
        "executor_dirs": {
            "creation":  "raw_results/Executor/2026-07-27_10-34-54_qubits-5_pennylane",
            "execution": "raw_results/Executor/2026-07-27_10-34-54_qubits-5_pennylane",
            "gradient":  "raw_results/Executor/2026-07-31_20-59-17_qubits-5_pennylane",
        },
    },
    8: {
        "pennylane_dir": "raw_results/Pennylane/2026-07-27_10-27-07_qubits-8",
        "executor_dirs": {
            "creation":  "raw_results/Executor/2026-07-27_10-35-32_qubits-8_pennylane",
            "execution": "raw_results/Executor/2026-07-27_10-35-32_qubits-8_pennylane",
            "gradient":  "raw_results/Executor/2026-07-31_20-59-25_qubits-8_pennylane",
        },
    },
    10: {
        "pennylane_dir": "raw_results/Pennylane/2026-07-27_10-27-40_qubits-10",
        "executor_dirs": {
            "creation":  "raw_results/Executor/2026-07-27_10-36-03_qubits-10_pennylane",
            "execution": "raw_results/Executor/2026-07-27_10-36-03_qubits-10_pennylane",
            "gradient":  "raw_results/Executor/2026-07-31_20-59-32_qubits-10_pennylane",
        },
    },
    12: {
        "pennylane_dir": "raw_results/Pennylane/2026-07-27_10-28-12_qubits-12",
        "executor_dirs": {
            "creation":  "raw_results/Executor/2026-07-27_10-36-28_qubits-12_pennylane",
            "execution": "raw_results/Executor/2026-07-27_10-36-28_qubits-12_pennylane",
            "gradient":  "raw_results/Executor/2026-07-31_20-59-39_qubits-12_pennylane",
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

    for q, run_cfg in RUNS.items():
        style = QUBIT_STYLE[q]
        for mode in MODES:
            df_pl = load(run_cfg["pennylane_dir"], "benchmark", mode)
            ax.plot(
                df_pl["total_gates"], df_pl["qnc_mem_avg"],
                linestyle=FRAMEWORK_STYLE["PennyLane (roh)"], color=MODE_COLOR[mode],
                **style,
            )
            x_last, y_last = df_pl["total_gates"].iloc[-1], df_pl["qnc_mem_avg"].iloc[-1]
            if x_last > end_points[mode][0]:
                end_points[mode] = (x_last, y_last)

            df_e = load(run_cfg["executor_dirs"][mode], "benchmark_executor_pennylane", mode)
            ax.plot(
                df_e["total_gates"], df_e["qnc_mem_avg"],
                linestyle=FRAMEWORK_STYLE["Executor (PennyLane-Backend)"], color=MODE_COLOR[mode],
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
        "PennyLane (roh) vs. Executor (PennyLane-Backend) – Speicherverbrauch (Peak) aller Modi "
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
    out_path = out_dir / "pennylane_vs_executor_qubits_mem.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
