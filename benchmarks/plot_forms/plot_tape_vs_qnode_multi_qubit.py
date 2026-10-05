"""Vergleichsplot Tape vs. QNode über mehrere Qubit-Zahlen.

Ein gemeinsames Diagramm für alle drei Benchmark-Modi (creation/execution/
gradient) und mehrere Qubit-Zahlen, im Stil von
``Results/FrameworkVergleich_5-12q_qiskit/qiskit_vs_executor_qubits_time.png``:
Farbe = Modus, Linienstil = Methode (Tape durchgezogen, QNode param-shift
gestrichelt, QNode Backprop gepunktet), Qubit-Zahl = Transparenz +
Liniendicke (blass/dünn = wenige Qubits, kräftig/dick = viele Qubits) statt
Marker-Form — bei vielen Linien in 3 überlappenden Farbclustern sind
unterschiedliche Marker-Formen kaum zu unterscheiden, eine Helligkeits-/
Dicken-Rampe für eine ordinale Größe wie die Qubit-Zahl liest sich sauberer.
Jede Modus-Gruppe wird zusätzlich direkt an der Linie beschriftet statt nur
über die Legende.

Ersetzt die Bänder-Darstellung aus ``alle_modi_ein_diagramm_time.png``
(Mittelwert über 5-12 Qubits + Streuband) durch eine Darstellung, die jede
Qubit-Zahl als eigene Linie zeigt.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

ROOT = Path(__file__).parent.parent.parent  # benchmarks/plot_forms/ -> Projekt-Wurzel

# Qubit-Zahl -> Tape-vs-QNode-Lauf-Ordner
RUNS = {
    5:  "raw_results/TapeVsQNode/2026-07-18_20-28-46_qubits-5",
    8:  "raw_results/TapeVsQNode/2026-07-18_20-29-07_qubits-8",
    10: "raw_results/TapeVsQNode/2026-07-18_20-29-39_qubits-10",
    12: "raw_results/TapeVsQNode/2026-07-18_20-30-00_qubits-12",
}

GATE_SET = "clifford"
FILE_PREFIX = "tape_vs_qnode"
MODES = ["creation", "execution", "gradient"]

MODE_COLOR = {
    "creation":  "tab:blue",
    "execution": "tab:orange",
    "gradient":  "mediumseagreen",
}
MODE_LABEL = {
    "creation":  "Erstellung",
    "execution": "Ausführung",
    "gradient":  "Gradient",
}

# Methode -> Linienstil
METHOD_STYLE = {
    "Tape":                          "-",
    "QNode (param-shift)":           "--",
    "QNode (Backprop, nur Gradient)": ":",
}

# Modus -> [(CSV-Spalte, Methode)] — Backprop (qbest) gibt es nur im Gradienten-Modus
MODE_SERIES = {
    "creation":  [("tape_avg", "Tape"), ("qnode_avg", "QNode (param-shift)")],
    "execution": [("tape_avg", "Tape"), ("qnode_avg", "QNode (param-shift)")],
    "gradient":  [("tape_avg", "Tape"), ("qps_avg", "QNode (param-shift)"),
                  ("qbest_avg", "QNode (Backprop, nur Gradient)")],
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

BACKPROP = "QNode (Backprop, nur Gradient)"


def annotation_label(mode: str, method: str) -> str:
    """Text der Direkt-Beschriftung am rechten Linienende.

    Im Gradienten-Modus laufen zwei Verfahren nebeneinander, deren Abstand
    ohne Benennung nicht interpretierbar ist — daher werden dort die beiden
    Cluster einzeln benannt statt nur der Modus.
    """
    if mode != "gradient":
        return MODE_LABEL[mode]
    return "Gradient (Backprop)" if method == BACKPROP else "Gradient (Parameter-Shift)"


def load(run_dir: str, mode: str) -> pd.DataFrame:
    f = ROOT / run_dir / f"{FILE_PREFIX}_{GATE_SET}_{mode}.csv"
    return pd.read_csv(f).sort_values("total_gates")


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Für die Direkt-Beschriftung: je Beschriftungs-Cluster der Endpunkt der
    # Linie mit dem größten x.  label -> (x, y, Farbe)
    end_points: dict[str, tuple[float, float, str]] = {}

    for q, run_dir in RUNS.items():
        style = QUBIT_STYLE[q]
        for mode in MODES:
            df = load(run_dir, mode)
            for column, method in MODE_SERIES[mode]:
                ax.plot(
                    df["total_gates"], df[column],
                    linestyle=METHOD_STYLE[method], color=MODE_COLOR[mode],
                    **style,
                )
                label = annotation_label(mode, method)
                x_last, y_last = df["total_gates"].iloc[-1], df[column].iloc[-1]
                if x_last > end_points.get(label, (0.0,))[0]:
                    end_points[label] = (x_last, y_last, MODE_COLOR[mode])

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

    # Direkt-Beschriftung am rechten Ende der am weitesten reichenden Linie
    for label, (x, y, color) in end_points.items():
        ax.annotate(
            label, xy=(x, y), xytext=(10, 0), textcoords="offset points",
            color=color, fontsize=13, fontweight="bold", va="center",
        )

    # Legende 1: Methode (Linienstil)
    method_handles = [
        plt.Line2D([0], [0], color="#555555", linestyle=ls, linewidth=2, label=label)
        for label, ls in METHOD_STYLE.items()
    ]
    legend1 = ax.legend(
        handles=method_handles, title="Methode  (Farbe = Modus)",
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
    out_path = out_dir / "tape_vs_qnode_qubits_time.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
