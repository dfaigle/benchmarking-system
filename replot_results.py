"""
Verwendung:
    python plot_results.py <Results\benchmark_non_clifford_creation_V2.csv>
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

MODE_LABELS = {
    "creation":  "Erstellungszeit",
    "execution": "Ausführungszeit",
    "gradient":  "Gradienten-Berechnungszeit",
}

METHOD_CFG = [
    ("Tape",             "tape_avg", "tape_std", "tab:blue"),
    ("QNode (no cache)", "qnc_avg",  "qnc_std",  "tab:orange"),
    ("QNode (cached)",   "qc_avg",   "qc_std",   "tab:green"),
]


def plot_results(df: pd.DataFrame) -> None:
    qubit_vals = sorted(df["num_qubits"].unique())
    gate_set   = df["gate_set"].iloc[0]
    mode       = df["benchmark_mode"].iloc[0]
    mode_label = MODE_LABELS.get(mode, mode)

    n_qubits = len(qubit_vals)
    fig, axes = plt.subplots(1, n_qubits, figsize=(6 * n_qubits, 5), squeeze=False)
    axes = axes[0]

    fig.suptitle(
        f"{mode_label}  |  Gate-Set: {gate_set!r}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, nq in zip(axes, qubit_vals):
        sub = df[df["num_qubits"] == nq].sort_values("total_gates")
        x   = sub["total_gates"].values

        for method_label, col_avg, col_std, color in METHOD_CFG:
            y = sub[col_avg].values
            s = sub[col_std].values

            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4,
                    label=method_label, color=color)
            ax.fill_between(x, np.maximum(y - s, 1e-9), y + s,
                            alpha=0.18, color=color)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Anzahl Gatter", fontsize=11)
        ax.set_ylabel("Zeit (s)", fontsize=11)
        ax.set_title(f"{nq} Qubits", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark-Ergebnisse aus CSV plotten")
    parser.add_argument("csv", type=Path, help="Results\benchmark_non_clifford_creation_V2.csv")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Fehler: Datei nicht gefunden: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)

    required = {"num_qubits", "total_gates", "gate_set", "benchmark_mode",
                "tape_avg", "tape_std", "qnc_avg", "qnc_std", "qc_avg", "qc_std"}
    missing = required - set(df.columns)
    if missing:
        print(f"Fehler: CSV fehlt Spalten: {missing}", file=sys.stderr)
        sys.exit(1)

    plot_results(df)


if __name__ == "__main__":
    main()
