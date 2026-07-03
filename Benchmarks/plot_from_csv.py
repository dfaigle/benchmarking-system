"""
Plottet Zeit + Speicher aus einer (auch unvollständigen) Benchmark-CSV.

Nützlich, wenn ein Lauf abgebrochen ist: die inkrementell geschriebene CSV
enthält alle bis dahin fertigen Log-Punkte und kann damit trotzdem geplottet
werden.

Das Framework wird automatisch am Spalten-Schema erkannt:
    PennyLane  → tape_ / qnc_ / qc_
    Qiskit     → qc_   / est_ / estt_
    Executor   → qnc_  / qc_          (Abstraktionsschicht, 2 Linien)

Beispiele:
    python plot_from_csv.py ../Results/Pennylane/benchmark_non_clifford_gradient_V1.csv
    python plot_from_csv.py ../Results/Executor/benchmark_executor_pennylane_clifford_execution_V1.csv
    python plot_from_csv.py <pfad.csv> --no-save     # nur anzeigen, nichts speichern
    python plot_from_csv.py <pfad.csv> --no-show     # nur speichern, kein Fenster
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

MODE_LABELS = {
    "creation":  "Erstellungszeit",
    "execution": "Ausführungszeit",
    "gradient":  "Gradienten-Berechnungszeit",
}

# =========================================================
# Framework anhand der Spalten erkennen
# =========================================================
# Reihenfolge wichtig: PennyLane hat tape_+qnc_+qc_, Qiskit hat qc_+est_+estt_,
# Executor hat nur qnc_+qc_. Daher erst auf die eindeutigen Präfixe testen
# (tape_ bzw. est_), Executor als Rest über qnc_ ohne tape_.

def detect_methods(columns) -> tuple[str, list]:
    cols = set(columns)
    if "tape_avg" in cols:
        methods = [
            ("Tape",             "tape", "tab:blue"),
            ("QNode (no cache)", "qnc",  "tab:orange"),
            ("QNode (cached)",   "qc",   "tab:green"),
        ]
        return "PennyLane", methods
    if "est_avg" in cols:
        methods = [
            ("QuantumCircuit + Statevector", "qc",   "tab:blue"),
            ("Estimator (kein Transpile)",   "est",  "tab:orange"),
            ("Estimator (transpiliert)",     "estt", "tab:green"),
        ]
        return "Qiskit", methods
    if "qnc_avg" in cols:
        methods = [
            ("Executor (no cache)", "qnc", "tab:orange"),
            ("Executor (cached)",   "qc",  "tab:green"),
        ]
        return "Executor", methods
    raise ValueError(
        "Unbekanntes CSV-Format: weder 'tape_avg' (PennyLane), 'est_avg' (Qiskit) "
        "noch 'qnc_avg' (Executor) in den Spalten gefunden."
    )

# =========================================================
# Generischer Plot (1 Subplot pro Qubit-Konfiguration)
# =========================================================

def plot_metric(
    df: pd.DataFrame,
    methods: list,
    framework: str,
    avg_suffix: str,
    std_suffix: str,
    ylabel: str,
    title: str,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    qubit_vals = sorted(df["num_qubits"].unique())
    gate_set   = df["gate_set"].iloc[0] if "gate_set" in df.columns else "?"

    n_qubits = len(qubit_vals)
    fig, axes = plt.subplots(1, n_qubits, figsize=(6 * n_qubits, 5), squeeze=False)
    axes = axes[0]

    fig.suptitle(
        f"{framework} – {title}  |  Gate-Set: {gate_set!r}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, nq in zip(axes, qubit_vals):
        sub = df[df["num_qubits"] == nq].sort_values("total_gates")
        x   = sub["total_gates"].values

        for method_label, prefix, color in methods:
            col_avg = f"{prefix}_{avg_suffix}"
            col_std = f"{prefix}_{std_suffix}"
            if col_avg not in sub.columns:
                continue
            y = sub[col_avg].values
            s = sub[col_std].values if col_std in sub.columns else np.zeros_like(y)

            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4,
                    label=method_label, color=color)
            ax.fill_between(x, np.maximum(y - s, 1e-9), y + s,
                            alpha=0.18, color=color)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Anzahl Gatter", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{nq} Qubits", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

# =========================================================
# Main
# =========================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plottet Zeit + Speicher aus einer (auch unvollständigen) Benchmark-CSV."
    )
    ap.add_argument("csv", type=Path, help="Pfad zur Ergebnis-CSV")
    ap.add_argument("--no-save", action="store_true", help="PNGs nicht speichern")
    ap.add_argument("--no-show", action="store_true", help="Fenster nicht anzeigen")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV nicht gefunden: {args.csv}")

    # on_bad_lines='skip' fängt eine evtl. abgeschnittene letzte Zeile ab
    df = pd.read_csv(args.csv, on_bad_lines="skip")
    df = df.dropna(subset=["num_qubits", "total_gates"])
    if df.empty:
        raise SystemExit("CSV enthält keine gültigen Datenzeilen.")
    df["num_qubits"]  = df["num_qubits"].astype(int)
    df["total_gates"] = df["total_gates"].astype(int)

    framework, methods = detect_methods(df.columns)
    # Executor-CSVs führen zusätzlich das Backend als Spalte → an den Titel hängen
    if "backend" in df.columns:
        framework = f"{framework} / {df['backend'].iloc[0]}"

    mode       = df["benchmark_mode"].iloc[0] if "benchmark_mode" in df.columns else ""
    mode_label = MODE_LABELS.get(mode, mode or "Laufzeit")

    qubits     = sorted(int(q) for q in df["num_qubits"].unique())
    points_per = {int(q): int(n) for q, n in df.groupby("num_qubits").size().items()}
    print(
        f"{framework}-CSV geladen: {len(df)} Zeilen, "
        f"Qubits={qubits}, Gatter-Punkte pro Qubit={points_per}"
    )

    stem      = args.csv.with_suffix("")   # Pfad ohne .csv
    time_png  = None if args.no_save else Path(f"{stem}_time.png")
    mem_png   = None if args.no_save else Path(f"{stem}_mem.png")

    # --- Laufzeit ---
    plot_metric(
        df, methods, framework, "avg", "std",
        ylabel="Zeit (s)", title=mode_label,
        save_path=time_png, show=not args.no_show,
    )

    # --- Speicher (nur wenn *_mem_avg vorhanden) ---
    has_mem = any(f"{prefix}_mem_avg" in df.columns for _, prefix, _ in methods)
    if has_mem:
        plot_metric(
            df, methods, framework, "mem_avg", "mem_std",
            ylabel="Peak-Speicher (MiB)",
            title=f"Speicherverbrauch (Peak) – {mode_label}",
            save_path=mem_png, show=not args.no_show,
        )
    else:
        print("Keine Speicher-Spalten (*_mem_avg) gefunden — Speicher-Plot übersprungen.")


if __name__ == "__main__":
    main()
