import pennylane as qml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gc
import time
from pathlib import Path
from datetime import datetime

# =========================================================
# Konfiguration
# =========================================================

RESULT_DIR = Path("../results")
RESULT_DIR.mkdir(exist_ok=True)

SEED    = 42
REPEATS = 5

QUBIT_CONFIGS = [2, 3, 4]

# Logarithmisch gestufte Gate-Anzahlen: ~10 Stufen von 10 bis 10 000
GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(10_000), 20)).astype(int)
)

# =========================================================
# Clifford-Gatter-Liste
# =========================================================

CLIFFORD_GATES = ["H", "S", "CNOT", "X", "Y", "Z"]

# =========================================================
# Nächste CSV-Version bestimmen
# =========================================================

version = 1
while True:
    CSV_FILE = RESULT_DIR / f"benchmark_clifford_V{version}.csv"
    if not CSV_FILE.exists():
        break
    version += 1

# =========================================================
# Gate-Sequenz generieren
# =========================================================

def generate_gate_sequence(num_qubits: int, total_gates: int, seed: int = 42) -> list:
    """
    Wählt zufällig Clifford-Gatter aus CLIFFORD_GATES und gibt eine Liste
    von (gate_name, wires)-Tupeln zurück.
    CNOT braucht 2 Qubits; bei num_qubits == 1 wird stattdessen H verwendet.
    """
    rng = np.random.default_rng(seed)
    sequence = []

    for _ in range(total_gates):
        gate = rng.choice(CLIFFORD_GATES)
        w = int(rng.integers(0, num_qubits))

        if gate == "CNOT":
            if num_qubits >= 2:
                target = int((w + 1) % num_qubits)
                sequence.append(("CNOT", [w, target]))
            else:
                sequence.append(("H", [w]))          # Fallback
        else:
            sequence.append((gate, [w]))

    return sequence


def apply_gate(gate_name: str, wires: list) -> None:
    """Wendet ein einzelnes Clifford-Gatter in einem PennyLane-Kontext an."""
    if gate_name == "H":
        qml.Hadamard(wires=wires[0])
    elif gate_name == "S":
        qml.S(wires=wires[0])
    elif gate_name == "CNOT":
        qml.CNOT(wires=wires)
    elif gate_name == "X":
        qml.PauliX(wires=wires[0])
    elif gate_name == "Y":
        qml.PauliY(wires=wires[0])
    elif gate_name == "Z":
        qml.PauliZ(wires=wires[0])

# =========================================================
# Circuit-Builder
# =========================================================

def build_exec_circuit(num_qubits: int, gate_sequence: list):
    """Reiner Ausführungs-Circuit (kein Parameter, probs-Output)."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=False)
    def circuit():
        for gate_name, wires in gate_sequence:
            apply_gate(gate_name, wires)
        return qml.probs(wires=range(num_qubits))

    return circuit


def build_grad_circuit(num_qubits: int, gate_sequence: list):
    """
    Gradient-Circuit: eine trainierbare RY-Schicht pro Qubit,
    dann der Clifford-Block, dann Erwartungswert für qml.grad.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method="best")
    def circuit(params):
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        for gate_name, wires in gate_sequence:
            apply_gate(gate_name, wires)
        return qml.expval(qml.PauliZ(0))

    return circuit

# =========================================================
# Timing-Hilfsfunktion
# =========================================================

def measure_runtime(func, repeats: int = 5) -> dict:
    func()          # Warm-up
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return {
        "avg": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }

# =========================================================
# Benchmark-Schleife
# =========================================================

results = []

for num_qubits in QUBIT_CONFIGS:
    for total_gates in GATE_CONFIGS:

        print(f"\nQubits={num_qubits}  Gates={total_gates}")

        # --- 1. Erstellungszeit: Gate-Sequenz generieren ---
        t0 = time.perf_counter()
        gate_sequence = generate_gate_sequence(num_qubits, total_gates, seed=SEED)
        creation_time = time.perf_counter() - t0
        print(f"  Creation:  {creation_time:.6f} s")

        # --- 2. Ausführungszeit ---
        exec_circuit = build_exec_circuit(num_qubits, gate_sequence)
        exec_stats   = measure_runtime(exec_circuit, REPEATS)
        print(f"  Execution: {exec_stats['avg']:.6f} s  (±{exec_stats['std']:.6f})")

        # --- 3. Gradienten-Berechnungszeit ---
        grad_circuit = build_grad_circuit(num_qubits, gate_sequence)
        params       = np.zeros(num_qubits)
        grad_fn      = qml.grad(grad_circuit)
        grad_stats   = measure_runtime(lambda: grad_fn(params.copy()), REPEATS)
        print(f"  Gradient:  {grad_stats['avg']:.6f} s  (±{grad_stats['std']:.6f})")

        results.append({
            "timestamp":     datetime.now().isoformat(),
            "num_qubits":    num_qubits,
            "total_gates":   total_gates,
            "creation_time": creation_time,
            **{f"exec_{k}": v for k, v in exec_stats.items()},
            **{f"grad_{k}": v for k, v in grad_stats.items()},
        })

# =========================================================
# CSV speichern
# =========================================================

df = pd.DataFrame(results)
df.to_csv(CSV_FILE, index=False)
print(f"\nErgebnisse gespeichert: {CSV_FILE}")

# =========================================================
# Plot-Funktion (kann auch separat aufgerufen werden)
# =========================================================

def plot_results(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """
    Erstellt drei nebeneinander liegende Log-Log-Plots:
      - Erstellungszeit
      - Ausführungszeit  (mit Fehlerband ±std)
      - Gradienten-Zeit  (mit Fehlerband ±std)
    """
    qubit_vals = sorted(df["num_qubits"].unique())
    palette    = plt.cm.Set2.colors

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Clifford-Circuit Benchmark  |  Zufällige Gatter aus {H, S, CNOT, X, Y, Z}",
        fontsize=13, fontweight="bold", y=1.01
    )

    metric_cfg = [
        ("Erstellungszeit",          "creation_time", None,       None),
        ("Ausführungszeit",          "exec_avg",       "exec_std", "exec_min"),
        ("Gradienten-Berechnungszeit","grad_avg",       "grad_std", "grad_min"),
    ]

    for ax, (title, col_avg, col_std, _) in zip(axes, metric_cfg):
        for i, nq in enumerate(qubit_vals):
            sub   = df[df["num_qubits"] == nq].sort_values("total_gates")
            color = palette[i % len(palette)]
            label = f"{nq} Qubits"

            x = sub["total_gates"].values
            y = sub[col_avg].values

            ax.plot(x, y, marker="o", linewidth=1.8, markersize=5,
                    label=label, color=color)

            if col_std and col_std in sub.columns:
                s = sub[col_std].values
                ax.fill_between(x, np.maximum(y - s, 1e-9), y + s,
                                alpha=0.18, color=color)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Anzahl Gatter", fontsize=11)
        ax.set_ylabel("Zeit (s)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")

    plt.show()


# =========================================================
# Plot direkt nach dem Benchmark ausführen
# =========================================================

PLOT_FILE = RESULT_DIR / f"benchmark_clifford_V{version}.png"
plot_results(df, save_path=PLOT_FILE)