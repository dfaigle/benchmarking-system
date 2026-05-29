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
# ➤  GATE-SET AUSWAHL  ←  hier anpassen
# =========================================================
#
#  Mögliche Werte:
#    "clifford"               → H, S, CNOT, PauliX/Y/Z
#    "non_clifford"           → T, RX, RY, RZ, Rot, CRX, CRY, CRZ,
#                               ControlledPhaseShift, Toffoli
#    "clifford_t"             → H, T, CNOT
#    "single_qubit_plus_cnot" → RX, RY, RZ, CNOT
#    "rot_cnot"               → Rot, CNOT
#
GATE_SET_CHOICE = "clifford"

# =========================================================
# Gate-Definitionen
# =========================================================

CLIFFORD_GATES = [
    "Hadamard", "S", "CNOT", "PauliX", "PauliY", "PauliZ"
]

NON_CLIFFORD_GATES = [
    "T", "RX", "RY", "RZ", "Rot",
    "CRX", "CRY", "CRZ", "ControlledPhaseShift", "Toffoli"
]

UNIVERSAL_GATE_SETS = {
    "clifford_t":              ["Hadamard", "T", "CNOT"],
    "single_qubit_plus_cnot":  ["RX", "RY", "RZ", "CNOT"],
    "rot_cnot":                ["Rot", "CNOT"],
}

# =========================================================
# Auflösung der Auswahl
# =========================================================

def resolve_gate_set(choice: str) -> list:
    mapping = {
        "clifford":     CLIFFORD_GATES,
        "non_clifford": NON_CLIFFORD_GATES,
        **UNIVERSAL_GATE_SETS,
    }
    if choice not in mapping:
        valid = ", ".join(f'"{k}"' for k in mapping)
        raise ValueError(f"Unbekanntes Gate-Set '{choice}'. Gültig: {valid}")
    return mapping[choice]

ACTIVE_GATE_SET = resolve_gate_set(GATE_SET_CHOICE)
print(f"Aktives Gate-Set: {GATE_SET_CHOICE!r}  →  {ACTIVE_GATE_SET}\n")

# =========================================================
# Konfiguration
# =========================================================

RESULT_DIR = Path("../results")
RESULT_DIR.mkdir(exist_ok=True)

SEED    = 42
REPEATS = 5

QUBIT_CONFIGS = [2, 3, 4]

# Logarithmisch gestufte Gate-Anzahlen: ~20 Stufen von 10 bis 10 000
GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(10_000), 20)).astype(int)
)

# =========================================================
# Versioned CSV
# =========================================================

version = 1
while True:
    CSV_FILE = RESULT_DIR / f"benchmark_{GATE_SET_CHOICE}_V{version}.csv"
    if not CSV_FILE.exists():
        break
    version += 1

# =========================================================
# Gate-Eigenschaften
# =========================================================

_ANGLE_1  = {"RX", "RY", "RZ", "CRX", "CRY", "CRZ", "ControlledPhaseShift"}
_ANGLE_3  = {"Rot"}
_NEEDS_2Q = {"CNOT", "CRX", "CRY", "CRZ", "ControlledPhaseShift"}
_NEEDS_3Q = {"Toffoli"}

# =========================================================
# Gate-Sequenz generieren
# =========================================================
# Jedes Element: (gate_name: str, wires: list[int], params: list[float])

def generate_gate_sequence(
    num_qubits: int,
    total_gates: int,
    gate_set: list,
    seed: int = 42,
) -> list:
    rng = np.random.default_rng(seed)
    sequence = []

    for _ in range(total_gates):
        gate = str(rng.choice(gate_set))
        w    = int(rng.integers(0, num_qubits))

        if gate in _NEEDS_3Q:
            if num_qubits >= 3:
                wires = [int(x) for x in rng.choice(num_qubits, size=3, replace=False)]
                sequence.append((gate, wires, []))
            else:
                sequence.append(("Hadamard", [w], []))
        elif gate in _NEEDS_2Q:
            if num_qubits >= 2:
                target = int((w + 1) % num_qubits)
                params = [float(rng.uniform(0, 2 * np.pi))] if gate != "CNOT" else []
                sequence.append((gate, [w, target], params))
            else:
                sequence.append(("Hadamard", [w], []))
        elif gate in _ANGLE_3:
            params = rng.uniform(0, 2 * np.pi, size=3).tolist()
            sequence.append((gate, [w], params))
        elif gate in _ANGLE_1:
            params = [float(rng.uniform(0, 2 * np.pi))]
            sequence.append((gate, [w], params))
        else:
            sequence.append((gate, [w], []))

    return sequence

# =========================================================
# Gate anwenden
# =========================================================

def apply_gate(gate_name: str, wires: list, params: list) -> None:
    if   gate_name == "Hadamard":             qml.Hadamard(wires=wires[0])
    elif gate_name == "S":                    qml.S(wires=wires[0])
    elif gate_name == "CNOT":                 qml.CNOT(wires=wires)
    elif gate_name == "PauliX":               qml.PauliX(wires=wires[0])
    elif gate_name == "PauliY":               qml.PauliY(wires=wires[0])
    elif gate_name == "PauliZ":               qml.PauliZ(wires=wires[0])
    elif gate_name == "T":                    qml.T(wires=wires[0])
    elif gate_name == "RX":                   qml.RX(params[0], wires=wires[0])
    elif gate_name == "RY":                   qml.RY(params[0], wires=wires[0])
    elif gate_name == "RZ":                   qml.RZ(params[0], wires=wires[0])
    elif gate_name == "Rot":                  qml.Rot(params[0], params[1], params[2], wires=wires[0])
    elif gate_name == "CRX":                  qml.CRX(params[0], wires=wires)
    elif gate_name == "CRY":                  qml.CRY(params[0], wires=wires)
    elif gate_name == "CRZ":                  qml.CRZ(params[0], wires=wires)
    elif gate_name == "ControlledPhaseShift": qml.ControlledPhaseShift(params[0], wires=wires)
    elif gate_name == "Toffoli":              qml.Toffoli(wires=wires)
    else:
        raise ValueError(f"Unbekanntes Gatter: {gate_name!r}")

# =========================================================
# Circuit-Builder  (3 Varianten)
# =========================================================

def build_tape_runner(num_qubits: int, gate_sequence: list):
    """
    Tape-Variante: kein QNode, direkte low-level Ausführung.
    Niedrigster Overhead, kein Caching, kein Tracing.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    def run():
        with qml.tape.QuantumTape() as tape:
            for gate_name, wires, params in gate_sequence:
                apply_gate(gate_name, wires, params)
            qml.probs(wires=range(num_qubits))
        return qml.execute([tape], dev)

    return run


def build_qnode_no_cache(num_qubits: int, gate_sequence: list):
    """
    QNode ohne Cache: bei jedem Aufruf wird der Circuit neu getract
    und kompiliert. Misst den vollen QNode-Overhead.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=False)
    def circuit():
        for gate_name, wires, params in gate_sequence:
            apply_gate(gate_name, wires, params)
        return qml.probs(wires=range(num_qubits))

    return circuit


def build_qnode_cached(num_qubits: int, gate_sequence: list):
    """
    QNode mit Cache: Circuit-Graph wird nach dem ersten Aufruf
    wiederverwendet. Zeigt den Gewinn durch Caching.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=True)
    def circuit():
        for gate_name, wires, params in gate_sequence:
            apply_gate(gate_name, wires, params)
        return qml.probs(wires=range(num_qubits))

    return circuit


def build_grad_circuit(num_qubits: int, gate_sequence: list):
    """
    Gradient-Circuit (QNode, diff_method='best'):
      • Trainierbare RY-Schicht pro Qubit  →  qml.grad-fähig
      • Dann der Clifford-/Gate-Block mit festen zufälligen Parametern
      • Ausgabe: Erwartungswert ⟨Z₀⟩
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, diff_method="best")
    def circuit(params):
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        for gate_name, wires, g_params in gate_sequence:
            apply_gate(gate_name, wires, g_params)
        return qml.expval(qml.PauliZ(0))

    return circuit

# =========================================================
# Timing
# =========================================================

def measure_runtime(func, repeats: int = 5) -> dict:
    func()      # Warm-up (füllt Cache, initialisiert JIT etc.)
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

        # --- Erstellungszeit: Gate-Sequenz generieren ---
        creation_times = []
        gate_sequence  = None
        for i in range(REPEATS + 1):
            t0 = time.perf_counter()
            gs = generate_gate_sequence(num_qubits, total_gates, ACTIVE_GATE_SET, seed=SEED)
            elapsed = time.perf_counter() - t0
            if i == 0:
                gate_sequence = gs   # erste Sequenz behalten
            else:
                creation_times.append(elapsed)   # Cold-Start (i=0) weglassen
        creation_avg = float(np.mean(creation_times))
        creation_std = float(np.std(creation_times))

        # --- Tape ---
        tape_stats = measure_runtime(build_tape_runner(num_qubits, gate_sequence), REPEATS)

        # --- QNode ohne Cache ---
        qnc_stats = measure_runtime(build_qnode_no_cache(num_qubits, gate_sequence), REPEATS)

        # --- QNode mit Cache ---
        qc_stats = measure_runtime(build_qnode_cached(num_qubits, gate_sequence), REPEATS)

        # --- Gradient (QNode) ---
        grad_fn    = qml.grad(build_grad_circuit(num_qubits, gate_sequence))
        params     = np.zeros(num_qubits)
        grad_stats = measure_runtime(lambda: grad_fn(params.copy()), REPEATS)

        print(
            f"  create={creation_avg:.5f}s  "
            f"tape={tape_stats['avg']:.5f}s  "
            f"qnode_nc={qnc_stats['avg']:.5f}s  "
            f"qnode_c={qc_stats['avg']:.5f}s  "
            f"grad={grad_stats['avg']:.5f}s"
        )

        results.append({
            "timestamp":      datetime.now().isoformat(),
            "gate_set":       GATE_SET_CHOICE,
            "num_qubits":     num_qubits,
            "total_gates":    total_gates,
            # Erstellungszeit
            "create_avg":     creation_avg,
            "create_std":     creation_std,
            # Tape
            **{f"tape_{k}": v for k, v in tape_stats.items()},
            # QNode ohne Cache
            **{f"qnc_{k}": v for k, v in qnc_stats.items()},
            # QNode mit Cache
            **{f"qc_{k}": v for k, v in qc_stats.items()},
            # Gradient
            **{f"grad_{k}": v for k, v in grad_stats.items()},
        })

# =========================================================
# CSV speichern
# =========================================================

df = pd.DataFrame(results)
df.to_csv(CSV_FILE, index=False)
print(f"\nErgebnisse gespeichert: {CSV_FILE}")

# =========================================================
# Plot  (5 Subplots)
#
#  Layout:
#   ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
#   │ Erstellungs- │     Tape     │  QNode       │  QNode       │  Gradient    │
#   │    zeit      │  Ausführung  │  (no cache)  │  (cached)    │  (QNode)     │
#   └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
#
#  Farbe  = Qubit-Anzahl
#  Fehlerband (±std) bei allen Subplots
# =========================================================

def plot_results(df: pd.DataFrame, save_path: Path | None = None) -> None:
    qubit_vals = sorted(df["num_qubits"].unique())
    palette    = plt.cm.Set2.colors
    gate_set   = df["gate_set"].iloc[0]

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    fig.suptitle(
        f"Benchmark  |  Gate-Set: {gate_set!r}  ({', '.join(ACTIVE_GATE_SET)})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # (subplot-titel, avg-spalte, std-spalte)
    metrics = [
        ("Erstellungszeit",                "create_avg", "create_std"),
        ("Ausführung\nTape",               "tape_avg",   "tape_std"),
        ("Ausführung\nQNode (no cache)",   "qnc_avg",    "qnc_std"),
        ("Ausführung\nQNode (cached)",     "qc_avg",     "qc_std"),
        ("Gradienten-\nBerechnungszeit",   "grad_avg",   "grad_std"),
    ]

    for ax, (title, col_avg, col_std) in zip(axes, metrics):
        for i, nq in enumerate(qubit_vals):
            sub   = df[df["num_qubits"] == nq].sort_values("total_gates")
            color = palette[i % len(palette)]
            x     = sub["total_gates"].values
            y     = sub[col_avg].values
            s     = sub[col_std].values

            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4,
                    label=f"{nq} Qubits", color=color)
            ax.fill_between(x, np.maximum(y - s, 1e-9), y + s,
                            alpha=0.18, color=color)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Anzahl Gatter", fontsize=10)
        ax.set_ylabel("Zeit (s)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # Trennlinie nach Erstellungszeit und vor Gradient
    for spine_ax in [axes[1], axes[4]]:
        for spine in spine_ax.spines.values():
            spine.set_linewidth(1.8)
            spine.set_edgecolor("#555")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")

    plt.show()


PLOT_FILE = RESULT_DIR / f"benchmark_{GATE_SET_CHOICE}_V{version}.png"
plot_results(df, save_path=PLOT_FILE)