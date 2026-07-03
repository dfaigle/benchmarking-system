import pennylane as qml
import pennylane.numpy as pnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gc
import time
import tracemalloc
import sys
from pathlib import Path
from datetime import datetime

# Windows-Konsole auf UTF-8 stellen (sonst UnicodeEncodeError bei → und Umlauten)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =========================================================
# ➤  GATE-SET AUSWAHL  ←  hier anpassen
# =========================================================
#
#  "clifford"               → H, S, CNOT, PauliX/Y/Z
#  "non_clifford"           → T, RX, RY, RZ, Rot, CRX, CRY, CRZ,
#                             ControlledPhaseShift, Toffoli
#  "clifford_t"             → H, T, CNOT
#  "single_qubit_plus_cnot" → RX, RY, RZ, CNOT
#  "rot_cnot"               → Rot, CNOT
#
GATE_SET_CHOICE = "clifford"

# =========================================================
# ➤  BENCHMARK-MODUS  ←  hier anpassen
# =========================================================
#
#  "creation"  → Zeit um den Circuit aufzubauen (kein Ausführen)
#                  Tape:         QuantumTape befüllen
#                  QNode:        Erster Aufruf  (Python-Tracing + Graph-Aufbau)
#
#  "execution" → Zeit der reinen Simulation nach dem Aufbau
#                  Tape:         qml.execute([tape], dev)
#                  QNode:        circuit()-Aufruf (nach Warm-up)
#
#  "gradient"  → Zeit der Gradienten-Berechnung via Parameter-Shift
#                  Tape:         qml.gradients.param_shift(tape) + execute
#                  QNode:        qml.grad(circuit)(params)
#
BENCHMARK_MODE = "execution"

# =========================================================
# Gate-Definitionen
# =========================================================

CLIFFORD_GATES = ["Hadamard", "S", "CNOT", "PauliX", "PauliY", "PauliZ"]

NON_CLIFFORD_GATES = [
    "T", "RX", "RY", "RZ", "Rot",
    "CRX", "CRY", "CRZ", "ControlledPhaseShift", "Toffoli",
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

VALID_MODES = {"creation", "execution", "gradient"}
if BENCHMARK_MODE not in VALID_MODES:
    raise ValueError(f"Unbekannter Modus '{BENCHMARK_MODE}'. Gültig: {VALID_MODES}")

ACTIVE_GATE_SET = resolve_gate_set(GATE_SET_CHOICE)
print(f"Gate-Set : {GATE_SET_CHOICE!r}  →  {ACTIVE_GATE_SET}")
print(f"Modus    : {BENCHMARK_MODE!r}\n")

# =========================================================
# Konfiguration
# =========================================================

RESULT_DIR = Path(__file__).parent.parent / "Results" / "Pennylane"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED    = 42
REPEATS = 4

QUBIT_CONFIGS = [5]

GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(1000), 20)).astype(int)
)
# [    10     16     26     43     70    113    183    298    483
#      785   1274   2069   3360   5456   8859  14384  23357  37927
#      61585 100000]

# a_n = 10 * (100000 / 10)^(n / 19)

# =========================================================
# Pro Lauf ein eigener Unterordner: Results/Pennylane/run_<N>/
# =========================================================

run = 1
while (RESULT_DIR / f"run_{run}").exists():
    run += 1
RUN_DIR = RESULT_DIR / f"run_{run}"
RUN_DIR.mkdir(parents=True)

_stub    = f"benchmark_{GATE_SET_CHOICE}_{BENCHMARK_MODE}"
CSV_FILE = RUN_DIR / f"{_stub}.csv"

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
# Jedes Element: (gate_name, wires, params)

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
            sequence.append((gate, [w], rng.uniform(0, 2 * np.pi, size=3).tolist()))
        elif gate in _ANGLE_1:
            sequence.append((gate, [w], [float(rng.uniform(0, 2 * np.pi))]))
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
# Runner-Builder je Modus
# =========================================================

# ------ CREATION ------
# Misst: wie lange braucht jede Abstraktion, um den Circuit aufzubauen?
# Device-Erstellung ist für alle drei ausgelagert (gleiche Baseline).
#
#   Tape:           QuantumTape-Kontext befüllen (kein execute)
#   QNode no cache: QNode-Funktion definieren + ersten Aufruf (Tracing)
#   QNode cached:   QNode-Funktion definieren + ersten Aufruf (Tracing + Caching)

def runner_creation_tape(num_qubits: int, gate_sequence: list):
    def run():
        with qml.tape.QuantumTape() as _tape:
            for gn, ws, ps in gate_sequence:
                apply_gate(gn, ws, ps)
            qml.probs(wires=range(num_qubits))

    return run


def runner_creation_qnode_nc(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    def run():
        @qml.qnode(dev, cache=False)
        def circuit():
            for gn, ws, ps in gate_sequence:
                apply_gate(gn, ws, ps)
            return qml.probs(wires=range(num_qubits))

        circuit()   # erster Aufruf = Tracing

    return run


def runner_creation_qnode_c(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    def run():
        @qml.qnode(dev, cache=True)
        def circuit():
            for gn, ws, ps in gate_sequence:
                apply_gate(gn, ws, ps)
            return qml.probs(wires=range(num_qubits))

        circuit()   # erster Aufruf = Tracing + Caching

    return run


# ------ EXECUTION ------
# Misst: wie lange dauert die Simulation nach dem Aufbau?
#
#   Tape:           qml.execute([tape], dev) — tape ist vorgebaut
#   QNode no cache: circuit()-Aufruf mit Tracing + Simulation bei jedem Aufruf (cache=False)
#   QNode cached:   circuit()-Aufruf mit gecachtem Graph, nur Simulation neu (cache=True)

def runner_execution_tape(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    # Tape einmalig aufbauen
    with qml.tape.QuantumTape() as tape:
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        qml.probs(wires=range(num_qubits))

    def run():
        qml.execute([tape], dev)

    return run


def runner_execution_qnode_nc(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=False)
    def circuit():
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        return qml.probs(wires=range(num_qubits))

    return circuit


def runner_execution_qnode_c(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=True)
    def circuit():
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        return qml.probs(wires=range(num_qubits))

    return circuit


# ------ GRADIENT ------
# Misst: reine Ausführungszeit der Gradienten-Berechnung via Parameter-Shift.
# Tape und grad_tapes werden einmalig vorgebaut (analog zur QNode-Vorbereitung).
#
#   Trainierbare Schicht: RY(params[i]) pro Qubit, dann Gate-Block
#   Ausgabe: Erwartungswert ⟨Z₀⟩
#
#   Tape:           qml.execute(grad_tapes, dev) — grad_tapes einmalig vorberechnet
#   QNode no cache: qml.grad(circuit)(params) — Tracing + Simulation bei jedem Aufruf
#   QNode cached:   qml.grad(circuit)(params) — nur Simulation (Graph gecacht)

def runner_gradient_tape(num_qubits: int, gate_sequence: list):
    dev    = qml.device("default.qubit", wires=num_qubits)
    params = pnp.array(np.zeros(num_qubits), requires_grad=True)

    with qml.tape.QuantumTape() as tape:
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        qml.expval(qml.PauliZ(0))

    grad_tapes, fn = qml.gradients.param_shift(tape)

    def run():
        results = qml.execute(grad_tapes, dev)
        return fn(results)

    return run


def runner_gradient_qnode_nc(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=False, diff_method="parameter-shift")
    def circuit(params):
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        return qml.expval(qml.PauliZ(0))

    grad_fn = qml.grad(circuit)
    params  = pnp.array(np.zeros(num_qubits), requires_grad=True)

    def run():
        grad_fn(params.copy())

    return run


def runner_gradient_qnode_c(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=True, diff_method="parameter-shift")
    def circuit(params):
        for i in range(num_qubits):
            qml.RY(params[i], wires=i)
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        return qml.expval(qml.PauliZ(0))

    grad_fn = qml.grad(circuit)
    params  = pnp.array(np.zeros(num_qubits), requires_grad=True)

    def run():
        grad_fn(params.copy())

    return run


# =========================================================
# Modus → Runner-Tripel auflösen
# =========================================================

MODE_RUNNERS = {
    "creation":  (runner_creation_tape,  runner_creation_qnode_nc,  runner_creation_qnode_c),
    "execution": (runner_execution_tape, runner_execution_qnode_nc, runner_execution_qnode_c),
    "gradient":  (runner_gradient_tape,  runner_gradient_qnode_nc,  runner_gradient_qnode_c),
}

MODE_LABELS = {
    "creation":  "Erstellungszeit",
    "execution": "Ausführungszeit",
    "gradient":  "Gradienten-Berechnungszeit",
}

build_tape, build_qnc, build_qc = MODE_RUNNERS[BENCHMARK_MODE]

# =========================================================
# Timing
# =========================================================

def measure_runtime(func, repeats: int = 5) -> dict:
    func()      # Warm-up
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
# Speicher-Messung (Peak via tracemalloc)
# =========================================================
# Misst den Peak der Python-Allokationen während func() läuft.
# Reproduzierbar und erfasst die hier dominante Last (Op-/Tape-Objekte).
# NumPy-C-Buffer werden bewusst nicht erfasst (bei 10-15 Qubits vernachlässigbar).
# Getrennt vom Timing, da tracemalloc die Laufzeit verfälschen würde.

def measure_memory(func, repeats: int = 5) -> dict:
    func()      # Warm-up (lazy Imports/Caches nicht mitmessen)
    peaks = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        func()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak / 1024**2)   # MiB
    return {
        "avg": float(np.mean(peaks)),
        "std": float(np.std(peaks)),
        "min": float(np.min(peaks)),
        "max": float(np.max(peaks)),
    }

# =========================================================
# Benchmark-Schleife
# =========================================================

results = []
first_write = True   # Header nur beim allerersten Schreiben

for num_qubits in QUBIT_CONFIGS:
    for total_gates in GATE_CONFIGS:

        print(f"Qubits={num_qubits}  Gates={total_gates}")

        gate_sequence = generate_gate_sequence(
            num_qubits, total_gates, ACTIVE_GATE_SET, seed=SEED
        )

        tape_runner = build_tape(num_qubits, gate_sequence)
        qnc_runner  = build_qnc(num_qubits,  gate_sequence)
        qc_runner   = build_qc(num_qubits,   gate_sequence)

        tape_stats = measure_runtime(tape_runner, REPEATS)
        qnc_stats  = measure_runtime(qnc_runner,  REPEATS)
        qc_stats   = measure_runtime(qc_runner,   REPEATS)

        tape_mem = measure_memory(tape_runner, REPEATS)
        qnc_mem  = measure_memory(qnc_runner,  REPEATS)
        qc_mem   = measure_memory(qc_runner,   REPEATS)

        print(
            f"  tape={tape_stats['avg']:.5f}s/{tape_mem['avg']:.1f}MiB  "
            f"qnode_nc={qnc_stats['avg']:.5f}s/{qnc_mem['avg']:.1f}MiB  "
            f"qnode_c={qc_stats['avg']:.5f}s/{qc_mem['avg']:.1f}MiB"
        )

        row = {
            "timestamp":      datetime.now().isoformat(),
            "gate_set":       GATE_SET_CHOICE,
            "benchmark_mode": BENCHMARK_MODE,
            "num_qubits":     num_qubits,
            "total_gates":    total_gates,
            **{f"tape_{k}":     v for k, v in tape_stats.items()},
            **{f"qnc_{k}":      v for k, v in qnc_stats.items()},
            **{f"qc_{k}":       v for k, v in qc_stats.items()},
            **{f"tape_mem_{k}": v for k, v in tape_mem.items()},
            **{f"qnc_mem_{k}":  v for k, v in qnc_mem.items()},
            **{f"qc_mem_{k}":   v for k, v in qc_mem.items()},
        }
        results.append(row)

        # Zwischenergebnis sofort anhängen → überlebt einen Absturz
        pd.DataFrame([row]).to_csv(CSV_FILE, mode="a", header=first_write, index=False)
        first_write = False

# =========================================================
# CSV speichern
# =========================================================

df = pd.DataFrame(results)
print(f"\nErgebnisse gespeichert (inkrementell während des Laufs): {CSV_FILE}")

# =========================================================
# Plot  (3 Subplots: Tape | QNode no cache | QNode cached)
# =========================================================

def plot_metric(
    df: pd.DataFrame,
    avg_suffix: str,
    std_suffix: str,
    ylabel: str,
    title: str,
    save_path: Path | None = None,
) -> None:
    qubit_vals = sorted(df["num_qubits"].unique())
    gate_set   = df["gate_set"].iloc[0]

    method_cfg = [
        ("Tape",              f"tape_{avg_suffix}", f"tape_{std_suffix}", "tab:blue"),
        ("QNode (no cache)",  f"qnc_{avg_suffix}",  f"qnc_{std_suffix}",  "tab:orange"),
        ("QNode (cached)",    f"qc_{avg_suffix}",   f"qc_{std_suffix}",   "tab:green"),
    ]

    n_qubits = len(qubit_vals)
    fig, axes = plt.subplots(1, n_qubits, figsize=(6 * n_qubits, 5), squeeze=False)
    axes = axes[0]

    fig.suptitle(
        f"{title}  |  Gate-Set: {gate_set!r}  ({', '.join(ACTIVE_GATE_SET)})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, nq in zip(axes, qubit_vals):
        sub = df[df["num_qubits"] == nq].sort_values("total_gates")
        x   = sub["total_gates"].values

        for method_label, col_avg, col_std, color in method_cfg:
            y = sub[col_avg].values
            s = sub[col_std].values

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

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")

    plt.show()


mode_label = MODE_LABELS[BENCHMARK_MODE]

# --- Laufzeit ---
PLOT_FILE = RUN_DIR / f"{_stub}_time.png"
plot_metric(
    df, avg_suffix="avg", std_suffix="std",
    ylabel="Zeit (s)", title=mode_label,
    save_path=PLOT_FILE,
)

# --- Speicherverbrauch (Peak, tracemalloc) ---
MEM_PLOT_FILE = RUN_DIR / f"{_stub}_mem.png"
plot_metric(
    df, avg_suffix="mem_avg", std_suffix="mem_std",
    ylabel="Peak-Speicher (MiB)", title=f"Speicherverbrauch (Peak) – {mode_label}",
    save_path=MEM_PLOT_FILE,
)