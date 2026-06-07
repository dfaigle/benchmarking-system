"""
Qiskit Benchmarking – analog zu benchmarkLog_V4.py (PennyLane)

Abstraktionsebenen:
  qc   → QuantumCircuit + Statevector        (roh, kein Primitiv)
  est  → QuantumCircuit + StatevectorEstimator  (Primitiv, kein Transpile)
  estt → transpile(circuit) + StatevectorEstimator (Primitiv + Kompilierung)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gc
import time
from pathlib import Path
from datetime import datetime

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit import ParameterVector

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
GATE_SET_CHOICE = "non_clifford"

# =========================================================
# ➤  BENCHMARK-MODUS  ←  hier anpassen
# =========================================================
#
#  "creation"  → Zeit um den Circuit aufzubauen
#                  qc:   QuantumCircuit aufbauen (kein Ausführen)
#                  est:  QuantumCircuit + erster Estimator-Run (Setup-Overhead)
#                  estt: QuantumCircuit + transpile + erster Estimator-Run
#
#  "execution" → Zeit der reinen Simulation nach dem Aufbau
#                  qc:   Statevector(circuit).expectation_value(Z₀)
#                  est:  StatevectorEstimator.run() – circuit vorgebaut
#                  estt: StatevectorEstimator.run() – circuit vortranspiliert
#
#  "gradient"  → Zeit der Gradienten-Berechnung via Parameter-Shift
#                  Trainierbare Schicht: RY(θᵢ) pro Qubit, dann Gate-Block
#                  Ausgabe: ⟨Z₀⟩-Gradient für alle θᵢ
#                  qc:   Statevector + manueller Param-Shift
#                  est:  StatevectorEstimator + manueller Param-Shift
#                  estt: StatevectorEstimator + Param-Shift (vorkompiliert)
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
    "clifford_t":             ["Hadamard", "T", "CNOT"],
    "single_qubit_plus_cnot": ["RX", "RY", "RZ", "CNOT"],
    "rot_cnot":               ["Rot", "CNOT"],
}


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

RESULT_DIR = Path(__file__).parent.parent / "Results" / "Qiskit"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED    = 42
REPEATS = 4

QUBIT_CONFIGS = [10]

GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(100000), 20)).astype(int)
)

# =========================================================
# Versioned CSV
# =========================================================

version = 1
while True:
    CSV_FILE = RESULT_DIR / f"qiskit_{GATE_SET_CHOICE}_{BENCHMARK_MODE}_V{version}.csv"
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

def generate_gate_sequence(num_qubits: int, total_gates: int, gate_set: list, seed: int = 42) -> list:
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
# Gate anwenden (Qiskit QuantumCircuit API)
# =========================================================
# PennyLane → Qiskit:
#   Hadamard → h | S → s | CNOT → cx | PauliX/Y/Z → x/y/z | T → t
#   RX/RY/RZ → rx/ry/rz | Rot → u (3-Winkel) | CRX/CRY/CRZ → crx/cry/crz
#   ControlledPhaseShift → cp | Toffoli → ccx

def apply_gate(qc: QuantumCircuit, gate_name: str, wires: list, params: list) -> None:
    if   gate_name == "Hadamard":             qc.h(wires[0])
    elif gate_name == "S":                    qc.s(wires[0])
    elif gate_name == "CNOT":                 qc.cx(wires[0], wires[1])
    elif gate_name == "PauliX":               qc.x(wires[0])
    elif gate_name == "PauliY":               qc.y(wires[0])
    elif gate_name == "PauliZ":               qc.z(wires[0])
    elif gate_name == "T":                    qc.t(wires[0])
    elif gate_name == "RX":                   qc.rx(params[0], wires[0])
    elif gate_name == "RY":                   qc.ry(params[0], wires[0])
    elif gate_name == "RZ":                   qc.rz(params[0], wires[0])
    elif gate_name == "Rot":                  qc.u(params[0], params[1], params[2], wires[0])
    elif gate_name == "CRX":                  qc.crx(params[0], wires[0], wires[1])
    elif gate_name == "CRY":                  qc.cry(params[0], wires[0], wires[1])
    elif gate_name == "CRZ":                  qc.crz(params[0], wires[0], wires[1])
    elif gate_name == "ControlledPhaseShift": qc.cp(params[0], wires[0], wires[1])
    elif gate_name == "Toffoli":              qc.ccx(wires[0], wires[1], wires[2])
    else:
        raise ValueError(f"Unbekanntes Gatter: {gate_name!r}")

# =========================================================
# Runner-Builder je Modus
# =========================================================

# ------ CREATION ------
# Misst: wie lange braucht jede Abstraktion, um den Circuit aufzubauen?
#
#   qc:   QuantumCircuit befüllen (kein execute)
#   est:  QuantumCircuit + erster Estimator-Run (Tracing + Setup)
#   estt: QuantumCircuit + transpile + erster Estimator-Run

def runner_creation_qc(num_qubits: int, gate_sequence: list):
    def run():
        qc = QuantumCircuit(num_qubits)
        for gn, ws, ps in gate_sequence:
            apply_gate(qc, gn, ws, ps)
    return run


def runner_creation_est(num_qubits: int, gate_sequence: list):
    observable = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    estimator  = StatevectorEstimator()

    def run():
        qc = QuantumCircuit(num_qubits)
        for gn, ws, ps in gate_sequence:
            apply_gate(qc, gn, ws, ps)
        estimator.run([(qc, observable)]).result()

    return run


def runner_creation_estt(num_qubits: int, gate_sequence: list):
    observable = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    estimator  = StatevectorEstimator()

    def run():
        qc = QuantumCircuit(num_qubits)
        for gn, ws, ps in gate_sequence:
            apply_gate(qc, gn, ws, ps)
        tc = transpile(qc, optimization_level=1)
        estimator.run([(tc, observable)]).result()

    return run


# ------ EXECUTION ------
# Misst: wie lange dauert die Simulation nach dem Aufbau?
# Circuit und (wo nötig) transpilierter Circuit sind vorgebaut.
#
#   qc:   Statevector(circuit).expectation_value(Z₀)
#   est:  StatevectorEstimator.run() – circuit vorgebaut
#   estt: StatevectorEstimator.run() – circuit vortranspiliert

def runner_execution_qc(num_qubits: int, gate_sequence: list):
    observable = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    qc = QuantumCircuit(num_qubits)
    for gn, ws, ps in gate_sequence:
        apply_gate(qc, gn, ws, ps)

    def run():
        Statevector(qc).expectation_value(observable)

    return run


def runner_execution_est(num_qubits: int, gate_sequence: list):
    observable = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    estimator  = StatevectorEstimator()
    qc = QuantumCircuit(num_qubits)
    for gn, ws, ps in gate_sequence:
        apply_gate(qc, gn, ws, ps)

    def run():
        estimator.run([(qc, observable)]).result()

    return run


def runner_execution_estt(num_qubits: int, gate_sequence: list):
    observable = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    estimator  = StatevectorEstimator()
    qc = QuantumCircuit(num_qubits)
    for gn, ws, ps in gate_sequence:
        apply_gate(qc, gn, ws, ps)
    tc = transpile(qc, optimization_level=1)

    def run():
        estimator.run([(tc, observable)]).result()

    return run


# ------ GRADIENT ------
# Misst: reine Ausführungszeit der Gradienten-Berechnung via Parameter-Shift.
# Trainierbare Schicht: RY(θᵢ) pro Qubit, dann Gate-Block.
# grad_i = [f(θᵢ + π/2) – f(θᵢ – π/2)] / 2
#
#   qc:   Statevector + manueller Param-Shift (assign_parameters per Shift)
#   est:  StatevectorEstimator + Param-Shift (Parameter-Binding im Primitiv)
#   estt: StatevectorEstimator + Param-Shift (vorkompilierter Circuit)

def runner_gradient_qc(num_qubits: int, gate_sequence: list):
    observable   = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    pv           = ParameterVector("θ", length=num_qubits)
    qc           = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(pv[i], i)
    for gn, ws, ps in gate_sequence:
        apply_gate(qc, gn, ws, ps)
    param_values = np.zeros(num_qubits)

    def run():
        for i in range(num_qubits):
            plus_vals  = param_values.copy(); plus_vals[i]  += np.pi / 2
            minus_vals = param_values.copy(); minus_vals[i] -= np.pi / 2
            ev_p = Statevector(qc.assign_parameters(dict(zip(pv, plus_vals)))).expectation_value(observable).real
            ev_m = Statevector(qc.assign_parameters(dict(zip(pv, minus_vals)))).expectation_value(observable).real
            _ = (ev_p - ev_m) / 2

    return run


def runner_gradient_est(num_qubits: int, gate_sequence: list):
    observable   = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    estimator    = StatevectorEstimator()
    pv           = ParameterVector("θ", length=num_qubits)
    qc           = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(pv[i], i)
    for gn, ws, ps in gate_sequence:
        apply_gate(qc, gn, ws, ps)
    param_values = np.zeros(num_qubits)

    def run():
        for i in range(num_qubits):
            plus_vals  = param_values.copy(); plus_vals[i]  += np.pi / 2
            minus_vals = param_values.copy(); minus_vals[i] -= np.pi / 2
            ev_p = float(estimator.run([(qc, observable, plus_vals)]).result()[0].data.evs)
            ev_m = float(estimator.run([(qc, observable, minus_vals)]).result()[0].data.evs)
            _ = (ev_p - ev_m) / 2

    return run


def runner_gradient_estt(num_qubits: int, gate_sequence: list):
    observable   = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    estimator    = StatevectorEstimator()
    pv           = ParameterVector("θ", length=num_qubits)
    qc           = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(pv[i], i)
    for gn, ws, ps in gate_sequence:
        apply_gate(qc, gn, ws, ps)
    tc           = transpile(qc, optimization_level=1)
    param_values = np.zeros(num_qubits)

    def run():
        for i in range(num_qubits):
            plus_vals  = param_values.copy(); plus_vals[i]  += np.pi / 2
            minus_vals = param_values.copy(); minus_vals[i] -= np.pi / 2
            ev_p = float(estimator.run([(tc, observable, plus_vals)]).result()[0].data.evs)
            ev_m = float(estimator.run([(tc, observable, minus_vals)]).result()[0].data.evs)
            _ = (ev_p - ev_m) / 2

    return run

# =========================================================
# Modus → Runner-Tripel auflösen
# =========================================================

MODE_RUNNERS = {
    "creation":  (runner_creation_qc,  runner_creation_est,  runner_creation_estt),
    "execution": (runner_execution_qc, runner_execution_est, runner_execution_estt),
    "gradient":  (runner_gradient_qc,  runner_gradient_est,  runner_gradient_estt),
}

MODE_LABELS = {
    "creation":  "Erstellungszeit",
    "execution": "Ausführungszeit",
    "gradient":  "Gradienten-Berechnungszeit",
}

build_qc, build_est, build_estt = MODE_RUNNERS[BENCHMARK_MODE]

# =========================================================
# Timing
# =========================================================

def measure_runtime(func, repeats: int = 5) -> dict:
    func()  # Warm-up
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

        print(f"Qubits={num_qubits}  Gates={total_gates}")

        gate_sequence = generate_gate_sequence(num_qubits, total_gates, ACTIVE_GATE_SET, seed=SEED)

        qc_stats   = measure_runtime(build_qc(num_qubits,   gate_sequence), REPEATS)
        est_stats  = measure_runtime(build_est(num_qubits,  gate_sequence), REPEATS)
        estt_stats = measure_runtime(build_estt(num_qubits, gate_sequence), REPEATS)

        print(
            f"  qc={qc_stats['avg']:.5f}s  "
            f"est={est_stats['avg']:.5f}s  "
            f"estt={estt_stats['avg']:.5f}s"
        )

        results.append({
            "timestamp":      datetime.now().isoformat(),
            "gate_set":       GATE_SET_CHOICE,
            "benchmark_mode": BENCHMARK_MODE,
            "num_qubits":     num_qubits,
            "total_gates":    total_gates,
            **{f"qc_{k}":   v for k, v in qc_stats.items()},
            **{f"est_{k}":  v for k, v in est_stats.items()},
            **{f"estt_{k}": v for k, v in estt_stats.items()},
        })

# =========================================================
# CSV speichern
# =========================================================

df = pd.DataFrame(results)
df.to_csv(CSV_FILE, index=False)
print(f"\nErgebnisse gespeichert: {CSV_FILE}")

# =========================================================
# Plot  (3 Subplots: QC + Statevector | Estimator | Estimator transpiliert)
# =========================================================

def plot_results(df: pd.DataFrame, save_path: Path | None = None) -> None:
    qubit_vals = sorted(df["num_qubits"].unique())
    gate_set   = df["gate_set"].iloc[0]
    mode       = df["benchmark_mode"].iloc[0]
    mode_label = MODE_LABELS[mode]

    method_cfg = [
        ("QuantumCircuit + Statevector", "qc_avg",   "qc_std",   "tab:blue"),
        ("Estimator (kein Transpile)",   "est_avg",  "est_std",  "tab:orange"),
        ("Estimator (transpiliert)",     "estt_avg", "estt_std", "tab:green"),
    ]

    n_qubits = len(qubit_vals)
    fig, axes = plt.subplots(1, n_qubits, figsize=(6 * n_qubits, 5), squeeze=False)
    axes = axes[0]

    fig.suptitle(
        f"Qiskit – {mode_label}  |  Gate-Set: {gate_set!r}",
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
        ax.set_ylabel("Zeit (s)", fontsize=11)
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


PLOT_FILE = RESULT_DIR / f"qiskit_{GATE_SET_CHOICE}_{BENCHMARK_MODE}_V{version}.png"
plot_results(df, save_path=PLOT_FILE)
