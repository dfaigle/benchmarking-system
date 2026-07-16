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
GATE_SET_CHOICE = "non_clifford"

# =========================================================
# ➤  BENCHMARK-MODUS  ←  hier anpassen
# =========================================================
#
#  "creation"  → Zeit um den Circuit aufzubauen (kein Ausführen)
#                  Tape:         QuantumTape befüllen
#                  QNode:        construct_tape  (Tracing + Tape-Aufbau, kein Execute)
#
#  "execution" → Zeit der reinen Simulation nach dem Aufbau
#                  Tape:         qml.execute([tape], dev)
#                  QNode:        circuit()-Aufruf (nach Warm-up)
#
#  "gradient"  → Zeit der Gradienten-Berechnung via Parameter-Shift
#                  Tape:         qml.gradients.param_shift(tape) + execute
#                  QNode:        qml.grad(circuit)(params)
#
#  "all"       → alle drei Modi nacheinander (creation → execution → gradient);
#                eigene CSV + Plots pro Modus, gemeinsamer run_<N>-Ordner
#
BENCHMARK_MODE = "all"

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

VALID_MODES = {"creation", "execution", "gradient", "all"}
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

QUBIT_CONFIGS = [5,10,15]

GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(100000), 20)).astype(int)
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

# CSV-/Plot-Dateinamen entstehen pro Modus in run_benchmark() bzw. im Haupt-Ablauf.

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
# Device-Erstellung ist ausgelagert (gleiche Baseline).
#
#   Tape:   QuantumTape-Kontext befüllen (kein execute)
#   QNode:  qml.workflow.construct_tape → Tracing + Tape-Aufbau (kein execute)
#
# cache=True/False ist hier irrelevant: der Cache betrifft nur
# Ausführungsergebnisse (keyed auf Tape-Hash), nicht das Tracing —
# daher genügt im Creation-Modus eine einzige QNode-Linie.

def runner_creation_tape(num_qubits: int, gate_sequence: list):
    def run():
        with qml.tape.QuantumTape() as _tape:
            for gn, ws, ps in gate_sequence:
                apply_gate(gn, ws, ps)
            qml.probs(wires=range(num_qubits))

    return run


def runner_creation_qnode(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        return qml.probs(wires=range(num_qubits))

    def run():
        qml.workflow.construct_tape(circuit)()   # nur Tracing + Tape-Aufbau

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
# Misst: reine Ausführungszeit der Gradienten-Berechnung.
#
#   Trainierbare Schicht: die ersten TRAINABLE_RATIO der Gatter (20%) werden
#   durch trainierbare RY(params[k]) ERSETZT, zyklisch auf die Qubits verteilt
#   (wires = k % num_qubits). Die restlichen 80% bleiben fester Gate-Block.
#   Die Gesamt-Gatterzahl bleibt damit total_gates; die Zahl der Ableitungs-
#   richtungen (Parameter) wächst nun mit total_gates mit.
#   Ausgabe: Erwartungswert ⟨Z₀⟩
#
#   HINWEIS (Entartung): Sobald n_trainable > num_qubits, liegen mehrere RY auf
#   demselben Qubit direkt hintereinander und kollabieren mathematisch zu einem
#   RY(Summe) — die Parameter sind dann redundant und der Circuit ist über-
#   parametrisiert. Für die reine ZEITMESSUNG irrelevant, da der Simulator jedes
#   RY einzeln als Matrixmultiplikation ausführt (kein Auto-Merge).
#
# QNode-Linien nutzen diff_method="best" (auf default.qubit → Backprop) —
# identisch zum Executor der Abstraktionsschicht (pennylane_executor.py,
# QNode mit diff_method="best"). Dadurch messen Roh-Benchmark und
# Executor-Benchmark denselben Gradienten-Algorithmus, und die Differenz
# Executor − Roh ist der reine Abstraktions-Overhead.
#
# ACHTUNG: Die Tape-Linie bleibt Parameter-Shift (qml.gradients.param_shift),
# das 2 Tapes PRO trainierbarem Parameter erzeugt. Da die Parameterzahl jetzt
# mit total_gates mitwächst (20%), wird die Tape-Linie bei großen GATE_CONFIGS
# sehr teuer (100000 Gatter → 20000 Params → 40000 Tapes) und ist praktisch der
# limitierende Faktor. Adjoint-Differentiation (device.compute_derivatives) wäre
# die parameterzahl-unabhängige Alternative. Die Tape-Linie ist zudem NICHT
# direkt mit den QNode-Linien (Backprop) vergleichbar.
#
#   Tape:           qml.execute(grad_tapes, dev) — grad_tapes einmalig vorberechnet
#   QNode no cache: qml.grad(circuit)(params) — Tracing + Simulation bei jedem Aufruf
#   QNode cached:   qml.grad(circuit)(params) — nur Simulation (Graph gecacht)


# Anteil der Gatter, der im Gradient-Modus durch trainierbare RY ersetzt wird.
TRAINABLE_RATIO = 0.2


def split_trainable(gate_sequence: list):
    """Teilt die Sequenz in (n_trainable, fixed_gates).

    Die ersten TRAINABLE_RATIO der Gatter werden durch trainierbare RY ersetzt,
    der Rest bleibt fest. Es gilt
        n_trainable + len(fixed_gates) == len(gate_sequence),
    sodass die Gesamt-Gatterzahl total_gates erhalten bleibt.
    """
    n_trainable = max(1, round(TRAINABLE_RATIO * len(gate_sequence)))
    fixed_gates = gate_sequence[n_trainable:]
    return n_trainable, fixed_gates


def runner_gradient_tape(num_qubits: int, gate_sequence: list):
    dev                      = qml.device("default.qubit", wires=num_qubits)
    n_trainable, fixed_gates = split_trainable(gate_sequence)
    params                   = pnp.array(np.zeros(n_trainable), requires_grad=True)

    with qml.tape.QuantumTape() as tape:
        for k in range(n_trainable):
            qml.RY(params[k], wires=k % num_qubits)
        for gn, ws, ps in fixed_gates:
            apply_gate(gn, ws, ps)
        qml.expval(qml.PauliZ(0))

    # Nur nach den trainierbaren RY ableiten. Ein rohes QuantumTape setzt
    # trainable_params sonst auf ALLE Parameter (inkl. der festen Winkel der
    # fixed_gates) und ignoriert requires_grad — anders als das QNode-Interface.
    # Ohne diese Zeile würde die Tape-Linie nach mehr Parametern ableiten als die
    # QNode-Linien und wäre nicht mit ihnen vergleichbar.
    tape.trainable_params = list(range(n_trainable))

    grad_tapes, fn = qml.gradients.param_shift(tape)

    def run():
        results = qml.execute(grad_tapes, dev)
        return fn(results)

    return run


def runner_gradient_qnode_nc(num_qubits: int, gate_sequence: list):
    dev                      = qml.device("default.qubit", wires=num_qubits)
    n_trainable, fixed_gates = split_trainable(gate_sequence)

    @qml.qnode(dev, cache=False, diff_method="best")
    def circuit(params):
        for k in range(n_trainable):
            qml.RY(params[k], wires=k % num_qubits)
        for gn, ws, ps in fixed_gates:
            apply_gate(gn, ws, ps)
        return qml.expval(qml.PauliZ(0))

    grad_fn = qml.grad(circuit)
    params  = pnp.array(np.zeros(n_trainable), requires_grad=True)

    def run():
        grad_fn(params.copy())

    return run


def runner_gradient_qnode_c(num_qubits: int, gate_sequence: list):
    dev                      = qml.device("default.qubit", wires=num_qubits)
    n_trainable, fixed_gates = split_trainable(gate_sequence)

    @qml.qnode(dev, cache=True, diff_method="best")
    def circuit(params):
        for k in range(n_trainable):
            qml.RY(params[k], wires=k % num_qubits)
        for gn, ws, ps in fixed_gates:
            apply_gate(gn, ws, ps)
        return qml.expval(qml.PauliZ(0))

    grad_fn = qml.grad(circuit)
    params  = pnp.array(np.zeros(n_trainable), requires_grad=True)

    def run():
        grad_fn(params.copy())

    return run


# =========================================================
# Modus → Runner-Liste auflösen
# =========================================================
# Jeder Eintrag: (csv_prefix, plot_label, plot_farbe, builder)
# Creation hat nur zwei Linien (nc/c dort bedeutungslos, s.o.).

MODE_RUNNERS = {
    "creation": [
        ("tape",  "Tape",  "tab:blue",   runner_creation_tape),
        ("qnode", "QNode", "tab:orange", runner_creation_qnode),
    ],
    "execution": [
        ("tape", "Tape",             "tab:blue",   runner_execution_tape),
        ("qnc",  "QNode (no cache)", "tab:orange", runner_execution_qnode_nc),
        ("qc",   "QNode (cached)",   "tab:green",  runner_execution_qnode_c),
    ],
    "gradient": [
        ("tape", "Tape",             "tab:blue",   runner_gradient_tape),
        ("qnc",  "QNode (no cache)", "tab:orange", runner_gradient_qnode_nc),
        ("qc",   "QNode (cached)",   "tab:green",  runner_gradient_qnode_c),
    ],
}

MODE_LABELS = {
    "creation":  "Erstellungszeit",
    "execution": "Ausführungszeit",
    "gradient":  "Gradienten-Berechnungszeit",
}

# "all" → alle Modi in Definitionsreihenfolge, sonst nur der gewählte
MODES_TO_RUN = list(MODE_RUNNERS) if BENCHMARK_MODE == "all" else [BENCHMARK_MODE]

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
# Benchmark-Schleife (ein Modus)
# =========================================================

def run_benchmark(mode: str) -> pd.DataFrame:
    runners  = MODE_RUNNERS[mode]
    csv_file = RUN_DIR / f"benchmark_{GATE_SET_CHOICE}_{mode}.csv"

    results = []
    first_write = True   # Header nur beim allerersten Schreiben

    for num_qubits in QUBIT_CONFIGS:
        for total_gates in GATE_CONFIGS:

            print(f"Qubits={num_qubits}  Gates={total_gates}")

            gate_sequence = generate_gate_sequence(
                num_qubits, total_gates, ACTIVE_GATE_SET, seed=SEED
            )

            row = {
                "timestamp":      datetime.now().isoformat(),
                "gate_set":       GATE_SET_CHOICE,
                "benchmark_mode": mode,
                "num_qubits":     num_qubits,
                "total_gates":    total_gates,
            }

            log_parts = []
            for prefix, _label, _color, builder in runners:
                runner = builder(num_qubits, gate_sequence)
                stats  = measure_runtime(runner, REPEATS)
                mem    = measure_memory(runner, REPEATS)
                row.update({f"{prefix}_{k}":     v for k, v in stats.items()})
                row.update({f"{prefix}_mem_{k}": v for k, v in mem.items()})
                log_parts.append(f"{prefix}={stats['avg']:.5f}s/{mem['avg']:.1f}MiB")

            print("  " + "  ".join(log_parts))
            results.append(row)

            # Zwischenergebnis sofort anhängen → überlebt einen Absturz
            pd.DataFrame([row]).to_csv(csv_file, mode="a", header=first_write, index=False)
            first_write = False

    print(f"\nErgebnisse gespeichert (inkrementell während des Laufs): {csv_file}")
    return pd.DataFrame(results)

# =========================================================
# Plot  (ein Subplot pro Qubit-Zahl, eine Linie pro Abstraktion)
# =========================================================

def plot_metric(
    df: pd.DataFrame,
    runners: list,
    avg_suffix: str,
    std_suffix: str,
    ylabel: str,
    title: str,
    save_path: Path | None = None,
) -> None:
    qubit_vals = sorted(df["num_qubits"].unique())
    gate_set   = df["gate_set"].iloc[0]

    method_cfg = [
        (label, f"{prefix}_{avg_suffix}", f"{prefix}_{std_suffix}", color)
        for prefix, label, color, _builder in runners
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


# =========================================================
# Haupt-Ablauf: gewählte Modi nacheinander ausführen
# =========================================================
# CSV + Plots werden direkt nach jedem Modus gespeichert (absturzsicher),
# angezeigt werden alle Plot-Fenster gesammelt am Ende.

for mode in MODES_TO_RUN:
    print(f"\n========== Modus: {mode!r} ==========\n")

    df         = run_benchmark(mode)
    runners    = MODE_RUNNERS[mode]
    mode_label = MODE_LABELS[mode]
    stub       = f"benchmark_{GATE_SET_CHOICE}_{mode}"

    # --- Laufzeit ---
    plot_metric(
        df, runners, avg_suffix="avg", std_suffix="std",
        ylabel="Zeit (s)", title=mode_label,
        save_path=RUN_DIR / f"{stub}_time.png",
    )

    # --- Speicherverbrauch (Peak, tracemalloc) ---
    plot_metric(
        df, runners, avg_suffix="mem_avg", std_suffix="mem_std",
        ylabel="Peak-Speicher (MiB)", title=f"Speicherverbrauch (Peak) – {mode_label}",
        save_path=RUN_DIR / f"{stub}_mem.png",
    )

plt.show()