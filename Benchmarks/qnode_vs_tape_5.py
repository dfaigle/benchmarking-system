"""Fairer Vergleich: rohes QuantumTape vs. QNode-Interface (nur PennyLane).

Zweck (Vergleichspaar 1 der Thesis): Was kostet das QNode-Interface gegenüber
dem direkten Arbeiten mit QuantumTapes — bei GLEICHER Arbeit pro Messaufruf?

Fairness-Prinzip: Beide Linien leisten in jedem Modus exakt dieselbe Arbeit
innerhalb des Messfensters. Nichts wird für eine Seite vorgebaut, was die
andere Seite pro Aufruf neu erledigen muss.

    creation  → beide bauen NUR (kein Execute):
                  Tape:  QuantumTape-Kontext befüllen
                  QNode: qml.workflow.construct_tape → Tracing + Tape-Aufbau
    execution → beide bauen + führen aus (pro Aufruf):
                  Tape:  QuantumTape IM run() befüllen + qml.execute([tape], dev)
                  QNode: circuit()-Aufruf (Tracing + Simulation), cache=False
                  Rückgabe beidseitig: qml.expval(Z₀)
    gradient  → ⟨Z₀⟩-Gradient über dieselbe trainierbare Schicht
                (TRAINABLE_RATIO der Gatter durch RY ersetzt), DREI Linien:
                  tape :  Tape IM run() bauen + param_shift + execute
                  qps  :  qml.grad, diff_method="parameter-shift" — gleicher
                          Algorithmus wie tape → das FAIRE Paar
                  qbest:  qml.grad, diff_method="best" (→ Backprop) —
                          Zusatzlinie: zeigt den Algorithmus-Vorteil von
                          Backprop; NICHT Teil des fairen Tape-Paars

Die Differenz qps − tape ist damit der reine Overhead des QNode-Interfaces
(Workflow-Maschinerie, Interface-Auflösung, Ergebnis-Postprocessing) bei
identischer Bau-, Simulations- und Algorithmus-Last. qbest − qps zeigt
separat, wie viel der bessere Algorithmus (Backprop) bei identischem
Interface einspart.

Bewusste Design-Entscheidungen (Abweichungen von logarithmic_benchmark_pennylane.py):

* Keine cached-QNode-Linien (überall cache=False): Der PennyLane-Cache
  betrifft nur Ausführungsergebnisse, vermeidet nie das Tracing und kostet
  nur Hash-Overhead — eine cached-Linie hätte hier keinen Informationswert
  (im Hauptbenchmark wird genau das separat gezeigt) und würde das faire
  Paar verzerren, da die Tape-Linie keinen Cache-Apparat trägt.
* „Nur ausführen" ist für den QNode prinzipiell nicht messbar (Tracing ist
  bei jedem Aufruf unvermeidbar). Die Referenz „vorgebautes Tape, nur
  Ausführung" existiert weiterhin im Hauptbenchmark (execution/Tape-Linie).
* Gradient nutzt dieselbe TRAINABLE_RATIO-Ersetzung wie die Haupt-Benchmarks
  (Wert MUSS mit logarithmic_benchmark_pennylane.py synchron sein).
  ACHTUNG Kosten: Parameter-Shift braucht 2 Ausführungen PRO Parameter und
  PRO Messaufruf — mit n_trainable = TRAINABLE_RATIO·total_gates wächst der
  Aufwand ~O(total_gates²) je Aufruf; die oberen GATE_CONFIGS sind für die
  tape-/qps-Linien damit extrem teuer. Nur qbest (Backprop: ein Vor-/
  Rückwärtslauf) bleibt davon unberührt.
* Gleiche Sequenz-Generierung/Seeds wie der Roh-Benchmark: bei gleichem
  GATE_SET_CHOICE entstehen exakt dieselben Schaltkreise wie dort.
"""

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
#  "non_clifford_comparable"→ T, RX, RY, RZ, CRX, CRY, CRZ,
#                             ControlledPhaseShift (ohne Rot/Toffoli)
#  "clifford_t"             → H, T, CNOT
#  "single_qubit_plus_cnot" → RX, RY, RZ, CNOT
#  "rot_cnot"               → Rot, CNOT
#
# Gleiche Namen/Listen wie logarithmic_benchmark_pennylane.py → bei gleichem
# Seed entstehen identische Schaltkreise wie im Roh-Benchmark.
#
GATE_SET_CHOICE = "non_clifford_comparable"

# =========================================================
# ➤  BENCHMARK-MODUS  ←  hier anpassen
# =========================================================
#
#  "creation"  → beide Linien: nur bauen (kein Execute)
#  "execution" → beide Linien: bauen + ausführen pro Aufruf
#  "gradient"  → beide Linien: Parameter-Shift-Gradient pro Aufruf
#  "all"       → alle drei Modi nacheinander; eigene CSV + Plots pro Modus
#
BENCHMARK_MODE = "all"

# Plots am Ende interaktiv anzeigen? Für Batch-/Headless-Läufe auf False setzen.
SHOW_PLOTS = True

# =========================================================
# Gate-Definitionen  (identisch zum Roh-Benchmark)
# =========================================================

CLIFFORD_GATES = ["Hadamard", "S", "CNOT", "PauliX", "PauliY", "PauliZ"]

NON_CLIFFORD_GATES = [
    "T", "RX", "RY", "RZ", "Rot",
    "CRX", "CRY", "CRZ", "ControlledPhaseShift", "Toffoli",
]

NON_CLIFFORD_COMPARABLE_GATES = [
    "T", "RX", "RY", "RZ",
    "CRX", "CRY", "CRZ", "ControlledPhaseShift",
]

UNIVERSAL_GATE_SETS = {
    "clifford_t":              ["Hadamard", "T", "CNOT"],
    "single_qubit_plus_cnot":  ["RX", "RY", "RZ", "CNOT"],
    "rot_cnot":                ["Rot", "CNOT"],
}


def resolve_gate_set(choice: str) -> list:
    mapping = {
        "clifford":                CLIFFORD_GATES,
        "non_clifford":            NON_CLIFFORD_GATES,
        "non_clifford_comparable": NON_CLIFFORD_COMPARABLE_GATES,
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
# Konfiguration  (identisch zum Roh-Benchmark)
# =========================================================

RESULT_DIR = Path(__file__).parent.parent / "Results" / "TapeVsQNode"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED    = 42
REPEATS = 4

QUBIT_CONFIGS = [5]

GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(10000), 20)).astype(int)
)

# =========================================================
# Pro Lauf ein eigener Unterordner: Results/TapeVsQNode/run_<N>/
# =========================================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

qubit_string = "-".join(map(str, QUBIT_CONFIGS))

RUN_DIR = RESULT_DIR / f"{timestamp}_qubits-{qubit_string}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
# =========================================================
# Gate-Eigenschaften  (identisch zum Roh-Benchmark)
# =========================================================

_ANGLE_1  = {"RX", "RY", "RZ", "CRX", "CRY", "CRZ", "ControlledPhaseShift"}
_ANGLE_3  = {"Rot"}
_NEEDS_2Q = {"CNOT", "CRX", "CRY", "CRZ", "ControlledPhaseShift"}
_NEEDS_3Q = {"Toffoli"}

# =========================================================
# Gate-Sequenz generieren  (identische Logik/Seed wie Roh-Benchmark)
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
# Builder laufen UNGEMESSEN (Setup: Device, params); nur run() wird getimt.
# Fairness: beide Linien leisten in run() dieselbe Arbeit — siehe Docstring.

# ------ CREATION ------
# Beide: nur bauen, kein Execute. Kein Device im Messfenster.

def runner_creation_tape(num_qubits: int, gate_sequence: list):
    def run():
        with qml.tape.QuantumTape() as _tape:
            for gn, ws, ps in gate_sequence:
                apply_gate(gn, ws, ps)
            qml.probs(wires=range(num_qubits))

    return run


def runner_creation_qnode(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=False)
    def circuit():
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        return qml.probs(wires=range(num_qubits))

    def run():
        qml.workflow.construct_tape(circuit)()   # nur Tracing + Tape-Aufbau

    return run


# ------ EXECUTION ------
# Beide: bauen + ausführen PRO AUFRUF. Das Tape wird bewusst IM Messfenster
# neu befüllt — nur so trägt es dieselbe Aufbau-Last wie das QNode-Tracing.

def runner_execution_tape(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    def run():
        with qml.tape.QuantumTape() as tape:     # Bauen — im Messfenster
            for gn, ws, ps in gate_sequence:
                apply_gate(gn, ws, ps)
            qml.expval(qml.PauliZ(0))
        qml.execute([tape], dev)                  # Ausführen

    return run

def runner_execution_qnode(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=False)
    def circuit():
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        return qml.expval(qml.PauliZ(0))

    return circuit                                # Tracing + Simulation pro Aufruf


# ------ GRADIENT ------
# ⟨Z₀⟩-Gradient über die trainierbare Schicht (jedes step-te Gatter durch ein
# trainierbares RY auf dem Wire des ersetzten Gatters ersetzt, verstreut über
# den ganzen Circuit — kein RY-Block am Anfang), komplett PRO AUFRUF (Bauen/
# Tracing → Ableiten → Ausführen → Zusammensetzen). Drei Linien, alle über
# DENSELBEN Circuit:
#
#   tape :  Tape bauen + param_shift + execute      — Parameter-Shift
#   qps  :  qml.grad, diff_method="parameter-shift" — gleicher Algorithmus wie
#           tape → das FAIRE Paar (qps − tape = QNode-Interface-Overhead)
#   qbest:  qml.grad, diff_method="best" (Backprop) — Zusatzlinie: zeigt den
#           Algorithmus-Vorteil von Backprop; NICHT mit tape vergleichbar
#
# ACHTUNG Kosten: Param-Shift = 2 Ausführungen pro Parameter pro Aufruf; mit
# n_trainable = TRAINABLE_RATIO·total_gates ist das ~O(total_gates²) je
# Messaufruf — obere GATE_CONFIGS für tape/qps sehr teuer. qbest (Backprop:
# ein Vor-/Rückwärtslauf) ist davon nicht betroffen.
#
# Hinweis: qml.grad liefert zusätzlich zum Gradienten den Funktionswert-Pfad
# der Autograd-Maschinerie — dieser Mehraufwand ist Teil des QNode/qml.grad-
# Interfaces und damit bewusst Teil des gemessenen Overheads.

# Anteil der Gatter, der im Gradient-Modus durch trainierbare RY ersetzt wird.
# MUSS mit TRAINABLE_RATIO in logarithmic_benchmark_pennylane.py (und den
# anderen Benchmarks) synchron sein.
TRAINABLE_RATIO = 0.3


def trainable_layout(n_gates: int):
    """Verteilung der trainierbaren RY über den GANZEN Circuit (kein RY-Block) —
    identisch zum Roh-Benchmark.

    Jedes ``step``-te Gatter (step = round(1/TRAINABLE_RATIO)) wird durch ein
    trainierbares RY ERSETZT; die übrigen Gatter bleiben fest. n_trainable ist
    deterministisch (hängt nur von total_gates ab, nicht vom Seed); die Gesamt-
    Gatterzahl bleibt total_gates.
    """
    step        = max(1, round(1 / TRAINABLE_RATIO))
    n_trainable = len(range(0, n_gates, step))
    return n_trainable, step


def runner_gradient_tape(num_qubits: int, gate_sequence: list):
    dev               = qml.device("default.qubit", wires=num_qubits)
    n_trainable, step = trainable_layout(len(gate_sequence))
    params            = pnp.array(np.zeros(n_trainable), requires_grad=True)

    def run():
        # Tape-Parameter-Indizes der RY beim Bauen mitzählen: da die RY zwischen
        # festen Winkelgattern verstreut liegen, sind ihre Indizes NICHT mehr
        # 0..n_trainable-1. param_counter zählt alle ins Tape eingebrachten
        # Parameter (RY: 1, festes Gatter: len(ps)).
        trainable_param_idx = []
        param_counter       = 0
        k                   = 0
        with qml.tape.QuantumTape() as tape:      # Bauen — im Messfenster
            for i, (gn, ws, ps) in enumerate(gate_sequence):
                if i % step == 0:
                    qml.RY(params[k], wires=ws[0])
                    trainable_param_idx.append(param_counter)
                    param_counter += 1
                    k += 1
                else:
                    apply_gate(gn, ws, ps)
                    param_counter += len(ps)
            qml.expval(qml.PauliZ(0))
        # Nur nach den RY ableiten (rohe Tapes markieren sonst ALLE Winkel
        # als trainierbar — anders als das QNode-Interface).
        tape.trainable_params = trainable_param_idx
        grad_tapes, fn = qml.gradients.param_shift(tape)
        fn(qml.execute(grad_tapes, dev))          # Ausführen + Zusammensetzen

    return run


def runner_gradient_qnode_ps(num_qubits: int, gate_sequence: list):
    dev               = qml.device("default.qubit", wires=num_qubits)
    n_trainable, step = trainable_layout(len(gate_sequence))

    @qml.qnode(dev, cache=False, diff_method="parameter-shift")
    def circuit(params):
        k = 0
        for i, (gn, ws, ps) in enumerate(gate_sequence):
            if i % step == 0:
                qml.RY(params[k], wires=ws[0])
                k += 1
            else:
                apply_gate(gn, ws, ps)
        return qml.expval(qml.PauliZ(0))

    grad_fn = qml.grad(circuit)
    params  = pnp.array(np.zeros(n_trainable), requires_grad=True)

    def run():
        grad_fn(params.copy())

    return run


def runner_gradient_qnode_best(num_qubits: int, gate_sequence: list):
    dev               = qml.device("default.qubit", wires=num_qubits)
    n_trainable, step = trainable_layout(len(gate_sequence))

    @qml.qnode(dev, cache=False, diff_method="best")
    def circuit(params):
        k = 0
        for i, (gn, ws, ps) in enumerate(gate_sequence):
            if i % step == 0:
                qml.RY(params[k], wires=ws[0])
                k += 1
            else:
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

MODE_RUNNERS = {
    "creation": [
        ("tape",  "Tape",  "tab:blue",   runner_creation_tape),
        ("qnode", "QNode", "tab:orange", runner_creation_qnode),
    ],
    "execution": [
        ("tape",  "Tape",  "tab:blue",   runner_execution_tape),
        ("qnode", "QNode", "tab:orange", runner_execution_qnode),
    ],
    "gradient": [
        ("tape",  "Tape (param-shift)",       "tab:blue",   runner_gradient_tape),
        ("qps",   "QNode (param-shift)",      "tab:orange", runner_gradient_qnode_ps),
        ("qbest", "QNode (best → Backprop)",  "tab:green",  runner_gradient_qnode_best),
    ],
}

MODE_LABELS = {
    "creation":  "Erstellungszeit (nur bauen)",
    "execution": "Bauen + Ausführen",
    "gradient":  "Gradient (Parameter-Shift, bauen + rechnen)",
}

# "all" → alle Modi in Definitionsreihenfolge, sonst nur der gewählte
MODES_TO_RUN = list(MODE_RUNNERS) if BENCHMARK_MODE == "all" else [BENCHMARK_MODE]

# =========================================================
# Timing  (identisch zum Roh-Benchmark)
# =========================================================

def measure_runtime(runner_factory, repeats: int = 5) -> dict:
    # Jede Wiederholung bekommt einen eigenen Seed (SEED + r) → ein anderer
    # Zufalls-Circuit pro Wiederholung. Der Aufbau (runner_factory) läuft
    # ungemessen, danach ein ungemessener Warm-up-Aufruf, dann der Zeit-Messlauf.
    times = []
    for r in range(repeats):
        runner = runner_factory(SEED + r)
        runner()                              # Warm-up (nicht gemessen)
        gc.collect()
        t0 = time.perf_counter()
        runner()
        times.append(time.perf_counter() - t0)
    return {
        "avg": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }

# =========================================================
# Speicher-Messung (Peak via tracemalloc, identisch zum Roh-Benchmark)
# =========================================================

def measure_memory(runner_factory, repeats: int = 5) -> dict:
    # Wie measure_runtime: eigener Seed (SEED + r) pro Wiederholung, Warm-up
    # (lazy Imports/Caches nicht mitmessen) ungemessen vor der Peak-Messung.
    peaks = []
    for r in range(repeats):
        runner = runner_factory(SEED + r)
        runner()                              # Warm-up (nicht gemessen)
        gc.collect()
        tracemalloc.start()
        runner()
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
    csv_file = RUN_DIR / f"tape_vs_qnode_{GATE_SET_CHOICE}_{mode}.csv"

    results = []
    first_write = True   # Header nur beim allerersten Schreiben

    for num_qubits in QUBIT_CONFIGS:
        for total_gates in GATE_CONFIGS:

            print(f"Qubits={num_qubits}  Gates={total_gates}")

            row = {
                "timestamp":      datetime.now().isoformat(),
                "gate_set":       GATE_SET_CHOICE,
                "benchmark_mode": mode,
                "num_qubits":     num_qubits,
                "total_gates":    total_gates,
            }

            log_parts = []
            for prefix, _label, _color, builder in runners:
                # Fabrik: erzeugt pro Seed eine frische Gatter-Sequenz und den
                # dazugehörigen Runner. Die Mess-Funktionen rufen sie mit
                # SEED + r auf → anderer Zufalls-Circuit pro Wiederholung.
                def make_runner(seed, _builder=builder):
                    seq = generate_gate_sequence(
                        num_qubits, total_gates, ACTIVE_GATE_SET, seed=seed
                    )
                    return _builder(num_qubits, seq)

                stats = measure_runtime(make_runner, REPEATS)
                mem   = measure_memory(make_runner, REPEATS)
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
# Plot  (ein Subplot pro Qubit-Zahl, zwei Linien: Tape vs. QNode)
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
        f"Tape vs. QNode – {title}  |  Gate-Set: {gate_set!r}  "
        f"({', '.join(ACTIVE_GATE_SET)})",
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

    if not SHOW_PLOTS:
        plt.close(fig)


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
    stub       = f"tape_vs_qnode_{GATE_SET_CHOICE}_{mode}"

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

if SHOW_PLOTS:
    plt.show()
