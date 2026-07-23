"""
Qiskit Benchmarking – analog zu benchmarkLog_V4.py (PennyLane)

Eine Linie (CSV-Präfix ``qc``):
  creation  → QuantumCircuit befüllen                    (kein Primitiv nötig)
  execution → StatevectorEstimator.run(circuit, ⟨Z₀⟩)
  gradient  → Parameter-Shift, ausgewertet über StatevectorEstimator

Warum execution/gradient über das Estimator-Primitiv laufen (und NICHT über
Statevector direkt): Der Executor der Abstraktionsschicht nutzt für Qiskit
intern OpTree + Qiskit-Estimator (s. QiskitExecutor._expectation_value bzw.
_expectation_value_derivatives — letzteres bricht ohne Estimator sogar mit
RuntimeError ab). Läge hier der rohe Statevector-Pfad, enthielte die Differenz
"Executor − Roh" zusätzlich Qiskits EIGENEN Primitiv-Overhead statt nur den
der Abstraktionsschicht — gemessen ~50 % des ausgewiesenen Overheads bei
kleinen und ~20 % bei großen Schaltkreisen.
Der Estimator ist zudem Qiskits idiomatisches Gegenstück zu PennyLanes QNode
(beide: Circuit + Observable → Erwartungswert). Damit sind Roh- und Executor-
Benchmark in BEIDEN Frameworks auf derselben Abstraktionsebene gepaart.
"""

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

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit import ParameterVector

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
#  "non_clifford_comparable"→ RX, RY, RZ, CRX, CRY, CRZ,
#                             ControlledPhaseShift  (ohne T/Rot/Toffoli —
#                             1:1 vergleichbar mit "non_clifford" des
#                             Executor-Benchmarks, s. Definition unten)
#  "clifford_plus_non_clifford"
#                           → beide obigen kombiniert (13 Gatter): H, S, CNOT,
#                             PauliX/Y/Z, RX, RY, RZ, CRX, CRY, CRZ,
#                             ControlledPhaseShift
#  "clifford_t"             → H, T, CNOT
#  "single_qubit_plus_cnot" → RX, RY, RZ, CNOT
#  "rot_cnot"               → Rot, CNOT
#
# ACHTUNG Vergleichbarkeit mit logarithmic_benchmark_abstraction.py:
# rng.choice zieht Indizes über die Gate-Liste. Nur wenn BEIDE Benchmarks
# eine Liste gleicher Länge und Reihenfolge nutzen, entsteht bei gleichem
# Seed dieselbe Gatter-Sequenz. Vergleichbare Sets: "clifford",
# "clifford_t", "single_qubit_plus_cnot", "non_clifford_comparable"
# (dort "non_clifford") und "clifford_plus_non_clifford" (gleicher Name dort).
# NICHT vergleichbar: "non_clifford" (10 statt 7 Gatter) und "rot_cnot".
#
GATE_SET_CHOICE = "clifford_plus_non_clifford"

# =========================================================
# ➤  BENCHMARK-MODUS  ←  hier anpassen
# =========================================================
#
#  "creation"  → Zeit um den Circuit aufzubauen
#                  qc:   QuantumCircuit aufbauen (kein Ausführen)
#
#  "execution" → Zeit der reinen Simulation nach dem Aufbau
#                  qc:   Statevector(circuit).expectation_value(⟨Z₀⟩) —
#                        Erwartungswert wie qml.expval(qml.PauliZ(0))
#
#  "gradient"  → Zeit der Gradienten-Berechnung via Parameter-Shift
#                  Trainierbare Schicht: TRAINABLE_RATIO der Gatter durch RY
#                  ersetzt (identisch zu PennyLane-/Executor-Benchmark)
#                  Ausgabe: ⟨Z₀⟩-Gradient für alle θₖ
#                  qc:   Statevector + manueller Param-Shift
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

# Spiegelt Position für Position das Set "non_clifford" des Executor-Benchmarks
# (["t","rx","ry","rz","crx","cry","crz","cp"]). Gleiche Länge + Reihenfolge
# UND pro Gattertyp gleich viele RNG-Aufrufe → identische Sequenz bei gleichem
# Seed. Bei Änderungen beide Listen synchron halten!
# Ohne "T" (7 statt 8 Gatter): T ist das einzige Gatter dieses Sets ohne Winkel
# und kann nie trainierbar werden — in qnode_vs_tape.py, wo die trainierbaren
# Winkel über den Circuit verstreut liegen statt in einem RY-Block, ließ es die
# Parameterzahl mit dem Seed schwanken. Hier mitgezogen, damit alle vier
# Benchmark-Dateien bei gleichem Seed denselben Schaltkreis erzeugen.
NON_CLIFFORD_COMPARABLE_GATES = [
    "RX", "RY", "RZ",
    "CRX", "CRY", "CRZ", "ControlledPhaseShift",
]

# Clifford + non-Clifford in EINEM Set (13 Gatter). Reihenfolge: erst Clifford,
# dann non-Clifford — MUSS in allen drei Benchmark-Dateien identisch sein, da
# rng.choice Indizes über die Liste zieht.
# Vergleichbar mit "clifford_plus_non_clifford" des Executor-Benchmarks: gleiche
# Länge, gleiche Reihenfolge, und pro Gattertyp gleich viele RNG-Aufrufe. Der
# kritische Fall ist CNOT/cx — beide Generatoren ziehen dafür KEINEN Winkel
# (hier: `params = [...] if gate != "CNOT" else []`), sonst würde der
# Zufallsstrom auseinanderlaufen.
# Hinweis: ~46 % der Gatter sind damit winkelfrei (vorher 0 %). n_trainable
# bleibt trotzdem seed-unabhängig, da trainable_layout nur von total_gates
# abhängt und apply_trainable das Gatter unabhängig von seinem Typ ersetzt.
CLIFFORD_PLUS_NON_CLIFFORD_GATES = CLIFFORD_GATES + NON_CLIFFORD_COMPARABLE_GATES

UNIVERSAL_GATE_SETS = {
    "clifford_t":             ["Hadamard", "T", "CNOT"],
    "single_qubit_plus_cnot": ["RX", "RY", "RZ", "CNOT"],
    "rot_cnot":               ["Rot", "CNOT"],
}


def resolve_gate_set(choice: str) -> list:
    mapping = {
        "clifford":                     CLIFFORD_GATES,
        "non_clifford":                 NON_CLIFFORD_GATES,
        "non_clifford_comparable":      NON_CLIFFORD_COMPARABLE_GATES,
        "clifford_plus_non_clifford":   CLIFFORD_PLUS_NON_CLIFFORD_GATES,
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

# Datei liegt in Benchmarks/qiskit/ → drei Ebenen hoch zur Projekt-Wurzel.
# (parent = qiskit/, parent.parent = Benchmarks/, parent.parent.parent = Wurzel)
RESULT_DIR = Path(__file__).parent.parent.parent / "Results" / "Qiskit"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED    = 42
REPEATS = 4

QUBIT_CONFIGS = [10]

GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(100000), 20)).astype(int)
)

# =========================================================
# Pro Lauf ein eigener Unterordner: Results/Qiskit/<timestamp>_qubits-<...>/
# =========================================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

qubit_string = "-".join(map(str, QUBIT_CONFIGS))

RUN_DIR = RESULT_DIR / f"{timestamp}_qubits-{qubit_string}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

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
# Misst: wie lange braucht die rohe Ebene, um den Circuit aufzubauen?
#
#   qc:   QuantumCircuit befüllen (kein execute)

def runner_creation_qc(num_qubits: int, gate_sequence: list):
    def run():
        qc = QuantumCircuit(num_qubits)
        for gn, ws, ps in gate_sequence:
            apply_gate(qc, gn, ws, ps)
    return run


# ------ EXECUTION ------
# Misst: wie lange dauert die Simulation nach dem Aufbau?
#
#   qc:   StatevectorEstimator.run([(circuit, ⟨Z₀⟩)]) — Erwartungswert über
#         dasselbe Primitiv, das der Executor intern nutzt (s. Modul-Docstring).
#         Gleiche Aufgabe wie qml.expval(qml.PauliZ(0)) im PennyLane-Roh- und
#         ex.expectation_value im Executor-Benchmark.
#         Circuit, Observable und Estimator im Setup gebaut (ungemessen), nur
#         der Auswertungs-Aufruf liegt im Messfenster.

def runner_execution_qc(num_qubits: int, gate_sequence: list):
    observable = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    estimator  = StatevectorEstimator()
    qc = QuantumCircuit(num_qubits)
    for gn, ws, ps in gate_sequence:
        apply_gate(qc, gn, ws, ps)

    def run():
        estimator.run([(qc, observable)]).result()

    return run


# ------ GRADIENT ------
# Misst: reine Ausführungszeit der Gradienten-Berechnung via Parameter-Shift.
#
#   Trainierbare Schicht: jedes step-te Gatter (step = round(1/TRAINABLE_RATIO))
#   wird durch ein trainierbares RY(θₖ) ERSETZT — die trainierbaren Parameter
#   liegen gleichmäßig über den GANZEN Circuit verstreut (kein RY-Block am
#   Anfang). Das RY sitzt auf dem Wire des ersetzten Gatters (ws[0]); die
#   übrigen Gatter bleiben fest. Die Gesamt-Gatterzahl bleibt total_gates —
#   identisch zum PennyLane-Roh- und zum Executor-Benchmark (dort: gleiche
#   Ersetzung, Executor-Qiskit rechnet ebenfalls Parameter-Shift via OpTree →
#   gleicher Algorithmus).
#   grad_k = [f(θₖ + π/2) − f(θₖ − π/2)] / 2
#
# ACHTUNG Kosten: Parameter-Shift braucht 2 Ausführungen PRO Parameter und
# PRO Messaufruf. Mit n_trainable = TRAINABLE_RATIO·total_gates wächst der
# Aufwand je Aufruf ~O(total_gates²) und wird bei großen GATE_CONFIGS sehr
# teuer — das gilt für Roh- UND Executor-Benchmark gleichermaßen und ist
# daher fair, aber laufzeitintensiv.
#
#   qc:   manueller Param-Shift, jede Auswertung über StatevectorEstimator
#         (Parameter-Binding im Primitiv) — dieselbe Ebene wie der Executor,
#         der intern ebenfalls Estimator-basierten Param-Shift rechnet.

# Anteil der Gatter, der im Gradient-Modus durch trainierbare RY ersetzt wird.
# MUSS mit TRAINABLE_RATIO in logarithmic_benchmark_pennylane.py und
# logarithmic_benchmark_abstraction.py synchron sein.
TRAINABLE_RATIO = 0.3


def trainable_layout(n_gates: int):
    """Verteilung der trainierbaren RY über den GANZEN Circuit (kein RY-Block) —
    identisch zu den anderen Benchmarks.

    Jedes ``step``-te Gatter (step = round(1/TRAINABLE_RATIO)) wird durch ein
    trainierbares RY ERSETZT; die übrigen Gatter bleiben fest. n_trainable ist
    deterministisch (hängt nur von total_gates ab, nicht vom Seed); die Gesamt-
    Gatterzahl bleibt total_gates.
    """
    step        = max(1, round(1 / TRAINABLE_RATIO))
    n_trainable = len(range(0, n_gates, step))
    return n_trainable, step


# Rotationsachsen, die an den trainierbaren Positionen zyklisch (über den Zähler
# k) eingesetzt werden. Alle drei sind 1-Parameter-Gatter → die Zähl-Logik
# (params[k], n_trainable) bleibt unverändert korrekt. Auf ein Set beschränken
# (z. B. ["RY"]) genügt, um nur eine Achse zu nutzen. MUSS mit TRAINABLE_GATES
# der anderen Benchmarks synchron sein.
TRAINABLE_GATES = ["RX", "RY", "RZ"]


def apply_trainable(qc: QuantumCircuit, k: int, param, wire: int) -> None:
    gate = TRAINABLE_GATES[k % len(TRAINABLE_GATES)]
    if   gate == "RX": qc.rx(param, wire)
    elif gate == "RY": qc.ry(param, wire)
    else:              qc.rz(param, wire)


def _build_trainable_circuit(num_qubits: int, gate_sequence: list):
    """Trainierbarer Circuit wie in PennyLane-/Executor-Benchmark: jedes step-te
    Gatter durch eine trainierbare Rotation RX/RY/RZ(θₖ) — zyklisch über k — auf
    wire ws[0] ersetzt, verstreut über den ganzen Circuit; die übrigen Gatter
    bleiben fest."""
    n_trainable, step = trainable_layout(len(gate_sequence))
    pv = ParameterVector("θ", length=n_trainable)
    qc = QuantumCircuit(num_qubits)
    k = 0
    for i, (gn, ws, ps) in enumerate(gate_sequence):
        if i % step == 0:
            apply_trainable(qc, k, pv[k], ws[0])   # RX / RY / RZ im Wechsel
            k += 1
        else:
            apply_gate(qc, gn, ws, ps)
    return qc, pv, n_trainable


def runner_gradient_qc(num_qubits: int, gate_sequence: list):
    observable = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=num_qubits)
    estimator  = StatevectorEstimator()
    qc, _pv, n_trainable = _build_trainable_circuit(num_qubits, gate_sequence)
    param_values = np.zeros(n_trainable)

    # Die Parameter werden IM Primitiv gebunden (dritter Eintrag des PUB) statt
    # vorher per assign_parameters — genauso macht es der Executor. Bewusst
    # NICHT gebatcht (alle Shifts in einem run()-Aufruf): der Executor wertet
    # ebenfalls sequenziell aus, und gemessen bringt Batching hier nur ~6 %.
    def run():
        for i in range(n_trainable):
            plus_vals  = param_values.copy(); plus_vals[i]  += np.pi / 2
            minus_vals = param_values.copy(); minus_vals[i] -= np.pi / 2
            ev_p = float(estimator.run([(qc, observable, plus_vals)]).result()[0].data.evs)
            ev_m = float(estimator.run([(qc, observable, minus_vals)]).result()[0].data.evs)
            _ = (ev_p - ev_m) / 2

    return run

# =========================================================
# Modus → Runner-Liste auflösen
# =========================================================
# Jeder Eintrag: (csv_prefix, plot_label, plot_farbe, builder)
#
# Eine Linie. Sie ist der Paar-Partner zum Executor-Benchmark
# (BACKEND_CHOICE="qiskit"): creation baut nur den Circuit (Executor:
# transpile_circuit, ebenfalls ohne Ausführung), execution und gradient laufen
# über dasselbe Estimator-Primitiv, das der Executor intern nutzt. Damit misst
# "Executor − Roh" in allen drei Modi ausschließlich die Abstraktionsschicht.
# Die frühere estt-Linie (vortranspiliert) ist entfernt: der Executor
# transpiliert bei lokalem Backend nicht (s. _convert_to_optree).
#
# Der CSV-Präfix bleibt "qc" — kompatibel zu bestehenden Läufen und zur
# Framework-Erkennung in plot_from_csv.py.

MODE_RUNNERS = {
    "creation": [
        ("qc", "Qiskit (QuantumCircuit)", "tab:blue", runner_creation_qc),
    ],
    "execution": [
        ("qc", "Qiskit (Estimator)", "tab:blue", runner_execution_qc),
    ],
    "gradient": [
        ("qc", "Qiskit (Estimator)", "tab:blue", runner_gradient_qc),
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
# Speicher-Messung (Peak via tracemalloc)
# =========================================================
# Misst den Peak der Python-Allokationen während func() läuft.
# Reproduzierbar und erfasst die hier dominante Last (Circuit-/Gate-Objekte).
# NumPy-C-Buffer werden bewusst nicht erfasst (bei 10-15 Qubits vernachlässigbar).
# Getrennt vom Timing, da tracemalloc die Laufzeit verfälschen würde.

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
    csv_file = RUN_DIR / f"qiskit_{GATE_SET_CHOICE}_{mode}.csv"

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
                    seq = generate_gate_sequence(num_qubits, total_gates, ACTIVE_GATE_SET, seed=seed)
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
        f"Qiskit – {title}  |  Gate-Set: {gate_set!r}",
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
    stub       = f"qiskit_{GATE_SET_CHOICE}_{mode}"

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
