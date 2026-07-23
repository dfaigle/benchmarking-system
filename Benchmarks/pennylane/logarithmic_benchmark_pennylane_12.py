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
#  "creation"  → Zeit um den Circuit aufzubauen (kein Ausführen)
#                  QNode:        construct_tape  (Tracing + Tape-Aufbau, kein Execute)
#
#  "execution" → Zeit der reinen Simulation nach dem Aufbau
#                  QNode:        circuit()-Aufruf (nach Warm-up)
#
#  "gradient"  → Zeit der Gradienten-Berechnung (Backprop, diff_method="best")
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
    "clifford_t":              ["Hadamard", "T", "CNOT"],
    "single_qubit_plus_cnot":  ["RX", "RY", "RZ", "CNOT"],
    "rot_cnot":                ["Rot", "CNOT"],
}

# =========================================================
# Auflösung der Auswahl
# =========================================================

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

# Datei liegt in Benchmarks/pennylane/ → drei Ebenen hoch zur Projekt-Wurzel.
# (parent = pennylane/, parent.parent = Benchmarks/, parent.parent.parent = Wurzel)
RESULT_DIR = Path(__file__).parent.parent.parent / "Results" / "Pennylane"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED    = 42
REPEATS = 4

QUBIT_CONFIGS = [12]
#QUBIT_CONFIGS = [3]   # Schnelltest

GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(100000), 20)).astype(int)
)
# [    10     16     26     43     70    113    183    298    483
#      785   1274   2069   3360   5456   8859  14384  23357  37927
#      61585 100000]

# a_n = 10 * (100000 / 10)^(n / 19)

# =========================================================
# Pro Lauf ein eigener Unterordner: Results/Pennylane/<timestamp>_qubits-<...>/
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
# Misst: wie lange braucht der QNode, um den Circuit aufzubauen?
# Device-Erstellung ist ausgelagert (gleiche Baseline).
#
#   QNode: qml.workflow.construct_tape → Tracing + Tape-Aufbau (kein execute)
#
# Es gibt bewusst nur EINE QNode-Linie (cache=False, s. Kommentar bei den
# Runner-Listen): der PennyLane-Cache betrifft nur Ausführungsergebnisse
# (keyed auf Tape-Hash) und kann hier — wo gar nicht ausgeführt wird —
# ohnehin nichts bewirken.

def runner_creation_qnode(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=False)
    def circuit():
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        return qml.state()

    def run():
        qml.workflow.construct_tape(circuit)()   # nur Tracing + Tape-Aufbau

    return run


# ------ EXECUTION ------
# Misst: wie lange dauert die Simulation nach dem Aufbau?
# Rückgabe ist qml.expval(⟨Z₀⟩) — dieselbe Aufgabe wie im Executor-Benchmark.
#
#   QNode: circuit()-Aufruf mit Tracing + Simulation bei jedem Aufruf (cache=False)

def runner_execution_qnode(num_qubits: int, gate_sequence: list):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=False)
    def circuit():
        for gn, ws, ps in gate_sequence:
            apply_gate(gn, ws, ps)
        return qml.expval(qml.PauliZ(0))

    return circuit


# ------ GRADIENT ------
# Misst: reine Ausführungszeit der Gradienten-Berechnung.
#
#   Trainierbare Schicht: jedes step-te Gatter (step = round(1/TRAINABLE_RATIO))
#   wird durch ein trainierbares RY(params[k]) ERSETZT — die trainierbaren
#   Parameter liegen damit GLEICHMÄSSIG über den gesamten Circuit verstreut
#   (kein RY-Block am Anfang mehr). Das RY sitzt auf dem Wire des ersetzten
#   Gatters (ws[0]). Die übrigen Gatter bleiben fest.
#   Die Gesamt-Gatterzahl bleibt damit total_gates; die Zahl der Ableitungs-
#   richtungen (Parameter) wächst mit total_gates mit.
#   Ausgabe: Erwartungswert ⟨Z₀⟩
#
# Die QNode-Linie nutzt diff_method="best" (auf default.qubit → Backprop) —
# identisch zum Executor der Abstraktionsschicht (pennylane_executor.py,
# QNode mit diff_method="best"). Dadurch messen Roh-Benchmark und
# Executor-Benchmark denselben Gradienten-Algorithmus, und die Differenz
# Executor − Roh ist der reine Abstraktions-Overhead.
# VORAUSSETZUNG dafür ist, dass beide Benchmarks denselben Circuit bauen:
# logarithmic_benchmark_abstraction.py nutzt dieselbe TRAINABLE_RATIO-Ersetzung
# (split_trainable/build_abstract_trainable dort) — die beiden TRAINABLE_RATIO-
# Werte müssen synchron bleiben.
#
#   QNode: qml.grad(circuit)(params) — Tracing + Simulation bei jedem Aufruf


# Anteil der Gatter, der im Gradient-Modus durch trainierbare RY ersetzt wird.
# MUSS mit TRAINABLE_RATIO in logarithmic_benchmark_abstraction.py synchron sein.
TRAINABLE_RATIO = 0.3


def trainable_layout(n_gates: int):
    """Verteilung der trainierbaren RY über den GANZEN Circuit (kein RY-Block).

    Jedes ``step``-te Gatter (step = round(1/TRAINABLE_RATIO)) wird durch ein
    trainierbares RY ERSETZT; die übrigen Gatter bleiben fest. Die RY liegen
    damit gleichmäßig über den gesamten Schaltkreis verstreut statt in einem
    Block am Anfang. n_trainable ist deterministisch (hängt nur von total_gates
    ab, nicht vom Seed); die Gesamt-Gatterzahl bleibt total_gates.
    """
    step        = max(1, round(1 / TRAINABLE_RATIO))
    n_trainable = len(range(0, n_gates, step))
    return n_trainable, step


# Rotationsachsen, die an den trainierbaren Positionen zyklisch (über den Zähler
# k) eingesetzt werden. Alle drei sind 1-Parameter-Gatter → die Zähl-Logik
# (params[k]) bleibt unverändert korrekt. Auf ein Set beschränken (z. B. ["RY"])
# genügt, um nur eine Achse zu nutzen. MUSS mit TRAINABLE_GATES der anderen
# Benchmarks synchron sein.
TRAINABLE_GATES = ["RX", "RY", "RZ"]


def apply_trainable(k: int, param, wire: int) -> None:
    gate = TRAINABLE_GATES[k % len(TRAINABLE_GATES)]
    if   gate == "RX": qml.RX(param, wires=wire)
    elif gate == "RY": qml.RY(param, wires=wire)
    else:              qml.RZ(param, wires=wire)


def runner_gradient_qnode(num_qubits: int, gate_sequence: list):
    dev               = qml.device("default.qubit", wires=num_qubits)
    n_trainable, step = trainable_layout(len(gate_sequence))

    @qml.qnode(dev, cache=False, diff_method="best")
    def circuit(params):
        k = 0
        for i, (gn, ws, ps) in enumerate(gate_sequence):
            if i % step == 0:
                apply_trainable(k, params[k], ws[0])   # RX / RY / RZ im Wechsel
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
#
# Nur EINE Linie: der QNode (cache=False). Die frühere cached-QNode-Linie
# betraf nur Ausführungsergebnisse (keyed auf Tape-Hash), vermied nie das
# Tracing und war durch den Hash-Overhead sogar ~10-20 % langsamer — ohne
# Informationswert, daher entfernt. Die frühere Tape-Linie ist ebenfalls
# entfernt: der faire Tape-vs-QNode-Vergleich lebt eigenständig in
# qnode_vs_tape.py.
#
# Der CSV-Präfix der QNode-Linie bleibt bewusst "qnc" (no cache): So bleiben
# die Spaltennamen zu den älteren Läufen und zum Executor-Benchmark
# (logarithmic_benchmark_abstraction.py) identisch — dort ist qnc ebenfalls die
# uncached Linie und damit der Paar-Partner für Overhead = Executor − Roh.

MODE_RUNNERS = {
    "creation": [
        ("qnc", "QNode", "tab:orange", runner_creation_qnode),
    ],
    "execution": [
        ("qnc", "QNode", "tab:orange", runner_execution_qnode),
    ],
    "gradient": [
        ("qnc", "QNode", "tab:orange", runner_gradient_qnode),
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
# Reproduzierbar und erfasst die hier dominante Last (Op-/Tape-Objekte).
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
    csv_file = RUN_DIR / f"benchmark_{GATE_SET_CHOICE}_{mode}.csv"

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