"""Performance-Benchmark des Executors (Abstraktionsschicht), backend-umschaltbar.

Eigenständiges Gegenstück zum reinen PennyLane-/Qiskit-Benchmark: hier läuft
dieselbe Arbeit (creation / execution / gradient) über den ``Executor`` — einmal
mit dem PennyLane-, einmal mit dem Qiskit-Backend (``BACKEND_CHOICE``).

Der Aufbau ist bewusst IDENTISCH zum Roh-Framework-Benchmark (gleiche Gate-Sets,
gleiche GATE_CONFIGS/QUBIT_CONFIGS, gleiche Mess-Methodik, gleiche CSV-Spalte
``qnc_``, gleiches Plot-Layout). Dadurch lassen sich die beiden Benchmarks
fair vergleichen: der Overhead deiner Abstraktion ergibt sich als spaltenweise
Differenz  Executor − Roh-Framework  für dieselbe Zelle.

Eine Linie, gemessen wird der Standardfall des Executors:

    qnc  (no cache) → Executor mit caching=False (Standardfall): rechnet bei
                      jedem Aufruf neu (abstrakter Circuit wird je Aufruf ins
                      native Format übersetzt und simuliert).

Die frühere cached-Linie (``qc``, ``caching=True``) ist entfernt: sie misst nach
dem Warm-up nur noch Cache-Treffer, also den Lookup statt der eigentlichen
Arbeit, und hat im Roh-Benchmark keinen sinnvollen Paar-Partner (der PennyLane-
Cache greift dort nicht). Der Spalten-Präfix ``qnc`` bleibt erhalten, damit alte
CSVs und die Vergleichs-Skripte weiter passen.

Hinweis Vergleichbarkeit: Identische Gatter-Sequenzen (gleicher Seed) entstehen
nur, wenn beide Benchmarks eine Gate-Liste GLEICHER Länge und Reihenfolge
nutzen — rng.choice zieht Indizes über die Liste. Zusätzlich muss pro Gattertyp
die Zahl der RNG-Aufrufe übereinstimmen (ein Gatter ohne Winkel verbraucht
keinen rng.uniform-Aufruf und verschiebt den Zufallsstrom). Das gilt für
"clifford", "clifford_t" und "single_qubit_plus_cnot" (gleicher Name in beiden
Skripten) sowie für "non_clifford" HIER ↔ "non_clifford_comparable" im
Roh-Benchmark (beide 7 Gatter, ohne T).
Das Roh-Set "non_clifford" (mit Rot/Toffoli, 10 Gatter) ist NICHT vergleichbar.
"""

import gc
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from qc_executor import Executor
from qc_executor.abstraction import (
    AbstractQuantumCircuit,
    AbstractQuantumOperator,
    ParameterVector,
)

# Windows-Konsole auf UTF-8 stellen (sonst UnicodeEncodeError bei → und Umlauten)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =========================================================
# ➤  BACKEND-AUSWAHL  ←  hier anpassen
# =========================================================
#
#  "pennylane"  → Executor.create("pennylane")
#  "qiskit"     → Executor.create("qiskit")
#
BACKEND_CHOICE = "pennylane"

# =========================================================
# ➤  GATE-SET AUSWAHL  ←  hier anpassen
# =========================================================
#
#  "clifford"               → h, s, cx, x, y, z
#  "non_clifford"           → rx, ry, rz, crx, cry, crz, cp   (7 Gatter, ohne t)
#  "clifford_plus_non_clifford"
#                           → beide obigen kombiniert (13 Gatter): h, s, cx,
#                             x, y, z, rx, ry, rz, crx, cry, crz, cp
#  "clifford_t"             → h, t, cx
#  "single_qubit_plus_cnot" → rx, ry, rz, cx
#
# Vergleichbarkeit mit dem Roh-Benchmark (identische Sequenz bei gleichem Seed):
#   "clifford", "clifford_t", "single_qubit_plus_cnot" → gleicher Name dort
#   "clifford_plus_non_clifford"                       → gleicher Name dort
#   "non_clifford"                                     → dort "non_clifford_comparable"
#                                                        wählen (ebenfalls 7 Gatter,
#                                                        gleiche Reihenfolge)!
# Das Roh-Set "non_clifford" (10 Gatter, mit Rot/Toffoli) erzeugt eine ANDERE
# Sequenz und ist NICHT vergleichbar.
#
GATE_SET_CHOICE = "clifford_plus_non_clifford"

# =========================================================
# ➤  BENCHMARK-MODUS  ←  hier anpassen
# =========================================================
#
#  "creation"  → Zeit, den abstrakten Circuit zu bauen und ins native Format
#                zu übersetzen (transpile_circuit) — OHNE Ausführung, analog
#                zum Roh-Benchmark (dort: construct_tape bzw. Circuit befüllen).
#  "execution" → Reine Ausführungszeit des Erwartungswerts ⟨Z₀⟩ nach dem Aufbau.
#  "gradient"  → Gradienten-Berechnung (⟨Z₀⟩; TRAINABLE_RATIO der Gatter sind
#                durch trainierbare RY ersetzt, verstreut über den ganzen
#                Circuit — identisch zum Roh-Benchmark, Gesamt-Gatterzahl bleibt
#                total_gates).
#  "all"       → alle drei Modi nacheinander (creation → execution → gradient);
#                eigene CSV + Plots pro Modus, gemeinsamer run_<N>-Ordner.
#
BENCHMARK_MODE = "all"

# Plots am Ende interaktiv anzeigen? Für Batch-/Headless-Läufe auf False setzen
# (die PNGs werden unabhängig davon immer gespeichert).
SHOW_PLOTS = True

# =========================================================
# Gate-Definitionen  (kanonische Namen, kleingeschrieben)
# =========================================================

# "non_clifford" spiegelt Position für Position NON_CLIFFORD_COMPARABLE_GATES des
# Roh-Benchmarks (["RX","RY","RZ","CRX","CRY","CRZ","ControlledPhaseShift"]).
# Gleiche Länge + Reihenfolge UND pro Gattertyp gleich viele RNG-Aufrufe →
# identische Sequenz bei gleichem Seed. Bei Änderungen beide Listen synchron
# halten!
# OHNE "t": T ist das einzige Gatter dieses Sets ohne Winkel — es verbraucht
# keinen rng.uniform-Aufruf und verschiebt damit den gesamten Zufallsstrom
# gegenüber dem Roh-Benchmark. Solange "t" hier drinstand (8 statt 7 Gatter),
# erzeugten Roh- und Executor-Benchmark bei gleichem Seed VERSCHIEDENE
# Schaltkreise und waren nicht vergleichbar.
GATE_SETS = {
    "clifford":               ["h", "s", "cx", "x", "y", "z"],
    "non_clifford":           ["rx", "ry", "rz", "crx", "cry", "crz", "cp"],
    "clifford_t":             ["h", "t", "cx"],
    "single_qubit_plus_cnot": ["rx", "ry", "rz", "cx"],
}

# Clifford + non-Clifford in EINEM Set (13 Gatter). Reihenfolge: erst Clifford,
# dann non-Clifford — MUSS mit CLIFFORD_PLUS_NON_CLIFFORD_GATES der Roh-
# Benchmarks identisch sein, da rng.choice Indizes über die Liste zieht.
# Der kritische Fall ist cx/CNOT: beide Generatoren ziehen dafür KEINEN Winkel
# (hier: cx liegt in _TWO_Q_0, nicht in _ANGLE_2), sonst würde der Zufallsstrom
# gegenüber dem Roh-Benchmark auseinanderlaufen.
# Hinweis: ~46 % der Gatter sind damit winkelfrei (vorher 0 %). n_trainable
# bleibt trotzdem seed-unabhängig, da trainable_layout nur von total_gates
# abhängt und apply_trainable das Gatter unabhängig von seinem Typ ersetzt.
GATE_SETS["clifford_plus_non_clifford"] = (
    GATE_SETS["clifford"] + GATE_SETS["non_clifford"]
)

VALID_MODES = {"creation", "execution", "gradient", "all"}
VALID_BACKENDS = {"pennylane", "qiskit"}


def resolve_gate_set(choice: str) -> list:
    if choice not in GATE_SETS:
        valid = ", ".join(f'"{k}"' for k in GATE_SETS)
        raise ValueError(f"Unbekanntes Gate-Set '{choice}'. Gültig: {valid}")
    return GATE_SETS[choice]


if BENCHMARK_MODE not in VALID_MODES:
    raise ValueError(f"Unbekannter Modus '{BENCHMARK_MODE}'. Gültig: {VALID_MODES}")
if BACKEND_CHOICE not in VALID_BACKENDS:
    raise ValueError(f"Unbekanntes Backend '{BACKEND_CHOICE}'. Gültig: {VALID_BACKENDS}")

ACTIVE_GATE_SET = resolve_gate_set(GATE_SET_CHOICE)
print(f"Backend  : {BACKEND_CHOICE!r}")
print(f"Gate-Set : {GATE_SET_CHOICE!r}  →  {ACTIVE_GATE_SET}")
print(f"Modus    : {BENCHMARK_MODE!r}\n")

# =========================================================
# Konfiguration  (identisch zum Roh-Benchmark)
# =========================================================

# Datei liegt in Benchmarks/abstract/ → drei Ebenen hoch zur Projekt-Wurzel.
# (parent = abstract/, parent.parent = Benchmarks/, parent.parent.parent = Wurzel)
RESULT_DIR = Path(__file__).parent.parent.parent / "Results" / "Executor"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED    = 42
REPEATS = 4

QUBIT_CONFIGS = [5]

GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(100000), 20)).astype(int)
)

#: Präfix der gemessenen Linie — identisch zum Roh-Benchmark, damit die CSVs
#: spaltenweise vergleichbar sind (beidseitig ohne Ergebnis-Cache).
METHODS = ["qnc"]

# =========================================================
# Pro Lauf ein eigener Unterordner:
#   Results/Executor/<timestamp>_qubits-<...>_<backend>/
# Das Backend steht mit im Ordnernamen, weil beide Backends in denselben
# Executor-Ordner schreiben — so sind pennylane- und qiskit-Läufe auf einen
# Blick unterscheidbar.
# =========================================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

qubit_string = "-".join(map(str, QUBIT_CONFIGS))

RUN_DIR = RESULT_DIR / f"{timestamp}_qubits-{qubit_string}_{BACKEND_CHOICE}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# CSV-/Plot-Dateinamen entstehen pro Modus in run_benchmark() bzw. im Haupt-Ablauf.

# =========================================================
# Gate-Eigenschaften  (kanonische Namen)
# =========================================================

_SINGLE_0 = {"h", "s", "t", "x", "y", "z", "sdag", "tdag"}  # 1 Qubit, kein Winkel
_ANGLE_1  = {"rx", "ry", "rz", "p"}                          # 1 Qubit, 1 Winkel
_TWO_Q_0  = {"cx", "cy", "cz", "swap"}                       # 2 Qubit, kein Winkel
_ANGLE_2  = {"crx", "cry", "crz", "cp"}                      # 2 Qubit, 1 Winkel

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

        if gate in _TWO_Q_0 or gate in _ANGLE_2:
            if num_qubits >= 2:
                target = int((w + 1) % num_qubits)
                params = [float(rng.uniform(0, 2 * np.pi))] if gate in _ANGLE_2 else []
                sequence.append((gate, [w, target], params))
            else:
                sequence.append(("h", [w], []))
        elif gate in _ANGLE_1:
            sequence.append((gate, [w], [float(rng.uniform(0, 2 * np.pi))]))
        else:
            sequence.append((gate, [w], []))

    return sequence

# =========================================================
# Abstrakten Schaltkreis bauen
# =========================================================

def apply_abstract(qc: AbstractQuantumCircuit, gate_name: str, wires: list, params: list) -> None:
    if   gate_name in _SINGLE_0: getattr(qc, gate_name)(wires[0])
    elif gate_name in _ANGLE_1:  getattr(qc, gate_name)(wires[0], params[0])
    elif gate_name in _TWO_Q_0:  getattr(qc, gate_name)(wires[0], wires[1])
    elif gate_name in _ANGLE_2:  getattr(qc, gate_name)(wires[0], wires[1], params[0])
    else:
        raise ValueError(f"Unbekanntes Gatter: {gate_name!r}")


def build_abstract(num_qubits: int, sequence: list) -> AbstractQuantumCircuit:
    qc = AbstractQuantumCircuit(num_qubits)
    for gn, ws, ps in sequence:
        apply_abstract(qc, gn, ws, ps)
    return qc


# Anteil der Gatter, der im Gradient-Modus durch trainierbare RY ersetzt wird.
# MUSS mit TRAINABLE_RATIO in logarithmic_benchmark_pennylane.py synchron sein,
# sonst bauen Roh- und Executor-Benchmark verschiedene Circuits und die
# Overhead-Differenz (Executor − Roh) ist im Gradient-Modus ungültig.
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


# Rotationsachsen, die an den trainierbaren Positionen zyklisch (über den Zähler
# k) eingesetzt werden. Alle drei sind 1-Parameter-Gatter → die Zähl-Logik
# (theta[k], n_trainable) bleibt unverändert korrekt. Auf ein Set beschränken
# (z. B. ["RY"]) genügt, um nur eine Achse zu nutzen. MUSS mit TRAINABLE_GATES
# der anderen Benchmarks synchron sein.
TRAINABLE_GATES = ["RX", "RY", "RZ"]


def apply_trainable(qc: AbstractQuantumCircuit, k: int, param, wire: int) -> None:
    # Achtung: die Abstraktionsschicht nimmt den Wire ZUERST, dann den Winkel
    # (qc.rx(wire, angle)) — umgekehrt zu Qiskit/PennyLane.
    gate = TRAINABLE_GATES[k % len(TRAINABLE_GATES)]
    if   gate == "RX": qc.rx(wire, param)
    elif gate == "RY": qc.ry(wire, param)
    else:              qc.rz(wire, param)


def build_abstract_trainable(
    num_qubits: int, sequence: list, step: int, theta: ParameterVector
) -> AbstractQuantumCircuit:
    """Trainierbare Rotationen verstreut über den ganzen Circuit — identisch zum
    Roh-Benchmark.

    Jedes step-te Gatter wird durch eine trainierbare Rotation RX/RY/RZ(theta[k])
    — zyklisch über k — auf dem Wire des ersetzten Gatters (ws[0]) ersetzt; die
    übrigen Gatter bleiben fest.
    """
    qc = AbstractQuantumCircuit(num_qubits)
    k = 0
    for i, (gn, ws, ps) in enumerate(sequence):
        if i % step == 0:
            apply_trainable(qc, k, theta[k], ws[0])   # RX / RY / RZ im Wechsel
            k += 1
        else:
            apply_abstract(qc, gn, ws, ps)
    return qc

# =========================================================
# Runner-Builder je Modus
# =========================================================
# Liefert ein dict {methoden_key: run_callable}. Der Aufbau von Executor und
# (abstraktem) Circuit passiert HIER (Setup, nicht gemessen); nur ``run()`` wird
# getimt.
#
#   qnc → Executor mit caching=False (Standardfall): rechnet bei jedem Aufruf neu

def build_runners(mode: str, num_qubits: int, sequence: list) -> dict:
    ex = Executor.create(BACKEND_CHOICE)   # ohne Ergebnis-Cache (Standardfall)

    # ------------------------------------------------ creation
    # NUR bauen + ins native Format übersetzen, NICHT ausführen — sonst ist der
    # Modus nicht mit den Roh-Benchmarks vergleichbar, die im creation-Modus
    # ebenfalls nur bauen (PennyLane: construct_tape ohne execute, Qiskit:
    # QuantumCircuit befüllen). Mit dem früheren ex.statevector(...) steckte eine
    # volle Statevector-Simulation im Messfenster: bei 10 Qubits/2000 Gattern
    # 1.58 s statt 0.018 s — die Differenz "Executor − Roh" maß damit zu ~99 %
    # die Simulation statt des Abstraktions-Overheads.
    # transpile_circuit() greift hier keinen Cache ab: caching=False → kein
    # Ergebnis-Cache, und build_abstract() liefert pro Aufruf ein neues Objekt.
    if mode == "creation":
        def r_qnc():
            ex.transpile_circuit(build_abstract(num_qubits, sequence))

        return {"qnc": r_qnc}

    # ------------------------------------------------ execution: ⟨Z₀⟩
    # Erwartungswert ⟨Z₀⟩ statt voller Statevector — dieselbe Aufgabe wie
    # qml.expval(qml.PauliZ(0)) im Roh-Benchmark, damit beide fair vergleichbar
    # sind. Circuit und Observable werden HIER (Setup, nicht gemessen) gebaut
    # bzw. transpiliert; nur der Erwartungswert-Aufruf liegt im Messfenster.
    if mode == "execution":
        z0 = "I" * (num_qubits - 1) + "Z"          # ⟨Z₀⟩ (little-endian)
        obs = AbstractQuantumOperator(paulis=[z0], coeffs=[1.0])

        qc_abs = build_abstract(num_qubits, sequence)
        op = ex.transpile_operator(obs)

        def r_qnc():
            ex.expectation_value(qc_abs, op)

        return {"qnc": r_qnc}

    # ------------------------------------------------ gradient
    if mode == "gradient":
        z0 = "I" * (num_qubits - 1) + "Z"          # ⟨Z₀⟩ (little-endian)
        obs = AbstractQuantumOperator(paulis=[z0], coeffs=[1.0])

        n_trainable, step = trainable_layout(len(sequence))
        values = np.zeros(n_trainable).tolist()

        theta = ParameterVector("x", n_trainable)
        qc_abs = build_abstract_trainable(num_qubits, sequence, step, theta)
        op = ex.transpile_operator(obs)

        def r_qnc():
            ex.expectation_value_derivatives(qc_abs, op, "x", **{"x": values})

        return {"qnc": r_qnc}

    raise ValueError(f"Unbekannter Modus '{mode}'.")


MODE_LABELS = {
    "creation":  "Erstellungszeit",
    "execution": "Ausführungszeit",
    "gradient":  "Gradienten-Berechnungszeit",
}

# "all" → alle drei Modi nacheinander, sonst nur der gewählte
MODES_TO_RUN = (
    ["creation", "execution", "gradient"] if BENCHMARK_MODE == "all" else [BENCHMARK_MODE]
)

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
# Speicher-Messung (Peak via tracemalloc)
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
    csv_file = RUN_DIR / f"benchmark_executor_{BACKEND_CHOICE}_{GATE_SET_CHOICE}_{mode}.csv"

    results = []
    first_write = True   # Header nur beim allerersten Schreiben
    #tauschen
    for num_qubits in QUBIT_CONFIGS:
        for total_gates in GATE_CONFIGS:

            print(f"Qubits={num_qubits}  Gates={total_gates}")

            # Fabrik pro Methode: erzeugt für einen Seed eine frische Gatter-
            # Sequenz und den zugehörigen Runner. Die Mess-Funktionen rufen sie
            # mit SEED + r auf → anderer Zufalls-Circuit pro Wiederholung.
            def make_runner_factory(m):
                def factory(seed):
                    seq = generate_gate_sequence(
                        num_qubits, total_gates, ACTIVE_GATE_SET, seed=seed
                    )
                    return build_runners(mode, num_qubits, seq)[m]
                return factory

            stats = {m: measure_runtime(make_runner_factory(m), REPEATS) for m in METHODS}
            mem   = {m: measure_memory(make_runner_factory(m), REPEATS) for m in METHODS}

            print(
                "  " + "  ".join(
                    f"{m}={stats[m]['avg']:.5f}s/{mem[m]['avg']:.1f}MiB" for m in METHODS
                )
            )

            row = {
                "timestamp":      datetime.now().isoformat(),
                "backend":        BACKEND_CHOICE,
                "gate_set":       GATE_SET_CHOICE,
                "benchmark_mode": mode,
                "num_qubits":     num_qubits,
                "total_gates":    total_gates,
            }
            for m in METHODS:
                row.update({f"{m}_{k}": v for k, v in stats[m].items()})
                row.update({f"{m}_mem_{k}": v for k, v in mem[m].items()})
            results.append(row)

            # Zwischenergebnis sofort anhängen — überlebt einen Absturz
            pd.DataFrame([row]).to_csv(csv_file, mode="a", header=first_write, index=False)
            first_write = False

    print(f"\nErgebnisse gespeichert (inkrementell während des Laufs): {csv_file}")
    return pd.DataFrame(results)

# =========================================================
# Plot  (eine Linie: Executor ohne Ergebnis-Cache; Layout wie Roh-Benchmark)
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
        ("Executor", "qnc", "tab:orange"),
    ]

    n_qubits = len(qubit_vals)
    fig, axes = plt.subplots(1, n_qubits, figsize=(6 * n_qubits, 5), squeeze=False)
    axes = axes[0]

    fig.suptitle(
        f"{title}  |  Backend: {BACKEND_CHOICE!r}  |  Gate-Set: {gate_set!r} "
        f"({', '.join(ACTIVE_GATE_SET)})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, nq in zip(axes, qubit_vals):
        sub = df[df["num_qubits"] == nq].sort_values("total_gates")
        x   = sub["total_gates"].values

        for method_label, prefix, color in method_cfg:
            y = sub[f"{prefix}_{avg_suffix}"].values
            s = sub[f"{prefix}_{std_suffix}"].values

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

    # Anzeigen passiert gesammelt am Ende (plt.show() im Haupt-Ablauf);
    # ohne Anzeige die Figur sofort schließen, um Speicher freizugeben.
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
    mode_label = MODE_LABELS[mode]
    stub       = f"benchmark_executor_{BACKEND_CHOICE}_{GATE_SET_CHOICE}_{mode}"

    # --- Laufzeit ---
    plot_metric(
        df, avg_suffix="avg", std_suffix="std",
        ylabel="Zeit (s)", title=mode_label,
        save_path=RUN_DIR / f"{stub}_time.png",
    )

    # --- Speicherverbrauch (Peak, tracemalloc) ---
    plot_metric(
        df, avg_suffix="mem_avg", std_suffix="mem_std",
        ylabel="Peak-Speicher (MiB)", title=f"Speicherverbrauch (Peak) – {mode_label}",
        save_path=RUN_DIR / f"{stub}_mem.png",
    )

if SHOW_PLOTS:
    plt.show()
