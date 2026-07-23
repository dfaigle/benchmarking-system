"""Performance-Benchmark: natives PennyLane (Roh-Framework, OHNE Abstraktionsschicht).

Eigenständiges Gegenstück zu ``examples/abstraction/benchmark_executor.py``:
dieselbe Arbeit (creation / execution / gradient) läuft hier direkt über
PennyLane – ohne ``qc_executor``-Import. Der Overhead der Abstraktionsschicht
ergibt sich anschließend als spaltenweise Differenz

    Executor-CSV  −  diese CSV     (gleiche Zelle: num_qubits × total_gates)

und wird von ``examples/abstraction/compare_overhead.py`` automatisch berechnet.

Der Aufbau ist bewusst IDENTISCH zum Executor-Benchmark: gleiche GATE_SETS,
gleiche ``generate_gate_sequence``-Logik (gleicher SEED → exakt dieselben
Gate-Sequenzen), gleiche QUBIT_/GATE_CONFIGS, gleiche Mess-Methodik und gleiche
CSV-Spalten ``qnc_/qc_``. Die Gatter werden über dasselbe Gate-Mapping
angewendet, das auch das PennyLane-Backend des Executors intern nutzt
(h → qml.Hadamard, cx → qml.CNOT, …) – nur eben direkt.

Zwei Linien – das native Pendant zu den zwei Executor-Linien:

    qnc  (QNode no cache) → idiomatisches PennyLane: EIN QNode, jeder Aufruf
                            baut das Tape neu und simuliert.
                            (Pendant zu Executor.create(..., caching=False))
    qc   (QNode cached)   → derselbe QNode plus simpler Ergebnis-Cache (dict,
                            Schlüssel = Hash der Gate-Sequenz): nach dem
                            Warm-up ist jeder identische Aufruf ein Lookup.
                            (Pendant zu Executor.create(..., caching=True))

Bewusster Unterschied zum Executor (Teil von dessen Overhead, soll mitgemessen
werden): das Device wird hier – wie in PennyLane üblich – mit fester Wire-Zahl
erzeugt (``qml.device("default.qubit", wires=n)``). Der Executor arbeitet mit
dynamischen Wires und fügt intern Identity-Gates ein, damit alle Wires im Tape
landen.
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
import pennylane as qml
import pennylane.numpy as pnp

# Windows-Konsole auf UTF-8 stellen (sonst UnicodeEncodeError bei → und Umlauten)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =========================================================
# ➤  GATE-SET AUSWAHL  ←  hier anpassen
# =========================================================
#
#  "clifford"               → h, s, cx, x, y, z
#  "clifford_t"             → h, t, cx
#  "single_qubit_plus_cnot" → rx, ry, rz, cx
#
# Namen und Gatter decken sich mit den gleichnamigen Sets des
# Executor-Benchmarks – nur so ist die Differenz der CSVs aussagekräftig.
#
GATE_SET_CHOICE = "clifford"

# =========================================================
# ➤  BENCHMARK-MODUS  ←  hier anpassen
# =========================================================
#
#  "creation"  → Zeit, aus der Gatter-Sequenz einen lauffähigen Zustand
#                herzustellen (Quantenfunktion + QNode bauen + erster Aufruf).
#  "execution" → Reine Ausführungszeit (statevector) nach dem Aufbau.
#  "gradient"  → Gradienten-Berechnung (⟨Z₀⟩ nach trainierbarer RY-Schicht).
#
BENCHMARK_MODE = "execution"

# Plots am Ende interaktiv anzeigen? Für Batch-/Headless-Läufe auf False setzen
# (die PNGs werden unabhängig davon immer gespeichert).
SHOW_PLOTS = True

# =========================================================
# Gate-Definitionen  (kanonische Namen, kleingeschrieben)
# =========================================================

GATE_SETS = {
    "clifford":               ["h", "s", "cx", "x", "y", "z"],
    "clifford_t":             ["h", "t", "cx"],
    "single_qubit_plus_cnot": ["rx", "ry", "rz", "cx"],
}

VALID_MODES = {"creation", "execution", "gradient"}


def resolve_gate_set(choice: str) -> list:
    if choice not in GATE_SETS:
        valid = ", ".join(f'"{k}"' for k in GATE_SETS)
        raise ValueError(f"Unbekanntes Gate-Set '{choice}'. Gültig: {valid}")
    return GATE_SETS[choice]


if BENCHMARK_MODE not in VALID_MODES:
    raise ValueError(f"Unbekannter Modus '{BENCHMARK_MODE}'. Gültig: {VALID_MODES}")

ACTIVE_GATE_SET = resolve_gate_set(GATE_SET_CHOICE)
print(f"Framework: natives PennyLane (qml {qml.__version__})")
print(f"Gate-Set : {GATE_SET_CHOICE!r}  →  {ACTIVE_GATE_SET}")
print(f"Modus    : {BENCHMARK_MODE!r}\n")

# =========================================================
# Konfiguration  (identisch zum Executor-Benchmark)
# =========================================================

RESULT_DIR = Path(__file__).parent.parent / "Results" / "PennyLane"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED    = 42
REPEATS = 4

QUBIT_CONFIGS = [5]

GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(1000), 20)).astype(int)
)

#: Präfixe der zwei Linien – identisch zum Executor-Benchmark, damit die CSVs
#: spaltenweise vergleichbar sind (no cache vs Ergebnis-Cache).
METHODS = ["qnc", "qc"]

# =========================================================
# Versioned CSV  (kein Überschreiben alter Läufe)
# =========================================================

version = 1
while True:
    CSV_FILE = (
        RESULT_DIR
        / f"benchmark_native_pennylane_{GATE_SET_CHOICE}_{BENCHMARK_MODE}_V{version}.csv"
    )
    if not CSV_FILE.exists():
        break
    version += 1

# =========================================================
# Gate-Eigenschaften  (kanonische Namen, identisch zum Executor-Benchmark)
# =========================================================

_SINGLE_0 = {"h", "s", "t", "x", "y", "z", "sdag", "tdag"}  # 1 Qubit, kein Winkel
_ANGLE_1  = {"rx", "ry", "rz", "p"}                          # 1 Qubit, 1 Winkel
_TWO_Q_0  = {"cx", "cy", "cz", "swap"}                       # 2 Qubit, kein Winkel
_ANGLE_2  = {"crx", "cry", "crz", "cp"}                      # 2 Qubit, 1 Winkel

# =========================================================
# Natives Gate-Mapping  (deckungsgleich mit dem PennyLane-Backend des Executors:
# src/qc_executor/pennylane/pennylane_gates.py – hier bewusst dupliziert, damit
# dieses Skript qc_executor-frei bleibt)
# =========================================================

GATE_MAP = {
    "h": qml.Hadamard,
    "s": qml.S,
    "t": qml.T,
    "x": qml.PauliX,
    "y": qml.PauliY,
    "z": qml.PauliZ,
    "cx": qml.CNOT,
    "cy": qml.CY,
    "cz": qml.CZ,
    "swap": qml.SWAP,
    "rx": qml.RX,
    "ry": qml.RY,
    "rz": qml.RZ,
    "p": qml.PhaseShift,
    "crx": qml.CRX,
    "cry": qml.CRY,
    "crz": qml.CRZ,
    "cp": qml.ControlledPhaseShift,
}

# =========================================================
# Gate-Sequenz generieren  (identische Logik/Seed wie Executor-Benchmark)
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


def sequence_key(sequence: list) -> int:
    """Hashbarer Schlüssel einer Gate-Sequenz (Pendant zum Circuit-Hash des
    Executor-Ergebnis-Caches; wird wie dort bei JEDEM Cache-Zugriff berechnet)."""
    return hash(tuple((gn, tuple(ws), tuple(ps)) for gn, ws, ps in sequence))

# =========================================================
# Nativen Schaltkreis bauen
# =========================================================

def apply_native(gate_name: str, wires: list, params: list) -> None:
    op = GATE_MAP.get(gate_name)
    if op is None:
        raise ValueError(f"Unbekanntes Gatter: {gate_name!r}")
    if gate_name in _ANGLE_1 or gate_name in _ANGLE_2:
        op(params[0], wires=wires)
    else:
        op(wires=wires)


def make_statevector_qnode(dev, sequence: list):
    """Quantenfunktion aus der Gate-Sequenz bauen und als QNode zurückgeben."""

    def qfunc():
        for gn, ws, ps in sequence:
            apply_native(gn, ws, ps)
        return qml.state()

    return qml.QNode(qfunc, dev)


def make_gradient_qnode(dev, num_qubits: int, sequence: list):
    """Trainierbare RY-Schicht (ein Parameter je Qubit) + Gate-Block, ⟨Z₀⟩."""

    def qfunc(theta):
        for i in range(num_qubits):
            qml.RY(theta[i], wires=[i])
        for gn, ws, ps in sequence:
            apply_native(gn, ws, ps)
        return qml.expval(qml.PauliZ(0))

    return qml.QNode(qfunc, dev, diff_method="best", max_diff=1)

# =========================================================
# Runner-Builder je Modus
# =========================================================
# Liefert ein dict {methoden_key: run_callable}. Der Aufbau von Device und
# QNode passiert HIER (Setup, nicht gemessen); nur ``run()`` wird getimt.
# Beide Linien bekommen exakt dieselbe Eingabe – der EINZIGE Unterschied ist
# der Ergebnis-Cache:
#
#   qnc → jeder Aufruf baut das Tape neu und simuliert (PennyLane-Standard)
#   qc  → Ergebnis-Cache: Cache-Treffer nach dem Warm-up

def build_runners(mode: str, num_qubits: int, sequence: list) -> dict:
    dev = qml.device("default.qubit", wires=num_qubits)

    # ------------------------------------------------ creation
    if mode == "creation":
        result_cache = {}

        def r_qnc():
            make_statevector_qnode(dev, sequence)()

        def r_qc():
            qnode = make_statevector_qnode(dev, sequence)
            key = sequence_key(sequence)
            if key not in result_cache:
                result_cache[key] = qnode()
            return result_cache[key]

        return {"qnc": r_qnc, "qc": r_qc}

    # ------------------------------------------------ execution
    if mode == "execution":
        qnode = make_statevector_qnode(dev, sequence)
        result_cache = {}

        def r_qnc():
            qnode()

        def r_qc():
            key = sequence_key(sequence)
            if key not in result_cache:
                result_cache[key] = qnode()
            return result_cache[key]

        return {"qnc": r_qnc, "qc": r_qc}

    # ------------------------------------------------ gradient
    if mode == "gradient":
        qnode = make_gradient_qnode(dev, num_qubits, sequence)
        values = pnp.zeros(num_qubits, requires_grad=True)
        result_cache = {}

        def r_qnc():
            qml.jacobian(qnode)(values)

        def r_qc():
            key = (sequence_key(sequence), tuple(float(v) for v in values))
            if key not in result_cache:
                result_cache[key] = qml.jacobian(qnode)(values)
            return result_cache[key]

        return {"qnc": r_qnc, "qc": r_qc}

    raise ValueError(f"Unbekannter Modus '{mode}'.")


MODE_LABELS = {
    "creation":  "Erstellungszeit",
    "execution": "Ausführungszeit",
    "gradient":  "Gradienten-Berechnungszeit",
}

# =========================================================
# Timing  (identisch zum Executor-Benchmark)
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
# Speicher-Messung (Peak via tracemalloc, identisch zum Executor-Benchmark)
# =========================================================

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

        sequence = generate_gate_sequence(
            num_qubits, total_gates, ACTIVE_GATE_SET, seed=SEED
        )

        runners = build_runners(BENCHMARK_MODE, num_qubits, sequence)

        stats = {m: measure_runtime(runners[m], REPEATS) for m in METHODS}
        mem   = {m: measure_memory(runners[m], REPEATS) for m in METHODS}

        print(
            f"  qnc={stats['qnc']['avg']:.5f}s  qc={stats['qc']['avg']:.5f}s"
        )

        row = {
            "timestamp":      datetime.now().isoformat(),
            "backend":        "pennylane_native",
            "gate_set":       GATE_SET_CHOICE,
            "benchmark_mode": BENCHMARK_MODE,
            "num_qubits":     num_qubits,
            "total_gates":    total_gates,
        }
        for m in METHODS:
            row.update({f"{m}_{k}": v for k, v in stats[m].items()})
            row.update({f"{m}_mem_{k}": v for k, v in mem[m].items()})
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
# Plot  (2 Linien: no cache vs Ergebnis-Cache; Layout wie Executor-Benchmark)
# =========================================================
# Farbkonzept über alle Benchmark-Plots hinweg: BLAU = natives PennyLane,
# ORANGE = Executor/Abstraktion. Die Cache-Variante wird NICHT über einen
# weiteren Farbton kodiert, sondern über den Linienstil (durchgezogen = ohne
# Cache, gestrichelt = mit Cache) – bleibt damit auch bei Farbfehlsicht lesbar.

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
        ("PennyLane nativ (no cache)", "qnc", "#1f77b4", "-",  "o"),
        ("PennyLane nativ (cached)",   "qc",  "#1f77b4", "--", "s"),
    ]

    n_qubits = len(qubit_vals)
    fig, axes = plt.subplots(1, n_qubits, figsize=(6 * n_qubits, 5), squeeze=False)
    axes = axes[0]

    fig.suptitle(
        f"{title}  |  natives PennyLane  |  Gate-Set: {gate_set!r} "
        f"({', '.join(ACTIVE_GATE_SET)})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, nq in zip(axes, qubit_vals):
        sub = df[df["num_qubits"] == nq].sort_values("total_gates")
        x   = sub["total_gates"].values

        for method_label, prefix, color, linestyle, marker in method_cfg:
            y = sub[f"{prefix}_{avg_suffix}"].values
            s = sub[f"{prefix}_{std_suffix}"].values

            ax.plot(x, y, marker=marker, linewidth=1.8, markersize=4,
                    linestyle=linestyle, label=method_label, color=color,
                    markerfacecolor="white" if linestyle == "--" else color)
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

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


mode_label = MODE_LABELS[BENCHMARK_MODE]
_stub = f"benchmark_native_pennylane_{GATE_SET_CHOICE}_{BENCHMARK_MODE}_V{version}"

# --- Laufzeit ---
plot_metric(
    df, avg_suffix="avg", std_suffix="std",
    ylabel="Zeit (s)", title=mode_label,
    save_path=RESULT_DIR / f"{_stub}.png",
)

# --- Speicherverbrauch (Peak, tracemalloc) ---
plot_metric(
    df, avg_suffix="mem_avg", std_suffix="mem_std",
    ylabel="Peak-Speicher (MiB)", title=f"Speicherverbrauch (Peak) – {mode_label}",
    save_path=RESULT_DIR / f"{_stub}_mem.png",
)
