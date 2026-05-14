import pennylane as qml
import numpy as np
import pandas as pd
import gc
import time
from pathlib import Path
from datetime import datetime

# =========================================================
# Konfiguration
# =========================================================

RESULT_DIR = Path("../results")
RESULT_DIR.mkdir(exist_ok=True)

CSV_FILE = RESULT_DIR / "benchmark_results.csv"

REPEATS = 20

QUBIT_CONFIGS = [2, 3, 4]
GATE_CONFIGS = [100, 1000, 5000]

# =========================================================
# Circuit Builder
# =========================================================

def apply_gates(num_qubits, total_gates):
    for i in range(total_gates):
        w = i % num_qubits

        if i % 3 == 0:
            qml.Hadamard(wires=w)
        elif i % 3 == 1:
            qml.PauliX(wires=w)
        else:
            qml.RX(np.pi / 2, wires=w)

# =========================================================
# Benchmarks
# =========================================================

def benchmark_tape(num_qubits, total_gates):
    dev = qml.device("default.qubit", wires=num_qubits)

    with qml.tape.QuantumTape() as tape:
        apply_gates(num_qubits, total_gates)
        qml.probs(wires=range(num_qubits))

    return qml.execute([tape], dev)


def benchmark_qnode_no_cache(num_qubits, total_gates):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, cache=False)
    def circuit():
        apply_gates(num_qubits, total_gates)
        return qml.probs(wires=range(num_qubits))

    return circuit()


def create_cached_qnode(num_qubits, total_gates):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        apply_gates(num_qubits, total_gates)
        return qml.probs(wires=range(num_qubits))

    return circuit

# =========================================================
# Timing
# =========================================================

def measure_runtime(func, repeats=10):
    times = []

    # Warmup
    func()

    for _ in range(repeats):
        gc.collect()

        start = time.perf_counter()
        func()
        end = time.perf_counter()

        times.append(end - start)

    return {
        "avg": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }

# =========================================================
# Main Benchmark Loop
# =========================================================

results = []

for num_qubits in QUBIT_CONFIGS:
    for total_gates in GATE_CONFIGS:

        print(f"\nQubits={num_qubits} Gates={total_gates}")

        cached_qnode = create_cached_qnode(
            num_qubits,
            total_gates         
        )

        benchmarks = [
            ("tape", lambda: benchmark_tape(num_qubits, total_gates)),
            ("qnode_no_cache", lambda: benchmark_qnode_no_cache(num_qubits, total_gates)),
            ("qnode_cached", cached_qnode),
        ]

        for name, func in benchmarks:
            #output file hier generieren

            print(f"Running {name} ...")

            stats = measure_runtime(func, REPEATS)
            results.append({
                "timestamp": datetime.now().isoformat(),
                "benchmark": name,
                "num_qubits": num_qubits,
                "total_gates": total_gates,
                "repeats": REPEATS,
                **stats
            })

# =========================================================
# Save CSV
# =========================================================

df = pd.DataFrame(results)

if CSV_FILE.exists():
    old = pd.read_csv(CSV_FILE)
    df = pd.concat([old, df], ignore_index=True)

df.to_csv(CSV_FILE, index=False)

print("\nSaved results:")
print(CSV_FILE)