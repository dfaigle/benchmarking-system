# Benchmarking System

Performance benchmarks for my bachelor thesis. They measure how much overhead the
[qc-executor](https://github.com/flaqship/qc-executor) abstraction layer adds compared to
using PennyLane or Qiskit directly.

The results from this repository are used in the thesis:
**[dfaigle/dfaigle-bachelor-Thesis](https://github.com/dfaigle/dfaigle-bachelor-Thesis)**.
The plots shown there are the ones in [`final_plotted_results/`](final_plotted_results/).

## Idea

The same random circuits (same gate set, seed and gate counts) are run twice:

```
raw framework:   PennyLane / Qiskit  ──────────────►  simulator
executor:        AbstractQuantumCircuit ─► Executor ─► PennyLane / Qiskit ─► simulator
```

The difference between the two runs is the cost of the abstraction. Each run
measures time and peak memory (`tracemalloc`) in three modes:

| Mode        | What is measured                                   |
|-------------|----------------------------------------------------|
| `creation`  | building the circuit (executor: incl. translation) |
| `execution` | computing the expectation value ⟨Z₀⟩               |
| `gradient`  | computing the gradient of ⟨Z₀⟩                     |

## Folder structure

```
benchmarks/
  pennylane_benchmark.py      raw PennyLane benchmark (gradient: backprop)
  qiskit_benchmark.py         raw Qiskit benchmark (gradient: parameter-shift)
  abstraction_benchmark.py    same benchmark through the Executor (PennyLane or Qiskit backend)
  qnode_vs_tape_benchmark.py  PennyLane QNode vs. tape (side comparison)
  plot_forms/                 scripts that combine several runs into comparison plots
raw_results/                  CSVs + per-run plots, one folder per run (<timestamp>_qubits-<n>)
final_plotted_results/        final comparison plots, these are the plots used in the thesis
vendor/executor/              qc-executor as git submodule
```

There is one script per benchmark. Qubit count, gate set, mode etc. are passed
as command-line flags (`python <script> --help` lists them):

| Flag          | Meaning                                                      | Default                       |
|---------------|--------------------------------------------------------------|-------------------------------|
| `--framework` | backend of the Executor, `pennylane` or `qiskit` (**`abstraction_benchmark.py` only, required**) | none |
| `--gates`     | gate set, e.g. `clifford`, `clifford_t`, `clifford_plus_non_clifford` (full list at the top of each script) | `clifford_plus_non_clifford` (`qnode_vs_tape`: `clifford`) |
| `--mode`      | `creation`, `execution`, `gradient` or `all`                 | `all`                         |
| `--qubits`    | one or more qubit counts, e.g. `--qubits 5 8 10 12`          | `5`                           |
| `--max-gates` | largest gate count; 20 log-spaced steps from 10 up to this value | `100000` (`qnode_vs_tape`: `10000`) |
| `--repeats`   | repetitions per cell (avg/std/min/max in CSV)                | `4`                           |

## Setup and usage

Setup (once):

```bash
git submodule update --init --recursive   # downloads the executor into vendor/executor
pip install -r requirements.txt
pip install ./vendor/executor
```

Running a benchmark or plot script — these are just examples, every script in
`benchmarks/` is started the same way:

```bash
python benchmarks/pennylane_benchmark.py --qubits 5                              # raw PennyLane, 5 qubits
python benchmarks/abstraction_benchmark.py --framework pennylane --qubits 5      # same, through the Executor
python benchmarks/qiskit_benchmark.py --qubits 5 8 --mode gradient --max-gates 1000
python benchmarks/plot_forms/plot_pennylane_vs_executor_multi_qubit.py           # comparison plot
```

Several values for `--qubits` run one after another in the same process. To run
qubit counts in parallel, start the script once per qubit count.

Results are written to `raw_results/<Pennylane|Qiskit|Executor|TapeVsQNode>/`.
The plot scripts read the run folders listed in their `RUNS` dict and write to
`final_plotted_results/`.

Long runs can also be started in Docker:

```bash
./build_and_run.sh                                                  # pennylane_benchmark.py, default flags
./build_and_run.sh benchmarks/qiskit_benchmark.py --qubits 8 --mode gradient
./build_and_run.sh benchmarks/abstraction_benchmark.py --framework pennylane --qubits 5 8 10 12
```

The script and its flags are passed straight to `python` inside the container.
`benchmarks/` is mounted read-only, results are written to `raw_results/` on the host.
