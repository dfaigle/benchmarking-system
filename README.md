# Benchmarking System

Performance benchmarks for my bachelor thesis. They measure how much overhead the
[qc-executor](https://github.com/flaqship/qc-executor) abstraction layer adds compared to
using PennyLane or Qiskit directly.

The results of these benchmarks are used in the thesis:
**[dfaigle/dfaigle-bachelor-Thesis](https://github.com/dfaigle/dfaigle-bachelor-Thesis)**.
The measured data and plots are not part of this repository; the benchmarks create
them locally when you run them (see [Results](#results)).

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
plot_forms/                   scripts that combine several runs into comparison plots
  plot_pennylane_vs_executor_multi_qubit.py      time,   PennyLane vs. Executor
  plot_pennylane_vs_executor_multi_qubit_mem.py  memory, PennyLane vs. Executor
  plot_qiskit_vs_executor_multi_qubit.py         time,   Qiskit vs. Executor
  plot_qiskit_vs_executor_multi_qubit_mem.py     memory, Qiskit vs. Executor
  plot_tape_vs_qnode_multi_qubit.py              PennyLane QNode vs. tape
vendor/executor/              qc-executor as git submodule
Dockerfile                    image with Python 3.12, requirements and the executor
build_and_run.sh              builds the image and starts a benchmark in Docker
requirements.txt              Python dependencies
```

Created locally when you run the scripts, ignored by git:

```
raw_results/                  CSVs + per-run plots, one folder per run (<timestamp>_qubits-<n>)
  Pennylane/                  from pennylane_benchmark.py
  Qiskit/                     from qiskit_benchmark.py
  Executor/                   from abstraction_benchmark.py (<timestamp>_qubits-<n>_<backend>)
  TapeVsQNode/                from qnode_vs_tape_benchmark.py
final_plotted_results/        comparison plots from plot_forms/
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

## Plots

The scripts in `plot_forms/` write these plots to `final_plotted_results/`. Each
plot combines all three modes and, in the thesis, the qubit counts 5, 8, 10 and 12
in one diagram:

| Plot | Script in `plot_forms/` | Shows |
|---|---|---|
| `pennylane_vs_executor_qubits_time_new.png` | `plot_pennylane_vs_executor_multi_qubit.py` | runtime, raw PennyLane vs. Executor (PennyLane backend) |
| `pennylane_vs_executor_qubits_mem.png` | `plot_pennylane_vs_executor_multi_qubit_mem.py` | peak memory, raw PennyLane vs. Executor (PennyLane backend) |
| `qiskit_vs_executor_qubits_time.png` | `plot_qiskit_vs_executor_multi_qubit.py` | runtime, raw Qiskit vs. Executor (Qiskit backend) |
| `qiskit_vs_executor_qubits_mem.png` | `plot_qiskit_vs_executor_multi_qubit_mem.py` | peak memory, raw Qiskit vs. Executor (Qiskit backend) |
| `tape_vs_qnode_qubits_time.png` | `plot_tape_vs_qnode_multi_qubit.py` | runtime, PennyLane tape vs. QNode (parameter-shift and backprop) |

How to read them:

- **x-axis:** number of gates, **y-axis:** time in seconds or peak memory in MiB
  (`tracemalloc`), both logarithmic.
- **Color** = mode (`creation`, `execution`, `gradient`); each group is labelled
  directly at its lines.
- **Line style** = implementation: raw framework solid, Executor dash-dotted
  (tape vs. QNode: tape solid, QNode parameter-shift dashed, QNode backprop dotted).
- **Transparency and line width** = qubit count: pale and thin for 5 qubits,
  strong and thick for 12 qubits.
- The gap between a solid and a dash-dotted line of the same color is the
  overhead of the abstraction layer for that mode.

Notes on the plots in the thesis:

- Not every `gradient` line reaches 100,000 gates, because these runs were stopped
  manually: Executor (PennyLane backend) ends at 23,357 gates, raw Qiskit at
  23,357 and Executor (Qiskit backend) at 14,384. Raw PennyLane `gradient` and all
  `creation`/`execution` lines go up to 100,000 gates (tape vs. QNode: 10,000).
- The Executor `gradient` measurement takes the second call after an unmeasured
  warm-up. The one-time translation of the abstract circuit to PennyLane is paid
  by the warm-up and is therefore **not** included in these lines; they show the
  steady state.

The run folders each plot is built from are listed in the `RUNS` dict at the top
of its script. These are the folder names of the thesis runs, which are not in
this repository. To create a plot, run the benchmarks first and enter your own
folder names in `RUNS`; otherwise the script stops because the folders are missing.

## Setup and usage

Setup (once):

```bash
git submodule update --init --recursive   # downloads the executor into vendor/executor
pip install -r requirements.txt
pip install ./vendor/executor
```

Running a benchmark or plot script — these are just examples, every script in
`benchmarks/` and `plot_forms/` is started the same way:

```bash
python benchmarks/pennylane_benchmark.py --qubits 5                              # raw PennyLane, 5 qubits
python benchmarks/abstraction_benchmark.py --framework pennylane --qubits 5      # same, through the Executor
python benchmarks/qiskit_benchmark.py --qubits 5 8 --mode gradient --max-gates 1000
python plot_forms/plot_pennylane_vs_executor_multi_qubit.py                      # comparison plot
```

Several values for `--qubits` run one after another in the same process. To run
qubit counts in parallel, start the script once per qubit count.

### Results

Results are written to `raw_results/<Pennylane|Qiskit|Executor|TapeVsQNode>/`.
The plot scripts read the run folders listed in their `RUNS` dict and write to
`final_plotted_results/`. Both folders are created automatically if they do not
exist and are listed in `.gitignore`.

### Docker

Long runs can also be started in Docker:

```bash
./build_and_run.sh                                                  # pennylane_benchmark.py, default flags
./build_and_run.sh benchmarks/qiskit_benchmark.py --qubits 8 --mode gradient
./build_and_run.sh benchmarks/abstraction_benchmark.py --framework pennylane --qubits 5 8 10 12
```

The script and its flags are passed straight to `python` inside the container.
`benchmarks/` is mounted read-only, results are written to `raw_results/` on the host.
