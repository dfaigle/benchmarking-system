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
  pennylane/               raw PennyLane benchmarks (gradient: backprop)
    qnode_vs_tape/         PennyLane QNode vs. tape (side comparison)
  qiskit/                  raw Qiskit benchmarks (gradient: parameter-shift)
  abstract/                same benchmarks through the Executor (PennyLane or Qiskit backend)
  plot_forms/              scripts that combine several runs into comparison plots
raw_results/               CSVs + per-run plots, one folder per run (<timestamp>_qubits-<n>)
final_plotted_results/     final comparison plots, these are the plots used in the thesis
vendor/executor/           qc-executor as git submodule
```

Each benchmark script exists once per qubit count (`_5`, `_8`, `_10`, `_12`).
Gate set (`GATE_SET_CHOICE`), mode (`BENCHMARK_MODE`) and, for the executor,
the backend (`BACKEND_CHOICE`) are set at the top of each script.

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
python benchmarks/pennylane/logarithmic_benchmark_pennylane_5.py              # raw PennyLane, 5 qubits
python benchmarks/abstract/logarithmic_benchmark_abstraction_pennylane_5.py   # same, through the Executor
python benchmarks/plot_forms/plot_pennylane_vs_executor_multi_qubit.py        # comparison plot
```

Results are written to `raw_results/<Pennylane|Qiskit|Executor|TapeVsQNode>/`.
The plot scripts read the run folders listed in their `RUNS` dict and write to
`final_plotted_results/`.

Long runs can also be started in Docker:

```bash
./build_and_run.sh
```

It runs `benchmarks/pennylane/logarithmic_benchmark_pennylane_5.py`. For another
benchmark, change that path in `build_and_run.sh`.
