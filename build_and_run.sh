# Aufruf:
#   ./build_and_run.sh                                              # pennylane_benchmark.py mit Standard-Flags
#   ./build_and_run.sh benchmarks/qiskit_benchmark.py --qubits 8 --mode gradient
#   ./build_and_run.sh benchmarks/abstraction_benchmark.py --framework pennylane --qubits 5 8 10 12
# Alle Argumente werden unveraendert an python im Container weitergereicht.

IMAGE="$(whoami)/abstraction-benchmark:0.1.0"

if [ $# -eq 0 ]; then
	set -- benchmarks/pennylane_benchmark.py
fi

if [ --build ]; then
	git submodule update --init --recursive
	docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USERNAME=$(whoami) -f Dockerfile -t "$IMAGE" .
fi

N_CPUS=16

# Vorher anlegen: fehlt der Ordner, erstellt Docker ihn als root, und der
# Container-User koennte dort nicht schreiben.
mkdir -p raw_results

docker run -d --rm \
    -v "$(pwd)/benchmarks:/app/benchmarks:ro" \
    -v "$(pwd)/raw_results:/app/raw_results" \
    --shm-size=2g \
    --cpus $N_CPUS \
    --memory 128g \
    -e OMP_NUM_THREADS=$N_CPUS \
    -e MKL_NUM_THREADS=$N_CPUS \
    -e OPENBLAS_NUM_THREADS=$N_CPUS \
    -e NUMEXPR_NUM_THREADS=$N_CPUS \
    -e MPLBACKEND=Agg \
    --name "$(whoami)_abstraction_benchmark" \
    "$IMAGE" \
    "$@"
