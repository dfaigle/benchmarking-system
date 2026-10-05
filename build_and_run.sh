if [ --build ]; then
	git submodule update --init --recursive
	docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USERNAME=$(whoami) -f Dockerfile -t "$(whoami)/abstraction-benchmark:0.1.0" .
fi

N_CPUS=16

docker run -d --rm \
    -v "$(pwd)/benchmarks/pennylane/logarithmic_benchmark_pennylane_5.py:/app/benchmarks/pennylane/logarithmic_benchmark_pennylane_5.py:ro" \
    -v "$(pwd)/raw_results:/app/raw_results" \
    --shm-size=2g \
    --cpus $N_CPUS \
    --memory 128g \
    -e OMP_NUM_THREADS=$N_CPUS \
    -e MKL_NUM_THREADS=$N_CPUS \
    -e OPENBLAS_NUM_THREADS=$N_CPUS \
    -e NUMEXPR_NUM_THREADS=$N_CPUS \
    --name "$(whoami)_abstraction_benchmark" \
    mow/abstraction-benchmark:0.1.0 \
    benchmarks/pennylane/logarithmic_benchmark_pennylane_5.py

