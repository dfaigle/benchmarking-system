if [ --build ]; then
	docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USERNAME=$(whoami) -f Dockerfile -t "$(whoami)/abstraction-benchmark:0.1.0" .
fi

N_CPUS=16

docker run -d --rm \
    -v "$(pwd)/Benchmarks/logarithmic_benchmark_pennylane.py:/app/logarithmic_benchmark_pennylane.py:ro" \
    -v "$(pwd)/Results:/results" \
    --shm-size=2g \
    --cpus $N_CPUS \
    --memory 128g \
    -e OMP_NUM_THREADS=$N_CPUS \
    -e MKL_NUM_THREADS=$N_CPUS \
    -e OPENBLAS_NUM_THREADS=$N_CPUS \
    -e NUMEXPR_NUM_THREADS=$N_CPUS \
    --name "$(whoami)_abstraction_benchmark" \
    mow/abstraction-benchmark:0.1.0 \
    logarithmic_benchmark_pennylane.py

