PYTHON ?= python3

.PHONY: help build test benchmark clean

help:
	@echo "Quantized GEMM Kernels - Command Menu:"
	@echo "  make build      - Compile high-performance C++ extension"
	@echo "  make test       - Run all PyTest numerical parity and shape tests"
	@echo "  make benchmark  - Run Roofline latency and memory bandwidth profiler"
	@echo "  make clean      - Clean build artifacts"

build:
	$(PYTHON) setup.py build_ext --inplace

test:
	$(PYTHON) -m pytest tests/ -v

benchmark:
	$(PYTHON) benchmarks/benchmark_roofline.py

clean:
	rm -rf build *.so *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

