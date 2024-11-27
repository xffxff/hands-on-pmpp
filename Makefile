# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -arch=sm_90

# Source files with correct path
SOURCES = parallel_scan/brent_kung_scan.cu parallel_scan/kogge_stone_scan.cu

# Generate executable names by replacing .cu with nothing
EXECUTABLES = $(basename $(notdir $(SOURCES)))

# Default target
all: $(EXECUTABLES)

# Rule to build each executable
%: parallel_scan/%.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Clean target
clean:
	rm -f $(EXECUTABLES)

# Help target
help:
	@echo "Available targets:"
	@echo "  all    - Build all kernels (default)"
	@echo "  clean  - Remove all built executables"
	@echo "  help   - Show this help message"
	@echo ""
	@echo "Individual targets:"
	@echo "  brent_kung_scan"
	@echo "  kogge_stone_scan" 