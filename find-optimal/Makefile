CC=g++
NVCC=g++
CFLAGS=
LIBS=
BUILDDIR=build
TESTDIR=tests
SRCDIR=src
INCLUDEDIR=include

# Test-specific variables
TEST_LIBS=$(shell pkg-config --cflags --libs gtest) -lgtest_main -lpthread -lcurl
TEST_SOURCES=$(wildcard $(TESTDIR)/*.cpp)
TEST_OBJECTS=$(TEST_SOURCES:$(TESTDIR)/%.cpp=$(BUILDDIR)/%.o)
TEST_EXECUTABLE=$(BUILDDIR)/run_tests

# Formatting variables
CLANG_FORMAT=clang-format
CPP_SOURCES=$(wildcard $(SRCDIR)/*.cpp)
HEADER_SOURCES=$(wildcard $(INCLUDEDIR)/*.h)
TEST_SOURCES_FORMAT=$(wildcard $(TESTDIR)/*.cpp)
ALL_SOURCES=$(CPP_SOURCES) $(HEADER_SOURCES) $(TEST_SOURCES_FORMAT)

OS := $(shell uname -s)
HAS_CUDA := $(shell which nvcc)

# Check for CUDA requirement
ifeq (,$(HAS_CUDA))
	$(error CUDA is required. Please install NVIDIA CUDA toolkit and ensure nvcc is in your PATH)
endif

# Set compiler to NVCC
NVCC=nvcc -x cu

# Auto-detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')

# Set CUDA compilation flags
ifdef JETSON_NANO
	NVCFLAGS=-arch=sm_53 -w -O3 --ptxas-options=-v -lineinfo --default-stream per-thread
else ifdef GPU_ARCH
	NVCFLAGS=-arch=sm_$(GPU_ARCH) -w -O3 --ptxas-options=-v -lineinfo --default-stream per-thread
else
	# Fallback for common architectures
	NVCFLAGS=-arch=sm_87 -w -O3 --ptxas-options=-v -lineinfo --default-stream per-thread
endif

# Set platform-specific flags
ifeq ($(OS),Darwin)
	CFLAGS = -I/opt/homebrew/include
	LIBS = -L/opt/homebrew/lib -largp -lpthread -lcurl -lssl -lcrypto -L/usr/local/cuda/lib64 -lcudart -lcuda
else
	LIBS = -L/usr/local/cuda/lib64 -lcudart -lcuda -lpthread -lcurl -lssl -lcrypto
endif

all: setup format-check explore-cachability find-optimal

$(BUILDDIR)/cli_parser.o: $(SRCDIR)/cli_parser.cpp $(INCLUDEDIR)/cli_parser.h
	$(CC) $(CFLAGS) -I$(INCLUDEDIR) -c $< -o $@


$(BUILDDIR)/utils.o: $(SRCDIR)/utils.cpp $(INCLUDEDIR)/utils.h
	$(CC) $(CFLAGS) -I$(INCLUDEDIR) -c $< -o $@

$(BUILDDIR)/logging.o: $(SRCDIR)/logging.cpp $(INCLUDEDIR)/logging.h
	$(CC) $(CFLAGS) -I$(INCLUDEDIR) -c $< -o $@

$(BUILDDIR)/gol.o: $(SRCDIR)/gol.cpp $(INCLUDEDIR)/gol.h
	$(NVCC) $(if $(NVCFLAGS), $(NVCFLAGS), $(CFLAGS)) -I$(INCLUDEDIR) -c $< -o $@

$(BUILDDIR)/find-optimal.o: $(SRCDIR)/find-optimal.cpp $(INCLUDEDIR)/gol.h
	$(NVCC) $(if $(NVCFLAGS), $(NVCFLAGS), $(CFLAGS)) -I$(INCLUDEDIR) -c $< -o $@

# CUDA-specific implementations
$(BUILDDIR)/gol_cuda.o: $(SRCDIR)/gol_cuda.cu $(INCLUDEDIR)/gol.h
	$(NVCC) $(NVCFLAGS) -I$(INCLUDEDIR) -c $< -o $@

CUDA_OBJECTS = $(BUILDDIR)/gol_cuda.o

find-optimal: $(BUILDDIR)/find-optimal.o $(BUILDDIR)/cli_parser.o $(BUILDDIR)/gol.o $(BUILDDIR)/utils.o $(BUILDDIR)/logging.o $(CUDA_OBJECTS)
	$(CC) $^ -o $(BUILDDIR)/$@ $(LIBS)

$(BUILDDIR)/explore-cachability.o: $(SRCDIR)/explore-cachability.cpp
	$(CC) $(CFLAGS) -I$(INCLUDEDIR) -c $< -o $@

explore-cachability: $(BUILDDIR)/explore-cachability.o
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS)

$(BUILDDIR)/validate_frame_search_completeness.o: $(SRCDIR)/validate_frame_search_completeness.cpp $(INCLUDEDIR)/frame_utils.h $(INCLUDEDIR)/gol_core.h
	$(CC) $(CFLAGS) -I$(INCLUDEDIR) -c $< -o $@

validate-frame-search-completeness: $(BUILDDIR)/validate_frame_search_completeness.o
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS)

setup:
	mkdir -p build

clean:
	rm -rf build *~ core

# Format targets
format-check: $(ALL_SOURCES)
	@echo "Checking if formatting is needed..."
	@$(CLANG_FORMAT) --dry-run --Werror $(ALL_SOURCES) && echo "✓ Code is properly formatted" || (echo "✗ Code needs formatting - run 'make format-fix'" && exit 1)

format-fix: $(ALL_SOURCES)
	@echo "Formatting code..."
	@$(CLANG_FORMAT) -i $(ALL_SOURCES)
	@echo "✓ Code formatted"

# Test targets
test: setup $(TEST_EXECUTABLE)
	$(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(TEST_OBJECTS)
	$(CC) $^ -o $@ $(TEST_LIBS)

$(BUILDDIR)/%.o: $(TESTDIR)/%.cpp
	$(CC) $(CFLAGS) -I$(INCLUDEDIR) -c $< -o $@ $(shell pkg-config --cflags gtest)

.PHONY: test format-fix format-check
