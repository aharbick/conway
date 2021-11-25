CC=g++
NVCC=nvcc
NVCFLAGS=-w --ptxas-options=-v -O3 -arch sm_53
NVLIBS=-L/usr/local/cuda/lib64 -lcudart -lcuda
CFLAGS=-std=c++11
BUILDDIR=build

DEPS = mt.h
_OBJ = find-optimal.o mt.o
OBJ = $(patsubst %,$(BUILDDIR)/%,$(_OBJ))
_CUDA_OBJ = find-cuda-optimal.o mt.o
CUDA_OBJ = $(patsubst %,$(BUILDDIR)/%,$(_CUDA_OBJ))

all: setup find-optimal find-cuda-optimal

$(BUILDDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

find-optimal: $(OBJ)
	$(CC) -pthread -o $(BUILDDIR)/$@ $^ $(CFLAGS)

$(BUILDDIR)/%.o: %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(NVCFLAGS)

find-cuda-optimal: $(CUDA_OBJ)
	$(CC) -o $(BUILDDIR)/$@ $^ $(NVLIBS)

setup:
	mkdir -p build

clean:
	rm -rf build *~ core
