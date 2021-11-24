CC=g++
NVCC=nvcc
NVCFLAGS=-w --ptxas-options=-v -O3
CFLAGS=-std=c++11
BUILDDIR=build

DEPS = mt.h
_OBJ = find-optimal.o mt.o
OBJ = $(patsubst %,$(BUILDDIR)/%,$(_OBJ))

all: setup find-optimal

$(BUILDDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

find-optimal: $(OBJ)
	$(CC) -lpthread -o $(BUILDDIR)/$@ $^ $(CFLAGS)

$(BUILDDIR)/%.o: %.cu
	$(NVCC) -c -o $@ $< $(NVCFLAGS)

find-cuda-optimal: $(OBJ) $(BUILDDIR)/gol.o
	$(CC) -lpthread -o $(BUILDDIR)/$@ $^ $(CFLAGS)

setup:
	mkdir -p build

clean:
	rm -rf build *~ core
