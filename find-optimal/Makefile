CC=g++
CFLAGS=
LIBS=
BUILDDIR=build

OS := $(shell uname -s)
HAS_CUDA := $(shell which nvcc)

ifeq ($(OS),Darwin)
	CFLAGS = -I/opt/homebrew/include
	LIBS = -L/opt/homebrew/lib -largp -lpthread
else
	ifneq (,$(HAS_CUDA))
		CC=nvcc
		CFLAGS=-w --ptxas-options=-v -O3 -arch sm_53 -DHAS_CUDA
		LIBS=-L/usr/local/cuda/lib64 -lcudart -lcuda -lpthread
	endif
endif

DEPS=mt.h cli.h utils.h gol.h
OBJ = $(BUILDDIR)/find-optimal.o $(BUILDDIR)/gol.o $(BUILDDIR)/utils.o

CUDA_DEPS=mt.h cli.h utils.h gol_cuda.h
CUDA_OBJ = $(BUILDDIR)/find-optimal.o $(BUILDDIR)/gol_cuda.o $(BUILDDIR)/utils.o

all: setup explore-cachability find-optimal $(if $(HAS_CUDA), find-cuda-optimal)

$(BUILDDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/%.o: %.cu $(CUDA_DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

find-optimal: $(OBJ)
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS) $(LIBS)

find-cuda-optimal: $(CUDA_OBJ)
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS) $(LIBS)

explore-cachability: $(BUILDDIR)/explore-cachability.o
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS)

setup:
	mkdir -p build

clean:
	rm -rf build *~ core
