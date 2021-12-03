CC=g++
CFLAGS=
LIBS=
BUILDDIR=build

OS := $(shell uname -s)
HAS_CUDA := $(shell which nvcc)

ifeq ($(OS),Darwin)
	CFLAGS = -I/opt/homebrew/include
	LIBS = -L/opt/homebrew/lib -largp
else
	ifneq (,$(HAS_CUDA))
		CC=nvcc
		CFLAGS=-w --ptxas-options=-v -O3 -arch sm_53
		LIBS=-L/usr/local/cuda/lib64 -lcudart -lcuda -pthread
	endif
endif

_OBJ = find-optimal.o gol.o utils.o
OBJ = $(patsubst %,$(BUILDDIR)/%,$(_OBJ))
DEPS=mt.h cli.h utils.h

all: setup explore-cachability find-optimal find-cuda-optimal

$(BUILDDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BUILDDIR)/%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

find-optimal: $(OBJ)
	$(CC) -pthread -o $(BUILDDIR)/$@ $^ $(CFLAGS) $(LIBS)

explore-cachability: $(BUILDDIR)/explore-cachability.o
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS)

setup:
	mkdir -p build

clean:
	rm -rf build *~ core
