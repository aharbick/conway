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
		CFLAGS=-w -O3 --ptxas-options=-v -lineinfo -arch sm_53 -DHAS_CUDA
		LIBS=-L/usr/local/cuda/lib64 -lcudart -lcuda -lpthread
	endif
endif

DEPS=mt.h cli.h utils.h gol.h
OBJ = $(BUILDDIR)/find-optimal.o $(BUILDDIR)/utils.o

all: setup explore-cachability find-optimal

$(BUILDDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

find-optimal: $(OBJ)
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS) $(LIBS)

explore-cachability: $(BUILDDIR)/explore-cachability.o
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS)

setup:
	mkdir -p build

clean:
	rm -rf build *~ core
