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
		CC=nvcc -arch=sm_53
		CFLAGS=-w -O3 --ptxas-options=-v -lineinfo
		LIBS=-L/usr/local/cuda/lib64 -lcudart -lcuda -lpthread
	endif
endif

all: setup find-optimal

$(BUILDDIR)/find-optimal.o: find-optimal.cpp mt.h gol.h
	$(CC) -x cu $(CFLAGS) -I. -dc $< -o $@

find-optimal: $(BUILDDIR)/find-optimal.o
	$(CC) $(CFLAGS) $^ -o $(BUILDDIR)/$@ $(LIBS)

explore-cachability: $(BUILDDIR)/explore-cachability.o
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS)

setup:
	mkdir -p build

clean:
	rm -rf build *~ core
