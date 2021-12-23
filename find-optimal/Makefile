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

all: setup explore-cachability find-optimal

$(BUILDDIR)/find-optimal.o: find-optimal.cpp mt.h gol.h
	$(CC) -x cu $(CFLAGS) -I. -c $< -o $@

find-optimal: $(BUILDDIR)/find-optimal.o
	$(CC) $^ -o $(BUILDDIR)/$@ $(LIBS)

$(BUILDDIR)/explore-cachability.o: explore-cachability.cpp mt.h
	$(CC) $(CFLAGS) -I. -c $< -o $@

explore-cachability: $(BUILDDIR)/explore-cachability.o
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS)

setup:
	mkdir -p build

clean:
	rm -rf build *~ core
