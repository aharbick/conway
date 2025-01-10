CC=g++
NVCC=g++
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
		NVCC=nvcc -x cu -arch=sm_53
		NVCFLAGS=-w -O3 --ptxas-options=-v -lineinfo --default-stream per-thread
		ifdef JETSON
			NVCFLAGS += -DJETSON_NANO
		endif
		LIBS=-L/usr/local/cuda/lib64 -lcudart -lcuda -lpthread
	endif
endif

all: setup explore-cachability find-optimal

$(BUILDDIR)/find-optimal.o: find-optimal.cpp mt.h gol.h frame_utils.h types.h display_utils.h
	$(NVCC) $(if $(NVCFLAGS), $(NVCFLAGS), $(CFLAGS)) -I. -c $< -o $@

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
