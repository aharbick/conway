CC=g++
NVCC=g++
CFLAGS=
LIBS=
BUILDDIR=build
TESTDIR=tests

# Test-specific variables
TEST_LIBS=$(shell pkg-config --cflags --libs gtest) -lgtest_main -lpthread -lcurl
TEST_SOURCES=$(wildcard $(TESTDIR)/*.cpp)
TEST_OBJECTS=$(TEST_SOURCES:$(TESTDIR)/%.cpp=$(BUILDDIR)/%.o)
TEST_EXECUTABLE=$(BUILDDIR)/run_tests

OS := $(shell uname -s)
HAS_CUDA := $(shell which nvcc)

ifeq ($(OS),Darwin)
	CFLAGS = -I/opt/homebrew/include
	LIBS = -L/opt/homebrew/lib -largp -lpthread -lcurl
else
	ifneq (,$(HAS_CUDA))
		NVCC=nvcc -x cu
		ifdef JETSON_NANO
			NVCFLAGS=-arch=sm_53 -w -O3 --ptxas-options=-v -lineinfo --default-stream per-thread
		else
			NVCFLAGS=-arch=sm_87 -w -O3 --ptxas-options=-v -lineinfo --default-stream per-thread
		endif
		LIBS=-L/usr/local/cuda/lib64 -lcudart -lcuda -lpthread -lcurl
	else
		LIBS=-lpthread -lcurl
	endif
endif

all: setup explore-cachability find-optimal

$(BUILDDIR)/find-optimal.o: find-optimal.cpp mt.h gol.h frame_utils.h types.h display_utils.h airtable_client.h
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

# Test targets
test: setup $(TEST_EXECUTABLE)
	$(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(TEST_OBJECTS)
	$(CC) $^ -o $@ $(TEST_LIBS)

$(BUILDDIR)/%.o: $(TESTDIR)/%.cpp
	$(CC) $(CFLAGS) -I. -c $< -o $@ $(shell pkg-config --cflags gtest)

.PHONY: test
