CC=g++
CFLAGS=-std=c++11
BUILDDIR=build

DEPS = #bloom.h
_OBJ = find-optimal.o #bloom.o
OBJ = $(patsubst %,$(BUILDDIR)/%,$(_OBJ))

all: setup find-optimal find-optimal-random

$(BUILDDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

find-optimal: $(OBJ)
	$(CC) -o $(BUILDDIR)/$@ $^ $(CFLAGS)

find-optimal-random: $(OBJ)
	$(CC) -DRANDOM -o $(BUILDDIR)/$@ $^ $(CFLAGS)

setup:
	mkdir -p build

clean:
	rm -f build/*.o *~ core
