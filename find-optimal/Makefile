CC=g++
CFLAGS=-std=c++11
BUILDDIR=build

DEPS = mt.h #bloom.h
_OBJ = find-optimal.o mt.o #bloom.o
OBJ = $(patsubst %,$(BUILDDIR)/%,$(_OBJ))

all: setup find-optimal

$(BUILDDIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

find-optimal: $(OBJ)
	$(CC) -lpthread -o $(BUILDDIR)/$@ $^ $(CFLAGS)

setup:
	mkdir -p build

clean:
	rm -rf build *~ core
