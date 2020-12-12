CXX=gcc
CXXFLAGS= -fopenmp -O3

SRC=src
BIN=bin

SRC_FILES=$(wildcard $(SRC)/*.c)
TARGETS=$(patsubst $(SRC)/%.c,$(BIN)/%,$(SRC_FILES))

all: bindir $(TARGETS)

bindir:
	mkdir -p $(BIN)

$(BIN)/%: $(SRC)/%.c
	$(CXX) -o $@ $< $(CXXFLAGS)

clean:
	rm -r $(BIN)
