CC=mpicc
CFLAGS= -std=c99

SRC=src
BIN=bin

SRC_FILES=$(wildcard $(SRC)/*.c)
TARGETS=$(patsubst $(SRC)/%.c,$(BIN)/%,$(SRC_FILES))

all: $(TARGETS)

$(BIN):
	mkdir -p $(BIN)

$(TARGETS): | $(BIN)

$(BIN)/%: $(SRC)/%.c
	$(CC) -o $@ $< $(CFLAGS)

clean:
	rm -r $(BIN)
