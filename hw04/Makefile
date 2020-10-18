CXX=gcc
CXXFLAGS= -fopenmp -O3 -lcblas

SRC=src
BIN=bin

SRC_FILES=$(wildcard $(SRC)/*.c)
TARGETS=$(patsubst $(SRC)/%.c,$(BIN)/%.o,$(SRC_FILES))

all: $(TARGETS)

$(BIN):
	mkdir -p $(BIN)

$(TARGETS): | $(BIN)

$(BIN)/%.o: $(SRC)/%.c
	$(CXX) -o $@ $< $(CXXFLAGS)

clean:
	rm -r $(BIN)

test_gen: $(BIN)/gen_random_graph.o
	./$(BIN)/gen_random_graph.o 10 0.3

test_power: $(BIN)/matrix_power.o
	./$(BIN)/matrix_power.o 10 < data/wiki_article_graph.txt

test_pagerank: $(BIN)/matrix_pagerank.o
	./$(BIN)/matrix_pagerank.o < data/wiki_article_graph.txt
