CXX=gcc
CXXFLAGS= -fopenmp -O3 -lcblas

SRC=src
BIN=bin

SRC_FILES=$(wildcard $(SRC)/*.c)
TARGETS=$(patsubst $(SRC)/%.c,$(BIN)/%,$(SRC_FILES))

all: $(TARGETS)

$(BIN):
	mkdir -p $(BIN)

$(TARGETS): | $(BIN)

$(BIN)/%: $(SRC)/%.c
	$(CXX) -o $@ $< $(CXXFLAGS)

clean:
	rm -r $(BIN)

test_gen: $(BIN)/gen_random_graph
	./$(BIN)/gen_random_graph 10 0.3

test_power: $(BIN)/matrix_power
	./$(BIN)/matrix_power 10 < data/wiki_article_graph.txt

test_pagerank: $(BIN)/matrix_pagerank
	./$(BIN)/matrix_pagerank < data/wiki_article_graph.txt
