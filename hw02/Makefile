CXX=g++
CXXFLAGS= -std=c++14 -O3

SRC=src
BIN=bin

HEADERS=${SRC}/generator.h ${SRC}/utils.h
SOURCES=${SRC}/benchmark.cpp ${SRC}/generator.cpp ${SRC}/utils.cpp

TARGET=${BIN}/benchmark

all: bindir build_benchmark

bindir:
	mkdir -p ${BIN}

build_benchmark: ${HEADERS} ${SOURCES}
	${CXX} ${CXXFLAGS} -o ${TARGET} ${SOURCES} -lcblas

run:
	bash run_benchmark.sh

clean:
	rm -r ${BIN}
