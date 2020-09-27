#include "generator.h"

void Generator::generate_vector(
        int N, std::vector<double> &vector) {

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dist(0, 1);

    vector.assign(N, 0);

    for (int i = 0; i < N; ++i) {
        vector[i] = dist(gen);
    }
}

void Generator::generate_vector(
        int N, double *vector) {

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dist(0, 1);

    for (int i = 0; i < N; ++i) {
        vector[i] = dist(gen);
    }
}

void Generator::generate_matrix(
        int N, std::vector<std::vector<double> > &matrix) {

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dist(0, 1);

    matrix.assign(N, std::vector<double>(N, 0));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = dist(gen);
        }
    }
}

void Generator::generate_matrix(
        int N, double *matrix) {

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> dist(0, 1);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i * N + j] = dist(gen);
        }
    }
}
