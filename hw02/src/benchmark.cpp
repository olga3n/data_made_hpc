#include <iostream>
#include <cstring>
#include <vector>

#include "generator.h"
#include "utils.h"

int main(int argc, char **argv) {

    if (argc == 3) {

        std::string type = argv[1];

        int N = std::stoi(argv[2]);

        Generator generator;
        Utils utils;

        if (std::strcmp(argv[1], "matrix_matrix") == 0) {

            std::vector<std::vector<double> > matrix_a, matrix_b, result;

            generator.generate_matrix(N, matrix_a);
            generator.generate_matrix(N, matrix_b);

            utils.multiply_matrix_matrix(matrix_a, matrix_b, result);

        } else if (std::strcmp(argv[1], "matrix_vector") == 0) {

            std::vector<std::vector<double> > matrix;
            std::vector<double> vector, result;

            generator.generate_vector(N, vector);
            generator.generate_matrix(N, matrix);

            utils.multiply_matrix_vector(matrix, vector, result);

        } else if (std::strcmp(argv[1], "matrix_matrix_cblas") == 0) {

            double *A = new double[N * N];
            double *B = new double[N * N];
            double *C = new double[N * N];

            generator.generate_matrix(N, A);
            generator.generate_matrix(N, B);

            utils.multiply_matrix_matrix_cblas(A, B, C, N);

            delete[] A;
            delete[] B;
            delete[] C;

        } else if (std::strcmp(argv[1], "matrix_vector_cblas") == 0) {

            double *A = new double[N * N];
            double *B = new double[N];
            double *C = new double[N];

            generator.generate_matrix(N, A);
            generator.generate_vector(N, B);

            utils.multiply_matrix_vector_cblas(A, B, C, N);

            delete[] A;
            delete[] B;
            delete[] C;
        }
    }

    return 0;
}
