#include "utils.h"

void Utils::multiply_matrix_matrix(
        double *A, double *B, double *C, int N) {

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {

            C[i * N + j] = 0;

            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void Utils::multiply_matrix_vector(
        double *A, double *B, double *C, int N) {

    for (int i = 0; i < N; ++i) {

        C[i] = 0;

        for (int j = 0; j < N; ++j) {
            C[i] += A[i * N + j] * B[j];
        }
    }
}

void Utils::multiply_matrix_matrix(
        std::vector<std::vector<double> > &matrix_a,
        std::vector<std::vector<double> > &matrix_b,
        std::vector<std::vector<double> > &result) {

    int N = matrix_a.size();

    result.assign(N, std::vector<double>(N, 0));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                result[i][j] += matrix_a[i][k] * matrix_b[k][j];
            }
        }
    }
}

void Utils::multiply_matrix_vector(
        std::vector<std::vector<double> > &matrix,
        std::vector<double> &vector,
        std::vector<double> &result){

    int N = matrix.size();

    result.assign(N, 0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void Utils::multiply_matrix_matrix_cblas(
        double *A, double *B, double *C, int N) {

    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
        1.0, A, N, B, N, 0.0, C, N);
}

void Utils::multiply_matrix_vector_cblas(
        double *A, double *B, double *C, int N) {

    cblas_dgemv(
        CblasRowMajor, CblasNoTrans, N, N,
        1.0, A, N, B, 1, 0.0, C, 1);
}
