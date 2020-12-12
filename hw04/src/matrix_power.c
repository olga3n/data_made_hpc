#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void matrix_multiplication(double *A, double *B, double *C, int N)
{
    int i, j, k;

    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            C[i * N + j] = 0;

            for (k = 0; k < N; ++k)
            {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void matrix_power(double *matrix, double *result, int N, int power)
{
    int max_power_bit = 0;

    while ((power >> max_power_bit) > 0)
    {
        max_power_bit += 1;
    }

    double *matrix_powers[max_power_bit];

    int i;

    for (i = 0; i < max_power_bit; ++i) {
        matrix_powers[i] = (double *) malloc(N * N * sizeof(double));
    }

    memcpy(matrix_powers[0], matrix, N * N * sizeof(double));

    for (i = 1; i < max_power_bit; ++i) {
        matrix_multiplication(
            matrix_powers[i - 1], matrix_powers[i - 1], matrix_powers[i], N);
    }

    double *buffer = (double *) malloc(N * N * sizeof(double));

    memcpy(result, matrix_powers[max_power_bit - 1], N * N * sizeof(double));

    for (i = 1; i < max_power_bit; ++i) {
        int current_bit_ind = max_power_bit - i - 1;

        if (power & (1 << current_bit_ind) != 0) {
            matrix_multiplication(
                result, matrix_powers[current_bit_ind], buffer, N);

            memcpy(result, buffer, N * N * sizeof(double));
        }
    }

    free(buffer);

    for (i = 0; i < max_power_bit; ++i) {
        free(matrix_powers[i]);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2 || atoi(argv[1]) < 1) {
        return 0;
    }

    int power = atoi(argv[1]);

    int i, j, N;

    scanf("%d", &N);

    double *matrix = (double *) malloc(N * N * sizeof(double));

    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            scanf("%lf", &matrix[i * N + j]);
        }
    }

    double *result = (double *) malloc(N * N * sizeof(double));

    matrix_power(matrix, result, N, power);

    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            printf("%.0lf ", result[i * N + j]);
        }

        printf("\n");
    }

    free(matrix);
    free(result);

    return 0;
}
