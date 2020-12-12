#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void matrix_vector_multiplication(double *A, double *B, double *C, int N)
{
    int i, j;

    for (i = 0; i < N; ++i)
    {
        C[i] = 0;

        for (j = 0; j < N; ++j)
        {
            C[i] += A[i * N + j] * B[j];
        }
    }
}

double calc_l1_dist(double *A, double *B, int N)
{
    double result = 0;

    int i;

    for (i = 0; i < N; ++i)
    {
        result += fabs(A[i] - B[i]);
    }

    return result;
}

void matrix_pagerank(
    double *matrix, double *result, int N,
    double eps, double damping, int max_iter)
{
    double *fixed_matrix = (double *) malloc(N * N * sizeof(double));

    int i, j;

    for (i = 0; i < N; ++i)
    {
        double row_sum = 0;

        for (j = 0; j < N; ++j)
        {
            row_sum += matrix[i * N + j];
        }

        for (j = 0; j < N; ++j)
        {
            if (row_sum != 0)
            {
                fixed_matrix[j * N + i] = matrix[i * N + j] / row_sum;
            }
            else
            {
                fixed_matrix[j * N + i] = 0;
            }
        }
    }

    double *curr_vector = (double *) malloc(N * sizeof(double));

    for (i = 0; i < N; ++i)
    {
        curr_vector[i] = 1 / N;
    }

    for (i = 0; i < max_iter; ++i)
    {
        matrix_vector_multiplication(fixed_matrix, curr_vector, result, N);

        for (j = 0; j < N; ++j)
        {
            result[j] = damping * result[j] + (1 - damping) / N;
        }

        double l1_dist = calc_l1_dist(curr_vector, result, N);

        if (l1_dist < eps) {
            break;
        }

        memcpy(curr_vector, result, N * sizeof(double));
    }

    free(fixed_matrix);
    free(curr_vector);
}

void calc_freq(double *matrix, double *freq, int N)
{
    int i, j;

    double sum_freq = 0;

    for (i = 0; i < N; ++i)
    {
        freq[i] = 0;

        for (j = 0; j < N; ++j)
        {
            freq[i] += matrix[j * N + i];
        }

        sum_freq += freq[i];
    }

    for (i = 0; i < N; ++i)
    {
        freq[i] /= sum_freq;
    }
}

int main(int argc, char *argv[])
{
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

    double *result = (double *) malloc(N * sizeof(double));

    double eps = 1e-6;
    double damping = 0.85;
    int max_iter = 1e6;

    matrix_pagerank(matrix, result, N, eps, damping, max_iter);

    double *freq = (double *) malloc(N * sizeof(double));

    calc_freq(matrix, freq, N);

    for (i = 0; i < N; ++i)
    {
        printf("%2d: pagerank=%lf, freq=%lf\n", i, result[i], freq[i]);
    }

    free(matrix);
    free(freq);
    free(result);

    return 0;
}
