#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_gen(int *matrix, int N, float edge_prob)
{
    int i, j;

    unsigned int seed = (unsigned) time(NULL);

    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            if ((float)rand_r(&seed) / RAND_MAX < edge_prob) {
                matrix[i * N + j] = 1;
            } else {
                matrix[i * N + j] = 0;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3 || atoi(argv[1]) < 1) {
        return 0;
    }

    int N = atoi(argv[1]);
    float edge_prob = atof(argv[2]);

    printf("%d\n", N);

    int *graph = (int *) malloc(N * N * sizeof(int));

    matrix_gen(graph, N, edge_prob);

    int i, j;

    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            printf("%d ", graph[i * N + j]);
        }

        printf("\n");
    }

    free(graph);

    return 0;
}
