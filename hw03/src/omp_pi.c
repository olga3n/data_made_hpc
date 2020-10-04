#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float pi_calculation(size_t N)
{
    int i, tid, seed;
    int hits = 0;
    unsigned int dt = (unsigned) time(NULL);

    #pragma omp parallel private(i, tid, seed) shared(hits)
    {
        tid = omp_get_thread_num();
        seed = dt ^ tid;

        #pragma omp for
        for (int i = 0; i < N; ++i)
        {
            float x = (float)rand_r(&seed) / RAND_MAX * 2 - 1;
            float y = (float)rand_r(&seed) / RAND_MAX * 2 - 1;

            if (x * x + y * y <= 1) {
                #pragma omp atomic
                hits += 1;
            }
        }
    }

    float pi = 4 * (float)hits / N;

    return pi;
}

int main (int argc, char *argv[])
{
    const size_t N = 1e7;

    float pi = pi_calculation(N);

    printf("%f\n", pi);

    return 0;
}
