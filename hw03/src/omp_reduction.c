#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int dotprod(int *a, int *b, size_t N)
{
    int i, tid;
    int sum = 0;

    #pragma omp parallel private(i, tid) shared(sum)
    {
        tid = omp_get_thread_num();

        #pragma omp for reduction(+:sum)
        for (i = 0; i < N; ++i)
        {
            sum += a[i] * b[i];
            printf("tid = %d i = %d\n", tid, i);
        }
    }

    return sum;
}

int main (int argc, char *argv[])
{
    const size_t N = 100;
    int i;

    int a[N], b[N];

    #pragma omp parallel for
    for (i = 0; i < N; ++i)
    {
        a[i] = b[i] = i;
    }

    int sum = dotprod(&a[0], &b[0], N);

    printf("Sum = %d\n", sum);

    return 0;
}
