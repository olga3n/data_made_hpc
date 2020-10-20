#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
    const size_t N = 100;
    const size_t chunk = 3;

    int i, tid;
    int a[N], b[N], c[N];

    #pragma omp parallel private(i, tid) shared(a, b, c, chunk)
    {
        #pragma omp parallel for schedule(static, chunk)
        for (i = 0; i < N; ++i)
        {
            a[i] = b[i] = i;
        }

        tid = omp_get_thread_num();

        #pragma omp for schedule(static, chunk)
        for (i = 0; i < N; ++i)
        {
            c[i] = a[i] + b[i];
            printf("tid = %d, c[%d] = %d\n", tid, i, c[i]);
        }
    } 

    return 0;
}
