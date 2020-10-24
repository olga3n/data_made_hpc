#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <stddef.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int rules[8];
    int ring_type = 0;
    int N = 100500;
    int iterations = 300;
    int verbose = 0;

    if (argc > 1)
    {
        for (int i = 0; i < 8; ++i)
        {
            rules[i] = 0;
        }

        FILE* input = fopen(argv[1], "r");

        for (int i = 0; i < 8; ++i) {
            int rule_in, rule_out;

            fscanf(input, "%d %d", &rule_in, &rule_out);

            int rule_in_code = (
                4 * ((rule_in - (rule_in % 100)) / 100) +
                2 * (((rule_in % 100) - (rule_in % 10)) / 10) +
                1 * (rule_in % 10)
            );

            rules[rule_in_code] = rule_out;
        }

        fclose(input);
    } else {
        printf(
            "Usage: %s rule_file [line/ring] [size] [iterations] [verbose]\n",
            argv[0]);

        return 1;
    }
 
    if (argc > 2 && strcmp(argv[2], "ring") == 0)
    {
        ring_type = 1;
    }

    if (argc > 3)
    {
        N = atoi(argv[3]);
    }
    
    if (argc > 4)
    {
        iterations = atoi(argv[4]);
    }

    if (argc > 5)
    {
        verbose = 1;
    }

    MPI_Init(NULL, NULL);

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int chunk_size;

    if (N % size)
    {
        chunk_size = N / size + 1;
    } else {
        chunk_size = N / size;
    }

    int current_chunk_size = chunk_size;

    if (rank == size - 1)
    {
        current_chunk_size = N - (size - 1) * chunk_size;
    }

    int buffer[current_chunk_size + 2];
    int next_buffer[current_chunk_size + 2];

    buffer[0] = 0;
    buffer[current_chunk_size + 1] = 0;

    unsigned int seed = (unsigned) time(NULL) ^ rank;

    for (int i = 1; i < current_chunk_size + 1; ++i)
    {
        if ((float)rand_r(&seed) / RAND_MAX > 0.5)
        {
            buffer[i] = 1;
        } else {
            buffer[i] = 0;
        }
    }

    int result[N];
    int recvcounts[size];
    int displs[size];

    for (int i = 0; i < size; ++i)
    {
        recvcounts[i] = (i < size - 1)?
            chunk_size: N - (size - 1) * chunk_size;

        displs[i] = i * chunk_size;
    }

    MPI_Request reqs[4];
    MPI_Status stats[4];

    for (int i = 0; i < iterations; ++i)
    {
        int left = rank - 1;
        int right = rank + 1;

        if (left < 0)
        {
            left = (ring_type)? size - 1: -1;
        }

        if (right >= size)
        {
            right = (ring_type)? 0: -1; 
        }

        int events = 0;
 
        if (left != -1)
        {
            MPI_Irecv(
                buffer, 1, MPI_INT, left, 0,
                MPI_COMM_WORLD, &reqs[events]);

            events += 1;
        } else {
            buffer[0] = 0;
        }

        if (right != -1)
        {
            MPI_Irecv(
                buffer + current_chunk_size + 1, 1, MPI_INT, right, 1,
                MPI_COMM_WORLD, &reqs[events]);

            events += 1;
        } else {
            buffer[current_chunk_size + 1] = 0;
        }

        if (left != -1)
        {
            MPI_Isend(
                buffer + 1, 1, MPI_INT, left, 1,
                MPI_COMM_WORLD, &reqs[events]);

            events += 1;
        }

        if (right != -1)
        {
            MPI_Isend(
                buffer + current_chunk_size, 1, MPI_INT, right, 0,
                MPI_COMM_WORLD, &reqs[events]);

            events += 1;
        }

        MPI_Waitall(events, reqs, stats);

        for (int j = 1; j < current_chunk_size + 1; ++j)
        {
            int rule_code = 4 * buffer[j - 1] + 2 * buffer[j] + buffer[j + 1];
            next_buffer[j] = rules[rule_code];
        }

        memcpy(buffer, next_buffer, (current_chunk_size + 2) * sizeof(int));

        if (verbose && i != iterations - 1)
        {
            MPI_Gatherv(
                buffer + 1, current_chunk_size, MPI_INT,
                result, recvcounts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                for (int j = 0; j < N; ++j)
                {
                    printf("%d", result[j]);
                }

                printf("\n");
            }
        }
    }

    MPI_Gatherv(
        buffer + 1, current_chunk_size, MPI_INT,
        result, recvcounts, displs, MPI_INT,
        0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        for (int i = 0; i < N; ++i)
        {
            printf("%d", result[i]);
        }

        printf("\n");
    }

    MPI_Finalize();

    return 0;
}
