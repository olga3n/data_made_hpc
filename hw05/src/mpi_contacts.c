#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <stddef.h>
#include <time.h>

typedef struct contact_message {
    int rank;
    int N;
    char hostname[MPI_MAX_PROCESSOR_NAME];
} contact;

int main(int argc, char *argv[])
{
    int N;

    MPI_Init(NULL, NULL);

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (argc > 1) {
        N = atoi(argv[1]);

        if (N > size - 1)
        {
            N = size - 1;
        }
    } else {
        N = size - 1;
    }

    MPI_Datatype mpi_contact_type;

    int nitems = 3;

    int blocklengths[3] = {
        1, 1, MPI_MAX_PROCESSOR_NAME
    };

    MPI_Aint offsets[3] = {
        offsetof(contact, rank),
        offsetof(contact, N),
        offsetof(contact, hostname)
    };

    MPI_Datatype types[3] = {
        MPI_INT, MPI_INT, MPI_CHAR
    };

    MPI_Type_create_struct(
        nitems, blocklengths, offsets, types, &mpi_contact_type);

    MPI_Type_commit(&mpi_contact_type);

    int tag = 1234;

    if (rank == 0)
    {
        int hostname_size;
        contact new_contact;

        new_contact.rank = rank;
        new_contact.N = N - 1;

        MPI_Get_processor_name(new_contact.hostname, &hostname_size);

        printf("Process %s, rank: %d\n",
            new_contact.hostname, new_contact.rank);

        if (N > 0)
        {
            int dest_lst_size = size - 1;
            int dest_lst[dest_lst_size];
            int curr_index = 0;

            for (int i = 1; i < size; ++i)
            {
                dest_lst[curr_index] = i;
                curr_index += 1;
            }

            unsigned int seed = (unsigned) time(NULL);
            const int dest = dest_lst[rand_r(&seed) % dest_lst_size];
            int cnt[1] = {1};

            MPI_Ssend(
                cnt, 1, MPI_INT,
                dest, tag, MPI_COMM_WORLD);

            MPI_Ssend(
                &new_contact, 1, mpi_contact_type,
                dest, tag, MPI_COMM_WORLD);
        }

    } else {
        int cnt[1];
        MPI_Status status;

        MPI_Recv(
            cnt, 1, MPI_INT,
            MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);

        int recv_contacts_size = cnt[0];

        contact contacts[recv_contacts_size + 1];

        MPI_Recv(
            contacts, recv_contacts_size, mpi_contact_type,
            MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);

        int hostname_size;

        contacts[recv_contacts_size].rank = rank;
        contacts[recv_contacts_size].N = contacts[recv_contacts_size - 1].N - 1;

        MPI_Get_processor_name(
            contacts[recv_contacts_size].hostname, &hostname_size);

        printf("\nRound: %d\n", N - contacts[recv_contacts_size - 1].N);

        printf("Process %s, rank: %d\n",
            contacts[recv_contacts_size].hostname,
            contacts[recv_contacts_size].rank);

        printf("Message from rank %d. Contacts: %d\n",
            status.MPI_SOURCE, recv_contacts_size);

        for (int i = 0; i < recv_contacts_size; ++i)
        {
            printf("  [Contact %d] Process %s, rank: %d\n",
                i, contacts[i].hostname, contacts[i].rank);
        }

        if (contacts[recv_contacts_size - 1].N > 0)
        {
            int dest_lst_size = size - cnt[0] - 1;
            int dest_lst[dest_lst_size];
            int curr_index = 0;

            for (int i = 0; i < size; ++i)
            {
                int flag = 0;
                
                for (int j = 0; j < recv_contacts_size; ++j)
                {
                    if (contacts[j].rank == i)
                    {
                        flag = 1;
                        break;
                    }
                }

                if (flag == 0 && rank != i)
                {
                    dest_lst[curr_index] = i;
                    curr_index += 1;
                }
            }

            unsigned int seed = (unsigned) time(NULL);
            const int dest = dest_lst[rand_r(&seed) % dest_lst_size];
            int next_cnt[1] = {recv_contacts_size + 1};

            MPI_Ssend(
                next_cnt, 1, MPI_INT,
                dest, tag, MPI_COMM_WORLD);

            MPI_Ssend(
                &contacts, recv_contacts_size + 1, mpi_contact_type,
                dest, tag, MPI_COMM_WORLD);
        }
    }

    MPI_Type_free(&mpi_contact_type);

    MPI_Finalize();

    return 0;
}
