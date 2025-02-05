#include "log.hpp"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * 各个进程组成一个环，数据在环中传递
 */

int main(int argc, char **argv)
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int token;
    // Receive from the lower process and send to the higher process. Take care
    // of the special case when you are the first process to prevent deadlock.
    if (world_rank != 0) {
        MPI_Recv(&token, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        logm("Process %d received token %d from process %d", world_rank, token, world_rank - 1);
    } else {
        // Set the token's value if you are process 0
        token = 88;
    }
    MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size, 0, MPI_COMM_WORLD);
    // Now process 0 can receive from the last process. This makes sure that at
    // least one MPI_Send is initialized before all MPI_Recvs (again, to prevent
    // deadlock)
    if (world_rank == 0) {
        MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        logm("Process %d received token %d from process %d", world_rank, token, world_size - 1);
    }
    MPI_Finalize();
}

/*
mpirun --allow-run-as-root -np 4 ./bin/101SendRecvRing
输出：
2025/01/15 21:19:19.959707 1608010432 18936 [101SendRecvRing.cpp:main:25] Process 1 received token 88 from process 0 
2025/01/15 21:19:19.959862 3828824768 18937 [101SendRecvRing.cpp:main:25] Process 2 received token 88 from process 1 
2025/01/15 21:19:19.960010 1649375936 18938 [101SendRecvRing.cpp:main:25] Process 3 received token 88 from process 2 
2025/01/15 21:19:19.960106 828320448 18935 [101SendRecvRing.cpp:main:36] Process 0 received token 88 from process 3 
*/