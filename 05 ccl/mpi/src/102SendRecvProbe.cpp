#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2) {
        fprintf(stderr, "Must use two processes for this example\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int number_amount;
    if (world_rank == 0) {
        const int MAX_NUMBERS = 100;
        int numbers[MAX_NUMBERS];
        // Pick a random amont of integers to send to process one
        srand(time(NULL));
        number_amount = (rand() / (float) RAND_MAX) * MAX_NUMBERS;
        // Send the amount of integers to process one
        MPI_Send(numbers, number_amount, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("rank0 sent %d numbers to rank1\n", number_amount);
    } else if (world_rank == 1) {
        MPI_Status status;
        // Probe for an incoming message from process zero
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);  // 参数为: src, tag, comm, status
        // 在 MPI_Get_count 函数中，使用者需要传递 MPI_Status 结构体，消息的datatype（数据类型），并返回 count。 变量
        // count 是已接收的 datatype 元素的数目。
        MPI_Get_count(&status, MPI_INT, &number_amount);
        // Allocate a buffer just big enough to hold the incoming numbers
        int *number_buf = (int *) malloc(sizeof(int) * number_amount);
        // Now receive the message with the allocated buffer
        MPI_Recv(number_buf, number_amount, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("rank1 dynamically received %d numbers from rank0.\n", number_amount);
        free(number_buf);
    }
    MPI_Finalize();
}

/*
mpirun --allow-run-as-root -np 2 ./bin/102SendRecvProbe
输出：
rank0 sent 45 numbers to rank1
rank1 dynamically received 45 numbers from rank0.
*/