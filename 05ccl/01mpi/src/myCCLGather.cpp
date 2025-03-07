#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int myGather(const void *sendBuff,
             void *recvBuff,
             size_t sendCount,
             MPI_Datatype dataType,
             int root,
             MPI_Comm comm,
             void *stream)
{
    int rank, rankNum, typeSize;
    const int tag = 100;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &rankNum);
    MPI_Type_size(dataType, &typeSize);
    std::vector<MPI_Request> recvReqs(rankNum);
    MPI_Request sendReq;

    // 根进程接收数据
    if (rank == root)
    {
        for (int i = 0; i < rankNum; i++)
        {
            void *curRecvBuff = (uint8_t *)recvBuff + typeSize * sendCount * i;
            MPI_Irecv(curRecvBuff, sendCount, dataType, i, tag, comm, &recvReqs[i]);
        }
    }

    // 所有进程发送数据
    MPI_Isend(sendBuff, sendCount, dataType, root, tag, comm, &sendReq);

    // 根进程等待所有接收操作完成
    if (rank == root)  
    {
        MPI_Waitall(rankNum, recvReqs.data(), MPI_STATUSES_IGNORE);
    }
    // 所有进程等待发送操作完成
    MPI_Wait(&sendReq, MPI_STATUS_IGNORE);

    // delete[] recvReqs;

    return 0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, rankNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankNum);

    int root      = 0;
    int sendCount = 2;  // 元素数量
    int recvCount = sendCount * rankNum;
    int *sendData = new int[sendCount];
    int *recvData = new int[recvCount];

    memset(recvData, -1, recvCount);
    for (int i = 0; i < sendCount; ++i)
    {
        sendData[i] = rank;
    }

    myGather(sendData, recvData, sendCount, MPI_INT, root, MPI_COMM_WORLD, nullptr);

    // 根进程打印收集到的数据
    if (rank == root)
    {
        printf("Collected data at root:\n");
        for (int i = 0; i < recvCount; i++)
        {
            printf("%d, ", recvData[i]);
        }
        printf("\n");
    }

    delete[] sendData;
    delete[] recvData;

    MPI_Finalize();
    return 0;
}