#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "log.hpp"

#define sliceSize (8)

// ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype, int root,
// ncclComm_t comm, lynStream_t stream);

// 假设不是 in-place，最后一个节点还要把数据传回来。因为在某国产硬件里，同芯片的拷贝是很慢的，不如借助其它节点传回来。
int myBroadcastRing(const void *sendBuff,
                    void *recvBuff,
                    size_t count,
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

    if (1 == rankNum)
    {
        return 0;
    }

    const int prevRank = (rank - 1 + rankNum) % rankNum;
    const int nextRank = (rank + 1 + rankNum) % rankNum;
    const int lastRank = (root - 1 + rankNum) % rankNum;

    int totalSize  = count * typeSize;
    int bufferNum  = (totalSize + sliceSize - 1) / sliceSize;
    int bufferSize = sliceSize;
    logd("bufferNum = %d, bufferSize = %d", bufferNum, bufferSize);

    for (int i = 0; i < bufferNum; ++i)
    {
        int offset              = i * bufferSize;
        int remainSize          = totalSize - offset;
        int currentSize         = remainSize > bufferSize ? bufferSize : remainSize;
        uint8_t *currentSendBuf = rank == root ? (uint8_t *)sendBuff + offset : (uint8_t *)recvBuff + offset;
        uint8_t *currentRecvBuf = (uint8_t *)recvBuff + offset;

        if (rank == root)
        {
            MPI_Send(currentSendBuf, currentSize, MPI_BYTE, nextRank, tag, comm);
            // logd("rank%d send %d bytes to rank%d", rank, currentSize, nextRank);
            MPI_Recv(currentRecvBuf, currentSize, MPI_BYTE, prevRank, tag, comm, MPI_STATUS_IGNORE);
            // logd("rank%d recv %d bytes from rank%d", rank, currentSize, prevRank);
        }
        else
        {
            MPI_Recv(currentRecvBuf, currentSize, MPI_BYTE, prevRank, tag, comm, MPI_STATUS_IGNORE);
            // logd("rank%d recv %d bytes from rank%d", rank, currentSize, prevRank);
            MPI_Send(currentSendBuf, currentSize, MPI_BYTE, nextRank, tag, comm);
            // logd("rank%d send %d bytes to rank%d", rank, currentSize, nextRank);
        }
    }

    return 0;
}

// In-place， 最后一个节点只需要收，不发送。
int myBcastRing(void *data, size_t count, MPI_Datatype dataType, int root, MPI_Comm comm, void *stream)
{
    int rank, rankNum, typeSize;
    const int tag = 100;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &rankNum);
    MPI_Type_size(dataType, &typeSize);

    if (1 == rankNum)
    {
        return 0;
    }

    const int prevRank = (rank - 1 + rankNum) % rankNum;
    const int nextRank = (rank + 1 + rankNum) % rankNum;
    const int lastRank = (root - 1 + rankNum) % rankNum;

    int totalSize  = count * typeSize;
    int bufferNum  = (totalSize + sliceSize - 1) / sliceSize;
    int bufferSize = sliceSize;
    logd("rank%d bufferNum = %d, bufferSize = %d", rank, bufferNum, bufferSize);

    for (int i = 0; i < bufferNum; ++i)
    {
        int offset          = i * bufferSize;
        int remainSize      = totalSize - offset;
        int currentSize     = remainSize > bufferSize ? bufferSize : remainSize;
        uint8_t *currentPtr = (uint8_t *)data + offset;

        if (rank == root)
        {
            MPI_Send(currentPtr, currentSize, MPI_BYTE, nextRank, tag, comm);
            // logi("rank%d send %d bytes to rank%d", rank, currentSize, nextRank);
        }
        else if (rank == lastRank)
        {
            MPI_Recv(currentPtr, currentSize, MPI_BYTE, prevRank, tag, comm, MPI_STATUS_IGNORE);
            // logm("rank%d recv %d bytes from rank%d", rank, currentSize, prevRank);
        }
        else
        {
            MPI_Recv(currentPtr, currentSize, MPI_BYTE, prevRank, tag, comm, MPI_STATUS_IGNORE);
            // logd("rank%d recv %d bytes from rank%d", rank, currentSize, prevRank);
            MPI_Send(currentPtr, currentSize, MPI_BYTE, nextRank, tag, comm);
            // logd("rank%d send %d bytes to rank%d", rank, currentSize, nextRank);
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, rankNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankNum);

    int testCase  = 2;
    int root      = rankNum - 1;
    int sendCount = 3;  // 元素数量
    int recvCount = sendCount;
    int *sendData = nullptr;
    int *recvData = new int[recvCount];
    memset(recvData, -1, recvCount);

    if (rank == root)
    {
        sendData = new int[sendCount];
        for (int i = 0; i < sendCount; ++i)
        {
            sendData[i] = i;
        }
    }

    switch (testCase)
    {
        case 1:
        {
            void *data = rank == root ? sendData : recvData;
            myBcastRing(data, sendCount, MPI_INT, root, MPI_COMM_WORLD, nullptr);
            break;
        }

        case 2:
        {
            myBroadcastRing(sendData, recvData, sendCount, MPI_INT, root, MPI_COMM_WORLD, nullptr);
            break;
        }

        default:
            loge("invalid test case id = %d", testCase);
    }

    // myBroadcast(sendData, recvData, sendCount, MPI_INT, root, MPI_COMM_WORLD, nullptr);

    // 每个进程打印收集到的数据
    printf("Broadcast: rank%d collected data:", rank);
    for (int i = 0; i < recvCount; i++)
    {
        printf("%d, ", recvData[i]);
    }
    printf("\n");

    if (sendData)
    {
        delete[] sendData;
        sendData = nullptr;
    }
    if (recvData)
    {
        delete[] recvData;
        recvData = nullptr;
    }

    MPI_Finalize();
    return 0;
}