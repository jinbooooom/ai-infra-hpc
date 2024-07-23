#include <arpa/inet.h>
#include <mpi.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>

#include <getopt.h>
#include <ifaddrs.h>
#include <string.h>
#include <sys/types.h>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <thread>
#include <vector>
#include "../../../utils/log/log.hpp"

int gNodeNum;  // 节点总数
int gRank;     // 当前的节点

int fun(int x)
{
    int y = x * x;
    if (gRank)
    {
        logd("y = %d", y);
    }
    else
    {
        logw("y = %d", y);
    }

    return y;
}

int main(int argc, char **argv)
{
    int ret = 0;

    std::string ip = get_local_ip();
    // printf("ip = %s\n", ip.c_str());
    if (std::string("192.168.76.3") == ip)
    {
        gRank = 0;
    }
    else
    {
        gRank = 1;
    }
    gNodeNum = 2;

    fun(gRank);

    printf("\n");
    int flag = 10;
    while (flag)
    {
        sleep(10);
        fun(flag--);
    }

    fun(100 + gRank);

    return 0;
}
