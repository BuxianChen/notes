# 并行设计导论

参考书: 并行程序设计导论
网站: [https://www.cs.usfca.edu/~peter/ipp/](https://www.cs.usfca.edu/~peter/ipp/)

# 第 1 章: 简介

## MPI, Pthreads, OpenMP简介

MPI (Message-Passing Interface), Pthreads (POSIX threads), OpenMP都是**C语言**的扩展, 用于编写并行程序.

并行系统主要分为两种:

* **共享内存系统**: 多核共享同一内存, 每个核都能读写这一内存的所有区域, 因此可以通过检测和更新共享内存中的数据来协调各个核.
* **分布式内存系统**: 每个核拥有自己的私有内存, 核之间需要使用类似于在网络中发送消息的方式进行显式通信.

MPI, Pthreads, OpenMP三者的简要区别如下:

* MPI与Pthreads是C语言的扩展库, 可以在C程序中使用扩展的类型定义、函数和宏, OpenMP包含了一个扩展库以及对C编译器的部分修改.
* Pthreads与OpenMP是为共享内存系统的编程而设计的, MPI是为分布式内存系统的编程而设计的, 它提供发送消息的机制来实现并行.
* 对于共享内存系统编程的两种扩展Pthreads与OpenMP来说, 两者的区别在于: ~~OpenMP是对C语言相对更高层次的扩展 (存疑)~~

## 并发, 并行, 分布式

* 并发指的是一个程序的多个任务在同一时间段内同时执行
* 并行指的是一个程序通过多个任务紧密协作来解决某个问题 (简单理解为单机器内部协调)
* 分布式指的是一个程序需要与其他程序协作来解决某个问题 (简单理解为多机器合作)

所以并行与分布式一定伴随着并发. 注意, 有些人会这样约定: **并行=共享内存, 分布式=分布式内存**.

## 冯·诺伊曼结构

# 第 2 章: Pthread

# 第 3 章: MPI

```c
// mpicc -g -Wall -o mpi_hello mpi_hello.c
// mpiexec -n 4 ./mpi_hello
#include <stdio.h>
#include <string.h>
#include <mpi.h>

const int MAX_STRING = 100;

int main() {
    char greeting[MAX_STRING];
    int comm_sz;
    int my_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank != 0) {
        sprintf(greeting, "Greeting from process %d of %d!", my_rank, comm_sz);
        MPI_Send(greeting, strlen(greeting)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    else {
        printf("Greeting from process %d of %d!\n", my_rank, comm_sz);
        for (int q = 1; q < comm_sz; q++) {
            MPI_Recv(greeting, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", greeting);
        }
    }

    MPI_Finalize();
    return 0;
}
```


```c
int MPI_Init(
    int* argc_p;     // argc_p 是 main 函数参数 argc 的指针
    char*** argv_p;  // argv_p 是 main 函数参数 argv 的指针
);
int MPI_Finalize(void);
MPI_Comm MPI_COMM_WORLD;  // communicator
int MPI_Comm_size(
    MPI_Comm comm,
    int * comm_sz_p
);
int MPI_Comm_rank(
    MPI_Comm comm,
    int * my_rank_p
);

int MPI_Send(
    void* msg_buf_p          /* in */,   // 发送的数据存储的指针
    int   msg_size           /* in */,   // 发送数据个数
    MPI_Datatype msg_type    /* in */,   // 发送数据类型, 估计是一个宏
    int   dest               /* in */,   // 发送的目标 rank
    int   tag                /* in */,   // 发送的 tag, 接收方也要接收这个 tag
    MPI_Comm communicator    /* in */    // 通信子, 一般就 MPI_COMM_WORLD
);

// MPI_ANY_SOURCE, MPI_ANY_TAG
int MPI_Recv(
    void*  msg_buf_p         /* out */,  // 接收的数据存储指针
    int    buf_size          /* in  */,  // 接收数据buffer最大值
    MPI_Datatype buf_type    /* in  */,  // 接收数据buffer数据类型
    int    source            /* in  */,  // 源 rank
    int    tag               /* in  */,  // tag, 与发送方匹配
    MPI_Comm communicator    /* in  */,  // 通信子, 一般就是 MPI_COMM_WORLD
    MPI_Status* status_p     /* out */,  // 不确定, 可以传 MPI_STATUS_IGNORE
)

struct MPI_Status {
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;  // 不确定数据类型
};

// 获取接收到的数据量
int MPI_Get_count(
    MPI_Status * status_p    /* in  */,
    MPI_Datatype type        /* in  */,
    int *        count_p     /* out */  // *count_p 即为数据量(字节数除以单个数据类型所占字节数)
)


// MPI 中, input_data_p 与 output_data_p 不能传同一个地址
int MPI_Reduce(
    void*  input_data_p      /* in  */,
    void*  output_data_p     /* out */,  // 非目标进程可以传 null
    int    count             /* in  */,
    MPI_Datatype datatype    /* in  */,
    MPI_Op operator          /* in  */,  // 例如: MPI_SUM
    int    dest_process      /* in  */,
    MPI_Comm comm            /* in  */
)

```

q 进程如果发送了两条消息给 r, 那么第一条消息会在第二条消息前可用 (但 r 可以先接收第 2 条消息, 再接收第 1 条消息?? 不确定)