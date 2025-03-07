/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_H_
#define NCCL_SOCKET_H_

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include "nccl.h"

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT 1000  // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES                                                    \
  2e4  // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES                                                   \
  3  // connection timed out retry times (each one can take 20s)
#define SOCKET_NAME_MAXLEN (NI_MAXHOST + NI_MAXSERV)
#define NCCL_SOCKET_MAGIC 0x564ab9f2fc4b9d6cULL

/* Common socket address storage structure for IPv4/IPv6 */
union ncclSocketAddress {
    struct sockaddr sa;
    struct sockaddr_in sin;
    struct sockaddr_in6 sin6;
};

enum ncclSocketState {
  ncclSocketStateNone = 0,
  ncclSocketStateInitialized = 1,
  ncclSocketStateAccepting = 2,
  ncclSocketStateAccepted = 3,
  ncclSocketStateConnecting = 4,
  ncclSocketStateConnectPolling = 5,
  ncclSocketStateConnected = 6,
  ncclSocketStateReady = 7,
  ncclSocketStateClosed = 8,
  ncclSocketStateError = 9,
  ncclSocketStateNum = 10
};

enum ncclSocketType {
  ncclSocketTypeUnknown = 0,
  ncclSocketTypeBootstrap = 1,
  ncclSocketTypeProxy = 2,
  ncclSocketTypeNetSocket = 3,
  ncclSocketTypeNetIb = 4
};

struct ncclSocket {
    int fd;
    int acceptFd;
    int timedOutRetries;
    int refusedRetries;
    union ncclSocketAddress addr;
    volatile uint32_t *abortFlag;
    int asyncFlag;
    enum ncclSocketState state;
    int salen;
    uint64_t magic;
    enum ncclSocketType type;
};

// Initialize a socket
ncclResult_t ncclSocketInit(struct ncclSocket *sock,
                            union ncclSocketAddress *addr = NULL,
                            uint64_t magic = NCCL_SOCKET_MAGIC,
                            enum ncclSocketType type = ncclSocketTypeUnknown,
                            volatile uint32_t *abortFlag = NULL,
                            int asyncFlag = 0);
// Connect to sock->addr. sock->fd is set after a successful call.
ncclResult_t ncclSocketConnect(struct ncclSocket *sock);
// Accept an incoming connection from listenSock->fd and keep the file
// descriptor in sock->fd, with the remote side IP/port in sock->addr.
ncclResult_t ncclSocketAccept(struct ncclSocket *sock,
                              struct ncclSocket *ulistenSock);

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1

ncclResult_t ncclSocketSend(struct ncclSocket *sock, void *ptr, int size);
ncclResult_t ncclSocketRecv(struct ncclSocket *sock, void *ptr, int size);
ncclResult_t ncclSocketTryRecv(struct ncclSocket *sock, void *ptr, int size,
                               int *closed, bool blocking);
ncclResult_t ncclSocketClose(struct ncclSocket *sock);

struct unexConn {
    int peer;
    int tag;
    struct ncclSocket sock;
    struct unexConn *next;
};

struct bootstrapState {
    struct ncclSocket listenSock;
    struct ncclSocket ringRecvSocket;
    struct ncclSocket ringSendSocket;
    union ncclSocketAddress *peerCommAddresses;
    union ncclSocketAddress *peerProxyAddresses;
    struct unexConn *unexpectedConnections;
    int cudaDev;
    int rank;
    int nranks;
    uint64_t magic;
    volatile uint32_t *abortFlag;
};

#endif
