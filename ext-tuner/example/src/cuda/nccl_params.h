#pragma once

#define NCCL_NUM_ALGORITHMS 6  // Tree/Ring/CollNet*
#define NCCL_ALGO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4
#define NCCL_ALGO_NVLS_TREE 5

#define NCCL_NUM_PROTOCOLS 3  // Simple/LL/LL128
#define NCCL_PROTO_UNDEF -1
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2

#define NCCL_SM_COPY 0
#define NCCL_COPY_ENGINE 1

#define NCCL_STEPS 8
#define WARP_SIZE 32
#define SIZEELEM (sizeof(int8_t))
// LL
#define EltPerLine (sizeof(uint64_t)/SIZEELEM)
// LL128
#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS - 1)
#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define WireWordPerSlice WARP_SIZE *NCCL_LL128_SHMEM_ELEMS_PER_THREAD

#define DataEltPerSlice                                                        \
  ((WireWordPerSlice-WireWordPerSlice/NCCL_LL128_LINEELEMS)*(sizeof(uint64_t)/SIZEELEM))

#define NCCL_MAX_NTHREADS 640

using ncclFunc_t = enum {
  ncclFuncBroadcast,
  ncclFuncReduce,
  ncclFuncAllGather,
  ncclFuncReduceScatter,
  ncclFuncAllReduce,
  ncclFuncSendRecv,
  ncclFuncSend,
  ncclFuncRecv,
  ncclNumFuncs,
  ncclFuncAll2All,
  ncclFuncAll2Allv
};

using LinkType = enum {
  // Local (myself)
  PATH_LOC = 0,
  // Connection traversing NVLink
  PATH_NVL = 1,
  // Connection through NVLink using an intermediate GPU
  PATH_NVB = 2,
  // Connection traversing at most a single PCIe bridge
  PATH_PIX = 3,
  // Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PATH_PXB = 4,
  // Connection between a GPU and a NIC using an intermediate GPU. Used to enable rail-local, aggregated network send/recv operations.
  PATH_PXN = 5,
  // Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PATH_PHB = 6,
  // Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  PATH_SYS = 7,
  // Connection through the network
  PATH_NET = 8
};

using Transports = enum {
  TRANSPORT_P2P = 0,
  TRANSPORT_P2P_CE = 1,
  TRANSPORT_SHM = 2,
  TRANSPORT_NET = 3,
  TRANSPORT_COLLNET = 4
};
