/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    ncclRing *ring = &ncclShmem.channel.ring;
    ssize_t chunkSize;
    if (args->native) {
      chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? REDUCE_CHUNKSTEPS : 1));
    } else {
      chunkSize = args->effectiveChunkSize;
    }
    ssize_t minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL)
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    if (Proto::Id == NCCL_PROTO_LL128) {
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = int(nthreads*(Proto::calcBytePerGrain()/sizeof(T))); // minChunkSizeLL128
    }

    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->count;
    const int rank = ncclShmem.comm.rank;
    const int prevRank = ring->userRanks[nranks-1];
    const int root = args->root;

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, args->redOpArg, 0, 0, 0, args->transportIndex);

    auto calcChunkSize = [&]__device__(ssize_t gridOffset)->int {
      int realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      } else if (Proto::Id == NCCL_PROTO_LL) {
        if (args->native) {
          realChunkSize = size-gridOffset < loopSize ? args->lastChunkSize : chunkSize;
        } else {
          realChunkSize = min(divUp(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);
        }
      } else if (Proto::Id == NCCL_PROTO_LL128) {
        realChunkSize = min(divUp(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);
      }
      return realChunkSize;
    };

    if (prevRank == root) {
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize;
        realChunkSize = calcChunkSize(gridOffset);
        ssize_t offset = gridOffset + bid*realChunkSize;
        int nelem = min(realChunkSize, size-offset);
        prims.send(offset, nelem);
      }
    }
    else if (rank == root) {
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize;
        realChunkSize = calcChunkSize(gridOffset);
        ssize_t offset = gridOffset + bid*realChunkSize;
        int nelem = min(realChunkSize, size-offset);
        prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
      }
    }
    else {
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize;
        realChunkSize = calcChunkSize(gridOffset);
        ssize_t offset = gridOffset + bid*realChunkSize;
        int nelem = min(realChunkSize, size-offset);
        prims.recvReduceSend(offset, nelem);
      }
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runRing<T, RedOp, ProtoLL128>(args);
  }
};
