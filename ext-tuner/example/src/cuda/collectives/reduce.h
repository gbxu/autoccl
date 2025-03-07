#pragma once
#include <cmath>
#include <vector>
#include "src/cuda/collectives/util.h"
#include "src/cuda/nccl_params.h"

void ncclGetValidCandidatesForReduce(
    std::vector<std::vector<int>> *candidates, int nrank,
    int count, std::vector<int> buffSizeDefaultDict, std::vector<int> algos,
    std::vector<int> protos, std::vector<int> smCopyOrCopyEngine, int32_t tuner_nChannels, std::vector<int> p2pLevels, int scale2) {
  for (int algo : algos) {
    for (int proto : protos) {
      int buffSize = buffSizeDefaultDict[proto];
      for (int isCopyEngineNotSmCopy : smCopyOrCopyEngine) {
        for (int p2pLevel : p2pLevels) {
        if (isCopyEngineNotSmCopy == 1) {
          if (proto != NCCL_PROTO_SIMPLE)
            continue;
          if (p2pLevel == PATH_LOC) {
            continue;
          }
        }
        bool small = false;
        for (int nc = 1; nc <= tuner_nChannels && !small;
             nc = (scale2 ? nc * 2 : nc + 1)) {
          for (int nt = 3 * WARP_SIZE;
               nt <= NCCL_MAX_NTHREADS && !small;
               nt = (scale2 ? nt * 2 : nt + WARP_SIZE)) {
            if (nt > NCCL_MAX_NTHREADS-WARP_SIZE && proto == NCCL_PROTO_SIMPLE) continue;
            int maxEffectiveChunkSize, minEffectiveChunkSize,
                effectiveChunksize, wireChunksize, lastIterEffectiveChunksize;
            int iteration;
            if (proto == NCCL_PROTO_LL) {
              maxEffectiveChunkSize =
                  static_cast<int>(buffSize / NCCL_STEPS / 2 / SIZEELEM);
              minEffectiveChunkSize = static_cast<int>(
                  nt * 8 / SIZEELEM);  // it can be any chunkSize in allgather
              int i = 1;
              do {
                effectiveChunksize = i * minEffectiveChunkSize;
                if ((effectiveChunksize <= maxEffectiveChunkSize) &&
                    (effectiveChunksize % EltPerLine == 0)) {
                  iteration = static_cast<int>(
                      ceil(1.0 * count / (nc * effectiveChunksize)));
                  // assert(count % (nrank * nc * effectiveChunksize) % (nrank *
                  // nc) == 0);
                  lastIterEffectiveChunksize =
                      count % (nc * effectiveChunksize) / (nc)*SIZEELEM;
                  wireChunksize =
                      effective2Wire(effectiveChunksize, proto) * SIZEELEM;
                  candidates->push_back({algo, proto, isCopyEngineNotSmCopy, p2pLevel, nc, nt,
                                         wireChunksize, iteration,
                                         lastIterEffectiveChunksize, 0});
                } else if (effectiveChunksize > maxEffectiveChunkSize) {
                  break;
                }
                i = (scale2 ? i * 2 : i + 1);
              } while (i <=
                       static_cast<int>(count / (nc * minEffectiveChunkSize)));
            } else if (proto == NCCL_PROTO_LL128) {  // LL128
              maxEffectiveChunkSize = static_cast<int>(
                  buffSize / NCCL_STEPS * NCCL_LL128_DATAELEMS /
                  NCCL_LL128_LINEELEMS / SIZEELEM);
              minEffectiveChunkSize = static_cast<int>(
                  nt * NCCL_LL128_SHMEM_ELEMS_PER_THREAD *
                  NCCL_LL128_DATAELEMS * 8 / NCCL_LL128_LINEELEMS /
                  SIZEELEM);  // TODO: we should let it divide 2 ?
              int i = 1;
              do {
                effectiveChunksize = i * minEffectiveChunkSize;
                if ((effectiveChunksize <= maxEffectiveChunkSize) &&
                    (effectiveChunksize % DataEltPerSlice == 0)) {
                  iteration = static_cast<int>(
                      ceil(1.0 * count / (nc * effectiveChunksize)));
                  // assert(count % (nrank * nc * effectiveChunksize) % (nrank *
                  // nc) == 0);
                  lastIterEffectiveChunksize =
                      count % (nc * effectiveChunksize) / (nc)*SIZEELEM;
                  wireChunksize =
                      effective2Wire(effectiveChunksize, proto) * SIZEELEM;
                  candidates->push_back({algo, proto, isCopyEngineNotSmCopy, p2pLevel, nc, nt,
                                         wireChunksize, iteration,
                                         lastIterEffectiveChunksize, 0});
                } else if (effectiveChunksize > maxEffectiveChunkSize) {
                  break;
                }
                i = (scale2 ? i * 2 : i + 1);
              } while (i <=
                       static_cast<int>(count / (nc * minEffectiveChunkSize)));
            } else if (proto == NCCL_PROTO_SIMPLE) {
              maxEffectiveChunkSize = static_cast<int>(
                  buffSize / NCCL_STEPS /
                  SIZEELEM);  // only occupy 1 buffer slot and send 1
              minEffectiveChunkSize = static_cast<int>(nt * 8 / SIZEELEM);
              int i = 1;
              do {
                effectiveChunksize = i * minEffectiveChunkSize;
                if (effectiveChunksize <= maxEffectiveChunkSize) {
                  iteration = static_cast<int>(
                      ceil(1.0 * count / (nc * effectiveChunksize)));
                  // assert(count % (nrank * nc * effectiveChunksize) % (nrank *
                  // nc) == 0);
                  lastIterEffectiveChunksize =
                      count % (nc * effectiveChunksize) / (nc)*SIZEELEM;
                  wireChunksize =
                      effective2Wire(effectiveChunksize, proto) * SIZEELEM;
                  candidates->push_back({algo, proto, isCopyEngineNotSmCopy, p2pLevel, nc, nt,
                                         wireChunksize, iteration,
                                         lastIterEffectiveChunksize, 0});
                } else {
                  break;
                }
                i = (scale2 ? i * 2 : i + 1);
              } while (i <=
                       static_cast<int>(count / (nc * minEffectiveChunkSize)));
            }
            small = (iteration == 1) && (count < 1048576);
          }
        }
        }
      }
    }
  }
}
