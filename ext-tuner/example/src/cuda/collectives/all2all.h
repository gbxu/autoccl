#pragma once
#include <cmath>
#include <vector>
#include "src/cuda/collectives/util.h"
#include "src/cuda/nccl_params.h"
#include "src/include/datatype.h"

#define NCCL_MAX_WORK_ELEMENTS_P2P 16
void ncclGetValidCandidatesForAll2all(
    std::vector<std::vector<int>> *candidates, int nrank,
    int count, std::vector<int> buffSizeDictAll2all, std::vector<int> algos,
    std::vector<int> protos, std::vector<int> smCopyOrCopyEngine, int32_t tuner_p2pnChannelsPerPeer, int32_t tuner_p2pnChannels, std::vector<int> p2pLevels, int scale2) {
  for (int algo : algos) {
    for (int proto : protos) {
      int buffSize = buffSizeDictAll2all[proto];
      // sendrecv should not use cudamemcpy due to the deadlock potential
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
        // make sure nc is power of 2
        for (int nc = 1; nc <= tuner_p2pnChannelsPerPeer && !small;
            nc = (scale2 ? nc * 2 : nc + 1)) {
          // use NCCL_MAX_WORK_ELEMENTS_P2P/2 as upper bound to limit one round
          if (DIVUP(nc*nrank, tuner_p2pnChannels) > NCCL_MAX_WORK_ELEMENTS_P2P/2) {
            break;
          }
          int colocate = DIVUP(nc*nrank, tuner_p2pnChannels);
          int max_nt = 512/2/colocate;
          max_nt = max_nt/WARP_SIZE*WARP_SIZE;
          for (int nt = WARP_SIZE;
               nt <= max_nt && !small;
               nt = (scale2 ? nt * 2 : nt + WARP_SIZE)) {
            int maxEffectiveChunkSize, minEffectiveChunkSize,
                effectiveChunksize, wireChunksize, lastIterEffectiveChunksize;
            int iteration;
            if (proto == NCCL_PROTO_LL) {
              maxEffectiveChunkSize =
                  static_cast<int>(buffSize / 2 / SIZEELEM);
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
                  buffSize / NCCL_LL128_DATAELEMS /
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
              maxEffectiveChunkSize =
                  static_cast<int>(buffSize / SIZEELEM);
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
