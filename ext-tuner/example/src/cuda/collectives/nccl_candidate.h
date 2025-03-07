#pragma once
#include <cmath>
#include <unordered_map>
#include <vector>
#include "src/include/datatype.h"
#include "src/include/internal/logging.h"
#include "src/cuda/nccl_params.h"
#include "src/cuda/collectives/util.h"
#include "src/cuda/collectives/all2all.h"
#include "src/cuda/collectives/all_gather.h"
#include "src/cuda/collectives/all_reduce.h"
#include "src/cuda/collectives/broadcast.h"
#include "src/cuda/collectives/reduce.h"
#include "src/cuda/collectives/reduce_scatter.h"

// clang-format off
// TODO: support the other algo for allreduce, e.g., NVLS-SIMPLE, NVLS_TREE-SIMPLE.
// clang-format on

void ncclGetValidCandidates(
    const Info &myInfo,
    const std::unordered_map<GIDTYPE, GroupInfo> &allGroupInfos,
    const Workload &workload, bool scale2, std::vector<Candidate> *candidates, std::vector<ConfigRange> *configRanges) {
  //  workload(groupID, colltype, nbytes)
  //  candidate(algorithm, protocol, smCopyOrCopyEngine, p2pLevel, nChannels, nThreads,
  //  wireChunksize, iteration, lastIterEffectiveChunksize)
  KEY groupID = workload[0];   // NOLINT
  KEY collType = workload[1];  // NOLINT
  KEY nBytes = workload[2];    // NOLINT

  auto &group = allGroupInfos.at(groupID);
  bool extraP2PCESupport = group.tunerEnvs.at("tuner_extraP2PCE");
  bool extraSHMSupport = group.tunerEnvs.at("tuner_extraSHM");
  int rank = group.rank;
  int nrank = group.nrank;
  int count = ceil(1.0 * nBytes / SIZEELEM);        // NOLINT
  std::vector<int> buffSizeDefaultDict = {524288 +  // NOLINT
                                              (8 * 16) * 128 * 8,
                                          4915200, 4194304};
  std::vector<int> algos = {NCCL_ALGO_TREE, NCCL_ALGO_RING};   // NOLINT
  std::vector<int> protos = {NCCL_PROTO_LL, NCCL_PROTO_LL128,  // NOLINT
                             NCCL_PROTO_SIMPLE};
  std::vector<int> smCopyOrCopyEngine = extraP2PCESupport ? 
    std::vector<int>{NCCL_SM_COPY, NCCL_COPY_ENGINE} : std::vector<int>{NCCL_SM_COPY};  // NOLINT
  // PATH_LOC, PATH_PIX, PATH_PXB, PATH_PHB, PATH_SYS
  std::vector<int> p2pLevels;
  if (!extraSHMSupport) {
    p2pLevels = std::vector<int>{PATH_SYS};  // use p2p anyway
  } else {
    if (group.tunerEnvs.at("tuner_p2pLevel") != -1) {
      p2pLevels = std::vector<int>{group.tunerEnvs.at("tuner_p2pLevel")};
    } else {
      p2pLevels = std::vector<int>{PATH_PIX, PATH_PXB, PATH_PHB, PATH_SYS};  // p2p or shm
    }
  }
  // scale2: true for exponetial; false for linear
  switch (collType) {
  case  ncclFuncAll2All:
    // sendrecv使用的buffer不一样
    CHECK(80*1024 <= group.tunerEnvs.at("tuner_p2pChunkSize"));
    ncclGetValidCandidatesForAll2all(candidates, nrank, count,
                                       {80*1024, group.tunerEnvs.at("tuner_p2pChunkSize"), group.tunerEnvs.at("tuner_p2pChunkSize")}, {NCCL_ALGO_RING}, (group.nnode > 1 ? std::vector<int>{NCCL_PROTO_SIMPLE} : std::vector<int>{NCCL_PROTO_LL, NCCL_PROTO_SIMPLE}),
                                       {0}, group.tunerEnvs.at("tuner_p2pnChannelsPerPeer"), group.tunerEnvs.at("tuner_p2pnChannels"), p2pLevels, scale2);
    break;
  case ncclFuncAllReduce:
    ncclGetValidCandidatesForAllreduce(group.tunerEnvs, candidates, nrank, count,
                                       buffSizeDefaultDict, algos, protos,
                                       smCopyOrCopyEngine, group.tunerEnvs.at("tuner_nChannels"), p2pLevels, scale2);
    break;
  case ncclFuncReduceScatter:
    ncclGetValidCandidatesForReducescatter(candidates, nrank, count,
                                           buffSizeDefaultDict, {NCCL_ALGO_RING}, protos,
                                           smCopyOrCopyEngine, group.tunerEnvs.at("tuner_nChannels"), p2pLevels, scale2);
    break;
  case ncclFuncAllGather:
    ncclGetValidCandidatesForAllgather(candidates, nrank, count,
                                       buffSizeDefaultDict, {NCCL_ALGO_RING}, protos,
                                       smCopyOrCopyEngine, group.tunerEnvs.at("tuner_nChannels"), p2pLevels, scale2);
    break;
  case ncclFuncReduce:
    ncclGetValidCandidatesForReduce(candidates, nrank, count,
                                       buffSizeDefaultDict, {NCCL_ALGO_RING}, protos,
                                       smCopyOrCopyEngine, group.tunerEnvs.at("tuner_nChannels"), p2pLevels, scale2);
    break;
  case ncclFuncBroadcast:
    ncclGetValidCandidatesForBroadcast(candidates, nrank, count,
                                       buffSizeDefaultDict, {NCCL_ALGO_RING}, protos,
                                       smCopyOrCopyEngine, group.tunerEnvs.at("tuner_nChannels"), p2pLevels, scale2);
    break;
  default:
    throw std::runtime_error("undefined collective!\n");
    break;
  }

  configRanges->clear();
  configRanges->emplace_back(true, ConfigRange::EMPTY, ConfigRange::EMPTY, ConfigRange::EMPTY, 1); // algo
  configRanges->emplace_back(true, ConfigRange::EMPTY, ConfigRange::EMPTY, ConfigRange::EMPTY, 1); // proto
  configRanges->emplace_back(true, ConfigRange::EMPTY, ConfigRange::EMPTY, ConfigRange::EMPTY, 1); // sm copy or copy engine
  configRanges->emplace_back(true, ConfigRange::EMPTY, ConfigRange::EMPTY, ConfigRange::EMPTY, 1); // p2p level to choose p2p
  configRanges->emplace_back(true, ConfigRange::EMPTY, ConfigRange::EMPTY, ConfigRange::EMPTY, 1); // nc
  configRanges->emplace_back(true, ConfigRange::EMPTY, ConfigRange::EMPTY, ConfigRange::EMPTY, 1); // nt
  configRanges->emplace_back(true, ConfigRange::EMPTY, ConfigRange::EMPTY, ConfigRange::EMPTY, 1); // wire chunk size
  configRanges->emplace_back(false, ConfigRange::EMPTY, ConfigRange::EMPTY, ConfigRange::EMPTY, 0); // iter
  configRanges->emplace_back(false, ConfigRange::EMPTY, ConfigRange::EMPTY, ConfigRange::EMPTY, 0); // lastIterEffectiveChunksize
  configRanges->emplace_back(false, ConfigRange::EMPTY, ConfigRange::EMPTY, ConfigRange::EMPTY, 0); // native

  return;
}
