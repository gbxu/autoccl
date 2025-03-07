/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TUNER_H_
#define NCCL_TUNER_H_

#include <map>
#include <string>
#include "nccl.h"
#include "devcomm.h"

// API to be implemented by external tuner
typedef struct {
  // Name of the tuner
  const char* name;

  // Initializes tuner states.
  // nRanks: number of ranks in current communicator. Each communicator initialize its own tuner.
  // nNodes: number of nodes in current communicator.
  // logFunction: a logFunction can be useful to integrate logging together with NCCL core.
  ncclResult_t (*init)(uint64_t commHash, size_t nRanks, size_t nNodes, size_t rank, size_t node, size_t device, std::map<std::string, int32_t> tunerEnvs, void* handler);

  // Gets info (algo, protocol, number of ctas and threads) for a given collective.
  // Inputs:
  //   - collType: collective type , e.g., allreduce, allgather…
  //   - nBytes: collective size in bytes
  //   - collNetTypeSupport: whether collnet supports this type
  //   - nvlsTypeSupport: whether nvlink sharp supports this time
  //   - numPipeOps: number of operations in the group
  //
  // Outputs:
  //   - algorithm: selected algorithm to be used for the given collective
  //   - protocol: selected protocol to be used for the given collective
  //   - nChannels: number of channels (hence SMs) to be used.
  //
  // If getCollInfo() does not return ncclSuccess, NCCL will fall back to the
  // default tuning for the given collective.
  // Also, the plugin is allowed to not set any output, or set only the
  // algorithm and protocol, but not only the algorithm or only the protocol.
  // Unset fields will be set automatically by NCCL.
  ncclResult_t (*getCandidate)(uint64_t commHash, ncclFunc_t collType, size_t nBytes,
                              int *algorithm, int *protocol, int* isCopyEngineNotSmCopy, int* p2pLevel,
                              int* nChannels, int* nThreads,
                              int* chunkSize, int* iteration, int* lastIterEffectiveChunksize, int* native);

  // Terminates the plugin and cleans up any resources that the plugin allocated.
  ncclResult_t (*destroy)(uint64_t commHash);
  // Profiles the communication
  ncclResult_t (*startProfiling)(uint64_t commHash, cudaStream_t stream, ncclFunc_t collType, size_t nBytes,
                              int algorithm, int protocol, int isCopyEngineNotSmCopy, int p2pLevel,
                              int nChannels, int nThreads,
                              int chunkSize, int iteration, int lastIterEffectiveChunksize, int native);
  ncclResult_t (*stopProfiling)(uint64_t commHash);
  ncclResult_t (*isNewWorkload)(uint64_t commHash, ncclFunc_t collType, size_t nBytes, bool* flag);
} ncclTuner_v1_t;

typedef ncclTuner_v1_t ncclTuner_t;

#define NCCL_TUNER_PLUGIN_SYMBOL "ncclTunerPlugin_v1"

#define TUNER_P2PCHUNKSIZE 128 * 1024
#define SIMPLE_P2PCHUNKSIZE_UPPER_BOUND 128*1024
#define LL128_P2PCHUNKSIZE_UPPER_BOUND 128*1024
#define LL_P2PCHUNKSIZE_UPPER_BOUND 80*1024
static_assert(SIMPLE_P2PCHUNKSIZE_UPPER_BOUND <= TUNER_P2PCHUNKSIZE);
static_assert(LL128_P2PCHUNKSIZE_UPPER_BOUND <= TUNER_P2PCHUNKSIZE);
static_assert(LL_P2PCHUNKSIZE_UPPER_BOUND <= TUNER_P2PCHUNKSIZE);
ncclResult_t ncclGetLevel(int* level, const char* disableEnv, const char* levelEnv);
#endif
