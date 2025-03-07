#include "plugin/cuda/plugin.h"
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <bitset>
#include "src/cuda/collectives/nccl_candidate.h"
#include "src/cuda/cuda_timer.h"
#include "src/cuda/nccl_combined_job.h"
#include "src/cuda/nccl_communicator.h"
#include "src/include/jobs/bruteforce_job.h"
#include "src/include/jobs/native_job.h"
#include "src/include/jobs/specific_job.h"
#include "src/include/jobs/simanneal_job.h"
#include "src/include/jobs/coordinate_descent_job.h"
#include "src/include/net/socket_communicator.h"
#include "src/include/optimizers/optimizer.h"
#include "src/include/optimizers/uniform_optimizer.h"
#include "src/include/optimizers/hybrid_optimizer.h"
#include "src/include/tuner.h"

std::unique_ptr<Optimizer> getOptimizer(CandidateFunc getValidCandidatesFunc) {
  enum TunerMode {
    TUNER_MODE_NATIVE = 0,
    TUNER_MODE_BRUTEFORCE = 1,
    TUNER_MODE_SPECIFIC = 2,
    TUNER_MODE_SIMANNEAL_COMBINED = 3,
    TUNER_MODE_SIMANNEAL = 4,
    TUNER_MODE_DESCENT_COMBINED = 5,
    TUNER_MODE_DESCENT = 6,
    TUNER_MODE_BRUTEFORCE_COMBINED = 7,
  };
  int32_t mode = Environment::get()->find("TUNER_MODE", 5);
  switch (mode) {
  case TunerMode::TUNER_MODE_NATIVE:
    return std::make_unique<HybridOptimizer<NativeJob>>(
        getValidCandidatesFunc);
  case TunerMode::TUNER_MODE_BRUTEFORCE:
    return std::make_unique<HybridOptimizer<BFJob>>(getValidCandidatesFunc);
  case TunerMode::TUNER_MODE_SPECIFIC:
    return std::make_unique<HybridOptimizer<SpecificJob>>(getValidCandidatesFunc);
  case TunerMode::TUNER_MODE_SIMANNEAL_COMBINED:
    return std::make_unique<HybridOptimizer<NcclCombinedJob<SimAnnealJob>>>(
        getValidCandidatesFunc);
  case TunerMode::TUNER_MODE_SIMANNEAL:
    return std::make_unique<HybridOptimizer<SimAnnealJob>>(getValidCandidatesFunc);
  case TunerMode::TUNER_MODE_DESCENT_COMBINED:
    return std::make_unique<HybridOptimizer<NcclCombinedJob<CoDesJob>>>(getValidCandidatesFunc);
  case TunerMode::TUNER_MODE_DESCENT:
    return std::make_unique<HybridOptimizer<CoDesJob>>(getValidCandidatesFunc);
  case TunerMode::TUNER_MODE_BRUTEFORCE_COMBINED:
    return std::make_unique<HybridOptimizer<NcclCombinedJob<BFJob>>>(getValidCandidatesFunc);
  default:
    throw std::runtime_error("undefined mode!\n");
    break;
  }
  INFO(Logger::LogSubSys::ALL) << "TUNER_MODE=" << mode;
}

#define __hidden __attribute__((visibility("hidden")))

// various workers for different communication groups
static std::map<uint64_t, std::shared_ptr<Tuner>> workers;

// work for multi communication groups
static Tuner worker;
static Tuner coordinator;
static const char *coordinatorStr = nullptr;

__hidden ncclResult_t pluginInit(uint64_t commHash, size_t nRanks,
                                 size_t nNodes, size_t rank, size_t node,
                                 size_t device, 
                                 std::map<std::string, int32_t> tunerEnvs, void *handler) {
  INFO(Logger::LogSubSys::ALL) << toDebugStr(tunerEnvs);
  int root = 0;
  Info myInfo(node, device);
  if (coordinatorStr == nullptr) {
    coordinatorStr = Environment::get()->find("TUNER_COORDINATOR");
  }
  if (!coordinatorStr) {
    std::shared_ptr<Tuner> currWorker = std::make_shared<Tuner>();
    if (!currWorker->started) {
      INFO(Logger::LogSubSys::ALL) << "worker: using seperate tuner with multi-thread NcclCommunicator";
      auto optimizer = getOptimizer(ncclGetValidCandidates);
      auto timer = std::make_unique<CudaTimer>();
      auto communicator = std::make_unique<NcclCommunicator>();
      currWorker->start(myInfo, std::move(communicator), nullptr,
                    std::move(optimizer), std::move(timer));
    }
    NcclCommunicator *ncclCommunicator =
        dynamic_cast<NcclCommunicator *>(currWorker->communicator.get());
    CHECK(ncclCommunicator != nullptr);
    ncclCommunicator->addGroup(commHash, root, rank, nRanks, nNodes, tunerEnvs, handler);
    workers[commHash] = currWorker;
  } else {
    std::unique_ptr<std::thread> coordinatorThread = nullptr;
    // for coordinator
    const char *role = Environment::get()->find("TUNER_ROLE");
    if (role && strcmp(role, "COORDINATOR") == 0) {
      if (!coordinator.started) {
        INFO(Logger::LogSubSys::ALL) << "coordinator: using co-tuner with SocketCommunicator";
        coordinatorThread =
            std::unique_ptr<std::thread>(new std::thread([myInfo]() {
              auto communicator = std::make_unique<SocketCommunicator>();
              coordinator.start(myInfo, std::move(communicator),
                                "COORDINATOR");
            }));
        pthread_setname_np(coordinatorThread->native_handle(),
                          "coordinatorThread");
      }
    }
    // for workers
    if (!worker.started) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(10000));  // waiting for coordinator
      INFO(Logger::LogSubSys::ALL) << "worker: using co-tuner with SocketCommunicator";
      auto optimizer = getOptimizer(ncclGetValidCandidates);
      auto timer = std::make_unique<CudaTimer>();
      auto communicator = std::make_unique<SocketCommunicator>();
      worker.start(myInfo, std::move(communicator), "WORKER",
                  std::move(optimizer), std::move(timer));
    }

    worker.communicator->addGroup(commHash, root, rank, nRanks, nNodes, tunerEnvs);

    if (coordinatorThread && coordinatorThread->joinable()) {
      coordinatorThread->join();
      coordinatorThread.reset();
    }
  }
  return ncclSuccess;
}

__hidden ncclResult_t pluginGetCollInfo(
    uint64_t commHash, ncclFunc_t collType, size_t nBytes,
    int *algorithm, int *protocol,
    int *isCopyEngineNotSmCopy, int *p2pLevel, int *nChannels, int *nThreads, int *chunkSize,
    int *iteration, int *lastIterEffectiveChunksize, int *native) {
  if ((collType != ncclFuncAllReduce) && (collType != ncclFuncBroadcast) &&
      (collType != ncclFuncAllGather) && (collType != ncclFuncReduceScatter) &&
      (collType != ncclFuncReduce) && (collType != ncclFuncAll2All)) {
    return ncclSuccess;
  }

  Workload workload = {commHash, collType, nBytes};  // NOLINT
  // clang-format off
  Candidate candidate = {  // NOLINT
      *algorithm, *protocol,  *isCopyEngineNotSmCopy, *p2pLevel, *nChannels,
      *nThreads,  *chunkSize, *iteration,        *lastIterEffectiveChunksize,
      *native};
  // clang-format on
  if (!coordinatorStr) {
    workers[commHash]->query(commHash, workload, &candidate);
  } else {
    worker.query(commHash, workload, &candidate);
  }
  *algorithm = candidate[0];
  *protocol = candidate[1];
  *isCopyEngineNotSmCopy = candidate[2];
  *p2pLevel = candidate[3];
  *nChannels = candidate[4];
  *nThreads = candidate[5];
  *chunkSize = candidate[6];
  // TODO(anonymous): without divup
  *iteration = candidate[7];
  // per channel, with nrank parallel mayebe
  *lastIterEffectiveChunksize = candidate[8];
  *native = candidate[9];
  return ncclSuccess;
}

__hidden ncclResult_t pluginDestroy(uint64_t commHash) {
  if (!coordinatorStr) {
    if (workers[commHash]->started) {
      workers[commHash]->communicator->removeGroup(commHash);
      if (workers[commHash]->communicator->getMyInfo().groupIDs.size() == 0) {
        workers[commHash]->stop();
      }
    }
  } else {
    if (worker.started) {
      auto *communicator =
          dynamic_cast<SocketCommunicator *>(worker.communicator.get());
      CHECK(communicator != nullptr);
      communicator->removeGroup(commHash);
      if (communicator->getMyInfo().groupIDs.size() == 0) {
        worker.stop();
        if (coordinator.started) {
          coordinator.stop();
        }
      }
    }
  }
  return ncclSuccess;
}

__hidden ncclResult_t pluginStartProfiling(
    uint64_t commHash, cudaStream_t stream, ncclFunc_t collType, size_t nBytes,
    int algorithm, int protocol,
    int isCopyEngineNotSmCopy, int p2pLevel, int nChannels, int nThreads, int chunkSize,
    int iteration, int lastIterEffectiveChunksize, int native) {
  if ((collType != ncclFuncAllReduce) && (collType != ncclFuncBroadcast) &&
      (collType != ncclFuncAllGather) && (collType != ncclFuncReduceScatter) &&
      (collType != ncclFuncReduce) && (collType != ncclFuncAll2All)) {
    return ncclSuccess;
  }
  Workload workload = {commHash, collType, nBytes};  // NOLINT
  Candidate candidate =                              // NOLINT
      {algorithm, protocol,  isCopyEngineNotSmCopy, p2pLevel, nChannels,
       nThreads,  chunkSize, iteration,        lastIterEffectiveChunksize,
       native};
  if (!coordinatorStr) {
    workers[commHash]->timer->begin(RecordKey(commHash, workload, candidate), false, stream);
  } else {
    worker.timer->begin(RecordKey(commHash, workload, candidate), false, stream);
  }
  return ncclSuccess;
}

__hidden ncclResult_t pluginStopProfiling(uint64_t commHash) {
  if (!coordinatorStr) {
    workers[commHash]->timer->end(commHash, false);
  } else {
    worker.timer->end(commHash, false);
  }
  return ncclSuccess;
}
__hidden ncclResult_t pluginIsNewWorkload(uint64_t commHash, ncclFunc_t collType, size_t nBytes, bool* flag) {
  Workload workload = {commHash, collType, nBytes};  // NOLINT
  if (!coordinatorStr) {
    *flag = workers[commHash]->isNewWorkload(workload);
  } else {
    *flag = worker.isNewWorkload(workload);
  }
  return ncclSuccess;
}

#define PLUGIN_NAME "Odysseus"  // The Odyssey: Finding the Path Home to Ithaca

extern const ncclTuner_v1_t ncclTunerPlugin_v1 = {
    .name = PLUGIN_NAME,
    .init = pluginInit,
    .getCandidate = pluginGetCollInfo,
    .destroy = pluginDestroy,
    .startProfiling = pluginStartProfiling,
    .stopProfiling = pluginStopProfiling,
    .isNewWorkload = pluginIsNewWorkload
    // .workload = {},
    // .candidate = {}
};
