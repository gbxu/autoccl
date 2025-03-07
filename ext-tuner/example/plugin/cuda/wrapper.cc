#include "src/cuda/collectives/nccl_candidate.h"

#define WORKLOAD_SIZE 3
#define CANDIDATE_SIZE 10

// Wrapper function to match Python ctypes expectations
extern "C" void ncclGetValidCandidatesWrapper(
    const GroupInfo* groupInfo,
    const std::pair<const char*, int>* pairs, size_t pairs_size,
    const uint64_t* workloadElemPtr, bool scale2, int32_t** candidateElemPtr, size_t* candidateElemCount) {
  Info myInfo;

  std::map<std::string, int32_t> tunerEnvs;
  for (size_t i = 0; i < pairs_size; ++i) {
      tunerEnvs[pairs[i].first] = pairs[i].second;
  }
  std::unordered_map<GIDTYPE, GroupInfo> allGroupInfos;
  allGroupInfos[groupInfo->groupID] = GroupInfo(
    groupInfo->groupID,
    groupInfo->root,
    groupInfo->rank,
    groupInfo->nrank,
    groupInfo->nnode,
    tunerEnvs);

  Workload workload(workloadElemPtr, workloadElemPtr+WORKLOAD_SIZE);
  std::vector<Candidate> candidates;
  std::vector<ConfigRange> configRanges;

  ncclGetValidCandidates(myInfo, allGroupInfos, workload, scale2, &candidates, &configRanges);

  *candidateElemCount = candidates.size() * CANDIDATE_SIZE;
  *candidateElemPtr = new int32_t[*candidateElemCount];

  size_t index = 0;
  for (const auto& candidate : candidates) {
    std::copy(candidate.begin(), candidate.end(), *candidateElemPtr + index);
    index += CANDIDATE_SIZE;
  }
}

// Function to free allocated candidate memory
extern "C" void freeCandidates(int32_t* candidateElemPtr) {
    delete[] candidateElemPtr;
}
