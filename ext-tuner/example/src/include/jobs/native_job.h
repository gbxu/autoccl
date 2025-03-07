#pragma once
#include "src/include/datatype.h"
#include "src/include/internal/logging.h"
#include "src/include/jobs/job.h"

#define DEBUG_DIST 0
struct NativeJob : public Job {
    NativeJob(Tuner *tuner, const GIDTYPE &groupID, const Workload &workload,
              const Candidate &startCandidate,
              std::vector<Candidate> &&validCandidates, std::vector<ConfigRange> configRanges,
              const int32_t warmupSteps, const int32_t nativeSteps, const int32_t pretrainSteps,
              const int32_t trainSteps, const int32_t roundMaxSteps,
              const int32_t optimumExpireSteps, const int32_t expireSteps)
        : Job(tuner, groupID, workload, startCandidate,
              std::move(validCandidates), configRanges, 0, 0, 0,
              0, roundMaxSteps, optimumExpireSteps, expireSteps) {
    }
    JobType getType() const override {
      return JobType::NativeJob;
    }
    static bool isDecentralized() { return DEBUG_DIST == 0; }
    void createResult(const SimpleQueryStatus &status, Result *result) override;

  protected:
    void addTrain(Result *jobResult) override {};
};

void NativeJob::createResult(const SimpleQueryStatus &status, Result *result) {
  if (this->isDone)
    return;
  this->isDone = true;
  result->workload = this->workload;
  result->roundCandidates.version = this->nextVersion++;
  result->roundCandidates.roundExpire = ExpireCandidate::FOREVER;
  auto nativeEC =
      ExpireCandidate(ExpireCandidate::FOREVER, this->startCandidate);
  this->optimum = nativeEC.candidate;
  result->roundCandidates.expireCandidates.push_back(std::move(nativeEC));
  this->currTotalExpire = ExpireCandidate::FOREVER;
  INFO(Logger::LogSubSys::OPTIMIZER) << " add permanent:" << toDebugStr(this->workload)
       << " candidate:" << toDebugStr(this->startCandidate);
}
