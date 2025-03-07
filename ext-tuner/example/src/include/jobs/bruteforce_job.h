#pragma once
#include "src/include/datatype.h"
#include "src/include/internal/logging.h"
#include "src/include/jobs/job.h"

struct BFJob : public Job {
    static bool isDecentralized() { return false; }
    BFJob(Tuner *tuner, const GIDTYPE &groupID, const Workload &workload,
          const Candidate &startCandidate,
          std::vector<Candidate> &&validCandidates, std::vector<ConfigRange> configRanges, const int32_t warmupSteps, const int32_t nativeSteps,
          const int32_t pretrainSteps, const int32_t trainSteps,
          const int32_t roundMaxSteps, const int32_t optimumExpireSteps,
          const int32_t expireSteps) : Job(tuner, groupID, workload, startCandidate, std::move(validCandidates), configRanges,
          warmupSteps, nativeSteps, pretrainSteps+trainSteps, 0, roundMaxSteps,
          optimumExpireSteps, expireSteps) {
      if (this->roundMaxSteps > this->pretrainSteps) {
        WARN(Logger::LogSubSys::OPTIMIZER) << "set pretrainSteps at least: " << this->roundMaxSteps;
      }
      this->accessByRand = Environment::get()->find("TUNER_BRUTEFOROCE_RAND", 1) > 0;
    }
    std::string debugStr() const override;

    JobType getType() const override {
      return JobType::BFJob;
    }

  protected:
    void addTrain(Result *jobResult) override {};
};

std::string BFJob::debugStr() const {
  std::stringstream ss;  // NOLINT
  ss << Job::debugStr();
  ss << " + BFJob[";
  ss << "]";
  return ss.str();
}
