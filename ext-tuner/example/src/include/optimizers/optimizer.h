#pragma once
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "src/include/jobs/job.h"
#include "src/include/tuner.h"
#include "src/include/internal/env.h"
#include "src/misc/math.h"

class Tuner;

class Optimizer {
  public:
    explicit Optimizer(const CandidateFunc &getValidCandidatesFunc)
        : getValidCandidatesFunc(getValidCandidatesFunc) {
      initEnv();
    }
    void initialize(Tuner *tuner) { this->tuner = tuner; }
    virtual void
    addJob(const Info &myInfo,
           const std::unordered_map<GIDTYPE, GroupInfo> &allGroupInfos,
           const GIDTYPE &groupID, const Workload &workload,
           Candidate &startCandidate) = 0;
    // TODO: multithread, workload-specific
    virtual void run(const std::map<Workload, SimpleQueryStatus> &status,
                     const std::vector<Record> &records,
                     ResultPack *resultPack) = 0;
    virtual void pick(const std::map<Workload, SimpleQueryStatus> &status,
                      DecisionPack *decisionPack) = 0;

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << "mode=" << Environment::get()->find("TUNER_MODE", "-1") << "\n";
      ss << "warmupSteps=" << warmupSteps << "\n";
      ss << "pretrainSteps=" << pretrainSteps << "\n";
      ss << "trainSteps=" << trainSteps << "\n";
      ss << "roundMaxSteps=" << roundMaxSteps << "\n";
      ss << "optimumExpireSteps=" << optimumExpireSteps << "\n";
      ss << "expireSteps=" << expireSteps << "\n";
      std::lock_guard<std::mutex> lock(this->jobsMutex);
      for (auto &pair : this->jobs) {
        const auto &workload = pair.first;
        const auto &job = pair.second;
        ss << job->debugStr() << "\n";;
      }
      return ss.str();
    }

  protected:
    mutable std::mutex jobsMutex;
    // pthread_mutex_t jobsMutex = PTHREAD_MUTEX_INITIALIZER;
    std::map<Workload, std::unique_ptr<Job>> jobs;
    Tuner *tuner;
    CandidateFunc getValidCandidatesFunc;
    int32_t warmupSteps;
    int32_t pretrainSteps;
    int32_t trainSteps;
    int32_t roundMaxSteps;
    int32_t optimumExpireSteps;
    int32_t expireSteps;

  private:
    void initEnv();
};

void Optimizer::initEnv() {
  warmupSteps = Environment::get()->find("TUNER_WARMUP_STEPS", 5);
  CHECK(warmupSteps >= 1);
  pretrainSteps = Environment::get()->find("TUNER_PRETRAIN_STEPS", 20);
  CHECK(pretrainSteps >= 1);
  trainSteps = Environment::get()->find("TUNER_TRAIN_STEPS", 50);
  CHECK(trainSteps >= 1);
  roundMaxSteps = Environment::get()->find("TUNER_ROUND_MAX_STEPS", 10);
  CHECK(roundMaxSteps >= 1);
  optimumExpireSteps = Environment::get()->find("TUNER_OPTIMUM_EXPIRE", 10);
  expireSteps = Environment::get()->find("TUNER_PROFILE_REPEAT", 1);
  CHECK(pretrainSteps + trainSteps >= expireSteps);
  CHECK(expireSteps > 0);
  INFO(Logger::LogSubSys::OPTIMIZER) << "set warmupSteps=" << warmupSteps
       << " pretrainSteps=" << pretrainSteps << " trainSteps=" << trainSteps
       << " roundMaxSteps=" << roundMaxSteps
       << " optimumExpireSteps=" << optimumExpireSteps
       << " expireSteps=" << expireSteps;
}
