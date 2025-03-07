#pragma once
#include <map>
#include <memory>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include "src/include/jobs/job.h"
#include "src/include/optimizers/optimizer.h"

template <typename T, typename Enable = void> struct UniformOptimizer;

template <typename T>
struct UniformOptimizer<
    T, typename std::enable_if<std::is_base_of<Job, T>::value>::type>
    : public Optimizer {
  public:
    explicit UniformOptimizer(const CandidateFunc &getValidCandidatesFunc)
        : Optimizer(getValidCandidatesFunc) {}
    void addJob(const Info &myInfo,
                const std::unordered_map<GIDTYPE, GroupInfo> &allGroupInfos,
                const GIDTYPE &groupID, const Workload &workload,
                Candidate &startCandidate) {
      // std::lock_guard<std::mutex> lock(this->jobsMutex);
      // pthread_mutex_lock(&this->jobsMutex);
      if (allGroupInfos.at(groupID).isLeader() || T::isDecentralized()) {
        std::vector<Candidate> validCandidates;  // NOLINT
        std::vector<ConfigRange> configRanges;
        this->getValidCandidatesFunc(myInfo, allGroupInfos, workload,
                                     false /*scale2*/, &validCandidates, &configRanges);

        this->jobsMutex.lock();
        CHECK(this->jobs.count(workload) == 0) << "old job";
        this->jobs[workload] = std::make_unique<T>(
            tuner, groupID, workload, startCandidate,
            std::move(validCandidates), configRanges, this->warmupSteps, this->expireSteps, this->pretrainSteps,
            this->trainSteps, this->roundMaxSteps, this->optimumExpireSteps,
            this->expireSteps);
        int32_t profile_more = Environment::get()->find("TUNER_PROFILE_MORE", 0);
        this->tuner->timer->setProfiling(workload,
                                         this->jobs[workload]->askedProfiling+profile_more);
        INFO(Logger::LogSubSys::OPTIMIZER) << "New T job:"
             << dynamic_cast<T *>(this->jobs[workload].get())->debugStr();
        this->jobsMutex.unlock();
      } else {
        this->tuner->timer->setProfiling(workload, 0);
      }
      // pthread_mutex_unlock(&this->jobsMutex);
    }

    void run(const std::map<Workload, SimpleQueryStatus> &status,
             const std::vector<Record> &records, ResultPack *resultPack) {
      std::lock_guard<std::mutex> lock(this->jobsMutex);
      // pthread_mutex_lock(&this->jobsMutex);
      for (const auto &record : records) {
        auto &workload = record.key.workload;
        if (this->jobs.count(workload) == 0)
          continue;  // follower has no optimizing jobs.
        auto job = dynamic_cast<T *>(this->jobs[workload].get());
        job->addRecord(record);
      }

      for (auto &pair : status) {
        const auto &workload = pair.first;
        if (this->jobs.count(workload) == 0)
          continue;  // follower has no optimizing jobs.
        auto jobResult = Result();
        this->jobs[workload]->createResult(pair.second, &jobResult);
        if (jobResult.roundCandidates.expireCandidates.size() == 0) {
          continue;
        }
        if (T::isDecentralized()) {
          // local
          resultPack->localResults.push_back(std::move(jobResult));
        } else {
          if (resultPack->distResults.count(this->jobs[workload]->groupID) ==
              0) {
            resultPack->distResults[this->jobs[workload]->groupID] =
                std::vector<Result>();
          }
          resultPack->distResults[this->jobs[workload]->groupID].push_back(
              std::move(jobResult));
        }
      }
      // pthread_mutex_unlock(&this->jobsMutex);
    }

    void pick(const std::map<Workload, SimpleQueryStatus> &status,
              DecisionPack *decisionPack) {
      std::lock_guard<std::mutex> lock(this->jobsMutex);
      // pthread_mutex_lock(&this->jobsMutex);
      for (auto &pair : status) {
        const auto &workload = pair.first;
        if (this->jobs.count(workload) == 0)
          continue;  // follower has no optimizing job
        auto jobDecision = Decision();
        this->jobs[workload]->makeDecision(pair.second, &jobDecision);
        if (jobDecision.version == RoundCandidates::INITVERSION) {
          continue;
        }
        if (T::isDecentralized()) {
          // local
          decisionPack->localDecisions.push_back(std::move(jobDecision));
        } else {
          // dist: group by groupID for broadcasting
          if (decisionPack->distDecisions.count(
                  this->jobs[workload]->groupID) == 0) {
            decisionPack->distDecisions[this->jobs[workload]->groupID] =
                std::vector<Decision>();
          }
          decisionPack->distDecisions[this->jobs[workload]->groupID].push_back(
              std::move(jobDecision));
        }
      }
      // pthread_mutex_unlock(&this->jobsMutex);
      return;
    }
};
