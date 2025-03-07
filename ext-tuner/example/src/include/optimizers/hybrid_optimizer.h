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

template <typename T, typename Enable = void> struct HybridOptimizer;

template <typename T>
struct HybridOptimizer<
    T, typename std::enable_if<std::is_base_of<Job, T>::value>::type>
    : public Optimizer {
  public:
    explicit HybridOptimizer(const CandidateFunc &getValidCandidatesFunc)
        : Optimizer(getValidCandidatesFunc) {
      this->getWhiteListRules(&this->whiteListRules);
      this->getWhiteListCases(&this->whiteListCases);
    }
    void addJob(const Info &myInfo,
                const std::unordered_map<GIDTYPE, GroupInfo> &allGroupInfos,
                const GIDTYPE &groupID, const Workload &workload,
                Candidate &startCandidate) {
      // std::lock_guard<std::mutex> lock(this->jobsMutex);
      // pthread_mutex_lock(&this->jobsMutex);
      bool isInWhiteList = this->checkWhiteList(allGroupInfos, groupID, workload);
      if (allGroupInfos.at(groupID).isLeader() || T::isDecentralized() || isInWhiteList) {
        std::vector<Candidate> validCandidates;  // NOLINT
        std::vector<ConfigRange> configRanges;
        this->getValidCandidatesFunc(myInfo, allGroupInfos, workload,
                                     false /*scale2*/, &validCandidates, &configRanges);

        this->jobsMutex.lock();
        CHECK(this->jobs.count(workload) == 0) << "old job";
        if (isInWhiteList) {
          this->jobs[workload] = std::make_unique<NativeJob>(
              tuner, groupID, workload, startCandidate,
              std::move(validCandidates), configRanges, this->warmupSteps, this->expireSteps, this->pretrainSteps,
              this->trainSteps, this->roundMaxSteps, this->optimumExpireSteps,
              this->expireSteps);
          WARN(Logger::LogSubSys::OPTIMIZER) << "Skip tuning workload=" << toDebugStr(workload);
        } else {
          this->jobs[workload] = std::make_unique<T>(
              tuner, groupID, workload, startCandidate,
              std::move(validCandidates), configRanges, this->warmupSteps, this->expireSteps, this->pretrainSteps,
              this->trainSteps, this->roundMaxSteps, this->optimumExpireSteps,
              this->expireSteps);
        }
        int32_t profile_more = Environment::get()->find("TUNER_PROFILE_MORE", 0);
        this->tuner->timer->setProfiling(workload,
                                         this->jobs[workload]->askedProfiling+profile_more);
        if (this->jobs[workload]->getType() == JobType::NativeJob) {
          INFO(Logger::LogSubSys::OPTIMIZER) << "In WhiteList, New Native job:"
              << dynamic_cast<NativeJob *>(this->jobs[workload].get())->debugStr();
        } else {
          INFO(Logger::LogSubSys::OPTIMIZER) << "New T job:"
              << dynamic_cast<T *>(this->jobs[workload].get())->debugStr();
        }
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
        if (this->jobs[workload]->getType() == JobType::NativeJob) {
          auto job = dynamic_cast<NativeJob *>(this->jobs[workload].get());
          job->addRecord(record);
        } else {
          auto job = dynamic_cast<T *>(this->jobs[workload].get());
          job->addRecord(record);
        }
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
        if (T::isDecentralized() || this->jobs[workload]->getType() == JobType::NativeJob) {
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
        if (T::isDecentralized() || this->jobs[workload]->getType() == JobType::NativeJob) {
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

  private:
    std::vector<std::vector<KeyRange>> whiteListRules;
    std::vector<std::pair<int32_t, Workload>> whiteListCases;

    bool checkWhiteList(const std::unordered_map<GIDTYPE, GroupInfo> &allGroupInfos,
      const GIDTYPE &groupID, const Workload &workload) {
      bool isInWhiteList = false;
      if (this->whiteListRules.size()) {
        for (const auto & whiteListRule : this->whiteListRules) {
          CHECK(1 + workload.size() == whiteListRule.size());
          // check cluster
          int32_t nrank = allGroupInfos.at(groupID).nrank;
          if (whiteListRule[0].begin != KeyRange::EMPTY && whiteListRule[0].end != KeyRange::EMPTY) {
            if (whiteListRule[0].begin > nrank || nrank > whiteListRule[0].end) {
              // not in this range, check the next rule
              continue;
            }
          }
          // check workload
          bool workloadMatch = true;
          for (size_t i = 0; i < workload.size(); i++) {
            if (whiteListRule[i+1].begin != KeyRange::EMPTY && whiteListRule[i+1].end != KeyRange::EMPTY) {
              if (whiteListRule[i+1].begin > workload[i] || workload[i] > whiteListRule[i+1].end) {
                // not in this range, check the next rule
                workloadMatch = false;
                break;
              }
            }
          }
          if (!workloadMatch) {
            // no match, try the next case
            continue;
          }
          // match all ranges
          isInWhiteList = true;
          break;
        }
      }
      if (this->whiteListCases.size() && !isInWhiteList) {
        for (const auto & whiteListCase : this->whiteListCases) {
          int32_t whiteListNrank = whiteListCase.first;
          Workload whiteListWorkload = whiteListCase.second;
          // check cluster
          int32_t nrank = allGroupInfos.at(groupID).nrank;
          if (whiteListNrank != -1 && nrank != whiteListNrank) {
            // no match, try the next case
            continue;
          }
          // check workload
          if (!this->compare(workload, whiteListWorkload)) {
            // no match, try the next case
            continue;
          }
          // match all ranges
          isInWhiteList = true;
          break;
        }
      }
      return isInWhiteList;
    }

    void getWhiteListRules(std::vector<std::vector<KeyRange>> *rules) {
      // example:
      // do not add any comments in each line
      // nrank range; coll range; sizePerRank range;
      // 8 8; 4 4; 1024 1024;
      // ;  ; 1024 1024;
      std::vector<std::string> lines;
      // all nodes should have this file.
      const char* str = Environment::get()->find("TUNER_WHITELIST_RULES_FILE", "");
      if (str && strlen(str) > 0) {
        std::string filename = std::string(str);
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
          lines.push_back(line);
        }
        file.close();
      }

      for (auto line : lines) {
        std::vector<KeyRange> ranges;
        std::istringstream iss(line);
        std::string rangeStr;
        while (std::getline(iss, rangeStr, ';')) {
          std::istringstream rangeIss(rangeStr);
          // [start, end]
          KEY start, end;
          if (rangeIss >> start >> end) {
            ranges.push_back(KeyRange(start, end));
          } else {
            // empyt
            ranges.push_back(KeyRange());
          }
        }
        // save rules
        rules->push_back(ranges);
      }
    }

    void getWhiteListCases(std::vector<std::pair<int32_t, Workload>> *nrankWorkloads) {
      // example:
      // do not add any comments in each line
      // nrank; workload;
      // 8 ; -1 4 1024;
      std::vector<std::string> lines;
      // all nodes should have this file.
      const char* str = Environment::get()->find("TUNER_WHITELIST_CASES_FILE", "");
      if (str && strlen(str) > 0) {
        std::string filename = std::string(str);
        std::ifstream file(filename);
        std::string line;
        while (std::getline(file, line)) {
          lines.push_back(line);
        }
        file.close();
      }

      for (auto line : lines) {
        std::istringstream iss(line);
        std::string clusterStr, workloadStr;
        std::getline(iss, clusterStr, ';');
        std::getline(iss, workloadStr, ';');

        std::istringstream clusterIss(clusterStr);
        std::istringstream workloadIss(workloadStr);

        KEY workloadElem;
        Workload workload;
        while (workloadIss >> workloadElem) workload.push_back(workloadElem);

        int32_t nrank;
        clusterIss >> nrank;

        // save whitelist case
        nrankWorkloads->push_back(std::make_pair(nrank, workload));
      }
    }

    template <typename W> bool compare(const W &validData, const W &userData) {
      CHECK(validData.size() == userData.size());
      for(int i=0; i<validData.size(); i++) {
        if (userData[i] == -1) {
          continue;
        } else if (userData[i] != validData[i]) {
          return false;
        }
      }
      return true;
    }

};
