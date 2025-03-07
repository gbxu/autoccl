#include <map>
#include <set>
#include <memory>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>
#include "src/include/jobs/native_job.h"

template <typename T, typename Enable = void> struct NcclCombinedJob;
template <typename T>
using IsEligibleJob =
    typename std::enable_if<std::is_base_of<Job, T>::value &&
                            !std::is_same<NativeJob, T>::value>::type;

template <typename T> struct NcclCombinedJob<T, IsEligibleJob<T>> : public Job {
    static bool isDecentralized() { return T::isDecentralized(); }
    NcclCombinedJob(Tuner *tuner, const GIDTYPE &groupID,
                    const Workload &workload, const Candidate &startCandidate,
                    std::vector<Candidate> &&validCandidates, std::vector<ConfigRange> configRanges,
                    const int32_t warmupSteps, const int32_t nativeSteps, const int32_t pretrainSteps,
                    const int32_t trainSteps, const int32_t roundMaxSteps,
                    const int32_t optimumExpireSteps,
                    const int32_t expireSteps)
        : Job(tuner, groupID, workload, startCandidate, std::move(validCandidates), configRanges,
              warmupSteps, nativeSteps, pretrainSteps, trainSteps, roundMaxSteps,
              optimumExpireSteps, expireSteps) {
      this->parallel = Environment::get()->find("TUNER_COMBINE_PARALLEL", 0) > 0;
      std::set<CandidateFilter> filters = {
          {NCCL_ALGO_TREE, NCCL_PROTO_LL, NCCL_SM_COPY, PATH_PXB},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, NCCL_SM_COPY, PATH_PXB},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, NCCL_SM_COPY, PATH_PXB},
          {NCCL_ALGO_RING, NCCL_PROTO_LL, NCCL_SM_COPY, PATH_PXB},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, NCCL_SM_COPY, PATH_PXB},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, NCCL_SM_COPY, PATH_PXB},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, NCCL_COPY_ENGINE, PATH_PXB},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, NCCL_COPY_ENGINE, PATH_PXB},

          {NCCL_ALGO_TREE, NCCL_PROTO_LL, NCCL_SM_COPY, PATH_SYS},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, NCCL_SM_COPY, PATH_SYS},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, NCCL_SM_COPY, PATH_SYS},
          {NCCL_ALGO_RING, NCCL_PROTO_LL, NCCL_SM_COPY, PATH_SYS},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, NCCL_SM_COPY, PATH_SYS},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, NCCL_SM_COPY, PATH_SYS},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, NCCL_COPY_ENGINE, PATH_SYS},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, NCCL_COPY_ENGINE, PATH_SYS},

          {NCCL_ALGO_TREE, NCCL_PROTO_LL, NCCL_SM_COPY, PATH_PHB},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, NCCL_SM_COPY, PATH_PHB},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, NCCL_SM_COPY, PATH_PHB},
          {NCCL_ALGO_RING, NCCL_PROTO_LL, NCCL_SM_COPY, PATH_PHB},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, NCCL_SM_COPY, PATH_PHB},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, NCCL_SM_COPY, PATH_PHB},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, NCCL_COPY_ENGINE, PATH_PHB},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, NCCL_COPY_ENGINE, PATH_PHB},

          {NCCL_ALGO_TREE, NCCL_PROTO_LL, NCCL_SM_COPY, PATH_PIX},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, NCCL_SM_COPY, PATH_PIX},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, NCCL_SM_COPY, PATH_PIX},
          {NCCL_ALGO_RING, NCCL_PROTO_LL, NCCL_SM_COPY, PATH_PIX},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, NCCL_SM_COPY, PATH_PIX},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, NCCL_SM_COPY, PATH_PIX},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, NCCL_COPY_ENGINE, PATH_PIX},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, NCCL_COPY_ENGINE, PATH_PIX},

          {NCCL_ALGO_TREE, NCCL_PROTO_LL, NCCL_SM_COPY, PATH_LOC},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, NCCL_SM_COPY, PATH_LOC},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, NCCL_SM_COPY, PATH_LOC},
          {NCCL_ALGO_RING, NCCL_PROTO_LL, NCCL_SM_COPY, PATH_LOC},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, NCCL_SM_COPY, PATH_LOC},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, NCCL_SM_COPY, PATH_LOC},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, NCCL_COPY_ENGINE, PATH_LOC},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, NCCL_COPY_ENGINE, PATH_LOC},
      };
      configRanges[0].used = false;
      configRanges[1].used = false;
      configRanges[2].used = false;
      configRanges[3].used = false;
      // CHECK(filters.size() == 16);

      std::map<CandidateFilter, std::vector<Candidate>> filteredCandidates;
      for (auto &filter : filters) {
        filteredCandidates[filter] = std::vector<Candidate>();
      }

      auto mapFilterCandidates = [](
          const std::vector<Candidate> &validCandidates,
          std::map<CandidateFilter, std::vector<Candidate>> *filteredCandidates) {
        for (const auto &candidate : validCandidates) {
          (*filteredCandidates)[{candidate[0], candidate[1], candidate[2], candidate[3]}].push_back(
              candidate);
        }
      };

      mapFilterCandidates(this->validCandidates, &filteredCandidates);

      int32_t filterCount = 0;
      for (auto &filter : filters) {
        if (filteredCandidates[filter].size()) {
          filterCount++;
        }
      }
      if (filterCount > 0) {
        this->subPretrainSteps = std::max(pretrainSteps/filterCount, 1);
        this->subTrainSteps = std::max(trainSteps/filterCount, 1);
      }

      this->askedProfiling = this->warmupSteps + this->nativeSteps;
      for (auto &filter : filters) {
        if (filteredCandidates[filter].size()) {
          TRACE(Logger::LogSubSys::OPTIMIZER) << " filter=" << toDebugStr(filter) << " filteredCandidates[0]="
                << toDebugStr(filteredCandidates[filter][0]);
          subjobs[filter] = std::make_unique<T>(
              tuner, groupID, workload, startCandidate,
              std::move(filteredCandidates[filter]), configRanges, 0, 0,
              this->subPretrainSteps, this->subTrainSteps, this->roundMaxSteps,
              this->optimumExpireSteps, this->expireSteps);
          this->askedProfiling += subjobs[filter]->askedProfiling;
          this->validFilters.push_back(filter);
        }
      }
      // CHECK(subjobs.size() > 0);

      int32_t moreSteps = (this->subPretrainSteps+this->subTrainSteps)*filterCount - this->pretrainSteps - this->trainSteps;
      if (moreSteps > 0) {
        WARN(Logger::LogSubSys::OPTIMIZER) << "Add more pretrain or train steps for NcclCombinedJob" << " :" << moreSteps;
      }

    }

    void addRecord(const Record &record) override;
    void createResult(const SimpleQueryStatus &status, Result *result) override;
    std::string debugStr() const override;

    JobType getType() const override {
      return JobType::NcclCombinedJob;
    }

  protected:
    void addTrain(Result *jobResult) override {};
  private:
    std::vector<CandidateFilter> validFilters;
    std::map<CandidateFilter, std::unique_ptr<T>> subjobs;
    std::map<CandidateFilter, bool> subjobDone;
    bool parallel = false;
    int32_t subPretrainSteps;
    int32_t subTrainSteps;
    int subjobFilterIndex = 0; // next visit subjob filter index
    void callSubjobs(const SimpleQueryStatus &status, Result *mergedResult);
};

template <typename T>
std::string NcclCombinedJob<T, IsEligibleJob<T>>::debugStr() const {
  std::stringstream ss;  // NOLINT
  ss << Job::debugStr();
  ss << " + NcclCombinedJob[";
  ss << " subPretrainSteps=" << subPretrainSteps << "; ";
  ss << " subTrainSteps=" << subTrainSteps << "; ";
  ss << "]";
  for (auto &pair : this->subjobs) {
    auto &subjob = pair.second;
    ss << "subjob=" << subjob->debugStr() << "; ";
  }
  return ss.str();
}

template <typename T>
void NcclCombinedJob<T, IsEligibleJob<T>>::addRecord(const Record &record) {
  if (this->currDropProfiled < this->warmupSteps) {
    this->currDropProfiled++;
    this->currTotalProfiled++;
  } else {
    CHECK(record.value.repeat == 1);
    CHECK(record.key.candidate.size() > 0);
    if (this->recordsTable.count(record.key) == 0) {
      this->recordsTable[record.key] = RecordValueList();
    }
    this->recordsTable[record.key].push_back(record.value);
    if (this->currTotalProfiled >= this->warmupSteps+this->nativeSteps) {
      auto &candidate = record.key.candidate;
      const auto &filter = {candidate[0], candidate[1], candidate[2], candidate[3]};
      if (subjobs.count(filter) > 0) {
        subjobs[filter]->addRecord(record);
      }
    }
    this->currTotalProfiled++;
  }
  TRACE(Logger::LogSubSys::OPTIMIZER) << " add record: " << record.debugStr()
    << " currDropProfiled=" << this->currDropProfiled
    << " currTotalProfiled=" << this->currTotalProfiled;
}

template <typename T>
void NcclCombinedJob<T, IsEligibleJob<T>>::createResult(
    const SimpleQueryStatus &status, Result *result) {
  if (this->isDone)
    return;
  result->workload = this->workload;

  if (this->currTotalExpire < this->warmupSteps + this->nativeSteps) {
    // -> warmup
    // add native candidate as warmup
    this->optimum = this->startCandidate;
    addWarmup(result);
  } else {
    callSubjobs(status, result);
  }

  // make a new version for the combined job
  if (result->roundCandidates.expireCandidates.size()) {
    result->roundCandidates.version = this->nextVersion++;
    TRACE(Logger::LogSubSys::OPTIMIZER) << " generate merged: " << result->debugStr()
          << ", job=" << this->debugStr();
  }
}

template <typename T>
void NcclCombinedJob<T, IsEligibleJob<T>>::callSubjobs(const SimpleQueryStatus &status, Result *mergedResult) {
  int endSubjobFilterIndex = this->subjobFilterIndex;
  CHECK(this->validFilters.size() > 0);
  do {
    auto &subjob = this->subjobs[this->validFilters[this->subjobFilterIndex]];
    auto tempResult = Result();
    subjob->createResult(status, &tempResult);

    // merge tempResult
    for (auto &expireCandidate : tempResult.roundCandidates.expireCandidates) {
      if (expireCandidate.expire == ExpireCandidate::FOREVER) {
        if (this->optimumExpireSteps > 0) {
          expireCandidate.expire = this->optimumExpireSteps;
        } else {
          continue;
        }
      }
      mergedResult->roundCandidates.roundExpire += expireCandidate.expire;
      mergedResult->roundCandidates.expireCandidates.push_back(expireCandidate);
    }

    // check optimum
    if (this->currTotalProfiled  > this->warmupSteps+this->nativeSteps) {
      const auto &optimumRecordKey =
          RecordKey(this->groupID, this->workload, this->optimum);
      CHECK(this->recordsTable.count(optimumRecordKey) != 0);
      float globalMin = this->recordsTable[optimumRecordKey].meanOfTopPercent(75);

      const auto &recordKey =
          RecordKey(this->groupID, this->workload, subjob->optimum);
      if (subjob->isDone) {
        CHECK(subjob->recordsTable.count(recordKey) != 0);
        float localMin = subjob->recordsTable[recordKey].meanOfTopPercent(75);
        this->subjobDone[this->validFilters[this->subjobFilterIndex]] = true;
        if (localMin < globalMin) {
          this->optimum = recordKey.candidate;
        }
      } else if (subjob->currOptimumBased > subjob->warmupSteps + subjob->nativeSteps + subjob->pretrainSteps + subjob->trainSteps * 0.1) {
        CHECK(subjob->recordsTable.count(recordKey) != 0);
        float localMin = subjob->recordsTable[recordKey].meanOfTopPercent(75);
        if (localMin > globalMin * 1.5) {
          this->subjobDone[this->validFilters[this->subjobFilterIndex]] = true;
          subjob->isDone = true;
          WARN(Logger::LogSubSys::OPTIMIZER) << " too slow, exit early. subjob=" << subjob->debugStr();
        }
      } else if (subjob->currOptimumBased > subjob->warmupSteps + subjob->nativeSteps + subjob->pretrainSteps + subjob->expireSteps * 5.0) {
        CHECK(subjob->recordsTable.count(recordKey) != 0);
        float localMin = subjob->recordsTable[recordKey].meanOfTopPercent(75);
        if (localMin > globalMin * 3) {
          this->subjobDone[this->validFilters[this->subjobFilterIndex]] = true;
          subjob->isDone = true;
          WARN(Logger::LogSubSys::OPTIMIZER) << " extremely slow, exit early. subjob=" << subjob->debugStr();
        }
      }
    }
    this->subjobFilterIndex = (this->subjobFilterIndex + 1) % this->validFilters.size();
    if (!this->parallel && mergedResult->roundCandidates.expireCandidates.size()) {
      break;
    }
  } while (this->subjobFilterIndex != endSubjobFilterIndex);

  if (this->subjobDone.size() == this->subjobs.size()) {
    this->updateOptimum();

    // set forever for the optimum
    this->isDone = true;
    mergedResult->roundCandidates.expireCandidates.clear();
    mergedResult->roundCandidates.roundExpire = ExpireCandidate::FOREVER;
    auto bestEC = ExpireCandidate(ExpireCandidate::FOREVER, this->optimum);
    mergedResult->roundCandidates.expireCandidates.push_back(bestEC);
    INFO(Logger::LogSubSys::OPTIMIZER) << " add permanent:" << toDebugStr(this->workload)
          << " candidate:" << toDebugStr(this->optimum);
  }
}
