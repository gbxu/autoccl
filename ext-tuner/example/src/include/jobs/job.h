#pragma once
#include <random>
#include "src/include/datatype.h"
#include "src/include/internal/logging.h"

enum class JobType {
  NativeJob,
  BFJob,
  SpecificJob,
  SimAnnealJob,
  CoDesJob,
  NcclCombinedJob,
};

class Tuner;
struct Job {
    Job(Tuner *tuner, const GIDTYPE &groupID, const Workload &workload,
        const Candidate &startCandidate,
        std::vector<Candidate> &&validCandidates, std::vector<ConfigRange> configRanges, const int32_t warmupSteps, const int32_t nativeSteps,
        const int32_t pretrainSteps, const int32_t trainSteps,
        const int32_t roundMaxSteps, const int32_t optimumExpireSteps,
        const int32_t expireSteps)
        : tuner(tuner), groupID(groupID), workload(workload),
          startCandidate(startCandidate),
          validCandidates(std::move(validCandidates)), configRanges(configRanges), warmupSteps(warmupSteps), nativeSteps(nativeSteps),
          pretrainSteps(pretrainSteps), trainSteps(trainSteps),
          roundMaxSteps(roundMaxSteps), optimumExpireSteps(optimumExpireSteps),
          expireSteps(expireSteps), restCandidates(this->validCandidates),
          gen(rd()), dis(0, this->validCandidates.size() - 1) {
      this->askedProfiling = Environment::get()->find("TUNER_PROFILE_ASKED", warmupSteps + nativeSteps + pretrainSteps + trainSteps);

      if (this->pretrainSteps%this->expireSteps != 0) {
        WARN(Logger::LogSubSys::OPTIMIZER) << "pretrainSteps " << this->pretrainSteps
          << " % expireSteps " << this->expireSteps << " != 0 may bring timer bias, especially for 'Broadcast' and 'Reduce'.";
      }

      if (this->trainSteps%this->expireSteps != 0) {
        WARN(Logger::LogSubSys::OPTIMIZER) << "trainSteps " << this->trainSteps
          << " % expireSteps " << this->expireSteps << " != 0 may bring timer bias for Broadcast.";
      }

      if (this->pretrainSteps/this->expireSteps < this->validCandidates.size() * 0.01) {
        INFO(Logger::LogSubSys::OPTIMIZER) << "pretrainSteps " << this->pretrainSteps
          << " / expireSteps " << this->expireSteps << "=" << this->pretrainSteps/this->expireSteps << " :" << " add more pretrainSteps may help.";
      }

      /******* traverse or random **** */
      int32_t visitCount = this->pretrainSteps / expireSteps;
      // The probability that an item is not accessed
      double noAccessProbViaRand = pow((this->validCandidates.size() - 1) * 1.0 /
                                          this->validCandidates.size(),
                                      visitCount);
      double noAccessProbViaSeq =
          this->validCandidates.size() > visitCount
              ? (this->validCandidates.size() - visitCount) * 1.0 /
                    this->validCandidates.size()
              : 1;
      if (noAccessProbViaRand > noAccessProbViaSeq) {
        this->accessByRand = false;
      } else {
        this->accessByRand = true;
      }
      this->visitStride =
          std::max(static_cast<int32_t>(std::floor(this->validCandidates.size() *
                                                  1.0 / visitCount)),
                  1);
      this->currVisitIndex = dis(gen);
    }
    Job() = default;
    virtual JobType getType() const = 0;
    virtual void createResult(const SimpleQueryStatus &status,
                              Result *result);
    virtual void makeDecision(const SimpleQueryStatus &status,
                              Decision *decision);
    virtual void addRecord(const Record &record);
    virtual std::string debugStr() const;

    GIDTYPE groupID;

    bool isDone = false;
    bool convergence = false;
    Candidate optimum;
    std::map<RecordKey, RecordValueList> recordsTable;
    int32_t askedProfiling;

    int32_t warmupSteps;
    int32_t nativeSteps;
    int32_t pretrainSteps;
    int32_t trainSteps;
    int32_t roundMaxSteps;
    int32_t optimumExpireSteps;
    int32_t expireSteps;

    int32_t currOptimumBased = 0;   // for after-train generating
    int32_t currTotalExpire = 0;           // trace generaing
  protected:
    Tuner *tuner;
    Workload workload;
    Candidate startCandidate;
    std::vector<Candidate> validCandidates;
    std::vector<ConfigRange> configRanges;

    int32_t currTotalProfiled = 0;  // for after-train generating
    int32_t currDropProfiled = 0;   // for dropping warmup

    int32_t nextVersion = 0;
    int32_t lastUrgent =
        RoundCandidates::INITVERSION - 1;  // for making decision

    std::vector<Candidate> restCandidates;
    void updateRestCandidates(Candidate &currCandidate);

    int32_t getNextIndex();
    int32_t getRandIndex();

    void addWarmup(Result *jobResult);
    void addPretrain(Result *jobResult);
    virtual void addTrain(Result *jobResult) = 0;
    void addBest(Result *jobResult);
    void updateOptimum();

    bool accessByRand;
  private:
    int32_t currVisitIndex;  // access by seq
    int32_t visitStride;     // access by seq
    std::random_device rd;  // the seed of getting random numbers
    std::mt19937 gen;       // Mersenne Twister
    std::uniform_int_distribution<> dis;
    bool hasProfiledAll();
};

std::string Job::debugStr() const {
  std::stringstream ss;  // NOLINT
  ss << "[Job:";
  ss << " groupID=" << groupID << ";";
  ss << " workload=" << toDebugStr(workload) << "; ";
  ss << " startCandidate=" << toDebugStr(startCandidate) << "; ";
  ss << " validCandidates.size=" << validCandidates.size() << "; ";
  ss << " warmupSteps=" << warmupSteps << "; ";
  ss << " nativeSteps=" << nativeSteps << "; ";
  ss << " pretrainSteps=" << pretrainSteps << "; ";
  ss << " trainSteps=" << trainSteps << "; ";
  ss << " roundMaxSteps=" << roundMaxSteps << "; ";
  ss << " optimumExpireSteps=" << optimumExpireSteps << "; ";
  ss << " expireSteps=" << expireSteps << "; ";
  ss << " isDone=" << isDone << "; ";
  ss << " currOptimumBased=" << currOptimumBased << "; ";
  ss << " currTotalProfiled=" << currTotalProfiled << "; ";
  ss << " currDropProfiled=" << currDropProfiled << "; ";
  ss << " nextVersion=" << nextVersion << "; ";
  ss << " lastUrgent=" << lastUrgent << "; ";
  ss << " currTotalExpire=" << currTotalExpire << "; ";
  ss << " askedProfiling=" << askedProfiling << "; ";
  ss << " optimum=" << toDebugStr(optimum) << "; ";
  ss << " accessByRand=" << accessByRand << "; ";
  ss << " currVisitIndex=" << currVisitIndex << "; ";
  ss << " visitStride=" << visitStride << "; ";
  ss << "]";
  return ss.str();
}

void Job::updateRestCandidates(Candidate &currCandidate) {
  auto it = std::find(this->restCandidates.begin(), this->restCandidates.end(), currCandidate);
  if (it != this->restCandidates.end()) this->restCandidates.erase(it);
}

int32_t Job::getRandIndex() {
  return this->dis(this->gen);
}

int32_t Job::getNextIndex() {
  if (this->accessByRand) {
    return this->getRandIndex();
  } else {
    int32_t curr = this->currVisitIndex;
    this->currVisitIndex = (this->currVisitIndex + this->visitStride) %
                           this->validCandidates.size();
    return curr;
  }
}

void Job::addRecord(const Record &record) {
  if (this->currDropProfiled < this->warmupSteps) {
    this->currDropProfiled++;
    this->currTotalProfiled++;
  } else {
    CHECK(record.value.repeat == 1);
    this->currTotalProfiled++;
    if (this->recordsTable.count(record.key) == 0) {
      this->recordsTable[record.key] = RecordValueList();
    }
    this->recordsTable[record.key].push_back(record.value);
  }
  TRACE(Logger::LogSubSys::OPTIMIZER) << " add record: " << record.debugStr()
    << " currDropProfiled=" << this->currDropProfiled
    << " currTotalProfiled=" << this->currTotalProfiled
    << " askedProfiling=" << this->askedProfiling
    << " currTotalExpire=" << this->currTotalExpire
    << " optimum=" << toDebugStr(this->optimum);
}

void Job::createResult(const SimpleQueryStatus &status, Result *result) {
  if (this->isDone)
    return;
  result->workload = this->workload;

  /****** generate ********/
  if (this->currTotalExpire < this->warmupSteps + this->nativeSteps) {
    // -> warmup
    addWarmup(result);
  } else if ((this->currTotalExpire <
              this->warmupSteps + this->nativeSteps + this->pretrainSteps) &&
             !this->restCandidates.empty()) {
    // warmup -> pretrain
    // generate directly
    addPretrain(result);
  } else if ((this->currTotalExpire < this->warmupSteps + this->nativeSteps +
                                          this->pretrainSteps +
                                          this->trainSteps) &&
             !this->restCandidates.empty() && !this->convergence) {
    // pretrain -> train
    // based on records
    if (!hasProfiledAll()) return;
    this->updateOptimum();
    addTrain(result);
  } else {
    // trained, use the best
    if (!hasProfiledAll()) return;
    this->updateOptimum();
    addBest(result);
  }
  if (result->roundCandidates.expireCandidates.size()) {
    result->roundCandidates.version = this->nextVersion++;
    TRACE(Logger::LogSubSys::OPTIMIZER) << " generate: " << result->debugStr()
          << ", job=" << this->debugStr();
  }
}

void Job::makeDecision(const SimpleQueryStatus &status, Decision *decision) {
  const auto &remainingExpire = status.remainingExpire;
  const auto &activeVersion = status.activeVersion;
  if (remainingExpire == ExpireCandidate::FOREVER)
    return;
  int32_t latestVersion = this->nextVersion - 1;
  if (latestVersion == RoundCandidates::INITVERSION)
    return;  // has not yet created results.
  if (activeVersion <= this->lastUrgent)
    return;  // had made decision for this->lastUrgent before
  if (activeVersion >= latestVersion)
    return;  // no new data
  // when: this->lastUrgent < activeVersion < latestVersion
  // make decision
  decision->workload = this->workload;
  if (this->isDone) {
    // use the latest candidate with ExpireCandidate::FOREVER, dropping the old optimums
    decision->version = latestVersion;
  } else {
    int32_t lastDecision = this->lastUrgent+1;
    // use every result, so move 1 step each time to avoid dropping results
    decision->version = lastDecision+1;
  }
  this->lastUrgent = activeVersion;  // set new lastUrgent
  TRACE(Logger::LogSubSys::OPTIMIZER) << " make decision: " << decision->debugStr() << ", then "
        << this->debugStr();
}

void Job::updateOptimum() {
  // update optimum
  float min = std::numeric_limits<float>::max();
  for (const auto &pair : this->recordsTable) {
    const auto &recordKey = pair.first;
    const auto &recordValueList = pair.second;
    auto curr = recordValueList.meanOfTopPercent(75);
    if (curr < min) {
      this->optimum = recordKey.candidate;
      min = curr;
    }
  }
  this->currOptimumBased = this->currTotalProfiled;
}

void Job::addWarmup(Result *jobResult) {
  // add native candidate as warmup
  auto warmupEC = ExpireCandidate(this->warmupSteps, this->startCandidate);
  jobResult->roundCandidates.roundExpire += this->warmupSteps;
  jobResult->roundCandidates.expireCandidates.push_back(std::move(warmupEC));
  this->currTotalExpire += this->warmupSteps;
  // add native candidate
  auto nativeEC = ExpireCandidate(this->nativeSteps, this->startCandidate);
  jobResult->roundCandidates.roundExpire += this->nativeSteps;
  jobResult->roundCandidates.expireCandidates.push_back(std::move(nativeEC));
  this->currTotalExpire += this->nativeSteps;
}

void Job::addPretrain(Result *jobResult) {
  // generate directly by random
  while (true) {
    int32_t expireSteps = std::min(this->warmupSteps+this->nativeSteps+this->pretrainSteps-this->currTotalExpire, this->expireSteps);
    if (expireSteps <= 0 || this->restCandidates.empty()) {
      break;
    }
    auto randCandidate = this->validCandidates[this->getNextIndex()];
    auto tempEC = ExpireCandidate(expireSteps, randCandidate);
    jobResult->roundCandidates.roundExpire += expireSteps;
    jobResult->roundCandidates.expireCandidates.push_back(tempEC);
    this->updateRestCandidates(randCandidate);
    this->currTotalExpire += expireSteps;
  }
}

bool Job::hasProfiledAll() {
  // based on a round
  // bool hasNewRoundRecord =
  //     this->currTotalProfiled >= this->currOptimumBased + this->roundMaxSteps;
  // if (hasNewRoundRecord)
  //   return true;
  bool profiledAll = this->currTotalProfiled >= std::min(this->askedProfiling, this->currTotalExpire);
  if (profiledAll) {
    TRACE(Logger::LogSubSys::OPTIMIZER) << " has profiled all:"
      << " currTotalProfiled=" << this->currTotalProfiled
      << " askedProfiling=" << this->askedProfiling
      << " currTotalExpire=" << this->currTotalExpire;
    return true;
  }
  return false;
}

void Job::addBest(Result *jobResult) {
  // update job status
  this->isDone = this->isDone ||
    this->currTotalProfiled >= this->currTotalExpire ||
    this->restCandidates.empty() ||
    this->convergence;

  // trained, use the best
  int32_t expire;
  if (this->isDone) {
    expire = ExpireCandidate::FOREVER;
  } else if (this->optimumExpireSteps > 0) {
    expire = this->optimumExpireSteps;
  } else {
    return;
  }
  auto bestEC = ExpireCandidate(expire, this->optimum);
  jobResult->roundCandidates.roundExpire = expire;
  jobResult->roundCandidates.expireCandidates.push_back(bestEC);
  if (expire != ExpireCandidate::FOREVER) {
    this->currTotalExpire += expire;
  } else {
    INFO(Logger::LogSubSys::OPTIMIZER) << " add permanent:" << toDebugStr(this->workload)
         << " candidate:" << toDebugStr(this->optimum);
    this->currTotalExpire = ExpireCandidate::FOREVER;
  }
}
