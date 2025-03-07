#pragma once
#include "src/include/datatype.h"
#include "src/include/internal/logging.h"
#include "src/include/jobs/job.h"
#include <fstream>
#include <string>

struct SpecificJob : public Job {
    SpecificJob(Tuner *tuner, const GIDTYPE &groupID, const Workload &workload,
              const Candidate &startCandidate,
              std::vector<Candidate> &&validCandidates, std::vector<ConfigRange> configRanges,
              const int32_t warmupSteps, const int32_t nativeSteps, const int32_t pretrainSteps,
              const int32_t trainSteps, const int32_t roundMaxSteps,
              const int32_t optimumExpireSteps, const int32_t expireSteps)
        : Job(tuner, groupID, workload, startCandidate,
              std::move(validCandidates), configRanges, 0, 0, 0,
              0, roundMaxSteps, optimumExpireSteps, expireSteps) {
      // get specific file
      std::vector<std::string> lines;
      this->getCandidatesFromFile(lines);
      /*
        example: nRanks; workload; candidates; expire
          8;-1 4 1048576;0 0 0 1 96 1536 1366 256; 1
          8;-1 4 2097152;0 0 0 1 96 1536 2731 512; -1
      */
      int check = Environment::get()->find("TUNER_SPECIFIC_CHECK", 1);
      for (auto line : lines) {
        std::istringstream iss(line);
        std::string clusterStr, workloadStr, candidateStr, expireStr;
        std::getline(iss, clusterStr, ';');
        std::getline(iss, workloadStr, ';');
        std::getline(iss, candidateStr, ';');
        if(!std::getline(iss, expireStr, ';')) {
          expireStr = "-1";
        }
        std::istringstream clusterIss(clusterStr);
        std::istringstream workloadIss(workloadStr);
        std::istringstream candidateIss(candidateStr);
        std::istringstream expireIss(expireStr);

        // get specificWorkload: {-1, 4, 1048576} in "-1 4 1048576;"
        KEY workloadElem;
        Workload specificWorkload; 
        while (workloadIss >> workloadElem) specificWorkload.push_back(workloadElem);

        // get nrank: 8 in "8;"
        int32_t nrank;
        clusterIss >> nrank;
        // Check nRanks in clusterID
        std::unordered_map<GIDTYPE, GroupInfo> allGroupInfos = this->tuner->communicator->getAllGroupInfos();
        if ((nrank != -1) && (nrank != allGroupInfos[groupID].nrank)) continue; // wrong nrank, continue

        // Check workload in clusterID
        Workload currentWorkload = this->workload;
        if (!this->compare(currentWorkload, specificWorkload)) continue;

        // get specificCandidate: {0, 0, 0, 1, 96, 1536, 1366, 256} in "0 0 0 1 96 1536 1366 256;"
        Candidate candidate;
        CONFIG candidateElem;
        while (candidateIss >> candidateElem) candidate.push_back(candidateElem);
        // Check if specificCandidate is valid
        bool isValidCandidate = false;
        if (check) {
          for (const auto& validCandidate : this->validCandidates) {
            if (this->compare(validCandidate, candidate)) {
              isValidCandidate = true;
              break;
            }
          }
        } else {
          isValidCandidate = true;
        }
        if (!isValidCandidate) {
          const auto &neighbours =findKPrefixMatchCandidates(this->validCandidates, candidate, 10);
          std::string similarCandidates;
          for (const auto& neighbour : neighbours) {
            similarCandidates += toDebugStr(neighbour) + "\n";
          }
          THROWERROR << " The candidate=" << toDebugStr(candidate) << "for workload=" << toDebugStr(this->workload) << " is invalid, do you mean: \n" << similarCandidates;
        }

        // get expire
        int32_t expire;
        expireIss >> expire;
        if (expire == -1) {
          expire = ExpireCandidate::FOREVER;
        }
        // save candidate
        auto specificEC = ExpireCandidate(expire, candidate);
        this->userExpireCandidates.push_back(specificEC);
      }
      // Check if candidates is NOT FOUND
      if (this->userExpireCandidates.size() == 0) THROWERROR << "Candidates for workload=" << toDebugStr(this->workload) << "not found";
    }

    JobType getType() const override {
      return JobType::SpecificJob;
    }

    static bool isDecentralized() { return true; }
    void createResult(const SimpleQueryStatus &status, Result *result) override;

    protected:
      void addTrain(Result *jobResult) override {};
    private:
      std::vector<ExpireCandidate> userExpireCandidates;
      void getCandidatesFromFile(std::vector<std::string>& lines);
      template <typename T> bool compare(const T &currentData, const T &userData);
};

void SpecificJob::createResult(const SimpleQueryStatus &status, Result *result) {
  if (this->isDone)
    return;
  result->workload = this->workload;

  int32_t currRoundExpire = 0;
  for (const auto& expireCandidate : this->userExpireCandidates) {
    result->roundCandidates.expireCandidates.push_back(expireCandidate);
    if (expireCandidate.expire == ExpireCandidate::FOREVER) {
      this->optimum = expireCandidate.candidate;
      this->currTotalExpire = ExpireCandidate::FOREVER;
      currRoundExpire = ExpireCandidate::FOREVER;
      break;
    } else {
      if (currRoundExpire != ExpireCandidate::FOREVER) {
        currRoundExpire += expireCandidate.expire;
      }
    }
  }
  result->roundCandidates.roundExpire = currRoundExpire;
  if (result->roundCandidates.expireCandidates.size()) {
    result->roundCandidates.version = this->nextVersion++;
    this->isDone = true;
    TRACE(Logger::LogSubSys::OPTIMIZER) << " generate: " << result->debugStr()
          << ", job=" << this->debugStr();
  }
}

void SpecificJob::getCandidatesFromFile(std::vector<std::string>& lines) {
    const char* str = Environment::get()->find("TUNER_SPECIFIC_FILE", ""); // all nodes should have this file.
    if (str && strlen(str) > 0) {
      std::string filename = std::string(str);
      std::ifstream file(filename);
      std::string line;
      while (std::getline(file, line)) {
        lines.push_back(line);
      }
      file.close();
    } else {
      str = getenv("TUNER_SPECIFIC");
      if (str && strlen(str) > 0) {
        lines.push_back(std::string(str));
      }
    }
}

template <typename T> 
bool SpecificJob::compare(const T &validData, const T &userData) {
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
