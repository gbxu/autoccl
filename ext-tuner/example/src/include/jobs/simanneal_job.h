#pragma once
#include "src/include/datatype.h"
#include "src/include/internal/logging.h"
#include "src/include/jobs/job.h"

struct SimAnnealJob : public Job {
    static bool isDecentralized() { return false; }
    SimAnnealJob(Tuner *tuner, const GIDTYPE &groupID, const Workload &workload,
                 const Candidate &startCandidate,
                 std::vector<Candidate> &&validCandidates, std::vector<ConfigRange> configRanges,
                 const int32_t warmupSteps, const int32_t nativeSteps, const int32_t pretrainSteps,
                 const int32_t trainSteps, const int32_t roundMaxSteps,
                 const int32_t optimumExpireSteps, const int32_t expireSteps)
        : Job(tuner, groupID, workload, startCandidate, std::move(validCandidates), configRanges,
              warmupSteps, nativeSteps, pretrainSteps, trainSteps, roundMaxSteps,
              optimumExpireSteps, expireSteps) {

      if (this->trainSteps/this->expireSteps < (std::log(Tmin)-std::log(T0))/std::log(delta)) {
        INFO(Logger::LogSubSys::OPTIMIZER) << "trainSteps" << this->trainSteps
          << "/expireSteps" << this->expireSteps << "=" << this->trainSteps/this->expireSteps
          << " : add more trainSteps may help.";
      }
    }
    JobType getType() const override {
      return JobType::SimAnnealJob;
    }

    std::string debugStr() const override;

  protected:
    void addTrain(Result *jobResult) override;
  private:
    struct Point {
      Candidate candidate;
      float duration;
    };
    Point currPoint;
    std::vector<Point> nextPoints;

    // need 37 times to cool down
    double T = 1000.0;  // Initial temperature
    double T0 = T;  // Initial temperature. keep unchanged. It is used to calculate the neighbor range
    double Tmin = 1e-8;  // Lower bound of temperature
    double delta = 0.5;  // TODO: dynamic speed

    void updateStatus();
    void surveyAround(std::vector<Candidate> *neighbors);
};

std::string SimAnnealJob::debugStr() const {
  std::stringstream ss;  // NOLINT
  ss << Job::debugStr();
  ss << " + SimAnnealJob[";
  ss << " T=" << T << "; ";
  ss << " T0=" << T0 << "; ";
  ss << " Tmin=" << Tmin << "; ";
  ss << " delta=" << delta << "; ";
  ss << "]";
  return ss.str();
}

void SimAnnealJob::updateStatus() {
  const auto &currRecordKey =
      RecordKey(this->groupID, this->workload, this->currPoint.candidate);
  CHECK(this->recordsTable.count(currRecordKey) != 0);
  this->currPoint.duration = this->recordsTable[currRecordKey].meanOfTopPercent(75);
  for (auto &point : this->nextPoints) {
    const auto &nextRecordKey =
        RecordKey(this->groupID, this->workload, point.candidate);
    CHECK(this->recordsTable.count(nextRecordKey) != 0);
    point.duration = this->recordsTable[nextRecordKey].meanOfTopPercent(75);
    if (point.duration < this->currPoint.duration) {
      this->currPoint = point;
      T = T * delta;
    } else if (exp(-(point.duration - this->currPoint.duration) / T) > rand() * 1.0 / RAND_MAX) {
      // 如果curr大于min，即为差解。curr-min越大，接受的概率越低
      // If curr > min, it is a bad solution. 
      // The larger the curr-min, the lower the probability of acceptance.
      this->currPoint = point;
      T = T * delta;
    }
  }
  this->convergence = T < Tmin;
}

void SimAnnealJob::surveyAround(std::vector<Candidate> *neighbors) {
  TRACE(Logger::LogSubSys::OPTIMIZER) << " survey around";
  // The lower the temperature, the smaller the step size
  double lr = T/T0;
  for (int32_t index = 0; index < this->configRanges.size(); index++) {
    if (!this->configRanges[index].used) continue;
    for (bool positive : {true, false}) {
      auto newNeighbors = findKCandidatesWithAxis(this->restCandidates/*validCandidates or restCandidates?*/,
                        this->currPoint.candidate , 1/*topk*/, index, positive, lr, &this->configRanges);
      neighbors->insert(neighbors->end(), newNeighbors.begin(), newNeighbors.end());
    }
  }
  for (auto neighbor : *neighbors) {
    this->nextPoints.push_back({neighbor, 0});
  }
  if (neighbors->empty()) {
    WARN(Logger::LogSubSys::OPTIMIZER) << "No neighbors for:" << toDebugStr(this->currPoint.candidate);
  }
}

void SimAnnealJob::addTrain(Result *jobResult) {
  std::vector<Candidate> neighbors;
  if (this->nextPoints.empty()) {
    this->currPoint.candidate = this->optimum;
  } else {
    updateStatus();
  }
  if (this->convergence) {
    return;
  }
  surveyAround(&neighbors);
  for (auto neighbor : neighbors) {
    int32_t expireSteps = std::min(this->warmupSteps+this->nativeSteps+this->pretrainSteps+this->trainSteps-this->currTotalExpire, this->expireSteps);
    if (expireSteps <= 0) {
      break;
    }
    auto tempEC = ExpireCandidate(expireSteps,
                                  neighbor);
    jobResult->roundCandidates.roundExpire += expireSteps;
    jobResult->roundCandidates.expireCandidates.push_back(tempEC);
    this->updateRestCandidates(neighbor);
    this->currTotalExpire += expireSteps;
  }
}
