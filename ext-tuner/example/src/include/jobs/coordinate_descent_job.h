#pragma once
#include "src/include/datatype.h"
#include "src/include/internal/logging.h"
#include "src/include/jobs/job.h"

struct CoDesJob : public Job {
    static bool isDecentralized() { return false; }
    CoDesJob(Tuner *tuner, const GIDTYPE &groupID, const Workload &workload,
          const Candidate &startCandidate,
          std::vector<Candidate> &&validCandidates, std::vector<ConfigRange> configRanges, const int32_t warmupSteps, const int32_t nativeSteps,
          const int32_t pretrainSteps, const int32_t trainSteps,
          const int32_t roundMaxSteps, const int32_t optimumExpireSteps,
          const int32_t expireSteps)
        : Job(tuner, groupID, workload, startCandidate, std::move(validCandidates), configRanges,
              warmupSteps, nativeSteps, pretrainSteps, trainSteps, roundMaxSteps,
              optimumExpireSteps, expireSteps) {
      this->allow_repeat_access = Environment::get()->find("TUNER_REPEAT_ACCESS", 0) > 0;
      this->topk_to_smooth = Environment::get()->find("TUNER_ACCESS_TOPK", 3);
      this->retry_to_jump_local_optimum = Environment::get()->find("TUNER_RETRY", 5);
      for (int32_t index = 0; index < this->configRanges.size(); index++) {
        if (!this->configRanges[index].used) continue;
        this->dimensions.push_back(index);
      }
      CHECK(this->dimensions.size() >= 1);

      if (this->trainSteps/this->expireSteps < this->dimensions.size()*2*8) {
        INFO(Logger::LogSubSys::OPTIMIZER) << "trainSteps" << this->trainSteps
          << "/expireSteps" << this->expireSteps << "=" << this->trainSteps/this->expireSteps
          << " : add more trainSteps may help.";
      }

    }

    JobType getType() const override {
      return JobType::CoDesJob;
    }

    std::string debugStr() const override;

  protected:
    void addTrain(Result *jobResult) override;
  private:
    struct Point {
      int32_t dimension;
      bool positive;
      Candidate candidate;
      float duration;
    };
    Point currPoint;
    std::vector<Point> nextPoints;

    int32_t retry = 0;
    int32_t retry_to_jump_local_optimum;
    std::vector<int32_t> dimensions;
    int32_t convergenceCount = 0;
    double lr = -1;

    bool allow_repeat_access;
    int32_t topk_to_smooth;

    bool updateStatus();
    void surveyWithAxis(std::vector<Candidate> *nextCandidates);
};

std::string CoDesJob::debugStr() const {
  std::stringstream ss;  // NOLINT
  ss << Job::debugStr();
  ss << " + CoDesJob[";
  ss << "]";
  return ss.str();
}

bool CoDesJob::updateStatus() {
  const auto &currRecordKey = 
        RecordKey(this->groupID, this->workload, this->currPoint.candidate);
  CHECK(this->recordsTable.count(currRecordKey) != 0);
  this->currPoint.duration = this->recordsTable[currRecordKey].meanOfTopPercent(75);

  Point betterPoint = this->currPoint;
  bool hasBetterPoint = false;
  for (auto point : this->nextPoints) {
    const auto &nextRecordKey = 
        RecordKey(this->groupID, this->workload, point.candidate);
    CHECK(this->recordsTable.count(nextRecordKey) != 0);
    point.duration = this->recordsTable[nextRecordKey].meanOfTopPercent(75);
    if (point.duration < betterPoint.duration) {
      // adjust learning rate by duration delta
      // this->lr = 1.0 / (1.0 + std::exp(-(betterPoint.duration - point.duration)));
      this->lr = (this->currPoint.duration - point.duration)/this->currPoint.duration*0.8;
      betterPoint = point;
      hasBetterPoint = true;
    }
  }
  if (hasBetterPoint) {
    this->convergenceCount = 0;
    this->currPoint = betterPoint;
  } else {
    // no better point or nextPoints is empty
    this->convergenceCount++;
    if (this->currPoint.positive) {
      this->currPoint.positive = false;
    } else {
      this->currPoint.positive = true;
      this->currPoint.dimension = (this->currPoint.dimension+1)%this->dimensions.size();
    }
    this->lr = -1;
  }
  if (this->convergenceCount == this->dimensions.size()*2) {
    return false;
  }
  return true;
}

void CoDesJob::surveyWithAxis(std::vector<Candidate> *neighbors) {
  TRACE(Logger::LogSubSys::OPTIMIZER) << " survey along with axis=" << this->dimensions[this->currPoint.dimension] << " " << this->currPoint.positive ? "+" : "-";
  this->nextPoints.clear();
  while (neighbors->size() == 0) {
    auto newNeighbors = findKCandidatesWithAxis(this->restCandidates/*validCandidates or restCandidates?*/,
                      this->currPoint.candidate, this->topk_to_smooth/*topk*/, this->dimensions[this->currPoint.dimension], this->currPoint.positive, this->lr, &this->configRanges);
    neighbors->insert(neighbors->end(), newNeighbors.begin(), newNeighbors.end());
    if (neighbors->size() > 0) break;
    else {
      if (!updateStatus()) {
        // this->currPoint.candidate = this->validCandidates[this->getRandIndex()];
        if (!this->allow_repeat_access) {
          if (this->restCandidates.size() > 0) {
            this->currPoint.candidate = this->restCandidates[this->getRandIndex() % this->restCandidates.size()];
            this->currPoint.dimension = 0;
            this->currPoint.positive = true;
            this->retry++;
            neighbors->push_back(this->currPoint.candidate);
          } else {
            this->currPoint.candidate = this->optimum;
            this->currPoint.dimension = 0;
            this->currPoint.positive = true;
            this->retry++;
            neighbors->push_back(this->currPoint.candidate);
          }
        } else {
          this->currPoint.candidate = this->validCandidates[this->getRandIndex()];
          this->currPoint.dimension = 0;
          this->currPoint.positive = true;
          this->retry++;
          neighbors->push_back(this->currPoint.candidate);
        }
      }
    }
  } 
  for (auto neighbor : *neighbors) {
    this->nextPoints.push_back({this->currPoint.dimension, this->currPoint.positive, neighbor, 0});
  }
}

void CoDesJob::addTrain(Result *jobResult) {
  std::vector<Candidate> neighbors;
  if (this->retry == 0) {
    this->currPoint.candidate = this->optimum;
    this->currPoint.dimension = 0;
    this->currPoint.positive = true;
    this->retry++;
    surveyWithAxis(&neighbors);
  } else if (this->retry < this->retry_to_jump_local_optimum) {
    if (updateStatus()) {
      surveyWithAxis(&neighbors);
    } else {
      // this->currPoint.candidate = this->validCandidates[this->getRandIndex()];
      if (!this->allow_repeat_access) {
        if (this->restCandidates.size() > 0) {
          this->currPoint.candidate = this->restCandidates[this->getRandIndex() % this->restCandidates.size()];
          this->currPoint.dimension = 0;
          this->currPoint.positive = true;
          this->retry++;
          neighbors.push_back(this->currPoint.candidate);
        } else {
          this->currPoint.candidate = this->optimum;
          this->currPoint.dimension = 0;
          this->currPoint.positive = true;
          this->retry++;
          neighbors.push_back(this->currPoint.candidate);
        }
      } else {
        this->currPoint.candidate = this->validCandidates[this->getRandIndex()];
        this->currPoint.dimension = 0;
        this->currPoint.positive = true;
        this->retry++;
        neighbors.push_back(this->currPoint.candidate);
      }
    }
  } else {
    // Try multiple times
    // if there is no better candidate for all dimensions (positive and negative) or nextPoints is empty
    this->convergence = true;
    return;
  }
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
  // CHECK(neighbors.size() > 0);
}
