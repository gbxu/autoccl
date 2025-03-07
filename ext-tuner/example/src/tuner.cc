#include "src/include/tuner.h"
#include <assert.h>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>

void Tuner::query(const GIDTYPE &groupID, const Workload &workload,
                  Candidate *candidate) {
  std::unique_lock<std::mutex> lock(this->mutex);
  if (this->store.count(workload) == 0) {
    this->store[workload] = QueryStatus();
    this->storeChanged = true;  // train() will call jobs.
    const auto& myInfo = this->communicator->getMyInfo();
    const auto& allGroupInfos =
        this->communicator->getAllGroupInfos();  // groupInfo will update
    this->optimizer->addJob(myInfo, allGroupInfos, groupID, workload,
                            *candidate);
    TRACE(Logger::LogSubSys::TUNER) << "added job to optimizer";
  }
  // wait for candidates
  this->cond.wait(lock, [this, workload] {
    if (this->store[workload].remainingExpire == 0) {
      TRACE(Logger::LogSubSys::TUNER) << "Query :" << toDebugStr(workload) << " waiting";
      return false;
    }
    return true;
  });
  this->store[workload].activeVersion =
      this->store[workload].queue.front().version;  // current active
                                                    // version
  auto &expireCandidate =
      this->store[workload]
          .queue.front()
          .expireCandidates[this->store[workload].currIndex];
  TRACE(Logger::LogSubSys::TUNER) << "Store=" << this->store[workload].debugStr()
        << " ActiveRound=" << this->store[workload].queue.front().debugStr();
  CHECK(expireCandidate.expire != 0);
  *candidate = expireCandidate.candidate;
  TRACE(Logger::LogSubSys::TUNER) << "Query :" << toDebugStr(workload)
        << " got:" << toDebugStr(expireCandidate.candidate);
  // decrease 1
  if (expireCandidate.expire != ExpireCandidate::FOREVER) {
    expireCandidate.expire--;
    CHECK(expireCandidate.expire >= 0);
    if (expireCandidate.expire == 0) {
      this->store[workload].currIndex++;
      CHECK(this->store[workload].currIndex <=
            this->store[workload].queue.front().expireCandidates.size());
    }
    if (this->store[workload].remainingExpire != ExpireCandidate::FOREVER) {
      this->store[workload].remainingExpire--;
      CHECK(expireCandidate.expire >= 0);
      this->storeChanged = true;
    }
  }
  // if running out
  if (static_cast<size_t>(this->store[workload].currIndex) ==
      this->store[workload].queue.front().expireCandidates.size()) {
    this->store[workload].queue.pop();  // version changed
    this->store[workload].currIndex = 0;
    this->storeChanged = true;
    if (this->store[workload].queue.empty()) {
      CHECK(this->store[workload].remainingExpire == 0);
    }
    if (this->store[workload].remainingExpire == 0) {
      CHECK(this->store[workload].queue.empty());
    }
  }
}

void Tuner::update(const Message &recved) {
  if (recved.meta.type == Meta::RESULTS) {  // process candidate results
    for (auto &result : recved.results) {
      const auto &workload = result.workload;
      const auto &roundCandidates = result.roundCandidates;
      if (cachedRounds.count(workload) == 0) {
        cachedRounds[workload] = std::queue<RoundCandidates>();
      }
      TRACE(Logger::LogSubSys::TUNER) << "cache candidate: workload=" << toDebugStr(workload)
            << " v=" << roundCandidates.version;
      cachedRounds[workload].push(std::move(roundCandidates));
    }
  } else if (recved.meta.type == Meta::DECISIONS) {  // process version decision
    for (auto &decision : recved.decisions) {
      TRACE(Logger::LogSubSys::TUNER) << "workload=" << toDebugStr(decision.workload)
            << " decision=" << decision.version;
      CHECK(cachedRounds.count(decision.workload) != 0);
      std::queue<RoundCandidates> saving;
      while (!cachedRounds[decision.workload].empty()) {
        const auto &roundCandidates = cachedRounds[decision.workload].front();
        const auto &cacheVersion = roundCandidates.version;
        if (cacheVersion > decision.version) {
          saving.push(std::move(roundCandidates));
          cachedRounds[decision.workload].pop();
          TRACE(Logger::LogSubSys::TUNER) << "workload=" << toDebugStr(decision.workload)
                << " save newer version=" << cacheVersion;
        } else if (cacheVersion == decision.version) {
          TRACE(Logger::LogSubSys::TUNER) << "workload="
                << toDebugStr(decision.workload) << " hit version=" << decision.version;
          {
            std::lock_guard<std::mutex> lock(this->mutex);
            if (store.count(decision.workload) == 0) {
              store[decision.workload] = QueryStatus();
            }
            if (roundCandidates.roundExpire == ExpireCandidate::FOREVER) {
              store[decision.workload].remainingExpire =
                  ExpireCandidate::FOREVER;
            } else {
              CHECK(roundCandidates.roundExpire > 0);
              CHECK(store[decision.workload].remainingExpire !=
                    ExpireCandidate::FOREVER);
              store[decision.workload].remainingExpire +=
                  roundCandidates.roundExpire;  // total expire
            }
            CHECK(roundCandidates.expireCandidates.size() > 0);
            store[decision.workload].queue.push(std::move(roundCandidates));
            this->storeChanged = true;
          }
          this->cond.notify_all();
          cachedRounds[decision.workload].pop();
          break;
        } else {
          TRACE(Logger::LogSubSys::TUNER) << "workload=" << toDebugStr(decision.workload)
                << " drop old version=" << cacheVersion;
          cachedRounds[decision.workload].pop();
        }
      }
      if (cachedRounds.count(decision.workload) == 0) {
        cachedRounds[decision.workload] = std::move(saving);
      } else {
        while (!saving.empty()) {
          cachedRounds[decision.workload].push(std::move(saving.front()));
          saving.pop();
        }
      }
    }
  }
}

void Tuner::train() {
  const auto& myInfo = communicator->getMyInfo();
  std::map<Workload, SimpleQueryStatus> workloadStatus;  // NOLINT
  while (!this->stopTrain) {
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      if (this->storeChanged) {
        for (auto &pair : this->store) {
          auto &workload = pair.first;
          auto &status = pair.second;
          workloadStatus[workload] = {status.remainingExpire,
                                      status.activeVersion};
        }
        this->storeChanged = false;
      }
    }

    DecisionPack decisionPack;
    this->optimizer->pick(workloadStatus, &decisionPack);

    if (decisionPack.localDecisions.size()) {
      Message msg(std::move(decisionPack.localDecisions));
      msg.meta.sender = myInfo.id;
      msg.meta.recver = myInfo.id;
      communicator->sendMsg(msg);
    }

    // broadcast version data from leader
    for (auto &pair : decisionPack.distDecisions) {
      if (pair.second.size() == 0)
        continue;
      Message msg(std::move(pair.second));
      msg.meta.sender = myInfo.id;
      msg.meta.recverGroupID = pair.first;
      communicator->broadcastMsg(msg);
    }

    std::vector<Record> records;  // NOLINT
    this->step(&records, false /*non-blocking*/);

    ResultPack resultPack;
    this->optimizer->run(workloadStatus, records, &resultPack);

    if (resultPack.localResults.size()) {
      Message msg(std::move(resultPack.localResults));
      msg.meta.sender = myInfo.id;
      msg.meta.recver = myInfo.id;
      communicator->sendMsg(msg);
    }

    // broadcast new candidates generated by the optimizer
    for (auto &pair : resultPack.distResults) {
      if (pair.second.size() == 0)
        continue;
      Message msg(std::move(pair.second));
      msg.meta.sender = myInfo.id;
      msg.meta.recverGroupID = pair.first;
      communicator->broadcastMsg(msg);
    }
  }
}

void Tuner::step(std::vector<Record> *records, bool blocking) {
  this->timer->tryGetRecords(records, blocking);
  for (const auto &record : *records) {
    this->database->add(record);
  }
}
