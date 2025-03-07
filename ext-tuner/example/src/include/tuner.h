#pragma once
#include <pthread.h>
#include <sys/stat.h>
#include <atomic>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include "src/include/datatype.h"
#include "src/include/internal/database.h"
#include "src/include/internal/env.h"
#include "src/include/internal/logging.h"
#include "src/include/internal/message.h"
#include "src/include/internal/threadsafe_queue.h"
#include "src/include/net/communicator.h"
#include "src/include/optimizers/optimizer.h"
#include "src/include/timer.h"

class Tuner {
  public:
    Tuner() {
      Logger::setVerboseLevel(
          Environment::get()->find("TUNER_VERBOSE", "WARN"));
      Logger::setSubSys(
          Environment::get()->find("TUNER_VERBOSE_SUBSYS", "ALL"));
    }
    void start(Info myInfo, std::unique_ptr<Communicator> communicator,
               const char *role, std::unique_ptr<Optimizer> optimizer = nullptr,
               std::unique_ptr<Timer> timer = nullptr) {
      CHECK(this->started != true);
      this->database = std::make_unique<Database>();
      // Environment::get()->find("TUNER_LOAD_DATABASE", "")
      this->communicator = std::move(communicator);
      this->communicator->setMyInfo(myInfo);
      this->communicator->setDataProcessor(
          std::bind(&Tuner::update, this, std::placeholders::_1));
      this->communicator->start(role);

      if (timer) {
        this->timer = std::move(timer);
        this->timer->start();
      } else {
        CHECK(strcmp(role, "COORDINATOR") == 0);
      }

      if (optimizer) {
        this->optimizer = std::move(optimizer);
        this->optimizer->initialize(this);
        this->trainThread =
            std::unique_ptr<std::thread>(new std::thread(&Tuner::train, this));
        pthread_setname_np(this->trainThread->native_handle(), "trainThread");
        INFO(Logger::LogSubSys::TUNER) << "tuner start training thread";
      } else {
        CHECK(strcmp(role, "COORDINATOR") == 0);
      }

      this->started = true;
    }
    void stop() {
      CHECK(this->communicator->getMyInfo().groupIDs.size() == 0);
      if (this->timer) {
        this->timer->stop();
        INFO(Logger::LogSubSys::TUNER) << "timer stoped";
        std::vector<Record> records;  // NOLINT
        this->step(&records, true);
        INFO(Logger::LogSubSys::TUNER) << "tuner.step() done";
      }
      if (this->optimizer) {
        INFO(Logger::LogSubSys::TUNER) << "stopping train";
        stopTrain = true;
        this->trainThread->join();
        INFO(Logger::LogSubSys::TUNER) << "train stoped";
      }

      this->communicator->stop();
      INFO(Logger::LogSubSys::TUNER) << "communicator stoped";

      try {
        if (!this->communicator->getMyInfo().isCoordinator()) {
          dump();
        }
      } catch (const std::exception &e) {
        std::cerr << "Exception caught: " << e.what() << '\n';
      } catch (...) {
        std::cerr << "Unknown exception caught\n";
      }
    }
    void query(const GIDTYPE &groupID, const Workload &workload,
               Candidate *startCandidate);
    void update(const Message &recved);
    void dump() {
      auto myInfo = this->communicator->getMyInfo();
      std::string path = Environment::get()->find("TUNER_DUMP_DATABASE", "./");
      path = path + std::string("/tunerdb_") + std::to_string(myInfo.nodeID) +
             std::string("_") + std::to_string(myInfo.deviceID) +
             std::string("/");
      struct stat buffer;
      if (stat(path.c_str(), &buffer) != 0) {
        std::string command = "mkdir -p " + path;  // make directory
        system(command.c_str());
      }
      std::string filename = path + std::string("info.txt");
      std::ofstream file(filename);  // NOLINT
      if (file.is_open()) {
        file << this->communicator->debugStr();
        file << this->optimizer->debugStr();
        file.close();
      }
      filename = path + std::string("log.txt");
      file.open(filename);
      if (file.is_open()) {
        file << this->database->logDebugStr();
        file.close();
      }
      filename = path + std::string("table.txt");
      file.open(filename);
      if (file.is_open()) {
        file << this->database->dataDebugStr();
        file.close();
      }
    }
    bool isNewWorkload(const Workload &workload) {
      if (this->workloadCount.count(workload) == 0) {
        this->workloadCount[workload] = 1;
        return true;
      } else {
        this->workloadCount[workload] += 1;
        return false;
      }
    }
    bool initialized = false;
    bool started = false;
    std::unique_ptr<Communicator> communicator;
    std::unique_ptr<Timer> timer;

  private:
    void train();
    void step(std::vector<Record> *records, bool blocking);
    std::unique_ptr<Optimizer> optimizer;
    std::unique_ptr<Database> database;
    std::unique_ptr<std::thread> trainThread;
    std::atomic<bool> stopTrain{false};
    std::map<Workload, QueryStatus> store;
    std::map<Workload, int> workloadCount;
    bool storeChanged = false;
    std::map<Workload, std::queue<RoundCandidates>> cachedRounds;
    std::mutex mutex;
    std::condition_variable cond;
    // pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    // pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
};
