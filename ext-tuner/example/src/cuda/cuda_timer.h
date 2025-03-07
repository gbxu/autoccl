#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <queue>
#include <tuple>
#include <vector>
#include "src/include/tuner.h"
#include "src/include/internal/logging.h"

constexpr int RECORD_NUM = 2048;

struct CudaRecord {
    cudaEvent_t begin_event;
    cudaEvent_t end_event;
    RecordKey recordKey;
    float duration;  // ms
    cudaStream_t stream;

    CudaRecord() : duration(0.0) {
      cudaError_t res = cudaSuccess;  // NOLINT
      res = cudaEventCreate(&begin_event);
      if (res != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA event");
      }
      res = cudaEventCreate(&end_event);
      if (res != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA event");
      }
    }
    CudaRecord(const CudaRecord &) = delete;
    CudaRecord &operator=(const CudaRecord &) = delete;
    CudaRecord(CudaRecord &&) = delete;
    CudaRecord &operator=(CudaRecord &&) = delete;

    ~CudaRecord() {
      cudaEventDestroy(begin_event);
      cudaEventDestroy(end_event);
    }
};

struct ProfilingStatus {
    int32_t askedProfiling = 0;
    int32_t profiledCount = 0;
    int32_t skipCount = 0;
};

void *cudaQuery(void *cudaTimer);

class CudaTimer : public Timer {
  public:
    CudaTimer() {  // NOHINT
      for (auto &cudaRecord : cudaRecords) {
        this->available_records.push(&cudaRecord);
      }
    }

    void start() override {
      pthread_create(&query_thread, NULL, cudaQuery, this);
      INFO(Logger::LogSubSys::TIMER) << "cuda timer start profiling thread";
    }

    void stop() override {
      INFO(Logger::LogSubSys::TIMER) << "stopping timer";
      pthread_mutex_lock(&this->mutex_profiling);
      stop_query = true;
      pthread_cond_signal(&this->cond_profiling);
      pthread_mutex_unlock(&this->mutex_profiling);
      pthread_join(query_thread, NULL);
    }

    void getSpecificRecords(RecordKey *specificRecordKey,
                            std::vector<Record> *records, bool blocking) {
      const auto &specificWorkload = specificRecordKey->workload;
      auto *specificCandidate = &(specificRecordKey->candidate);  // ptr
      std::vector<CudaRecord *> temp_available_records;           // NOLINT
      std::queue<CudaRecord *> profiling_records_copy;            // NOLINT
      bool hasSpecific = false;
      while (blocking) {  // waiting for all events
        hasSpecific = false;
        pthread_mutex_lock(&this->mutex_profiling);
        profiling_records_copy = this->profiling_records;
        pthread_mutex_unlock(&this->mutex_profiling);
        while (!profiling_records_copy.empty()) {
          CudaRecord *record = profiling_records_copy.front();
          profiling_records_copy.pop();
          if ((specificWorkload == record->recordKey.workload) &&
              (specificCandidate->empty() ||
               *specificCandidate == record->recordKey.candidate)) {
            hasSpecific = true;
          }
        }
        if (!hasSpecific)
          break;

        // waiting for the specific events
        std::queue<CudaRecord *> profiled_records_copy;       // NOLINT
        std::queue<CudaRecord *> profiled_records_remaining;  // NOLINT
        pthread_mutex_lock(&this->mutex_profiled);
        profiled_records_copy = this->profiled_records;
        pthread_mutex_unlock(&this->mutex_profiled);
        while (!profiled_records_copy.empty()) {
          CudaRecord *record = profiled_records_copy.front();
          profiled_records_copy.pop();
          if ((specificWorkload == record->recordKey.workload) &&
              (specificCandidate->empty() ||
               *specificCandidate == record->recordKey.candidate)) {
            if (specificCandidate->empty())
              *specificCandidate = record->recordKey.candidate;
            temp_available_records.push_back(record);
          } else {
            profiled_records_remaining.push(record);
          }
        }
        pthread_mutex_lock(&this->mutex_profiled);
        this->profiled_records = profiled_records_remaining;
        pthread_mutex_unlock(&this->mutex_profiled);
        if (!temp_available_records.empty())
          break;
      }

      // waiting for the specific events
      std::queue<CudaRecord *> profiled_records_copy;       // NOLINT
      std::queue<CudaRecord *> profiled_records_remaining;  // NOLINT
      pthread_mutex_lock(&this->mutex_profiled);
      profiled_records_copy = this->profiled_records;
      pthread_mutex_unlock(&this->mutex_profiled);
      while (!profiled_records_copy.empty()) {
        CudaRecord *record = profiled_records_copy.front();
        profiled_records_copy.pop();
        if ((specificWorkload == record->recordKey.workload) &&
            (specificCandidate->empty() ||
             *specificCandidate == record->recordKey.candidate)) {
          if (specificCandidate->empty())
            *specificCandidate = record->recordKey.candidate;
          temp_available_records.push_back(record);
        } else {
          profiled_records_remaining.push(record);
        }
      }
      pthread_mutex_lock(&this->mutex_profiled);
      this->profiled_records = profiled_records_remaining;
      pthread_mutex_unlock(&this->mutex_profiled);

      for (CudaRecord *record : temp_available_records) {
        records->push_back(Record(RecordKey(record->recordKey),
                                  RecordValue(record->duration, 1)));
      }

      if (!temp_available_records.empty()) {
        pthread_mutex_lock(&this->mutex_available);
        for (CudaRecord *record : temp_available_records) {
          available_records.push(record);
        }
        pthread_cond_signal(&this->cond_available);
        pthread_mutex_unlock(&this->mutex_available);
      }

      return;
    }

    void tryGetRecords(std::vector<Record> *records, bool blocking) override {
      std::vector<CudaRecord *> temp_available_records;  // NOLINT
      while (blocking) {  // waiting for all events
        pthread_mutex_lock(&this->mutex_profiling);
        if (this->profiling_records.empty()) {
          pthread_mutex_unlock(&this->mutex_profiling);
          break;
        }
        pthread_mutex_unlock(&this->mutex_profiling);
      }
      pthread_mutex_lock(&this->mutex_profiled);
      while (!this->profiled_records.empty()) {
        CudaRecord *record = this->profiled_records.front();
        temp_available_records.push_back(record);
        this->profiled_records.pop();
      }
      pthread_mutex_unlock(&this->mutex_profiled);

      for (CudaRecord *record : temp_available_records) {
        records->push_back(Record(RecordKey(record->recordKey),
                                  RecordValue(record->duration, 1)));
      }

      if (!temp_available_records.empty()) {
        pthread_mutex_lock(&this->mutex_available);
        for (CudaRecord *record : temp_available_records) {
          available_records.push(record);
        }
        pthread_cond_signal(&this->cond_available);
        pthread_mutex_unlock(&this->mutex_available);
      }

      return;
    }

    void setProfiling(const Workload &workload,
                      int32_t askedProfiling = -1) override {
      this->statusMutex.lock();
      if (askedProfiling > 300 || askedProfiling == -1) {
        INFO(Logger::LogSubSys::TIMER) << "Inserting a lot of cuda events will behave like CPU blocking: "
                "workload="
             << toDebugStr(workload) << " least profiling:" << askedProfiling;
      }
      if (status.count(workload) == 0) {
        status[workload] = {askedProfiling, 0, 0};
      } else {
        status[workload].askedProfiling = askedProfiling;
      }
      this->statusMutex.unlock();
      INFO(Logger::LogSubSys::TIMER) << "set timer for workload=" << toDebugStr(workload)
           << " askedProfiling=" << askedProfiling;
    }
    void begin(const RecordKey &recordKey, bool blocking,
               void *stream) override {
      this->statusMutex.lock();
      if (status.count(recordKey.workload) == 0) {
        status[recordKey.workload] = {0, 0, 0};
        INFO(Logger::LogSubSys::TIMER) << "Call setProfiling in each job before timer.begin(), while "
                "sometimes the follower may recv candidate before it calls "
                "optimizer.addJob()";
      }
      if (status[recordKey.workload].askedProfiling != -1 &&
          status[recordKey.workload].profiledCount >=
              status[recordKey.workload].askedProfiling) {
        status[recordKey.workload].skipCount++;
        this->statusMutex.unlock();
        return;
      }
      status[recordKey.workload].profiledCount++;
      this->statusMutex.unlock();

      CudaRecord *record = nullptr;

      pthread_mutex_lock(&this->mutex_available);
      while (available_records.empty() && blocking) {
        WARN(Logger::LogSubSys::TIMER) << "cuda event is empty!";
        pthread_cond_wait(&this->cond_available, &this->mutex_available);
      }
      if (!available_records.empty()) {
        record = available_records.front();
        available_records.pop();
      }
      pthread_mutex_unlock(&this->mutex_available);

      if (record) {
        record->recordKey = recordKey;
        record->stream = (cudaStream_t)stream;
        cudaError_t res = cudaSuccess;  // NOLINT
        res = cudaEventRecord(record->begin_event, record->stream);
        if (res != cudaSuccess) {
          throw std::runtime_error("Failed to record begin event.");
        }
        using_records.push(record);
      }
    }

    void end(GIDTYPE groupID, bool blocking) override {
      if (using_records.empty()) {
        // throw std::runtime_error("Failed to record end event without start
        // event"); // TODO: check before calling now.
        return;
      }

      // Find the record with recordKey
      CudaRecord *record = nullptr;
      std::queue<CudaRecord *> using_records_copy;  // NOLINT
      while (!using_records.empty()) {
        record = using_records.front();
        using_records.pop();
        CHECK(record->recordKey.groupID == groupID);
        if (record->recordKey.groupID == groupID) {
          // Record found, update the end_event and add it back to using_records
          cudaError_t res = cudaSuccess;  // NOLINT
          res = cudaEventRecord(record->end_event, record->stream);
          if (res != cudaSuccess) {
            throw std::runtime_error("Failed to record end event.");
          }
          break;
        } else {
          WARN(Logger::LogSubSys::TIMER) << "begin-end is not a pair";
          record = nullptr;
          // Record not found, keep it in using_records
          using_records_copy.push(record);
        }
      }
      // Add the records from using_records_copy to using_records
      while (!using_records.empty()) {
        using_records_copy.push(using_records.front());
        using_records.pop();
      }
      using_records = using_records_copy;

      if (blocking) {
        cudaError_t res = cudaSuccess;  // NOLINT
        res = cudaStreamSynchronize(record->stream);
        // res = cudaDeviceSynchronize();
        if (res != cudaSuccess) {
          throw std::runtime_error(
              "Failed to synchronize with non-null CUDA stream.");
        }
      }

      pthread_mutex_lock(&this->mutex_profiling);
      this->profiling_records.push(record);
      pthread_cond_signal(&this->cond_profiling);
      pthread_mutex_unlock(&this->mutex_profiling);
    }

    CudaRecord cudaRecords[RECORD_NUM];
    pthread_t query_thread;
    bool stop_query = false;
    std::queue<CudaRecord *> available_records;  // NOLINT
    std::queue<CudaRecord *> using_records;      // NOLINT
    std::queue<CudaRecord *> profiling_records;  // NOLINT
    std::queue<CudaRecord *> profiled_records;   // NOLINT
    pthread_mutex_t mutex_available = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond_available = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex_profiling = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond_profiling = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex_profiled = PTHREAD_MUTEX_INITIALIZER;
    // pthread_cond_t cond_profiled = PTHREAD_COND_INITIALIZER;
    cudaStream_t stream = NULL;

  private:
    std::map<Workload, ProfilingStatus> status;
    std::mutex statusMutex;
};

void *cudaQuery(void *cudaTimer_) {
  auto *cudaTimer = reinterpret_cast<CudaTimer *>(cudaTimer_);
  cudaError_t res = cudaSuccess;  // NOLINT
  CudaRecord *curr_record = nullptr;
  while (true) {
    pthread_mutex_lock(&cudaTimer->mutex_profiling);
    while (!cudaTimer->stop_query && cudaTimer->profiling_records.empty()) {
      pthread_cond_wait(&cudaTimer->cond_profiling,
                        &cudaTimer->mutex_profiling);
    }
    if (cudaTimer->stop_query && cudaTimer->profiling_records.empty() &&
        !curr_record) {
      pthread_mutex_unlock(&cudaTimer->mutex_profiling);
      break;
    }
    if (!cudaTimer->profiling_records.empty()) {
      curr_record = cudaTimer->profiling_records.front();
    }
    pthread_mutex_unlock(&cudaTimer->mutex_profiling);
    while (curr_record) {
      res = cudaEventQuery(curr_record->end_event);
      if (res != cudaErrorNotReady) {
        if (res == cudaSuccess) {
          res = cudaEventElapsedTime(&curr_record->duration,
                                     curr_record->begin_event,
                                     curr_record->end_event);  // ms
          if (res != cudaSuccess) {
            const char *msg = nullptr;
            msg = cudaGetErrorString(res);
            std::stringstream safe_call_ss;  // NOLINT
            safe_call_ss
                << "Failed to query elapsed time from CUDA events, file:"
                << __FILE__ << ", line:" << __LINE__ << ", msg:" << msg;
            throw std::runtime_error(safe_call_ss.str());
          }
          TRACE(Logger::LogSubSys::TIMER) << "workload="<< toDebugStr(curr_record->recordKey.workload) << " candidate=" << toDebugStr(curr_record->recordKey.candidate) << " nccl kernel done.";
          pthread_mutex_lock(&cudaTimer->mutex_profiled);
          // curr_record->duration *= 1000000; // ms -> ns
          cudaTimer->profiled_records.push(curr_record);
          pthread_mutex_unlock(&cudaTimer->mutex_profiled);
          curr_record = NULL;
          pthread_mutex_lock(&cudaTimer->mutex_profiling);
          cudaTimer->profiling_records.pop();
          pthread_mutex_unlock(&cudaTimer->mutex_profiling);
        } else {
          const char *msg = nullptr;
          msg = cudaGetErrorString(res);
          std::stringstream safe_call_ss;  // NOLINT
          safe_call_ss << "Failed to query CUDA events, file:" << __FILE__
                       << ", line:" << __LINE__ << ", msg:" << msg;
          throw std::runtime_error(safe_call_ss.str());
        }
      }
    }
  }
  return NULL;
}
