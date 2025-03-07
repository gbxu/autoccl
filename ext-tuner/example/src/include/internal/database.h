#pragma once
#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "src/include/datatype.h"
#include "src/include/internal/logging.h"
#include "src/include/timer.h"

class Database {
  public:
    // TODO: reload from previous database
    RecordValue fetct(const RecordKey &key) {
      std::lock_guard<std::mutex> lock(this->mutex);
      // pthread_mutex_lock(&this->mutex);
      if (table.count(key) == 0) {
        // pthread_mutex_unlock(&this->mutex);
        return {-1, 0};
      } else {
        auto item = table[key];
        // pthread_mutex_unlock(&this->mutex);
        return item;
      }
    }

    void add(const Record &record) {
      std::lock_guard<std::mutex> lock(this->mutex);
      // pthread_mutex_lock(&this->mutex);
      if (table.count(record.key) == 0) {
        table[record.key] = record.value;
      } else {
        table[record.key] =
            RecordValue((table[record.key].duration * table[record.key].repeat +
                         record.value.duration * record.value.repeat) /
                            (table[record.key].repeat + record.value.repeat),
                        (table[record.key].repeat + record.value.repeat));
      }
      log.push_back(record);
      // pthread_mutex_unlock(&this->mutex);
    }
    std::string dataDebugStr();
    std::string logDebugStr();

  private:
    std::map<RecordKey, RecordValue> table;
    std::vector<Record> log;
    std::mutex mutex;
    // pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
};

std::string Database::dataDebugStr() {
  std::lock_guard<std::mutex> lock(this->mutex);
  // pthread_mutex_lock(&this->mutex);
  std::stringstream ss;  // NOLINT
  // the sorted records
  std::vector<std::pair<RecordKey, RecordValue>> sortedRecords;

  // group records by map
  std::map<Workload, std::vector<std::pair<RecordKey, RecordValue>>> groupedRecords;
  for (const auto& entry : this->table) {
    groupedRecords[entry.first.workload].emplace_back(entry);
  }
  // sort each group by duration value
  for (auto& group : groupedRecords) {
    auto& records = group.second;
    std::sort(records.begin(), records.end(), [](const auto& a, const auto& b) {
        return a.second.duration < b.second.duration;
    });
    // update sortedRecords
    sortedRecords.insert(sortedRecords.end(), records.begin(), records.end());
  }

  for (const auto &pair : sortedRecords) {
    ss << pair.first.debugStr();
    ss << pair.second.debugStr() << "\n";
  }
  // pthread_mutex_unlock(&this->mutex);
  return ss.str();
}

std::string Database::logDebugStr() {
  std::lock_guard<std::mutex> lock(this->mutex);
  // pthread_mutex_lock(&this->mutex);
  std::stringstream ss;  // NOLINT
  for (const auto &record : log) {
    ss << record.debugStr() << "\n";
  }
  // pthread_mutex_unlock(&this->mutex);
  return ss.str();

  // TODO: chrome timeline
}
