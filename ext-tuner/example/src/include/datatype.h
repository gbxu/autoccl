#pragma once
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using GIDTYPE = uint64_t;
using KEY = uint64_t;
static_assert(
    std::is_same<KEY, GIDTYPE>::value,
    "KEY and GIDTYPE must be the same type due to Workload including groupID.");
using CONFIG = int32_t;

using Workload = std::vector<KEY>;
using Candidate = std::vector<CONFIG>;

std::string toDebugStr(Workload workload) {
  std::stringstream ss;  // NOLINT
  for (const auto &key : workload) {
    ss << key << " ";
  }
  ss << ";";
  return ss.str();
}

std::string toDebugStr(Candidate candidate) {
  std::stringstream ss;  // NOLINT
  for (const auto &config : candidate) {
    ss << config << " ";
  }
  ss << ";";
  return ss.str();
}

std::string toDebugStr(const std::map<std::string, int32_t> dict) {
  std::stringstream ss;  // NOLINT
  for (const auto &pair : dict) {
    ss << pair.first << ": " << pair.second << "; ";
  }
  ss << ";";
  return ss.str();
}

struct ExpireCandidate {
    static const int32_t FOREVER;
    int32_t expire;
    Candidate candidate;

    ExpireCandidate(int32_t expire, const Candidate &candidate)
        : expire(expire), candidate(candidate) {}
    ExpireCandidate() = default;

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << "[" << expire << "; ";
      ss << toDebugStr(candidate);
      ss << "] ";
      return ss.str();
    }

    template <class Archive> void serialize(Archive &ar) {
      ar(expire, candidate);
    }
};
const int32_t ExpireCandidate::FOREVER = std::numeric_limits<int32_t>::min();

struct RoundCandidates {
    static const int32_t INITVERSION;
    int32_t version;
    int32_t roundExpire;
    std::vector<ExpireCandidate> expireCandidates;

    RoundCandidates() = default;
    RoundCandidates(int32_t version, int32_t roundExpire,
                    const std::vector<ExpireCandidate> &expireCandidates)
        : version(version), roundExpire(roundExpire),
          expireCandidates(expireCandidates) {}

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << "RoundCandidates=[" << version << "; ";
      ss << roundExpire << "; ";
      if (expireCandidates.size() >= 4) {
        for (size_t i = 0; i < 2; ++i) {
          ss << expireCandidates[i].debugStr() << ", ";
        }
        ss << "... ";
        for (size_t i = expireCandidates.size() - 2;
             i < expireCandidates.size(); ++i) {
          ss << expireCandidates[i].debugStr() << ", ";
        }
      } else {
        for (const auto &expireCandidate : expireCandidates) {
          ss << expireCandidate.debugStr() << ", ";
        }
      }
      ss << "] ";
      return ss.str();
    }

    // cereal
    template <class Archive> void serialize(Archive &ar) {
      ar(version, roundExpire, expireCandidates);
    }
};
const int32_t RoundCandidates::INITVERSION = -1;

struct Result {
    Workload workload;  // including groupID
    RoundCandidates roundCandidates;

    Result() = default;
    Result(const Workload &workload, const RoundCandidates &roundCandidates)
        : workload(workload), roundCandidates(roundCandidates) {}

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << "[";
      for (const auto &key : workload) {
        ss << key << " ";
      }
      ss << "; ";
      ss << roundCandidates.debugStr() << "] ";
      return ss.str();
    }

    // cereal for serialization and de-serializaion
    template <class Archive> void serialize(Archive &ar) {
      ar(workload, roundCandidates);
    }
};

struct ResultPack {
    std::vector<Result> localResults;
    std::map<GIDTYPE, std::vector<Result>> distResults;

    ResultPack() = default;
};

struct Decision {
    Workload workload;
    int32_t version = RoundCandidates::INITVERSION;

    Decision() = default;
    Decision(const Workload &workload, int32_t version)
        : workload(workload), version(version) {}

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << "[";
      for (const auto &key : workload) {
        ss << key << " ";
      }
      ss << "; ";
      ss << version << "] ";
      return ss.str();
    }

    // cereal
    template <class Archive> void serialize(Archive &ar) {
      ar(workload, version);
    }
};

struct DecisionPack {
    std::vector<Decision> localDecisions;
    std::map<GIDTYPE, std::vector<Decision>> distDecisions;

    DecisionPack() = default;
};

struct SimpleQueryStatus {
    int32_t remainingExpire;
    int32_t activeVersion;

    SimpleQueryStatus() = default;
    SimpleQueryStatus(int32_t remainingExpire, int32_t activeVersion)
        : remainingExpire(remainingExpire), activeVersion(activeVersion) {}
};

struct QueryStatus {
    int32_t remainingExpire = 0;
    int32_t activeVersion = RoundCandidates::INITVERSION;
    int32_t currIndex = 0;
    std::queue<RoundCandidates> queue;

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << "QueryStatus=[remainingExpire=" << remainingExpire
         << " , activeVersion=" << activeVersion << " currIndex=" << currIndex
         << " RoundCandidatesQueueSize=" << queue.size() << "]";
      return ss.str();
    }

    QueryStatus() = default;
};

struct GroupInfo {
    static const int32_t EMPTY;
    GroupInfo()
        : groupID(std::numeric_limits<GIDTYPE>::max()), root(EMPTY),
          rank(EMPTY), nrank(EMPTY), nnode(EMPTY) {}
    GroupInfo(GIDTYPE groupID, int32_t root, int32_t rank, int32_t nrank, int32_t nnode, std::map<std::string, int32_t> tunerEnvs)
        : groupID(groupID), root(root), rank(rank), nrank(nrank), nnode(nnode),
          members(nrank), tunerEnvs(tunerEnvs) {}
    GIDTYPE groupID;
    int32_t root;  // LEADER if rank == root else FOLLOWER
    int32_t rank;
    int32_t nrank;
    int32_t nnode;
    std::vector<int32_t> members;  // ids
    // TODO(anonymous): gather tunerEnvs of all ranks
    std::map<std::string, int32_t> tunerEnvs;
    bool isLeader() const { return root == rank; }
    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << "groupID=" << groupID << " , root=" << root << " rank=" << rank
         << " nrank=" << nrank << " [";
      for (auto &member : members) {
        ss << member << " ";
      }
      ss << "]";
      return ss.str();
    }

    // cereal
    template <class Archive> void serialize(Archive &ar) {
      ar(groupID, root, rank, nrank, members, tunerEnvs);
    }
};
const int32_t GroupInfo::EMPTY = std::numeric_limits<int32_t>::min();

struct Info {
    static const int32_t EMPTY;
    Info() : id(EMPTY), port(EMPTY) {}
    Info(int32_t nodeID, int32_t deviceID)
        : id(EMPTY), port(EMPTY), role(WORKER), nodeID(nodeID),
          deviceID(deviceID) {}
    std::string hostname;
    int32_t port;
    int32_t id;
    int32_t nodeID;
    int32_t deviceID;
    std::vector<GIDTYPE> groupIDs;                // groupIDs
    enum Role : uint8_t { COORDINATOR, WORKER };  // cotuning, scheduling
    Role role;                                    // local info
    static const char *roleNames[];
    bool isCoordinator() const { return role == COORDINATOR; }
    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << "hostname=" << hostname << " port=" << port << " id=" << id
         << " nodeID=" << nodeID << " deviceID=" << deviceID
        << " groupIDs=";
      ss << "[";
      for (const auto &groupID : groupIDs) {
        ss << groupID << ",";
      }
      ss << "]";
      ss << " role=" << roleNames[role];
      return ss.str();
    }

    template <class Archive> void serialize(Archive &ar) {
      ar(hostname, port, id, nodeID, deviceID, groupIDs, role);
    }
};
const int32_t Info::EMPTY = std::numeric_limits<int32_t>::min();
const char *Info::roleNames[] = {"COORDINATOR", "WORKER"};

struct RecordKey {
    GIDTYPE groupID;
    Workload workload;
    Candidate candidate;

    RecordKey() = default;
    RecordKey(const GIDTYPE &groupID, const Workload &workload,
              const Candidate &candidate)
        : groupID(groupID), workload(workload), candidate(candidate) {}

    bool operator<(const RecordKey &other) const {
      if (groupID != other.groupID) {
        return groupID < other.groupID;
      } else if (workload != other.workload) {
        return workload < other.workload;
      } else {
        return candidate < other.candidate;
      }
    }

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << groupID << "; ";
      ss << toDebugStr(workload);
      ss << toDebugStr(candidate);
      return ss.str();
    }
};

struct RecordValue {
    float duration; // duration: ms
    float beginTs;  // timestamp: ms
    float endTs;  // timestamp: ms
    int32_t repeat;

    RecordValue() = default;
    RecordValue(float duration, int32_t repeat)
        : duration(duration), repeat(repeat) {}

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << duration << ";";
      ss << repeat << ";";
      return ss.str();
    }
};

struct RecordValueList : public std::vector<RecordValue> {
  // mean
  double mean() const {
    if (this->empty()) return 0.0;
    double sum = std::accumulate(this->begin(), this->end(), 0.0,
      [](double acc, const RecordValue& rv) { return acc + rv.duration; });
    return sum / this->size();
  }

  // max
  double max() const {
    if (this->empty()) return 0.0;
    return std::max_element(this->begin(), this->end(),
      [](const RecordValue& a, const RecordValue& b) { return a.duration < b.duration; })->duration;
  }

  // min
  double min() const {
    if (this->empty()) return 0.0;
    return std::min_element(this->begin(), this->end(),
      [](const RecordValue& a, const RecordValue& b) { return a.duration < b.duration; })->duration;
  }

  // stddev
  double stddev() const {
    if (this->size() < 2) return 0.0;
    double mean_value = mean();
    double accum = 0.0;
    for (const auto& rv : *this) {
      accum += (rv.duration - mean_value) * (rv.duration - mean_value);
    }
    return std::sqrt(accum / (this->size() - 1));
  }

  // var
  double var() const {
    if (this->size() < 2) return 0.0;
    double mean_value = mean();
    double accum = 0.0;
    for (const auto& rv : *this) {
      accum += (rv.duration - mean_value) * (rv.duration - mean_value);
    }
    return accum / (this->size() - 1);
  }

  // median
  double median() const {
    if (this->empty()) return 0.0;
    std::vector<RecordValue> sorted_data = *this;
    std::sort(sorted_data.begin(), sorted_data.end(),
      [](const RecordValue& a, const RecordValue& b) { return a.duration < b.duration; });
    size_t size = sorted_data.size();
    if (size % 2 == 0) {
      return (sorted_data[size / 2 - 1].duration + sorted_data[size / 2].duration) / 2.0;
    } else {
      return sorted_data[size / 2].duration;
    }
  }

  // percentile
  float percentile(float percent) const {
    if (this->empty()) return 0.0;
    if (percent < 0 || percent > 100) throw std::out_of_range("Percent must be between 0 and 100");

    std::vector<RecordValue> sorted_data = *this;
    std::sort(sorted_data.begin(), sorted_data.end(),
      [](const RecordValue& a, const RecordValue& b) { return a.duration < b.duration; });

    float index = (percent / 100.0) * (sorted_data.size() - 1);
    size_t lower = std::floor(index);
    size_t upper = std::ceil(index);
    if (lower == upper) {
      return sorted_data[lower].duration;
    } else {
      float weight = index - lower;
      return sorted_data[lower].duration * (1 - weight) + sorted_data[upper].duration * weight;
    }
  }

  // meanOfTopPercent
  double meanOfTopPercent(float percent) const {
    if (this->empty()) return 0.0;
    if (percent < 0 || percent > 100) throw std::out_of_range("Percent must be between 0 and 100");

    std::vector<RecordValue> sorted_data = *this;
    std::sort(sorted_data.begin(), sorted_data.end(),
      [](const RecordValue& a, const RecordValue& b) { return a.duration < b.duration; });

    size_t count = std::ceil((percent / 100.0) * sorted_data.size());
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
      sum += sorted_data[i].duration;
    }
    return sum / count;
  }

  std::string debugStr() const {
    std::stringstream ss;  // NOLINT
    for (const auto& rv : *this) {
      ss << rv.debugStr();
    }
    return ss.str();
  }
};

struct Record {
    RecordKey key;
    RecordValue value;

    Record() = default;
    Record(const RecordKey &key, const RecordValue &value)
        : key(key), value(value) {}

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << key.debugStr();
      ss << value.debugStr();
      return ss.str();
    }
};

struct KeyRange {
    static const KEY EMPTY;
    KEY begin;  // [
    KEY end;    // ]

    KeyRange(KEY begin = EMPTY, KEY end = EMPTY)
        : begin(begin), end(end) {}
};
const KEY KeyRange::EMPTY = std::numeric_limits<KEY>::max();

struct ConfigRange {
    static const CONFIG EMPTY;
    CONFIG begin;  // [
    CONFIG end;    // ]
    CONFIG unit;
    double weight;  // weighted
    double lr;  // learning rate
    bool used; // used or placehold

    ConfigRange(bool used, CONFIG begin = EMPTY, CONFIG end = EMPTY, CONFIG unit = EMPTY, double weight = 0, double lr = 0)
        : used(used), begin(begin), end(end), unit(unit), weight(weight), lr(lr) {}
};
const CONFIG ConfigRange::EMPTY = std::numeric_limits<CONFIG>::min();

using CandidateFunc = std::function<void(
    const Info &, const std::unordered_map<GIDTYPE, GroupInfo> &,
    const Workload &, bool scale2, std::vector<Candidate> *, std::vector<ConfigRange> *)>;

using CandidateFilter = Candidate;

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

std::ostream &operator<<(std::ostream &os, const Info::Role &cmd) {
  os << Info::roleNames[cmd];
  return os;
}
