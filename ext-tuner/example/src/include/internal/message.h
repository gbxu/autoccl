#pragma once
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <limits>
#include <queue>
#include <string>
#include <tuple>
#include <vector>
#include "src/include/datatype.h"
#include "src/include/internal/logging.h"

struct Meta {
    Meta()
        : sender(EMPTY), recver(EMPTY),
          recverGroupID(std::numeric_limits<GIDTYPE>::min()), cmd(NONE) {}
    static const int32_t EMPTY;
    int32_t sender;         // global id
    int32_t recver;         // global id
    GIDTYPE recverGroupID;  // group ID
    enum Command : uint8_t {
      REGISTER,
      ADD_GROUP,
      REMOVE_GROUP,
      BARRIER,
      DATA,
      TERMINATE,
      NONE,
      COUNT
    };
    static const char *commandNames[];
    Command cmd;
    // additional variables for ctrl
    std::vector<Info> infos;  // request+response: REGISTER
    std::vector<GroupInfo>
        groupInfos;          // request+response: ADD_GROUP, REMOVE_GROUP
    GIDTYPE barrierGroupID;  // request+response: BARRIER
    enum Type : uint8_t { RESULTS, DECISIONS };
    static const char *typeNames[];
    Type type;  // DATA type

    template <class Archive> void serialize(Archive &ar) {
      ar(sender, recver, recverGroupID, cmd, infos, groupInfos, barrierGroupID,
         type);
    }

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << "sender=" << sender;
      ss << " recver=" << recver;
      ss << " recverGroupID=" << recverGroupID;
      ss << " cmd=" << commandNames[cmd] << "; ";
      switch (cmd) {
      case REGISTER:
        for (auto &info : infos) {
          ss << info.debugStr() << ", ";
        }
        break;
      case BARRIER:
        ss << barrierGroupID << ";";
        break;
      case DATA:
        ss << typeNames[type] << ";";
        break;
      default:
        break;
      }
      return ss.str();
    }
};
const int32_t Meta::EMPTY = std::numeric_limits<int32_t>::min();
const char *Meta::commandNames[] = {"REGISTER", "ADD_GROUP", "REMOVE_GROUP",
                                    "BARRIER",  "DATA",      "TERMINATE",
                                    "NONE"};
const char *Meta::typeNames[] = {"RESULTS", "DECISIONS"};
static_assert(sizeof(Meta::commandNames) / sizeof(Meta::commandNames[0]) ==
                  Meta::COUNT,
              "commandNames array size does not match Command enum count");

struct Message {
    Meta meta;
    std::vector<Result> results;
    std::vector<Decision> decisions;
    Message() = default;
    explicit Message(std::vector<Result> &&results)
        : results(std::move(results)) {
      CHECK(this->results.size() > 0);
      meta.cmd = Meta::DATA;
      meta.type = Meta::RESULTS;
    }
    explicit Message(std::vector<Decision> &&decisions)
        : decisions(std::move(decisions)) {
      CHECK(this->decisions.size() > 0);
      meta.cmd = Meta::DATA;
      meta.type = Meta::DECISIONS;
    }

    template <class Archive> void serialize(Archive &ar) {
      ar(meta, results, decisions);
    }

    // serialize to binary data
    void serialize(std::vector<uint8_t> *buffer) const {
      std::ostringstream os;
      cereal::BinaryOutputArchive archive(os);
      archive(*this);
      const std::string &tmp = os.str();
      buffer->assign(tmp.begin(), tmp.end());
    }

    // de-serialize from binary data
    void deserialize(const uint8_t *buffer, size_t size) {
      std::string str(reinterpret_cast<const char *>(buffer), size);
      std::istringstream is(str);
      cereal::BinaryInputArchive archive(is);
      archive(*this);
    }

    std::string debugStr() const {
      std::stringstream ss;  // NOLINT
      ss << "meta=" << meta.debugStr() << "; ";
      if (meta.type == Meta::RESULTS) {
        for (const auto &result : results) {
          ss << result.debugStr();
        }
      } else if (meta.type == Meta::DECISIONS) {
        for (const auto &decision : decisions) {
          ss << decision.debugStr();
        }
      }
      return ss.str();
    }
};

std::ostream &operator<<(std::ostream &os, const Meta::Command &cmd) {
  CHECK(cmd >= Meta::REGISTER && cmd < Meta::COUNT);
  os << Meta::commandNames[cmd];
  return os;
}

std::ostream &operator<<(std::ostream &os, const Meta::Type &type) {
  os << Meta::typeNames[type];
  return os;
}
