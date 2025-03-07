#pragma once
#include <vector>
#include "src/include/datatype.h"

class Timer {
  public:
    virtual void begin(const RecordKey &recordKey, bool blocking,
                       void *context) = 0;
    virtual void end(GIDTYPE groupID, bool blocking) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void tryGetRecords(std::vector<Record> *records, bool blocking) = 0;
    virtual void setProfiling(const Workload &workload,
                              int32_t askedProfiling) = 0;
};
