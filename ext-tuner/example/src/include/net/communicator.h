#pragma once
#include <algorithm>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "src/include/internal/env.h"
#include "src/include/internal/message.h"
#include "src/include/net/utils.h"

class Communicator {
  public:
    Communicator() = default;
    virtual void broadcastMsg(const Message &msg) = 0;
    virtual void sendMsg(const Message &msg) = 0;
    void setMyInfo(Info info) { this->myInfo = info; }
    void setDataProcessor(std::function<void(const Message &)> func) {
      dataProcessor = std::move(func);
    }
    const Info& getMyInfo() { return this->myInfo; }
    std::unordered_map<GIDTYPE, GroupInfo> getAllGroupInfos() {
      std::lock_guard<std::mutex> lock(allGroupInfosMutex);
      return allGroupInfos;
    }
    virtual void start(const char *role);
    virtual void stop();
    virtual void addGroup(GIDTYPE groupID, int32_t root, int32_t rank,
                          int32_t nrank, int32_t nnode, std::map<std::string, int32_t> tunerEnvs);
    virtual void removeGroup(GIDTYPE groupID);
    virtual void barrier(GIDTYPE groupID);
    std::string debugStr() { return history.str(); }

  protected:
    Info myInfo;  // get from coordinator and system;
    std::unordered_map<GIDTYPE, GroupInfo> allGroupInfos;  // groupID to Group
    Info coordinatorInfo;  // get from enviroment
    std::stringstream history;
    std::mutex allGroupInfosMutex;
    std::atomic<bool> isStarted{false};
    virtual void recvMsg(Message *msg) = 0;
    virtual void startConnect(const Info &info) = 0;
    virtual void startListen(const Info &info) = 0;
    std::unique_ptr<std::thread> receivingThread;
    int32_t worldSize;  // used by coordinator
    int32_t terminateCount = 0;
    void receiving() {
      while (true) {
        Message msg;
        recvMsg(&msg);
        TRACE(Logger::LogSubSys::NET) << myInfo.role << " processing msg:" << msg.meta.debugStr();
        if (msg.meta.cmd == Meta::REGISTER) {
          processRegister(msg);
        } else if (msg.meta.cmd == Meta::ADD_GROUP) {
          processAddGroup(msg);
        } else if (msg.meta.cmd == Meta::REMOVE_GROUP) {
          processRemoveGroup(msg);
        } else if (msg.meta.cmd == Meta::BARRIER) {
          processBarrier(msg);
        } else if (msg.meta.cmd == Meta::DATA) {
          processDataMsg(msg);
        } else if (msg.meta.cmd == Meta::TERMINATE) {
          if (myInfo.isCoordinator()) {
            processTerminate(msg);
            if (terminateCount == worldSize) {  // excluding coordinator
              break;
            }
          } else {
            processTerminate(msg);
            break;
          }
        } else {
          THROWERROR << "invalid msg type.";
        }
      }
    }

  private:
    void processRegister(const Message &msg);
    void processAddGroup(const Message &msg);
    void processRemoveGroup(const Message &msg);
    void processDataMsg(const Message &msg);
    void processTerminate(const Message &msg);
    void processBarrier(const Message &msg);

    std::function<void(const Message &)> dataProcessor;
    std::unordered_map<int32_t, bool> barrierDone;  // groupID -> release
    std::mutex barrierDoneMutex;
    std::condition_variable barrierDoneCond;
    std::vector<Info> allInfos;
    std::unordered_map<GIDTYPE, int32_t>
        barrierCounts;  // groupID -> count; used by coordinator
};

void Communicator::start(const char *role) {
  CHECK(role != nullptr);
  // get coordinator info
  std::string coordinatorStr =
      Environment::get()->find("TUNER_COORDINATOR", "");
  CHECK(coordinatorStr.size() != 0);
  size_t pos = coordinatorStr.find(':');  // NOHINT
  coordinatorInfo.hostname = coordinatorStr.substr(0, pos);
  coordinatorInfo.port = std::stoi(coordinatorStr.substr(pos + 1));
  worldSize = Environment::get()->find("TUNER_WORLDSIZE", 0);
  CHECK(worldSize > 0) << " work size = " << worldSize;
  coordinatorInfo.id = worldSize;
  // get my node info
  if (strcmp(role, "COORDINATOR") == 0) {
    myInfo = coordinatorInfo;
    myInfo.role = Info::COORDINATOR;
  } else if (strcmp(role, "WORKER") == 0) {
    std::string addr = getMatchInterfaceIpPort(coordinatorStr.c_str());
    INFO(Logger::LogSubSys::NET) << myInfo.role << " Get listen interface:ip:port=" << addr;
    std::stringstream ss(addr);
    std::string interface, hostname;  // NOHINT
    int32_t port = 0;                 // NOHINT
    std::getline(ss, interface, ':');
    std::getline(ss, hostname, ':');
    ss >> port;
    myInfo.role = Info::WORKER;
    myInfo.hostname = hostname;
    myInfo.port = port;
  } else {
    THROWERROR << " invalid communicator role.";
  }
  // start receiving
  receivingThread = std::unique_ptr<std::thread>(
      new std::thread(&Communicator::receiving, this));
  pthread_setname_np(receivingThread->native_handle(), "receivingThread");

  // start the listen thread to generate new recver socket.
  // this thread will finish when receiving from stop msg
  startListen(myInfo);

  // connect to the coordinator
  startConnect(coordinatorInfo);

  // register workers to scheduler
  if (!myInfo.isCoordinator()) {
    // let the coordinator know myself
    Message msg;
    msg.meta.recver = coordinatorInfo.id;
    msg.meta.cmd = Meta::REGISTER;
    msg.meta.infos.push_back(myInfo);
    this->sendMsg(msg);
  }

  while (!isStarted) {  // after recving and processing the "register" response
                        // from the coordinator
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  TRACE(Logger::LogSubSys::NET) << myInfo.role << myInfo.debugStr() << " started";
}

void Communicator::stop() {
  INFO(Logger::LogSubSys::NET) << myInfo.role << " stopping communicator";
  // communicator shold stop after releasing all groups
  CHECK(myInfo.groupIDs.size() == 0)
      << " still have some groups.";
  if (!myInfo.isCoordinator()) {
    Message msg;
    msg.meta.sender = myInfo.id;
    // send msg to self
    msg.meta.recver = myInfo.id;
    msg.meta.cmd = Meta::TERMINATE;
    this->sendMsg(msg);
    // coordinator should receive from "terminate" from all workers
    msg.meta.recver =
        coordinatorInfo.id;
    this->sendMsg(msg);
  }
  this->receivingThread->join();  // stop thread
}

void Communicator::addGroup(GIDTYPE groupID, int32_t root, int32_t rank,
                            int32_t nrank, int32_t nnode, std::map<std::string, int32_t> tunerEnvs) {
  TRACE(Logger::LogSubSys::NET) << myInfo.role << " addGroup: groupID=" << groupID << ", root=" << root
        << ", rank=" << rank << ", nrank=" << nrank << toDebugStr(tunerEnvs);
  CHECK(myInfo.isCoordinator() != true) << "coordinator cannot add groups.";
  // waitAddGroup = false;
  Message msg;
  msg.meta.sender = myInfo.id;
  msg.meta.recver = coordinatorInfo.id;
  msg.meta.cmd = Meta::ADD_GROUP;
  GroupInfo groupInfo = GroupInfo(groupID, root, rank, nrank, nnode, tunerEnvs);
  msg.meta.groupInfos.push_back(groupInfo);
  this->sendMsg(msg);
  // while (!waitAddGroup) {
  //   std::this_thread::sleep_for(std::chrono::milliseconds(10));
  // }
  // waitAddGroup = false;
  myInfo.groupIDs.push_back(groupID);
  this->allGroupInfosMutex.lock();
  allGroupInfos[groupID] = groupInfo;
  this->allGroupInfosMutex.unlock();
  barrier(groupID);
}

void Communicator::removeGroup(GIDTYPE groupID) {
  TRACE(Logger::LogSubSys::NET) << myInfo.role << " removeGroup: groupID=" << groupID;
  CHECK(myInfo.isCoordinator() != true) << "coordinator cannot remove groups.";
  Message msg;
  msg.meta.sender = myInfo.id;
  msg.meta.recver = coordinatorInfo.id;
  msg.meta.cmd = Meta::REMOVE_GROUP;
  GroupInfo groupInfo;
  groupInfo.groupID = groupID;
  msg.meta.groupInfos.push_back(groupInfo);
  this->sendMsg(msg);
  myInfo.groupIDs.erase(
      std::find(myInfo.groupIDs.begin(), myInfo.groupIDs.end(), groupID));
  this->allGroupInfosMutex.lock();
  allGroupInfos.erase(groupID);
  this->allGroupInfosMutex.unlock();
}

void Communicator::barrier(GIDTYPE groupID) {
  TRACE(Logger::LogSubSys::NET) << myInfo.role << " barrier groupID=" << groupID;
  {
    std::lock_guard<std::mutex> lock(this->allGroupInfosMutex);
    if (allGroupInfos[groupID].nrank <= 1)
      return;
  }
  Message msg;
  msg.meta.sender = myInfo.id;
  msg.meta.recver = coordinatorInfo.id;
  msg.meta.cmd = Meta::BARRIER;
  msg.meta.barrierGroupID = groupID;
  this->sendMsg(msg);

  std::unique_lock<std::mutex> ulk(barrierDoneMutex);
  barrierDone[groupID] = false;
  barrierDoneCond.wait(ulk, [this, groupID] { return barrierDone[groupID]; });
}

void Communicator::processRegister(const Message &msg) {
  if (myInfo.isCoordinator()) {
    // add node
    CHECK(msg.meta.infos.size() == 1);
    allInfos.push_back(msg.meta.infos[0]);
    if (allInfos.size() == worldSize) {  // without coordinator
      // sort the nodes according their ip and port,
      std::sort(
          allInfos.begin(), allInfos.end(), [](const Info &a, const Info &b) {
            return (a.hostname.compare(b.hostname) | (a.port < b.port)) > 0;
          });
      // assign global id
      // register之后， coordinator 收到所有节点的信息，然后 连接
      // workers。从而区别 worker id -> senders
      for (int32_t id = 0; id < allInfos.size(); id++) {
        auto &info = allInfos[id];
        info.id = id;
        TRACE(Logger::LogSubSys::NET) << myInfo.role << " assign id=" << id << " to info "
              << info.debugStr();
        startConnect(
            info);  // coordinator节点连接所有worker节点，方便下发指令。
      }
      Message msg;
      msg.meta.sender = coordinatorInfo.id;
      msg.meta.infos = allInfos;
      msg.meta.cmd = Meta::REGISTER;
      for (int32_t id = 0; id < allInfos.size(); id++) {
        msg.meta.recver = id;
        sendMsg(msg);
      }
      isStarted = true;  // coordinator 收到所有节点，也可以继续了
    }
  } else {
    // workers get the updated info from the coordinator
    CHECK(msg.meta.infos.size() == worldSize);
    for (const auto &info : msg.meta.infos) {
      if (info.hostname == myInfo.hostname && info.port == myInfo.port) {
        myInfo = info;
        break;
      }
    }
    CHECK(myInfo.id != Info::EMPTY);
    startConnect(myInfo);  // now has id
    history << myInfo.debugStr() << "\n";
    allInfos = msg.meta.infos;
    // is ready now, the main thread restarts.
    isStarted = true;
  }
}

void Communicator::processAddGroup(const Message &msg) {
  if (myInfo.isCoordinator()) {
    this->allGroupInfosMutex.lock();
    CHECK(msg.meta.groupInfos.size() == 1);
    auto groupInfo = msg.meta.groupInfos[0];
    if (allGroupInfos.count(groupInfo.groupID) == 0) {
      groupInfo.rank = GroupInfo::EMPTY;
      groupInfo.members.clear();
      groupInfo.members.push_back(msg.meta.sender);
      allGroupInfos[groupInfo.groupID] = groupInfo;
    } else {
      allGroupInfos[groupInfo.groupID].members.push_back(msg.meta.sender);
    }
    if (allGroupInfos[groupInfo.groupID].members.size() ==
        allGroupInfos[groupInfo.groupID].nrank) {
      Message msg;
      msg.meta.sender = coordinatorInfo.id;
      msg.meta.recverGroupID = groupInfo.groupID;
      msg.meta.groupInfos.push_back(allGroupInfos[groupInfo.groupID]);
      msg.meta.cmd = Meta::ADD_GROUP;
      this->allGroupInfosMutex.unlock();
      broadcastMsg(msg);
    } else {
      this->allGroupInfosMutex.unlock();
    }
  } else {
    this->allGroupInfosMutex.lock();
    allGroupInfos[msg.meta.groupInfos[0].groupID].members =
        msg.meta.groupInfos[0].members;
    auto &groupInfo = allGroupInfos[msg.meta.groupInfos[0].groupID];
    history << groupInfo.debugStr() << "\n";
    if (groupInfo.isLeader()) {
      this->allGroupInfosMutex.unlock();
      // the leader should connect all members of its group
      for (auto &id : groupInfo.members) {
        startConnect(allInfos[id]);
      }
    } else {
      this->allGroupInfosMutex.unlock();
    }
    // waitAddGroup = true;
  }
}

void Communicator::processRemoveGroup(const Message &msg) {
  std::lock_guard<std::mutex> lock(allGroupInfosMutex);
  CHECK(myInfo.isCoordinator());
  CHECK(msg.meta.groupInfos.size() == 1);
  auto &groupInfo = msg.meta.groupInfos[0];
  auto &members = allGroupInfos[groupInfo.groupID].members;
  members.erase(std::find(members.begin(), members.end(),
                          static_cast<int32_t>(msg.meta.sender)));
  if (members.size() == 0) {
    allGroupInfos.erase(groupInfo.groupID);
  }
}

void Communicator::processDataMsg(const Message &msg) { dataProcessor(msg); }

void Communicator::processTerminate(const Message &msg) {
  isStarted = false;
  terminateCount++;
}

void Communicator::processBarrier(const Message &msg) {
  if (myInfo.isCoordinator()) {
    auto &barrierGroupID = msg.meta.barrierGroupID;
    if (barrierCounts.count(barrierGroupID) == 0) {
      barrierCounts[barrierGroupID] = 0;
    }
    barrierCounts[barrierGroupID]++;
    this->allGroupInfosMutex.lock();
    int32_t nrank = allGroupInfos[barrierGroupID].nrank;  // NOHINT
    this->allGroupInfosMutex.unlock();
    if (barrierCounts[barrierGroupID] == nrank) {
      barrierCounts[barrierGroupID] = 0;
      Message msg;
      msg.meta.sender = coordinatorInfo.id;
      msg.meta.recverGroupID = barrierGroupID;
      msg.meta.barrierGroupID = barrierGroupID;
      msg.meta.cmd = Meta::BARRIER;
      broadcastMsg(msg);
    }
  } else {
    {
      std::lock_guard<std::mutex> lock(barrierDoneMutex);
      barrierDone[msg.meta.barrierGroupID] = true;
    }
    barrierDoneCond.notify_all();
  }
}
