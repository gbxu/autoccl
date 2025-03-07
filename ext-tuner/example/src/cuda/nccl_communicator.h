#include <map>
#include <memory>
#include <vector>
#include "src/cuda/nccl_socket.h"
#include "src/include/internal/threadsafe_queue.h"
#include "src/include/tuner.h"

#define USE_BOOTSTRAP 1
struct NcclGroup {
    NcclGroup() = default;
    explicit NcclGroup(struct bootstrapState *handler) : handler(handler) {}
    struct bootstrapState *handler;
    std::vector<std::unique_ptr<struct ncclSocket>> senders;
    std::vector<std::unique_ptr<struct ncclSocket>> recvers;
};

class NcclCommunicator : public Communicator {
  public:
    NcclCommunicator() = default;
    void start(const char *role = nullptr) override {
      receivingThread = std::unique_ptr<std::thread>(
          new std::thread([this]() { receiving(); }));
      pthread_setname_np(receivingThread->native_handle(), "receivingThread");
      history << myInfo.debugStr() << "\n";
    }

    void stop() override {
      INFO(Logger::LogSubSys::NET) << myInfo.role << " stopping communicator";
      CHECK(ncclGroups.empty()) << " still has some groups";
      Message msg;
      msg.meta.recver = myInfo.id;
      msg.meta.cmd = Meta::TERMINATE;
      this->sendMsg(msg);
      this->receivingThread->join();  // stop thread
    }

    void addGroup(GIDTYPE groupID, int32_t root, int32_t rank,
                  int32_t nrank, int32_t nnode, std::map<std::string, int32_t> tunerEnvs) override {
      THROWERROR << "addGroup is not supported by NcclCommunicator";
    }
    void addGroup(GIDTYPE groupID, int32_t root, int32_t rank, int32_t nrank,
                  int32_t nnode, std::map<std::string, int32_t> tunerEnvs, void *ncclHandler) {
      TRACE(Logger::LogSubSys::NET) << myInfo.role << " addGroup: groupID=" << groupID
            << ", root=" << root << ", rank=" << rank << ", nrank=" << nrank << ", nnode=" << nnode << ", bootstrap=" << ncclHandler;
      auto *handler = (struct bootstrapState *)ncclHandler;
      std::lock_guard<std::mutex> lk(allGroupInfosMutex);
      ncclGroups[groupID] = NcclGroup(handler);
      auto &senders = ncclGroups[groupID].senders;
      auto &recvers = ncclGroups[groupID].recvers;
      myInfo.groupIDs.push_back(groupID);
      GroupInfo groupInfo = GroupInfo(groupID, root, rank, nrank, nnode, tunerEnvs);
      history << groupInfo.debugStr() << "\n";
      allGroupInfos[groupID] = groupInfo;
#if USE_BOOTSTRAP
      for (int32_t peer = 0; peer < nrank; peer++) {
        senders.push_back(
            std::make_unique<ncclSocket>());  // using index to map sender
        recvers.push_back(std::make_unique<ncclSocket>());
      }
      if (rank == root) {
        for (int32_t peer = 0; peer < nrank; peer++) {
          if (rank != peer) {
            CHECK(ncclSocketInit(senders[peer].get(),
                                 handler->peerCommAddresses + peer,
                                 handler->magic, ncclSocketTypeBootstrap) == 0);
          }
        }
      } else {
        CHECK(ncclSocketInit(recvers[root].get()) ==
              0);  // only 1 recver for root
      }

      if (rank == root) {
        for (int32_t peer = 0; peer < nrank; peer++) {
          if (rank != peer) {
            CHECK(ncclSocketConnect(senders[peer].get()) == 0);
            INFO(Logger::LogSubSys::NET) << myInfo.role << " nccl communicator connecting: " << rank
                 << " to " << peer;
          }
        }
      } else {
        CHECK(ncclSocketAccept(recvers[root].get(), &handler->listenSock) == 0);
        INFO(Logger::LogSubSys::NET) << myInfo.role << " nccl communicator connecting: " << rank
             << " from " << root;
      }
#endif
      INFO(Logger::LogSubSys::NET) << myInfo.role << " nccl communicator connected";
    }

    void removeGroup(GIDTYPE groupID) override {
      TRACE(Logger::LogSubSys::NET) << myInfo.role << " removeGroup: groupID=" << groupID;
      std::lock_guard<std::mutex> lk(allGroupInfosMutex);
      auto &senders = ncclGroups[groupID].senders;
      auto &recvers = ncclGroups[groupID].recvers;
      int32_t root = allGroupInfos[groupID].root;    // NOHINT
      int32_t rank = allGroupInfos[groupID].rank;    // NOHINT
      int32_t nrank = allGroupInfos[groupID].nrank;  // NOHINT
#if USE_BOOTSTRAP
      if (rank == root) {
        for (int32_t peer = 0; peer < nrank; peer++) {
          if (rank != peer)
            CHECK(ncclSocketClose(senders[peer].get()) == 0);
        }
      } else {
        CHECK(ncclSocketClose(recvers[root].get()) == 0);
      }
#endif
      myInfo.groupIDs.erase(
          std::find(myInfo.groupIDs.begin(), myInfo.groupIDs.end(), groupID));
      allGroupInfos.erase(groupID);
      ncclGroups.erase(groupID);
    }
    void barrier(GIDTYPE groupID) override {
      THROWERROR << "barrier is not supported by NcclCommunicator";
    }

    void broadcastMsg(const Message &msg) override {
      TRACE(Logger::LogSubSys::NET) << myInfo.role << " broadcast msg=" << msg.debugStr();
      std::lock_guard<std::mutex> lk(allGroupInfosMutex);
      std::vector<uint8_t> buffer;  // NOHINT
      msg.serialize(&buffer);
      int32_t size = buffer.size();                                 // NOHINT
      if (allGroupInfos.count(msg.meta.recverGroupID) == 0) {
        // group is removed.
        return;
      }
      int32_t root = allGroupInfos[msg.meta.recverGroupID].root;    // NOHINT
      int32_t rank = allGroupInfos[msg.meta.recverGroupID].rank;    // NOHINT
      int32_t nrank = allGroupInfos[msg.meta.recverGroupID].nrank;  // NOHINT
      auto &senders = ncclGroups[msg.meta.recverGroupID].senders;   // NOHINT
      CHECK(rank == root);
      for (int32_t peer = 0; peer < nrank; peer++) {
        if (rank != peer) {
#if USE_BOOTSTRAP
          ncclSocketSend(senders[peer].get(), static_cast<void *>(&size),
                         sizeof(int32_t));
          ncclSocketSend(senders[peer].get(),
                         static_cast<void *>(buffer.data()), buffer.size());
          TRACE(Logger::LogSubSys::NET) << myInfo.role << " broadcast msg: send " << rank << " to "
                << peer;
#endif
        } else {
          Message msgCopy = msg;
          msgs.push(std::move(msgCopy));
          TRACE(Logger::LogSubSys::NET) << myInfo.role << " broadcast msg: send " << rank << " to "
                << peer;
        }
      }
      return;
    }

    void sendMsg(const Message &msg) override {
      CHECK(msg.meta.recver == myInfo.id && myInfo.id == Meta::EMPTY);
      if (msg.meta.recver == myInfo.id) {
        Message msgCopy = msg;
        msgs.push(std::move(msgCopy));
      }
    }

    void recvMsg(Message *msg) override {
      std::vector<uint8_t> buffer;  // NOHINT
      int32_t size = 0;             // NOHINT
      while (true) {
        if (!msgs.empty()) {
          msgs.waitPop(msg);
          TRACE(Logger::LogSubSys::NET) << myInfo.role << " get local msg=" << msg->debugStr();
          return;
        }
        std::lock_guard<std::mutex> lk(allGroupInfosMutex);
#if USE_BOOTSTRAP
        for (auto &pair : ncclGroups) {
          auto &groupID = pair.first;
          auto &root = allGroupInfos[groupID].root;
          auto &rank = allGroupInfos[groupID].rank;
          if (allGroupInfos[groupID].isLeader())
            continue;
          auto &recvers = pair.second.recvers;
          auto &recver = recvers[root];
          int closed = 0;
          ncclSocketTryRecv(recver.get(), static_cast<void *>(&size),
                            sizeof(int32_t), &closed, false /*blocking*/);
          if (size > 0) {
            buffer.resize(size);
            ncclSocketRecv(recver.get(), static_cast<void *>(buffer.data()),
                           size);
            msg->deserialize(buffer.data(), size);
            TRACE(Logger::LogSubSys::NET) << myInfo.role << " get remote msg=" << msg->debugStr();
            return;
          }
        }
#endif
      }
    }

  private:
    void startConnect(const Info &info) override{};
    void startListen(const Info &info) override{};
    ThreadsafeQueue<Message> msgs;
    std::map<GIDTYPE, NcclGroup> ncclGroups;
};
