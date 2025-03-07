#pragma once
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "src/include/tuner.h"

class SocketCommunicator : public Communicator {
  public:
    SocketCommunicator() = default;
    SocketCommunicator(const SocketCommunicator &) = delete;
    SocketCommunicator &operator=(const SocketCommunicator &) = delete;
    SocketCommunicator(SocketCommunicator &&) = delete;
    SocketCommunicator &operator=(SocketCommunicator &&) = delete;

    ~SocketCommunicator() = default;

    void stop() override;
    void sendMsg(const Message &msg) override;
    void broadcastMsg(const Message &msg) override;

  protected:
    void startListen(const Info &info) override;
    void startConnect(const Info &info) override;
    void recvMsg(Message *msg) override;

  private:
    std::map<int32_t, int32_t> senders;
    std::mutex sendersMutex;
    std::vector<int32_t> recvers;
    std::mutex recversMutex;
    void *listener;
    std::unique_ptr<std::thread> listenThread;
    std::atomic<bool> isListen{false};
};

void SocketCommunicator::stop() {
  Communicator::stop();
  isListen = false;
  listenThread->join();
  for (const auto &sender : senders) {
    SYSCHECK(close(sender.second), "close sender");
  }
  senders.clear();
  std::lock_guard<std::mutex> lock(recversMutex);
  for (int32_t recver : recvers) {
    SYSCHECK(close(recver), "close recver");
  }
  recvers.clear();
}

void SocketCommunicator::startListen(const Info &info) {
  std::ostringstream oss;  // NOHINT
  oss << info.hostname << ":" << info.port;
  INFO(Logger::LogSubSys::NET) << myInfo.role << " listening " << oss.str();
  union SocketAddress socketAddr;
  CHECK(SocketGetAddrFromString(&socketAddr, oss.str().c_str()) == 0);
  /* IPv4/IPv6 support */
  int32_t family = socketAddr.sa.sa_family;
  CHECK(family == AF_INET || family == AF_INET6);
  /* Connect to a hostname / port */
  int32_t listener = socket(family, SOCK_STREAM, 0);
  CHECK(listener != -1);
  // set port
  if (family == AF_INET) {
    socketAddr.sin.sin_port = htons(info.port);
  } else {
    socketAddr.sin6.sin6_port = htons(info.port);
  }
  // Port is forced by env. Make sure we get the port.
  int32_t opt = 1;
  SYSCHECK(setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)),
           "setsockopt");

  socklen_t salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                        : sizeof(struct sockaddr_in6);

  SYSCHECK(bind(listener, &socketAddr.sa, salen), "bind");
  SYSCHECK(getsockname(listener, &socketAddr.sa, &salen), "getsockname");
  SYSCHECK(listen(listener, worldSize + 1), "listen");

  // 设置套接字为非阻塞模式
  // non-blocking mode for socket
  int32_t flags = fcntl(listener, F_GETFL, 0);
  SYSCHECK(fcntl(listener, F_SETFL, flags | O_NONBLOCK), "fcntl");
  isListen = true;
  // accept the connection of clients
  listenThread = std::make_unique<std::thread>([this, socketAddr, listener]() {
    int32_t family = socketAddr.sa.sa_family;
    int32_t newRecvSocket = -1;
    socklen_t salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                          : sizeof(struct sockaddr_in6);
    while (isListen) {
      newRecvSocket =
          accept(listener, const_cast<sockaddr *>(&socketAddr.sa), &salen);
      if (newRecvSocket != -1) {
        if (newRecvSocket >= FD_SETSIZE) {
          WARN(Logger::LogSubSys::NET) << "newRecvSocket >= FD_SETSIZE, It's invalid for select(). "
                  "Re-connect is needed. "
                  "newRecvSocket="
               << newRecvSocket << " FD_SETSIZE=" << FD_SETSIZE;
        }
        std::lock_guard<std::mutex> lock(recversMutex);
        recvers.push_back(newRecvSocket);
      } else {
        if (errno == EWOULDBLOCK || errno == EAGAIN) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 10 ms
        } else {
          SYSCHECK(-1, "accept");
        }
      }
    }
    SYSCHECK(close(listener), "close listener");
  });
}

void SocketCommunicator::startConnect(const Info &info) {
  if (senders.count(info.id))
    return;
  std::ostringstream oss;  // NOHINT
  oss << info.hostname << ":" << info.port;
  INFO(Logger::LogSubSys::NET) << myInfo.role << " connecting " << oss.str();
  union SocketAddress socketAddr;
  CHECK(SocketGetAddrFromString(&socketAddr, oss.str().c_str()) == 0);
  /* IPv4/IPv6 support */
  int32_t family = socketAddr.sa.sa_family;
  CHECK(family == AF_INET || family == AF_INET6);
  /* Connect to a hostname / port */
  int32_t newSendSocket = socket(family, SOCK_STREAM, 0);
  CHECK(newSendSocket != -1);
  // set port
  if (family == AF_INET) {
    socketAddr.sin.sin_port = htons(info.port);
  } else {
    socketAddr.sin6.sin6_port = htons(info.port);
  }
  // connecting the peer
  socklen_t salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                        : sizeof(struct sockaddr_in6);
  SYSCHECK(connect(newSendSocket, &socketAddr.sa, salen), "connect");
  {
    std::lock_guard<std::mutex> lock(sendersMutex);
    senders[info.id] = newSendSocket;
  }
}

void SocketCommunicator::sendMsg(const Message &msg) {
  TRACE(Logger::LogSubSys::NET) << myInfo.role << " sendMsg:" << msg.debugStr();
  std::lock_guard<std::mutex> lk(sendersMutex);
  std::vector<uint8_t> buffer;  // NOHINT
  msg.serialize(&buffer);
  int32_t size = buffer.size();  // NOHINT
  CHECK(senders.count(msg.meta.recver) != 0);
  int32_t netSize = htonl(size);
  SYSCHECK(
      send(senders[msg.meta.recver], &netSize, sizeof(int32_t), MSG_NOSIGNAL),
      "send");
  SYSCHECK(send(senders[msg.meta.recver], buffer.data(), size, MSG_NOSIGNAL),
           "send");
}
#if 0
void SocketCommunicator::recvMsg(Message *msg) {
  while (true) {
    fd_set readfds;
    struct timeval tv;
    int32_t retval, maxfd = 0;

    FD_ZERO(&readfds);
    {
      std::lock_guard<std::mutex> lock(recversMutex);
      for (const auto &recver : recvers) {
        if (recver >= 0 && recver < FD_SETSIZE) {
          FD_SET(recver, &readfds);
          if (recver > maxfd) {
            maxfd = recver;
          }
        } else {
          WARN(Logger::LogSubSys::NET) << "Invalid file descriptor " << recver;
        }
      }
    }
    tv.tv_sec = 5;  // Timeout after 5 seconds
    tv.tv_usec = 0;

    retval = select(maxfd + 1, &readfds, NULL, NULL, &tv);
    if (retval > 0) {
      std::lock_guard<std::mutex> lock(recversMutex);
      for (const auto &recver : recvers) {
        if (FD_ISSET(recver, &readfds)) {
          int32_t netSize = 0;
          retval = recv(recver, static_cast<void *>(&netSize), sizeof(int32_t),
                        MSG_DONTWAIT);
          if (retval <= 0) {
            // Handle errors or disconnection
            continue;
          }
          int32_t size = ntohl(netSize);
          if (size > 0) {
            TRACE(Logger::LogSubSys::NET) << myInfo.role << " recvMsg size=" << size;
            std::vector<uint8_t> buffer;
            buffer.resize(size);
            int32_t received = 0;
            while (received < size) {
              retval =
                  recv(recver, buffer.data() + received, size - received, 0);
              if (retval <= 0) {
                // Handle errors or disconnection
                break;
              }
              received += retval;
            }
            if (received == size) {
              msg->deserialize(buffer.data(), buffer.size());
              TRACE(Logger::LogSubSys::NET) << myInfo.role << " Received message cmd=" << msg->meta.cmd
                    << "; size=" << size;
              // or push into msg queue
              return;
            }
          }
        }
      }
    } else if (retval == 0) {
      // Timeout, handle accordingly
      continue;
    } else {
      perror("select()");
    }
  }
}
#else
void SocketCommunicator::recvMsg(Message *msg) {
  while (true) {
    std::lock_guard<std::mutex> lock(recversMutex);
    for (const auto &recver : recvers) {
      int32_t netSize = 0;
      int32_t retval = recv(recver, static_cast<void *>(&netSize),
                            sizeof(int32_t), MSG_DONTWAIT);
      if (retval <= 0) {
        // Handle errors or disconnection
        continue;
      }
      int32_t size = ntohl(netSize);
      if (size > 0) {
        TRACE(Logger::LogSubSys::NET) << myInfo.role << " recvMsg size=" << size;
        std::vector<uint8_t> buffer;  // NOHINT
        buffer.resize(size);
        int32_t received = 0;
        while (received < size) {
          retval = recv(recver, buffer.data() + received, size - received, 0);
          if (retval <= 0) {
            // Handle errors or disconnection
            break;
          }
          received += retval;
        }
        if (received == size) {
          msg->deserialize(buffer.data(), buffer.size());
          TRACE(Logger::LogSubSys::NET) << myInfo.role << " Received message cmd=" << msg->meta.cmd
                << "; size=" << size;
          return;
        }
      }
    }
  }
}
#endif

void SocketCommunicator::broadcastMsg(const Message &msg) {
  TRACE(Logger::LogSubSys::NET) << myInfo.role << " broadcast msg=" << msg.debugStr();
  std::vector<uint8_t> buffer;  // NOHINT
  msg.serialize(&buffer);
  int32_t size = buffer.size();  // NOHINT
  int32_t netSize = htonl(size);
  std::vector<int32_t> members;
  {
    std::lock_guard<std::mutex> lock(this->allGroupInfosMutex);
    members = allGroupInfos[msg.meta.recverGroupID].members;
  }
  for (const auto &member : members) {
    int32_t sender;
    {
      std::lock_guard<std::mutex> lock(this->sendersMutex);
      CHECK(senders.count(member) != 0)
          << " groupID=" << msg.meta.recverGroupID << " member=" << member;
      sender = senders[member];
    }
    SYSCHECK(send(sender, &netSize, sizeof(int32_t), MSG_NOSIGNAL), "send");
    SYSCHECK(send(sender, buffer.data(), size, MSG_NOSIGNAL), "send");
  }
}
