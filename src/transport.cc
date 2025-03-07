/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "bootstrap.h"
#define ENABLE_TIMER 0
#include "timer.h"

struct ncclTransport* ncclTransports[NTRANSPORTS] = {
  &p2pTransport,
#if NTRANSPORTS == 5
  &p2pTransportCE,
#endif
  &shmTransport,
  &netTransport,
  &collNetTransport
};

template <int type>
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect, int channelId, int peer, int connIndex, int* transportType) {
  struct ncclPeerInfo* myInfo = comm->peerInfo+comm->rank;
  struct ncclPeerInfo* peerInfo = comm->peerInfo+peer;
  struct ncclConnector* connector = (type == 1) ? comm->channels[channelId].peers[peer]->send + connIndex :
                                                  comm->channels[channelId].peers[peer]->recv + connIndex;
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports[t];
    if (transport == &p2pTransportCE) continue;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    int p2pRet = 1;
    int currP2pLevel;
    if (transport == &p2pTransport) {
      // Check topology / p2p level.
      NCCLCHECK(ncclTopoCheckP2p(comm->topo, myInfo->busId, peerInfo->busId, &p2pRet, NULL, NULL, (*comm->tunerEnvs)["tuner_extraP2PCE_disable"] && (*comm->tunerEnvs)["tuner_extraSHM_disable"], (*comm->tunerEnvs)["tuner_p2pLevel"], &currP2pLevel));
    }
    NCCLCHECK(transport->canConnect(&ret, comm->topo, graph, myInfo, peerInfo));
    if (ret && p2pRet) {
      connector->transportComm = transportComm;
      // net or shm will use connect+0 if not p2pSupport
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
      if (transportType) *transportType = t;
      if (transport==&p2pTransport) {
        comm->channels[channelId].peers[peer]->p2pLevel = currP2pLevel; // keep for tuner
        comm->channels[channelId].peers[peer]->transportMask = 0b00000001;
        for (int i=1; i<TRANSPORT_NUM; ++i) {
          // for p2p_ce and shm
          struct ncclConnector* connector_extra = (type == 1) ? comm->channels[channelId].peers[peer]->send + connIndex + NCCL_MAX_CONNS*i:
                                                    comm->channels[channelId].peers[peer]->recv + connIndex + NCCL_MAX_CONNS*i;
          connector_extra->transportComm = NULL;
          struct ncclTransport *transport_extra = NULL;
          if (comm->tuner == NULL) {
            continue;
          }
          if (i == 1) {
            // p2p_ce
            transport_extra = ncclTransports[TRANSPORT_P2P_CE];
          } else if (i == 2) {
            // shm
            transport_extra = ncclTransports[TRANSPORT_SHM];
          } else {
            return ncclInternalError;
          }
          if ((*comm->tunerEnvs)["tuner_extraP2PCE_disable"] && transport_extra == &p2pTransportCE) continue;
          if ((*comm->tunerEnvs)["tuner_extraSHM_disable"] && transport_extra == &shmTransport) continue;
          int p2pCERet = 1;
          if (transport == &p2pTransportCE) {
            // Check topology / p2p level.
            int intermediateRank;
            NCCLCHECK(ncclTopoCheckP2p(comm->topo, myInfo->busId, peerInfo->busId, &p2pCERet, NULL, &intermediateRank, false, (*comm->tunerEnvs)["tuner_p2pLevel"], NULL));
            if (intermediateRank != -1) {
              p2pCERet = 0;
            }
          }
          NCCLCHECK(transport_extra->canConnect(&ret, comm->topo, graph, myInfo, peerInfo));
          if (ret && p2pCERet) {
            connector_extra->transportComm = type == 1 ? &transport_extra->send : &transport_extra->recv;
            NCCLCHECK(connector_extra->transportComm->setup(comm, graph, myInfo, peerInfo, connect+i, connector_extra, channelId, connIndex));
            comm->channels[channelId].peers[peer]->transportMask |= (1 << i);
            if (transport_extra == &p2pTransportCE) {
              (*comm->tunerEnvs)["tuner_extraP2PCE"] = 1;
            }
            if (transport_extra == &shmTransport) {
              (*comm->tunerEnvs)["tuner_extraSHM"] = 1;
            }
            TRACE(NCCL_INIT, "%s:%d channel=%d peer=%d %s connIndex=%d transportComm=%p(extra) transport=%d", __FILE__, __LINE__, channelId, peer, graph ? (type ? "send(coll)" : "recv(coll)") : (type ? "send" : "recv"), connIndex+NCCL_MAX_CONNS*i, connector_extra->transportComm, i);
          }
        }
      } else if (transport==&shmTransport || transport==&netTransport) {
        // shared memory or network
        comm->channels[channelId].peers[peer]->p2pLevel = PATH_DIS; // No p2p
        comm->channels[channelId].peers[peer]->transportMask |= 1 << t;
      } else if (transport==&collNetTransport) {
        WARN("Use collNetTransport for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
        comm->channels[channelId].peers[peer]->p2pLevel = PATH_DIS; // No p2p
        comm->channels[channelId].peers[peer]->transportMask |= 1 << t;
      }
      TRACE(NCCL_INIT, "%s:%d channel=%d peer=%d %s connIndex=%d transportComm=%p(curr) transport=%d transportMask=%x", __FILE__, __LINE__, channelId, peer, graph ? (type ? "send(coll)" : "recv(coll)") : (type ? "send" : "recv"), connIndex, connector->transportComm, t, comm->channels[channelId].peers[peer]->transportMask);
      return ncclSuccess;
    }
  }
  WARN("No transport found for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  return ncclSystemError;
}

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  struct ncclChannel* channel = &comm->channels[channelId];
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->recv[connIndex].connected) continue;
    comm->connectRecv[peer][channel->id] = true;
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || peer == comm->rank || channel->peers[peer]->send[connIndex].connected) continue;
    comm->connectSend[peer][channel->id] = true;
  }
  return ncclSuccess;
}

void dumpData(struct ncclConnect* data, int ndata) {
  for (int n=0; n<ndata; n++) {
    printf("[%d] ", n);
    uint8_t* d = (uint8_t*)data;
    for (int i=0; i<sizeof(struct ncclConnect); i++) printf("%02x", d[i]);
    printf("\n");
  }
}

ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex, int* highestTransportType/*=NULL*/) {
  // Stream used during transport setup; need for P2P pre-connect + CUDA Graph
  ncclResult_t ret = ncclSuccess;
  int highestType = TRANSPORT_P2P;  // track highest transport type
  struct ncclConnect** data = (ncclConnect**) malloc(sizeof(ncclConnect*) * comm->nRanks); // Store intermediate send/recvData structs for connect
  struct ncclConnect** recvData = (ncclConnect**) malloc(sizeof(ncclConnect*) * comm->nRanks); // Points to entries inside data for given recv connection within a channel
  struct ncclConnect** sendData = (ncclConnect**) malloc(sizeof(ncclConnect*) * comm->nRanks); // Points to entries inside data for given send connection within a channel

  NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->hostStream), ret, fail);
  // First time initialization
  for (int i=1; i<comm->nRanks; i++) {
    int bootstrapTag = (i<<8) + (graph ? graph->id+1 : 0);
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;

    // Data[i] contains all ncclConnect information for all send and receive connections with a given send and recv peer
    // This data is packed in the array based on the number of sendChannels and recvChannels connected with these peers
    // The first N entries contain recvData, connection information for recv connections
    // The next M entries contain sendData, connection information for send connections
    // It's not guaranteed that each entry of data has the same number of total or send/recv specific connections
    // 2 for send and recv
    // TUNER_MAXCHANNELS for channels
    // TRANSPORT_NUM for extra intra-node gpu transport: P2PCE(Copy Engine not SM copy), SHM
    data[i] = (ncclConnect*) malloc(sizeof(ncclConnect) * 2 * TUNER_MAXCHANNELS * TRANSPORT_NUM);
    recvData[i] = data[i];
    int sendChannels = 0, recvChannels = 0;
    int type;
    TIME_START(0);
    for (int c=0; c<TUNER_MAXCHANNELS; c++) {
      if (comm->connectRecv[recvPeer][c]) {
        NCCLCHECKGOTO(selectTransport<0>(comm, graph, recvData[i]+recvChannels*TRANSPORT_NUM, c, recvPeer, connIndex, &type), ret, fail);
        recvChannels += 1;
        if (type > highestType) highestType = type;
      }
    }
    TIME_STOP(0);
    TIME_START(1);
    sendData[i] = recvData[i]+recvChannels*TRANSPORT_NUM;
    for (int c=0; c<TUNER_MAXCHANNELS; c++) {
      if (comm->connectSend[sendPeer][c]) {
        NCCLCHECKGOTO(selectTransport<1>(comm, graph, sendData[i]+sendChannels*TRANSPORT_NUM, c, sendPeer, connIndex, &type), ret, fail);
        sendChannels += 1;
        if (type > highestType) highestType = type;
      }
    }
    TIME_STOP(1);

    TIME_START(2);
    TRACE(NCCL_INIT, "%s:%d recvChannels=%d for recvPeer=%d, sendChannels=%d for sendPeer=%d", __FILE__, __LINE__, recvChannels, recvPeer, sendChannels, sendPeer);
    if (sendPeer == recvPeer) {
      if (recvChannels+sendChannels) {
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data[i], sizeof(struct ncclConnect)*(recvChannels+sendChannels)*TRANSPORT_NUM), ret, fail);
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data[i], sizeof(struct ncclConnect)*(recvChannels+sendChannels)*TRANSPORT_NUM), ret, fail);
        sendData[i] = data[i];
        recvData[i] = data[i]+sendChannels*TRANSPORT_NUM;
      }
    } else {
      if (recvChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData[i], sizeof(struct ncclConnect)*recvChannels*TRANSPORT_NUM), ret, fail);
      if (sendChannels) NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData[i], sizeof(struct ncclConnect)*sendChannels*TRANSPORT_NUM), ret, fail);
      if (sendChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData[i], sizeof(struct ncclConnect)*sendChannels*TRANSPORT_NUM), ret, fail);
      if (recvChannels) NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData[i], sizeof(struct ncclConnect)*recvChannels*TRANSPORT_NUM), ret, fail);
    }
    TIME_STOP(2);
  }

  // Loop until all channels with all ranks have been connected
  bool allChannelsConnected;
  allChannelsConnected = false;
  while (!allChannelsConnected) {
    allChannelsConnected = true;
    for (int i=1; i<comm->nRanks; i++) {
      int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
      int sendPeer = (comm->rank + i) % comm->nRanks;

      int sendDataOffset = 0;
      int recvDataOffset = 0;
      for (int c=0; c<TUNER_MAXCHANNELS; c++) {
          TIME_START(3);
          if (comm->connectSend[sendPeer][c]) {
            for (int extra_index=0; extra_index<TRANSPORT_NUM; extra_index++) {
              struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex + NCCL_MAX_CONNS*extra_index;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                if (conn->transportComm != NULL) {
                  NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[i] + sendDataOffset, 1, comm->rank, conn), ret, fail);
                  if (ret == ncclSuccess) {
                    TRACE(NCCL_INIT, "%s:%d channel=%d Peer=%d send connIndex=%d transportComm=%p connected", __FILE__, __LINE__, c, sendPeer, connIndex + NCCL_MAX_CONNS*extra_index, conn->transportComm);
                    struct ncclDevChannelPeer* addr;
                    conn->connected = 1;
                    /* comm->channels[c].devPeers[sendPeer]->send[connIndex] is a device memory access. */
                    // CUDACHECKGOTO(cudaMemcpy(&addr, &comm->channels[c].devPeers[sendPeer], sizeof(struct ncclDevChannelPeer*), cudaMemcpyDeviceToHost), ret, fail);
                    // CUDACHECKGOTO(cudaMemcpyAsync(&addr->send[connIndex + NCCL_MAX_CONNS*extra_index], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), ret, fail);
                    CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex + NCCL_MAX_CONNS*extra_index], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), ret, fail);
                  } else if (ret == ncclInProgress) {
                    allChannelsConnected = false;
                  }
                }
                sendDataOffset += 1;
              }
            }
          }
          TIME_STOP(3);

          // Start with recv channels
          TIME_START(4);
          if (comm->connectRecv[recvPeer][c]) {
            for (int extra_index=0; extra_index<TRANSPORT_NUM; extra_index++) {
              struct ncclConnector* conn = comm->channels[c].peers[recvPeer]->recv + connIndex + NCCL_MAX_CONNS*extra_index;
              // This connector hasn't completed connection yet
              if (conn->connected == 0) {
                if (conn->transportComm != NULL) {
                  NCCLCHECKGOTO(conn->transportComm->connect(comm, recvData[i] + recvDataOffset, 1, comm->rank, conn), ret, fail);
                  if (ret == ncclSuccess) {
                    TRACE(NCCL_INIT, "%s:%d channel=%d Peer=%d recv connIndex=%d transportComm=%p connected", __FILE__, __LINE__, c, recvPeer, connIndex + NCCL_MAX_CONNS*extra_index, conn->transportComm);
                    struct ncclDevChannelPeer* addr;
                    conn->connected = 1;
                    /* comm->channels[c].devPeers[recvPeer]->recv[connIndex] is a device memory access. */
                    // CUDACHECKGOTO(cudaMemcpy(&addr, &comm->channels[c].devPeers[recvPeer], sizeof(struct ncclDevChannelPeer*), cudaMemcpyDeviceToHost), ret, fail);
                    // CUDACHECKGOTO(cudaMemcpyAsync(&addr->recv[connIndex+NCCL_MAX_CONNS*extra_index], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), ret, fail);
                    CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[recvPeer]->recv[connIndex + NCCL_MAX_CONNS*extra_index], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, comm->sharedRes->hostStream.cudaStream), ret, fail);
                  } else if (ret == ncclInProgress) {
                    allChannelsConnected = false;
                  }
                }
                recvDataOffset += 1;
              }
            }
          }
          TIME_STOP(4);
      }
    }
  }

  // Clear all connect masks and free each connectInfo array
  for (int i=1; i<comm->nRanks; i++) {
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    memset(comm->connectRecv[recvPeer], 0, TUNER_MAXCHANNELS * sizeof(bool));
    memset(comm->connectSend[sendPeer], 0, TUNER_MAXCHANNELS * sizeof(bool));
    free(data[i]);
  }

  free(data);
  free(sendData);
  free(recvData);

  if (highestTransportType != NULL) *highestTransportType = highestType;
  TIME_PRINT("P2P Setup/Connect");
exit:
  NCCLCHECK(ncclStrongStreamWaitStream(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, &comm->sharedRes->hostStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream));
  return ret;
fail:
  goto exit;
}

extern struct ncclTransport collNetTransport;

// All ranks must participate in collNetSetup call
// We do not NCCLCHECK this call because we would fall back to P2P network in case CollNet setup fails
int ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type) {
  int fail = 1;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nMasters = comm->nNodes;
  int rankInCollNet = -1;
  int isMaster = (rank == masterRank) ? 1 : 0;
  struct {
    int collNetRank;
    ncclConnect connect;
  } sendrecvExchange;

  // check if we can connect to collnet, whose root is the nranks-th rank
  struct ncclPeerInfo *myInfo = comm->peerInfo+rank, *peerInfo = comm->peerInfo+nranks;
  peerInfo->rank = nranks;

  // send master receives connect info from peer recv master
  if (isMaster && type == collNetSend) {
    NCCLCHECK(bootstrapRecv(comm->bootstrap, masterPeer, collNetGraph->id, &sendrecvExchange, sizeof(sendrecvExchange)));
    rankInCollNet = sendrecvExchange.collNetRank;
    TRACE(NCCL_INIT, "CollNet [send] : rank %d collNetRank %d collNetNranks %d received connect from rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }

  // select
  struct ncclChannelPeer* root = channel->peers[nranks];
  // connector index: 0 for recv, 1 for send
  struct ncclConnector* conn = (type == collNetRecv) ? root->recv+type : root->send+type;
  struct ncclTransportComm* transportComm = (type == collNetRecv) ? &(collNetTransport.recv) : &(collNetTransport.send);
  conn->transportComm = transportComm;
  // setup
  struct ncclConnect myConnect;
  if (isMaster) {
    NCCLCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, collNetGraphChannelId, type));
  }
  // prepare connect handles
  ncclResult_t res;
  struct {
    int isMaster;
    ncclConnect connect;
  } *allConnects = NULL;
  ncclConnect *masterConnects = NULL;
  NCCLCHECK(ncclCalloc(&masterConnects, nMasters));
  if (type == collNetRecv) {  // recv side: AllGather
    // all ranks must participate
    NCCLCHECK(ncclCalloc(&allConnects, nranks));
    allConnects[rank].isMaster = isMaster;
    memcpy(&(allConnects[rank].connect), &myConnect, sizeof(struct ncclConnect));
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), res, cleanup);
    // consolidate
    int c = 0;
    for (int r = 0; r < nranks; r++) {
      if (allConnects[r].isMaster) {
        memcpy(masterConnects+c, &(allConnects[r].connect), sizeof(struct ncclConnect));
        if (r == rank) rankInCollNet = c;
        c++;
      }
    }
  } else { // send side : copy in connect info received from peer recv master
    if (isMaster) memcpy(masterConnects+rankInCollNet, &(sendrecvExchange.connect), sizeof(struct ncclConnect));
  }
  // connect
  if (isMaster) {
    NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, rankInCollNet, conn), res, cleanup);
    struct ncclDevChannelPeer* devRoot;
    CUDACHECKGOTO(cudaMemcpy(&devRoot, channel->devPeers + nranks, sizeof(struct ncclDevChannelPeer*), cudaMemcpyDeviceToHost), res, cleanup);
    struct ncclConnInfo* devConnInfo = (type == collNetRecv) ? devRoot->recv + type : devRoot->send + type;
    CUDACHECKGOTO(cudaMemcpy(devConnInfo, &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice), res, cleanup);
  }
  // recv side sends connect info to send side
  if (isMaster && type == collNetRecv) {
    sendrecvExchange.collNetRank = rankInCollNet;
    memcpy(&sendrecvExchange.connect, masterConnects+rankInCollNet, sizeof(struct ncclConnect));
    NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, masterPeer, collNetGraph->id, &sendrecvExchange, sizeof(sendrecvExchange)), res, cleanup);
    TRACE(NCCL_INIT, "CollNet [recv] : rank %d collNetRank %d collNetNranks %d sent connect to rank %d", rank, rankInCollNet, nMasters, masterPeer);
  }
  fail = 0;
cleanup:
  if (allConnects != NULL) free(allConnects);
  if (masterConnects != NULL) free(masterConnects);
  return fail;
}

ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail) {
  // AllGather collNet setup results
  int allGatherFailures[NCCL_MAX_LOCAL_RANKS] = {0};
  allGatherFailures[comm->localRank] = collNetSetupFail;
  NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, allGatherFailures, sizeof(int)));
  for (int i=0; i<comm->localRanks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }
  if (collNetSetupFail) {
    if (comm->localRank == 0) WARN("Cannot initialize CollNet, using point-to-point network instead");
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm) {
  // Free collNet resources
  for (int r=0; r<comm->nChannels; r++) {
    struct ncclChannel* channel = comm->channels+r;
    struct ncclChannelPeer* peer = channel->peers[comm->nRanks];
    if (peer) {
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        for (int b=0; b<NCCL_MAX_CONNS*TRANSPORT_NUM; b++) {
          struct ncclConnector* send = peer->send + b;
          if (send->transportResources && send->transportComm) NCCLCHECK(send->transportComm->free(send));
          send->transportResources = NULL; // avoid double free
        }
        for (int b=0; b<NCCL_MAX_CONNS*TRANSPORT_NUM; b++) {
          struct ncclConnector* recv = peer->recv + b;
          if (recv->transportResources && recv->transportComm) NCCLCHECK(recv->transportComm->free(recv));
          recv->transportResources = NULL; // avoid double free
        }
      }
    }
  }
  return ncclSuccess;
}
