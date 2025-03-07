#include "graph/topo.h"
#include "proxy.h"

static ncclResult_t ncclChannelComputeBaseForTuner(struct ncclComm* comm, int peer, int coll, int*channelBase) {
  int p2pGroupSize = NCCL_MAX_WORK_ELEMENTS_P2P/2;
  int peerNode = comm->rankToNode[peer];
  int peerIndex = comm->rankToLocalRank[peer];
  int nsteps = comm->maxLocalRanks;
  int rankIndex = comm->rankToLocalRank[comm->rank];
  int step, delta;
  if (coll == ncclFuncSend) {
    step = (nsteps + peerIndex - rankIndex)%nsteps;
    delta = (comm->nNodes + peerNode - comm->node) % comm->nNodes;
  } else if (coll == ncclFuncRecv) {
    step = (nsteps + rankIndex - peerIndex)%nsteps;
    delta = (comm->nNodes + comm->node - peerNode) % comm->nNodes;
  } else {
    return ncclInternalError;
  }
  // The send of the current GPU is the recv of the opposite GPU, so send and recv need to be on the same channel of different GPUs.
  // In order to use more threadblocks, we try to map different peers to different channels
  // However, in order to make peers of the same node adjacent to each other and facilitate the merged communication of the net, the simplest method is the native version: fall to different worker nodes of the same channel
  // TODO(anonymous) or the value that falls into p2pSlotsForTuner[base] needs to be adjacent
  if (comm->nNodes > 1) {
    int maxSlots = (*comm->tunerEnvs)["tuner_p2pnChannels"]*NCCL_MAX_WORK_ELEMENTS_P2P/2;
    int nodeBaseSlot = comm->p2pSlotsForTuner[(delta+step/p2pGroupSize)%maxSlots];
    int slotPerSendRecv = maxSlots/comm->nNodes/nsteps;
    int baseSlot = (nodeBaseSlot+step*slotPerSendRecv) % maxSlots;
    int baseChannel = baseSlot/(NCCL_MAX_WORK_ELEMENTS_P2P/2); // from slot to channel
    *channelBase = baseChannel;
  } else {
    *channelBase = step;
  }
  return ncclSuccess;
}
static ncclResult_t ncclChannelComputeFromBaseForTuner(struct ncclComm* comm, int base, int channelInc, int*channelId, int peer, int coll) {
  if (comm->nNodes > 1) {
    int baseChannel = base;
    *channelId = (baseChannel+channelInc) % (*comm->tunerEnvs)["tuner_p2pnChannels"];
    TRACE(NCCL_COLL, "native=0, peer=%d %s baseChannel=%d channelInc=%d -> channel=%d", peer, (coll == ncclFuncSend ? "send" : "recv"), baseChannel, channelInc, *channelId);
  } else {
    int baseChannel = comm->p2pChannelsForTuner[base%(*comm->tunerEnvs)["tuner_p2pnChannels"]];
    *channelId = (baseChannel+channelInc) % (*comm->tunerEnvs)["tuner_p2pnChannels"];
    TRACE(NCCL_COLL, "native=0, peer=%d %s step=%d baseChannel=%d channelInc=%d -> channel=%d", peer, (coll == ncclFuncSend ? "send" : "recv"), base, baseChannel, channelInc, *channelId);
  }
  return ncclSuccess;
}
static ncclResult_t ncclChannelComputeForTuner(struct ncclComm* comm, int peer, int channelInc, int coll, int*channelId) {
  int base;
  NCCLCHECK(ncclChannelComputeBaseForTuner(comm, peer, coll, &base));
  NCCLCHECK(ncclChannelComputeFromBaseForTuner(comm, base, channelInc, channelId, peer, coll));
  return ncclSuccess;
}

static ncclResult_t computeCollForTuner(struct ncclInfo* info /* input */, int* workFuncIndex, struct ncclWorkElem* work, struct ncclProxyOp* proxyOp /* output */) {
  info->algorithm = info->comm->tasks.candidate.algorithm;
  info->protocol = info->comm->tasks.candidate.protocol;
  info->nChannels = info->comm->tasks.candidate.nChannels;
  info->nThreads = info->comm->tasks.candidate.nThreads;
  info->comm->tasks.candidate.nThreadsTotal = info->comm->tasks.candidate.nThreads;
  info->chunkSize = info->comm->tasks.candidate.wireChunksize;
  proxyOp->isCopyEngineNotSmCopy = info->comm->tasks.candidate.isCopyEngineNotSmCopy;
  proxyOp->p2pLevel = info->comm->tasks.candidate.p2pLevel;

  NCCLCHECK(getPatternInfo(info));
  // get info->nstepsPerLoop and info->nchunksPerLoop
  NCCLCHECK(getLoopInfo(info));

  work->sendbuff = info->sendbuff;
  work->recvbuff = info->recvbuff;
  work->root = info->root;
  work->count = info->count;
  work->redOpArg = info->opFull.scalarArg;
  work->redOpArgIsPtr = info->opFull.scalarArgIsPtr;
  work->native = 0;

  if (info->comm->nRanks == 1) {
    // one-rank reduce index
    *workFuncIndex = 1 + int(info->datatype);
    return ncclSuccess;
  }

  *workFuncIndex = FUNC_INDEX(info->coll, info->opFull.op, info->datatype, info->algorithm, info->protocol);

  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int slicePerChunk  = chunkSteps/sliceSteps;
  // Optimize pipeline
  if (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) {
    if (info->coll == ncclFuncAllReduce || 
        info->coll == ncclFuncAllGather || 
        info->coll == ncclFuncReduceScatter) {
      // minSliceSteps: 1,2,4,4
      int minSliceSteps = DIVUP(info->chunkSize/*sliceSize*/, info->comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS);
      if ((*info->comm->tunerEnvs)["tuner_chunkPipeline_disable"] > 0) {
        if (minSliceSteps == 1) minSliceSteps = 2; // TODO: make it tunable
      }
      if (minSliceSteps == 3) minSliceSteps = 4;
      else if (minSliceSteps > NCCL_STEPS/2) {
        WARN("chunkSize is too large.");
        return ncclInternalError;
      }
      sliceSteps = minSliceSteps;
      if (sliceSteps < minSliceSteps || sliceSteps > NCCL_STEPS/2) {
        return ncclInternalError;
      }
      // slicePerChunk: 4,2,1,1
      slicePerChunk = chunkSteps/sliceSteps;
    }
  }
  work->userSliceSteps = sliceSteps;

  int chunkEffectiveSize = info->chunkSize;
  if (info->protocol == NCCL_PROTO_LL) chunkEffectiveSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkEffectiveSize = (info->chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;

  if (UINT8_MAX /*255*/ < info->nChannels) {
    return ncclInternalError;
  }
  work->nChannels = info->nChannels;
  work->useAllReduceSimpleTreeUpDown = 0;
  work->useAllReduceLL128LLTreeUpDown = 0;
  // Extra warp for sync
  if (info->coll == ncclFuncAllReduce) {
    if (info->algorithm == NCCL_ALGO_TREE) {
      if (info->protocol == NCCL_PROTO_LL128 || info->protocol == NCCL_PROTO_LL) {
        if ((*info->comm->tunerEnvs)["tuner_treeupdown_allreduce_ll128_ll"] == 1) {
          work->useAllReduceLL128LLTreeUpDown = 1;
        } else {
          work->useAllReduceLL128LLTreeUpDown = 0;
        }
      }
    }
  }
  if (info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->algorithm == NCCL_ALGO_TREE) {
      if (info->coll == ncclFuncAllReduce) {
        if ((*info->comm->tunerEnvs)["tuner_treeupdown_allreduce_simple"] == 1) {
          work->useAllReduceSimpleTreeUpDown = 1;
          // tree updown
          info->nThreads += WARP_SIZE;
          info->comm->tasks.candidate.nThreadsTotal += WARP_SIZE;
        } else {
          work->useAllReduceSimpleTreeUpDown = 0;
          // tree split
          info->nThreads += 4*WARP_SIZE;
          info->comm->tasks.candidate.nThreadsTotal += 4*WARP_SIZE;
        }
      } else {
        return ncclInternalError;
      }
    }
    if (info->algorithm == NCCL_ALGO_RING) {
      info->nThreads += WARP_SIZE;
      info->comm->tasks.candidate.nThreadsTotal += WARP_SIZE;
    }
  }
  work->nWarps = info->nThreads / WARP_SIZE;
  uint64_t effectiveChunkCount = chunkEffectiveSize / ncclTypeSize(info->datatype);
  if ((1ULL<<EFFECTIVECHUNKSIZE_BITS)-1 < effectiveChunkCount) {
    return ncclInternalError;
  }
  work->effectiveChunkSize = effectiveChunkCount; // effective chunksize for gpu (for simple, is slice size)

  proxyOp->dtype = info->datatype;
  proxyOp->redOp = info->opFull.op==ncclDevPreMulSum || info->opFull.op==ncclDevSumPostDiv ? ncclSum : // Network sees avg as sum
                     info->op;
  proxyOp->root = info->root;

  // Compute nSteps for proxies
  int nLoops = (int)(DIVUP(info->nBytes/*total*/, (((size_t)(info->nChannels))*info->nchunksPerLoop*chunkEffectiveSize*slicePerChunk)));
  proxyOp->nsteps = info->nstepsPerLoop * nLoops * chunkSteps;
  proxyOp->sliceSteps = sliceSteps;
  proxyOp->chunkSteps = chunkSteps;
  proxyOp->chunkSize = slicePerChunk*info->chunkSize; // wire chunksize for net
  proxyOp->protocol = info->protocol;
  proxyOp->pattern = info->pattern;
  int stepSize   = info->comm->buffSizes[info->protocol]/NCCL_STEPS;

  proxyOp->nbytes = stepSize*proxyOp->sliceSteps;
  // TODO: receive actual size?
  // proxyOp->nbytes = slicePerChunk*info->chunkSize;

  INFO(NCCL_COLL,"reget: coll=%d, nbytes=%d, count per rank=%d nbytes, nRanks=%d => algorithm=%d, useAllReduceSimpleTreeUpDown=%d, useAllReduceLL128LLTreeUpDown=%d, protocol=%d, [gpu] nc parallel=%d, nrank parallel=%d; nLoops=%d; primcall per loop=%d+1, chunkSteps=%d, sliceSteps=%d; effective SliceSize=%dT, T%d=%d bytes, no lastChunkSize, nt=%d, [proxy] opCount=%lx, nsteps=%d, chunkSteps=%d, wirechunkSize=%d bytes, nbytes=%d",
                info->coll, info->nBytes, info->count*ncclTypeSize(info->datatype), info->comm->nRanks,
                info->algorithm, work->useAllReduceSimpleTreeUpDown, work->useAllReduceLL128LLTreeUpDown, info->protocol, 
                info->nChannels, info->nchunksPerLoop, nLoops, info->nstepsPerLoop, chunkSteps, sliceSteps, work->effectiveChunkSize, info->datatype, ncclTypeSize(info->datatype), info->nThreads,
                proxyOp->opCount, proxyOp->nsteps, proxyOp->chunkSteps, proxyOp->chunkSize, proxyOp->nbytes);
  return ncclSuccess;
}

static ncclResult_t scheduleCollTasksToPlanForTuner(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int* nWorkBudget
  ) {
  struct ncclTasks* tasks = &comm->tasks;
  while (tasks->nTasksColl != 0) {
    struct ncclTaskColl* head = ncclIntruQueueHead(&tasks->collQueue);
    struct ncclInfo info = {};
    info.comm = comm;
    info.coll = head->func;
    info.sendbuff = head->sendbuff;
    info.recvbuff = head->recvbuff;
    info.count = head->count;
    info.root = head->root;
    info.datatype = head->datatype;
    info.opFull = head->op; // C++ struct assignment
    info.op = (ncclRedOp_t)(int)head->op.op;
    info.chunkSteps = head->chunkSteps;
    // Can be optimized to increase pipeline
    info.sliceSteps = head->sliceSteps;
    NCCLCHECK(ncclInfoSetDerived(&info, comm->nRanks));

    int workFuncIndex;
    struct ncclWorkElem workElem = {};
    struct ncclProxyOp proxyOp = {};
    NCCLCHECK(computeCollForTuner(&info, &workFuncIndex, &workElem, &proxyOp));

    if (*nWorkBudget < info.nChannels) return ncclSuccess; // Ensure room for addCollToPlan()

    bool regBufUsed = false;
    void* regBufSend[NCCL_MAX_LOCAL_RANKS];
    void* regBufRecv[NCCL_MAX_LOCAL_RANKS];
    if (plan->persistent && ncclParamGraphRegister() &&
        info.algorithm == NCCL_ALGO_COLLNET_DIRECT &&   // limited to CollNetDirect for now
        comm->intraHighestTransportType == TRANSPORT_P2P && // only when all ranks can p2p each other
        comm->intraRanks < comm->localRanks) { // only with inter-process & intra-node peers
      NCCLCHECK(registerIntraNodeBuffers(comm, plan, &info, &regBufUsed, regBufSend, regBufRecv));
    }

    NCCLCHECK(addCollToPlan(comm, plan, nWorkBudget, workFuncIndex, &workElem, &proxyOp,
      (*comm->tunerEnvs)["tuner_nChannels"], info.nChannels, info.nBytes, regBufUsed, regBufSend, regBufRecv));
    tasks->nTasksColl -= 1;
    tasks->collBytesTotal -= info.nBytes;
    ncclIntruQueueDequeue(&tasks->collQueue);
    head = ncclIntruQueueHead(&tasks->collQueue);

    plan->threadPerBlock = std::max(plan->threadPerBlock, info.nThreads);
    if (!plan->kernelSpecialized) {
      plan->kernelFn = ncclKerns[workFuncIndex].kernelFn;
      plan->kernelSpecialized = ncclKerns[workFuncIndex].specialized;
    }
  }
  return ncclSuccess;
}

ncclResult_t chooseTransport(struct ncclComm* comm, int channelId, int peer, uint8_t isCopyEngineNotSmCopy, uint8_t p2pLevel, uint8_t* transportIndex) {
  if (peer < 0) return ncclSuccess;
  struct ncclChannel* channel = &comm->channels[channelId];
  *transportIndex = 0; // net or shm will use 0 if not p2pSupport
  bool p2pSupport = (channel->peers[peer]->transportMask & (1 << TRANSPORT_P2P)) != 0;
  bool p2pCeSupport = (channel->peers[peer]->transportMask & (1 << TRANSPORT_P2P_CE)) != 0;
  bool shmSupport = (channel->peers[peer]->transportMask & (1 << TRANSPORT_SHM)) != 0;
  if (p2pSupport && (p2pCeSupport || shmSupport) && TRANSPORT_NUM > 1) {
    if (channel->peers[peer]->p2pLevel <= p2pLevel) {
      *transportIndex = (isCopyEngineNotSmCopy == 1 && p2pCeSupport) ? TRANSPORT_P2P_CE: TRANSPORT_P2P;
      // bool use_multi_transports = false;
      // if (shmSupport && use_multi_transports) {
      //   if (channel->peers[peer]->p2pLevel == PATH_NVL ||
      //       channel->peers[peer]->p2pLevel == PATH_NVB) {
      //     if (channelId%2 == 1) {
      //       *transportIndex = TRANSPORT_SHM;
      //     }
      //   }
      // }
    } else {
      if (!shmSupport) {
        WARN("No shm transport.");
      } else {
        *transportIndex = TRANSPORT_SHM;
      }
    }
  }
  TRACE(NCCL_COLL, "[Tuner] channel=%d peer=%d level=%d p2pSupport=%d p2pCeSupport=%d shmSupport=%d: isCopyEngineNotSmCopy=%d p2pLevel=%d -> transportIndex=%d", channelId, peer, channel->peers[peer]->p2pLevel, p2pSupport, p2pCeSupport, shmSupport, isCopyEngineNotSmCopy, p2pLevel, *transportIndex);
  return ncclSuccess;
}

static ncclResult_t saveCollTransports(struct ncclComm* comm, struct ncclProxyOp * proxyOp, struct ncclWorkElem * workElem, int channelId, int isCopyEngineNotSmCopy, int p2pLevel) {
  struct ncclChannel* channel = &comm->channels[channelId];
  switch (proxyOp->pattern) {
  case ncclPatternRing:
  case ncclPatternRingTwice:
  case ncclPatternPipelineFrom:
  case ncclPatternPipelineTo: {
      struct ncclRing* ring = &channel->ring;
      int recvIndex = 0;
      int sendIndex = 0;
      if (ring->prev >= 0 && NeedProxy(0/*proxyRecv*/, proxyOp->pattern, proxyOp->root, ring, comm->nRanks)) {
        chooseTransport(comm, channelId, ring->prev, isCopyEngineNotSmCopy, p2pLevel, &workElem->transportIndex[recvIndex++]);
      }
      if (ring->next >= 0 && NeedProxy(1/*proxySend*/, proxyOp->pattern, proxyOp->root, ring, comm->nRanks)) {
        chooseTransport(comm, channelId, ring->next, isCopyEngineNotSmCopy, p2pLevel, &workElem->transportIndex[1+sendIndex++]);
      }
    } break;
  case ncclPatternTreeUp:
  case ncclPatternTreeDown:
  case ncclPatternTreeUpDown: {
      struct ncclTree* tree = &channel->tree;
      bool is_runTreeSplit = false;
      if (comm->tasks.workload.collType == ncclFuncAllReduce && 
          comm->tasks.candidate.algorithm == NCCL_ALGO_TREE) {
        if (comm->tasks.candidate.native == 1) {
          if (comm->tasks.candidate.protocol == NCCL_PROTO_LL128 || comm->tasks.candidate.protocol == NCCL_PROTO_LL) {
            is_runTreeSplit = (*comm->tunerEnvs)["tuner_treeupdown_allreduce_ll128_ll_native"] == 0;
          } else if (comm->tasks.candidate.protocol == NCCL_PROTO_SIMPLE) {
            is_runTreeSplit = (*comm->tunerEnvs)["tuner_treeupdown_allreduce_simple_native"] == 0;
          } else {
            return ncclInternalError;
          }
        } else {
          if (comm->tasks.candidate.protocol == NCCL_PROTO_LL128 || comm->tasks.candidate.protocol == NCCL_PROTO_LL) {
            is_runTreeSplit = (*comm->tunerEnvs)["tuner_treeupdown_allreduce_ll128_ll"] == 0;
          } else if (comm->tasks.candidate.protocol == NCCL_PROTO_SIMPLE) {
            is_runTreeSplit = (*comm->tunerEnvs)["tuner_treeupdown_allreduce_simple"] == 0;
          } else {
            return ncclInternalError;
          }
        }
      }
      if (is_runTreeSplit && tree->up == -1) { // runTreeSplit
        // for tree->up != -1, the rank with nthreadsSplit will be like runTreeUpDown.
        int recvIndex = 0;
        int sendIndex = 0;
        // recv
        for (int i=0; i<NCCL_MAX_TREE_ARITY_TOP; i++) {
          if (tree->down[i] >= 0) {
            chooseTransport(comm, channelId, tree->down[i], isCopyEngineNotSmCopy, p2pLevel, &workElem->transportIndex[recvIndex++]);
          }
        }
        // send
        for (int i=0; i< NCCL_MAX_TREE_ARITY_TOP; i++) {
          if (tree->down[i] >= 0) {
            chooseTransport(comm, channelId, tree->down[i], isCopyEngineNotSmCopy, p2pLevel, &workElem->transportIndex[NCCL_MAX_TREE_ARITY_TOP+sendIndex++]);
          }
        }
      } else { // runTreeUpDown
        // reduce
        if (proxyOp->pattern != ncclPatternTreeDown) { // Tree up
          int recvIndex = 0;
          int sendIndex = 0;
          // recv
          for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) {
            if (tree->down[i] >= 0) {
              chooseTransport(comm, channelId, tree->down[i], isCopyEngineNotSmCopy, p2pLevel, &workElem->transportIndex[recvIndex++]);
            }
          }
          // send
          if (tree->up >= 0) {
            chooseTransport(comm, channelId, tree->up, isCopyEngineNotSmCopy, p2pLevel, &workElem->transportIndex[NCCL_MAX_TREE_ARITY+sendIndex++]);
          }
        }
        // broadcast
        if (proxyOp->pattern != ncclPatternTreeUp) { // Tree down
          int recvIndex = 0;
          int sendIndex = 0;
          // recv
          if (tree->up >= 0) {
            chooseTransport(comm, channelId, tree->up, isCopyEngineNotSmCopy, p2pLevel, &workElem->transportIndex[NCCL_MAX_TREE_ARITY+1+recvIndex++]);
          }
          // send
          for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) {
            if (tree->down[i] >= 0) {
              chooseTransport(comm, channelId, tree->down[i], isCopyEngineNotSmCopy, p2pLevel, &workElem->transportIndex[NCCL_MAX_TREE_ARITY+1+1+sendIndex++]);
            }
          }
        }
      }
    } break;
  case ncclPatternCollnetChain: {
    } break;
  case ncclPatternCollnetDirect: {
    } break;
  case ncclPatternNvls: {
    } break;
  case ncclPatternNvlsTree: {
    } break;
  case ncclPatternSend:
  case ncclPatternRecv: {
      int index = 0;
      if (proxyOp->root == comm->rank) return ncclSuccess;
      if (proxyOp->root >= 0) {
        chooseTransport(comm, channelId, proxyOp->root, isCopyEngineNotSmCopy, p2pLevel, &workElem->transportIndex[index++]);
      }
    } break;
  }
  return ncclSuccess;
}

static ncclResult_t addP2pToPlanForTuner(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int* nWorkBudget,
    bool isSendNotRecv, int peer, int chunk, void *addr, size_t bytes, bool fuseOk
  ) {
  struct ncclInfo info = {
    isSendNotRecv ? ncclFuncSend : ncclFuncRecv,
    isSendNotRecv ? "Send" : "Recv",
    nullptr, addr, bytes, ncclInt8, ncclSum, peer, comm, (cudaStream_t)0,
    /*Args*/1, 1
  };

  int channelId;
  // Call ncclChannelCompute to calculate the specific channelId to be used
  NCCLCHECK(ncclChannelComputeForTuner(comm, peer, chunk, info.coll, &channelId));
  info.channelId = channelId;

  // set protocol
  info.protocol = comm->tasks.candidate.protocol;

  // info->chunkSize is the amount of data to be processed per iteration. Avoid it from exceeding the bounds.
  // SIMPLE bound = 128KB;  LL128 bound = 128KB; LL bound = 80 KB
  int wireChunksize = comm->tasks.candidate.wireChunksize;
  info.chunkSize = wireChunksize;

  // elem.effectiveChunkSize and proxyOp.chunkSize are the processing size of each iteration.
  // Avoid exceeding the boundaries.
  struct ncclProxyOp proxyOp = {};
  proxyOp.channelId = info.channelId;
  proxyOp.sliceSteps = 1;
  proxyOp.chunkSteps = 1;
  proxyOp.dtype = info.datatype;
  proxyOp.protocol = info.protocol;
  proxyOp.root = info.root;
  if (info.coll == ncclFuncSend) {
    proxyOp.pattern = ncclPatternSend;
  } else if (info.coll == ncclFuncRecv) {
    proxyOp.pattern = ncclPatternRecv;
  } else {
    WARN("P2p operation is neither send or recv");
    return ncclInternalError;
  }
  proxyOp.chunkSize = info.chunkSize;

  int chunkEffectiveSize = wireChunksize;
  if (info.protocol == NCCL_PROTO_LL) chunkEffectiveSize /= 2;
  if (info.protocol == NCCL_PROTO_LL128) chunkEffectiveSize = (chunkEffectiveSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  proxyOp.nsteps = DIVUP(info.count, chunkEffectiveSize);
  if (proxyOp.nsteps == 0) proxyOp.nsteps = 1;

  int netRecvStepSize = info.comm->buffSizes[proxyOp.protocol]/NCCL_STEPS;
  if (proxyOp.protocol == NCCL_PROTO_SIMPLE) netRecvStepSize = SIMPLE_P2PCHUNKSIZE_UPPER_BOUND;
  if (proxyOp.protocol == NCCL_PROTO_LL128) netRecvStepSize = LL128_P2PCHUNKSIZE_UPPER_BOUND;
  if (proxyOp.protocol == NCCL_PROTO_LL) netRecvStepSize = LL_P2PCHUNKSIZE_UPPER_BOUND;
  proxyOp.nbytes = netRecvStepSize; // TODO: receive actual size?

  struct ncclWorkElemP2p elem = {0};
  elem.native = 0;
  elem.proto = info.protocol;
  elem.peer = peer;
  int nThreadsPerGroup = comm->tasks.candidate.nThreads;
  // The current elem's nWarps
  // But it will be updated by finishWorkP2p later
  elem.nWarps = nThreadsPerGroup/WARP_SIZE;
  elem.p2pType = isSendNotRecv ? ncclWorkP2pTypeSend : ncclWorkP2pTypeRecv;
  elem.buffLo32 = uint32_t(reinterpret_cast<uintptr_t>(addr));
  elem.buffHi32 = reinterpret_cast<uintptr_t>(addr)>>32;
  // The total number of tasks for the current workelem
  elem.countLo32 = uint32_t(bytes);
  elem.countHi32 = bytes>>32;
  elem.effectiveChunkSize = chunkEffectiveSize;

  proxyOp.isCopyEngineNotSmCopy = comm->tasks.candidate.isCopyEngineNotSmCopy;
  proxyOp.p2pLevel = comm->tasks.candidate.p2pLevel;

  uint8_t transportIndex;
  chooseTransport(comm, channelId, peer, proxyOp.isCopyEngineNotSmCopy, proxyOp.p2pLevel, &transportIndex);
  elem.transportIndex = transportIndex;

  *nWorkBudget += plan->channels[channelId].nWork;
  appendWorkElemP2p(comm, plan, channelId, &elem, fuseOk, false);
  *nWorkBudget -= plan->channels[channelId].nWork;

  // Calculate the opCount after appendWorkElemP2p since it will always return
  // with channel->nWork equal to one plus the work index this p2p settled in.
  proxyOp.opCount = uint64_t(plan->channels[channelId].nWork)<<1 | 1;
  NCCLCHECK(addProxyOpIfNeeded(comm, plan, &proxyOp));
  return ncclSuccess;
}

static ncclResult_t scheduleP2pTasksToPlanForTuner(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int* nWorkBudget
  ) {
  // TODO(anonymous): to support all2allv and sendrecv

  struct ncclTasks* tasks = &comm->tasks;
  struct ncclTasks::Peer* peers = tasks->peers;
  int const *sendOrder = tasks->p2pSendOrder;
  int const *recvOrder = tasks->p2pRecvOrder;

  if (!plan->kernelSpecialized) {
    plan->kernelFn = ncclKerns[FUNC_INDEX_P2P].kernelFn;
    plan->kernelSpecialized = ncclKerns[FUNC_INDEX_P2P].specialized;
  }

  // nc: Compute how much to split operations
  int p2pnChannelsPerPeerSendOrRecv = comm->tasks.candidate.nChannels;

  // nt
  comm->tasks.candidate.nThreadsTotal = 0;

  while (tasks->nTasksP2p != 0) {
    for (int i=0; i < tasks->p2pOrderSteps; i++) {
      int sendPeer = sendOrder[i];
      int recvPeer = recvOrder[i];
      struct ncclTaskP2p* send = sendPeer != -1 ? ncclIntruQueueHead(&peers[sendPeer].sendQueue) : NULL;
      struct ncclTaskP2p* recv = recvPeer != -1 ? ncclIntruQueueHead(&peers[recvPeer].recvQueue) : NULL;
      if (sendPeer == comm->rank) {
        if (recvPeer != comm->rank) {
          WARN("Sendrecv plan not aligned for self");
          return ncclInternalError;
        }
        if (send && recv == nullptr) {
          WARN("Trying to send to self without a matching recv");
          return ncclInvalidUsage;
        }
        if (send == nullptr && recv) {
          WARN("Trying to recv to self without a matching send");
          return ncclInvalidUsage;
        }
      }
      if (send != nullptr || recv != nullptr) {
        char* recvPtr = recv ? (char*)recv->buff : nullptr;
        char* sendPtr = send ? (char*)send->buff : nullptr;
        ssize_t recvBytes = recv ? recv->bytes : 0;
        ssize_t sendBytes = send ? send->bytes : 0;
        ssize_t recvChunkBytesMax = divUp(recvBytes, p2pnChannelsPerPeerSendOrRecv);
        ssize_t sendChunkBytesMax = divUp(sendBytes, p2pnChannelsPerPeerSendOrRecv);
        // Zero size send/recv are syncs, encode here with -1.
        recvBytes = recv && recvBytes == 0 ? -1 : recvBytes;
        sendBytes = send && sendBytes == 0 ? -1 : sendBytes;
        // Advance to current chunk. Syncs will always have chunk=0 so no effect on the -1.
        if (recv) recvPtr   += recv->chunk*recvChunkBytesMax;
        if (recv) recvBytes -= recv->chunk*recvChunkBytesMax;
        if (send) sendPtr   += send->chunk*sendChunkBytesMax;
        if (send) sendBytes -= send->chunk*sendChunkBytesMax;

        do {
          ssize_t recvChunkBytes = std::min(recvBytes, recvChunkBytesMax); // -1 preserved
          ssize_t sendChunkBytes = std::min(sendBytes, sendChunkBytesMax);
          if (recvChunkBytes != 0) {
            if (recvChunkBytes == -1) recvChunkBytes = 0;
            if (*nWorkBudget < 1) return ncclSuccess; // ensure room in budget
            TRACE(NCCL_COLL, "[Tuner] schedule p2ptask to plan: recv Peer=%d recv->chunk=%d, recvPtr=%p, recvChunkBytes=%d", recvPeer, recv->chunk, recvPtr, recvChunkBytes);
            NCCLCHECK(addP2pToPlanForTuner(comm, plan, nWorkBudget, /*isSendNotRecv=*/false, recvPeer, recv->chunk, recvPtr, recvChunkBytes, true));
            recvPtr += recvChunkBytes;
            recvBytes -= recvChunkBytes;
            recv->chunk += 1;
            if (recvBytes <= 0) {
              recvBytes = 0; // in case still -1
              ncclIntruQueueDequeue(&peers[recvPeer].recvQueue);
              tasks->nTasksP2p -= 1;
            }
          }
          if (sendChunkBytes != 0) {
            if (sendChunkBytes == -1) sendChunkBytes = 0;
            if (*nWorkBudget < 1) return ncclSuccess; // ensure room in budget
            TRACE(NCCL_COLL, "[Tuner] schedule p2ptask to plan: send Peer=%d send->chunk=%d, sendPtr=%p, sendChunkBytes=%d", sendPeer, send->chunk, sendPtr, sendChunkBytes);
            NCCLCHECK(addP2pToPlanForTuner(comm, plan, nWorkBudget, /*isSendNotRecv=*/true, sendPeer, send->chunk, sendPtr, sendChunkBytes, true));
            sendPtr += sendChunkBytes;
            sendBytes -= sendChunkBytes;
            send->chunk += 1;
            if (sendBytes <= 0) {
              sendBytes = 0; // in case still -1
              ncclIntruQueueDequeue(&peers[sendPeer].sendQueue);
              tasks->nTasksP2p -= 1;
            }
          }
        } while (sendBytes != 0 || recvBytes != 0);
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t connectPeers(struct ncclComm* comm, bool native, int p2pnChannelsPerPeerSendOrRecv) {
  struct ncclTasks* tasks = &comm->tasks;
  struct ncclTasks::Peer* peers = NULL;
  TRACE(NCCL_COLL, "[Tuner] connect peers for native=%d; p2pnChannelsPerPeerSendOrRecv=%d", native, p2pnChannelsPerPeerSendOrRecv);
  peers = comm->tasks.peers;
  int const *sendOrder = tasks->p2pSendOrder;
  int const *recvOrder = tasks->p2pRecvOrder;

  auto connect = [&](int peerRank, bool isSendNotRecv) -> ncclResult_t {
    int channelBaseId;
    if (native) {
      NCCLCHECK(ncclChannelComputeBase(comm, peerRank, isSendNotRecv ? ncclFuncSend : ncclFuncRecv, &channelBaseId));
    } else {
      NCCLCHECK(ncclChannelComputeBaseForTuner(comm, peerRank, isSendNotRecv ? ncclFuncSend : ncclFuncRecv, &channelBaseId));
    }
    if (!(isSendNotRecv ? peers[peerRank].sendSeen : peers[peerRank].recvSeen)) {
      (isSendNotRecv ? peers[peerRank].sendSeen : peers[peerRank].recvSeen) = true;
      for (int c=0; c < p2pnChannelsPerPeerSendOrRecv; c++) {
        int channelId;
        if (native) {
          NCCLCHECK(ncclChannelComputeFromBase(comm, channelBaseId, c, &channelId));
        } else {
          NCCLCHECK(ncclChannelComputeFromBaseForTuner(comm, channelBaseId, c, &channelId, peerRank, (isSendNotRecv ? ncclFuncSend : ncclFuncRecv)));
        }
        if (isSendNotRecv) {
          if (comm->channels[channelId].peers[peerRank]->send[1].connected == 0) { // P2P uses only 1 connector
            comm->connectSend[peerRank][channelId] = true;
            TRACE(NCCL_COLL, "[Tuner] channel=%d send peer=%d connIndex=1 need to connect.", channelId, peerRank);
          } else {
            TRACE(NCCL_COLL, "[Tuner] channel=%d send peer=%d connIndex=1 connected.", channelId, peerRank);
          }
        } else {
          if (comm->channels[channelId].peers[peerRank]->recv[1].connected == 0) { // P2P uses only 1 connector
            comm->connectRecv[peerRank][channelId] = true;
            TRACE(NCCL_COLL, "[Tuner] channel=%d recv peer=%d connIndex=1 need to connect.", channelId, peerRank);
          } else {
            TRACE(NCCL_COLL, "[Tuner] channel=%d recv peer=%d connIndex=1 connected.", channelId, peerRank);
          }
        }
      }
    }
    return ncclSuccess;
  };

  if (!native) {
    // To honor "conn->buffs[NCCL_PROTO_LL] != nullptr", we need connect peers when tasks appended.
    // To connect peers again, we need clear the status here.
    for (int i=0; i < tasks->p2pOrderSteps; i++) {
      int sendPeer = sendOrder[i];
      int recvPeer = recvOrder[i];
      struct ncclTaskP2p* send = sendPeer != -1 ? ncclIntruQueueHead(&peers[sendPeer].sendQueue) : NULL;
      struct ncclTaskP2p* recv = recvPeer != -1 ? ncclIntruQueueHead(&peers[recvPeer].recvQueue) : NULL;
      if (send != nullptr) {
        int peerRank = send->peer;
        peers[peerRank].sendSeen = false;
      }
      if (recv != nullptr) {
        int peerRank = recv->peer;
        peers[peerRank].recvSeen = false;
      }
    }
  }
  for (int i=0; i < tasks->p2pOrderSteps; i++) {
    int sendPeer = sendOrder[i];
    int recvPeer = recvOrder[i];
    TRACE(NCCL_COLL,"[Tuner] Process sendPeer=%d, recvPeer=%d by p2pOrder", sendPeer, recvPeer);
    struct ncclTaskP2p* send = sendPeer != -1 ? ncclIntruQueueHead(&peers[sendPeer].sendQueue) : NULL;
    struct ncclTaskP2p* recv = recvPeer != -1 ? ncclIntruQueueHead(&peers[recvPeer].recvQueue) : NULL;
    if (send != nullptr) {
      int peerRank = send->peer;
      connect(peerRank, /*isSendNotRecv=*/true);
    }
    if (recv != nullptr) {
      int peerRank = recv->peer;
      connect(peerRank, /*isSendNotRecv=*/false);
    }
  }
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
  return ncclSuccess;
}

ncclResult_t cleanUpPlans(struct ncclComm* comm) {
  // modified from groupCleanup()
  while (!ncclIntruQueueEmpty(&comm->planQueue)) {
    struct ncclKernelPlan* plan = ncclIntruQueueDequeue(&comm->planQueue);
    // Persistent plans will be reclaimed via the callbackQueue when the
    // graph drops its UserObject reference.
    if (!plan->persistent) {
      for (int c = 0; c < TUNER_MAXCHANNELS; c++) {
        while (!ncclIntruQueueEmpty(&plan->channels[c].proxyOpQueue)) {
          struct ncclProxyOp* pxop = ncclIntruQueueDequeue(&plan->channels[c].proxyOpQueue);
          ncclMemoryPoolFree(&comm->memPool_ncclProxyOp, pxop);
        }
      }
      ncclMemoryPoolFree(&comm->memPool_ncclKernelPlan, plan);
    }
  }
  return ncclSuccess;
}

ncclResult_t cleanUpTasks(struct ncclComm* comm, bool native) {
  // modified from groupCleanup()
  // Here we just reset the pointer in the queue
  // and the memory release is handled by memScope
  if (native) {
    for (int i = 0; i < comm->nRanks; i++) {
      comm->tasks.peers[i].sendSeen = false;
      comm->tasks.peers[i].recvSeen = false;
    }
    // Reset comm->tasks to empty.
    comm->tasks.nTasksColl = 0;
    comm->tasks.nTasksP2p = 0;
    ncclIntruQueueConstruct(&comm->tasks.collQueue);
    comm->tasks.collBytesTotal = 0;
    for (int i = 0; i < comm->nRanks; i++) {
      ncclIntruQueueConstruct(&comm->tasks.peers[i].sendQueue);
      ncclIntruQueueConstruct(&comm->tasks.peers[i].recvQueue);
    }
  } else {
    for (int i = 0; i < comm->nRanks; i++) {
      comm->tasks.backup.peers[i].sendSeen = false;
      comm->tasks.backup.peers[i].recvSeen = false;
    }
    comm->tasks.backup.nTasksColl = 0;
    comm->tasks.backup.nTasksP2p = 0;
    ncclIntruQueueConstruct(&comm->tasks.backup.collQueue);
    comm->tasks.backup.collBytesTotal = 0;
    for (int i = 0; i < comm->nRanks; i++) {
      ncclIntruQueueConstruct(&comm->tasks.backup.peers[i].sendQueue);
      ncclIntruQueueConstruct(&comm->tasks.backup.peers[i].recvQueue);
    }
  }
  return ncclSuccess;
}

ncclResult_t reloadTasks(struct ncclComm* comm) {
  // modified from groupCleanup()
  // Here we just reset the pointer in the queue
  // and the memory release is handled by memScope
  for (int i = 0; i < comm->nRanks; i++) {
    struct ncclTaskP2p* head;
    // send
    head = ncclIntruQueueHead(&comm->tasks.backup.peers[i].sendQueue);
    while (head != nullptr) {
      struct ncclTaskP2p* p2p = ncclMemoryStackAlloc<struct ncclTaskP2p>(&comm->memScoped);
      memcpy(p2p, head, sizeof(struct ncclTaskP2p));
      ncclIntruQueueEnqueue(
        &comm->tasks.peers[i].sendQueue,
        p2p);
      head = head->next;
      TRACE(NCCL_COLL,"[Tuner] copy send TaskP2p for peer=%d", i);
    }
    // recv
    head = ncclIntruQueueHead(&comm->tasks.backup.peers[i].recvQueue);
    while (head != nullptr) {
      struct ncclTaskP2p* p2p = ncclMemoryStackAlloc<struct ncclTaskP2p>(&comm->memScoped);
      memcpy(p2p, head, sizeof(struct ncclTaskP2p));
      ncclIntruQueueEnqueue(
        &comm->tasks.peers[i].recvQueue,
        p2p);
      head = head->next;
      TRACE(NCCL_COLL,"[Tuner] copy recv TaskP2p for peer=%d", i);
    }
  }

  // coll
  struct ncclTaskColl* head = ncclIntruQueueHead(&comm->tasks.backup.collQueue);
  while (head != nullptr) {
    struct ncclTaskColl* t = ncclMemoryStackAlloc<struct ncclTaskColl>(&comm->memScoped);
    memcpy(t, head, sizeof(struct ncclTaskColl));
    ncclIntruQueueEnqueue(
      &comm->tasks.collQueue,
      t);
    head = head->next;
  }
  comm->tasks.nTasksP2p = comm->tasks.backup.nTasksP2p;
  comm->tasks.nTasksColl = comm->tasks.backup.nTasksColl;
  comm->tasks.collBytesTotal = comm->tasks.backup.collBytesTotal;
  return ncclSuccess;
}

ncclResult_t getPlans(struct ncclComm* comm, int* nPlans, bool native, bool honor) {
  // modified from ncclLaunchPrepare()
  ncclResult_t result = ncclSuccess;
  struct ncclTasks* tasks = &comm->tasks;
  bool persistent = ncclCudaGraphValid(tasks->capturingGraph);
  TRACE(NCCL_COLL,"[Tuner] Process nTasksColl=%d, nTasksP2p=%d, native=%d, honor=%d, persistent=%d", tasks->nTasksColl, tasks->nTasksP2p, native, honor, persistent);
  do {
    struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
    ncclIntruQueueEnqueue(&comm->planQueue, plan);
    *nPlans += 1;
    plan->comm = comm;
    plan->reclaimer.fn = reclaimPlan;
    plan->persistent = persistent;

    // Non-persistent kernels fill up at most half of our fifo per kernel.
    int nWorkBudget = plan->persistent ? INT_MAX : comm->workFifoDepth/2;
    int nWorkBudgetOld = nWorkBudget;

    // Drain coll tasks first. This is essential since we partition tasks based
    // on the work budget and p2p work isn't collective. If we were to drain p2p
    // first, the place where we cut the kernel could vary by rank which would
    // cause the "shortest channel first" channel picker to have divergent results.
    if (native) {
      if (tasks->nTasksColl != 0 && tasks->nTasksP2p == 0) {
        NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, plan, &nWorkBudget, honor), result, failure);
        /******** clean the extra warps for simple info *******/
        // if (honor) {
        //   if (info.protocol == NCCL_PROTO_SIMPLE) {
        //     if (info.algorithm == NCCL_ALGO_RING) info.nThreads -= WARP_SIZE;
        //     if (info.algorithm == NCCL_ALGO_TREE) info.nThreads -= 4*WARP_SIZE;
        //     INFO(NCCL_COLL, "[Tuner] clean the extra warps for simple");
        //   }
        // }
      } else if (tasks->nTasksColl == 0 && tasks->nTasksP2p != 0) {
        NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, plan, &nWorkBudget, honor), result, failure);
      } else {
        WARN("[Tuner] No support grouped calls: nTasksColl=%d, nTasksP2p=%d", tasks->nTasksColl, tasks->nTasksP2p);
        result = ncclInvalidUsage;
        return result;
      }
    } else {
      if (tasks->nTasksColl != 0 && tasks->nTasksP2p == 0) {
        NCCLCHECKGOTO(scheduleCollTasksToPlanForTuner(comm, plan, &nWorkBudget), result, failure);
      } else if (tasks->nTasksColl == 0 && tasks->nTasksP2p != 0) {
        NCCLCHECKGOTO(scheduleP2pTasksToPlanForTuner(comm, plan, &nWorkBudget), result, failure);
      } else {
        WARN("[Tuner] No support grouped calls: nTasksColl=%d, nTasksP2p=%d", tasks->nTasksColl, tasks->nTasksP2p);
        result = ncclInvalidUsage;
        return result;
      }
    }
    if (nWorkBudget == nWorkBudgetOld) {
      // We weren't able to fit any tasks into our budget which means now we're
      // stuck in an infinite loop. We defer this check until here, instead of
      // doing it in comm init, to permit testing with insanely shallow queues
      // for cases where that's expected to still work (e.g. few channels).
      WARN("'NCCL_WORK_FIFO_DEPTH=%d' is too small. Minimum value is %d", comm->workFifoDepth, native ? (2*MAXCHANNELS) : (2*TUNER_MAXCHANNELS));
      result = ncclInvalidUsage;
      return result;
    }
    finishPlan(comm, plan, native);
  } while ((tasks->nTasksColl + tasks->nTasksP2p) != 0);
failure:
  return result;
}

ncclResult_t setWorkload(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  // our tuner support coll and all2all (not all2allv) now.
  comm->tasks.workload.commHash = comm->commHash;
  if (comm->tasks.nTasksP2p) {
    struct ncclTasks::Peer* peers = comm->tasks.peers;
    int const *sendOrder = comm->tasks.p2pSendOrder;
    int const *recvOrder = comm->tasks.p2pRecvOrder;
    for (int i=0; i < comm->tasks.p2pOrderSteps; i++) {
      int sendPeer = sendOrder[i];
      int recvPeer = recvOrder[i];
      const struct ncclTaskP2p* send = sendPeer != -1 ? ncclIntruQueueHead(&peers[sendPeer].sendQueue) : NULL;
      const struct ncclTaskP2p* recv = recvPeer != -1 ? ncclIntruQueueHead(&peers[recvPeer].recvQueue) : NULL;
      if (send != nullptr) {
        comm->tasks.workload.nBytes = send->bytes;
        break;
      }
      if (recv != nullptr) {
        comm->tasks.workload.nBytes = recv->bytes;
        break;
      }
    }
    comm->tasks.workload.collType = ncclFuncAll2All;
  } else {
    struct ncclTaskColl* head = ncclIntruQueueHead(&comm->tasks.collQueue);
    comm->tasks.workload.collType = head->func;
     // count is per rank
     // count*sizeof(T) != nBytes when AllGather and ReduceScatter
    comm->tasks.workload.nBytes = head->count*ncclTypeSize(head->datatype);
  }
failure:
  return result;
}

ncclResult_t cleanTasks(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  NCCLCHECKGOTO(cleanUpTasks(comm, true), result, failure);
  NCCLCHECKGOTO(cleanUpTasks(comm, false), result, failure);
failure:
  return result;
}

ncclResult_t resetWorkloadAndCandidate(struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  comm->tasks.candidate.algorithm = -1;
  comm->tasks.candidate.protocol = -1;
  comm->tasks.candidate.isCopyEngineNotSmCopy = -1;
  comm->tasks.candidate.p2pLevel = -1;
  comm->tasks.candidate.nChannels = -1;
  comm->tasks.candidate.nThreads = -1;
  comm->tasks.candidate.wireChunksize = -1;
  comm->tasks.candidate.iteration = -1;
  comm->tasks.candidate.lastIterEffectiveChunksize = -1;
  comm->tasks.candidate.native = -1;
  comm->tasks.candidate.initialized = false;
  comm->tasks.candidate.nThreadsTotal = -1;
  comm->tasks.workload.commHash = 0;
  // comm->tasks.workload.collType
  comm->tasks.workload.nBytes = 0;
failure:
  return result;
}

ncclResult_t tunePlans(struct ncclComm* comm, int* nPlans) {
  ncclResult_t result = ncclSuccess;
  bool isNew = false;
  NCCLCHECKGOTO(setWorkload(comm), result, failure);
  comm->tuner->isNewWorkload(
    comm->tasks.workload.commHash,
    comm->tasks.workload.collType,
    comm->tasks.workload.nBytes,
    &isNew);
  if (isNew) {
    // If there is a tuner, the tuner will first call native, and then may form multiple plans as references, and finally form our plan
    TRACE(NCCL_COLL, "[Tuner] Get native plan for new workload.");
    NCCLCHECKGOTO(getPlans(comm, nPlans, true/*native*/, true/*honor*/), result, failure);
    *nPlans = 0;
    NCCLCHECK(cleanUpPlans(comm));
    NCCLCHECKGOTO(cleanUpTasks(comm, true), result, failure);
    NCCLCHECKGOTO(reloadTasks(comm), result, failure);
    INFO(NCCL_COLL, "native: commHash=%llu, coll=%d, nBytes per rank=%ld, algorithm=%d, protocol=%d, isCopyEngineNotSmCopy=%d, p2pLevel=%d, nChannels=%d, nThreads=%d, wireChunkSize=%ld, iteration=%d, lastIterEffectiveChunksize=%d, native=%d", 
              comm->tasks.workload.commHash, comm->tasks.workload.collType, comm->tasks.workload.nBytes, 
              comm->tasks.candidate.algorithm, comm->tasks.candidate.protocol, comm->tasks.candidate.isCopyEngineNotSmCopy, comm->tasks.candidate.p2pLevel, 
              comm->tasks.candidate.nChannels, comm->tasks.candidate.nThreads, comm->tasks.candidate.wireChunksize, comm->tasks.candidate.iteration, comm->tasks.candidate.lastIterEffectiveChunksize, 
              comm->tasks.candidate.native);
  }
  // TODO(anonymous): from plans to native candidate.
  comm->tuner->getCandidate(
    comm->tasks.workload.commHash,
    comm->tasks.workload.collType,
    comm->tasks.workload.nBytes,
    &comm->tasks.candidate.algorithm,
    &comm->tasks.candidate.protocol,
    &comm->tasks.candidate.isCopyEngineNotSmCopy,
    &comm->tasks.candidate.p2pLevel,
    &comm->tasks.candidate.nChannels,
    &comm->tasks.candidate.nThreads,
    &comm->tasks.candidate.wireChunksize,
    &comm->tasks.candidate.iteration,
    &comm->tasks.candidate.lastIterEffectiveChunksize,
    &comm->tasks.candidate.native);
  INFO(NCCL_COLL, "after tuner: commHash=%llu, coll=%d, nBytes per rank=%ld, algorithm=%d, protocol=%d, isCopyEngineNotSmCopy=%d, p2pLevel=%d, nChannels=%d, nThreads=%d, wireChunkSize=%ld, iteration=%d, lastIterEffectiveChunksize=%d, native=%d, stream=%p, nThreadsTotal=%d",
            comm->tasks.workload.commHash, comm->tasks.workload.collType, comm->tasks.workload.nBytes, 
            comm->tasks.candidate.algorithm, comm->tasks.candidate.protocol, comm->tasks.candidate.isCopyEngineNotSmCopy, comm->tasks.candidate.p2pLevel, 
            comm->tasks.candidate.nChannels, comm->tasks.candidate.nThreads, comm->tasks.candidate.wireChunksize, comm->tasks.candidate.iteration, comm->tasks.candidate.lastIterEffectiveChunksize, 
            comm->tasks.candidate.native,
            comm->tasks.streamRecent,
            comm->tasks.candidate.nThreadsTotal);

  if (comm->tasks.candidate.native == 1) {
    // if (comm->tasks.nTasksP2p) {
    //   NCCLCHECKGOTO(connectPeers(comm, true, comm->p2pnChannelsPerPeer), result, failure);
    // }
    // to get the native plan
    TRACE(NCCL_COLL, "[Tuner] Tuner ask to using native plan.");
    NCCLCHECKGOTO(getPlans(comm, nPlans, true/*native*/, false/*honor*/), result, failure);
  } else {
    int p2pnChannelsPerPeerSendOrRecv = comm->tasks.candidate.nChannels;
    // Because src/transport.cc requires connections to be paired, the connections must be of the same size and have complementary information
    // p2pnChannelsPerPeerSendOrRecv advances connections tuner_p2pnChannelsPerPeer
    if (comm->tasks.nTasksP2p) {
      NCCLCHECKGOTO(connectPeers(comm, false, p2pnChannelsPerPeerSendOrRecv), result, failure);
    }
    // to get the our plan
    TRACE(NCCL_COLL, "[Tuner] Get our plan based on tuned candidate.");
    NCCLCHECKGOTO(getPlans(comm, nPlans, false/*native*/, false/*honor*/), result, failure);
  }
  NCCLCHECKGOTO(cleanTasks(comm), result, failure);

  // start profiling all plans
  comm->tuner->startProfiling(
    comm->tasks.workload.commHash,
    comm->tasks.streams->stream,
    comm->tasks.workload.collType,
    comm->tasks.workload.nBytes,
    comm->tasks.candidate.algorithm,
    comm->tasks.candidate.protocol,
    comm->tasks.candidate.isCopyEngineNotSmCopy,
    comm->tasks.candidate.p2pLevel,
    comm->tasks.candidate.nChannels,
    comm->tasks.candidate.nThreads,
    comm->tasks.candidate.wireChunksize,
    comm->tasks.candidate.iteration,
    comm->tasks.candidate.lastIterEffectiveChunksize,
    comm->tasks.candidate.native);
  NCCLCHECKGOTO(resetWorkloadAndCandidate(comm), result, failure);
failure:
  return result;
}
