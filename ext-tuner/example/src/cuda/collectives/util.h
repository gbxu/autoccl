#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include "src/cuda/nccl_params.h"

int effective2Wire(int effectiveChunksize, int proto) {
  int wireChunksize = -1;
  if (proto == NCCL_PROTO_LL)
    wireChunksize = static_cast<int>(effectiveChunksize * 2);
  else if (proto == NCCL_PROTO_LL128)
    wireChunksize = static_cast<int>(effectiveChunksize * 16 / 15);
  else if (proto == NCCL_PROTO_SIMPLE)
    wireChunksize = effectiveChunksize;
  return wireChunksize;
}
int wire2Effective(int wireChunksize, int proto) {
  int effectiveChunksize = -1;
  if (proto == NCCL_PROTO_LL)
    effectiveChunksize = static_cast<int>(wireChunksize / 2);
  else if (proto == NCCL_PROTO_LL128)
    effectiveChunksize = static_cast<int>(wireChunksize * 15 / 16);
  else if (proto == NCCL_PROTO_SIMPLE)
    effectiveChunksize = wireChunksize;
  return effectiveChunksize;
}
