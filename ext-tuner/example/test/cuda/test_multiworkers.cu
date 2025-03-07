#include "src/cuda/cuda_timer.h"
#include "src/include/jobs/native_job.h"
#include "src/include/net/socket_communicator.h"
#include "src/include/optimizers/uniform_optimizer.h"
#include "src/include/tuner.h"

static __device__ inline uint64_t GlobalTimer64() {
  // return time in nanoseconds
  volatile uint64_t first_reading = 0;
  volatile uint32_t second_reading = 0;
  uint32_t high_bits_first = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
  high_bits_first = first_reading >> 32;
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
  if (high_bits_first == second_reading) {
    return first_reading;
  }
  return ((uint64_t)second_reading) << 32;
}

#if __CUDA_ARCH__ >= 700
#define NANOSLEEP(num) __nanosleep(num) /* idle-sleepMs */
#else
#define NANOSLEEP(num)                                                         \
  do {                                                                         \
    uint64_t start_time = GlobalTimer64();                                     \
    for (;;) {                                                                 \
      if ((GlobalTimer64() - start_time) > size_t(num))                        \
        break; /* busy-sleeping*/                                              \
    }                                                                          \
  } while (0)
#endif

__global__ void sleepKernel(uint64_t sleepNs) {
  NANOSLEEP(sleepNs);
  return;  // empty kernel to skip nvprof overhead!
}

void sleepMsKernelCall(cudaStream_t stream, uint64_t sleepMs) {
  sleepKernel<<<dim3(1, 1, 1), dim3(1, 1, 1), 0, stream>>>(sleepMs * 1000000);
}

void profiling(void (*func)(cudaStream_t, uint64_t), int32_t warmup,
               int32_t repeat, GIDTYPE groupID, Tuner &worker,
               int32_t sleepMs) {
  cudaStream_t stream;  // NOLINT
  CUDA_SAFE_CALL(cudaStreamCreate(&stream));
  // warm up
  uint64_t begin = std::chrono::steady_clock::now().time_since_epoch().count();
  for (int i = 0; i < warmup; i++) {
    func(stream, 10);
  }
  sleepMsKernelCall(stream, 10);  // 10 ms
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
  uint64_t warmup_end =
      std::chrono::steady_clock::now().time_since_epoch().count();
  uint64_t warmup_duration_ms =  // NOLINT
      1e3 * (warmup_end - begin) *
      std::chrono::steady_clock::duration::period::num /
      std::chrono::steady_clock::duration::period::den;
  printf("Call warmup, timecost: %llu ms\n", warmup_duration_ms);
  begin = std::chrono::steady_clock::now().time_since_epoch().count();

  Workload workload = {groupID, 10};  // NOLINT
  CONFIG threadblock = 2, threadPerThreadblock = 32;
  for (int i = 0; i < repeat; i++) {
    Candidate candidate = {threadblock, threadPerThreadblock};
    worker.query(groupID, workload, &candidate);
    worker.timer->begin(RecordKey(groupID, workload, candidate), false, stream);
    func(stream, sleepMs);
    worker.timer->end(groupID, false);
  }
  uint64_t call_end =
      std::chrono::steady_clock::now().time_since_epoch().count();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  uint64_t work_end =
      std::chrono::steady_clock::now().time_since_epoch().count();
  uint64_t call_duration_ms = 1e3 * (call_end - begin) *  // NOLINT
                              std::chrono::steady_clock::duration::period::num /
                              std::chrono::steady_clock::duration::period::den;
  uint64_t work_duration_ms = 1e3 * (work_end - begin) *  // NOLINT
                              std::chrono::steady_clock::duration::period::num /
                              std::chrono::steady_clock::duration::period::den;
  printf("Call repeat=%d, timecost: %llu ms\n", repeat, call_duration_ms);
  printf("Work repeat=%d, timecost: %llu ms\n", repeat, work_duration_ms);
  CUDA_SAFE_CALL(cudaStreamDestroy(stream));
}

void getValidCandidates(
    const Info &myInfo,
    const std::unordered_map<GIDTYPE, GroupInfo> &allGroupInfos,
    const Workload &workload, bool scale2, std::vector<Candidate> *candidates) {
  candidates->push_back({2, 32});
  candidates->push_back({1, 64});
  return;
}

void RunCoordinator() {
  Info myInfo;
  Tuner coordinator;
  auto communicator = std::make_unique<SocketCommunicator>();
  coordinator.start(myInfo, std::move(communicator), "COORDINATOR");
  coordinator.stop();
}

void RunWorker(int32_t nodeID, int32_t deviceID, std::map<std::string, int32_t> tunerEnvs,
               int32_t worldSize, uint64_t sleepMs) {
  CUDA_SAFE_CALL(cudaDeviceReset());
  CUDA_SAFE_CALL(cudaSetDevice(deviceID));
  Tuner worker;
  Info myInfo(nodeID, deviceID);
  auto optimizer =
      std::make_unique<UniformOptimizer<NativeJob>>(getValidCandidates);
  auto timer = std::make_unique<CudaTimer>();
  auto communicator = std::make_unique<SocketCommunicator>();
  worker.start(myInfo, std::move(communicator), "WORKER", std::move(optimizer),
               std::move(timer));

  GIDTYPE groupID = 0;
  int32_t root = 0, rank = deviceID, nrank = worldSize, nnode = 1;
  worker.communicator->addGroup(groupID, root, rank, nrank, nnode, tunerEnvs);
  profiling(sleepMsKernelCall, 5, 100, groupID, worker, sleepMs);
  worker.communicator->removeGroup(groupID);
  worker.stop();
}

int main(int argc, char *argv[]) {
  uint64_t sleepMs = strtoll(argv[1], NULL, 0);

  setenv("TUNER_COORDINATOR", "localhost:12390", true);
  setenv("TUNER_WORLDSIZE", "2", true);

  std::thread t0(RunCoordinator);
  pthread_setname_np(t0.native_handle(), "RunCoordinator");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::map<std::string, int32_t> tunerEnvs;
  std::thread t1(RunWorker, 0, 0, tunerEnvs, 2, sleepMs);
  pthread_setname_np(t1.native_handle(), "RunWorker0");
  std::thread t2(RunWorker, 0, 1, tunerEnvs, 2, sleepMs);
  pthread_setname_np(t2.native_handle(), "RunWorker1");

  t0.join();
  t1.join();
  t2.join();
  return 0;
}
