#include <chrono>
#include <sstream>
#include <thread>
#include "plugin/cuda/plugin.h"
#include "test/cuda/tuner_loader.h"

void testLoader() {
  ncclTuner_t *tuner = nullptr;
  ncclLoadTunerPlugin(&tuner);
  ncclCloseTunerPlugin(&tuner);
}

int main() {
  testLoader();
  printf("test_loader pass\n");
  return 0;
}
