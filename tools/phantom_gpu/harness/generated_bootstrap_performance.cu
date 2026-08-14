#include "common/rt_api.h"
#include "rt_phantom/rt_phantom.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern CIPHERTEXT bootstrap_full(CIPHERTEXT, CIPHERTEXT);

namespace generated_performance {

std::vector<double> call_seconds;
double context_setup_seconds = 0.0;
double context_teardown_seconds = 0.0;

void CheckCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

CIPHERTEXT TimedBootstrapFull(CIPHERTEXT input, CIPHERTEXT zero) {
  CheckCuda(cudaDeviceSynchronize(), "pre-bootstrap synchronization failed");
  const auto start = std::chrono::steady_clock::now();
  CIPHERTEXT result = bootstrap_full(input, zero);
  CheckCuda(cudaDeviceSynchronize(), "post-bootstrap synchronization failed");
  const auto stop = std::chrono::steady_clock::now();
  call_seconds.push_back(std::chrono::duration<double>(stop - start).count());
  return result;
}

void TimedPrepareContext() {
  const auto start = std::chrono::steady_clock::now();
  Prepare_context();
  CheckCuda(cudaDeviceSynchronize(), "context setup synchronization failed");
  context_setup_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
}

void TimedFinalizeContext() {
  CheckCuda(cudaDeviceSynchronize(), "pre-teardown synchronization failed");
  const auto start = std::chrono::steady_clock::now();
  Finalize_context();
  context_teardown_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
}

void WriteTimings(const char *path) {
  std::ofstream output(path, std::ios::binary);
  if (!output)
    throw std::runtime_error(std::string("cannot create timing output ") + path);
  output << "kind\tindex\twarmup\tseconds\n" << std::setprecision(17);
  output << "context_setup\t-1\t0\t" << context_setup_seconds << '\n';
  for (std::size_t index = 0; index < call_seconds.size(); ++index)
    output << "bootstrap_call\t" << index << '\t' << (index == 0 ? 1 : 0)
           << '\t'
           << call_seconds[index] << '\n';
  output << "context_teardown\t-1\t0\t" << context_teardown_seconds << '\n';
  output.flush();
  if (!output)
    throw std::runtime_error(std::string("cannot write timing output ") + path);
}

} // namespace generated_performance

#define bootstrap_full generated_performance::TimedBootstrapFull
#define Finalize_context generated_performance::TimedFinalizeContext
#define Prepare_context generated_performance::TimedPrepareContext
#define main GeneratedCorrectnessMain
#include "tools/phantom_gpu/harness/generated_bootstrap_phantom_correctness.cu"
#undef main
#undef Prepare_context
#undef Finalize_context
#undef bootstrap_full

int main(int argc, char **argv) {
  if (argc != 17) {
    std::cerr << "usage: generated_bootstrap_performance <correctness args...> "
                 "<timing-output>\n";
    return 2;
  }
  const int status = GeneratedCorrectnessMain(argc - 1, argv);
  try {
    generated_performance::WriteTimings(argv[16]);
  } catch (const std::exception &error) {
    std::cerr << "generated performance timing write failed: " << error.what()
              << '\n';
    return 1;
  }
  return status;
}
