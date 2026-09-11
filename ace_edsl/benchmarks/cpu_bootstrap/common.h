// Shared workload and result format for the preliminary CPU comparison.
#pragma once
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace bench {
constexpr unsigned N = 65536, Slots = N / 2, Q = 31, P = 11;
constexpr unsigned InputQ = 2, OutputQ = 14, ScaleBits = 56;
constexpr double MaxError = 0.02;

inline void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
inline double Now() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline std::vector<std::complex<double>> Input() {
  std::vector<std::complex<double>> x(Slots);
  // Identical deterministic input, independent of each library's encryption RNG.
  for (unsigned i = 0; i < Slots; ++i)
    x[i] = {0.0625 * (static_cast<int>(i % 17) - 8), 0.0};
  return x;
}
inline unsigned Repetitions(int argc, char** argv) {
  Require(argc == 2, "expected measured repetition count");
  char* end = nullptr;
  long n = std::strtol(argv[1], &end, 10);
  Require(*end == '\0' && n > 0 && n <= 100, "invalid repetition count");
  return n;
}
inline double Error(const std::vector<std::complex<double>>& x,
                    const std::vector<std::complex<double>>& y) {
  Require(x.size() == y.size(), "wrong output slot count");
  double worst = 0;
  for (unsigned i = 0; i < x.size(); ++i) {
    Require(std::isfinite(y[i].real()) && std::isfinite(y[i].imag()),
            "nonfinite bootstrap output");
    worst = std::max(worst, std::abs(x[i] - y[i]));
  }
  if (worst > MaxError) {
    std::fprintf(stderr, "maximum error %.12g exceeds %.12g; first outputs:", worst, MaxError);
    for (unsigned i = 0; i < 4; ++i)
      std::fprintf(stderr, " (%.9g,%.9g)", y[i].real(), y[i].imag());
    std::fprintf(stderr, "\n");
    throw std::runtime_error("bootstrap exceeds common error tolerance");
  }
  return worst;
}
inline void CheckScale(double scale) {
  Require(std::isfinite(scale) && scale > 0 &&
          std::abs(std::log2(scale) - ScaleBits) < 0.01,
          "scale differs from 2^56");
}
inline void Result(const char* impl, unsigned sample, double seconds,
                   double error, unsigned raw_q, unsigned raw_sf,
                   double scale) {
  std::printf("CPU_BTS_RESULT={\"implementation\":\"%s\",\"sample\":%u,"
      "\"timed\":%s,\"seconds\":%.9f,\"maximum_error\":%.12g,"
      "\"ring_dimension\":%u,\"slots\":%u,\"context_q\":%u,\"p_count\":%u,"
      "\"raised_q\":%u,\"input_q\":%u,\"input_scale_degree\":1,"
      "\"output_q\":%u,\"output_scale_degree\":1,\"output_scale\":%.17g,"
      "\"raw_output_q\":%u,\"raw_output_scale_degree\":%u,"
      "\"threads\":%d,\"status\":\"pass\"}\n",
      impl, sample, sample ? "true" : "false", seconds, error,
      N, Slots, Q, P, Q, InputQ, OutputQ, scale, raw_q, raw_sf,
      omp_get_max_threads());
  std::fflush(stdout);
}
}  // namespace bench
