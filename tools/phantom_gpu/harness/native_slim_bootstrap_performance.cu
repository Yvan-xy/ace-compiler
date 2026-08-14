#include "boot/Bootstrapper.cuh"
#include "phantom.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kPolynomialDegree = 65536;
constexpr std::size_t kLogicalSlots = 32768;
constexpr std::uint32_t kFirstModulusBits = 60;
constexpr std::uint32_t kScalingModulusBits = 56;
constexpr std::size_t kDataQCount = 26;
constexpr std::size_t kSpecialPCount = 9;
constexpr std::size_t kDeclaredQPartCount = 3;
constexpr std::size_t kCompilerInputActiveQ = 1;
// Phantom's Bootstrapper tests `chain_depth() >= 3`. The depth excludes the
// first Q prime, so its minimum is four active Q primes.
constexpr std::size_t kSlimInputChainDepth = 3;
constexpr std::size_t kSlimInputActiveQ = 4;
constexpr std::size_t kCallCount = 21;
constexpr int kHammingWeight = 192;
constexpr long kInverseDegree = 1;
constexpr double kThreshold = 0.01;

constexpr std::array<std::uint64_t, kDataQCount> kDataQ = {
    1152921504606584833ULL, 72057594069123073ULL, 72057594004111361ULL,
    72057594065453057ULL,   72057594006863873ULL, 72057594063093761ULL,
    72057594011189249ULL,   72057594062831617ULL, 72057594020626433ULL,
    72057594062438401ULL,   72057594021150721ULL, 72057594061651969ULL,
    72057594023903233ULL,   72057594058899457ULL, 72057594027704321ULL,
    72057594058375169ULL,   72057594029015041ULL, 72057594057195521ULL,
    72057594030981121ULL,   72057594047889409ULL, 72057594034913281ULL,
    72057594042646529ULL,   72057594035306497ULL, 72057594040680449ULL,
    72057594036879361ULL,   72057594038321153ULL};

constexpr std::array<std::uint64_t, kSpecialPCount> kSpecialP = {
    1152921504598720513ULL, 1152921504597016577ULL,
    1152921504595968001ULL, 1152921504592822273ULL,
    1152921504592429057ULL, 1152921504589938689ULL,
    1152921504586530817ULL, 1152921504583647233ULL,
    1152921504581419009ULL};

struct PhaseTimings {
  double context = 0;
  double keys = 0;
  double polynomial = 0;
  double rotation_keys = 0;
  double linear_transform = 0;
  double input = 0;
};

void Require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckCuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

double Seconds(Clock::time_point start, Clock::time_point stop) {
  return std::chrono::duration<double>(stop - start).count();
}

template <typename Function> double TimeSynchronized(Function &&function) {
  CheckCuda(cudaDeviceSynchronize(), "pre-phase synchronization failed");
  const auto start = Clock::now();
  function();
  CheckCuda(cudaDeviceSynchronize(), "post-phase synchronization failed");
  return Seconds(start, Clock::now());
}

std::vector<phantom::arith::Modulus>
CoefficientModuli(const std::string &policy) {
  if (policy == "phantom-generated") {
    std::vector<int> bit_sizes{kFirstModulusBits};
    bit_sizes.insert(bit_sizes.end(), kDataQCount - 1,
                     kScalingModulusBits);
    bit_sizes.insert(bit_sizes.end(), kSpecialPCount, kFirstModulusBits);
    return phantom::arith::CoeffModulus::Create(kPolynomialDegree,
                                                 bit_sizes);
  }
  Require(policy == "exact-ant", "unknown coefficient-modulus policy");
  std::vector<phantom::arith::Modulus> result;
  result.reserve(kDataQ.size() + kSpecialP.size());
  for (const auto value : kDataQ)
    result.emplace_back(value);
  for (const auto value : kSpecialP)
    result.emplace_back(value);
  return result;
}

std::vector<double> InputValues() {
  std::vector<double> result(kLogicalSlots);
  for (std::size_t index = 0; index < result.size(); ++index) {
    const double first = std::sin(double(index) * 0.001953125);
    const double second = std::cos(double(index) * 0.0009765625);
    result[index] = 0.02 * first + 0.01 * second;
  }
  return result;
}

double MaximumError(const std::vector<double> &actual,
                    const std::vector<double> &expected) {
  Require(actual.size() == expected.size(), "decoded output size changed");
  double maximum = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double error = std::abs(actual[index] - expected[index]);
    Require(std::isfinite(error), "decoded output contains a nonfinite value");
    maximum = std::max(maximum, error);
  }
  return maximum;
}

std::vector<double> Decode(phantom::CKKSEvaluator &evaluator,
                           PhantomCiphertext &ciphertext) {
  PhantomPlaintext plain;
  std::vector<double> result;
  evaluator.decryptor.decrypt(ciphertext, plain);
  evaluator.encoder.decode(plain, result);
  return result;
}

void CheckGpu(const std::string &expected) {
  int count = 0;
  CheckCuda(cudaGetDeviceCount(&count), "cannot count CUDA devices");
  Require(count == 1, "expected exactly one CUDA device");
  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, 0),
            "cannot inspect CUDA device");
  Require(expected == properties.name, "CUDA device name mismatch");
  Require(properties.major == 8 && properties.minor == 0,
          "expected SM80 compute capability");
}

void WriteTimings(const std::string &path,
                  const std::vector<double> &seconds) {
  std::ofstream output(path, std::ios::binary);
  Require(bool(output), "cannot create native timing output");
  output << "call_index\twarmup\tseconds\n" << std::setprecision(17);
  for (std::size_t index = 0; index < seconds.size(); ++index)
    output << index << '\t' << (index == 0 ? 1 : 0) << '\t' << seconds[index]
           << '\n';
  output.flush();
  Require(bool(output), "cannot write native timing output");
}

void WriteSummary(const std::string &path, const PhaseTimings &phases,
                  double first_error, double last_error,
                  const PhantomCiphertext &last_output,
                  const std::vector<double> &calls,
                  const std::string &modulus_policy) {
  const bool correct = first_error <= kThreshold && last_error <= kThreshold;
  std::ofstream output(path, std::ios::binary);
  Require(bool(output), "cannot create native summary output");
  output << std::setprecision(17) << "{\n"
         << "  \"schema_version\": "
            "\"ace.phantom.slim-bootstrap-performance/1.0.0\",\n"
         << "  \"status\": \"" << (correct ? "pass" : "fail") << "\",\n"
         << "  \"algorithm\": \"Bootstrapper::slim_bootstrap\",\n"
         << "  \"domain\": \"full-packed-real-only\",\n"
         << "  \"polynomial_degree\": " << kPolynomialDegree << ",\n"
         << "  \"logical_slots\": " << kLogicalSlots << ",\n"
         << "  \"first_modulus_bits\": " << kFirstModulusBits << ",\n"
         << "  \"scaling_modulus_bits\": " << kScalingModulusBits << ",\n"
         << "  \"data_q_count\": " << kDataQCount << ",\n"
         << "  \"special_p_count\": " << kSpecialPCount << ",\n"
         << "  \"declared_q_part_count\": " << kDeclaredQPartCount << ",\n"
         << "  \"hamming_weight\": " << kHammingWeight << ",\n"
         << "  \"inverse_degree\": " << kInverseDegree << ",\n"
         << "  \"compiler_input_active_q\": " << kCompilerInputActiveQ
         << ",\n"
         << "  \"native_slim_input_active_q\": " << kSlimInputActiveQ
         << ",\n"
         << "  \"native_slim_input_chain_depth\": "
         << kSlimInputChainDepth << ",\n"
         << "  \"modulus_policy\": \"" << modulus_policy << "\",\n"
         << "  \"input_compatibility\": "
            "\"slim_bootstrap requires chain depth three, which is four "
            "active Q primes\",\n"
         << "  \"exact_ordered_qp\": "
         << (modulus_policy == "exact-ant" ? "true" : "false") << ",\n"
         << "  \"call_count\": " << calls.size() << ",\n"
         << "  \"warmup_count\": 1,\n"
         << "  \"measured_count\": " << calls.size() - 1 << ",\n"
         << "  \"correctness_threshold\": " << kThreshold << ",\n"
         << "  \"first_maximum_absolute_error\": " << first_error << ",\n"
         << "  \"last_maximum_absolute_error\": " << last_error << ",\n"
         << "  \"output_active_q_count\": "
         << last_output.coeff_modulus_size() << ",\n"
         << "  \"output_raw_scale\": " << last_output.scale() << ",\n"
         << "  \"setup_seconds\": {\n"
         << "    \"context\": " << phases.context << ",\n"
         << "    \"keys\": " << phases.keys << ",\n"
         << "    \"polynomial\": " << phases.polynomial << ",\n"
         << "    \"rotation_keys\": " << phases.rotation_keys << ",\n"
         << "    \"linear_transform\": " << phases.linear_transform << ",\n"
         << "    \"input\": " << phases.input << "\n"
         << "  }\n"
         << "}\n";
  output.flush();
  Require(bool(output), "cannot write native summary output");
}

} // namespace

int main(int argc, char **argv) {
  try {
    Require(argc == 5,
            "usage: native_slim_bootstrap_performance <expected-gpu> "
            "<timing-output> <summary-output> "
            "<exact-ant|phantom-generated>");
    const std::string modulus_policy = argv[4];
    const auto progress = [&](const std::string &phase) {
      std::cerr << "native-slim-progress\t" << modulus_policy << '\t' << phase
                << '\n';
      std::cerr.flush();
    };
    progress("gpu-check-start");
    CheckGpu(argv[1]);
    progress("gpu-check-complete");

    PhaseTimings phases;
    const double scale = std::ldexp(1.0, kScalingModulusBits);
    phantom::EncryptionParameters parameters(phantom::scheme_type::ckks);
    parameters.set_poly_modulus_degree(kPolynomialDegree);
    parameters.set_coeff_modulus(CoefficientModuli(modulus_policy));
    parameters.set_special_modulus_size(kSpecialPCount);
    parameters.set_secret_key_hamming_weight(kHammingWeight);
    parameters.set_sparse_slots(kLogicalSlots);

    progress("context-start");
    const auto context_start = Clock::now();
    PhantomContext context(parameters);
    CheckCuda(cudaDeviceSynchronize(), "context synchronization failed");
    phases.context = Seconds(context_start, Clock::now());
    progress("context-complete");

    progress("keys-start");
    const auto key_start = Clock::now();
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;
    PhantomCKKSEncoder encoder(context);
    phantom::CKKSEvaluator evaluator(&context, &public_key, &secret_key,
                                     &encoder, &relin_keys, &galois_keys,
                                     scale);
    CheckCuda(cudaDeviceSynchronize(), "key synchronization failed");
    phases.keys = Seconds(key_start, Clock::now());
    progress("keys-complete");

    constexpr long loge = 10;
    constexpr long logn = 15;
    constexpr long logNh = 15;
    constexpr long total_level = 25;
    constexpr long boundary_K = 25;
    constexpr long sin_cos_degree = 59;
    constexpr long scale_factor = 2;
    Bootstrapper bootstrapper(loge, logn, logNh, total_level, scale,
                              boundary_K, sin_cos_degree, scale_factor,
                              kInverseDegree, &evaluator);
    bootstrapper.set_slim_relu(false);

    progress("polynomial-start");
    phases.polynomial =
        TimeSynchronized([&] { bootstrapper.prepare_mod_polynomial(); });
    progress("polynomial-complete");

    std::vector<int> rotation_steps{0};
    for (int index = 0; index < logNh; ++index)
      rotation_steps.push_back(1 << index);
    progress("rotation-keys-start");
    phases.rotation_keys = TimeSynchronized([&] {
      bootstrapper.addLeftRotKeys_Linear_to_vector_3(rotation_steps);
      evaluator.decryptor.create_galois_keys_from_steps(
          rotation_steps, *evaluator.galois_keys);
      bootstrapper.slot_vec.push_back(logn);
    });
    progress("rotation-keys-complete");
    progress("linear-transform-start");
    phases.linear_transform =
        TimeSynchronized([&] { bootstrapper.generate_LT_coefficient_3(); });
    progress("linear-transform-complete");

    const std::vector<double> clear_input = InputValues();
    PhantomPlaintext plain;
    PhantomCiphertext base_input;
    progress("input-start");
    phases.input = TimeSynchronized([&] {
      evaluator.encoder.encode(clear_input, scale, plain);
      evaluator.encryptor.encrypt(plain, base_input);
      while (context.get_context_data(base_input.params_id()).chain_depth() >
             kSlimInputChainDepth)
        evaluator.evaluator.mod_switch_to_next_inplace(base_input);
    });
    progress("input-complete");
    Require(base_input.coeff_modulus_size() == kSlimInputActiveQ,
            "native slim input does not have four active Q primes");
    const std::vector<double> expected = Decode(evaluator, base_input);

    std::vector<double> call_seconds;
    call_seconds.reserve(kCallCount);
    double first_error = 0;
    double last_error = 0;
    PhantomCiphertext last_output;
    for (std::size_t call = 0; call < kCallCount; ++call) {
      progress("bootstrap-call-" + std::to_string(call) + "-start");
      PhantomCiphertext owned_input = base_input;
      PhantomCiphertext output;
      call_seconds.push_back(TimeSynchronized(
          [&] { bootstrapper.slim_bootstrap(output, owned_input); }));
      if (call == 0)
        first_error = MaximumError(Decode(evaluator, output), expected);
      if (call + 1 == kCallCount) {
        last_error = MaximumError(Decode(evaluator, output), expected);
        last_output = output;
      }
      progress("bootstrap-call-" + std::to_string(call) + "-complete");
    }
    WriteTimings(argv[2], call_seconds);
    WriteSummary(argv[3], phases, first_error, last_error, last_output,
                 call_seconds, modulus_policy);
    Require(first_error <= kThreshold && last_error <= kThreshold,
            "native slim result exceeds correctness threshold");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "native slim bootstrap performance failed: " << error.what()
              << '\n';
    return 1;
  } catch (const char *error) {
    std::cerr << "native slim bootstrap performance failed: " << error << '\n';
    return 1;
  }
}
