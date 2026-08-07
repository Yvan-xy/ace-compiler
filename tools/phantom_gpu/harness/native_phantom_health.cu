#include "phantom.h"
#include "fullpacked_bts_profile.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void RequireCuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

}  // namespace

int main() {
  try {
    static_assert(ace::phantom_profile::kCudaArchitecture == 80);
    int device_count = 0;
    RequireCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count != 1) {
      std::cerr << "expected exactly one CUDA device, found " << device_count
                << '\n';
      return 1;
    }

    cudaDeviceProp properties{};
    RequireCuda(cudaGetDeviceProperties(&properties, 0),
                "cudaGetDeviceProperties");
    if (std::string(properties.name).find("A100") == std::string::npos) {
      std::cerr << "expected an A100, found " << properties.name << '\n';
      return 1;
    }

    std::vector<int> bit_sizes(
        ace::phantom_profile::kDataQBitSizes.begin(),
        ace::phantom_profile::kDataQBitSizes.end());
    bit_sizes.insert(bit_sizes.end(),
                     ace::phantom_profile::kSpecialPBitSizes.begin(),
                     ace::phantom_profile::kSpecialPBitSizes.end());

    phantom::EncryptionParameters parameters(phantom::scheme_type::ckks);
    parameters.set_poly_modulus_degree(
        ace::phantom_profile::kPolynomialDegree);
    parameters.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
        ace::phantom_profile::kPolynomialDegree, bit_sizes));
    parameters.set_special_modulus_size(
        ace::phantom_profile::kSpecialPBitSizes.size());
    parameters.set_secret_key_hamming_weight(
        ace::phantom_profile::kSecretKeyHammingWeight);

    PhantomContext context(parameters);
    PhantomCKKSEncoder encoder(context);
    if (encoder.slot_count() != ace::phantom_profile::kActiveSlotCount) {
      std::cerr << "expected " << ace::phantom_profile::kActiveSlotCount
                << " active slots, found " << encoder.slot_count() << '\n';
      return 1;
    }
    PhantomSecretKey secret_key(context);

    std::vector<std::complex<double>> input(encoder.slot_count());
    input[0] = {0.125, -0.25};
    input[1] = {-1.5, 2.25};
    input[2] = {3.0, 0.5};
    input[3] = {-0.75, -4.0};
    const double scale =
        std::ldexp(1.0, ace::phantom_profile::kScalingModulusBits);

    PhantomPlaintext plaintext;
    encoder.encode(context, input, scale, plaintext);
    PhantomCiphertext ciphertext;
    secret_key.encrypt_symmetric(context, plaintext, ciphertext);
    PhantomPlaintext decrypted;
    secret_key.decrypt(context, ciphertext, decrypted);

    std::vector<std::complex<double>> output;
    encoder.decode(context, decrypted, output);
    RequireCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (output.size() != input.size()) {
      std::cerr << "decoded " << output.size() << " slots, expected "
                << input.size() << '\n';
      return 1;
    }

    double max_error = 0.0;
    for (std::size_t index = 0; index < input.size(); ++index) {
      max_error = std::max(max_error, std::abs(input[index] - output[index]));
    }
    if (max_error > 1.0e-4) {
      std::cerr << "native Phantom health error " << max_error
                << " exceeds tolerance\n";
      return 1;
    }

    std::cout << "{\"status\":\"pass\",\"gpu\":\""
              << properties.name << "\",\"device_count\":" << device_count
              << ",\"max_error\":" << max_error
              << ",\"profile_sha256\":\""
              << ace::phantom_profile::kProfileSha256 << "\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native Phantom health failed: " << error.what() << '\n';
    return 1;
  }
}
