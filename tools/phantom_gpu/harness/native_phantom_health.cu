#include "phantom.h"
#include "common/rt_api.h"
#include "rt_phantom/phantom_api.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ACE_CONTEXT_MANIFEST_SHA256
#error "ACE_CONTEXT_MANIFEST_SHA256 must identify the compiler-emitted manifest"
#endif

extern "C" {
int Get_input_count() { return 0; }
int Get_output_count() { return 0; }
DATA_SCHEME* Get_encode_scheme(int) { return nullptr; }
DATA_SCHEME* Get_decode_scheme(int) { return nullptr; }
}

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
    const PHANTOM_CONTEXT_MANIFEST* manifest = Get_phantom_context_manifest();
    if (manifest == nullptr || manifest->_data_q_bit_sizes == nullptr ||
        manifest->_special_p_bit_sizes == nullptr) {
      throw std::runtime_error("compiler context manifest is incomplete");
    }
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

    std::vector<int> bit_sizes(manifest->_data_q_bit_sizes,
                               manifest->_data_q_bit_sizes +
                                   manifest->_data_q_count);
    bit_sizes.insert(bit_sizes.end(), manifest->_special_p_bit_sizes,
                     manifest->_special_p_bit_sizes +
                         manifest->_special_p_count);

    phantom::EncryptionParameters parameters(phantom::scheme_type::ckks);
    parameters.set_poly_modulus_degree(manifest->_poly_degree);
    parameters.set_coeff_modulus(phantom::arith::CoeffModulus::Create(
        manifest->_poly_degree, bit_sizes));
    parameters.set_special_modulus_size(manifest->_special_p_count);
    parameters.set_secret_key_hamming_weight(manifest->_hamming_weight);

    PhantomContext context(parameters);
    PhantomCKKSEncoder encoder(context);
    if (encoder.slot_count() != manifest->_logical_slots) {
      std::cerr << "expected " << manifest->_logical_slots
                << " active slots, found " << encoder.slot_count() << '\n';
      return 1;
    }
    PhantomSecretKey secret_key(context);

    std::vector<std::complex<double>> input(encoder.slot_count());
    input[0] = {0.125, -0.25};
    input[1] = {-1.5, 2.25};
    input[2] = {3.0, 0.5};
    input[3] = {-0.75, -4.0};
    const double scale = std::ldexp(1.0, manifest->_scaling_modulus_bits);

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
              << ",\"context_manifest_sha256\":\""
              << ACE_CONTEXT_MANIFEST_SHA256 << "\"}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native Phantom health failed: " << error.what() << '\n';
    return 1;
  }
}
