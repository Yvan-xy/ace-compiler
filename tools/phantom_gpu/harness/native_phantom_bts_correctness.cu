#include "boot/Bootstrapper.cuh"
#include "phantom.h"

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Complex = std::complex<double>;
using Json = nlohmann::json;

constexpr char kCompatibilitySchema[] =
    "ace.phantom.native-bts-compatibility/1.0.0";
constexpr char kClearSchema[] = "ace.phantom.native-bts-clear-inputs/1.0.0";
constexpr char kRawSchema[] = "ace.phantom.native-bts-raw-execution/1.0.0";

[[noreturn]] void Fail(const std::string &message) {
  throw std::runtime_error(message);
}

void Require(bool condition, const std::string &message) {
  if (!condition)
    Fail(message);
}

Json LoadJson(const std::string &path) {
  std::ifstream input(path);
  Require(bool(input), "cannot open JSON " + path);
  Json value;
  input >> value;
  Require(bool(input) && value.is_object(), "cannot parse JSON " + path);
  return value;
}

std::vector<std::uint8_t> ReadBytes(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  Require(bool(input), "cannot open " + path);
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  Require(size >= 0, "cannot size " + path);
  input.seekg(0);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (!bytes.empty())
    input.read(reinterpret_cast<char *>(bytes.data()), size);
  Require(bool(input), "cannot read " + path);
  return bytes;
}

void RequireFresh(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  Require(!input.good(), "refusing to replace output " + path);
}

void WriteBytes(const std::string &path,
                const std::vector<std::uint8_t> &bytes) {
  RequireFresh(path);
  std::ofstream output(path, std::ios::binary);
  Require(bool(output), "cannot create " + path);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  output.flush();
  Require(bool(output), "cannot write " + path);
}

void WriteJson(const std::string &path, const Json &value) {
  RequireFresh(path);
  std::ofstream output(path, std::ios::binary);
  Require(bool(output), "cannot create " + path);
  output << value.dump(2) << '\n';
  output.flush();
  Require(bool(output), "cannot write " + path);
}

void CheckGpu(const std::string &expected) {
  Require(expected.find('/') == std::string::npos &&
              expected.find('\\') == std::string::npos,
          "expected GPU name must be a basename");
  int count = 0;
  Require(cudaGetDeviceCount(&count) == cudaSuccess && count == 1,
          "expected exactly one CUDA device");
  cudaDeviceProp properties{};
  Require(cudaGetDeviceProperties(&properties, 0) == cudaSuccess,
          "cannot inspect CUDA device");
  Require(expected == properties.name, "CUDA device name mismatch");
}

int ExactLog2(std::size_t value, const std::string &name) {
  Require(value >= 2 && (value & (value - 1)) == 0,
          name + " must be a power of two");
  int result = 0;
  while ((std::size_t{1} << result) != value)
    ++result;
  return result;
}

std::vector<Complex> ReadCase(const std::vector<std::uint8_t> &payload,
                              const Json &record, std::size_t slots) {
  Require(record.at("value_count").get<std::size_t>() == slots,
          "clear case is not full-slot");
  const std::size_t offset = record.at("offset_bytes");
  const std::size_t bytes = slots * 2 * sizeof(double);
  Require(offset <= payload.size() && bytes <= payload.size() - offset,
          "clear case exceeds its binary payload");
  std::vector<Complex> result(slots);
  for (std::size_t index = 0; index < slots; ++index) {
    double real = 0, imaginary = 0;
    std::memcpy(&real, payload.data() + offset + index * 16, sizeof(double));
    std::memcpy(&imaginary, payload.data() + offset + index * 16 + 8,
                sizeof(double));
    Require(std::isfinite(real) && std::isfinite(imaginary),
            "clear input contains NaN or infinity");
    result[index] = {real, imaginary};
  }
  return result;
}

Json Metric(const std::vector<Complex> &actual,
            const std::vector<Complex> &expected, double threshold,
            const std::string &label) {
  Require(actual.size() == expected.size() && !actual.empty(),
          label + " has unequal or empty vectors");
  double maximum = -1, sum = 0, squares = 0;
  std::size_t maximum_index = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double error = std::abs(actual[index] - expected[index]);
    Require(std::isfinite(error), label + " has a nonfinite error");
    if (error > maximum) {
      maximum = error;
      maximum_index = index;
    }
    sum += error;
    squares += error * error;
  }
  if (maximum > threshold) {
    std::ostringstream message;
    message << std::setprecision(std::numeric_limits<double>::max_digits10)
            << label << " exceeds its frozen threshold: maximum=" << maximum
            << ", index=" << maximum_index
            << ", actual=" << actual[maximum_index]
            << ", expected=" << expected[maximum_index]
            << ", threshold=" << threshold;
    Fail(message.str());
  }
  return {{"status", "pass"},
          {"comparison_count", actual.size()},
          {"threshold", threshold},
          {"maximum_absolute_error", maximum},
          {"maximum_absolute_error_index", maximum_index},
          {"mean_absolute_error", sum / actual.size()},
          {"root_mean_square_error", std::sqrt(squares / actual.size())},
          {"estimated_precision_bits",
           maximum == 0 ? Json(nullptr) : Json(-std::log2(maximum))},
          {"estimated_precision_is_infinite", maximum == 0}};
}

double MaximumDifference(const std::vector<Complex> &left,
                         const std::vector<Complex> &right) {
  Require(left.size() == right.size(), "repeat vectors differ in length");
  double result = 0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result = std::max(result, std::abs(left[index] - right[index]));
  return result;
}

void AppendDouble(std::vector<std::uint8_t> &output, double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (unsigned shift = 0; shift < 64; shift += 8)
    output.push_back(static_cast<std::uint8_t>(bits >> shift));
}

struct Inputs {
  Json context;
  Json compatibility;
  Json clear_record;
  std::vector<std::uint8_t> clear_values;
  std::string compatibility_sha256;
  std::string mode;
  std::size_t slots = 0;
  std::size_t degree = 0;
  std::size_t data_q_count = 0;
  std::size_t special_p_count = 0;
  int scaling_bits = 0;
  double scale = 0;
};

Inputs Authenticate(char **argv) {
  Inputs input{LoadJson(argv[1]), LoadJson(argv[2]), LoadJson(argv[3]),
               ReadBytes(argv[4]), argv[5], argv[7]};
  Require(input.compatibility.at("schema_version") == kCompatibilitySchema &&
              input.compatibility.at("status") == "compatible",
          "native compatibility record is not passing");
  Require(input.clear_record.at("schema_version") == kClearSchema &&
              input.clear_record.at("status") == "pass",
          "clear-input record is not passing");
  Require(input.mode == "all" || input.mode == "sentinel",
          "mode must be all or sentinel");
  const auto &compiler = input.compatibility.at("compiler_context");
  input.degree = compiler.at("polynomial_degree");
  input.slots = compiler.at("logical_slots");
  input.data_q_count = compiler.at("data_q_bit_sizes").size();
  input.special_p_count = compiler.at("special_p_bit_sizes").size();
  input.scaling_bits = compiler.at("scaling_modulus_bits");
  input.scale = std::ldexp(1.0, input.scaling_bits);
  Require(input.context.at("polynomial_degree") == input.degree &&
              input.context.at("logical_slot_capacity") == input.slots &&
              input.context.at("data_q_bit_sizes") ==
                  compiler.at("data_q_bit_sizes") &&
              input.context.at("special_p_bit_sizes") ==
                  compiler.at("special_p_bit_sizes") &&
              input.context.at("q_part_count") == compiler.at("q_part_count") &&
              input.context.at("hamming_weight") == compiler.at("hamming_weight") &&
              input.context.at("security_level") == compiler.at("security_level") &&
              input.context.at("input_level") == compiler.at("input_level") &&
              input.context.at("scaling_modulus_bits") == input.scaling_bits,
          "compiler-emitted context differs from native compatibility");
  Require(input.degree == input.slots * 2 &&
              ExactLog2(input.slots, "logical slots") ==
                  input.compatibility.at("native_configuration")
                      .at("constructor")
                      .at("logn") &&
              input.compatibility.at("representability").at("status") ==
                  "pass" &&
              input.compatibility.at("representability")
                      .at("context_adjusted") == false &&
              input.compatibility.at("representability")
                      .at("manual_parameter_profile_used") == false,
          "native configuration is not an exact representation of the context");
  Require(input.clear_record.at("logical_slots") == input.slots &&
              input.clear_record.at("case_order") ==
                  input.compatibility.at("correctness_contract")
                      .at("case_order") &&
              input.clear_record.at("values").at("size_bytes") ==
                  input.clear_values.size(),
          "clear inputs differ from compatibility");
  return input;
}

Json Metadata(const PhantomCiphertext &cipher, std::size_t slots,
              int scaling_bits) {
  const std::size_t active_q_count = cipher.coeff_modulus_size();
  Require(active_q_count > 0, "native output has no active Q modulus");
  const double scale_degree = std::log2(cipher.scale()) / scaling_bits;
  const auto rounded = static_cast<std::int64_t>(std::llround(scale_degree));
  return {{"ace_level", active_q_count - 1},
          {"active_q_count", active_q_count},
          {"phantom_chain_index", cipher.chain_index()},
          {"raw_scale", cipher.scale()},
          {"scale_degree", rounded},
          {"ciphertext_size", cipher.size()},
          {"logical_slots", slots},
          {"ntt_state", cipher.is_ntt_form()}};
}

Json InputMetadata(const PhantomCiphertext &cipher, std::size_t slots) {
  return {{"active_q_count", cipher.coeff_modulus_size()},
          {"phantom_chain_index", cipher.chain_index()},
          {"raw_scale", cipher.scale()},
          {"ciphertext_size", cipher.size()},
          {"logical_slots", slots},
          {"ntt_state", cipher.is_ntt_form()}};
}

void ValidateOutputMetadata(const Json &observed, const Json &invariants,
                            int scaling_bits, std::size_t data_q_count) {
  const std::size_t active_q_count = observed.at("active_q_count");
  Require(active_q_count >=
                  invariants.at("minimum_active_q_count").get<std::size_t>() &&
              active_q_count <=
                  invariants.at("maximum_active_q_count").get<std::size_t>(),
          "native output active Q count is outside the emitted context");
  Require(observed.at("ace_level").get<std::size_t>() + 1 ==
              active_q_count,
          "native output ACE level does not use Phantom's Q-count-minus-one convention");
  Require(observed.at("phantom_chain_index").get<std::size_t>() ==
              1 + data_q_count - active_q_count,
          "native output chain index is inconsistent with its active Q count");
  for (const char *key : {"scale_degree", "ciphertext_size",
                          "logical_slots", "ntt_state"})
    Require(observed.at(key) == invariants.at(key),
            std::string("native output invariant differs at ") + key);
  const double coordinate =
      std::log2(observed.at("raw_scale").get<double>()) / scaling_bits;
  Require(std::isfinite(coordinate) &&
              std::abs(coordinate -
                       invariants.at("scale_degree").get<double>()) <=
                  1.0e-4,
          "native output raw scale differs from the expected coordinate");
}

struct Runtime {
  explicit Runtime(const Inputs &input)
      : parameters(phantom::scheme_type::ckks), scale(input.scale) {
    const auto &compiler = input.compatibility.at("compiler_context");
    parameters.set_poly_modulus_degree(input.degree);
    std::vector<int> modulus_bits =
        compiler.at("data_q_bit_sizes").get<std::vector<int>>();
    const auto special =
        compiler.at("special_p_bit_sizes").get<std::vector<int>>();
    modulus_bits.insert(modulus_bits.end(), special.begin(), special.end());
    parameters.set_coeff_modulus(
        phantom::arith::CoeffModulus::Create(input.degree, modulus_bits));
    parameters.set_special_modulus_size(special.size());
    parameters.set_secret_key_hamming_weight(
        compiler.at("hamming_weight").get<int>());
    parameters.set_sparse_slots(input.slots);
    context = std::make_unique<PhantomContext>(parameters);
    encoder = std::make_unique<PhantomCKKSEncoder>(*context);
    secret_key = std::make_unique<PhantomSecretKey>(*context);
    public_key =
        std::make_unique<PhantomPublicKey>(secret_key->gen_publickey(*context));
    relin_key =
        std::make_unique<PhantomRelinKey>(secret_key->gen_relinkey(*context));
    galois_key = std::make_unique<PhantomGaloisKey>();
    evaluator = std::make_unique<phantom::CKKSEvaluator>(
        context.get(), public_key.get(), secret_key.get(), encoder.get(),
        relin_key.get(), galois_key.get(), scale);
    Require(encoder->logical_slot_count() == input.slots &&
                context->first_context_data().parms().coeff_modulus().size() ==
                    input.data_q_count &&
                context->get_first_index() == 1,
            "constructed native context does not represent the manifest");

    const auto &constructor =
        input.compatibility.at("native_configuration").at("constructor");
    bootstrapper = std::make_unique<Bootstrapper>(
        constructor.at("loge").get<long>(),
        constructor.at("logn").get<long>(),
        constructor.at("logNh").get<long>(),
        constructor.at("total_level").get<long>(),
        constructor.at("final_scale").get<double>(),
        constructor.at("boundary_K").get<long>(),
        constructor.at("sin_cos_degree").get<long>(),
        constructor.at("scale_factor").get<long>(),
        constructor.at("inverse_degree").get<long>(), evaluator.get(),
        constructor.at("enable_slim_relu").get<bool>());
    bootstrapper->prepare_mod_polynomial();
    std::vector<int> steps{0};
    for (int bit = 0; bit < constructor.at("logNh").get<int>(); ++bit)
      steps.push_back(1 << bit);
    bootstrapper->addLeftRotKeys_Linear_to_vector_3(steps);
    const int post_step = input.compatibility.at("correctness_contract")
                              .at("post_rotation_step");
    if (std::find(steps.begin(), steps.end(), post_step) == steps.end())
      steps.push_back(post_step);
    evaluator->decryptor.create_galois_keys_from_steps(steps, *galois_key);
    bootstrapper->slot_vec.push_back(constructor.at("logn").get<long>());
    bootstrapper->generate_LT_coefficient_3();
  }

  ~Runtime() {
    if (bootstrapper && bootstrapper->mod_reducer) {
      delete bootstrapper->mod_reducer->poly_generator;
      delete bootstrapper->mod_reducer->inverse_poly_generator;
      bootstrapper->mod_reducer->poly_generator = nullptr;
      bootstrapper->mod_reducer->inverse_poly_generator = nullptr;
      delete bootstrapper->mod_reducer;
      bootstrapper->mod_reducer = nullptr;
    }
  }

  PhantomCiphertext Encrypt(const std::vector<Complex> &values) {
    const std::size_t bottom =
        context->get_first_index() +
        context->first_context_data().parms().coeff_modulus().size() - 1;
    const auto plain = encoder->encode(*context, values, scale, bottom);
    PhantomCiphertext cipher;
    public_key->encrypt_asymmetric(*context, plain, cipher);
    return cipher;
  }

  std::vector<Complex> Decrypt(const PhantomCiphertext &cipher,
                               std::size_t slots) {
    auto values =
        secret_key->decrypt_decode_complex(*context, cipher, *encoder);
    Require(values.size() == slots, "native decrypt is not full-slot");
    for (const auto &value : values)
      Require(std::isfinite(value.real()) && std::isfinite(value.imag()),
              "native decrypt contains NaN or infinity");
    return values;
  }

  phantom::EncryptionParameters parameters;
  double scale;
  std::unique_ptr<PhantomContext> context;
  std::unique_ptr<PhantomCKKSEncoder> encoder;
  std::unique_ptr<PhantomSecretKey> secret_key;
  std::unique_ptr<PhantomPublicKey> public_key;
  std::unique_ptr<PhantomRelinKey> relin_key;
  std::unique_ptr<PhantomGaloisKey> galois_key;
  std::unique_ptr<phantom::CKKSEvaluator> evaluator;
  std::unique_ptr<Bootstrapper> bootstrapper;
};

std::pair<Json, std::vector<std::uint8_t>> Execute(const Inputs &input) {
  const auto &contract = input.compatibility.at("correctness_contract");
  const auto &native_config = input.compatibility.at("native_configuration");
  const double threshold = contract.at("provider_clear_maximum_absolute");
  const double repeat_threshold = contract.at("repeat_maximum_absolute");
  const int configured_calls = contract.at("calls_per_case");
  const int calls = input.mode == "all" ? configured_calls : 1;
  const std::string sentinel = contract.at("sanitizer_sentinel");
  std::vector<Json> selected;
  for (const auto &record : input.clear_record.at("records")) {
    const std::string case_id = record.at("case_id");
    if (input.mode == "all" || case_id == sentinel)
      selected.push_back(record);
  }
  Require(!selected.empty() &&
              (input.mode == "all" || selected.size() == 1),
          "native execution case selection is invalid");

  Json records = Json::array(), order = Json::array();
  std::vector<std::uint8_t> output_values;
  std::size_t invocation_count = 0, completion_count = 0;
  {
    Runtime runtime(input);
    for (const auto &clear_descriptor : selected) {
      const std::string case_id = clear_descriptor.at("case_id");
      order.push_back(case_id);
      const auto clear = ReadCase(input.clear_values, clear_descriptor,
                                  input.slots);
      PhantomCiphertext encrypted = runtime.Encrypt(clear);
      const Json input_metadata = InputMetadata(encrypted, input.slots);
      Require(input_metadata == native_config.at("input"),
              "native encrypted input metadata differs from compatibility");

      std::vector<std::vector<Complex>> call_values;
      Json call_metrics = Json::array(), call_metadata = Json::array();
      PhantomCiphertext primary;
      for (int call = 0; call < calls; ++call) {
        PhantomCiphertext owned_input = encrypted;
        PhantomCiphertext result;
        ++invocation_count;
        runtime.bootstrapper->bootstrap_3(result, owned_input);
        ++completion_count;
        auto decoded = runtime.Decrypt(result, input.slots);
        Json metadata = Metadata(result, input.slots, input.scaling_bits);
        ValidateOutputMetadata(metadata, native_config.at("output_invariants"),
                               input.scaling_bits, input.data_q_count);
        call_metrics.push_back(
            Metric(decoded, clear, threshold,
                   "native-phantom/" + case_id + "/call-" +
                       std::to_string(call)));
        call_metadata.push_back(metadata);
        call_values.push_back(std::move(decoded));
        if (call == 0)
          primary = result;
        result.release();
        owned_input.release();
      }
      double repeat_maximum = 0;
      for (std::size_t call = 1; call < call_values.size(); ++call)
        repeat_maximum = std::max(
            repeat_maximum,
            MaximumDifference(call_values.front(), call_values[call]));
      Require(repeat_maximum <= repeat_threshold,
              "native bootstrap repeat threshold exceeded");

      const auto &multiply = contract.at("post_multiply");
      Require(multiply.at("plaintext_scale_degree").get<int>() == 0,
              "post multiply scale degree must be zero");
      const Complex multiplier(multiply.at("real").get<double>(),
                               multiply.at("imaginary").get<double>());
      std::vector<Complex> multiply_plain_values(input.slots, multiplier);
      auto plain = runtime.encoder->encode(*runtime.context,
                                           multiply_plain_values, 1.0,
                                           primary.chain_index());
      PhantomCiphertext multiply_result;
      runtime.evaluator->evaluator.multiply_plain(primary, plain,
                                                   multiply_result);
      auto multiply_values = runtime.Decrypt(multiply_result, input.slots);
      auto multiply_expected = clear;
      for (auto &value : multiply_expected)
        value *= multiplier;
      const Json multiply_metric =
          Metric(multiply_values, multiply_expected,
                 threshold * std::max(1.0, std::abs(multiplier)),
                 "native-phantom/" + case_id + "/post-multiply");
      const Json multiply_metadata =
          Metadata(multiply_result, input.slots, input.scaling_bits);
      ValidateOutputMetadata(multiply_metadata,
                             native_config.at("output_invariants"),
                             input.scaling_bits, input.data_q_count);

      const int step = contract.at("post_rotation_step");
      PhantomCiphertext rotation_result;
      runtime.evaluator->evaluator.rotate_vector(
          primary, step, *runtime.galois_key, rotation_result);
      auto rotation_values = runtime.Decrypt(rotation_result, input.slots);
      std::vector<Complex> rotation_expected(input.slots);
      for (std::size_t index = 0; index < input.slots; ++index)
        rotation_expected[index] =
            clear[(index + step + input.slots) % input.slots];
      const Json rotation_metric =
          Metric(rotation_values, rotation_expected, threshold,
                 "native-phantom/" + case_id + "/post-rotation");
      const Json rotation_metadata =
          Metadata(rotation_result, input.slots, input.scaling_bits);
      ValidateOutputMetadata(rotation_metadata,
                             native_config.at("output_invariants"),
                             input.scaling_bits, input.data_q_count);

      const std::size_t output_offset = output_values.size();
      for (const auto &value : call_values.front()) {
        AppendDouble(output_values, value.real());
        AppendDouble(output_values, value.imag());
      }
      records.push_back(
          {{"case_id", case_id},
           {"offset_bytes", output_offset},
           {"value_count", input.slots},
           {"metadata",
            {{"input", input_metadata},
             {"bootstrap", call_metadata.front()},
             {"metrics_vs_clear",
              Metric(call_values.front(), clear, threshold,
                     "native-phantom/" + case_id + "/primary")},
             {"repeatability",
              {{"calls", calls},
               {"independently_owned_executions", true},
               {"maximum_absolute_difference", repeat_maximum},
               {"threshold", repeat_threshold},
               {"call_metrics", call_metrics},
               {"call_metadata", call_metadata}}},
             {"post_operations",
              {{"ciphertext_plaintext_multiply",
                {{"metric", multiply_metric},
                 {"metadata", multiply_metadata}}},
               {"rotation",
                {{"metric", rotation_metric},
                 {"metadata", rotation_metadata},
                 {"step", step}}}}},
             {"ownership",
              {{"all_owned_objects_released", true},
               {"bootstrap_inputs_released", calls},
               {"bootstrap_results_released", calls},
               {"post_operation_objects_released", true},
               {"teardown_completed", true}}}}}});
      plain.release();
      multiply_result.release();
      rotation_result.release();
      primary.release();
      encrypted.release();
    }
    Require(cudaDeviceSynchronize() == cudaSuccess,
            "native execution synchronization failed");
  }
  Require(cudaDeviceSynchronize() == cudaSuccess,
          "native teardown synchronization failed");
  return {
      {{"schema_version", kRawSchema},
       {"status", "pass"},
       {"mode", input.mode},
       {"compatibility_sha256", input.compatibility_sha256},
       {"case_order", order},
       {"logical_slots", input.slots},
       {"records", records},
       {"execution",
        {{"status", "pass"},
         {"bootstrap_3_invocation_count", invocation_count},
         {"bootstrap_3_completion_count", completion_count},
         {"direct_bootstrap_3_callsite", true},
         {"skip_count", 0},
         {"fallback_count", 0},
         {"identity_path_count", 0},
         {"nonfinite_count", 0},
         {"teardown_completed", true}}}},
      std::move(output_values)};
}
} // namespace

int main(int argc, char **argv) {
  try {
    Require(argc == 11,
            "usage: native_phantom_bts_correctness <context> "
            "<compatibility> <clear-record> <clear-values> "
            "<compatibility-sha256> <expected-gpu-basename> <all|sentinel> "
            "<raw-record-output> <raw-values-output> <reserved-proof-token>");
    Require(std::string(argv[10]) == "direct-bootstrap-3-no-fallback",
            "native bootstrap proof token is missing");
    RequireFresh(argv[8]);
    RequireFresh(argv[9]);
    CheckGpu(argv[6]);
    Inputs input = Authenticate(argv);
    auto [record, values] = Execute(input);
    WriteBytes(argv[9], values);
    WriteJson(argv[8], record);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "native Phantom bootstrap correctness failed: "
              << error.what() << '\n';
    return 1;
  }
}
