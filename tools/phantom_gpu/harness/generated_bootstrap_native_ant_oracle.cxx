#include "generated_bootstrap_ant_common.h"

#include "ckks/bootstrap.h"
#include "ckks/evaluator.h"
#include "common/rt_api.h"
#include "util/type.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using generated_bootstrap_ant::AppendValues;
using generated_bootstrap_ant::BinaryDescriptor;
using generated_bootstrap_ant::Complex;
using generated_bootstrap_ant::Decode;
using generated_bootstrap_ant::EncryptComplex;
using generated_bootstrap_ant::Fail;
using generated_bootstrap_ant::FinalizeValueFile;
using generated_bootstrap_ant::FixtureCase;
using generated_bootstrap_ant::Json;
using generated_bootstrap_ant::LoadQualificationInputs;
using generated_bootstrap_ant::MaterializeCases;
using generated_bootstrap_ant::Metric;
using generated_bootstrap_ant::Provenance;
using generated_bootstrap_ant::QualificationInputs;
using generated_bootstrap_ant::Require;
using generated_bootstrap_ant::RuntimeContextAttestation;
using generated_bootstrap_ant::VerifyPrimeChain;
using generated_bootstrap_ant::VerifyRuntimeContext;
using generated_bootstrap_ant::WriteBytes;
using generated_bootstrap_ant::WriteJson;

constexpr char kProvider[] = "native-ant";
constexpr char kProviderSchema[] =
    "ace.phantom.bootstrap-native-ant/1.0.0";

CKKS_PARAMS* context_parameters = nullptr;

void ConfigureContext(const QualificationInputs& inputs) {
  const Json& rotations = inputs.resources.at("rotation_steps");
  Require(rotations.is_array(), "resource rotation_steps is not an array");
  const std::size_t allocation =
      sizeof(CKKS_PARAMS) + rotations.size() * sizeof(std::int32_t);
  context_parameters =
      static_cast<CKKS_PARAMS*>(std::calloc(1U, allocation));
  Require(context_parameters != nullptr, "cannot allocate ANT context view");
  context_parameters->_provider = LIB_ANT;
  context_parameters->_poly_degree =
      inputs.context.at("polynomial_degree").get<std::uint32_t>();
  context_parameters->_sec_level =
      inputs.context.at("security_level").get<std::size_t>();
  context_parameters->_mul_depth =
      inputs.context.at("data_q_bit_sizes").size() - 1U;
  context_parameters->_input_level = inputs.input_level;
  context_parameters->_first_mod_size =
      inputs.context.at("first_modulus_bits").get<std::size_t>();
  context_parameters->_scaling_mod_size =
      inputs.context.at("scaling_modulus_bits").get<std::size_t>();
  context_parameters->_num_q_parts =
      inputs.context.at("q_part_count").get<std::size_t>();
  context_parameters->_hamming_weight =
      inputs.context.at("hamming_weight").get<std::size_t>();
  context_parameters->_num_rot_idx = rotations.size();
  for (std::size_t index = 0; index < rotations.size(); ++index) {
    context_parameters->_rot_idxs[index] =
        rotations.at(index).get<std::int32_t>();
  }
}

void PrepareExplicitBootstrap(const QualificationInputs& inputs) {
  CKKS_BTS_CTX* bootstrap_context =
      Get_bts_ctx(reinterpret_cast<CKKS_EVALUATOR*>(Eval()));
  Require(bootstrap_context != nullptr, "native bootstrap context is null");
  VL_UI32* budgets = Alloc_value_list(UI32_TYPE, 2U);
  VL_UI32* dimensions = Alloc_value_list(UI32_TYPE, 2U);
  UI32_VALUE_AT(budgets, 0) = inputs.encode_budget;
  UI32_VALUE_AT(budgets, 1) = inputs.decode_budget;
  UI32_VALUE_AT(dimensions, 0) = 0U;
  UI32_VALUE_AT(dimensions, 1) = 0U;
  Bootstrap_setup(bootstrap_context, budgets, dimensions, inputs.slots);
  Bootstrap_keygen(bootstrap_context, inputs.slots);
  Free_value_list(dimensions);
  Free_value_list(budgets);
}

Json AttestationJson(const CKKS_BOOTSTRAP_EXECUTION_ATTESTATION& value) {
  return {{"invocation_count", value.invocation_count},
          {"early_identity_copy_count", value.early_copy_return_count},
          {"full_execution_count", value.full_execution_count},
          {"coeffs_to_slots_entry_count", value.coeffs_to_slots_entry_count},
          {"coeffs_to_slots_completion_count",
           value.coeffs_to_slots_completion_count},
          {"eval_mod_entry_count", value.eval_mod_entry_count},
          {"eval_mod_completion_count", value.eval_mod_completion_count},
          {"slots_to_coeffs_entry_count", value.slots_to_coeffs_entry_count},
          {"slots_to_coeffs_completion_count",
           value.slots_to_coeffs_completion_count},
          {"full_completion_count", value.full_completion_count}};
}

void RequireFullExecution(
    const CKKS_BOOTSTRAP_EXECUTION_ATTESTATION& value) {
  Require(value.invocation_count == 1U &&
              value.early_copy_return_count == 0U &&
              value.full_execution_count == 1U &&
              value.coeffs_to_slots_entry_count == 1U &&
              value.coeffs_to_slots_completion_count == 1U &&
              value.eval_mod_entry_count == 1U &&
              value.eval_mod_completion_count == 1U &&
              value.slots_to_coeffs_entry_count == 1U &&
              value.slots_to_coeffs_completion_count == 1U &&
              value.full_completion_count == 1U,
          "native bootstrap did not complete its full mathematical path");
}

Json RunOracle(const QualificationInputs& inputs, const std::string& executable,
               const std::string& output_values,
               const Json& context_attestation) {
  const std::vector<FixtureCase> cases = MaterializeCases(inputs);
  Json records = Json::array();
  Json order = Json::array();
  std::vector<std::uint8_t> payload;
  std::uint64_t invocation_count = 0U;
  for (const FixtureCase& fixture_case : cases) {
    order.push_back(fixture_case.id);
    CIPHER source = EncryptComplex(fixture_case.clear, inputs.input_level);
    CIPHER result = Alloc_ciphertext();
    Reset_bootstrap_execution_attestation();
    Eval_bootstrap_ciph(result, source, 0U, inputs.slots);
    CKKS_BOOTSTRAP_EXECUTION_ATTESTATION attestation{};
    Get_bootstrap_execution_attestation(&attestation);
    RequireFullExecution(attestation);
    ++invocation_count;
    const std::vector<Complex> values = Decode(result, inputs.slots);
    const Json metric =
        Metric(values, fixture_case.clear, inputs.maximum_error,
               std::string(kProvider) + ":" + fixture_case.id);
    records.push_back(AppendValues(
        payload, fixture_case.id, kProvider, values,
        {{"recipe", fixture_case.recipe},
         {"metrics_vs_clear", metric},
         {"execution_attestation", AttestationJson(attestation)}}));
    Free_ciphertext(result);
    Free_ciphertext(source);
  }
  const std::vector<std::uint8_t> value_file =
      FinalizeValueFile(payload, records, inputs);
  WriteBytes(output_values, value_file);
  return {{"schema_version", kProviderSchema},
          {"provider", kProvider},
          {"fixture_sha256", inputs.fixture_sha256},
          {"bindings", inputs.bindings},
          {"case_order", order},
          {"logical_slots", inputs.slots},
          {"binary", BinaryDescriptor(value_file, payload, records)},
          {"records", records},
          {"execution",
           {{"status", "pass"},
            {"skip_count", 0},
            {"timeout_count", 0},
            {"fallback_count", 0},
            {"nonfinite_count", 0},
            {"native_bootstrap_invocation_count", invocation_count},
            {"early_identity_copy_count", 0},
            {"context_attestation", context_attestation},
            {"provenance", Provenance(inputs, executable)}}}};
}

}  // namespace

extern "C" {
CKKS_PARAMS* Get_context_params() {
  if (context_parameters == nullptr) std::abort();
  return context_parameters;
}
RT_DATA_INFO* Get_rt_data_info() { return nullptr; }
int Get_input_count() { return 0; }
int Get_output_count() { return 0; }
DATA_SCHEME* Get_encode_scheme(int) { return nullptr; }
DATA_SCHEME* Get_decode_scheme(int) { return nullptr; }
}

int main(int argc, char** argv) {
  try {
    if (argc != 15) {
      std::cerr
          << "usage: generated_bootstrap_native_ant_oracle <fixture.json> "
             "<ace-source-manifest.json> <phantom-source-manifest.json> "
             "<compiler-invocation.json> <raw.air> <post-ckks.air> "
             "<context.json> <resource.json> <constant.json> "
             "<bootstrap-semantics.json> "
             "<post-operation.air> <post-operation-attestation.json> "
             "<output.json> <output.bin>\n";
      return 2;
    }
    const QualificationInputs inputs = LoadQualificationInputs(
        {argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
         argv[8], argv[9], argv[10], argv[11], argv[12]});
    ConfigureContext(inputs);
    setenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM", "1", 1);
    Prepare_context();
    VerifyRuntimeContext(inputs);
    VerifyPrimeChain(inputs.context);
    PrepareExplicitBootstrap(inputs);
    const Json record = RunOracle(inputs, argv[0], argv[14],
                                  RuntimeContextAttestation(inputs, kProvider));
    Finalize_context();
    std::free(context_parameters);
    context_parameters = nullptr;
    WriteJson(argv[13], record);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native ANT bootstrap oracle failed: " << error.what() << '\n';
    return 1;
  }
}
