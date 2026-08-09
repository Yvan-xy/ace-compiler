#include "generated_bootstrap_ant_common.h"

#include "common/rt_api.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifndef ACE_POST_CKKS_AIR_SHA256
#error "ACE_POST_CKKS_AIR_SHA256 must bind the linked generated program"
#endif
#ifndef ACE_GENERATED_DSL_ANT_SOURCE_SHA256
#error "ACE_GENERATED_DSL_ANT_SOURCE_SHA256 must bind the linked source"
#endif

#define ACE_ORACLE_STRINGIFY_INNER(value) #value
#define ACE_ORACLE_STRINGIFY(value) ACE_ORACLE_STRINGIFY_INNER(value)

extern CIPHERTEXT bootstrap_full(CIPHERTEXT input,
                                 CIPHERTEXT encrypted_zero);

extern "C" {
int Get_input_count() { return 0; }
int Get_output_count() { return 0; }
DATA_SCHEME* Get_encode_scheme(int) { return nullptr; }
DATA_SCHEME* Get_decode_scheme(int) { return nullptr; }
}

namespace {

using generated_bootstrap_ant::AppendValues;
using generated_bootstrap_ant::BinaryDescriptor;
using generated_bootstrap_ant::Complex;
using generated_bootstrap_ant::Decode;
using generated_bootstrap_ant::EncryptComplex;
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

constexpr char kProvider[] = "generated-ant";
constexpr char kProviderSchema[] =
    "ace.phantom.bootstrap-generated-ant/1.0.0";

void VerifyLinkedProgram(const QualificationInputs& inputs) {
  const Json& bindings = inputs.semantics.at("bindings");
  Require(bindings.at("post_ckks_air_sha256") ==
              ACE_ORACLE_STRINGIFY(ACE_POST_CKKS_AIR_SHA256),
          "linked program post-CKKS AIR binding differs from semantics");
  Require(bindings.at("generated_dsl_ant_source_sha256") ==
              ACE_ORACLE_STRINGIFY(ACE_GENERATED_DSL_ANT_SOURCE_SHA256),
          "linked generated DSL/ANT source binding differs from semantics");
  Require(bindings.at("post_ckks_air_sha256") ==
              inputs.bindings.at("post_ckks_air_sha256"),
          "linked program does not use the fixture's canonical post-CKKS AIR");
}

Json RunOracle(const QualificationInputs& inputs, const std::string& executable,
               const std::string& output_values,
               const Json& context_attestation) {
  const std::vector<FixtureCase> cases = MaterializeCases(inputs);
  const std::vector<Complex> zero_values(inputs.slots, Complex{});
  Json records = Json::array();
  Json order = Json::array();
  std::vector<std::uint8_t> payload;
  std::uint64_t invocation_count = 0U;
  for (const FixtureCase& fixture_case : cases) {
    order.push_back(fixture_case.id);
    CIPHER source = EncryptComplex(fixture_case.clear, inputs.input_level);
    CIPHER encrypted_zero = EncryptComplex(zero_values, inputs.input_level);
    CIPHERTEXT result = bootstrap_full(*source, *encrypted_zero);
    ++invocation_count;
    const std::vector<Complex> values = Decode(&result, inputs.slots);
    const Json metric = Metric(values, fixture_case.clear, inputs.maximum_error);
    records.push_back(AppendValues(
        payload, fixture_case.id, kProvider, values,
        {{"recipe", fixture_case.recipe}, {"metrics_vs_clear", metric}}));
    Zero_ciph(&result);
    Free_ciphertext(encrypted_zero);
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
            {"generated_bootstrap_invocation_count", invocation_count},
            {"linked_post_ckks_air_sha256",
             inputs.bindings.at("post_ckks_air_sha256")},
            {"linked_generated_source_sha256",
             inputs.semantics.at("bindings").at(
                 "generated_dsl_ant_source_sha256")},
            {"context_attestation", context_attestation},
            {"provenance", Provenance(inputs, executable)}}}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 15) {
      std::cerr
          << "usage: generated_bootstrap_dsl_ant_oracle <fixture.json> "
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
    VerifyLinkedProgram(inputs);
    setenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM", "1", 1);
    Prepare_context();
    VerifyRuntimeContext(inputs);
    VerifyPrimeChain(inputs.context);
    const Json record = RunOracle(inputs, argv[0], argv[14],
                                  RuntimeContextAttestation(inputs, kProvider));
    Finalize_context();
    WriteJson(argv[13], record);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "generated DSL/ANT bootstrap oracle failed: " << error.what()
              << '\n';
    return 1;
  }
}
