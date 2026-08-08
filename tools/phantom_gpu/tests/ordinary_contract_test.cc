#include "phantom_ordinary_contract.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace contract = ace::phantom::ordinary;

template <typename FN>
void ExpectContractFailure(FN&& fn) {
  bool failed = false;
  try {
    fn();
  } catch (const contract::ContractException&) {
    failed = true;
  }
  assert(failed);
}

void CheckChainLayout(std::size_t q_count, std::size_t first_chain) {
  const std::size_t middle_level = (q_count + 1) / 2;
  const std::size_t last_chain = first_chain + q_count - 1;

  assert(contract::AceLevelToChainIndex(q_count, q_count, first_chain) ==
         first_chain);
  assert(contract::AceLevelToChainIndex(middle_level, q_count, first_chain) ==
         first_chain + q_count - middle_level);
  assert(contract::AceLevelToChainIndex(1, q_count, first_chain) == last_chain);
  assert(contract::TargetQCountToChainIndex(q_count, q_count, first_chain) ==
         first_chain);
  assert(contract::ChainIndexToActiveQ(first_chain, q_count, first_chain) ==
         q_count);
  assert(contract::ChainIndexToActiveQ(last_chain, q_count, first_chain) == 1);
  assert(contract::ChainIndexToAceLevel(last_chain, q_count, first_chain) == 1);

  ExpectContractFailure(
      [=] { contract::AceLevelToChainIndex(0, q_count, first_chain); });
  ExpectContractFailure([=] {
    contract::AceLevelToChainIndex(q_count + 1, q_count, first_chain);
  });
  if (first_chain != 0) {
    ExpectContractFailure([=] {
      contract::ChainIndexToActiveQ(first_chain - 1, q_count, first_chain);
    });
  }
  ExpectContractFailure([=] {
    contract::ChainIndexToActiveQ(last_chain + 1, q_count, first_chain);
  });
}

std::size_t ParseSize(const char* text, const char* name) {
  std::size_t consumed = 0;
  const std::string value(text);
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || parsed == 0 ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return static_cast<std::size_t>(parsed);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    throw std::invalid_argument(
        "usage: ordinary_contract_test degree slots data_q_count scale_bits");
  }
  const std::size_t degree = ParseSize(argv[1], "polynomial degree");
  const std::size_t slots = ParseSize(argv[2], "logical slots");
  const std::size_t q_count = ParseSize(argv[3], "data-Q count");
  const std::size_t scaling_bits = ParseSize(argv[4], "scaling bits");
  if (slots < 2 ||
      slots > static_cast<std::size_t>(
                  std::numeric_limits<std::int64_t>::max() - 1) ||
      q_count == std::numeric_limits<std::size_t>::max() ||
      scaling_bits < 2) {
    throw std::invalid_argument("compiler context exceeds contract-test range");
  }

  CheckChainLayout(q_count, 0);
  CheckChainLayout(q_count + 1, q_count);

  const std::int64_t signed_slots = static_cast<std::int64_t>(slots);
  const std::array<std::int64_t, 9> steps = {
      -signed_slots - 1, -signed_slots, -signed_slots + 1, -1, 0,
      1,                 signed_slots - 1, signed_slots, signed_slots + 1};
  const std::array<int, 9> expected = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
  for (std::size_t index = 0; index < steps.size(); ++index) {
    assert(contract::NormalizeRotation(steps[index], slots) == expected[index]);
  }
  ExpectContractFailure([] { contract::NormalizeRotation(1, 0); });

  assert(contract::DeriveLogicalSlots(degree, contract::PackingConvention::kFull) ==
         slots);
  assert(contract::ValidateLogicalSlots(
             degree, contract::PackingConvention::kFull, slots) == slots);
  ExpectContractFailure([=] {
    contract::ValidateLogicalSlots(degree, contract::PackingConvention::kFull,
                                   slots - 1);
  });

  for (std::int64_t degree : {0, 1, 2, 3}) {
    const double scale = contract::ScaleForDegree(degree, scaling_bits);
    assert(contract::ScaleDegree(scale, scaling_bits) == degree);
  }
  ExpectContractFailure([=] {
    contract::ScaleDegree(std::numeric_limits<double>::quiet_NaN(),
                          scaling_bits);
  });
  ExpectContractFailure([=] {
    contract::ScaleDegree(std::numeric_limits<double>::infinity(),
                          scaling_bits);
  });
  ExpectContractFailure(
      [=] { contract::ScaleDegree(-1.0, scaling_bits); });
  ExpectContractFailure(
      [=] {
        contract::ScaleDegree(
            std::exp2(static_cast<double>(scaling_bits) / 2.0), scaling_bits);
      });
  return 0;
}
