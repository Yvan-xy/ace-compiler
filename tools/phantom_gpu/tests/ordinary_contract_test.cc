#include "phantom_ordinary_contract.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

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

int main() {
  CheckChainLayout(27, 1);
  CheckChainLayout(12, 3);

  constexpr std::size_t slots = 8192;
  const std::array<std::int64_t, 9> steps = {
      -8193, -8192, -8191, -1, 0, 1, 8191, 8192, 8193};
  const std::array<int, 9> expected = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
  for (std::size_t index = 0; index < steps.size(); ++index) {
    assert(contract::NormalizeRotation(steps[index], slots) == expected[index]);
  }
  ExpectContractFailure([] { contract::NormalizeRotation(1, 0); });

  assert(contract::DeriveLogicalSlots(
             16384, contract::PackingConvention::kFull) == slots);
  assert(contract::ValidateLogicalSlots(
             16384, contract::PackingConvention::kFull, slots) == slots);
  ExpectContractFailure([] {
    contract::ValidateLogicalSlots(16384, contract::PackingConvention::kFull,
                                   4096);
  });

  constexpr std::size_t scaling_bits = 56;
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
      [=] { contract::ScaleDegree(std::exp2(28.0), scaling_bits); });
  return 0;
}
