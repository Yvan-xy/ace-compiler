#include "phantom.h"
#include "fullpacked_bts_profile.h"

#include <cstddef>
#include <vector>

int main() {
  static_assert(ace::phantom_profile::kCudaArchitecture == 80);
  std::vector<int> bit_sizes(
      ace::phantom_profile::kDataQBitSizes.begin(),
      ace::phantom_profile::kDataQBitSizes.end());
  bit_sizes.insert(bit_sizes.end(),
                   ace::phantom_profile::kSpecialPBitSizes.begin(),
                   ace::phantom_profile::kSpecialPBitSizes.end());
  const std::vector<phantom::arith::Modulus> moduli =
      phantom::arith::CoeffModulus::Create(
          ace::phantom_profile::kPolynomialDegree, bit_sizes);
  return moduli.size() == bit_sizes.size() ? 0 : 1;
}
