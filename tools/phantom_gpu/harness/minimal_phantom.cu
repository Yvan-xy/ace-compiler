#include "phantom.h"
#include "rt_phantom/phantom_api.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

int main() {
  const PHANTOM_CONTEXT_MANIFEST* manifest = Get_phantom_context_manifest();
  if (manifest == nullptr || manifest->_data_q_bit_sizes == nullptr ||
      manifest->_special_p_bit_sizes == nullptr) {
    throw std::runtime_error("compiler context manifest is incomplete");
  }
  std::vector<int> bit_sizes(manifest->_data_q_bit_sizes,
                             manifest->_data_q_bit_sizes +
                                 manifest->_data_q_count);
  bit_sizes.insert(bit_sizes.end(), manifest->_special_p_bit_sizes,
                   manifest->_special_p_bit_sizes +
                       manifest->_special_p_count);
  const std::vector<phantom::arith::Modulus> moduli =
      phantom::arith::CoeffModulus::Create(manifest->_poly_degree, bit_sizes);
  return moduli.size() == bit_sizes.size() ? 0 : 1;
}
