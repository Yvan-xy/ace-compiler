#include "phantom.h"

#include <cstddef>
#include <vector>

int main() {
  const std::vector<phantom::arith::Modulus> moduli =
      phantom::arith::CoeffModulus::Create(8192, {50, 40, 50});
  return moduli.size() == std::size_t{3} ? 0 : 1;
}
