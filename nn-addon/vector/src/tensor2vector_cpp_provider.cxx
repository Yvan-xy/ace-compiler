//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "nn/vector/tensor2vector_planning.h"
#include "nn/vector/tensor2vector_cpp_conv_provider.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <locale>
#include <sstream>
#include <type_traits>

#include "nn/vector/vector_utils.h"

namespace nn {
namespace vector {

namespace {

using air::base::PRIMITIVE_TYPE;

bool Host_is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

std::optional<size_t> Primitive_width(PRIMITIVE_TYPE type) {
  switch (type) {
    case PRIMITIVE_TYPE::BOOL:
    case PRIMITIVE_TYPE::INT_S8:
    case PRIMITIVE_TYPE::INT_U8:
      return 1;
    case PRIMITIVE_TYPE::INT_S16:
    case PRIMITIVE_TYPE::INT_U16:
      return 2;
    case PRIMITIVE_TYPE::INT_S32:
    case PRIMITIVE_TYPE::INT_U32:
    case PRIMITIVE_TYPE::FLOAT_32:
      return 4;
    case PRIMITIVE_TYPE::INT_S64:
    case PRIMITIVE_TYPE::INT_U64:
    case PRIMITIVE_TYPE::FLOAT_64:
    case PRIMITIVE_TYPE::COMPLEX_32:
      return 8;
    case PRIMITIVE_TYPE::COMPLEX_64:
      return 16;
    default:
      return std::nullopt;
  }
}

bool Checked_mul(int64_t lhs, int64_t rhs, int64_t* result) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool Checked_add(int64_t lhs, int64_t rhs, int64_t* result) {
  if (lhs < 0 || rhs < 0 ||
      lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return false;
  }
  *result = lhs + rhs;
  return true;
}

bool Product(const std::vector<int64_t>& shape, int64_t* result) {
  int64_t count = 1;
  for (int64_t dimension : shape) {
    if (dimension <= 0 || !Checked_mul(count, dimension, &count)) return false;
  }
  *result = count;
  return true;
}

uint32_t Ceil_log2(int64_t value) {
  uint32_t result = 0;
  int64_t current = 1;
  while (current < value) {
    current <<= 1;
    ++result;
  }
  return result;
}

bool Is_power_of_two_i64(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int64_t Next_power_of_two_checked(int64_t value) {
  if (value <= 1) return 1;
  int64_t result = 1;
  while (result < value) {
    if (result > std::numeric_limits<int64_t>::max() / 2) return 0;
    result *= 2;
  }
  return result;
}

std::string Payload_hash(const VECTOR_KERNEL_RANKED_TYPE_PLAN& type,
                         const std::vector<uint8_t>& bytes) {
  std::ostringstream header;
  header.imbue(std::locale::classic());
  header << "vector-kernel-constant:v1\n"
         << "element=" << Vector_kernel_primitive_type_name(type._element_type)
         << '\n'
         << "rank=" << type._shape.size() << '\n'
         << "shape=";
  for (size_t idx = 0; idx < type._shape.size(); ++idx) {
    if (idx != 0) header << ',';
    header << type._shape[idx];
  }
  header << "\npayload:\n";
  std::string preimage = header.str();
  if (!bytes.empty()) {
    preimage.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  return "sha256:" + Vector_kernel_sha256(preimage);
}

template <typename T>
std::vector<uint8_t> To_little_endian_bytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  for (size_t idx = 0; idx < values.size(); ++idx) {
    std::array<uint8_t, sizeof(T)> element{};
    std::memcpy(element.data(), &values[idx], sizeof(T));
    if (!Host_is_little_endian()) std::reverse(element.begin(), element.end());
    std::copy(element.begin(), element.end(),
              bytes.begin() + idx * sizeof(T));
  }
  return bytes;
}

template <typename T>
bool From_little_endian_bytes(const std::vector<uint8_t>& bytes,
                              std::vector<T>* values) {
  if (bytes.size() % sizeof(T) != 0) return false;
  values->resize(bytes.size() / sizeof(T));
  for (size_t idx = 0; idx < values->size(); ++idx) {
    std::array<uint8_t, sizeof(T)> element{};
    std::copy(bytes.begin() + idx * sizeof(T),
              bytes.begin() + (idx + 1) * sizeof(T), element.begin());
    if (!Host_is_little_endian()) std::reverse(element.begin(), element.end());
    std::memcpy(&(*values)[idx], element.data(), sizeof(T));
  }
  return true;
}

VECTOR_KERNEL_TYPED_PAYLOAD Make_payload(
    const std::string& role, PRIMITIVE_TYPE element_type,
    std::vector<int64_t> shape, std::vector<uint8_t> bytes) {
  VECTOR_KERNEL_RANKED_TYPE_PLAN type{element_type, std::move(shape)};
  const std::string hash = Payload_hash(type, bytes);
  return VECTOR_KERNEL_TYPED_PAYLOAD{role, std::move(type), std::move(bytes),
                                     hash};
}

VECTOR_KERNEL_CONSTANT_PLAN Descriptor(
    const VECTOR_KERNEL_TYPED_PAYLOAD& payload) {
  return VECTOR_KERNEL_CONSTANT_PLAN{payload._role, payload._type,
                                     payload._content_hash};
}

const VECTOR_KERNEL_TYPED_PAYLOAD* Find_source(
    const VECTOR_KERNEL_PLANNING_REQUEST& request, const std::string& role) {
  for (const VECTOR_KERNEL_TYPED_PAYLOAD& payload :
       request._source_constants) {
    if (payload._role == role) return &payload;
  }
  return nullptr;
}

const VECTOR_KERNEL_ATTRIBUTE_RECORD* Find_attribute(
    const VECTOR_KERNEL_PLANNING_REQUEST& request, const std::string& name) {
  for (const VECTOR_KERNEL_ATTRIBUTE_RECORD& attribute : request._attributes) {
    if (attribute._name == name) return &attribute;
  }
  return nullptr;
}

bool Attribute_i64(const VECTOR_KERNEL_PLANNING_REQUEST& request,
                   const std::string& name, std::vector<int64_t>* values) {
  const VECTOR_KERNEL_ATTRIBUTE_RECORD* attribute =
      Find_attribute(request, name);
  if (attribute == nullptr) return false;
  if (attribute->_type._element_type == PRIMITIVE_TYPE::INT_S64) {
    return From_little_endian_bytes(attribute->_bytes, values);
  }
  if (attribute->_type._element_type == PRIMITIVE_TYPE::INT_S32) {
    std::vector<int32_t> narrow;
    if (!From_little_endian_bytes(attribute->_bytes, &narrow)) return false;
    values->assign(narrow.begin(), narrow.end());
    return true;
  }
  return false;
}

std::vector<int64_t> Attribute_or(
    const VECTOR_KERNEL_PLANNING_REQUEST& request, const std::string& name,
    std::vector<int64_t> fallback) {
  std::vector<int64_t> value;
  return Attribute_i64(request, name, &value) ? value : std::move(fallback);
}

bool Request_mask_enabled(const VECTOR_KERNEL_PLANNING_REQUEST& request) {
  if (!request._options._mask_fuse) return true;
  return Find_attribute(request, "mask") != nullptr;
}

std::vector<int32_t> Duplication_rotations(int64_t replications,
                                           int64_t input_size) {
  std::vector<int32_t> rotations;
  for (int64_t idx = 1; idx < replications; ++idx) {
    const int64_t value = -idx * input_size;
    if (value < std::numeric_limits<int32_t>::min()) return {};
    rotations.push_back(static_cast<int32_t>(value));
  }
  return rotations;
}

VECTOR_KERNEL_PROVIDER_CALL_RESULT Provider_failure(std::string message) {
  return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Failure(std::move(message));
}

bool Validate_gemm_request(const VECTOR_KERNEL_PLANNING_REQUEST& request,
                           const VECTOR_KERNEL_TYPED_PAYLOAD** weight,
                           const VECTOR_KERNEL_TYPED_PAYLOAD** bias,
                           std::vector<float>* weight_values,
                           std::string* diagnostic) {
  if (request._operand_types.size() != 3) {
    *diagnostic = "Gemm planning requires exactly three operands";
    return false;
  }
  *weight = Find_source(request, "weight");
  *bias = Find_source(request, "bias");
  if (*weight == nullptr || *bias == nullptr ||
      (*weight)->_type._element_type != PRIMITIVE_TYPE::FLOAT_32 ||
      (*bias)->_type._element_type != PRIMITIVE_TYPE::FLOAT_32 ||
      (*weight)->_type._shape.size() != 2 ||
      (*bias)->_type._shape.size() != 1 ||
      !From_little_endian_bytes((*weight)->_bytes, weight_values)) {
    *diagnostic = "Gemm planning requires owned f32 weight/bias payloads";
    return false;
  }
  int64_t weight_count = 0;
  if (!Product((*weight)->_type._shape, &weight_count) ||
      static_cast<uint64_t>(weight_count) != weight_values->size()) {
    *diagnostic = "Gemm weight payload does not match its shape";
    return false;
  }
  return true;
}

VECTOR_KERNEL_PROVIDER_CALL_RESULT Build_baseline_gemm(
    const VECTOR_KERNEL_PLANNING_REQUEST& request) {
  const VECTOR_KERNEL_TYPED_PAYLOAD* source_weight = nullptr;
  const VECTOR_KERNEL_TYPED_PAYLOAD* source_bias = nullptr;
  std::vector<float> weight_values;
  std::string diagnostic;
  if (!Validate_gemm_request(request, &source_weight, &source_bias,
                             &weight_values, &diagnostic)) {
    return Provider_failure(std::move(diagnostic));
  }

  int64_t height = source_weight->_type._shape[0];
  const int64_t original_height = height;
  const int64_t width = source_weight->_type._shape[1];
  const int64_t slots = request._target._num_slots;
  if (height <= 0 || width <= 0 || height > std::numeric_limits<int>::max() ||
      width > std::numeric_limits<int>::max() || slots <= 0) {
    return Provider_failure("baseline Gemm dimensions are out of range");
  }

  std::vector<std::vector<float>> matrix(
      static_cast<size_t>(height),
      std::vector<float>(static_cast<size_t>(width)));
  for (int64_t row = 0; row < height; ++row) {
    std::copy(weight_values.begin() + row * width,
              weight_values.begin() + (row + 1) * width,
              matrix[static_cast<size_t>(row)].begin());
  }

  if (width == slots || width == slots / 2) {
    int64_t padded_height = height;
    while (padded_height <= width && width % padded_height != 0) {
      ++padded_height;
    }
    if (padded_height > width) {
      return Provider_failure("baseline Gemm height padding has no divisor");
    }
    matrix.resize(static_cast<size_t>(padded_height),
                  std::vector<float>(static_cast<size_t>(width), 0.0F));
    height = padded_height;
  }

  int64_t padded_width = width;
  int64_t search_limit = 0;
  if (!Checked_mul(height, width, &search_limit)) {
    return Provider_failure("baseline Gemm width search overflows");
  }
  while (padded_width <= search_limit && padded_width % height != 0) {
    ++padded_width;
  }
  if (padded_width > search_limit) {
    return Provider_failure("baseline Gemm width padding has no solution");
  }

  std::vector<float> diagonal;
  int64_t diagonal_count = 0;
  if (!Checked_mul(height, padded_width, &diagonal_count) ||
      diagonal_count > std::numeric_limits<int32_t>::max()) {
    return Provider_failure("baseline Gemm diagonal payload is too large");
  }
  diagonal.reserve(static_cast<size_t>(diagonal_count));
  for (int64_t position = 0; position < height; ++position) {
    for (int64_t idx = 0; idx < padded_width; ++idx) {
      const int64_t row = idx % height;
      const int64_t column = (position + idx) % padded_width;
      diagonal.push_back(column < width
                             ? matrix[static_cast<size_t>(row)]
                                     [static_cast<size_t>(column)]
                             : 0.0F);
    }
  }

  const int64_t input_duplications = padded_width == slots ? 1 : 2;
  int64_t result_width = 0;
  if (!Checked_mul(input_duplications, padded_width, &result_width)) {
    return Provider_failure("baseline Gemm result width overflows");
  }
  const VECTOR_KERNEL_RANKED_TYPE_PLAN& input_type =
      request._operand_types[0];

  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> payloads;
  payloads.push_back(Make_payload(
      "weight", PRIMITIVE_TYPE::FLOAT_32, {height, padded_width},
      To_little_endian_bytes(diagonal)));
  payloads.push_back(VECTOR_KERNEL_TYPED_PAYLOAD{
      "bias", source_bias->_type, source_bias->_bytes,
      Payload_hash(source_bias->_type, source_bias->_bytes)});
  const bool need_mask = Request_mask_enabled(request);
  if (need_mask) {
    payloads.push_back(Make_payload(
        "mask", PRIMITIVE_TYPE::FLOAT_32, {height},
        To_little_endian_bytes(
            std::vector<float>(static_cast<size_t>(height), 1.0F))));
  }

  std::vector<VECTOR_KERNEL_LOOP_PLAN> loops{
      VECTOR_KERNEL_LOOP_PLAN{"gemm", 0, static_cast<int32_t>(height), 1, 0}};
  std::vector<VECTOR_KERNEL_ROTATION_PLAN> rotations;
  const std::vector<int32_t> duplicate_rotations =
      Duplication_rotations(input_duplications, padded_width);
  if (!duplicate_rotations.empty()) {
    rotations.push_back(
        VECTOR_KERNEL_ROTATION_PLAN{"input-duplication", duplicate_rotations});
  }
  std::vector<int32_t> gemm_rotations;
  for (int64_t idx = 0; idx < height; ++idx) {
    gemm_rotations.push_back(static_cast<int32_t>(idx));
  }
  rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{"gemm", gemm_rotations});

  std::vector<VECTOR_KERNEL_REDUCTION_PLAN> reductions;
  const int64_t reduction_factor = padded_width / height;
  if (reduction_factor > 1) {
    const uint32_t loop_count = Ceil_log2(reduction_factor);
    loops.push_back(VECTOR_KERNEL_LOOP_PLAN{
        "block-reduction", 0, static_cast<int32_t>(loop_count), 1, 0});
    std::vector<int32_t> shifts;
    for (uint32_t idx = 0; idx < loop_count; ++idx) {
      const int64_t shift = (int64_t{1} << idx) * height;
      if (shift > std::numeric_limits<int32_t>::max()) {
        return Provider_failure("baseline Gemm reduction shift overflows s32");
      }
      shifts.push_back(static_cast<int32_t>(shift));
    }
    rotations.push_back(
        VECTOR_KERNEL_ROTATION_PLAN{"block-reduction", shifts});
    reductions.push_back(VECTOR_KERNEL_REDUCTION_PLAN{
        "block-reduction", VECTOR_KERNEL_REDUCTION_KIND::POWER_OF_TWO,
        reduction_factor, height, 0});
  }

  std::vector<VECTOR_KERNEL_CONSTANT_PLAN> constants;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD& payload : payloads) {
    constants.push_back(Descriptor(payload));
  }
  VECTOR_KERNEL_COMMON_PLAN common{
      {input_type},
      {},
      std::move(constants),
      VECTOR_KERNEL_RANKED_TYPE_PLAN{request._operand_types[0]._element_type,
                                     {result_width}},
      std::move(loops),
      {VECTOR_KERNEL_SLICE_PLAN{
          "weight", VECTOR_KERNEL_AFFINE_INDEX_PLAN{{1}, 0, false},
          padded_width}},
      std::move(rotations),
      std::move(reductions),
      VECTOR_KERNEL_MASK_PLAN{
          need_mask ? VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX
                    : VECTOR_KERNEL_MASK_POLICY::NONE,
          need_mask ? height : 0},
      VECTOR_KERNEL_SLOT_PLAN{
          VECTOR_KERNEL_SLOT_POLICY::ABSENT_NATIVE_BASELINE, 0}};
  VECTOR_KERNEL_PLAN plan = BASELINE_GEMM_PLAN{
      std::move(common), height, padded_width, input_duplications};

  std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION> runtime{
      VECTOR_KERNEL_RUNTIME_PREPARATION{
          "input", 0,
          VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::PACKED_VECTOR, input_type,
          padded_width, input_duplications, 0, duplicate_rotations, 0}};
  (void)original_height;
  return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Success(
      VECTOR_KERNEL_PROVIDER_RESULT{std::move(plan), std::move(payloads),
                                    std::move(runtime), {}, "cpp"});
}

bool Compute_pad_gemm_checked(int64_t* n, int64_t* k, int64_t slots,
                              std::string* diagnostic) {
  int64_t value_n = *n;
  int64_t value_k = *k;
  const int64_t original_n = value_n;
  const int64_t original_k = value_k;
  if (value_n <= 0 || value_k <= 0 || slots <= 0) {
    *diagnostic = "fast Gemm padding requires positive dimensions";
    return false;
  }
  if (value_n > value_k) std::swap(value_n, value_k);
  if (Is_power_of_two_i64(value_k) && !Is_power_of_two_i64(value_n)) {
    const int64_t limit = Next_power_of_two_checked(value_n);
    if (limit == 0) {
      *diagnostic = "fast Gemm next-power-of-two overflows";
      return false;
    }
    int64_t padded = value_n;
    while (padded <= limit && value_k % padded != 0) ++padded;
    if (padded <= limit) value_n = padded;
  }
  if (value_k == slots || value_k == slots / 2) {
    int64_t limit = 0;
    if (!Checked_mul(value_n, value_k, &limit)) {
      *diagnostic = "fast Gemm divisor search overflows";
      return false;
    }
    int64_t padded = value_n;
    while (padded <= limit && value_k % padded != 0) ++padded;
    if (padded > limit) {
      *diagnostic = "fast Gemm divisor search failed";
      return false;
    }
    value_n = padded;
  }
  const int64_t increase_n = (value_k - value_n % value_k) % value_k;
  const int64_t increase_k = (value_n - value_k % value_n) % value_n;
  if (increase_n <= increase_k) {
    if (!Checked_add(value_n, increase_n, &value_n)) return false;
  } else if (!Checked_add(value_k, increase_k, &value_k)) {
    return false;
  }
  if (original_n > original_k) std::swap(value_n, value_k);
  if (value_k > slots / 2 && value_k < slots) {
    value_k = slots;
    while (value_n <= slots && value_n % value_k != 0 &&
           value_k % value_n != 0) {
      ++value_n;
    }
  }
  if (!((value_k <= slots / 2) || value_k == slots) ||
      !((value_n % value_k == 0) || (value_k % value_n == 0))) {
    *diagnostic = "fast Gemm padded shape violates divisibility/slot rules";
    return false;
  }
  *n = value_n;
  *k = value_k;
  return true;
}

struct IMRA_COST {
  int64_t _block_size = 1;
  int64_t _blocks_per_partition = 1;
  int64_t _packed_partitions = 1;
  int64_t _cost = 0;
};

bool Compute_imra_cost(int64_t nd, int64_t kd, int64_t slots,
                       IMRA_COST* result, std::string* diagnostic) {
  int64_t min_cost = 0;
  if (!Checked_mul(nd, kd, &min_cost)) {
    *diagnostic = "IMRA initial cost overflows";
    return false;
  }
  result->_blocks_per_partition = nd;
  for (int64_t block_size = 1; block_size <= nd; ++block_size) {
    if (nd % block_size != 0) continue;
    const int64_t blocks = nd / block_size;
    for (int64_t packed = 1; packed <= blocks; ++packed) {
      if (blocks % packed != 0) continue;
      int64_t packed_width = 0;
      int64_t required = 0;
      if (!Checked_mul(kd, packed, &packed_width) ||
          !Checked_add(packed_width, nd, &required)) {
        *diagnostic = "IMRA capacity calculation overflows";
        return false;
      }
      if ((required > slots && kd != slots) ||
          (kd == slots && packed > 1)) {
        continue;
      }
      const int64_t replications = (required + kd - 1) / kd;
      const int64_t cost = static_cast<int64_t>(Ceil_log2(replications)) +
                           (block_size - 1) + (blocks / packed - 1) +
                           (packed - 1);
      if (cost < min_cost ||
          (cost == min_cost && packed > result->_packed_partitions)) {
        min_cost = cost;
        result->_block_size = block_size;
        result->_blocks_per_partition = blocks;
        result->_packed_partitions = packed;
      }
    }
  }
  result->_cost = min_cost;
  return true;
}

VECTOR_KERNEL_PROVIDER_CALL_RESULT Build_fast_gemm(
    const VECTOR_KERNEL_PLANNING_REQUEST& request) {
  const VECTOR_KERNEL_TYPED_PAYLOAD* source_weight = nullptr;
  const VECTOR_KERNEL_TYPED_PAYLOAD* source_bias = nullptr;
  std::vector<float> weight_values;
  std::string diagnostic;
  if (!Validate_gemm_request(request, &source_weight, &source_bias,
                             &weight_values, &diagnostic)) {
    return Provider_failure(std::move(diagnostic));
  }
  const int64_t n = source_weight->_type._shape[0];
  const int64_t k = source_weight->_type._shape[1];
  int64_t np = n;
  int64_t kp = k;
  if (!Compute_pad_gemm_checked(&np, &kp, request._target._num_slots,
                                &diagnostic)) {
    return Provider_failure(std::move(diagnostic));
  }
  const int64_t nd = std::min(np, kp);
  const int64_t kd = std::max(np, kp);
  if (np > std::numeric_limits<int32_t>::max() ||
      kp > std::numeric_limits<int32_t>::max()) {
    return Provider_failure("fast Gemm padded dimensions exceed s32");
  }

  int64_t padded_count = 0;
  if (!Checked_mul(np, kp, &padded_count) ||
      static_cast<uint64_t>(padded_count) >
          std::numeric_limits<size_t>::max()) {
    return Provider_failure("fast Gemm padded weight size overflows");
  }
  std::vector<std::vector<float>> padded(
      static_cast<size_t>(np),
      std::vector<float>(static_cast<size_t>(kp), 0.0F));
  for (int64_t row = 0; row < n; ++row) {
    std::copy(weight_values.begin() + row * k,
              weight_values.begin() + (row + 1) * k,
              padded[static_cast<size_t>(row)].begin());
  }

  std::vector<std::vector<float>> irma(
      static_cast<size_t>(nd),
      std::vector<float>(static_cast<size_t>(kd), 0.0F));
  for (int64_t height = 0; height < nd; ++height) {
    int64_t row = (np - height) % np;
    int64_t column = 0;
    for (int64_t width = 0; width < kd; ++width) {
      irma[static_cast<size_t>(height)][static_cast<size_t>(width)] =
          padded[static_cast<size_t>(row)][static_cast<size_t>(column)];
      row = (row + 1) % np;
      column = (column + 1) % kp;
    }
  }

  IMRA_COST cost;
  if (!Compute_imra_cost(nd, kd, request._target._num_slots, &cost,
                         &diagnostic)) {
    return Provider_failure(std::move(diagnostic));
  }
  const int64_t shift = 1;
  const int64_t shift_buffer = (nd / cost._packed_partitions) * shift;
  int64_t packed_row_width = 0;
  const int64_t actual_shift_buffer =
      (cost._packed_partitions == 1 && kd == request._target._num_slots)
          ? 0
          : shift_buffer;
  if (!Checked_add(kd, actual_shift_buffer, &packed_row_width) ||
      !Checked_mul(cost._packed_partitions, packed_row_width,
                   &packed_row_width) ||
      packed_row_width > request._target._num_slots) {
    return Provider_failure("fast Gemm packed width exceeds slots");
  }
  const int64_t grid_size =
      cost._blocks_per_partition / cost._packed_partitions;

  std::vector<float> packed_values;
  for (int64_t grid = 0; grid < grid_size; ++grid) {
    for (int64_t block = 0; block < cost._block_size; ++block) {
      const int64_t base_row = grid * cost._block_size + block;
      for (int64_t partition = 0;
           partition < cost._packed_partitions; ++partition) {
        const int64_t row =
            base_row + partition * (nd / cost._packed_partitions);
        std::vector<float> rotated = irma[static_cast<size_t>(row)];
        const int64_t rotate_by =
            (row - base_row) * shift + (block % cost._block_size);
        if (rotate_by < 0 || rotate_by > kd) {
          return Provider_failure("fast Gemm IRMA rotation is out of range");
        }
        std::rotate(rotated.begin(), rotated.begin() + rotate_by,
                    rotated.end());
        rotated.insert(rotated.end(), rotated.begin(),
                       rotated.begin() + actual_shift_buffer);
        packed_values.insert(packed_values.end(), rotated.begin(),
                             rotated.end());
      }
    }
  }
  int64_t packed_height = 0;
  if (!Checked_mul(grid_size, cost._block_size, &packed_height)) {
    return Provider_failure("fast Gemm packed height overflows");
  }
  int64_t packed_count_expected = 0;
  if (!Checked_mul(packed_height, packed_row_width,
                   &packed_count_expected) ||
      static_cast<uint64_t>(packed_count_expected) != packed_values.size()) {
    return Provider_failure("fast Gemm packed payload size mismatch");
  }

  const VECTOR_KERNEL_RANKED_TYPE_PLAN& input_type =
      request._operand_types[0];
  const int64_t replications =
      kp == request._target._num_slots
          ? 1
          : (cost._packed_partitions *
                     (kd + actual_shift_buffer) +
                 k - 1) /
                k;
  std::vector<int32_t> alignments;
  for (int64_t idx = 0; idx < cost._block_size; ++idx) {
    alignments.push_back(static_cast<int32_t>(idx));
  }

  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> payloads;
  payloads.push_back(Make_payload(
      "weight", PRIMITIVE_TYPE::FLOAT_32,
      {packed_height, packed_row_width},
      To_little_endian_bytes(packed_values)));
  payloads.push_back(VECTOR_KERNEL_TYPED_PAYLOAD{
      "bias", source_bias->_type, source_bias->_bytes,
      Payload_hash(source_bias->_type, source_bias->_bytes)});
  payloads.push_back(Make_payload(
      "rotation-table", PRIMITIVE_TYPE::INT_S32,
      {static_cast<int64_t>(alignments.size())},
      To_little_endian_bytes(alignments)));
  const bool clear_mask =
      !request._options._mask_fuse ||
      (!request._options._is_last_operation &&
       n != request._target._num_slots);
  if (clear_mask) {
    payloads.push_back(Make_payload(
        "mask", PRIMITIVE_TYPE::FLOAT_32, {n},
        To_little_endian_bytes(
            std::vector<float>(static_cast<size_t>(n), 1.0F))));
  }

  std::vector<VECTOR_KERNEL_ROTATION_PLAN> rotations;
  const std::vector<int32_t> duplicate_rotations =
      Duplication_rotations(replications, kp);
  if (!duplicate_rotations.empty()) {
    rotations.push_back(
        VECTOR_KERNEL_ROTATION_PLAN{"input-duplication", duplicate_rotations});
  }
  rotations.push_back(
      VECTOR_KERNEL_ROTATION_PLAN{"blocking-alignment", alignments});
  std::vector<int32_t> grid_rotations;
  for (int64_t idx = 0; idx < grid_size; ++idx) {
    grid_rotations.push_back(
        static_cast<int32_t>(idx * cost._block_size));
  }
  rotations.push_back(
      VECTOR_KERNEL_ROTATION_PLAN{"grid", grid_rotations});

  std::vector<VECTOR_KERNEL_REDUCTION_PLAN> reductions;
  auto append_reduction = [&](const std::string& role, int64_t factor,
                              int64_t block_width, int64_t padding) {
    if (factor <= 1) return;
    std::vector<int32_t> shifts;
    if (Is_power_of_two_i64(factor)) {
      for (uint32_t idx = 0; idx < Ceil_log2(factor); ++idx) {
        shifts.push_back(static_cast<int32_t>(
            (int64_t{1} << idx) * (block_width + padding)));
      }
    } else {
      for (int64_t idx = 1; idx < factor; ++idx) {
        shifts.push_back(static_cast<int32_t>(
            idx * (block_width + padding)));
      }
    }
    rotations.push_back(VECTOR_KERNEL_ROTATION_PLAN{role, shifts});
    reductions.push_back(VECTOR_KERNEL_REDUCTION_PLAN{
        role,
        Is_power_of_two_i64(factor)
            ? VECTOR_KERNEL_REDUCTION_KIND::POWER_OF_TWO
            : VECTOR_KERNEL_REDUCTION_KIND::LINEAR,
        factor, block_width, padding});
  };
  append_reduction("packed-partitions", cost._packed_partitions, kd,
                   actual_shift_buffer);
  append_reduction("kp-over-np", kp / np, np, 0);

  std::vector<VECTOR_KERNEL_CONSTANT_PLAN> constants;
  for (const VECTOR_KERNEL_TYPED_PAYLOAD& payload : payloads) {
    constants.push_back(Descriptor(payload));
  }
  VECTOR_KERNEL_COMMON_PLAN common{
      {input_type},
      {},
      std::move(constants),
      VECTOR_KERNEL_RANKED_TYPE_PLAN{request._operand_types[0]._element_type,
                                     {packed_row_width}},
      {VECTOR_KERNEL_LOOP_PLAN{
           "grid", 0, static_cast<int32_t>(grid_size), 1, 0},
       VECTOR_KERNEL_LOOP_PLAN{
           "block", 0, static_cast<int32_t>(cost._block_size), 1, 1}},
      {VECTOR_KERNEL_SLICE_PLAN{
          "weight",
          VECTOR_KERNEL_AFFINE_INDEX_PLAN{{cost._block_size, 1}, 0, false},
          packed_row_width}},
      std::move(rotations),
      std::move(reductions),
      VECTOR_KERNEL_MASK_PLAN{
          clear_mask ? VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX
                     : VECTOR_KERNEL_MASK_POLICY::NONE,
          clear_mask ? n : 0},
      VECTOR_KERNEL_SLOT_PLAN{
          VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS,
          static_cast<uint32_t>(n)}};
  VECTOR_KERNEL_PLAN plan = FAST_GEMM_PLAN{
      std::move(common),
      n,
      k,
      np,
      kp,
      nd,
      kd,
      cost._block_size,
      cost._blocks_per_partition,
      cost._packed_partitions,
      shift,
      actual_shift_buffer,
      grid_size,
      replications};
  std::vector<VECTOR_KERNEL_RUNTIME_PREPARATION> runtime{
      VECTOR_KERNEL_RUNTIME_PREPARATION{
          "input", 0,
          VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::BLOCKING_ROTATIONS,
          input_type, kp, replications, cost._block_size, alignments, 0}};
  return VECTOR_KERNEL_PROVIDER_CALL_RESULT::Success(
      VECTOR_KERNEL_PROVIDER_RESULT{std::move(plan), std::move(payloads),
                                    std::move(runtime), {}, "cpp"});
}

}  // namespace

VECTOR_KERNEL_PROVIDER_CALL_RESULT CPP_VECTOR_KERNEL_PLAN_PROVIDER::Plan(
    const VECTOR_KERNEL_PLANNING_REQUEST& request) const {
  if (request._operation == VECTOR_KERNEL_OPERATION::GEMM) {
    if (request._requested_plan_kind ==
            VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV ||
        request._requested_plan_kind ==
            VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV) {
      return Provider_failure("Conv plan kind cannot plan a Gemm operation");
    }
    if (request._requested_plan_kind ==
        VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM) {
      return Build_baseline_gemm(request);
    }
    return Build_fast_gemm(request);
  }
  return Plan_cpp_vector_kernel_conv(request);
}

}  // namespace vector
}  // namespace nn
