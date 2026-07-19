//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "nn/vector/tensor2vector_plan.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <type_traits>

#include "air/base/st.h"

namespace nn {
namespace vector {

namespace {

constexpr uint32_t SHA256_K[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

constexpr uint32_t Rotate_right(uint32_t value, uint32_t amount) {
  return (value >> amount) | (value << (32U - amount));
}

class SHA256 {
public:
  SHA256()
      : _state{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

  void Update(const uint8_t* data, size_t size) {
    AIR_ASSERT(data != nullptr || size == 0);
    AIR_ASSERT(_total_bytes <= std::numeric_limits<uint64_t>::max() - size);
    _total_bytes += size;
    while (size != 0) {
      const size_t copy_size = std::min(size, _block.size() - _block_size);
      std::memcpy(_block.data() + _block_size, data, copy_size);
      _block_size += copy_size;
      data += copy_size;
      size -= copy_size;
      if (_block_size == _block.size()) {
        Transform(_block.data());
        _block_size = 0;
      }
    }
  }

  std::string Final_hex() {
    AIR_ASSERT(_total_bytes <= std::numeric_limits<uint64_t>::max() / 8U);
    const uint64_t bit_count = _total_bytes * 8U;
    _block[_block_size++]    = 0x80U;
    if (_block_size > 56) {
      std::fill(_block.begin() + _block_size, _block.end(), 0U);
      Transform(_block.data());
      _block_size = 0;
    }
    std::fill(_block.begin() + _block_size, _block.begin() + 56, 0U);
    for (uint32_t idx = 0; idx < 8; ++idx) {
      _block[56 + idx] =
          static_cast<uint8_t>(bit_count >> (56U - idx * 8U));
    }
    Transform(_block.data());

    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << std::hex << std::setfill('0');
    for (uint32_t word : _state) os << std::setw(8) << word;
    return os.str();
  }

private:
  void Transform(const uint8_t* block) {
    uint32_t words[64];
    for (uint32_t idx = 0; idx < 16; ++idx) {
      words[idx] = (static_cast<uint32_t>(block[idx * 4]) << 24U) |
                   (static_cast<uint32_t>(block[idx * 4 + 1]) << 16U) |
                   (static_cast<uint32_t>(block[idx * 4 + 2]) << 8U) |
                   static_cast<uint32_t>(block[idx * 4 + 3]);
    }
    for (uint32_t idx = 16; idx < 64; ++idx) {
      const uint32_t s0 = Rotate_right(words[idx - 15], 7U) ^
                          Rotate_right(words[idx - 15], 18U) ^
                          (words[idx - 15] >> 3U);
      const uint32_t s1 = Rotate_right(words[idx - 2], 17U) ^
                          Rotate_right(words[idx - 2], 19U) ^
                          (words[idx - 2] >> 10U);
      words[idx] = words[idx - 16] + s0 + words[idx - 7] + s1;
    }

    uint32_t a = _state[0];
    uint32_t b = _state[1];
    uint32_t c = _state[2];
    uint32_t d = _state[3];
    uint32_t e = _state[4];
    uint32_t f = _state[5];
    uint32_t g = _state[6];
    uint32_t h = _state[7];
    for (uint32_t idx = 0; idx < 64; ++idx) {
      const uint32_t sum1 = Rotate_right(e, 6U) ^ Rotate_right(e, 11U) ^
                            Rotate_right(e, 25U);
      const uint32_t choose = (e & f) ^ (~e & g);
      const uint32_t temp1 = h + sum1 + choose + SHA256_K[idx] + words[idx];
      const uint32_t sum0 = Rotate_right(a, 2U) ^ Rotate_right(a, 13U) ^
                            Rotate_right(a, 22U);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2    = sum0 + majority;

      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    _state[0] += a;
    _state[1] += b;
    _state[2] += c;
    _state[3] += d;
    _state[4] += e;
    _state[5] += f;
    _state[6] += g;
    _state[7] += h;
  }

  std::array<uint32_t, 8> _state;
  std::array<uint8_t, 64> _block{};
  size_t                  _block_size  = 0;
  uint64_t                _total_bytes = 0;
};

const char* Primitive_type_name(air::base::PRIMITIVE_TYPE type) {
  using air::base::PRIMITIVE_TYPE;
  switch (type) {
    case PRIMITIVE_TYPE::INT_S8:
      return "s8";
    case PRIMITIVE_TYPE::INT_S16:
      return "s16";
    case PRIMITIVE_TYPE::INT_S32:
      return "s32";
    case PRIMITIVE_TYPE::INT_S64:
      return "s64";
    case PRIMITIVE_TYPE::INT_U8:
      return "u8";
    case PRIMITIVE_TYPE::INT_U16:
      return "u16";
    case PRIMITIVE_TYPE::INT_U32:
      return "u32";
    case PRIMITIVE_TYPE::INT_U64:
      return "u64";
    case PRIMITIVE_TYPE::FLOAT_32:
      return "f32";
    case PRIMITIVE_TYPE::FLOAT_64:
      return "f64";
    case PRIMITIVE_TYPE::FLOAT_80:
      return "f80";
    case PRIMITIVE_TYPE::FLOAT_128:
      return "f128";
    case PRIMITIVE_TYPE::COMPLEX_32:
      return "c32";
    case PRIMITIVE_TYPE::COMPLEX_64:
      return "c64";
    case PRIMITIVE_TYPE::COMPLEX_80:
      return "c80";
    case PRIMITIVE_TYPE::COMPLEX_128:
      return "c128";
    case PRIMITIVE_TYPE::VOID:
      return "void";
    case PRIMITIVE_TYPE::BOOL:
      return "bool";
    case PRIMITIVE_TYPE::END:
      return "end";
  }
  return "invalid";
}

size_t Primitive_element_width(air::base::PRIMITIVE_TYPE type) {
  using air::base::PRIMITIVE_TYPE;
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
    case PRIMITIVE_TYPE::FLOAT_80:
    case PRIMITIVE_TYPE::FLOAT_128:
    case PRIMITIVE_TYPE::COMPLEX_80:
    case PRIMITIVE_TYPE::COMPLEX_128:
    case PRIMITIVE_TYPE::VOID:
    case PRIMITIVE_TYPE::END:
      AIR_ASSERT_MSG(false, "unsupported canonical constant element type");
      return 0;
  }
  AIR_ASSERT_MSG(false, "invalid canonical constant element type");
  return 0;
}

size_t Primitive_component_width(air::base::PRIMITIVE_TYPE type) {
  using air::base::PRIMITIVE_TYPE;
  switch (type) {
    case PRIMITIVE_TYPE::COMPLEX_32:
      return 4;
    case PRIMITIVE_TYPE::COMPLEX_64:
      return 8;
    default:
      return Primitive_element_width(type);
  }
}

bool Host_is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

const char* Helper_kind_token(VECTOR_KERNEL_PLAN_KIND kind) {
  switch (kind) {
    case VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM:
      return "baseline_gemm";
    case VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV:
      return "baseline_conv";
    case VECTOR_KERNEL_PLAN_KIND::FAST_GEMM:
      return "fast_gemm";
    case VECTOR_KERNEL_PLAN_KIND::FAST_CONV:
      return "fast_conv";
  }
  return "invalid";
}

const char* Mask_policy_name(VECTOR_KERNEL_MASK_POLICY policy) {
  switch (policy) {
    case VECTOR_KERNEL_MASK_POLICY::NONE:
      return "none";
    case VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX:
      return "clear-valid-prefix";
    case VECTOR_KERNEL_MASK_POLICY::COLLECTIVE_REDUCTION:
      return "collective-reduction";
  }
  return "invalid";
}

const char* Slot_policy_name(VECTOR_KERNEL_SLOT_POLICY policy) {
  switch (policy) {
    case VECTOR_KERNEL_SLOT_POLICY::ABSENT_NATIVE_BASELINE:
      return "absent-native-baseline";
    case VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS:
      return "logical-output-elements";
    case VECTOR_KERNEL_SLOT_POLICY::EXPLICIT:
      return "explicit";
  }
  return "invalid";
}

const char* Reduction_kind_name(VECTOR_KERNEL_REDUCTION_KIND kind) {
  switch (kind) {
    case VECTOR_KERNEL_REDUCTION_KIND::POWER_OF_TWO:
      return "power-of-two";
    case VECTOR_KERNEL_REDUCTION_KIND::LINEAR:
      return "linear";
    case VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_SINGLE_BLOCK:
      return "collective-single-block";
    case VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_BLOCKS:
      return "collective-blocks";
  }
  return "invalid";
}

void Append_string(std::ostream& os, const std::string& value) {
  os << value.size() << ':' << value;
}

template <typename T>
void Append_integer_vector(std::ostream& os, const std::vector<T>& values) {
  os << '[';
  for (size_t idx = 0; idx < values.size(); ++idx) {
    if (idx != 0) os << ',';
    os << static_cast<int64_t>(values[idx]);
  }
  os << ']';
}

void Append_ranked_type(std::ostream&                         os,
                        const VECTOR_KERNEL_RANKED_TYPE_PLAN& type) {
  os << Primitive_type_name(type._element_type);
  Append_integer_vector(os, type._shape);
}

void Append_common_plan(std::ostream&                     os,
                        const VECTOR_KERNEL_COMMON_PLAN& common) {
  os << "|vector-inputs=" << common._runtime_vector_inputs.size() << '{';
  for (size_t idx = 0; idx < common._runtime_vector_inputs.size(); ++idx) {
    if (idx != 0) os << ';';
    Append_ranked_type(os, common._runtime_vector_inputs[idx]);
  }
  os << '}';

  os << "|scalar-inputs=" << common._runtime_scalar_inputs.size() << '{';
  for (size_t idx = 0; idx < common._runtime_scalar_inputs.size(); ++idx) {
    if (idx != 0) os << ';';
    os << Primitive_type_name(common._runtime_scalar_inputs[idx]);
  }
  os << '}';

  os << "|constants=" << common._constants.size() << '{';
  for (size_t idx = 0; idx < common._constants.size(); ++idx) {
    if (idx != 0) os << ';';
    const VECTOR_KERNEL_CONSTANT_PLAN& constant = common._constants[idx];
    Append_string(os, constant._role);
    os << '=';
    Append_ranked_type(os, constant._type);
    os << '#';
    Append_string(os, constant._content_hash);
  }
  os << '}';

  os << "|result=";
  Append_ranked_type(os, common._result_type);

  os << "|loops=" << common._loops.size() << '{';
  for (size_t idx = 0; idx < common._loops.size(); ++idx) {
    if (idx != 0) os << ';';
    const VECTOR_KERNEL_LOOP_PLAN& loop = common._loops[idx];
    Append_string(os, loop._role);
    os << '(' << loop._lower << ',' << loop._upper << ',' << loop._step << ','
       << loop._nesting_depth << ')';
  }
  os << '}';

  os << "|slices=" << common._slices.size() << '{';
  for (size_t idx = 0; idx < common._slices.size(); ++idx) {
    if (idx != 0) os << ';';
    const VECTOR_KERNEL_SLICE_PLAN& slice = common._slices[idx];
    Append_string(os, slice._role);
    os << '(';
    Append_integer_vector(os, slice._index._iv_coefficients);
    os << ',' << slice._index._constant << ','
       << (slice._index._uses_sharding_offset ? 1 : 0) << ',' << slice._width
       << ')';
  }
  os << '}';

  os << "|rotations=" << common._rotations.size() << '{';
  for (size_t idx = 0; idx < common._rotations.size(); ++idx) {
    if (idx != 0) os << ';';
    const VECTOR_KERNEL_ROTATION_PLAN& rotation = common._rotations[idx];
    Append_string(os, rotation._role);
    Append_integer_vector(os, rotation._candidates);
  }
  os << '}';

  os << "|reductions=" << common._reductions.size() << '{';
  for (size_t idx = 0; idx < common._reductions.size(); ++idx) {
    if (idx != 0) os << ';';
    const VECTOR_KERNEL_REDUCTION_PLAN& reduction = common._reductions[idx];
    Append_string(os, reduction._role);
    os << '(' << Reduction_kind_name(reduction._kind) << ','
       << reduction._factor << ',' << reduction._block_width << ','
       << reduction._padding << ')';
  }
  os << '}';

  os << "|mask=" << Mask_policy_name(common._mask._policy) << ':'
     << common._mask._valid_length;
  os << "|slot=" << Slot_policy_name(common._slot._policy) << ':'
     << common._slot._value;
}

}  // namespace

const char* Vector_kernel_primitive_type_name(
    air::base::PRIMITIVE_TYPE type) {
  return Primitive_type_name(type);
}

std::string Vector_kernel_sha256(std::string_view bytes) {
  SHA256 hash;
  hash.Update(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
  return hash.Final_hex();
}

std::string Build_vector_kernel_constant_hash(
    air::base::PRIMITIVE_TYPE element_type,
    const std::vector<int64_t>& shape, const void* payload,
    size_t payload_byte_count) {
  const size_t element_width   = Primitive_element_width(element_type);
  const size_t component_width = Primitive_component_width(element_type);

  size_t element_count = 1;
  for (int64_t dimension : shape) {
    AIR_ASSERT_MSG(dimension >= 0,
                   "canonical constant shape has a negative dimension");
    const size_t unsigned_dimension = static_cast<size_t>(dimension);
    AIR_ASSERT_MSG(
        unsigned_dimension == 0 ||
            element_count <=
                std::numeric_limits<size_t>::max() / unsigned_dimension,
        "canonical constant element count overflows size_t");
    element_count *= unsigned_dimension;
  }
  AIR_ASSERT_MSG(
      element_count <= std::numeric_limits<size_t>::max() / element_width,
      "canonical constant payload size overflows size_t");
  AIR_ASSERT_MSG(payload_byte_count == element_count * element_width,
                 "canonical constant payload size does not match its shape");
  AIR_ASSERT(payload != nullptr || payload_byte_count == 0);

  std::ostringstream header;
  header.imbue(std::locale::classic());
  header << "vector-kernel-constant:v1\n"
         << "element=" << Primitive_type_name(element_type) << '\n'
         << "rank=" << shape.size() << '\n'
         << "shape=";
  for (size_t idx = 0; idx < shape.size(); ++idx) {
    if (idx != 0) header << ',';
    header << shape[idx];
  }
  header << "\npayload:\n";

  std::string preimage = header.str();
  const auto* raw      = static_cast<const uint8_t*>(payload);
  if (payload_byte_count == 0) {
    // Nothing to append. Keeping this branch explicit avoids passing a null
    // pointer to std::string::append for a valid zero-element constant.
  } else if (Host_is_little_endian()) {
    preimage.append(reinterpret_cast<const char*>(raw), payload_byte_count);
  } else {
    for (size_t offset = 0; offset < payload_byte_count;
         offset += component_width) {
      for (size_t idx = 0; idx < component_width; ++idx) {
        preimage.push_back(
            static_cast<char>(raw[offset + component_width - idx - 1]));
      }
    }
  }
  return "sha256:" + Vector_kernel_sha256(preimage);
}

std::string Build_vector_kernel_constant_hash(
    air::base::CONST_CONSTANT_PTR constant) {
  AIR_ASSERT(constant != air::base::Null_ptr);
  AIR_ASSERT_MSG(constant->Kind() == air::base::CONSTANT_KIND::ARRAY,
                 "vector-kernel constant must be an AIR ARRAY");
  air::base::TYPE_PTR type = constant->Type();
  AIR_ASSERT(type->Is_array());
  air::base::ARRAY_TYPE_PTR array_type = type->Cast_to_arr();
  air::base::TYPE_PTR       element    = array_type->Elem_type();
  AIR_ASSERT_MSG(element->Is_prim(),
                 "vector-kernel ARRAY element must be primitive");
  return Build_vector_kernel_constant_hash(
      element->Cast_to_prim()->Encoding(), array_type->Shape(),
      constant->Array_buffer(), constant->Array_byte_len());
}

VECTOR_KERNEL_PLAN_KIND Get_vector_kernel_plan_kind(
    const VECTOR_KERNEL_PLAN& plan) {
  switch (plan.index()) {
    case 0:
      return VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM;
    case 1:
      return VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV;
    case 2:
      return VECTOR_KERNEL_PLAN_KIND::FAST_GEMM;
    case 3:
      return VECTOR_KERNEL_PLAN_KIND::FAST_CONV;
  }
  return VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM;
}

const char* Vector_kernel_plan_kind_name(VECTOR_KERNEL_PLAN_KIND kind) {
  switch (kind) {
    case VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM:
      return "baseline-gemm";
    case VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV:
      return "baseline-conv";
    case VECTOR_KERNEL_PLAN_KIND::FAST_GEMM:
      return "fast-gemm";
    case VECTOR_KERNEL_PLAN_KIND::FAST_CONV:
      return "fast-conv";
  }
  return "invalid";
}

std::string Build_vector_kernel_specialization_key(
    const VECTOR_KERNEL_PLAN& plan) {
  std::ostringstream os;
  os.imbue(std::locale::classic());
  os << "vector-kernel-plan:v" << VECTOR_KERNEL_PLAN_SCHEMA_VERSION
     << "|kind="
     << Vector_kernel_plan_kind_name(Get_vector_kernel_plan_kind(plan));

  std::visit(
      [&os](const auto& typed_plan) {
        using PLAN = std::decay_t<decltype(typed_plan)>;
        Append_common_plan(os, typed_plan._common);
        if constexpr (std::is_same_v<PLAN, BASELINE_GEMM_PLAN>) {
          os << "|height=" << typed_plan._height
             << "|width=" << typed_plan._width
             << "|input-duplications=" << typed_plan._input_duplications;
        } else if constexpr (std::is_same_v<PLAN, BASELINE_CONV_PLAN>) {
          os << "|channel-in=" << typed_plan._channel_in
             << "|channel-out=" << typed_plan._channel_out
             << "|output-height=" << typed_plan._output_height
             << "|output-width=" << typed_plan._output_width
             << "|kernel-hw=" << typed_plan._kernel_hw
             << "|stride=" << typed_plan._stride
             << "|input-duplications=" << typed_plan._input_duplications;
        } else if constexpr (std::is_same_v<PLAN, FAST_GEMM_PLAN>) {
          os << "|n=" << typed_plan._n << "|k=" << typed_plan._k
             << "|np=" << typed_plan._np << "|kp=" << typed_plan._kp
             << "|nd=" << typed_plan._nd << "|kd=" << typed_plan._kd
             << "|block-size=" << typed_plan._block_size
             << "|blocks-per-partition="
             << typed_plan._blocks_per_partition
             << "|packed-partitions=" << typed_plan._packed_partitions
             << "|shift=" << typed_plan._shift
             << "|shift-buffer=" << typed_plan._shift_buffer
             << "|grid-size=" << typed_plan._grid_size
             << "|input-replications=" << typed_plan._input_replications;
        } else if constexpr (std::is_same_v<PLAN, FAST_CONV_PLAN>) {
          os << "|channel-in=" << typed_plan._channel_in
             << "|channel-out=" << typed_plan._channel_out
             << "|output-height=" << typed_plan._output_height
             << "|output-width=" << typed_plan._output_width
             << "|kernel-hw=" << typed_plan._kernel_hw
             << "|group=" << typed_plan._group
             << "|stride=" << typed_plan._stride
             << "|input-size=" << typed_plan._input_size
             << "|output-size=" << typed_plan._output_size
             << "|num-slots=" << typed_plan._num_slots
             << "|num-grid=" << typed_plan._num_grid
             << "|num-block=" << typed_plan._num_block
             << "|width-block=" << typed_plan._width_block
             << "|width-block-data=" << typed_plan._width_block_data
             << "|width-block-pad=" << typed_plan._width_block_pad
             << "|position-block=" << typed_plan._position_block
             << "|capacity-block=" << typed_plan._capacity_block
             << "|input-duplications=" << typed_plan._input_duplications
             << "|blocking-outer-depth="
             << typed_plan._blocking_outer_depth
             << "|cyclic-roll=" << (typed_plan._cyclic_roll ? 1 : 0)
             << "|sharding-offset=";
          if (typed_plan._sharding_offset.has_value()) {
            os << Primitive_type_name(typed_plan._sharding_offset->_type)
               << '*' << typed_plan._sharding_offset->_scale;
          } else {
            os << "none";
          }
        }
      },
      plan);
  return os.str();
}

std::string Build_vector_kernel_helper_name(const VECTOR_KERNEL_PLAN& plan) {
  const std::string key = Build_vector_kernel_specialization_key(plan);
  return std::string("__ace_vkernel_") +
         Helper_kind_token(Get_vector_kernel_plan_kind(plan)) + '_' +
         Vector_kernel_sha256(key);
}

}  // namespace vector
}  // namespace nn
