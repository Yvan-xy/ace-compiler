//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "nn/vector/tensor2vector_prepared_lowering.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <type_traits>

#include "air/base/meta_info.h"
#include "air/base/st.h"
#include "air/core/opcode.h"
#include "nn/core/opcode.h"
#include "nn/vector/tensor2vector_ctx.h"
#include "nn/vector/tensor2vector_util.h"
#include "nn/vector/vector_utils.h"

namespace nn {
namespace vector {

namespace {

using air::base::ARRAY_TYPE_PTR;
using air::base::ATTR_ITER;
using air::base::ATTR_PTR;
using air::base::CONST_TYPE_PTR;
using air::base::CONSTANT_KIND;
using air::base::CONSTANT_PTR;
using air::base::GLOB_SCOPE;
using air::base::NODE_PTR;
using air::base::PRIMITIVE_TYPE;
using air::base::TYPE_PTR;

bool Host_is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t *>(&value) == 1;
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

size_t Primitive_component_width(PRIMITIVE_TYPE type) {
  if (type == PRIMITIVE_TYPE::COMPLEX_32)
    return 4;
  if (type == PRIMITIVE_TYPE::COMPLEX_64)
    return 8;
  return Primitive_width(type).value_or(0);
}

bool Checked_mul_size(size_t lhs, size_t rhs, size_t *value) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    return false;
  }
  *value = lhs * rhs;
  return true;
}

bool Checked_element_count(const std::vector<int64_t> &shape, size_t *count) {
  if (shape.empty())
    return false;
  size_t value = 1;
  for (int64_t dimension : shape) {
    if (dimension <= 0 ||
        static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max() ||
        !Checked_mul_size(value, static_cast<size_t>(dimension), &value)) {
      return false;
    }
  }
  *count = value;
  return true;
}

std::vector<uint8_t> Host_to_little_endian(const void *data, size_t byte_count,
                                           PRIMITIVE_TYPE type) {
  const uint8_t *raw = static_cast<const uint8_t *>(data);
  std::vector<uint8_t> result(raw, raw + byte_count);
  const size_t component_width = Primitive_component_width(type);
  if (!Host_is_little_endian() && component_width > 1) {
    for (size_t offset = 0; offset < result.size(); offset += component_width) {
      std::reverse(result.begin() + offset,
                   result.begin() + offset + component_width);
    }
  }
  return result;
}

std::vector<uint8_t> Little_to_host_endian(const std::vector<uint8_t> &bytes,
                                           PRIMITIVE_TYPE type) {
  std::vector<uint8_t> result = bytes;
  const size_t component_width = Primitive_component_width(type);
  if (!Host_is_little_endian() && component_width > 1) {
    for (size_t offset = 0; offset < result.size(); offset += component_width) {
      std::reverse(result.begin() + offset,
                   result.begin() + offset + component_width);
    }
  }
  return result;
}

std::string Payload_hash(const VECTOR_KERNEL_RANKED_TYPE_PLAN &type,
                         const std::vector<uint8_t> &bytes) {
  std::ostringstream header;
  header.imbue(std::locale::classic());
  header << "vector-kernel-constant:v1\n"
         << "element=" << Vector_kernel_primitive_type_name(type._element_type)
         << '\n'
         << "rank=" << type._shape.size() << '\n'
         << "shape=";
  for (size_t idx = 0; idx < type._shape.size(); ++idx) {
    if (idx != 0)
      header << ',';
    header << type._shape[idx];
  }
  header << "\npayload:\n";
  std::string preimage = header.str();
  if (!bytes.empty()) {
    preimage.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  }
  return "sha256:" + Vector_kernel_sha256(preimage);
}

std::optional<VECTOR_KERNEL_RANKED_TYPE_PLAN>
Structural_type(TYPE_PTR type, bool flatten_nested, std::string *diagnostic) {
  if (type == air::base::Null_ptr || !type->Is_array()) {
    *diagnostic = "vector-kernel value must have an AIR array type";
    return std::nullopt;
  }
  std::vector<int64_t> shape;
  TYPE_PTR current = type;
  do {
    ARRAY_TYPE_PTR array = current->Cast_to_arr();
    const std::vector<int64_t> dimensions = array->Shape();
    if (dimensions.empty()) {
      *diagnostic = "vector-kernel array type must have positive rank";
      return std::nullopt;
    }
    shape.insert(shape.end(), dimensions.begin(), dimensions.end());
    current = array->Elem_type();
    if (!flatten_nested && current->Is_array()) {
      *diagnostic = "nested array operand type is not a ranked primitive";
      return std::nullopt;
    }
  } while (current->Is_array());
  if (!current->Is_prim()) {
    *diagnostic = "vector-kernel array element type must be primitive";
    return std::nullopt;
  }
  const PRIMITIVE_TYPE primitive = current->Cast_to_prim()->Encoding();
  if (!Primitive_width(primitive).has_value()) {
    *diagnostic = "vector-kernel array element representation is unsupported";
    return std::nullopt;
  }
  size_t ignored = 0;
  if (!Checked_element_count(shape, &ignored)) {
    *diagnostic = "vector-kernel array shape is invalid or overflows";
    return std::nullopt;
  }
  return VECTOR_KERNEL_RANKED_TYPE_PLAN{primitive, std::move(shape)};
}

std::optional<VECTOR_KERNEL_TYPED_PAYLOAD>
Copy_constant_payload(const std::string &role, CONSTANT_PTR constant,
                      std::string *diagnostic) {
  if (constant == air::base::Null_ptr ||
      constant->Kind() != CONSTANT_KIND::ARRAY) {
    *diagnostic = "vector-kernel source role '" + role +
                  "' is not an inline ARRAY constant";
    return std::nullopt;
  }
  std::optional<VECTOR_KERNEL_RANKED_TYPE_PLAN> type =
      Structural_type(constant->Type(), true, diagnostic);
  if (!type.has_value())
    return std::nullopt;
  size_t count = 0;
  size_t expected_bytes = 0;
  const size_t width = *Primitive_width(type->_element_type);
  if (!Checked_element_count(type->_shape, &count) ||
      !Checked_mul_size(count, width, &expected_bytes) ||
      constant->Array_byte_len() != expected_bytes) {
    *diagnostic = "vector-kernel source role '" + role +
                  "' payload size does not match its type";
    return std::nullopt;
  }
  std::vector<uint8_t> bytes = Host_to_little_endian(
      constant->Array_buffer(), expected_bytes, type->_element_type);
  const std::string hash = Payload_hash(*type, bytes);
  return VECTOR_KERNEL_TYPED_PAYLOAD{role, std::move(*type), std::move(bytes),
                                     hash};
}

std::vector<uint8_t> Attribute_bytes(ATTR_PTR attribute) {
  const std::string_view value = attribute->Value();
  if (attribute->Count() == 0 ||
      !Primitive_width(attribute->Type()).has_value()) {
    return std::vector<uint8_t>(value.begin(), value.end());
  }
  return Host_to_little_endian(value.data(), value.size(), attribute->Type());
}

void Append_attribute_snapshot(
    std::vector<VECTOR_KERNEL_ATTRIBUTE_RECORD> *records, ATTR_PTR attribute) {
  const std::string_view value = attribute->Value();
  PRIMITIVE_TYPE type = attribute->Type();
  uint32_t count = attribute->Count();
  if (count == 0 || !Primitive_width(type).has_value()) {
    type = PRIMITIVE_TYPE::INT_U8;
    count = static_cast<uint32_t>(value.size());
  }
  // Empty strings do not affect current Conv/Gemm planning and cannot be
  // represented by v1's positive ranked dimensions.
  if (count == 0)
    return;
  records->push_back(VECTOR_KERNEL_ATTRIBUTE_RECORD{
      attribute->Key(), VECTOR_KERNEL_RANKED_TYPE_PLAN{type, {count}},
      Attribute_bytes(attribute)});
}

std::vector<uint8_t> S32_bytes(int32_t value) {
  return Host_to_little_endian(&value, sizeof(value), PRIMITIVE_TYPE::INT_S32);
}

bool Decompose_sharded_weight(NODE_PTR weight, CONSTANT_PTR *constant,
                              NODE_PTR *index) {
  if (weight == air::base::Null_ptr ||
      weight->Opcode() != air::core::OPC_ILD || weight->Num_child() != 1) {
    return false;
  }
  NODE_PTR address = weight->Child(0);
  if (address->Opcode() != air::core::OPC_ARRAY ||
      address->Num_child() != 2) {
    return false;
  }
  NODE_PTR base = address->Child(0);
  if (base->Opcode() != air::core::OPC_LDCA) {
    return false;
  }
  *constant = base->Const();
  *index = address->Child(1);
  return *constant != air::base::Null_ptr &&
         *index != air::base::Null_ptr;
}

CONSTANT_PTR Source_constant_for_weight(NODE_PTR node, bool *sharded) {
  NODE_PTR weight = node->Child(1);
  *sharded = weight->Opcode() == air::core::OPC_ILD;
  if (!*sharded) {
    if (weight->Opcode() != air::core::OPC_LDC) {
      return air::base::Null_ptr;
    }
    return weight->Const();
  }
  CONSTANT_PTR constant = air::base::Null_ptr;
  NODE_PTR index = air::base::Null_ptr;
  return Decompose_sharded_weight(weight, &constant, &index)
             ? constant
             : air::base::Null_ptr;
}

const VECTOR_KERNEL_TYPED_PAYLOAD *
Find_constant(const PREPARED_VECTOR_KERNEL_PLAN &prepared,
              const std::string &role) {
  for (const VECTOR_KERNEL_TYPED_PAYLOAD &payload : prepared.Constants()) {
    if (payload._role == role)
      return &payload;
  }
  return nullptr;
}

CONSTANT_PTR Materialize_constant(TENSOR2VECTOR_CTX &ctx,
                                  const VECTOR_KERNEL_TYPED_PAYLOAD &payload,
                                  const air::base::SPOS &spos) {
  GLOB_SCOPE *glob = ctx.Container()->Glob_scope();
  CONST_TYPE_PTR element = glob->Prim_type(payload._type._element_type);
  size_t count = 0;
  AIR_ASSERT(Checked_element_count(payload._type._shape, &count));
  AIR_ASSERT(count <= static_cast<size_t>(std::numeric_limits<int64_t>::max()));
  std::vector<uint8_t> host =
      Little_to_host_endian(payload._bytes, payload._type._element_type);
  const std::string name = "vector_kernel_" + payload._role;
  return New_array_const(glob, name.c_str(), static_cast<int64_t>(count),
                         element, payload._type._shape, host.data(), spos);
}

CONSTANT_PTR Materialize_optional_constant(
    TENSOR2VECTOR_CTX &ctx, const PREPARED_VECTOR_KERNEL_PLAN &prepared,
    const std::string &role, const air::base::SPOS &spos) {
  const VECTOR_KERNEL_TYPED_PAYLOAD *payload = Find_constant(prepared, role);
  return payload == nullptr ? air::base::Null_ptr
                            : Materialize_constant(ctx, *payload, spos);
}

std::vector<int>
Decode_s32_payload(const VECTOR_KERNEL_TYPED_PAYLOAD &payload) {
  AIR_ASSERT(payload._type._element_type == PRIMITIVE_TYPE::INT_S32);
  AIR_ASSERT(payload._bytes.size() % sizeof(int32_t) == 0);
  std::vector<int> values(payload._bytes.size() / sizeof(int32_t));
  for (size_t idx = 0; idx < values.size(); ++idx) {
    std::array<uint8_t, sizeof(int32_t)> element{};
    std::copy(payload._bytes.begin() + idx * sizeof(int32_t),
              payload._bytes.begin() + (idx + 1) * sizeof(int32_t),
              element.begin());
    if (!Host_is_little_endian())
      std::reverse(element.begin(), element.end());
    int32_t value = 0;
    std::memcpy(&value, element.data(), sizeof(value));
    values[idx] = value;
  }
  return values;
}

void Set_slot(NODE_PTR node, uint32_t value) {
  node->Set_attr(nn::core::ATTR::SLOT, &value, 1);
}

NODE_PTR Emit_prepared_fast_conv_native(
    TENSOR2VECTOR_CTX &ctx, TENSOR2VECTOR_UTIL &util, NODE_PTR input2d,
    NODE_PTR weight, NODE_PTR bias, const SHARD_MAP_PARAMS *params,
    bool need_mask, NODE_PTR new_offset, const SPOS &spos,
    const FAST_CONV_PLAN &prepared_plan,
    const FAST_CONV_PREPARED_CONSTANTS &prepared_constants) {
  ctx.Incr_num_vloop();
  const int channel_in = static_cast<int>(prepared_plan._channel_in);
  const int channel_out = static_cast<int>(prepared_plan._channel_out);
  const int output_height = static_cast<int>(prepared_plan._output_height);
  const int output_width = static_cast<int>(prepared_plan._output_width);
  const int group = static_cast<int>(prepared_plan._group);
  GLOB_SCOPE *gscope = ctx.Container()->Glob_scope();
  FUNC_SCOPE *fscope = ctx.Container()->Parent_func_scope();

  int input_size = channel_in * output_height * output_width;
  int output_size = channel_out * output_height * output_width;
  int num_grid = params->_num_grid;
  int num_block = params->_num_block;

  int width_block = params->_width_block;
  int width_block_data = params->_width_block_data;
  int width_block_pad = params->_width_block_pad;
  int cap_block = params->_cap_block;
  int position_block = params->_position_block;
  int num_dup_input = params->_num_dup_input;

  int64_t num_slots = prepared_plan._num_slots;

  // Get and check input type: lda input2d -> vector of vector
  AIR_ASSERT_MSG((input2d->Opcode() == air::core::OPC_LDA),
                 "Input is not an LDA node");
  ARRAY_TYPE_PTR type2d = input2d->Addr_datum()->Type()->Cast_to_arr();
  std::vector<int64_t> shape2d = type2d->Shape();
  AIR_ASSERT_MSG(((shape2d.size() == 1) && (shape2d[0] == cap_block)),
                 "input2d shape check");

  ARRAY_TYPE_PTR type2d_elem = type2d->Elem_type()->Cast_to_arr();

  CONSTANT_PTR weight_const = weight->Const();

  // Build var: result, result_grid, input_dup
  std::vector<int64_t> result_shape{1, channel_out, output_height,
                                    output_width};
  TYPE_PTR vtype = New_array_type(gscope, "type_result_n", ctx.Get_num_vloop(),
                                  type2d_elem->Elem_type(), result_shape, spos);

  // VECTOR result_var = 0
  ADDR_DATUM_PTR result_var =
      util.Gen_st_0_to_var_stmt("result_n", vtype, spos);

  std::string result_grid_str =
      (std::string("result_grid_n") + std::to_string(ctx.Get_num_vloop()));
  ADDR_DATUM_PTR result_grid_var =
      fscope->New_var(vtype, result_grid_str.c_str(), spos);

  std::string dup_str =
      (std::string("input_dup_n") + std::to_string(ctx.Get_num_vloop()));
  ADDR_DATUM_PTR input_dup_var = fscope->New_var(vtype, dup_str.c_str(), spos);

  CONST_TYPE_PTR s32_type = gscope->Prim_type(PRIMITIVE_TYPE::INT_S32);

  // 3) IR Gen for Code: grid-block two-level loop
  // Here MetaKernel is block-level computation.
  // for i in range(0, num_grid):
  //   result_grid = zeros(num_slots)
  //   for j in range(0, cap_block):
  //     result_grid += input_grid_var[j] * weight[i*cap_block+j]
  //   result_var += roll(result_grid, i*position_block)
  STMT_PTR grid_loop = util.New_loop("num_grid", 0, num_grid, spos);
  STMT_LIST grid_sl =
      STMT_LIST::Enclosing_list(grid_loop->Node()->Child(3)->End_stmt());

  // result_grid = zeros(num_slots)
  STMT_PTR st0_result_grid_stmt = ctx.Container()->New_st(
      ctx.Container()->New_zero(vtype, spos), result_grid_var, spos);
  grid_sl.Append(st0_result_grid_stmt);

  STMT_PTR cap_block_loop = util.New_loop("cap_block", 0, cap_block, spos);
  STMT_LIST cap_block_sl =
      STMT_LIST::Enclosing_list(cap_block_loop->Node()->Child(3)->End_stmt());

  // weight[i*cap_block+j]
  NODE_PTR slice_index_node = ctx.Container()->New_bin_arith(
      air::core::OPCODE::ADD, s32_type,
      ctx.Container()->New_ld(cap_block_loop->Node()->Iv(), spos),
      ctx.Container()->New_bin_arith(
          air::core::OPCODE::MUL, s32_type,
          ctx.Container()->New_ld(grid_loop->Node()->Iv(), spos),
          ctx.Container()->New_intconst(s32_type, cap_block, spos), spos),
      spos);

  if (new_offset != air::base::Null_ptr) {
    // sharding
    slice_index_node = ctx.Container()->New_bin_arith(
        air::core::OPCODE::ADD, s32_type, new_offset, slice_index_node, spos);
  }
  NODE_PTR weight_grid = util.New_slice(
      weight, slice_index_node,
      ctx.Container()->New_intconst(s32_type, num_block * width_block, spos),
      spos);

  // input_grid_var[j]
  NODE_PTR input_array2 = ctx.Container()->New_array(input2d, 1, spos);
  ctx.Container()->Set_array_idx(
      input_array2, 0,
      ctx.Container()->New_ld(cap_block_loop->Node()->Iv(), spos));
  NODE_PTR ild_input = ctx.Container()->New_ild(input_array2, spos);

  // result_var += input_grid_var[j] * weight[i*cap_block+j]
  NODE_PTR vmul_node = util.New_mul(ild_input, weight_grid, spos);
  NODE_PTR vadd_node = util.New_add(
      ctx.Container()->New_ld(result_grid_var, spos), vmul_node, spos);
  STMT_PTR vadd_store =
      ctx.Container()->New_st(vadd_node, result_grid_var, spos);

  cap_block_sl.Append(vadd_store);

  NODE_PTR cin_add_node = ctx.Container()->New_ld(result_grid_var, spos);

  // special case: introcude Roll_cyclic. the left-roll destroy existed data
  // The condition is strict. TODO :)
  bool is_cyclic = prepared_plan._cyclic_roll;
  ctx.Trace(TF_LOWER, "is_cyclic=", is_cyclic, " output_size=", output_size,
            "\n");

  NODE_PTR cin_roll_node2;
  if (is_cyclic) {
    ctx.Trace(TF_LOWER, "Roll_cyclic: output_size=", output_size,
              " position_block=", position_block, "\n");
    cin_roll_node2 =
        util.Roll_cyclic(result_grid_var, output_size, position_block, num_grid,
                         grid_loop->Node()->Iv(), spos, &prepared_constants);
  } else {
    // result_var += roll(result_grid, i*position_block)
    std::vector<int> roll_num_left;
    for (int i = 0; i < num_grid; i++)
      roll_num_left.push_back(i * position_block);
    cin_roll_node2 = util.New_roll(
        cin_add_node,
        ctx.Container()->New_bin_arith(
            air::core::OPCODE::MUL, s32_type,
            ctx.Container()->New_ld(grid_loop->Node()->Iv(), spos),
            ctx.Container()->New_intconst(s32_type, position_block, spos),
            spos),
        roll_num_left, spos);
  }

  NODE_PTR cin_add_node2 = util.New_add(
      ctx.Container()->New_ld(result_var, spos), cin_roll_node2, spos);
  STMT_PTR cin_add_st =
      ctx.Container()->New_st(cin_add_node2, result_var, spos);

  grid_sl.Append(cap_block_loop);
  grid_sl.Append(cin_add_st);
  ctx.Prepend(grid_loop);

  STMT_PTR vadd_bias_stmt;
  const VECTOR_KERNEL_REDUCTION_PLAN *prepared_reduction = nullptr;
  if (!prepared_plan._common._reductions.empty()) {
    AIR_ASSERT(prepared_plan._common._reductions.size() == 1);
    prepared_reduction = &prepared_plan._common._reductions.front();
  }
  const bool needs_reduction = prepared_reduction != nullptr;
  if (needs_reduction) {
    // Here we get results of all blocks in a vecotr. Reduce it.
    // num_block == 1: 2xslots for rotate-left. so width_block_pad:
    PREG_PTR epi_preg;
    const bool use_single_block =
        prepared_reduction->_kind ==
        VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_SINGLE_BLOCK;
    if (use_single_block) {
      epi_preg = util.Gen_collective_reduce_stmt(
          result_var, type2d_elem->Elem_type(), spos, width_block_data,
          output_size, need_mask, num_slots, &prepared_constants);
    } else {
      AIR_ASSERT(prepared_reduction->_kind ==
                 VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_BLOCKS);
      width_block_pad = prepared_reduction->_padding;
      epi_preg = util.Gen_collective_reduce_stmt(
          result_var, type2d_elem->Elem_type(), spos, num_block,
          width_block_data, width_block_pad, output_size, need_mask, num_slots,
          &prepared_constants);
    }
    vadd_bias_stmt = ctx.Container()->New_st(
        util.New_add(ctx.Container()->New_ldp(epi_preg, spos), bias, spos),
        result_var, spos);
  } else {
    vadd_bias_stmt = ctx.Container()->New_st(
        util.New_add(ctx.Container()->New_ld(result_var, spos), bias, spos),
        result_var, spos);
  }

  ctx.Prepend(vadd_bias_stmt);

  NODE_PTR ld_result = ctx.Container()->New_ld(result_var, spos);
  return ld_result;
}

} // namespace

VECTOR_KERNEL_REQUEST_BUILD_RESULT Build_vector_kernel_planning_request(
    TENSOR2VECTOR_CTX &ctx, NODE_PTR source_node,
    VECTOR_KERNEL_REQUESTED_PLAN_KIND requested_kind) {
  const air::base::OPCODE opcode = source_node->Opcode();
  VECTOR_KERNEL_OPERATION operation;
  if (opcode == air::base::OPCODE(nn::core::NN, nn::core::OPCODE::GEMM) ||
      opcode == air::base::OPCODE(nn::core::NN, nn::core::OPCODE::MATMUL)) {
    operation = VECTOR_KERNEL_OPERATION::GEMM;
  } else if (opcode ==
             air::base::OPCODE(nn::core::NN, nn::core::OPCODE::CONV)) {
    operation = VECTOR_KERNEL_OPERATION::CONV;
  } else {
    return {std::nullopt, "source node is not a vector-kernel Conv/Gemm"};
  }
  if (source_node->Num_child() != 3) {
    return {std::nullopt, "vector-kernel Conv/Gemm requires three operands"};
  }

  std::string diagnostic;
  std::vector<VECTOR_KERNEL_RANKED_TYPE_PLAN> operand_types;
  operand_types.reserve(source_node->Num_child());
  for (uint32_t idx = 0; idx < source_node->Num_child(); ++idx) {
    std::optional<VECTOR_KERNEL_RANKED_TYPE_PLAN> type =
        Structural_type(source_node->Child(idx)->Rtype(), false, &diagnostic);
    if (!type.has_value())
      return {std::nullopt, std::move(diagnostic)};
    operand_types.push_back(std::move(*type));
  }
  std::optional<VECTOR_KERNEL_RANKED_TYPE_PLAN> result_type =
      Structural_type(source_node->Rtype(), false, &diagnostic);
  if (!result_type.has_value()) {
    return {std::nullopt, std::move(diagnostic)};
  }

  std::vector<VECTOR_KERNEL_ATTRIBUTE_RECORD> attributes;
  if (air::base::META_INFO::Has_prop<air::base::OPR_PROP::ATTR>(opcode)) {
    for (ATTR_ITER iter = source_node->Begin_attr();
         iter != source_node->End_attr(); ++iter) {
      Append_attribute_snapshot(&attributes, *iter);
    }
  }

  bool sharded = false;
  CONSTANT_PTR weight = Source_constant_for_weight(source_node, &sharded);
  std::optional<VECTOR_KERNEL_TYPED_PAYLOAD> weight_payload =
      Copy_constant_payload("weight", weight, &diagnostic);
  if (!weight_payload.has_value()) {
    return {std::nullopt, std::move(diagnostic)};
  }
  NODE_PTR bias_node = source_node->Child(2);
  CONSTANT_PTR bias_constant = air::base::Null_ptr;
  if (bias_node->Opcode() ==
      air::base::OPCODE(air::core::CORE, air::core::OPCODE::LDC)) {
    bias_constant = bias_node->Const();
  }
  std::optional<VECTOR_KERNEL_TYPED_PAYLOAD> bias_payload =
      Copy_constant_payload("bias", bias_constant, &diagnostic);
  if (!bias_payload.has_value()) {
    return {std::nullopt, std::move(diagnostic)};
  }
  if (sharded) {
    attributes.push_back(VECTOR_KERNEL_ATTRIBUTE_RECORD{
        "weight-sharded",
        VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::INT_S32, {1}},
        S32_bytes(1)});
  }
  std::vector<PRIMITIVE_TYPE> runtime_scalar_types;
  if (sharded) {
    NODE_PTR source_scalar = Find_vector_kernel_source_scalar(source_node);
    if (source_scalar == air::base::Null_ptr ||
        source_scalar->Rtype() == air::base::Null_ptr ||
        !source_scalar->Rtype()->Is_prim()) {
      return {std::nullopt,
              "sharded Conv weight offset must have a primitive type"};
    }
    runtime_scalar_types.push_back(
        source_scalar->Rtype()->Cast_to_prim()->Encoding());
  }
  std::vector<VECTOR_KERNEL_TYPED_PAYLOAD> constants;
  constants.push_back(std::move(*weight_payload));
  constants.push_back(std::move(*bias_payload));

  const int64_t slots = ctx.Get_slot();
  if (slots <= 0) {
    return {std::nullopt, "vector-kernel slot target must be positive"};
  }
  return {VECTOR_KERNEL_PLANNING_REQUEST{
              operation, std::move(attributes), std::move(operand_types),
              std::move(*result_type),
              VECTOR_KERNEL_OPTION_SNAPSHOT{ctx.Conv_parallel(),
                                            ctx.Mask_fuse(), ctx.Selective_ss(),
                                            ctx.Sharding(), ctx.Is_last_op()},
              VECTOR_KERNEL_TARGET_SNAPSHOT{slots, MIN_SLOT_ALLOWED,
                                            MAX_SLOT_ALLOWED},
              std::move(constants), requested_kind,
              std::move(runtime_scalar_types)},
          {}};
}

NODE_PTR Find_vector_kernel_source_scalar(NODE_PTR source_node) {
  if (source_node->Opcode() !=
      air::base::OPCODE(nn::core::NN, nn::core::OPCODE::CONV)) {
    return air::base::Null_ptr;
  }
  CONSTANT_PTR constant = air::base::Null_ptr;
  NODE_PTR index = air::base::Null_ptr;
  return Decompose_sharded_weight(source_node->Child(1), &constant, &index)
             ? index
             : air::base::Null_ptr;
}

NODE_PTR Emit_prepared_vector_kernel_native(
    TENSOR2VECTOR_CTX &ctx, const PREPARED_VECTOR_KERNEL_PLAN &prepared,
    NODE_PTR ranked_input, const std::vector<NODE_PTR> &scalar_actuals,
    const air::base::SPOS &spos) {
  const VECTOR_KERNEL_TYPED_PAYLOAD *weight_payload =
      Find_constant(prepared, "weight");
  const VECTOR_KERNEL_TYPED_PAYLOAD *bias_payload =
      Find_constant(prepared, "bias");
  AIR_ASSERT(weight_payload != nullptr && bias_payload != nullptr);
  CONSTANT_PTR weight_constant =
      Materialize_constant(ctx, *weight_payload, spos);
  CONSTANT_PTR bias_constant = Materialize_constant(ctx, *bias_payload, spos);
  NODE_PTR weight = ctx.Container()->New_ldc(weight_constant, spos);
  NODE_PTR bias = ctx.Container()->New_ldc(bias_constant, spos);
  TENSOR2VECTOR_UTIL util(ctx);

  return std::visit(
      [&](const auto &plan) -> NODE_PTR {
        using PLAN = std::decay_t<decltype(plan)>;
        const VECTOR_KERNEL_COMMON_PLAN &common = plan._common;
        if constexpr (std::is_same_v<PLAN, BASELINE_GEMM_PLAN>) {
          AIR_ASSERT(scalar_actuals.empty());
          const bool need_mask =
              common._mask._policy != VECTOR_KERNEL_MASK_POLICY::NONE;
          CONSTANT_PTR mask =
              Materialize_optional_constant(ctx, prepared, "mask", spos);
          AIR_ASSERT(!need_mask || mask != air::base::Null_ptr);
          return util.New_gemm_metakernel(ranked_input, weight, bias, need_mask,
                                          spos, plan._input_duplications, mask);
        } else if constexpr (std::is_same_v<PLAN, BASELINE_CONV_PLAN>) {
          AIR_ASSERT(scalar_actuals.empty());
          const VECTOR_KERNEL_TYPED_PAYLOAD *rotations =
              Find_constant(prepared, "rotation-table");
          AIR_ASSERT(rotations != nullptr);
          std::vector<int> values = Decode_s32_payload(*rotations);
          NODE_PTR result = util.New_conv_metakernel(
              ranked_input, weight, bias, std::move(values),
              static_cast<int>(plan._channel_in),
              static_cast<int>(plan._channel_out),
              static_cast<int>(plan._output_height),
              static_cast<int>(plan._output_width),
              static_cast<int>(plan._kernel_hw), static_cast<int>(plan._stride),
              spos, plan._input_duplications, true);
          Set_slot(result, common._slot._value);
          return result;
        } else if constexpr (std::is_same_v<PLAN, FAST_GEMM_PLAN>) {
          AIR_ASSERT(scalar_actuals.empty());
          const VECTOR_KERNEL_TYPED_PAYLOAD *rotations =
              Find_constant(prepared, "rotation-table");
          AIR_ASSERT(rotations != nullptr);
          std::vector<int> alignment = Decode_s32_payload(*rotations);
          NODE_PTR input_block = util.Blocking_rot(
              ranked_input, alignment, static_cast<int>(plan._kp),
              static_cast<int>(plan._input_replications),
              static_cast<int>(plan._block_size), spos);
          air::base::ADDR_DATUM_PTR result = util.New_gemm_metakernel_fast(
              input_block, weight, bias, plan._np, plan._kp,
              static_cast<int>(plan._block_size),
              static_cast<int>(plan._grid_size), spos);
          air::base::STMT_PTR comment = ctx.Container()->New_comment(
              (std::string("gemm result reduce->Ps=") +
               std::to_string(plan._packed_partitions))
                  .c_str(),
              spos);
          ctx.Prepend(comment);
          if (plan._packed_partitions > 1) {
            result = util.Reduce_add_intra(
                result, static_cast<int>(plan._packed_partitions),
                static_cast<int>(plan._kd),
                static_cast<int>(plan._shift_buffer), spos);
          }
          comment = ctx.Container()->New_comment(
              (std::string("gemm result reduce->(kp/np)=") +
               std::to_string(plan._kp / plan._np))
                  .c_str(),
              spos);
          ctx.Prepend(comment);
          if (plan._kp > plan._np) {
            result = util.Reduce_add_intra(
                result, static_cast<int>(plan._kp / plan._np),
                static_cast<int>(plan._np), 0, spos);
          }
          comment = ctx.Container()->New_comment("gemm add bias", spos);
          ctx.Prepend(comment);
          NODE_PTR add =
              util.New_add(ctx.Container()->New_ld(result, spos), bias, spos);
          ctx.Prepend(ctx.Container()->New_st(add, result, spos));
          if (common._mask._policy != VECTOR_KERNEL_MASK_POLICY::NONE) {
            CONSTANT_PTR mask =
                Materialize_optional_constant(ctx, prepared, "mask", spos);
            AIR_ASSERT(mask != air::base::Null_ptr);
            util.Gen_clear_data_stmt(
                result, common._mask._valid_length,
                ranked_input->Rtype()->Cast_to_arr()->Elem_type(), spos, mask);
          }
          NODE_PTR loaded = ctx.Container()->New_ld(result, spos);
          Set_slot(loaded, common._slot._value);
          return loaded;
        } else {
          const VECTOR_KERNEL_TYPED_PAYLOAD *rotations =
              Find_constant(prepared, "rotation-table");
          AIR_ASSERT(rotations != nullptr);
          std::vector<int> alignment = Decode_s32_payload(*rotations);
          AIR_ASSERT(plan._blocking_outer_depth <=
                     std::numeric_limits<int>::max());
          NODE_PTR input_block = util.Blocking_rot(
              ranked_input, alignment, static_cast<int>(plan._input_size),
              static_cast<int>(plan._input_duplications),
              static_cast<int>(plan._capacity_block), spos,
              static_cast<int>(plan._blocking_outer_depth));

          NODE_PTR scaled_offset = air::base::Null_ptr;
          if (plan._sharding_offset.has_value()) {
            AIR_ASSERT(scalar_actuals.size() == 1);
            AIR_ASSERT(scalar_actuals[0] != air::base::Null_ptr);
            AIR_ASSERT(scalar_actuals[0]->Rtype()->Is_prim());
            AIR_ASSERT(scalar_actuals[0]
                           ->Rtype()
                           ->Cast_to_prim()
                           ->Encoding() == plan._sharding_offset->_type);
            CONST_TYPE_PTR s32 = ctx.Container()->Glob_scope()->Prim_type(
                PRIMITIVE_TYPE::INT_S32);
            scaled_offset = ctx.Container()->New_bin_arith(
                air::core::OPCODE::MUL, s32, scalar_actuals[0],
                ctx.Container()->New_intconst(
                    s32, plan._sharding_offset->_scale, spos),
                spos);
          } else {
            AIR_ASSERT(scalar_actuals.empty());
          }
          FAST_CONV_PREPARED_CONSTANTS prepared_constants;
          prepared_constants._cyclic_mask_left = Materialize_optional_constant(
              ctx, prepared, "cyclic-mask-left", spos);
          prepared_constants._cyclic_mask_right = Materialize_optional_constant(
              ctx, prepared, "cyclic-mask-right", spos);
          prepared_constants._collective_mask = Materialize_optional_constant(
              ctx, prepared, "collective-mask", spos);
          prepared_constants._collective_gap_mask =
              Materialize_optional_constant(ctx, prepared,
                                            "collective-gap-mask", spos);
          SHARD_MAP_PARAMS params{static_cast<int>(plan._num_grid),
                                  static_cast<int>(plan._num_block),
                                  static_cast<int>(plan._width_block),
                                  static_cast<int>(plan._width_block_data),
                                  static_cast<int>(plan._width_block_pad),
                                  static_cast<int>(plan._position_block),
                                  static_cast<int>(plan._capacity_block),
                                  static_cast<int>(plan._input_duplications)};
          NODE_PTR result = Emit_prepared_fast_conv_native(
              ctx, util, input_block, weight, bias, &params,
              common._mask._policy != VECTOR_KERNEL_MASK_POLICY::NONE,
              scaled_offset, spos, plan, prepared_constants);
          Set_slot(result, common._slot._value);
          return result;
        }
      },
      prepared.Plan());
}

} // namespace vector
} // namespace nn
