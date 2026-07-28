//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "air/base/meta_info.h"
#include "air/base/node.h"
#include "air/base/st.h"
#include "air/base/st_attr.h"
#include "air/base/st_iter.h"
#include "air/base/st_type.h"
#include "air/core/opcode.h"
#include "nn/core/attr.h"
#include "nn/core/opcode.h"
#include "nn/vector/config.h"
#include "nn/vector/tensor2vector_ctx.h"
#include "nn/vector/tensor2vector_plan.h"
#include "nn/vector/tensor2vector_planning.h"
#include "nn/vector/tensor2vector_prepared_lowering.h"
#include "nn/vector/tensor2vector_util.h"
#include "nn/vector/vector_ctx.h"
#include "nn/vector/vector_opcode.h"
#include "nn/vector/vector_utils.h"
#include "tensor2vector_air_normalizer.h"

using namespace air::base;
using namespace nn::vector;

namespace {

struct NATIVE_GEMM_IR {
  std::unique_ptr<GLOB_SCOPE> _glob;
  FUNC_SCOPE*                 _func_scope = nullptr;
  NODE_PTR                    _result;
};

NATIVE_GEMM_IR Build_native_gemm(const char* function_name, uint32_t line,
                                 int64_t height, int64_t width,
                                 bool need_mask) {
  NATIVE_GEMM_IR fixture;
  fixture._glob = std::make_unique<GLOB_SCOPE>(0, true);
  GLOB_SCOPE* glob = fixture._glob.get();
  const SPOS  spos(0, line, 1, 0);

  TYPE_PTR f32 = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_32);
  TYPE_PTR input_type =
      New_array_type(glob, std::string(function_name) + "_input", f32,
                     {width}, spos);
  TYPE_PTR result_type =
      New_array_type(glob, std::string(function_name) + "_result", f32,
                     {2 * width}, spos);

  STR_PTR  name = glob->New_str(function_name);
  FUNC_PTR func = glob->New_func(name, spos);
  func->Set_parent(glob->Comp_env_id());
  SIGNATURE_TYPE_PTR signature = glob->New_sig_type();
  glob->New_ret_param(result_type, signature);
  glob->New_param(glob->New_str("packed_input"), input_type, signature, spos);
  signature->Set_complete();
  glob->New_entry_point(signature, func, name, spos);

  fixture._func_scope = &glob->New_func_scope(func);
  CONTAINER* container = &fixture._func_scope->Container();
  STMT_PTR   entry     = container->New_func_entry(spos);
  NODE_PTR   body      = entry->Node()->Last_child();

  std::vector<float> weight(height * width);
  for (size_t idx = 0; idx < weight.size(); ++idx) {
    weight[idx] = static_cast<float>(idx + 1) / 8.0F;
  }
  std::vector<float> bias(height);
  for (size_t idx = 0; idx < bias.size(); ++idx) {
    bias[idx] = static_cast<float>(idx + 1) / 16.0F;
  }
  CONSTANT_PTR weight_constant = New_array_const(
      glob, std::string(function_name) + "_weight", height * width, f32,
      {height, width}, weight.data(), spos);
  CONSTANT_PTR bias_constant = New_array_const(
      glob, std::string(function_name) + "_bias", height, f32, {height},
      bias.data(), spos);

  VECTOR_CTX    vector_ctx;
  VECTOR_CONFIG config;
  vector_ctx.Update_slot(128);
  TENSOR2VECTOR_CTX lowering_ctx(container, vector_ctx, nullptr, config);
  lowering_ctx.Set_cur_func_scope(fixture._func_scope);
  TENSOR2VECTOR_UTIL util(lowering_ctx);
  lowering_ctx.Push(body, body);
  fixture._result = util.New_gemm_metakernel(
      container->New_ld(fixture._func_scope->Formal(0), spos),
      container->New_ldc(weight_constant, spos),
      container->New_ldc(bias_constant, spos), need_mask, spos);
  container->Stmt_list().Append(container->New_retv(fixture._result, spos));
  lowering_ctx.Pop(body, body);
  return fixture;
}

struct NATIVE_GEMM_STATS {
  uint32_t                      _loops       = 0;
  uint32_t                      _vector_muls = 0;
  uint32_t                      _slices      = 0;
  uint32_t                      _slot_attrs  = 0;
  std::vector<std::vector<int>> _rotations;
};

void Collect_stats(NODE_PTR node, NATIVE_GEMM_STATS& stats) {
  if (node->Is_do_loop()) ++stats._loops;
  if (node->Opcode() ==
      OPCODE(VECTOR_DOMAIN::ID, VECTOR_OPCODE::MUL)) {
    ++stats._vector_muls;
  }
  if (node->Opcode() ==
      OPCODE(VECTOR_DOMAIN::ID, VECTOR_OPCODE::SLICE)) {
    ++stats._slices;
  }
  if (META_INFO::Has_prop<OPR_PROP::ATTR>(node->Opcode())) {
    uint32_t count = 0;
    const uint32_t* slot =
        node->Attr<uint32_t>(nn::core::ATTR::SLOT, &count);
    if (slot != nullptr) ++stats._slot_attrs;
    const int* rotations = node->Attr<int>(nn::core::ATTR::RNUM, &count);
    if (rotations != nullptr) {
      stats._rotations.emplace_back(rotations, rotations + count);
    }
  }

  if (node->Is_block()) {
    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
         stmt          = stmt->Next()) {
      Collect_stats(stmt->Node(), stats);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      Collect_stats(node->Child(idx), stats);
    }
  }
}

NATIVE_GEMM_STATS Get_stats(const NATIVE_GEMM_IR& fixture) {
  NATIVE_GEMM_STATS stats;
  const CONTAINER& container = fixture._func_scope->Container();
  Collect_stats(container.Stmt(fixture._func_scope->Entry_stmt_id())->Node(),
                stats);
  return stats;
}

std::string Snapshot(const NATIVE_GEMM_IR& fixture) {
  return nn::vector::test::Normalize_vector_kernel_function(
      *fixture._func_scope);
}

std::vector<uint8_t> Float_bytes(const std::vector<float>& values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(float));
  for (float value : values) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    for (uint32_t shift = 0; shift != 32; shift += 8) {
      bytes.push_back(static_cast<uint8_t>((bits >> shift) & 0xffU));
    }
  }
  return bytes;
}

std::vector<uint8_t> Int32_bytes(const std::vector<int32_t>& values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(int32_t));
  for (int32_t value : values) {
    const uint32_t bits = static_cast<uint32_t>(value);
    for (uint32_t shift = 0; shift != 32; shift += 8) {
      bytes.push_back(static_cast<uint8_t>((bits >> shift) & 0xffU));
    }
  }
  return bytes;
}

VECTOR_KERNEL_TYPED_PAYLOAD Float_payload(
    std::string role, std::vector<int64_t> shape,
    const std::vector<float>& values) {
  VECTOR_KERNEL_RANKED_TYPE_PLAN type{PRIMITIVE_TYPE::FLOAT_32,
                                      std::move(shape)};
  std::vector<uint8_t> bytes = Float_bytes(values);
  const std::string hash = Build_vector_kernel_constant_hash(
      type._element_type, type._shape, values.data(), bytes.size());
  return VECTOR_KERNEL_TYPED_PAYLOAD{
      std::move(role), std::move(type), std::move(bytes), hash};
}

VECTOR_KERNEL_ATTRIBUTE_RECORD Int32_attribute(
    std::string name, const std::vector<int32_t>& values) {
  return VECTOR_KERNEL_ATTRIBUTE_RECORD{
      std::move(name),
      VECTOR_KERNEL_RANKED_TYPE_PLAN{
          PRIMITIVE_TYPE::INT_S32,
          {static_cast<int64_t>(values.size())}},
      Int32_bytes(values)};
}

std::vector<float> Frozen_conv_source_weight(int64_t channel_in,
                                             int64_t channel_out,
                                             int64_t kernel) {
  std::vector<float> weight(
      static_cast<size_t>(channel_out * channel_in * kernel * kernel));
  for (size_t idx = 0; idx < weight.size(); ++idx) {
    weight[idx] =
        static_cast<float>(static_cast<int32_t>(idx % 17) - 8) / 8.0F;
  }
  return weight;
}

VECTOR_KERNEL_PLANNING_REQUEST Gemm_oracle_request(
    VECTOR_KERNEL_REQUESTED_PLAN_KIND kind,
    const VECTOR_KERNEL_OPTION_SNAPSHOT& options,
    const VECTOR_KERNEL_TARGET_SNAPSHOT& target) {
  const std::vector<float> weight{
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  const std::vector<float> bias{0.25F, -0.5F};
  return VECTOR_KERNEL_PLANNING_REQUEST{
      VECTOR_KERNEL_OPERATION::GEMM,
      {},
      {VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {4}},
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {2, 4}},
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {2}}},
      VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {2}},
      options,
      target,
      {Float_payload("weight", {2, 4}, weight),
       Float_payload("bias", {2}, bias)},
      kind};
}

VECTOR_KERNEL_PLANNING_REQUEST Conv_oracle_request(
    VECTOR_KERNEL_REQUESTED_PLAN_KIND kind, int64_t channel_in,
    int64_t channel_out, int64_t kernel,
    const VECTOR_KERNEL_OPTION_SNAPSHOT& options,
    const VECTOR_KERNEL_TARGET_SNAPSHOT& target) {
  constexpr int64_t height = 2;
  std::vector<float> weight =
      Frozen_conv_source_weight(channel_in, channel_out, kernel);
  std::vector<float> bias(static_cast<size_t>(channel_out));
  for (size_t idx = 0; idx < bias.size(); ++idx) {
    bias[idx] = static_cast<float>(idx) / 4.0F;
  }
  return VECTOR_KERNEL_PLANNING_REQUEST{
      VECTOR_KERNEL_OPERATION::CONV,
      {Int32_attribute("group", {1}),
       Int32_attribute("strides", {1, 1}),
       Int32_attribute("pads",
                       {static_cast<int32_t>(kernel / 2),
                        static_cast<int32_t>(kernel / 2),
                        static_cast<int32_t>(kernel / 2),
                        static_cast<int32_t>(kernel / 2)})},
      {VECTOR_KERNEL_RANKED_TYPE_PLAN{
           PRIMITIVE_TYPE::FLOAT_32,
           {1, channel_in, height, height}},
       VECTOR_KERNEL_RANKED_TYPE_PLAN{
           PRIMITIVE_TYPE::FLOAT_32,
           {channel_out, channel_in, kernel, kernel}},
       VECTOR_KERNEL_RANKED_TYPE_PLAN{
           PRIMITIVE_TYPE::FLOAT_32, {channel_out}}},
      VECTOR_KERNEL_RANKED_TYPE_PLAN{
          PRIMITIVE_TYPE::FLOAT_32,
          {1, channel_out, height, height}},
      options,
      target,
      {Float_payload("weight",
                     {channel_out, channel_in, kernel, kernel}, weight),
       Float_payload("bias", {channel_out}, bias)},
      kind};
}

struct FROZEN_BASELINE_GEMM_PLAN {
  std::vector<int64_t> _weight_shape{2, 4};
  std::vector<float> _weight{1.0F, 6.0F, 3.0F, 8.0F,
                             2.0F, 7.0F, 4.0F, 5.0F};
  std::vector<int64_t> _bias_shape{2};
  std::vector<float> _bias{0.25F, -0.5F};
  int64_t _height = 2;
  int64_t _width = 4;
  int64_t _input_duplications = 2;
  bool _need_mask = true;
};

struct FROZEN_BASELINE_CONV_PLAN {
  std::vector<int64_t> _weight_shape{4, 8};
  std::vector<float> _weight;
  std::vector<int64_t> _bias_shape{8};
  std::vector<float> _bias{0.0F, 0.0F, 0.0F, 0.0F,
                           0.25F, 0.25F, 0.25F, 0.25F};
  std::vector<int> _alignment{0};
  int64_t _channel_in = 4;
  int64_t _channel_out = 2;
  int64_t _output_height = 2;
  int64_t _output_width = 2;
  int64_t _kernel_hw = 1;
  int64_t _stride = 1;
  int64_t _input_duplications = 2;
  uint32_t _slot = 8;
};

struct FROZEN_FAST_GEMM_PLAN {
  std::vector<int64_t> _weight_shape{2, 6};
  std::vector<float> _weight{1.0F, 6.0F, 3.0F, 8.0F, 1.0F, 6.0F,
                             5.0F, 2.0F, 7.0F, 4.0F, 5.0F, 2.0F};
  std::vector<int64_t> _bias_shape{2};
  std::vector<float> _bias{0.25F, -0.5F};
  std::vector<int> _alignment{0};
  int64_t _n = 2;
  int64_t _k = 4;
  int64_t _np = 2;
  int64_t _kp = 4;
  int64_t _nd = 2;
  int64_t _kd = 4;
  int64_t _block_size = 1;
  int64_t _blocks_per_partition = 2;
  int64_t _packed_partitions = 1;
  int64_t _shift = 1;
  int64_t _shift_buffer = 2;
  int64_t _grid_size = 2;
  int64_t _input_replications = 2;
  bool _need_mask = true;
  uint32_t _slot = 2;
};

struct FROZEN_FAST_CONV_PLAN {
  std::vector<int64_t> _weight_shape{18, 16};
  std::vector<float> _weight;
  std::vector<int64_t> _bias_shape{16};
  std::vector<float> _bias{0.0F,  0.0F,  0.0F,  0.0F,
                           0.25F, 0.25F, 0.25F, 0.25F,
                           0.5F,  0.5F,  0.5F,  0.5F,
                           0.75F, 0.75F, 0.75F, 0.75F};
  std::vector<int> _alignment{-3, -2, -1, -1, 0, 1, 1, 2, 3};
  int64_t _channel_in = 2;
  int64_t _channel_out = 4;
  int64_t _output_height = 2;
  int64_t _output_width = 2;
  int64_t _kernel_hw = 9;
  int64_t _group = 1;
  int64_t _stride = 1;
  int64_t _input_size = 8;
  int64_t _output_size = 16;
  int64_t _num_slots = 32;
  int64_t _num_grid = 2;
  int64_t _num_block = 1;
  int64_t _width_block = 16;
  int64_t _width_block_data = 16;
  int64_t _width_block_pad = 0;
  int64_t _position_block = 4;
  int64_t _capacity_block = 9;
  int64_t _input_duplications = 2;
  int64_t _blocking_outer_depth = 0;
  bool _need_mask = true;
  uint32_t _slot = 16;
};

std::vector<float> Frozen_conv_im2col_weight(int64_t channel_in,
                                             int64_t channel_out,
                                             int64_t kernel, bool fast) {
  constexpr int64_t height = 2;
  constexpr int64_t width = 2;
  std::vector<float> source =
      Frozen_conv_source_weight(channel_in, channel_out, kernel);
  const int64_t kernel_hw = kernel * kernel;
  const int64_t row_count = channel_in * kernel_hw;
  const int64_t plane = height * width;
  FPMAT im2col(static_cast<size_t>(row_count),
               FPVEC(static_cast<size_t>(channel_out * plane), 0.0F));
  const int64_t pad = (kernel - 1) / 2;
  for (int64_t output = 0; output < channel_out; ++output) {
    for (int64_t row = 0; row < row_count; ++row) {
      const int64_t kernel_index = row % kernel_hw;
      const int64_t source_row =
          (row + output * kernel_hw) % row_count;
      const float value = source[static_cast<size_t>(
          output * row_count + source_row)];
      const int64_t kernel_row = kernel_index / kernel;
      const int64_t kernel_column = kernel_index % kernel;
      for (int64_t h = 0; h < height; ++h) {
        for (int64_t w = 0; w < width; ++w) {
          const bool in_bounds =
              h + kernel_row >= pad && w + kernel_column >= pad &&
              h + kernel_row < height + pad &&
              w + kernel_column < width + pad;
          if (in_bounds) {
            im2col[static_cast<size_t>(row)][static_cast<size_t>(
                output * plane + h * width + w)] = value;
          }
        }
      }
    }
  }

  std::vector<float> packed;
  packed.reserve(static_cast<size_t>(channel_in * kernel * kernel *
                                     channel_out * height * width));
  if (!fast) {
    for (const FPVEC& row : im2col) {
      packed.insert(packed.end(), row.begin(), row.end());
    }
    return packed;
  }

  for (int64_t channel = 0; channel < channel_in; ++channel) {
    for (int64_t offset = 0; offset < kernel * kernel; ++offset) {
      FPVEC row = im2col[static_cast<size_t>(
          channel * kernel * kernel + offset)];
      const int64_t rotation = channel * height * width;
      std::rotate(row.begin(), row.end() - rotation, row.end());
      packed.insert(packed.end(), row.begin(), row.end());
    }
  }
  return packed;
}

using FROZEN_NATIVE_PLAN =
    std::variant<FROZEN_BASELINE_GEMM_PLAN, FROZEN_BASELINE_CONV_PLAN,
                 FROZEN_FAST_GEMM_PLAN, FROZEN_FAST_CONV_PLAN>;

struct FROZEN_NATIVE_CASE {
  const char* _name;
  VECTOR_KERNEL_OPERATION _operation;
  VECTOR_KERNEL_REQUESTED_PLAN_KIND _requested;
  VECTOR_KERNEL_PLAN_KIND _expected;
  VECTOR_KERNEL_RANKED_TYPE_PLAN _runtime_input_type;
  VECTOR_KERNEL_RANKED_TYPE_PLAN _runtime_result_type;
  VECTOR_KERNEL_OPTION_SNAPSHOT _options;
  VECTOR_KERNEL_TARGET_SNAPSHOT _target;
  FROZEN_NATIVE_PLAN _legacy_plan;
  const char* _air_hash;
};

std::vector<FROZEN_NATIVE_CASE> Frozen_native_cases() {
  const VECTOR_KERNEL_OPTION_SNAPSHOT options{false, false, false, false,
                                               false};
  const VECTOR_KERNEL_TARGET_SNAPSHOT target{32, 1, MAX_SLOT_ALLOWED};

  FROZEN_BASELINE_CONV_PLAN baseline_conv;
  baseline_conv._weight = Frozen_conv_im2col_weight(4, 2, 1, false);
  FROZEN_FAST_CONV_PLAN fast_conv;
  fast_conv._weight = Frozen_conv_im2col_weight(2, 4, 3, true);

  return {
      {"baseline-gemm", VECTOR_KERNEL_OPERATION::GEMM,
       VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
       VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM,
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {4}},
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {8}}, options,
       target, FROZEN_BASELINE_GEMM_PLAN{},
       "331c9ae48495848035c537898a050bbf"
       "d33b765f26d7e5f6e83080fa774c3daa"},
      {"baseline-conv", VECTOR_KERNEL_OPERATION::CONV,
       VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV,
       VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV,
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {16}},
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {1, 2, 2, 2}},
       options, target, std::move(baseline_conv),
       "ea2010d9b97b4f863b8b159952f5be8"
       "747b39341c1dae705a7e0e5cc1ff306bc"},
      {"fast-gemm", VECTOR_KERNEL_OPERATION::GEMM,
       VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM,
       VECTOR_KERNEL_PLAN_KIND::FAST_GEMM,
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {4}},
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {6}}, options,
       target, FROZEN_FAST_GEMM_PLAN{},
       "e45727cd8897005f1e3281cd53123b43"
       "c8aa902a36edf86efee4b1769c1543d2"},
      {"fast-conv", VECTOR_KERNEL_OPERATION::CONV,
       VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV,
       VECTOR_KERNEL_PLAN_KIND::FAST_CONV,
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {8}},
       VECTOR_KERNEL_RANKED_TYPE_PLAN{PRIMITIVE_TYPE::FLOAT_32, {1, 4, 2, 2}},
       options, target, std::move(fast_conv),
       "8fd43b06baabf303330b6f2bfa161b8"
       "178c55b43f9f6bd5efa2c559e8c736ee3"},
  };
}

VECTOR_KERNEL_PLANNING_REQUEST Frozen_oracle_request(
    const FROZEN_NATIVE_CASE& test_case) {
  if (test_case._operation == VECTOR_KERNEL_OPERATION::GEMM) {
    return Gemm_oracle_request(test_case._requested, test_case._options,
                               test_case._target);
  }
  if (test_case._requested ==
      VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV) {
    return Conv_oracle_request(test_case._requested, 4, 2, 1,
                               test_case._options, test_case._target);
  }
  return Conv_oracle_request(test_case._requested, 2, 4, 3,
                             test_case._options, test_case._target);
}

std::shared_ptr<const PREPARED_VECTOR_KERNEL_PLAN> Resolve_oracle_plan(
    const VECTOR_KERNEL_PLANNING_REQUEST& request) {
  const VECTOR_KERNEL_SELECTION selection{
      VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
      VECTOR_KERNEL_IMPLEMENTATION::NATIVE,
      request._requested_plan_kind,
      VECTOR_KERNEL_FALLBACK_POLICY::ERROR};
  const VECTOR_KERNEL_RESOLUTION_RESULT resolution =
      Resolve_vector_kernel_plan(request, selection, nullptr);
  EXPECT_TRUE(resolution.Ok()) << resolution._diagnostic;
  return resolution._prepared;
}

size_t Element_count(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (int64_t dimension : shape) {
    AIR_ASSERT(dimension > 0);
    count *= static_cast<size_t>(dimension);
  }
  return count;
}

CONSTANT_PTR Materialize_frozen_f32(GLOB_SCOPE* glob, const std::string& role,
                                    const std::vector<int64_t>& shape,
                                    const std::vector<float>& values,
                                    const SPOS& spos) {
  AIR_ASSERT(Element_count(shape) == values.size());
  std::vector<float> host = values;
  return New_array_const(
      glob, ("legacy_" + role).c_str(),
      static_cast<int64_t>(values.size()),
      glob->Prim_type(PRIMITIVE_TYPE::FLOAT_32), shape, host.data(), spos);
}

void Set_slot(NODE_PTR node, uint32_t value) {
  node->Set_attr(nn::core::ATTR::SLOT, &value, 1);
}

NODE_PTR Emit_legacy_native(
    TENSOR2VECTOR_CTX& ctx, const FROZEN_NATIVE_CASE& fixture,
    NODE_PTR ranked_input, const SPOS& spos) {
  GLOB_SCOPE* glob = ctx.Container()->Glob_scope();
  TENSOR2VECTOR_UTIL util(ctx);

  return std::visit(
      [&](const auto& plan) -> NODE_PTR {
        using PLAN = std::decay_t<decltype(plan)>;
        NODE_PTR weight = ctx.Container()->New_ldc(
            Materialize_frozen_f32(glob, "weight", plan._weight_shape,
                                   plan._weight, spos),
            spos);
        NODE_PTR bias = ctx.Container()->New_ldc(
            Materialize_frozen_f32(glob, "bias", plan._bias_shape,
                                   plan._bias, spos),
            spos);
        if constexpr (std::is_same_v<PLAN, FROZEN_BASELINE_GEMM_PLAN>) {
          AIR_ASSERT(plan._weight_shape ==
                     std::vector<int64_t>({plan._height, plan._width}));
          return util.New_gemm_metakernel(ranked_input, weight, bias,
                                          plan._need_mask, spos);
        } else if constexpr (
            std::is_same_v<PLAN, FROZEN_BASELINE_CONV_PLAN>) {
          NODE_PTR result = util.New_conv_metakernel(
              ranked_input, weight, bias, plan._alignment,
              static_cast<int>(plan._channel_in),
              static_cast<int>(plan._channel_out),
              static_cast<int>(plan._output_height),
              static_cast<int>(plan._output_width),
              static_cast<int>(plan._kernel_hw),
              static_cast<int>(plan._stride), spos);
          Set_slot(result, plan._slot);
          return result;
        } else if constexpr (std::is_same_v<PLAN, FROZEN_FAST_GEMM_PLAN>) {
          NODE_PTR input_block = util.Blocking_rot(
              ranked_input, plan._alignment, static_cast<int>(plan._kp),
              static_cast<int>(plan._input_replications),
              static_cast<int>(plan._block_size), spos);
          ADDR_DATUM_PTR result = util.New_gemm_metakernel_fast(
              input_block, weight, bias, plan._np, plan._kp,
              static_cast<int>(plan._block_size),
              static_cast<int>(plan._grid_size), spos);
          STMT_PTR comment = ctx.Container()->New_comment(
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
          ctx.Prepend(ctx.Container()->New_comment("gemm add bias", spos));
          NODE_PTR add =
              util.New_add(ctx.Container()->New_ld(result, spos), bias, spos);
          ctx.Prepend(ctx.Container()->New_st(add, result, spos));
          if (plan._need_mask) {
            util.Gen_clear_data_stmt(
                result, plan._n,
                ranked_input->Rtype()->Cast_to_arr()->Elem_type(), spos);
          }
          NODE_PTR loaded = ctx.Container()->New_ld(result, spos);
          Set_slot(loaded, plan._slot);
          return loaded;
        } else {
          NODE_PTR input_block = util.Blocking_rot(
              ranked_input, plan._alignment,
              static_cast<int>(plan._input_size),
              static_cast<int>(plan._input_duplications),
              static_cast<int>(plan._capacity_block), spos,
              static_cast<int>(plan._blocking_outer_depth));
          SHARD_MAP_PARAMS params{
              static_cast<int>(plan._num_grid),
              static_cast<int>(plan._num_block),
              static_cast<int>(plan._width_block),
              static_cast<int>(plan._width_block_data),
              static_cast<int>(plan._width_block_pad),
              static_cast<int>(plan._position_block),
              static_cast<int>(plan._capacity_block),
              static_cast<int>(plan._input_duplications)};
          NODE_PTR result = util.New_conv_metakernel_fast(
              input_block, weight, bias, plan._alignment,
              static_cast<int>(plan._channel_in),
              static_cast<int>(plan._channel_out),
              static_cast<int>(plan._output_height),
              static_cast<int>(plan._output_width),
              static_cast<int>(plan._kernel_hw),
              static_cast<int>(plan._group),
              static_cast<int>(plan._stride), &params,
              plan._need_mask, Null_ptr, spos);
          Set_slot(result, plan._slot);
          return result;
        }
      },
      fixture._legacy_plan);
}

struct NATIVE_VARIANT_IR {
  std::unique_ptr<GLOB_SCOPE> _glob;
  FUNC_SCOPE* _func_scope = nullptr;
};

template <typename EMITTER>
NATIVE_VARIANT_IR Build_native_variant(const char* function_name,
                                       uint32_t line,
                                       const FROZEN_NATIVE_CASE& test_case,
                                       int64_t active_slots,
                                       EMITTER&& emitter) {
  NATIVE_VARIANT_IR ir;
  ir._glob = std::make_unique<GLOB_SCOPE>(0, true);
  GLOB_SCOPE* glob = ir._glob.get();
  const SPOS spos(0, line, 1, 0);
  TYPE_PTR input_type = New_array_type(
      glob, std::string(function_name) + "_input",
      glob->Prim_type(test_case._runtime_input_type._element_type),
      test_case._runtime_input_type._shape, spos);
  TYPE_PTR result_type = New_array_type(
      glob, std::string(function_name) + "_result",
      glob->Prim_type(test_case._runtime_result_type._element_type),
      test_case._runtime_result_type._shape, spos);

  STR_PTR name = glob->New_str(function_name);
  FUNC_PTR function = glob->New_func(name, spos);
  function->Set_parent(glob->Comp_env_id());
  SIGNATURE_TYPE_PTR signature = glob->New_sig_type();
  glob->New_ret_param(result_type, signature);
  glob->New_param(glob->New_str("packed_input"), input_type, signature, spos);
  signature->Set_complete();
  glob->New_entry_point(signature, function, name, spos);

  ir._func_scope = &glob->New_func_scope(function);
  CONTAINER* container = &ir._func_scope->Container();
  STMT_PTR entry = container->New_func_entry(spos);
  NODE_PTR body = entry->Node()->Last_child();

  VECTOR_CTX vector_ctx;
  const int64_t emission_slots =
      active_slots == 0 ? test_case._target._num_slots : active_slots;
  AIR_ASSERT(emission_slots > 0 && emission_slots <= MAX_SLOT_ALLOWED);
  vector_ctx.Update_slot(static_cast<uint32_t>(emission_slots));
  VECTOR_CONFIG config;
  config._max_slots = static_cast<uint64_t>(emission_slots);
  config._conv_parallel = test_case._options._conv_parallel;
  config._mask_fuse = test_case._options._mask_fuse;
  config._selective_ss = test_case._options._selective_strided_slice;
  config._sharding = test_case._options._sharding;
  TENSOR2VECTOR_CTX lowering_ctx(container, vector_ctx, nullptr, config);
  lowering_ctx.Set_cur_func_scope(ir._func_scope);
  lowering_ctx.Push(body, body);
  NODE_PTR input = container->New_ld(ir._func_scope->Formal(0), spos);
  NODE_PTR result = emitter(lowering_ctx, input, spos);
  container->Stmt_list().Append(container->New_retv(result, spos));
  lowering_ctx.Pop(body, body);
  return ir;
}

class Tensor2VectorNativeAirOracle : public ::testing::Test {
protected:
  void SetUp() override {
    _had_core   = META_INFO::Valid_domain(air::core::CORE);
    _had_nn     = META_INFO::Valid_domain(nn::core::NN);
    _had_vector = META_INFO::Valid_domain(VECTOR_DOMAIN::ID);
    if (!_had_core) ASSERT_TRUE(air::core::Register_core());
    if (!_had_nn) ASSERT_TRUE(nn::core::Register_nn());
    if (!_had_vector) ASSERT_TRUE(Register_vector_domain());
  }

  void TearDown() override {
    // META_INFO has no per-domain unregister API. Rebuild the three domains
    // used by ut_nnvector so this fixture leaves the process state unchanged.
    META_INFO::Remove_all();
    if (_had_core) EXPECT_TRUE(air::core::Register_core());
    if (_had_nn) EXPECT_TRUE(nn::core::Register_nn());
    if (_had_vector) EXPECT_TRUE(Register_vector_domain());
  }

private:
  bool _had_core   = false;
  bool _had_nn     = false;
  bool _had_vector = false;
};

}  // namespace

TEST_F(Tensor2VectorNativeAirOracle,
       BaselineGemmFourByFourDirectNativeOracle) {
  VECTOR_CONFIG default_config;
  EXPECT_FALSE(default_config.Python_dsl());

  NATIVE_GEMM_IR fixture =
      Build_native_gemm("baseline_gemm_4x4", 11, 4, 4, false);
  ASSERT_TRUE(fixture._glob->Verify_ir());
  ASSERT_EQ(fixture._result->Rtype()->Cast_to_arr()->Shape(),
            std::vector<int64_t>({8}));

  const NATIVE_GEMM_STATS stats = Get_stats(fixture);
  EXPECT_EQ(stats._loops, 1U);
  EXPECT_EQ(stats._vector_muls, 1U);
  EXPECT_EQ(stats._slices, 1U);
  EXPECT_EQ(stats._slot_attrs, 0U);
  EXPECT_EQ(stats._rotations,
            (std::vector<std::vector<int>>{{-4}, {0, 1, 2, 3}}));

  const std::string normalized = Snapshot(fixture);
  EXPECT_EQ(Vector_kernel_sha256(normalized),
            "7ce2561a9967a2465146865c27b99636"
            "8afb362aa8221ef3e824f0231210fb4a");

  NATIVE_GEMM_IR renamed =
      Build_native_gemm("renamed_native_oracle", 91, 4, 4, false);
  ASSERT_TRUE(renamed._glob->Verify_ir());
  EXPECT_EQ(normalized, Snapshot(renamed));
}

TEST_F(Tensor2VectorNativeAirOracle,
       BaselineGemmTwoByEightReductionAndMaskOracle) {
  NATIVE_GEMM_IR fixture =
      Build_native_gemm("baseline_gemm_2x8", 17, 2, 8, true);
  ASSERT_TRUE(fixture._glob->Verify_ir());
  ASSERT_EQ(fixture._result->Rtype()->Cast_to_arr()->Shape(),
            std::vector<int64_t>({16}));

  const NATIVE_GEMM_STATS stats = Get_stats(fixture);
  EXPECT_EQ(stats._loops, 2U);
  EXPECT_EQ(stats._vector_muls, 2U);
  EXPECT_EQ(stats._slices, 1U);
  EXPECT_EQ(stats._slot_attrs, 0U);
  EXPECT_EQ(stats._rotations,
            (std::vector<std::vector<int>>{{-8}, {0, 1}, {2, 4}}));

  EXPECT_EQ(Vector_kernel_sha256(Snapshot(fixture)),
            "fe7aa356576094f22cc23ca600fea792"
            "b8545340bae40312051d7650c8123e34");
}

TEST_F(Tensor2VectorNativeAirOracle,
       PreparedCppNativeMatchesLegacyNativeForAllFourPlanKinds) {
  uint32_t line = 301;
  for (const FROZEN_NATIVE_CASE& test_case : Frozen_native_cases()) {
    SCOPED_TRACE(test_case._name);
    const VECTOR_KERNEL_PLANNING_REQUEST request =
        Frozen_oracle_request(test_case);
    const std::shared_ptr<const PREPARED_VECTOR_KERNEL_PLAN> prepared =
        Resolve_oracle_plan(request);
    ASSERT_NE(prepared, nullptr);
    ASSERT_EQ(Get_vector_kernel_plan_kind(prepared->Plan()),
              test_case._expected);

    NATIVE_VARIANT_IR legacy = Build_native_variant(
        "legacy_native_oracle", line++, test_case, 0,
        [&](TENSOR2VECTOR_CTX& ctx, NODE_PTR input, const SPOS& spos) {
          return Emit_legacy_native(ctx, test_case, input, spos);
        });
    NATIVE_VARIANT_IR planned = Build_native_variant(
        "prepared_native_oracle", line++, test_case, 0,
        [&](TENSOR2VECTOR_CTX& ctx, NODE_PTR input, const SPOS& spos) {
          return Emit_prepared_vector_kernel_native(ctx, *prepared, input, {},
                                                    spos);
        });
    NATIVE_VARIANT_IR changed_context = Build_native_variant(
        "prepared_changed_context_oracle", line++, test_case,
        test_case._target._num_slots * 2,
        [&](TENSOR2VECTOR_CTX& ctx, NODE_PTR input, const SPOS& spos) {
          return Emit_prepared_vector_kernel_native(ctx, *prepared, input, {},
                                                    spos);
        });
    ASSERT_TRUE(legacy._glob->Verify_ir());
    ASSERT_TRUE(planned._glob->Verify_ir());
    ASSERT_TRUE(changed_context._glob->Verify_ir());

    const std::string legacy_air =
        nn::vector::test::Normalize_vector_kernel_function(
            *legacy._func_scope);
    const std::string prepared_air =
        nn::vector::test::Normalize_vector_kernel_function(
            *planned._func_scope);
    EXPECT_EQ(Vector_kernel_sha256(legacy_air), test_case._air_hash);

    const nn::vector::test::VECTOR_KERNEL_AIR_COMPARE_RESULT comparison =
        nn::vector::test::Compare_normalized_vector_kernel_air(
            legacy_air, prepared_air);
    const size_t context_begin = comparison._mismatch_offset > 160
                                     ? comparison._mismatch_offset - 160
                                     : 0;
    EXPECT_TRUE(comparison._equal)
        << comparison._message << "\nlegacy: "
        << comparison._lhs.substr(context_begin, 320) << "\nprepared: "
        << comparison._rhs.substr(context_begin, 320);

    const nn::vector::test::VECTOR_KERNEL_AIR_COMPARE_RESULT
        context_comparison =
            nn::vector::test::Compare_normalized_vector_kernel_air(
                prepared_air,
                nn::vector::test::Normalize_vector_kernel_function(
                    *changed_context._func_scope));
    EXPECT_TRUE(context_comparison._equal) << context_comparison._message;
  }
}
