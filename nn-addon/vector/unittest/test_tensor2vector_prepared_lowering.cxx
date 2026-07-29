//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "air/base/meta_info.h"
#include "air/base/st.h"
#include "air/core/opcode.h"
#include "nn/core/attr.h"
#include "nn/core/opcode.h"
#include "nn/vector/config.h"
#include "nn/vector/skip_lowering.h"
#include "nn/vector/tensor2vector_dsl.h"
#include "nn/vector/tensor2vector_planning.h"
#include "nn/vector/vector_gen.h"
#include "nn/vector/vector_opcode.h"
#include "tensor2vector_air_normalizer.h"

using namespace air::base;
using namespace nn::vector;
using namespace nn::vector::test;

namespace {

enum class SOURCE_KERNEL {
  GEMM,
  CONV,
};

struct SOURCE_KERNEL_IR {
  std::unique_ptr<GLOB_SCOPE> _glob;
  FUNC_ID                     _function_id;
  std::string                 _function_name;
};

SOURCE_KERNEL_IR Build_source_kernel(SOURCE_KERNEL kernel,
                                     const char* function_name,
                                     uint32_t line,
                                     bool dsl_identity_shape = false,
                                     std::vector<int64_t> gemm_input_shape =
                                         {}) {
  SOURCE_KERNEL_IR fixture;
  fixture._glob = std::make_unique<GLOB_SCOPE>(0, true);
  fixture._function_name = function_name;
  GLOB_SCOPE& glob = *fixture._glob;
  const SPOS spos(0, line, 1, 0);
  TYPE_PTR f32 = glob.Prim_type(PRIMITIVE_TYPE::FLOAT_32);

  std::vector<int64_t> input_shape;
  std::vector<int64_t> weight_shape;
  std::vector<int64_t> bias_shape;
  std::vector<int64_t> result_shape;
  if (kernel == SOURCE_KERNEL::GEMM) {
    // Width/height 128 match the driver's minimum slot target, so the
    // baseline prepared input and result types are both [128]. This lets the
    // DSL recipe be a realistic identity kernel without creating an AIR type
    // during recipe selection.
    const int64_t width = dsl_identity_shape ? 128 : 4;
    const int64_t height = dsl_identity_shape ? 128 : 2;
    input_shape = gemm_input_shape.empty() ? std::vector<int64_t>{width}
                                           : std::move(gemm_input_shape);
    weight_shape = {height, width};
    bias_shape = {height};
    result_shape = {height};
  } else {
    input_shape = {1, 1, 4, 4};
    weight_shape = {1, 1, 3, 3};
    bias_shape = {1};
    result_shape = {1, 1, 4, 4};
  }

  TYPE_PTR input_type = New_array_type(
      &glob, std::string(function_name) + "_input", f32, input_shape, spos);
  TYPE_PTR result_type = New_array_type(
      &glob, std::string(function_name) + "_result", f32, result_shape, spos);
  STR_PTR name = glob.New_str(function_name);
  FUNC_PTR function = glob.New_func(name, spos);
  function->Set_parent(glob.Comp_env_id());
  SIGNATURE_TYPE_PTR signature = glob.New_sig_type();
  glob.New_ret_param(result_type, signature);
  glob.New_param("input", input_type, signature, spos);
  signature->Set_complete();
  glob.New_entry_point(signature, function, name, spos);

  FUNC_SCOPE& scope = glob.New_func_scope(function);
  fixture._function_id = scope.Id();
  CONTAINER& container = scope.Container();
  container.New_func_entry(spos);

  size_t weight_count = 1;
  for (int64_t dimension : weight_shape) {
    weight_count *= static_cast<size_t>(dimension);
  }
  std::vector<float> weight(weight_count);
  for (size_t idx = 0; idx < weight.size(); ++idx) {
    weight[idx] = static_cast<float>((idx % 19) + 1) / 32.0F;
  }
  std::vector<float> bias(static_cast<size_t>(bias_shape[0]));
  for (size_t idx = 0; idx < bias.size(); ++idx) {
    bias[idx] = static_cast<float>(idx + 1) / 64.0F;
  }
  CONSTANT_PTR weight_constant = New_array_const(
      &glob, std::string(function_name) + "_weight", weight.size(), f32,
      weight_shape, weight.data(), spos);
  CONSTANT_PTR bias_constant = New_array_const(
      &glob, std::string(function_name) + "_bias", bias.size(), f32,
      bias_shape, bias.data(), spos);

  const nn::core::OPCODE source_opcode =
      kernel == SOURCE_KERNEL::GEMM ? nn::core::OPCODE::GEMM
                                    : nn::core::OPCODE::CONV;
  NODE_PTR source = container.New_cust_node(
      OPCODE(nn::core::NN, source_opcode), result_type, spos);
  source->Set_child(0, container.New_ld(scope.Formal(0), spos));
  source->Set_child(1, container.New_ldc(weight_constant, spos));
  source->Set_child(2, container.New_ldc(bias_constant, spos));
  if (kernel == SOURCE_KERNEL::CONV) {
    int group = 1;
    int strides[] = {1, 1};
    int pads[] = {1, 1, 1, 1};
    source->Set_attr(nn::core::ATTR::GROUP, &group, 1);
    source->Set_attr(nn::core::ATTR::STRIDE, strides, 2);
    source->Set_attr(nn::core::ATTR::PAD, pads, 4);
  }
  container.Stmt_list().Append(container.New_retv(source, spos));
  return fixture;
}

SOURCE_KERNEL_IR Build_sharding_conv_kernel(const char* function_name,
                                            uint32_t line) {
  SOURCE_KERNEL_IR fixture;
  fixture._glob = std::make_unique<GLOB_SCOPE>(0, true);
  fixture._function_name = function_name;
  GLOB_SCOPE& glob = *fixture._glob;
  const SPOS spos(0, line, 1, 0);
  TYPE_PTR f32 = glob.Prim_type(PRIMITIVE_TYPE::FLOAT_32);
  const std::vector<int64_t> input_shape{1, 1, 12, 16};
  const std::vector<int64_t> weight_shape{2, 1, 3, 3};
  const std::vector<int64_t> bias_shape{2};
  const std::vector<int64_t> result_shape{1, 2, 12, 16};
  TYPE_PTR input_type = New_array_type(
      &glob, std::string(function_name) + "_input", f32, input_shape, spos);
  TYPE_PTR result_type = New_array_type(
      &glob, std::string(function_name) + "_result", f32, result_shape,
      spos);

  STR_PTR name = glob.New_str(function_name);
  FUNC_PTR function = glob.New_func(name, spos);
  function->Set_parent(glob.Comp_env_id());
  SIGNATURE_TYPE_PTR signature = glob.New_sig_type();
  glob.New_ret_param(result_type, signature);
  glob.New_param("input", input_type, signature, spos);
  signature->Set_complete();
  glob.New_entry_point(signature, function, name, spos);

  FUNC_SCOPE& scope = glob.New_func_scope(function);
  fixture._function_id = scope.Id();
  CONTAINER& container = scope.Container();
  container.New_func_entry(spos);

  std::vector<float> weight(18);
  for (size_t idx = 0; idx < weight.size(); ++idx) {
    weight[idx] = static_cast<float>((idx % 11) + 1) / 16.0F;
  }
  std::vector<float> bias{0.125F, -0.25F};
  CONSTANT_PTR weight_constant = New_array_const(
      &glob, std::string(function_name) + "_weight", weight.size(), f32,
      weight_shape, weight.data(), spos);
  CONSTANT_PTR bias_constant = New_array_const(
      &glob, std::string(function_name) + "_bias", bias.size(), f32,
      bias_shape, bias.data(), spos);

  NODE_PTR conv = container.New_cust_node(
      OPCODE(nn::core::NN, nn::core::OPCODE::CONV), result_type, spos);
  conv->Set_child(0, container.New_ld(scope.Formal(0), spos));
  conv->Set_child(1, container.New_ldc(weight_constant, spos));
  conv->Set_child(2, container.New_ldc(bias_constant, spos));
  int group = 1;
  int strides[] = {1, 1};
  int pads[] = {1, 1, 1, 1};
  conv->Set_attr(nn::core::ATTR::GROUP, &group, 1);
  conv->Set_attr(nn::core::ATTR::STRIDE, strides, 2);
  conv->Set_attr(nn::core::ATTR::PAD, pads, 4);

  ADDR_DATUM_PTR output = scope.New_var(result_type, "output", spos);
  container.Stmt_list().Append(container.New_st(conv, output, spos));
  container.Stmt_list().Append(
      container.New_retv(container.New_ld(output, spos), spos));
  return fixture;
}

struct AIR_COUNTS {
  uint32_t _calls = 0;
  uint32_t _ldps = 0;
  uint32_t _vector_nodes = 0;
  uint32_t _reshapes = 0;
  uint32_t _nn_gemms = 0;
  uint32_t _nn_convs = 0;
  std::vector<STMT_PTR> _call_statements;
};

void Collect_counts(NODE_PTR node, AIR_COUNTS* counts) {
  if (node->Is_call()) {
    ++counts->_calls;
    counts->_call_statements.push_back(node->Stmt());
  }
  if (node->Opcode() == air::core::OPC_LDP) ++counts->_ldps;
  if (node->Domain() == VECTOR_DOMAIN::ID) ++counts->_vector_nodes;
  if (node->Opcode() == nn::vector::OPC_RESHAPE) ++counts->_reshapes;
  if (node->Opcode() ==
      OPCODE(nn::core::NN, nn::core::OPCODE::GEMM)) {
    ++counts->_nn_gemms;
  }
  if (node->Opcode() ==
      OPCODE(nn::core::NN, nn::core::OPCODE::CONV)) {
    ++counts->_nn_convs;
  }
  if (node->Is_block()) {
    for (STMT_PTR statement = node->Begin_stmt();
         statement != node->End_stmt(); statement = statement->Next()) {
      Collect_counts(statement->Node(), counts);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      Collect_counts(node->Child(idx), counts);
    }
  }
}

AIR_COUNTS Get_counts(const FUNC_SCOPE& scope) {
  AIR_COUNTS counts;
  Collect_counts(scope.Container().Entry_node(), &counts);
  return counts;
}

struct PLACEMENT_STATS {
  std::vector<uint32_t> _reshape_store_depths;
  std::vector<uint32_t> _call_depths;
};

void Collect_placement(NODE_PTR node, uint32_t block_depth,
                       PLACEMENT_STATS* stats) {
  if (node->Opcode() == air::core::OPC_STP && node->Num_child() > 0 &&
      node->Child(0)->Opcode() == nn::vector::OPC_RESHAPE) {
    stats->_reshape_store_depths.push_back(block_depth);
  }
  if (node->Is_call()) stats->_call_depths.push_back(block_depth);
  if (node->Is_block()) {
    for (STMT_PTR statement = node->Begin_stmt();
         statement != node->End_stmt(); statement = statement->Next()) {
      Collect_placement(statement->Node(), block_depth, stats);
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      NODE_PTR child = node->Child(idx);
      Collect_placement(child, block_depth + (child->Is_block() ? 1U : 0U),
                        stats);
    }
  }
}

NODE_PTR Find_preg_load(NODE_PTR node, PREG_ID preg_id) {
  if (node->Opcode() == air::core::OPC_LDP && node->Preg_id() == preg_id) {
    return node;
  }
  if (node->Is_block()) {
    for (STMT_PTR statement = node->Begin_stmt();
         statement != node->End_stmt(); statement = statement->Next()) {
      NODE_PTR found = Find_preg_load(statement->Node(), preg_id);
      if (found != Null_ptr) return found;
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      NODE_PTR found = Find_preg_load(node->Child(idx), preg_id);
      if (found != Null_ptr) return found;
    }
  }
  return Null_ptr;
}

bool Is_s32_scale(NODE_PTR node, uint64_t scale) {
  return node->Opcode() == air::core::OPC_MUL && node->Num_child() == 2 &&
         node->Child(0)->Rtype()->Is_prim() &&
         node->Child(0)->Rtype()->Cast_to_prim()->Encoding() ==
             PRIMITIVE_TYPE::INT_S32 &&
         node->Child(1)->Opcode() == air::core::OPC_INTCONST &&
         node->Child(1)->Intconst() == scale;
}

bool Has_scaled_weight_slice(NODE_PTR node, uint64_t scale) {
  if (node->Opcode() ==
          OPCODE(VECTOR_DOMAIN::ID, VECTOR_OPCODE::SLICE) &&
      node->Num_child() >= 2) {
    NODE_PTR index = node->Child(1);
    if (index->Opcode() == air::core::OPC_ADD &&
        index->Num_child() == 2 && Is_s32_scale(index->Child(0), scale)) {
      return true;
    }
  }
  if (node->Is_block()) {
    for (STMT_PTR statement = node->Begin_stmt();
         statement != node->End_stmt(); statement = statement->Next()) {
      if (Has_scaled_weight_slice(statement->Node(), scale)) return true;
    }
  } else {
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
      if (Has_scaled_weight_slice(node->Child(idx), scale)) return true;
    }
  }
  return false;
}

uint32_t Function_count(GLOB_SCOPE& glob) {
  uint32_t count = 0;
  for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = glob.Begin_func_scope();
       iter != glob.End_func_scope(); ++iter) {
    ++count;
  }
  return count;
}

FUNC_SCOPE* Find_helper(GLOB_SCOPE& glob, FUNC_ID caller_id) {
  FUNC_SCOPE* helper = nullptr;
  for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = glob.Begin_func_scope();
       iter != glob.End_func_scope(); ++iter) {
    if ((*iter).Id() == caller_id) continue;
    EXPECT_EQ(helper, nullptr);
    helper = &*iter;
  }
  return helper;
}

FUNC_SCOPE* Find_function(GLOB_SCOPE& glob, const std::string& name) {
  for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = glob.Begin_func_scope();
       iter != glob.End_func_scope(); ++iter) {
    if (name == (*iter).Owning_func()->Name()->Char_str()) return &*iter;
  }
  return nullptr;
}

TYPE_PTR Find_ranked_type(GLOB_SCOPE& glob,
                          const VECTOR_KERNEL_RANKED_TYPE_PLAN& expected) {
  for (TYPE_ITER iter = glob.Begin_type(); iter != glob.End_type(); ++iter) {
    TYPE_PTR type = *iter;
    if (!type->Is_array()) continue;
    ARRAY_TYPE_PTR array = type->Cast_to_arr();
    TYPE_PTR element = array->Elem_type();
    if (array->Shape() == expected._shape && element->Is_prim() &&
        element->Cast_to_prim()->Encoding() == expected._element_type) {
      return type;
    }
  }
  return Null_ptr;
}

std::unique_ptr<GLOB_SCOPE> Lower(
    SOURCE_KERNEL_IR& source, VECTOR_CTX& vector_ctx,
    const VECTOR_CONFIG& config) {
  EXPECT_TRUE(source._glob->Verify_ir());
  return std::unique_ptr<GLOB_SCOPE>(
      Vector_driver(source._glob.get(), vector_ctx, nullptr, config));
}

class FORWARDING_CPP_PROVIDER final : public VECTOR_KERNEL_PLAN_PROVIDER {
public:
  const char* Name() const override { return "forwarding-cpp-test"; }

  VECTOR_KERNEL_PROVIDER_CALL_RESULT Plan(
      const VECTOR_KERNEL_PLANNING_REQUEST& request) const override {
    ++_call_count;
    _last_requested = request._requested_plan_kind;
    CPP_VECTOR_KERNEL_PLAN_PROVIDER cpp;
    VECTOR_KERNEL_PROVIDER_CALL_RESULT result = cpp.Plan(request);
    if (result._result.has_value()) {
      _last_plan_kind =
          Get_vector_kernel_plan_kind(result._result->_plan);
      result._result->_provenance = Name();
    }
    return result;
  }

  mutable uint32_t _call_count = 0;
  mutable VECTOR_KERNEL_REQUESTED_PLAN_KIND _last_requested =
      VECTOR_KERNEL_REQUESTED_PLAN_KIND::AUTO;
  mutable VECTOR_KERNEL_PLAN_KIND _last_plan_kind =
      VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM;
};

class SHARDED_CPP_PROVIDER final : public VECTOR_KERNEL_PLAN_PROVIDER {
public:
  const char* Name() const override { return "sharded-cpp-test"; }

  VECTOR_KERNEL_PROVIDER_CALL_RESULT Plan(
      const VECTOR_KERNEL_PLANNING_REQUEST& request) const override {
    ++_call_count;
    _input_shape = request._operand_types[0]._shape;
    _weight_operand_shape = request._operand_types[1]._shape;
    _source_weight_shape = request._source_constants.front()._type._shape;
    _runtime_scalar_types = request._runtime_scalar_types;
    CPP_VECTOR_KERNEL_PLAN_PROVIDER cpp;
    VECTOR_KERNEL_PROVIDER_CALL_RESULT result = cpp.Plan(request);
    if (result._result.has_value()) {
      result._result->_provenance = Name();
      _plan_kind = Get_vector_kernel_plan_kind(result._result->_plan);
      if (const FAST_CONV_PLAN* plan =
              std::get_if<FAST_CONV_PLAN>(&result._result->_plan)) {
        _outer_block_depth = plan->_blocking_outer_depth;
        if (plan->_sharding_offset.has_value()) {
          _offset_type = plan->_sharding_offset->_type;
          _offset_scale = plan->_sharding_offset->_scale;
        }
      }
      if (!result._result->_runtime_preparations.empty()) {
        const VECTOR_KERNEL_RUNTIME_PREPARATION& runtime =
            result._result->_runtime_preparations.front();
        _runtime_input_shape = runtime._result_type._shape;
        _runtime_kind = runtime._kind;
      }
      if (!result._result->_scalar_preparations.empty()) {
        const VECTOR_KERNEL_SCALAR_PREPARATION& scalar =
            result._result->_scalar_preparations.front();
        _scalar_role = scalar._role;
        _scalar_source_operand = scalar._source_operand;
        _scalar_type = scalar._type;
        _scalar_scale = scalar._scale;
      }
    }
    return result;
  }

  mutable uint32_t _call_count = 0;
  mutable std::vector<int64_t> _input_shape;
  mutable std::vector<int64_t> _weight_operand_shape;
  mutable std::vector<int64_t> _source_weight_shape;
  mutable std::vector<PRIMITIVE_TYPE> _runtime_scalar_types;
  mutable std::vector<int64_t> _runtime_input_shape;
  mutable VECTOR_KERNEL_RUNTIME_PREPARATION_KIND _runtime_kind =
      VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::PACKED_VECTOR;
  mutable VECTOR_KERNEL_PLAN_KIND _plan_kind =
      VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV;
  mutable int64_t _outer_block_depth = 0;
  mutable PRIMITIVE_TYPE _offset_type = PRIMITIVE_TYPE::END;
  mutable int64_t _offset_scale = 0;
  mutable std::string _scalar_role;
  mutable uint32_t _scalar_source_operand = 0;
  mutable PRIMITIVE_TYPE _scalar_type = PRIMITIVE_TYPE::END;
  mutable int64_t _scalar_scale = 0;
};

void Expect_sharded_provider_contract(const SHARDED_CPP_PROVIDER& provider) {
  EXPECT_EQ(provider._call_count, 1U);
  EXPECT_EQ(provider._plan_kind, VECTOR_KERNEL_PLAN_KIND::FAST_CONV);
  EXPECT_EQ(provider._input_shape, (std::vector<int64_t>{1, 1, 8, 16}));
  EXPECT_EQ(provider._weight_operand_shape,
            (std::vector<int64_t>{1, 1, 3, 3}));
  EXPECT_EQ(provider._source_weight_shape,
            (std::vector<int64_t>{2, 1, 1, 3, 3}));
  EXPECT_EQ(provider._runtime_scalar_types,
            (std::vector<PRIMITIVE_TYPE>{PRIMITIVE_TYPE::INT_S32}));
  EXPECT_EQ(provider._runtime_input_shape, (std::vector<int64_t>{128}));
  EXPECT_EQ(provider._runtime_kind,
            VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::BLOCKING_ROTATIONS);
  EXPECT_EQ(provider._outer_block_depth, 1);
  EXPECT_EQ(provider._offset_type, PRIMITIVE_TYPE::INT_S32);
  EXPECT_EQ(provider._offset_scale, 9);
  EXPECT_EQ(provider._scalar_role, "weight-offset");
  EXPECT_EQ(provider._scalar_source_operand, 1U);
  EXPECT_EQ(provider._scalar_type, PRIMITIVE_TYPE::INT_S32);
  EXPECT_EQ(provider._scalar_scale, 9);
}

class Tensor2VectorPreparedLowering : public ::testing::Test {
protected:
  void SetUp() override {
    _had_core = META_INFO::Valid_domain(air::core::CORE);
    _had_nn = META_INFO::Valid_domain(nn::core::NN);
    _had_vector = META_INFO::Valid_domain(VECTOR_DOMAIN::ID);
    if (!_had_core) ASSERT_TRUE(air::core::Register_core());
    if (!_had_nn) ASSERT_TRUE(nn::core::Register_nn());
    if (!_had_vector) ASSERT_TRUE(Register_vector_domain());

    const std::set<std::string>& skip_ops =
        SKIP_LOWERING_REGISTRY::Instance().Get_skip_ops();
    _saved_skip_ops.assign(skip_ops.begin(), skip_ops.end());
    Clear_skip_lowering_ops();
  }

  void TearDown() override {
    Clear_skip_lowering_ops();
    Set_skip_lowering_ops(_saved_skip_ops);

    META_INFO::Remove_all();
    if (_had_core) EXPECT_TRUE(air::core::Register_core());
    if (_had_nn) EXPECT_TRUE(nn::core::Register_nn());
    if (_had_vector) EXPECT_TRUE(Register_vector_domain());
  }

private:
  bool _had_core = false;
  bool _had_nn = false;
  bool _had_vector = false;
  std::vector<std::string> _saved_skip_ops;
};

}  // namespace

TEST_F(Tensor2VectorPreparedLowering,
       DefaultAndExplicitCppNativeAutoProduceIdenticalAir) {
  for (SOURCE_KERNEL kernel : {SOURCE_KERNEL::GEMM, SOURCE_KERNEL::CONV}) {
    SCOPED_TRACE(kernel == SOURCE_KERNEL::GEMM ? "Gemm" : "Conv");
    SOURCE_KERNEL_IR default_source =
        Build_source_kernel(kernel, "prepared_default", 101);
    VECTOR_CTX default_ctx;
    VECTOR_CONFIG default_config;
    std::unique_ptr<GLOB_SCOPE> default_lowered =
        Lower(default_source, default_ctx, default_config);
    ASSERT_NE(default_lowered, nullptr);
    ASSERT_TRUE(default_lowered->Verify_ir());

    SOURCE_KERNEL_IR explicit_source =
        Build_source_kernel(kernel, "prepared_explicit", 137);
    VECTOR_CTX explicit_ctx;
    VECTOR_CONFIG explicit_config;
    explicit_config._plan_provider = "cpp";
    explicit_config._kernel_impl = "native";
    explicit_config._plan_kind = "auto";
    explicit_config._fallback = "error";
    std::unique_ptr<GLOB_SCOPE> explicit_lowered =
        Lower(explicit_source, explicit_ctx, explicit_config);
    ASSERT_NE(explicit_lowered, nullptr);
    ASSERT_TRUE(explicit_lowered->Verify_ir());

    const FUNC_SCOPE& default_scope =
        default_lowered->Open_func_scope(default_source._function_id);
    const FUNC_SCOPE& explicit_scope =
        explicit_lowered->Open_func_scope(explicit_source._function_id);
    EXPECT_EQ(Normalize_vector_kernel_function(default_scope),
              Normalize_vector_kernel_function(explicit_scope));

    const AIR_COUNTS counts = Get_counts(default_scope);
    EXPECT_EQ(counts._nn_gemms, 0U);
    EXPECT_EQ(counts._nn_convs, 0U);
    EXPECT_GT(counts._vector_nodes, 0U);
  }
}

TEST_F(Tensor2VectorPreparedLowering,
       RankedGemmDefaultNativeDoesNotInsertReshape) {
  SOURCE_KERNEL_IR source = Build_source_kernel(
      SOURCE_KERNEL::GEMM, "prepared_ranked_gemm", 181, false, {2, 2});
  VECTOR_CTX vector_ctx;
  VECTOR_CONFIG config;
  std::unique_ptr<GLOB_SCOPE> lowered = Lower(source, vector_ctx, config);

  ASSERT_NE(lowered, nullptr);
  ASSERT_TRUE(lowered->Verify_ir());
  const FUNC_SCOPE& caller =
      lowered->Open_func_scope(source._function_id);
  const AIR_COUNTS counts = Get_counts(caller);
  EXPECT_EQ(counts._nn_gemms, 0U);
  EXPECT_EQ(counts._reshapes, 0U);
  EXPECT_GT(counts._vector_nodes, 0U);
}

TEST_F(Tensor2VectorPreparedLowering,
       ForcedCppNativePlansReachValidVectorAir) {
  struct CASE {
    SOURCE_KERNEL _kernel;
    const char* _plan_kind;
    VECTOR_KERNEL_REQUESTED_PLAN_KIND _requested;
    VECTOR_KERNEL_PLAN_KIND _expected;
  };
  const CASE cases[] = {
      {SOURCE_KERNEL::GEMM, "baseline-gemm",
       VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM,
       VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM},
      {SOURCE_KERNEL::GEMM, "fast-gemm",
       VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_GEMM,
       VECTOR_KERNEL_PLAN_KIND::FAST_GEMM},
      {SOURCE_KERNEL::CONV, "baseline-conv",
       VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_CONV,
       VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV},
      {SOURCE_KERNEL::CONV, "fast-conv",
       VECTOR_KERNEL_REQUESTED_PLAN_KIND::FAST_CONV,
       VECTOR_KERNEL_PLAN_KIND::FAST_CONV},
  };

  uint32_t line = 211;
  for (const CASE& test_case : cases) {
    SCOPED_TRACE(test_case._plan_kind);
    SOURCE_KERNEL_IR source =
        Build_source_kernel(test_case._kernel, test_case._plan_kind, line++);
    FORWARDING_CPP_PROVIDER provider;
    VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY providers;
    ASSERT_TRUE(providers.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
                                   &provider));
    VECTOR_CTX vector_ctx;
    vector_ctx.Set_vector_kernel_plan_provider_registry(&providers);
    VECTOR_CONFIG config;
    config._plan_provider = "cpp";
    config._kernel_impl = "native";
    config._plan_kind = test_case._plan_kind;
    config._fallback = "error";
    std::unique_ptr<GLOB_SCOPE> lowered = Lower(source, vector_ctx, config);
    ASSERT_NE(lowered, nullptr);
    ASSERT_TRUE(lowered->Verify_ir());
    ASSERT_EQ(Function_count(*lowered), 1U);
    EXPECT_EQ(provider._call_count, 1U);
    EXPECT_EQ(provider._last_requested, test_case._requested);
    EXPECT_EQ(provider._last_plan_kind, test_case._expected);

    const FUNC_SCOPE& caller =
        lowered->Open_func_scope(source._function_id);
    const AIR_COUNTS counts = Get_counts(caller);
    EXPECT_EQ(counts._calls, 0U);
    EXPECT_EQ(counts._nn_gemms, 0U);
    EXPECT_EQ(counts._nn_convs, 0U);
    EXPECT_GT(counts._vector_nodes, 0U);
  }
}

TEST_F(Tensor2VectorPreparedLowering,
       InjectedValidatedProviderReachesNativeHandler) {
  SOURCE_KERNEL_IR source =
      Build_source_kernel(SOURCE_KERNEL::GEMM, "prepared_fake_provider", 307);
  FORWARDING_CPP_PROVIDER provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY providers;
  ASSERT_TRUE(providers.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
                                 &provider));
  VECTOR_CTX vector_ctx;
  vector_ctx.Set_vector_kernel_plan_provider_registry(&providers);
  VECTOR_CONFIG config;
  config._plan_provider = "cpp";
  config._kernel_impl = "native";
  config._plan_kind = "baseline-gemm";
  config._fallback = "error";

  std::unique_ptr<GLOB_SCOPE> lowered = Lower(source, vector_ctx, config);
  ASSERT_NE(lowered, nullptr);
  ASSERT_TRUE(lowered->Verify_ir());
  EXPECT_EQ(provider._call_count, 1U);
  EXPECT_EQ(provider._last_requested,
            VECTOR_KERNEL_REQUESTED_PLAN_KIND::BASELINE_GEMM);

  const FUNC_SCOPE& caller =
      lowered->Open_func_scope(source._function_id);
  const AIR_COUNTS counts = Get_counts(caller);
  EXPECT_EQ(counts._calls, 0U);
  EXPECT_EQ(counts._nn_gemms, 0U);
  EXPECT_GT(counts._vector_nodes, 0U);
}

TEST_F(Tensor2VectorPreparedLowering,
       PreparedPlanRecipeCreatesExactTypedCallLdpBridge) {
  SOURCE_KERNEL_IR source = Build_source_kernel(
      SOURCE_KERNEL::GEMM, "prepared_dsl_recipe", 401, true);
  FORWARDING_CPP_PROVIDER provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY providers;
  ASSERT_TRUE(providers.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
                                 &provider));
  uint32_t recipe_count = 0;
  uint32_t body_builder_count = 0;
  std::string expected_key;
  std::string expected_name;
  VECTOR_KERNEL_LOWERING_REGISTRY recipes;
  ASSERT_TRUE(recipes.Register(
      VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM,
      [&](const PREPARED_VECTOR_KERNEL_PLAN& prepared,
          const std::vector<NODE_PTR>& actuals, GLOB_SCOPE& destination,
          const VECTOR_KERNEL_DESTINATION_ABI& abi) {
        ++recipe_count;
        EXPECT_EQ(Get_vector_kernel_plan_kind(prepared.Plan()),
                  VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM);
        EXPECT_EQ(prepared.Provenance(), provider.Name());
        EXPECT_EQ(actuals.size(), 1U);
        EXPECT_EQ(actuals[0]->Container()->Glob_scope(), &destination);
        EXPECT_EQ(abi._formal_types.size(), actuals.size());
        if (!abi._formal_types.empty()) {
          EXPECT_TRUE(
              abi._formal_types[0]->Is_compatible_type(actuals[0]->Rtype()));
        }
        EXPECT_TRUE(
            abi._result_type->Is_compatible_type(actuals[0]->Rtype()));
        expected_key = prepared.Specialization_key();
        expected_name = prepared.Helper_name();

        VECTOR_KERNEL_HELPER_SPEC spec;
        spec._formal_types = abi._formal_types;
        spec._result_type = abi._result_type;
        spec._build_body =
            [&](FUNC_SCOPE& helper, NODE_PTR, const SPOS& spos) {
              ++body_builder_count;
              return helper.Container().New_ld(helper.Formal(0), spos);
            };
        return spec;
      }));

  VECTOR_CTX vector_ctx;
  vector_ctx.Set_vector_kernel_plan_provider_registry(&providers);
  vector_ctx.Set_vector_kernel_lowering_registry(&recipes);
  VECTOR_CONFIG config;
  config._plan_provider = "cpp";
  config._kernel_impl = "dsl";
  config._plan_kind = "baseline-gemm";
  config._fallback = "error";
  std::unique_ptr<GLOB_SCOPE> lowered = Lower(source, vector_ctx, config);

  ASSERT_NE(lowered, nullptr);
  ASSERT_TRUE(lowered->Verify_ir());
  EXPECT_EQ(provider._call_count, 1U);
  EXPECT_EQ(recipe_count, 1U);
  EXPECT_EQ(body_builder_count, 1U);
  EXPECT_EQ(Function_count(*lowered), 2U);

  FUNC_SCOPE& caller = lowered->Open_func_scope(source._function_id);
  FUNC_SCOPE* helper = Find_helper(*lowered, caller.Id());
  ASSERT_NE(helper, nullptr);
  EXPECT_EQ(expected_name, helper->Owning_func()->Name()->Char_str());
  EXPECT_FALSE(expected_key.empty());

  const AIR_COUNTS caller_counts = Get_counts(caller);
  ASSERT_EQ(caller_counts._calls, 1U);
  EXPECT_EQ(caller_counts._ldps, 1U);
  EXPECT_EQ(caller_counts._nn_gemms, 0U);
  EXPECT_EQ(caller_counts._nn_convs, 0U);
  const AIR_COUNTS helper_counts = Get_counts(*helper);
  EXPECT_EQ(helper_counts._calls, 0U);
  EXPECT_EQ(helper_counts._nn_gemms, 0U);
  EXPECT_EQ(helper_counts._nn_convs, 0U);

  NODE_PTR caller_body = caller.Container().Entry_node()->Last_child();
  STMT_PTR terminal = Null_ptr;
  for (STMT_PTR statement = caller_body->Begin_stmt();
       statement != caller_body->End_stmt(); statement = statement->Next()) {
    terminal = statement;
  }
  ASSERT_NE(terminal, Null_ptr);
  ASSERT_EQ(terminal->Node()->Opcode(), air::core::OPC_RETV);
  NODE_PTR replacement = terminal->Node()->Child(0);
  ASSERT_EQ(replacement->Opcode(), air::core::OPC_LDP);
  STMT_PTR call = caller_counts._call_statements.front();
  ASSERT_EQ(call->Node()->Num_arg(), 1U);
  std::vector<NODE_PTR> actuals{call->Node()->Child(0)};
  const VECTOR_KERNEL_AIR_COMPARE_RESULT bridge =
      Check_vector_kernel_call_bridge(caller, call, actuals, replacement,
                                      *helper);
  EXPECT_TRUE(bridge._equal) << bridge._message;
}

TEST_F(Tensor2VectorPreparedLowering,
       ExplicitPreparedDslSelectionOverridesLegacySkipRegistry) {
  Set_skip_lowering_ops(
      {"nn::core::gemm", "nn::core::matmul", "nn::core::conv"});

  struct CASE {
    SOURCE_KERNEL _kernel;
    const char* _plan_kind;
    VECTOR_KERNEL_PLAN_KIND _expected_kind;
  };
  const CASE cases[] = {
      {SOURCE_KERNEL::GEMM, "baseline-gemm",
       VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM},
      {SOURCE_KERNEL::CONV, "baseline-conv",
       VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV},
  };

  uint32_t line = 503;
  for (const CASE& test_case : cases) {
    SCOPED_TRACE(test_case._plan_kind);
    EXPECT_TRUE(Should_skip_lowering(
        "nn::core",
        test_case._kernel == SOURCE_KERNEL::GEMM ? "gemm" : "conv"));
    SOURCE_KERNEL_IR source = Build_source_kernel(
        test_case._kernel, test_case._plan_kind, line++,
        test_case._kernel == SOURCE_KERNEL::GEMM);
    const FUNC_ID caller_id = source._function_id;

    uint32_t recipe_count = 0;
    uint32_t body_builder_count = 0;
    VECTOR_KERNEL_LOWERING_REGISTRY recipes;
    ASSERT_TRUE(recipes.Register(
        test_case._expected_kind,
        [&, caller_id](const PREPARED_VECTOR_KERNEL_PLAN& prepared,
                       const std::vector<NODE_PTR>& actuals,
                       GLOB_SCOPE& destination,
                       const VECTOR_KERNEL_DESTINATION_ABI&) {
          ++recipe_count;
          EXPECT_EQ(Get_vector_kernel_plan_kind(prepared.Plan()),
                    test_case._expected_kind);
          EXPECT_EQ(actuals.size(), 1U);
          EXPECT_EQ(actuals[0]->Container()->Glob_scope(), &destination);

          FUNC_SCOPE& caller = destination.Open_func_scope(caller_id);
          TYPE_PTR result_type =
              caller.Owning_func()
                  ->Entry_point()
                  ->Type()
                  ->Cast_to_sig()
                  ->Ret_param()
                  ->Type();
          VECTOR_KERNEL_HELPER_SPEC spec;
          spec._formal_types = {actuals[0]->Rtype()};
          spec._result_type = result_type;
          spec._build_body =
              [&, result_type](FUNC_SCOPE& helper, NODE_PTR,
                               const SPOS& spos) {
                ++body_builder_count;
                return helper.Container().New_zero(result_type, spos);
              };
          return spec;
        }));

    VECTOR_CTX vector_ctx;
    vector_ctx.Set_vector_kernel_lowering_registry(&recipes);
    VECTOR_CONFIG config;
    config._plan_provider = "cpp";
    config._kernel_impl = "dsl";
    config._plan_kind = test_case._plan_kind;
    config._fallback = "error";
    std::unique_ptr<GLOB_SCOPE> lowered = Lower(source, vector_ctx, config);

    ASSERT_NE(lowered, nullptr);
    ASSERT_TRUE(lowered->Verify_ir());
    EXPECT_EQ(recipe_count, 1U);
    EXPECT_EQ(body_builder_count, 1U);
    EXPECT_EQ(Function_count(*lowered), 2U);

    FUNC_SCOPE& caller = lowered->Open_func_scope(caller_id);
    FUNC_SCOPE* helper = Find_helper(*lowered, caller.Id());
    ASSERT_NE(helper, nullptr);
    const AIR_COUNTS caller_counts = Get_counts(caller);
    ASSERT_EQ(caller_counts._calls, 1U);
    EXPECT_GE(caller_counts._ldps, 1U);
    EXPECT_EQ(caller_counts._nn_gemms, 0U);
    EXPECT_EQ(caller_counts._nn_convs, 0U);
    const AIR_COUNTS helper_counts = Get_counts(*helper);
    EXPECT_EQ(helper_counts._calls, 0U);
    EXPECT_EQ(helper_counts._nn_gemms, 0U);
    EXPECT_EQ(helper_counts._nn_convs, 0U);

    NODE_PTR caller_body = caller.Container().Entry_node()->Last_child();
    STMT_PTR terminal = Null_ptr;
    for (STMT_PTR statement = caller_body->Begin_stmt();
         statement != caller_body->End_stmt(); statement = statement->Next()) {
      terminal = statement;
    }
    ASSERT_NE(terminal, Null_ptr);
    ASSERT_EQ(terminal->Node()->Opcode(), air::core::OPC_RETV);
    NODE_PTR replacement = terminal->Node()->Child(0);
    ASSERT_EQ(replacement->Opcode(), air::core::OPC_LDP);
    STMT_PTR call = caller_counts._call_statements.front();
    ASSERT_EQ(call->Node()->Num_arg(), 1U);
    std::vector<NODE_PTR> actuals{call->Node()->Child(0)};
    const VECTOR_KERNEL_AIR_COMPARE_RESULT bridge =
        Check_vector_kernel_call_bridge(caller, call, actuals, replacement,
                                        *helper);
    EXPECT_TRUE(bridge._equal) << bridge._message;
  }
}

TEST_F(Tensor2VectorPreparedLowering,
       LegacySkipClonesOnlyAfterValidCppNativePreflight) {
  Set_skip_lowering_ops(
      {"nn::core::gemm", "nn::core::matmul", "nn::core::conv"});
  struct CASE {
    SOURCE_KERNEL _kernel;
    const char* _plan_kind;
  };
  const CASE cases[] = {
      {SOURCE_KERNEL::GEMM, "baseline-gemm"},
      {SOURCE_KERNEL::CONV, "baseline-conv"},
  };
  uint32_t line = 571;
  for (const CASE& test_case : cases) {
    SCOPED_TRACE(test_case._plan_kind);
    SOURCE_KERNEL_IR source = Build_source_kernel(
        test_case._kernel, test_case._plan_kind, line++);
    FORWARDING_CPP_PROVIDER provider;
    VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY providers;
    ASSERT_TRUE(providers.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
                                   &provider));
    VECTOR_CTX vector_ctx;
    vector_ctx.Set_vector_kernel_plan_provider_registry(&providers);
    VECTOR_CONFIG config;
    config._plan_provider = "cpp";
    config._kernel_impl = "native";
    config._plan_kind = test_case._plan_kind;
    config._fallback = "error";
    std::unique_ptr<GLOB_SCOPE> lowered = Lower(source, vector_ctx, config);

    ASSERT_NE(lowered, nullptr);
    ASSERT_TRUE(lowered->Verify_ir());
    EXPECT_EQ(provider._call_count, 1U);
    const FUNC_SCOPE& caller =
        lowered->Open_func_scope(source._function_id);
    const AIR_COUNTS counts = Get_counts(caller);
    EXPECT_EQ(counts._calls, 0U);
    EXPECT_EQ(counts._vector_nodes, 0U);
    EXPECT_EQ(counts._nn_gemms,
              test_case._kernel == SOURCE_KERNEL::GEMM ? 1U : 0U);
    EXPECT_EQ(counts._nn_convs,
              test_case._kernel == SOURCE_KERNEL::CONV ? 1U : 0U);
  }
}

TEST_F(Tensor2VectorPreparedLowering,
       LegacySkipReportsCppNativeFallbackExactlyOnce) {
  Set_skip_lowering_ops(
      {"nn::core::gemm", "nn::core::matmul", "nn::core::conv"});
  uint32_t line = 591;
  for (SOURCE_KERNEL kernel : {SOURCE_KERNEL::GEMM, SOURCE_KERNEL::CONV}) {
    SCOPED_TRACE(kernel == SOURCE_KERNEL::GEMM ? "Gemm" : "Conv");
    SOURCE_KERNEL_IR source =
        Build_source_kernel(kernel, "skipped_provider_fallback", line++);
    VECTOR_CTX vector_ctx;
    VECTOR_CONFIG config;
    config._plan_provider = "python";
    config._kernel_impl = "native";
    config._plan_kind = "auto";
    config._fallback = "cpp-native";

    testing::internal::CaptureStderr();
    std::unique_ptr<GLOB_SCOPE> lowered = Lower(source, vector_ctx, config);
    const std::string stderr_output =
        testing::internal::GetCapturedStderr();

    ASSERT_NE(lowered, nullptr);
    ASSERT_TRUE(lowered->Verify_ir());
    const std::string expected_diagnostic =
        "vector-kernel fallback: requested provider unavailable; using "
        "cpp/native\n";
    ASSERT_GE(stderr_output.size(), expected_diagnostic.size());
    EXPECT_EQ(stderr_output.substr(stderr_output.size() -
                                   expected_diagnostic.size()),
              expected_diagnostic);
    EXPECT_EQ(stderr_output.find(expected_diagnostic),
              stderr_output.rfind(expected_diagnostic));

    const FUNC_SCOPE& caller =
        lowered->Open_func_scope(source._function_id);
    const AIR_COUNTS counts = Get_counts(caller);
    EXPECT_EQ(counts._calls, 0U);
    EXPECT_EQ(counts._vector_nodes, 0U);
    EXPECT_EQ(counts._nn_gemms,
              kernel == SOURCE_KERNEL::GEMM ? 1U : 0U);
    EXPECT_EQ(counts._nn_convs,
              kernel == SOURCE_KERNEL::CONV ? 1U : 0U);
  }
}

TEST_F(Tensor2VectorPreparedLowering,
       LegacySkipDoesNotHideUnavailableProviderOrInvalidSelector) {
  Set_skip_lowering_ops(
      {"nn::core::gemm", "nn::core::matmul", "nn::core::conv"});
  uint32_t line = 601;
  for (SOURCE_KERNEL kernel : {SOURCE_KERNEL::GEMM, SOURCE_KERNEL::CONV}) {
    SCOPED_TRACE(kernel == SOURCE_KERNEL::GEMM ? "Gemm" : "Conv");
    SOURCE_KERNEL_IR missing_provider = Build_source_kernel(
        kernel, "skipped_missing_provider", line++);
    VECTOR_CTX missing_provider_ctx;
    VECTOR_CONFIG missing_provider_config;
    missing_provider_config._plan_provider = "python";
    missing_provider_config._kernel_impl = "native";
    missing_provider_config._plan_kind = "auto";
    missing_provider_config._fallback = "error";
    EXPECT_DEATH(
        {
          std::unique_ptr<GLOB_SCOPE> lowered =
              Lower(missing_provider, missing_provider_ctx,
                    missing_provider_config);
        },
        "requested vector-kernel plan provider is unavailable");

    SOURCE_KERNEL_IR invalid_selector = Build_source_kernel(
        kernel, "skipped_invalid_selector", line++);
    VECTOR_CTX invalid_selector_ctx;
    VECTOR_CONFIG invalid_selector_config;
    invalid_selector_config._plan_provider = "cpp";
    invalid_selector_config._kernel_impl = "invalid";
    invalid_selector_config._plan_kind = "auto";
    invalid_selector_config._fallback = "error";
    EXPECT_DEATH(
        {
          std::unique_ptr<GLOB_SCOPE> lowered =
              Lower(invalid_selector, invalid_selector_ctx,
                    invalid_selector_config);
        },
        "invalid vector-kernel kernel_impl");
  }
}

TEST_F(Tensor2VectorPreparedLowering,
       ShardedFastConvPreparedNativeUsesOuterDepthAndS32Offset) {
  SOURCE_KERNEL_IR source =
      Build_sharding_conv_kernel("prepared_sharded_native", 641);
  SHARDED_CPP_PROVIDER provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY providers;
  ASSERT_TRUE(providers.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
                                 &provider));
  VECTOR_CTX vector_ctx;
  vector_ctx.Set_vector_kernel_plan_provider_registry(&providers);
  VECTOR_CONFIG config;
  config._sharding = true;
  config._max_slots = 128;
  config._plan_provider = "cpp";
  config._kernel_impl = "native";
  config._plan_kind = "fast-conv";
  config._fallback = "error";
  std::unique_ptr<GLOB_SCOPE> lowered = Lower(source, vector_ctx, config);

  ASSERT_NE(lowered, nullptr);
  ASSERT_TRUE(lowered->Verify_ir());
  Expect_sharded_provider_contract(provider);
  FUNC_SCOPE* caller = Find_function(*lowered, source._function_name);
  ASSERT_NE(caller, nullptr);
  const AIR_COUNTS counts = Get_counts(*caller);
  EXPECT_EQ(counts._calls, 0U);
  EXPECT_EQ(counts._nn_convs, 0U);
  EXPECT_GT(counts._vector_nodes, 0U);
  EXPECT_TRUE(Has_scaled_weight_slice(caller->Container().Entry_node(), 9));

  PLACEMENT_STATS placement;
  Collect_placement(caller->Container().Entry_node(), 0, &placement);
  ASSERT_EQ(placement._reshape_store_depths.size(), 1U);
  EXPECT_TRUE(placement._call_depths.empty());
  EXPECT_EQ(placement._reshape_store_depths.front(), 3U);
}

TEST_F(Tensor2VectorPreparedLowering,
       ShardedFastConvPreparedDslCreatesVectorAndS32TypedBridge) {
  SOURCE_KERNEL_IR source =
      Build_sharding_conv_kernel("prepared_sharded_dsl", 677);
  SHARDED_CPP_PROVIDER provider;
  VECTOR_KERNEL_PLAN_PROVIDER_REGISTRY providers;
  ASSERT_TRUE(providers.Register(VECTOR_KERNEL_PLAN_PROVIDER_KIND::CPP,
                                 &provider));
  uint32_t recipe_count = 0;
  uint32_t body_builder_count = 0;
  VECTOR_KERNEL_LOWERING_REGISTRY recipes;
  ASSERT_TRUE(recipes.Register(
      VECTOR_KERNEL_PLAN_KIND::FAST_CONV,
      [&](const PREPARED_VECTOR_KERNEL_PLAN& prepared,
          const std::vector<NODE_PTR>& actuals, GLOB_SCOPE& destination,
          const VECTOR_KERNEL_DESTINATION_ABI&) {
        ++recipe_count;
        EXPECT_EQ(Get_vector_kernel_plan_kind(prepared.Plan()),
                  VECTOR_KERNEL_PLAN_KIND::FAST_CONV);
        EXPECT_EQ(prepared.Provenance(), provider.Name());
        EXPECT_EQ(actuals.size(), 2U);
        AIR_ASSERT(actuals.size() == 2U);
        EXPECT_EQ(actuals[0]->Container()->Glob_scope(), &destination);
        EXPECT_EQ(actuals[1]->Container()->Glob_scope(), &destination);
        EXPECT_TRUE(actuals[0]->Rtype()->Is_array());
        EXPECT_EQ(actuals[0]->Rtype()->Cast_to_arr()->Shape(),
                  (std::vector<int64_t>{128}));
        EXPECT_TRUE(actuals[1]->Rtype()->Is_prim());
        EXPECT_EQ(actuals[1]->Rtype()->Cast_to_prim()->Encoding(),
                  PRIMITIVE_TYPE::INT_S32);
        EXPECT_EQ(prepared.Runtime_preparations().size(), 1U);
        AIR_ASSERT(prepared.Runtime_preparations().size() == 1U);
        EXPECT_EQ(prepared.Runtime_preparations().front()._outer_block_depth,
                  1U);
        EXPECT_EQ(prepared.Scalar_preparations().size(), 1U);
        AIR_ASSERT(prepared.Scalar_preparations().size() == 1U);
        EXPECT_EQ(prepared.Scalar_preparations().front()._role,
                  "weight-offset");
        EXPECT_EQ(prepared.Scalar_preparations().front()._type,
                  PRIMITIVE_TYPE::INT_S32);
        EXPECT_EQ(prepared.Scalar_preparations().front()._scale, 9);

        const VECTOR_KERNEL_COMMON_PLAN& common = std::visit(
            [](const auto& plan) -> const VECTOR_KERNEL_COMMON_PLAN& {
              return plan._common;
            },
            prepared.Plan());
        TYPE_PTR result_type = Find_ranked_type(destination,
                                                common._result_type);
        AIR_ASSERT(result_type != Null_ptr);
        EXPECT_EQ(result_type->Cast_to_arr()->Shape(),
                  (std::vector<int64_t>{1, 1, 8, 16}));

        VECTOR_KERNEL_HELPER_SPEC spec;
        spec._formal_types = {actuals[0]->Rtype(), actuals[1]->Rtype()};
        spec._result_type = result_type;
        spec._build_body =
            [&, result_type](FUNC_SCOPE& helper, NODE_PTR,
                             const SPOS& spos) {
              ++body_builder_count;
              return helper.Container().New_zero(result_type, spos);
            };
        return spec;
      }));

  VECTOR_CTX vector_ctx;
  vector_ctx.Set_vector_kernel_plan_provider_registry(&providers);
  vector_ctx.Set_vector_kernel_lowering_registry(&recipes);
  VECTOR_CONFIG config;
  config._sharding = true;
  config._max_slots = 128;
  config._plan_provider = "cpp";
  config._kernel_impl = "dsl";
  config._plan_kind = "fast-conv";
  config._fallback = "error";
  std::unique_ptr<GLOB_SCOPE> lowered = Lower(source, vector_ctx, config);

  ASSERT_NE(lowered, nullptr);
  ASSERT_TRUE(lowered->Verify_ir());
  Expect_sharded_provider_contract(provider);
  EXPECT_EQ(recipe_count, 1U);
  EXPECT_EQ(body_builder_count, 1U);
  EXPECT_EQ(Function_count(*lowered), 2U);
  FUNC_SCOPE* caller = Find_function(*lowered, source._function_name);
  ASSERT_NE(caller, nullptr);
  FUNC_SCOPE* helper = Find_helper(*lowered, caller->Id());
  ASSERT_NE(helper, nullptr);
  ASSERT_EQ(helper->Formal_cnt(), 2U);
  EXPECT_TRUE(helper->Formal(0)->Type()->Is_array());
  EXPECT_EQ(helper->Formal(0)->Type()->Cast_to_arr()->Shape(),
            (std::vector<int64_t>{128}));
  EXPECT_TRUE(helper->Formal(1)->Type()->Is_prim());
  EXPECT_EQ(helper->Formal(1)->Type()->Cast_to_prim()->Encoding(),
            PRIMITIVE_TYPE::INT_S32);

  const AIR_COUNTS caller_counts = Get_counts(*caller);
  ASSERT_EQ(caller_counts._calls, 1U);
  EXPECT_EQ(caller_counts._nn_convs, 0U);
  STMT_PTR call = caller_counts._call_statements.front();
  ASSERT_EQ(call->Node()->Num_arg(), 2U);
  NODE_PTR replacement = Find_preg_load(
      caller->Container().Entry_node(), call->Node()->Ret_preg_id());
  ASSERT_NE(replacement, Null_ptr);
  const std::vector<NODE_PTR> actuals{call->Node()->Child(0),
                                      call->Node()->Child(1)};
  const VECTOR_KERNEL_AIR_COMPARE_RESULT bridge =
      Check_vector_kernel_call_bridge(*caller, call, actuals, replacement,
                                      *helper);
  EXPECT_TRUE(bridge._equal) << bridge._message;

  PLACEMENT_STATS placement;
  Collect_placement(caller->Container().Entry_node(), 0, &placement);
  ASSERT_EQ(placement._reshape_store_depths.size(), 1U);
  ASSERT_EQ(placement._call_depths.size(), 1U);
  EXPECT_EQ(placement._reshape_store_depths.front(), 3U);
  EXPECT_EQ(placement._call_depths.front(), 4U);
}

TEST_F(Tensor2VectorPreparedLowering,
       MissingProviderHasErrorOrDiagnosedCppNativeFallback) {
  struct CASE {
    SOURCE_KERNEL _kernel;
    const char*   _plan_kind;
  };
  const CASE cases[] = {
      {SOURCE_KERNEL::GEMM, "baseline-gemm"},
      {SOURCE_KERNEL::CONV, "baseline-conv"},
  };

  uint32_t line = 607;
  for (const CASE& test_case : cases) {
    SCOPED_TRACE(test_case._plan_kind);
    SOURCE_KERNEL_IR error_source = Build_source_kernel(
        test_case._kernel, "missing_provider_error", line++);
    VECTOR_CTX error_ctx;
    VECTOR_CONFIG error_config;
    error_config._plan_provider = "python";
    error_config._kernel_impl = "dsl";
    error_config._plan_kind = test_case._plan_kind;
    error_config._fallback = "error";
    EXPECT_DEATH(
        {
          std::unique_ptr<GLOB_SCOPE> lowered =
              Lower(error_source, error_ctx, error_config);
        },
        "requested vector-kernel plan provider is unavailable");

    SOURCE_KERNEL_IR fallback_source = Build_source_kernel(
        test_case._kernel, "missing_provider_fallback", line++);
    VECTOR_CTX fallback_ctx;
    VECTOR_CONFIG fallback_config;
    fallback_config._plan_provider = "python";
    fallback_config._kernel_impl = "dsl";
    fallback_config._plan_kind = test_case._plan_kind;
    fallback_config._fallback = "cpp-native";

    testing::internal::CaptureStderr();
    std::unique_ptr<GLOB_SCOPE> lowered =
        Lower(fallback_source, fallback_ctx, fallback_config);
    const std::string stderr_output =
        testing::internal::GetCapturedStderr();

    ASSERT_NE(lowered, nullptr);
    ASSERT_TRUE(lowered->Verify_ir());
    const std::string expected_diagnostic =
        "vector-kernel fallback: requested provider unavailable; using "
        "cpp/native\n";
    ASSERT_GE(stderr_output.size(), expected_diagnostic.size());
    EXPECT_EQ(stderr_output.substr(stderr_output.size() -
                                   expected_diagnostic.size()),
              expected_diagnostic);
    EXPECT_EQ(stderr_output.find(expected_diagnostic),
              stderr_output.rfind(expected_diagnostic));

    EXPECT_EQ(Function_count(*lowered), 1U);
    const FUNC_SCOPE& caller =
        lowered->Open_func_scope(fallback_source._function_id);
    const AIR_COUNTS counts = Get_counts(caller);
    EXPECT_EQ(counts._calls, 0U);
    EXPECT_EQ(counts._nn_gemms, 0U);
    EXPECT_EQ(counts._nn_convs, 0U);
    EXPECT_GT(counts._vector_nodes, 0U);
  }
}

TEST_F(Tensor2VectorPreparedLowering,
       MissingDslRecipeHasErrorOrDiagnosedCppNativeFallback) {
  struct CASE {
    SOURCE_KERNEL _kernel;
    const char*   _plan_kind;
  };
  const CASE cases[] = {
      {SOURCE_KERNEL::GEMM, "baseline-gemm"},
      {SOURCE_KERNEL::CONV, "baseline-conv"},
  };

  uint32_t line = 617;
  for (const CASE& test_case : cases) {
    SCOPED_TRACE(test_case._plan_kind);
    SOURCE_KERNEL_IR error_source = Build_source_kernel(
        test_case._kernel, "missing_recipe_error", line++);
    VECTOR_CTX error_ctx;
    VECTOR_CONFIG error_config;
    error_config._plan_provider = "cpp";
    error_config._kernel_impl = "dsl";
    error_config._plan_kind = test_case._plan_kind;
    error_config._fallback = "error";
    EXPECT_DEATH(
        {
          std::unique_ptr<GLOB_SCOPE> lowered =
              Lower(error_source, error_ctx, error_config);
        },
        "no DSL recipe is registered for the validated plan kind");

    SOURCE_KERNEL_IR fallback_source = Build_source_kernel(
        test_case._kernel, "missing_recipe_fallback", line++);
    VECTOR_CTX fallback_ctx;
    VECTOR_CONFIG fallback_config;
    fallback_config._plan_provider = "cpp";
    fallback_config._kernel_impl = "dsl";
    fallback_config._plan_kind = test_case._plan_kind;
    fallback_config._fallback = "cpp-native";

    testing::internal::CaptureStderr();
    std::unique_ptr<GLOB_SCOPE> lowered =
        Lower(fallback_source, fallback_ctx, fallback_config);
    const std::string stderr_output =
        testing::internal::GetCapturedStderr();

    ASSERT_NE(lowered, nullptr);
    ASSERT_TRUE(lowered->Verify_ir());
    const std::string expected_diagnostic =
        "vector-kernel fallback: DSL recipe unavailable; using cpp/native\n";
    ASSERT_GE(stderr_output.size(), expected_diagnostic.size());
    EXPECT_EQ(stderr_output.substr(stderr_output.size() -
                                   expected_diagnostic.size()),
              expected_diagnostic);
    EXPECT_EQ(stderr_output.find(expected_diagnostic),
              stderr_output.rfind(expected_diagnostic));

    EXPECT_EQ(Function_count(*lowered), 1U);
    const FUNC_SCOPE& caller =
        lowered->Open_func_scope(fallback_source._function_id);
    const AIR_COUNTS counts = Get_counts(caller);
    EXPECT_EQ(counts._calls, 0U);
    EXPECT_EQ(counts._nn_gemms, 0U);
    EXPECT_EQ(counts._nn_convs, 0U);
    EXPECT_GT(counts._vector_nodes, 0U);
  }
}
