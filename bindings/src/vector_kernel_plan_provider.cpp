//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "vector_kernel_plan_provider.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace pyace {
namespace {

namespace vk = nn::vector;
using air::base::PRIMITIVE_TYPE;

[[noreturn]] void Invalid(const std::string& path,
                          const std::string& diagnostic) {
  throw std::invalid_argument("Python vector-kernel provider result " + path +
                              " " + diagnostic);
}

std::string Item_path(const std::string& path, size_t index) {
  return path + "[" + std::to_string(index) + "]";
}

py::dict Require_dict(py::handle value, const std::string& path) {
  if (!PyDict_CheckExact(value.ptr())) {
    Invalid(path, "must be a dict");
  }
  return py::reinterpret_borrow<py::dict>(value);
}

py::tuple Require_tuple(py::handle value, const std::string& path) {
  if (!PyTuple_CheckExact(value.ptr())) {
    Invalid(path, "must be a tuple");
  }
  return py::reinterpret_borrow<py::tuple>(value);
}

py::handle Field(const py::dict& data, const char* name,
                 const std::string& path) {
  PyObject* value = PyDict_GetItemString(data.ptr(), name);
  if (value == nullptr) {
    Invalid(path, "is missing required field '" + std::string(name) + "'");
  }
  return py::handle(value);
}

void Require_exact_keys(const py::dict& data,
                        std::initializer_list<const char*> keys,
                        const std::string& path) {
  if (static_cast<size_t>(py::len(data)) != keys.size()) {
    Invalid(path, "contains missing or extra fields");
  }
  for (const char* key : keys) {
    if (PyDict_GetItemString(data.ptr(), key) == nullptr) {
      Invalid(path, "is missing required field '" + std::string(key) + "'");
    }
  }
}

std::string Require_string(py::handle value, const std::string& path) {
  if (!PyUnicode_CheckExact(value.ptr())) {
    Invalid(path, "must be a string");
  }
  Py_ssize_t size = 0;
  const char* text = PyUnicode_AsUTF8AndSize(value.ptr(), &size);
  if (text == nullptr) {
    throw py::error_already_set();
  }
  return std::string(text, static_cast<size_t>(size));
}

bool Require_bool(py::handle value, const std::string& path) {
  if (!PyBool_Check(value.ptr())) {
    Invalid(path, "must be a bool");
  }
  return value.ptr() == Py_True;
}

int64_t Require_i64(py::handle value, const std::string& path) {
  if (!PyLong_CheckExact(value.ptr())) {
    Invalid(path, "must be an integer");
  }
  int overflow = 0;
  const long long parsed =
      PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
  if (overflow != 0 || (parsed == -1 && PyErr_Occurred())) {
    PyErr_Clear();
    Invalid(path, "must fit a signed 64-bit integer");
  }
  if (parsed < std::numeric_limits<int64_t>::min() ||
      parsed > std::numeric_limits<int64_t>::max()) {
    Invalid(path, "must fit a signed 64-bit integer");
  }
  return static_cast<int64_t>(parsed);
}

int32_t Require_i32(py::handle value, const std::string& path) {
  const int64_t parsed = Require_i64(value, path);
  if (parsed < std::numeric_limits<int32_t>::min() ||
      parsed > std::numeric_limits<int32_t>::max()) {
    Invalid(path, "must fit a signed 32-bit integer");
  }
  return static_cast<int32_t>(parsed);
}

uint32_t Require_u32(py::handle value, const std::string& path) {
  if (!PyLong_CheckExact(value.ptr())) {
    Invalid(path, "must be an integer");
  }
  const unsigned long long parsed = PyLong_AsUnsignedLongLong(value.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    Invalid(path, "must fit an unsigned 32-bit integer");
  }
  if (parsed > std::numeric_limits<uint32_t>::max()) {
    Invalid(path, "must fit an unsigned 32-bit integer");
  }
  return static_cast<uint32_t>(parsed);
}

PRIMITIVE_TYPE Parse_primitive(py::handle value, const std::string& path) {
  const std::string name = Require_string(value, path);
  if (name == "bool") return PRIMITIVE_TYPE::BOOL;
  if (name == "s8") return PRIMITIVE_TYPE::INT_S8;
  if (name == "s16") return PRIMITIVE_TYPE::INT_S16;
  if (name == "s32") return PRIMITIVE_TYPE::INT_S32;
  if (name == "s64") return PRIMITIVE_TYPE::INT_S64;
  if (name == "u8") return PRIMITIVE_TYPE::INT_U8;
  if (name == "u16") return PRIMITIVE_TYPE::INT_U16;
  if (name == "u32") return PRIMITIVE_TYPE::INT_U32;
  if (name == "u64") return PRIMITIVE_TYPE::INT_U64;
  if (name == "f32") return PRIMITIVE_TYPE::FLOAT_32;
  if (name == "f64") return PRIMITIVE_TYPE::FLOAT_64;
  if (name == "c32") return PRIMITIVE_TYPE::COMPLEX_32;
  if (name == "c64") return PRIMITIVE_TYPE::COMPLEX_64;
  Invalid(path, "has unsupported primitive type '" + name + "'");
}

size_t Primitive_width(PRIMITIVE_TYPE type) {
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
      throw std::invalid_argument(
          "unsupported vector-kernel NumPy primitive type");
  }
}

const char* Numpy_dtype(PRIMITIVE_TYPE type) {
  switch (type) {
    case PRIMITIVE_TYPE::BOOL: return "|b1";
    case PRIMITIVE_TYPE::INT_S8: return "|i1";
    case PRIMITIVE_TYPE::INT_S16: return "<i2";
    case PRIMITIVE_TYPE::INT_S32: return "<i4";
    case PRIMITIVE_TYPE::INT_S64: return "<i8";
    case PRIMITIVE_TYPE::INT_U8: return "|u1";
    case PRIMITIVE_TYPE::INT_U16: return "<u2";
    case PRIMITIVE_TYPE::INT_U32: return "<u4";
    case PRIMITIVE_TYPE::INT_U64: return "<u8";
    case PRIMITIVE_TYPE::FLOAT_32: return "<f4";
    case PRIMITIVE_TYPE::FLOAT_64: return "<f8";
    case PRIMITIVE_TYPE::COMPLEX_32: return "<c8";
    case PRIMITIVE_TYPE::COMPLEX_64: return "<c16";
    default:
      throw std::invalid_argument(
          "unsupported vector-kernel NumPy primitive type");
  }
}

bool Host_is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

size_t Expected_byte_count(const vk::VECTOR_KERNEL_RANKED_TYPE_PLAN& type,
                           const std::string& path,
                           bool require_positive_rank) {
  if (require_positive_rank && type._shape.empty()) {
    Invalid(path, "must have positive rank");
  }
  size_t count = 1;
  for (size_t index = 0; index < type._shape.size(); ++index) {
    const int64_t dimension = type._shape[index];
    if (dimension <= 0 ||
        static_cast<uint64_t>(dimension) >
            std::numeric_limits<size_t>::max()) {
      Invalid(Item_path(path + ".shape", index),
              "must be a positive size_t dimension");
    }
    const size_t converted = static_cast<size_t>(dimension);
    if (count > std::numeric_limits<size_t>::max() / converted) {
      Invalid(path, "shape element count overflows size_t");
    }
    count *= converted;
  }
  const size_t width = Primitive_width(type._element_type);
  if (count > std::numeric_limits<size_t>::max() / width) {
    Invalid(path, "payload byte count overflows size_t");
  }
  return count * width;
}

py::tuple Shape_tuple(const std::vector<int64_t>& shape) {
  py::tuple result(shape.size());
  for (size_t index = 0; index < shape.size(); ++index) {
    result[index] = py::int_(shape[index]);
  }
  return result;
}

py::dict Type_binding(const vk::VECTOR_KERNEL_RANKED_TYPE_PLAN& type) {
  py::dict result;
  result["element_type"] =
      vk::Vector_kernel_primitive_type_name(type._element_type);
  result["shape"] = Shape_tuple(type._shape);
  return result;
}

py::array Readonly_request_array(
    const py::object& numpy,
    const vk::VECTOR_KERNEL_RANKED_TYPE_PLAN& type,
    const std::vector<uint8_t>& bytes, const std::string& path) {
  const size_t expected = Expected_byte_count(type, path, true);
  if (bytes.size() != expected) {
    throw std::invalid_argument(path +
                                " byte count does not match its type");
  }

  const char* raw = bytes.empty()
                        ? ""
                        : reinterpret_cast<const char*>(bytes.data());
  py::bytes storage(raw, bytes.size());
  py::object flat = numpy.attr("frombuffer")(
      storage, py::arg("dtype") = Numpy_dtype(type._element_type));
  py::object reshaped = flat.attr("reshape")(
      Shape_tuple(type._shape), py::arg("order") = "C");
  py::array array = py::cast<py::array>(reshaped);
  if ((array.flags() & py::array::c_style) == 0 ||
      py::cast<bool>(array.attr("flags").attr("writeable"))) {
    throw std::runtime_error(path +
                             " did not create a read-only C-contiguous array");
  }
  return array;
}

py::dict Build_request_binding(
    const vk::VECTOR_KERNEL_PLANNING_REQUEST& request,
    const py::object& numpy) {
  py::dict data;
  data["operation"] = vk::Vector_kernel_operation_name(request._operation);

  py::tuple attributes(request._attributes.size());
  for (size_t index = 0; index < request._attributes.size(); ++index) {
    const vk::VECTOR_KERNEL_ATTRIBUTE_RECORD& attribute =
        request._attributes[index];
    py::dict item;
    item["name"] = attribute._name;
    item["values"] = Readonly_request_array(
        numpy, attribute._type, attribute._bytes,
        "request.attributes[" + std::to_string(index) + "].values");
    attributes[index] = std::move(item);
  }
  data["attributes"] = std::move(attributes);

  py::tuple operand_types(request._operand_types.size());
  for (size_t index = 0; index < request._operand_types.size(); ++index) {
    operand_types[index] = Type_binding(request._operand_types[index]);
  }
  data["operand_types"] = std::move(operand_types);
  data["declared_result_type"] =
      Type_binding(request._declared_result_type);

  py::dict options;
  options["conv_parallel"] = request._options._conv_parallel;
  options["mask_fuse"] = request._options._mask_fuse;
  options["selective_strided_slice"] =
      request._options._selective_strided_slice;
  options["sharding"] = request._options._sharding;
  options["is_last_operation"] = request._options._is_last_operation;
  data["options"] = std::move(options);

  py::dict target;
  target["num_slots"] = request._target._num_slots;
  target["min_slots"] = request._target._min_slots;
  target["max_slots"] = request._target._max_slots;
  data["target"] = std::move(target);

  py::tuple source_constants(request._source_constants.size());
  for (size_t index = 0; index < request._source_constants.size(); ++index) {
    const vk::VECTOR_KERNEL_TYPED_PAYLOAD& payload =
        request._source_constants[index];
    py::dict item;
    item["role"] = payload._role;
    item["values"] = Readonly_request_array(
        numpy, payload._type, payload._bytes,
        "request.source_constants[" + std::to_string(index) + "].values");
    item["content_hash"] = payload._content_hash;
    source_constants[index] = std::move(item);
  }
  data["source_constants"] = std::move(source_constants);
  data["requested_plan_kind"] =
      vk::Vector_kernel_requested_plan_kind_name(
          request._requested_plan_kind);

  py::tuple runtime_scalar_types(request._runtime_scalar_types.size());
  for (size_t index = 0; index < request._runtime_scalar_types.size();
       ++index) {
    runtime_scalar_types[index] = vk::Vector_kernel_primitive_type_name(
        request._runtime_scalar_types[index]);
  }
  data["runtime_scalar_types"] = std::move(runtime_scalar_types);
  return data;
}

template <typename RECORD, typename PARSER>
std::vector<RECORD> Parse_record_tuple(py::handle value,
                                       const std::string& path,
                                       PARSER&& parser) {
  py::tuple tuple = Require_tuple(value, path);
  std::vector<RECORD> records;
  records.reserve(tuple.size());
  for (size_t index = 0; index < tuple.size(); ++index) {
    records.push_back(parser(tuple[index], Item_path(path, index)));
  }
  return records;
}

std::vector<int64_t> Parse_i64_tuple(py::handle value,
                                     const std::string& path) {
  py::tuple tuple = Require_tuple(value, path);
  std::vector<int64_t> result;
  result.reserve(tuple.size());
  for (size_t index = 0; index < tuple.size(); ++index) {
    result.push_back(Require_i64(tuple[index], Item_path(path, index)));
  }
  return result;
}

std::vector<int32_t> Parse_i32_tuple(py::handle value,
                                     const std::string& path) {
  py::tuple tuple = Require_tuple(value, path);
  std::vector<int32_t> result;
  result.reserve(tuple.size());
  for (size_t index = 0; index < tuple.size(); ++index) {
    result.push_back(Require_i32(tuple[index], Item_path(path, index)));
  }
  return result;
}

vk::VECTOR_KERNEL_RANKED_TYPE_PLAN Parse_ranked_type(
    py::handle value, const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(data, {"element_type", "shape"}, path);
  return vk::VECTOR_KERNEL_RANKED_TYPE_PLAN{
      Parse_primitive(Field(data, "element_type", path),
                      path + ".element_type"),
      Parse_i64_tuple(Field(data, "shape", path), path + ".shape")};
}

vk::VECTOR_KERNEL_REDUCTION_KIND Parse_reduction_kind(
    py::handle value, const std::string& path) {
  const std::string name = Require_string(value, path);
  if (name == "power-of-two") {
    return vk::VECTOR_KERNEL_REDUCTION_KIND::POWER_OF_TWO;
  }
  if (name == "linear") return vk::VECTOR_KERNEL_REDUCTION_KIND::LINEAR;
  if (name == "collective-single-block") {
    return vk::VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_SINGLE_BLOCK;
  }
  if (name == "collective-blocks") {
    return vk::VECTOR_KERNEL_REDUCTION_KIND::COLLECTIVE_BLOCKS;
  }
  Invalid(path, "has unknown reduction kind '" + name + "'");
}

vk::VECTOR_KERNEL_MASK_POLICY Parse_mask_policy(
    py::handle value, const std::string& path) {
  const std::string name = Require_string(value, path);
  if (name == "none") return vk::VECTOR_KERNEL_MASK_POLICY::NONE;
  if (name == "clear-valid-prefix") {
    return vk::VECTOR_KERNEL_MASK_POLICY::CLEAR_VALID_PREFIX;
  }
  if (name == "collective-reduction") {
    return vk::VECTOR_KERNEL_MASK_POLICY::COLLECTIVE_REDUCTION;
  }
  Invalid(path, "has unknown mask policy '" + name + "'");
}

vk::VECTOR_KERNEL_SLOT_POLICY Parse_slot_policy(
    py::handle value, const std::string& path) {
  const std::string name = Require_string(value, path);
  if (name == "absent-native-baseline") {
    return vk::VECTOR_KERNEL_SLOT_POLICY::ABSENT_NATIVE_BASELINE;
  }
  if (name == "logical-output-elements") {
    return vk::VECTOR_KERNEL_SLOT_POLICY::LOGICAL_OUTPUT_ELEMENTS;
  }
  if (name == "explicit") return vk::VECTOR_KERNEL_SLOT_POLICY::EXPLICIT;
  Invalid(path, "has unknown slot policy '" + name + "'");
}

vk::VECTOR_KERNEL_RUNTIME_PREPARATION_KIND Parse_runtime_preparation_kind(
    py::handle value, const std::string& path) {
  const std::string name = Require_string(value, path);
  if (name == "packed-vector") {
    return vk::VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::PACKED_VECTOR;
  }
  if (name == "flatten-packed-vector") {
    return vk::VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::FLATTEN_PACKED_VECTOR;
  }
  if (name == "blocking-rotations") {
    return vk::VECTOR_KERNEL_RUNTIME_PREPARATION_KIND::BLOCKING_ROTATIONS;
  }
  Invalid(path, "has unknown runtime preparation kind '" + name + "'");
}

vk::VECTOR_KERNEL_CONSTANT_PLAN Parse_constant_plan(
    py::handle value, const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(data, {"role", "type", "content_hash"}, path);
  return vk::VECTOR_KERNEL_CONSTANT_PLAN{
      Require_string(Field(data, "role", path), path + ".role"),
      Parse_ranked_type(Field(data, "type", path), path + ".type"),
      Require_string(Field(data, "content_hash", path),
                     path + ".content_hash")};
}

vk::VECTOR_KERNEL_LOOP_PLAN Parse_loop(py::handle value,
                                       const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(data,
                     {"role", "lower", "upper", "step",
                      "nesting_depth"},
                     path);
  return vk::VECTOR_KERNEL_LOOP_PLAN{
      Require_string(Field(data, "role", path), path + ".role"),
      Require_i32(Field(data, "lower", path), path + ".lower"),
      Require_i32(Field(data, "upper", path), path + ".upper"),
      Require_i32(Field(data, "step", path), path + ".step"),
      Require_u32(Field(data, "nesting_depth", path),
                  path + ".nesting_depth")};
}

vk::VECTOR_KERNEL_AFFINE_INDEX_PLAN Parse_affine_index(
    py::handle value, const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(
      data, {"iv_coefficients", "constant", "uses_sharding_offset"}, path);
  return vk::VECTOR_KERNEL_AFFINE_INDEX_PLAN{
      Parse_i64_tuple(Field(data, "iv_coefficients", path),
                      path + ".iv_coefficients"),
      Require_i64(Field(data, "constant", path), path + ".constant"),
      Require_bool(Field(data, "uses_sharding_offset", path),
                   path + ".uses_sharding_offset")};
}

vk::VECTOR_KERNEL_SLICE_PLAN Parse_slice(py::handle value,
                                         const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(data, {"role", "index", "width"}, path);
  return vk::VECTOR_KERNEL_SLICE_PLAN{
      Require_string(Field(data, "role", path), path + ".role"),
      Parse_affine_index(Field(data, "index", path), path + ".index"),
      Require_i64(Field(data, "width", path), path + ".width")};
}

vk::VECTOR_KERNEL_ROTATION_PLAN Parse_rotation(
    py::handle value, const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(data, {"role", "candidates"}, path);
  return vk::VECTOR_KERNEL_ROTATION_PLAN{
      Require_string(Field(data, "role", path), path + ".role"),
      Parse_i32_tuple(Field(data, "candidates", path),
                      path + ".candidates")};
}

vk::VECTOR_KERNEL_REDUCTION_PLAN Parse_reduction(
    py::handle value, const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(
      data, {"role", "kind", "factor", "block_width", "padding"}, path);
  return vk::VECTOR_KERNEL_REDUCTION_PLAN{
      Require_string(Field(data, "role", path), path + ".role"),
      Parse_reduction_kind(Field(data, "kind", path), path + ".kind"),
      Require_i64(Field(data, "factor", path), path + ".factor"),
      Require_i64(Field(data, "block_width", path), path + ".block_width"),
      Require_i64(Field(data, "padding", path), path + ".padding")};
}

vk::VECTOR_KERNEL_MASK_PLAN Parse_mask(py::handle value,
                                       const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(data, {"policy", "valid_length"}, path);
  return vk::VECTOR_KERNEL_MASK_PLAN{
      Parse_mask_policy(Field(data, "policy", path), path + ".policy"),
      Require_i64(Field(data, "valid_length", path),
                  path + ".valid_length")};
}

vk::VECTOR_KERNEL_SLOT_PLAN Parse_slot(py::handle value,
                                       const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(data, {"policy", "value"}, path);
  return vk::VECTOR_KERNEL_SLOT_PLAN{
      Parse_slot_policy(Field(data, "policy", path), path + ".policy"),
      Require_u32(Field(data, "value", path), path + ".value")};
}

vk::VECTOR_KERNEL_COMMON_PLAN Parse_common(py::handle value,
                                           const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(
      data,
      {"runtime_vector_inputs", "runtime_scalar_inputs", "constants",
       "result_type", "loops", "slices", "rotations", "reductions",
       "mask", "slot"},
      path);

  std::vector<vk::VECTOR_KERNEL_RANKED_TYPE_PLAN> vector_inputs =
      Parse_record_tuple<vk::VECTOR_KERNEL_RANKED_TYPE_PLAN>(
          Field(data, "runtime_vector_inputs", path),
          path + ".runtime_vector_inputs", Parse_ranked_type);
  std::vector<PRIMITIVE_TYPE> scalar_inputs =
      Parse_record_tuple<PRIMITIVE_TYPE>(
          Field(data, "runtime_scalar_inputs", path),
          path + ".runtime_scalar_inputs", Parse_primitive);
  std::vector<vk::VECTOR_KERNEL_CONSTANT_PLAN> constants =
      Parse_record_tuple<vk::VECTOR_KERNEL_CONSTANT_PLAN>(
          Field(data, "constants", path), path + ".constants",
          Parse_constant_plan);
  vk::VECTOR_KERNEL_RANKED_TYPE_PLAN result_type = Parse_ranked_type(
      Field(data, "result_type", path), path + ".result_type");
  std::vector<vk::VECTOR_KERNEL_LOOP_PLAN> loops =
      Parse_record_tuple<vk::VECTOR_KERNEL_LOOP_PLAN>(
          Field(data, "loops", path), path + ".loops", Parse_loop);
  std::vector<vk::VECTOR_KERNEL_SLICE_PLAN> slices =
      Parse_record_tuple<vk::VECTOR_KERNEL_SLICE_PLAN>(
          Field(data, "slices", path), path + ".slices", Parse_slice);
  std::vector<vk::VECTOR_KERNEL_ROTATION_PLAN> rotations =
      Parse_record_tuple<vk::VECTOR_KERNEL_ROTATION_PLAN>(
          Field(data, "rotations", path), path + ".rotations",
          Parse_rotation);
  std::vector<vk::VECTOR_KERNEL_REDUCTION_PLAN> reductions =
      Parse_record_tuple<vk::VECTOR_KERNEL_REDUCTION_PLAN>(
          Field(data, "reductions", path), path + ".reductions",
          Parse_reduction);
  vk::VECTOR_KERNEL_MASK_PLAN mask =
      Parse_mask(Field(data, "mask", path), path + ".mask");
  vk::VECTOR_KERNEL_SLOT_PLAN slot =
      Parse_slot(Field(data, "slot", path), path + ".slot");

  return vk::VECTOR_KERNEL_COMMON_PLAN{
      std::move(vector_inputs), std::move(scalar_inputs),
      std::move(constants), std::move(result_type), std::move(loops),
      std::move(slices), std::move(rotations), std::move(reductions),
      std::move(mask), std::move(slot)};
}

vk::VECTOR_KERNEL_PLAN Parse_plan(py::handle value,
                                  const std::string& path) {
  py::dict data = Require_dict(value, path);
  const std::string kind =
      Require_string(Field(data, "kind", path), path + ".kind");

  if (kind == "baseline-gemm") {
    Require_exact_keys(data,
                       {"kind", "common", "height", "width",
                        "input_duplications"},
                       path);
    vk::VECTOR_KERNEL_COMMON_PLAN common =
        Parse_common(Field(data, "common", path), path + ".common");
    return vk::BASELINE_GEMM_PLAN{
        std::move(common),
        Require_i64(Field(data, "height", path), path + ".height"),
        Require_i64(Field(data, "width", path), path + ".width"),
        Require_i64(Field(data, "input_duplications", path),
                    path + ".input_duplications")};
  }

  if (kind == "baseline-conv") {
    Require_exact_keys(
        data,
        {"kind", "common", "channel_in", "channel_out", "output_height",
         "output_width", "kernel_hw", "stride", "input_duplications"},
        path);
    vk::VECTOR_KERNEL_COMMON_PLAN common =
        Parse_common(Field(data, "common", path), path + ".common");
    return vk::BASELINE_CONV_PLAN{
        std::move(common),
        Require_i64(Field(data, "channel_in", path), path + ".channel_in"),
        Require_i64(Field(data, "channel_out", path), path + ".channel_out"),
        Require_i64(Field(data, "output_height", path),
                    path + ".output_height"),
        Require_i64(Field(data, "output_width", path),
                    path + ".output_width"),
        Require_i64(Field(data, "kernel_hw", path), path + ".kernel_hw"),
        Require_i64(Field(data, "stride", path), path + ".stride"),
        Require_i64(Field(data, "input_duplications", path),
                    path + ".input_duplications")};
  }

  if (kind == "fast-gemm") {
    Require_exact_keys(
        data,
        {"kind", "common", "n", "k", "np", "kp", "nd", "kd",
         "block_size", "blocks_per_partition", "packed_partitions",
         "shift", "shift_buffer", "grid_size", "input_replications"},
        path);
    vk::VECTOR_KERNEL_COMMON_PLAN common =
        Parse_common(Field(data, "common", path), path + ".common");
    return vk::FAST_GEMM_PLAN{
        std::move(common),
        Require_i64(Field(data, "n", path), path + ".n"),
        Require_i64(Field(data, "k", path), path + ".k"),
        Require_i64(Field(data, "np", path), path + ".np"),
        Require_i64(Field(data, "kp", path), path + ".kp"),
        Require_i64(Field(data, "nd", path), path + ".nd"),
        Require_i64(Field(data, "kd", path), path + ".kd"),
        Require_i64(Field(data, "block_size", path), path + ".block_size"),
        Require_i64(Field(data, "blocks_per_partition", path),
                    path + ".blocks_per_partition"),
        Require_i64(Field(data, "packed_partitions", path),
                    path + ".packed_partitions"),
        Require_i64(Field(data, "shift", path), path + ".shift"),
        Require_i64(Field(data, "shift_buffer", path),
                    path + ".shift_buffer"),
        Require_i64(Field(data, "grid_size", path), path + ".grid_size"),
        Require_i64(Field(data, "input_replications", path),
                    path + ".input_replications")};
  }

  if (kind == "fast-conv") {
    Require_exact_keys(
        data,
        {"kind", "common", "channel_in", "channel_out", "output_height",
         "output_width", "kernel_hw", "group", "stride", "input_size",
         "output_size", "num_slots", "num_grid", "num_block",
         "width_block", "width_block_data", "width_block_pad",
         "position_block", "capacity_block", "input_duplications",
         "blocking_outer_depth", "cyclic_roll", "sharding_offset"},
        path);
    vk::VECTOR_KERNEL_COMMON_PLAN common =
        Parse_common(Field(data, "common", path), path + ".common");
    std::optional<vk::VECTOR_KERNEL_SHARDING_OFFSET_PLAN> sharding_offset;
    py::handle offset_value = Field(data, "sharding_offset", path);
    if (!offset_value.is_none()) {
      py::dict offset =
          Require_dict(offset_value, path + ".sharding_offset");
      Require_exact_keys(offset, {"type", "scale"},
                         path + ".sharding_offset");
      sharding_offset.emplace(vk::VECTOR_KERNEL_SHARDING_OFFSET_PLAN{
          Parse_primitive(Field(offset, "type", path + ".sharding_offset"),
                          path + ".sharding_offset.type"),
          Require_i64(Field(offset, "scale", path + ".sharding_offset"),
                      path + ".sharding_offset.scale")});
    }
    return vk::FAST_CONV_PLAN{
        std::move(common),
        Require_i64(Field(data, "channel_in", path), path + ".channel_in"),
        Require_i64(Field(data, "channel_out", path), path + ".channel_out"),
        Require_i64(Field(data, "output_height", path),
                    path + ".output_height"),
        Require_i64(Field(data, "output_width", path),
                    path + ".output_width"),
        Require_i64(Field(data, "kernel_hw", path), path + ".kernel_hw"),
        Require_i64(Field(data, "group", path), path + ".group"),
        Require_i64(Field(data, "stride", path), path + ".stride"),
        Require_i64(Field(data, "input_size", path), path + ".input_size"),
        Require_i64(Field(data, "output_size", path), path + ".output_size"),
        Require_i64(Field(data, "num_slots", path), path + ".num_slots"),
        Require_i64(Field(data, "num_grid", path), path + ".num_grid"),
        Require_i64(Field(data, "num_block", path), path + ".num_block"),
        Require_i64(Field(data, "width_block", path), path + ".width_block"),
        Require_i64(Field(data, "width_block_data", path),
                    path + ".width_block_data"),
        Require_i64(Field(data, "width_block_pad", path),
                    path + ".width_block_pad"),
        Require_i64(Field(data, "position_block", path),
                    path + ".position_block"),
        Require_i64(Field(data, "capacity_block", path),
                    path + ".capacity_block"),
        Require_i64(Field(data, "input_duplications", path),
                    path + ".input_duplications"),
        Require_i64(Field(data, "blocking_outer_depth", path),
                    path + ".blocking_outer_depth"),
        Require_bool(Field(data, "cyclic_roll", path),
                     path + ".cyclic_roll"),
        std::move(sharding_offset)};
  }

  Invalid(path + ".kind", "has unknown plan kind '" + kind + "'");
}

PRIMITIVE_TYPE Primitive_for_numpy_dtype(const py::dtype& dtype,
                                         const std::string& path) {
  const std::string kind =
      Require_string(dtype.attr("kind"), path + ".kind");
  const int64_t itemsize =
      Require_i64(dtype.attr("itemsize"), path + ".itemsize");
  const std::string byteorder =
      Require_string(dtype.attr("byteorder"), path + ".byteorder");

  if (itemsize > 1) {
    if (byteorder == ">" ||
        (byteorder == "=" && !Host_is_little_endian())) {
      Invalid(path, "must use canonical little-endian byte order");
    }
    if (byteorder != "<" && byteorder != "=") {
      Invalid(path, "has unsupported byte order '" + byteorder + "'");
    }
  } else if (byteorder != "|") {
    Invalid(path, "one-byte dtype must have byte-order marker '|'");
  }

  if (kind == "b" && itemsize == 1) return PRIMITIVE_TYPE::BOOL;
  if (kind == "i" && itemsize == 1) return PRIMITIVE_TYPE::INT_S8;
  if (kind == "i" && itemsize == 2) return PRIMITIVE_TYPE::INT_S16;
  if (kind == "i" && itemsize == 4) return PRIMITIVE_TYPE::INT_S32;
  if (kind == "i" && itemsize == 8) return PRIMITIVE_TYPE::INT_S64;
  if (kind == "u" && itemsize == 1) return PRIMITIVE_TYPE::INT_U8;
  if (kind == "u" && itemsize == 2) return PRIMITIVE_TYPE::INT_U16;
  if (kind == "u" && itemsize == 4) return PRIMITIVE_TYPE::INT_U32;
  if (kind == "u" && itemsize == 8) return PRIMITIVE_TYPE::INT_U64;
  if (kind == "f" && itemsize == 4) return PRIMITIVE_TYPE::FLOAT_32;
  if (kind == "f" && itemsize == 8) return PRIMITIVE_TYPE::FLOAT_64;
  if (kind == "c" && itemsize == 8) return PRIMITIVE_TYPE::COMPLEX_32;
  if (kind == "c" && itemsize == 16) return PRIMITIVE_TYPE::COMPLEX_64;
  Invalid(path, "is not an exact supported vector-kernel dtype");
}

vk::VECTOR_KERNEL_TYPED_PAYLOAD Parse_payload(py::handle value,
                                              const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(data, {"role", "values", "content_hash"}, path);
  py::handle values = Field(data, "values", path);
  if (!py::isinstance<py::array>(values)) {
    Invalid(path + ".values", "must be a NumPy ndarray");
  }
  py::array array = py::reinterpret_borrow<py::array>(values);
  if ((array.flags() & py::array::c_style) == 0) {
    Invalid(path + ".values", "must be C-contiguous");
  }
  const PRIMITIVE_TYPE primitive =
      Primitive_for_numpy_dtype(array.dtype(), path + ".values.dtype");
  py::buffer_info info = array.request();
  if (info.ndim <= 0) {
    Invalid(path + ".values", "must have positive rank");
  }
  if (info.itemsize != static_cast<py::ssize_t>(Primitive_width(primitive))) {
    Invalid(path + ".values", "item size does not match its dtype");
  }

  std::vector<int64_t> shape;
  shape.reserve(static_cast<size_t>(info.ndim));
  for (size_t index = 0; index < static_cast<size_t>(info.ndim); ++index) {
    const py::ssize_t dimension = info.shape[index];
    if (dimension <= 0 ||
        static_cast<uint64_t>(dimension) >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      Invalid(Item_path(path + ".values.shape", index),
              "must be a positive signed 64-bit dimension");
    }
    shape.push_back(static_cast<int64_t>(dimension));
  }
  vk::VECTOR_KERNEL_RANKED_TYPE_PLAN type{primitive, std::move(shape)};
  const size_t byte_count =
      Expected_byte_count(type, path + ".values", true);
  if (info.size < 0 ||
      static_cast<uint64_t>(info.size) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    Invalid(path + ".values", "shape and byte count are inconsistent");
  }
  const size_t element_count = static_cast<size_t>(info.size);
  const size_t item_size = static_cast<size_t>(info.itemsize);
  if (element_count > std::numeric_limits<size_t>::max() / item_size ||
      byte_count != element_count * item_size) {
    Invalid(path + ".values", "shape and byte count are inconsistent");
  }

  std::vector<uint8_t> bytes(byte_count);
  if (byte_count != 0) {
    if (info.ptr == nullptr) {
      Invalid(path + ".values", "has a null data pointer");
    }
    std::memcpy(bytes.data(), info.ptr, byte_count);
  }
  return vk::VECTOR_KERNEL_TYPED_PAYLOAD{
      Require_string(Field(data, "role", path), path + ".role"),
      std::move(type), std::move(bytes),
      Require_string(Field(data, "content_hash", path),
                     path + ".content_hash")};
}

vk::VECTOR_KERNEL_RUNTIME_PREPARATION Parse_runtime_preparation(
    py::handle value, const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(
      data,
      {"role", "source_operand", "kind", "result_type",
       "logical_input_size", "replications", "blocking_width",
       "rotation_candidates", "outer_block_depth"},
      path);
  return vk::VECTOR_KERNEL_RUNTIME_PREPARATION{
      Require_string(Field(data, "role", path), path + ".role"),
      Require_u32(Field(data, "source_operand", path),
                  path + ".source_operand"),
      Parse_runtime_preparation_kind(Field(data, "kind", path),
                                     path + ".kind"),
      Parse_ranked_type(Field(data, "result_type", path),
                        path + ".result_type"),
      Require_i64(Field(data, "logical_input_size", path),
                  path + ".logical_input_size"),
      Require_i64(Field(data, "replications", path),
                  path + ".replications"),
      Require_i64(Field(data, "blocking_width", path),
                  path + ".blocking_width"),
      Parse_i32_tuple(Field(data, "rotation_candidates", path),
                      path + ".rotation_candidates"),
      Require_u32(Field(data, "outer_block_depth", path),
                  path + ".outer_block_depth")};
}

vk::VECTOR_KERNEL_SCALAR_PREPARATION Parse_scalar_preparation(
    py::handle value, const std::string& path) {
  py::dict data = Require_dict(value, path);
  Require_exact_keys(data, {"role", "source_operand", "type", "scale"},
                     path);
  return vk::VECTOR_KERNEL_SCALAR_PREPARATION{
      Require_string(Field(data, "role", path), path + ".role"),
      Require_u32(Field(data, "source_operand", path),
                  path + ".source_operand"),
      Parse_primitive(Field(data, "type", path), path + ".type"),
      Require_i64(Field(data, "scale", path), path + ".scale")};
}

const vk::VECTOR_KERNEL_COMMON_PLAN& Common_plan(
    const vk::VECTOR_KERNEL_PLAN& plan) {
  return std::visit(
      [](const auto& typed_plan) -> const vk::VECTOR_KERNEL_COMMON_PLAN& {
        return typed_plan._common;
      },
      plan);
}

bool Equal_type(const vk::VECTOR_KERNEL_RANKED_TYPE_PLAN& lhs,
                const vk::VECTOR_KERNEL_RANKED_TYPE_PLAN& rhs) {
  return lhs._element_type == rhs._element_type && lhs._shape == rhs._shape;
}

void Validate_payload_shapes(
    const vk::VECTOR_KERNEL_PLAN& plan,
    const std::vector<vk::VECTOR_KERNEL_TYPED_PAYLOAD>& payloads) {
  const vk::VECTOR_KERNEL_COMMON_PLAN& common = Common_plan(plan);
  for (size_t index = 0; index < payloads.size(); ++index) {
    const vk::VECTOR_KERNEL_TYPED_PAYLOAD& payload = payloads[index];
    const auto descriptor = std::find_if(
        common._constants.begin(), common._constants.end(),
        [&](const vk::VECTOR_KERNEL_CONSTANT_PLAN& candidate) {
          return candidate._role == payload._role;
        });
    if (descriptor != common._constants.end() &&
        !Equal_type(descriptor->_type, payload._type)) {
      Invalid(Item_path("constants", index) + ".values",
              "dtype or shape does not match its plan descriptor");
    }
  }
}

vk::VECTOR_KERNEL_PROVIDER_RESULT Parse_provider_result(py::handle value) {
  py::dict data = Require_dict(value, "result");
  Require_exact_keys(data,
                     {"plan", "constants", "runtime_preparations",
                      "scalar_preparations", "provenance"},
                     "result");
  vk::VECTOR_KERNEL_PLAN plan =
      Parse_plan(Field(data, "plan", "result"), "plan");
  std::vector<vk::VECTOR_KERNEL_TYPED_PAYLOAD> constants =
      Parse_record_tuple<vk::VECTOR_KERNEL_TYPED_PAYLOAD>(
          Field(data, "constants", "result"), "constants", Parse_payload);
  Validate_payload_shapes(plan, constants);
  std::vector<vk::VECTOR_KERNEL_RUNTIME_PREPARATION> runtime_preparations =
      Parse_record_tuple<vk::VECTOR_KERNEL_RUNTIME_PREPARATION>(
          Field(data, "runtime_preparations", "result"),
          "runtime_preparations", Parse_runtime_preparation);
  std::vector<vk::VECTOR_KERNEL_SCALAR_PREPARATION> scalar_preparations =
      Parse_record_tuple<vk::VECTOR_KERNEL_SCALAR_PREPARATION>(
          Field(data, "scalar_preparations", "result"),
          "scalar_preparations", Parse_scalar_preparation);
  std::string provenance = Require_string(
      Field(data, "provenance", "result"), "provenance");
  return vk::VECTOR_KERNEL_PROVIDER_RESULT{
      std::move(plan), std::move(constants),
      std::move(runtime_preparations), std::move(scalar_preparations),
      std::move(provenance)};
}

class REQUEST_VIEW_EXPIRER {
public:
  explicit REQUEST_VIEW_EXPIRER(const py::object& view)
      : _expire(view.attr("_expire")) {}

  ~REQUEST_VIEW_EXPIRER() {
    if (_expired) return;
    try {
      _expire();
    } catch (...) {
      // Preserve the callback/conversion exception already in flight.  The
      // helper-owned expiration method is retried here only as an unwind guard.
      if (PyErr_Occurred()) PyErr_Clear();
    }
  }

  void Expire() {
    _expire();
    _expired = true;
  }

private:
  py::object _expire;
  bool _expired = false;
};

}  // namespace

PYTHON_VECTOR_KERNEL_PLAN_PROVIDER::PYTHON_VECTOR_KERNEL_PLAN_PROVIDER(
    py::object callback)
    : _callback(std::move(callback)) {
  if (_callback.is_none() || !PyCallable_Check(_callback.ptr())) {
    throw py::type_error("Python vector-kernel plan provider must be callable");
  }
}

vk::VECTOR_KERNEL_PROVIDER_CALL_RESULT
PYTHON_VECTOR_KERNEL_PLAN_PROVIDER::Plan(
    const vk::VECTOR_KERNEL_PLANNING_REQUEST& request) const {
  py::gil_scoped_acquire acquire;
  try {
    py::object planning =
        py::module_::import("ace_edsl.edsl.vector.planning");
    py::object numpy = py::module_::import("numpy");
    py::dict request_data = Build_request_binding(request, numpy);
    py::object request_view =
        planning.attr("_planning_request_from_binding")(
            std::move(request_data));
    REQUEST_VIEW_EXPIRER expire(request_view);
    py::object callback_result = _callback(request_view);
    expire.Expire();
    py::object result_data =
        planning.attr("_provider_result_binding_data")(
            std::move(callback_result));
    return vk::VECTOR_KERNEL_PROVIDER_CALL_RESULT::Success(
        Parse_provider_result(result_data));
  } catch (const py::error_already_set& error) {
    return vk::VECTOR_KERNEL_PROVIDER_CALL_RESULT::Failure(
        std::string("Python vector-kernel provider raised: ") + error.what());
  } catch (const std::exception& error) {
    return vk::VECTOR_KERNEL_PROVIDER_CALL_RESULT::Failure(error.what());
  } catch (...) {
    return vk::VECTOR_KERNEL_PROVIDER_CALL_RESULT::Failure(
        "Python vector-kernel provider failed with an unknown exception");
  }
}

}  // namespace pyace
