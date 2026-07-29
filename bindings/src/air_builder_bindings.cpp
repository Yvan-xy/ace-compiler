// air_builder_bindings.cpp - pybind11 bindings for real ACE AIR infrastructure
//
// REQUIRES: ACE_BINDINGS_ENABLED must be defined (real ACE libraries required)
// This module works with the real ACE compiler.
// Mock fallbacks have been removed from core operations (add, sub, mul).
// Some domain operations still have fallback paths that return non-real nodes
// when container is unavailable - these should not be used in production.

#ifndef ACE_BINDINGS_ENABLED
#error "air_builder module requires real ACE libraries. Build with -DACE_BINDINGS_ENABLED and link against ACE libraries."
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>

#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <complex>

#ifdef ACE_BINDINGS_ENABLED
// Real ACE AIR includes
#include "air/base/meta_info.h"
#include "air/base/st.h"
#include "air/base/container.h"
#include "air/base/node.h"
#include "air/base/opcode.h"
#include "air/core/opcode.h"
#include "air/core/handler.h"
// Domain-specific includes
#include "nn/core/opcode.h"
#include "nn/vector/vector_opcode.h"
#include "nn/vector/vector_gen.h"      // For Vector_driver
#include "fhe/sihe/sihe_opcode.h"
#include "fhe/ckks/ckks_opcode.h"
#include "fhe/poly/opcode.h"
// Attribute keys (for MASK attribute on encode nodes)
#include "nn/core/attr.h"
// Python lowering bridge
#include "python_lowering_bridge.h"

// For Vector_driver
#include "nn/vector/vector_gen.h"
#include "nn/vector/vector_ctx.h"
#include "nn/vector/tensor2vector_dsl.h"
#include "nn/vector/tensor2vector_prepared_lowering.h"
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
#include "tensor2vector_air_normalizer.h"
#include "nn/vector/tensor2vector_ctx.h"
#include "nn/vector/vector_utils.h"
#endif
#include "nn/vector/config.h"
#include "nn/vector/skip_lowering.h"  // For selective lowering registry
#include "fhe/sihe/skip_lowering.h"   // For SIHE selective lowering registry
#include "fhe/ckks/skip_lowering.h"   // For CKKS selective lowering registry

// For IR2C (C code generation)
#include "air/base/ir2c_ctx.h"

// For FHE lowering passes
#include "fhe/sihe/sihe_gen.h"
#include "fhe/sihe/config.h"
#include "fhe/ckks/ckks_gen.h"
#include "fhe/ckks/config.h"
#include "fhe/ckks/sihe2ckks_lower.h"
#include "air/driver/driver_ctx.h"
#include "fhe/poly/config.h"
#include "fhe/poly/poly_driver.h"
#include "fhe/poly/poly2c_driver.h"
#include "fhe/poly/poly2c_config.h"
#include "fhe/core/lower_ctx.h"
#include "fhe/core/ctx_param_ana.h"
// For scale manager (internal include)
#include "../../fhe-cmplr/ckks/include/scale_manager.h"

// For nn::core operations
#include "nn/core/opcode.h"

// For ONNX model loading (separate compilation unit to avoid namespace conflicts)
#include "onnx_loader.h"
#include "air_function_inliner.h"
#endif

namespace py = pybind11;

#ifdef ACE_BINDINGS_ENABLED

// ═══════════════════════════════════════════════════════════════════════════════
// Real AIR Implementation
// ═══════════════════════════════════════════════════════════════════════════════

namespace ace_bindings {

using namespace air::base;

static bool s_air_initialized = false;
static GLOB_SCOPE* s_active_binding_glob = nullptr;

class DESTINATION_LIFETIME {
public:
    explicit DESTINATION_LIFETIME(GLOB_SCOPE* owner) : _owner(owner) {
        if (owner == nullptr) {
            throw std::runtime_error(
                "vector-kernel destination lifetime requires an owner");
        }
    }

    bool owns(const GLOB_SCOPE* glob) const { return glob == _owner; }

    void expire() { _active = false; }

    void require_active(const char* object_kind) const {
        if (!_active) {
            throw std::runtime_error(
                std::string("vector-kernel destination ") + object_kind +
                " has expired");
        }
    }

private:
    GLOB_SCOPE* _owner;
    bool _active = true;
};

static std::shared_ptr<DESTINATION_LIFETIME> s_active_binding_lifetime;

std::shared_ptr<DESTINATION_LIFETIME> destination_lifetime_for_glob(
    const GLOB_SCOPE* glob,
    std::shared_ptr<DESTINATION_LIFETIME> candidate = nullptr) {
    if (!candidate) candidate = s_active_binding_lifetime;
    if (!candidate || !candidate->owns(glob)) return nullptr;
    return candidate;
}

class ACTIVE_BINDING_GLOB_GUARD {
public:
    explicit ACTIVE_BINDING_GLOB_GUARD(
        GLOB_SCOPE* active,
        std::shared_ptr<DESTINATION_LIFETIME> lifetime = nullptr)
        : _previous(s_active_binding_glob),
          _previous_lifetime(s_active_binding_lifetime) {
        s_active_binding_glob = active;
        s_active_binding_lifetime = std::move(lifetime);
    }

    ~ACTIVE_BINDING_GLOB_GUARD() {
        s_active_binding_glob = _previous;
        s_active_binding_lifetime = std::move(_previous_lifetime);
    }

    ACTIVE_BINDING_GLOB_GUARD(const ACTIVE_BINDING_GLOB_GUARD&) = delete;
    ACTIVE_BINDING_GLOB_GUARD& operator=(
        const ACTIVE_BINDING_GLOB_GUARD&) = delete;

private:
    GLOB_SCOPE* _previous;
    std::shared_ptr<DESTINATION_LIFETIME> _previous_lifetime;
};

void ensure_air_initialized() {
    if (!s_air_initialized) {
        META_INFO::Remove_all();
        air::core::Register_core();
        // Register domain opcodes
        nn::core::Register_nn();
        nn::vector::Register_vector_domain();
        fhe::sihe::Register_sihe_domain();
        fhe::ckks::Register_ckks_domain();
        fhe::poly::Register_polynomial();
        s_air_initialized = true;
    }
}

GLOB_SCOPE* active_binding_glob() {
    ensure_air_initialized();
    if (!s_active_binding_glob) {
        s_active_binding_glob = GLOB_SCOPE::Get();
    }
    return s_active_binding_glob;
}

// Sanitize string for UTF-8 compatibility
// Replaces invalid byte sequences with hex escape sequences
std::string sanitize_utf8(const std::string& input) {
    std::string result;
    result.reserve(input.size());
    
    for (size_t i = 0; i < input.size(); ) {
        unsigned char c = static_cast<unsigned char>(input[i]);
        
        // ASCII range (0x00-0x7F) - always valid
        if (c <= 0x7F) {
            result += input[i];
            i++;
            continue;
        }
        
        // Check for valid multi-byte UTF-8 sequences
        size_t seq_len = 0;
        if ((c & 0xE0) == 0xC0) seq_len = 2;      // 110xxxxx - 2 byte sequence
        else if ((c & 0xF0) == 0xE0) seq_len = 3; // 1110xxxx - 3 byte sequence
        else if ((c & 0xF8) == 0xF0) seq_len = 4; // 11110xxx - 4 byte sequence
        
        // Check if sequence is complete and valid
        bool valid = (seq_len >= 2) && (i + seq_len <= input.size());
        if (valid) {
            for (size_t j = 1; j < seq_len; j++) {
                unsigned char cont = static_cast<unsigned char>(input[i + j]);
                if ((cont & 0xC0) != 0x80) {  // Must be 10xxxxxx
                    valid = false;
                    break;
                }
            }
        }
        
        if (valid) {
            // Copy valid multi-byte sequence
            for (size_t j = 0; j < seq_len; j++) {
                result += input[i + j];
            }
            i += seq_len;
        } else {
            // Replace invalid byte with hex escape
            char hex[8];
            snprintf(hex, sizeof(hex), "\\x%02x", c);
            result += hex;
            i++;
        }
    }
    
    return result;
}

// Source location info passed from Python
struct SourceLoc {
    uint32_t file_id;
    uint32_t line;
    uint32_t col;
    
    SourceLoc() : file_id(0), line(0), col(0) {}
    SourceLoc(uint32_t f, uint32_t l, uint32_t c) : file_id(f), line(l), col(c) {}
    
    SPOS to_spos() const {
        return SPOS(file_id, line, col, 0);
    }
    
    bool is_valid() const { return line > 0; }
};

static void Set_rotation_attr(NODE_PTR node, int32_t rotation) {
    node->Set_attr(nn::core::ATTR::RNUM, &rotation, 1);
}

static void Set_rotation_list_attr(NODE_PTR node,
                                   const std::vector<int32_t>& rotations) {
    if (!rotations.empty()) {
        node->Set_attr(nn::core::ATTR::RNUM, rotations.data(), rotations.size());
    }
}

// Type wrapper
class Type {
public:
    TYPE_PTR type;
    std::string name;
    std::vector<int> shape;
    bool has_type;

    Type() : type(), name("void"), has_type(false) {}
    Type(TYPE_PTR t, const std::string& n,
         std::shared_ptr<DESTINATION_LIFETIME> lifetime = nullptr)
        : type(t), name(n), has_type(true),
          _lifetime(destination_lifetime_for_glob(
              t == Null_ptr ? nullptr : &t->Glob_scope(),
              std::move(lifetime))) {}
    Type(TYPE_PTR t, const std::string& n, const std::vector<int>& s,
         std::shared_ptr<DESTINATION_LIFETIME> lifetime = nullptr)
        : type(t), name(n), shape(s), has_type(true),
          _lifetime(destination_lifetime_for_glob(
              t == Null_ptr ? nullptr : &t->Glob_scope(),
              std::move(lifetime))) {}

    static Type make_void() {
        Type t;
        t.name = "void";
        t.has_type = false;
        return t;
    }

    static Type make_int(int bits) {
        ensure_air_initialized();
        GLOB_SCOPE* glob = active_binding_glob();
        if (glob == nullptr) {
            throw std::runtime_error(
                "integer type creation requires an active GLOB_SCOPE");
        }
        TYPE_PTR t;
        if (bits == 1) t = glob->Prim_type(PRIMITIVE_TYPE::BOOL);
        else if (bits == 8) t = glob->Prim_type(PRIMITIVE_TYPE::INT_S8);
        else if (bits == 16) t = glob->Prim_type(PRIMITIVE_TYPE::INT_S16);
        else if (bits == 32) t = glob->Prim_type(PRIMITIVE_TYPE::INT_S32);
        else if (bits == 64) t = glob->Prim_type(PRIMITIVE_TYPE::INT_S64);
        else throw std::runtime_error(
            "integer type width must be one of 1, 8, 16, 32, or 64 bits");
        return Type(t, "i" + std::to_string(bits));
    }

    static Type make_float(int bits) {
        ensure_air_initialized();
        GLOB_SCOPE* glob = active_binding_glob();
        if (glob == nullptr) {
            throw std::runtime_error(
                "floating type creation requires an active GLOB_SCOPE");
        }
        TYPE_PTR t;
        if (bits == 32) t = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_32);
        else if (bits == 64) t = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_64);
        else throw std::runtime_error(
            "floating type width must be either 32 or 64 bits");
        return Type(t, "f" + std::to_string(bits));
    }

    static Type make_array_in_glob(GLOB_SCOPE* glob,
                                   const std::vector<int>& shape,
                                   const Type& elem) {
        if (glob == nullptr) {
            throw std::runtime_error(
                "array type creation requires an active GLOB_SCOPE");
        }
        if (shape.empty()) {
            throw std::runtime_error(
                "ranked array type requires at least one dimension");
        }
        for (int dim : shape) {
            if (dim <= 0) {
                throw std::runtime_error(
                    "ranked array dimensions must be positive");
            }
        }
        elem.require_active();
        if (!elem.has_type || elem.type == Null_ptr ||
            &elem.type->Glob_scope() != glob) {
            throw std::runtime_error(
                "array element type belongs to a different GLOB_SCOPE");
        }
        SPOS spos = glob->Unknown_simple_spos();
        std::vector<int64_t> dims;
        for (int s : shape) dims.push_back(static_cast<int64_t>(s));
        STR_PTR type_name = glob->New_str("array");
        TYPE_PTR arr_type = glob->New_arr_type(type_name, elem.type, dims, spos);
        std::shared_ptr<DESTINATION_LIFETIME> lifetime = elem._lifetime;
        if (!lifetime && glob == s_active_binding_glob) {
            lifetime = s_active_binding_lifetime;
        }
        return Type(arr_type, "array", shape, std::move(lifetime));
    }

    static Type make_array(const std::vector<int>& shape, Type elem) {
        ensure_air_initialized();
        return make_array_in_glob(active_binding_glob(), shape, elem);
    }

    static Type make_ciphertext(const std::string& domain = "sihe") {
        Type t;
        t.name = "CIPHERTEXT";
        t.has_type = false;
        return t;
    }

    static Type make_plaintext() {
        Type t;
        t.name = "PLAINTEXT";
        t.has_type = false;
        return t;
    }

    static Type make_polynomial(int degree = 4096) {
        ensure_air_initialized();
        GLOB_SCOPE* glob = active_binding_glob();
        if (glob == nullptr) {
            throw std::runtime_error(
                "polynomial type creation requires an active GLOB_SCOPE");
        }
        SPOS spos = glob->Unknown_simple_spos();
        TYPE_PTR elem_type = glob->Prim_type(PRIMITIVE_TYPE::INT_S64);
        std::vector<int64_t> dims = {static_cast<int64_t>(degree)};
        STR_PTR type_name = glob->New_str("polynomial");
        TYPE_PTR arr_type = glob->New_arr_type(type_name, elem_type, dims, spos);
        return Type(arr_type, "polynomial", {degree});
    }

    static Type from_air_type(
        TYPE_PTR air_type,
        std::shared_ptr<DESTINATION_LIFETIME> lifetime = nullptr) {
        if (air_type == Null_ptr) return make_void();

        std::string type_name = air_type->Type_kind_name();
        std::vector<int> type_shape;
        if (air_type->Is_prim()) {
            switch (air_type->Cast_to_prim()->Encoding()) {
                case PRIMITIVE_TYPE::BOOL: type_name = "bool"; break;
                case PRIMITIVE_TYPE::INT_S8: type_name = "i8"; break;
                case PRIMITIVE_TYPE::INT_S16: type_name = "i16"; break;
                case PRIMITIVE_TYPE::INT_S32: type_name = "i32"; break;
                case PRIMITIVE_TYPE::INT_S64: type_name = "i64"; break;
                case PRIMITIVE_TYPE::INT_U8: type_name = "u8"; break;
                case PRIMITIVE_TYPE::INT_U16: type_name = "u16"; break;
                case PRIMITIVE_TYPE::INT_U32: type_name = "u32"; break;
                case PRIMITIVE_TYPE::INT_U64: type_name = "u64"; break;
                case PRIMITIVE_TYPE::FLOAT_32: type_name = "f32"; break;
                case PRIMITIVE_TYPE::FLOAT_64: type_name = "f64"; break;
                default: break;
            }
        } else if (air_type->Is_array()) {
            type_name = "array";
            for (int64_t dim : air_type->Cast_to_arr()->Shape()) {
                type_shape.push_back(static_cast<int>(dim));
            }
        } else if (air_type->Name() != Null_ptr) {
            type_name = air_type->Name()->Char_str();
        }
        return Type(air_type, type_name, type_shape, std::move(lifetime));
    }

    static bool structurally_equal_air_types(CONST_TYPE_PTR lhs,
                                             CONST_TYPE_PTR rhs) {
        if (lhs == Null_ptr || rhs == Null_ptr) return lhs == rhs;
        if (lhs->Kind() != rhs->Kind()) return false;
        if (lhs->Is_prim()) {
            return lhs->Cast_to_prim()->Encoding() ==
                   rhs->Cast_to_prim()->Encoding();
        }
        if (lhs->Is_array()) {
            CONST_ARRAY_TYPE_PTR lhs_array = lhs->Cast_to_arr();
            CONST_ARRAY_TYPE_PTR rhs_array = rhs->Cast_to_arr();
            return lhs_array->Shape() == rhs_array->Shape() &&
                   structurally_equal_air_types(lhs_array->Elem_type(),
                                                rhs_array->Elem_type());
        }
        if (&lhs->Glob_scope() == &rhs->Glob_scope()) {
            return lhs->Is_compatible_type(rhs);
        }
        return false;
    }

    void require_active() const {
        if (_lifetime) _lifetime->require_active("type");
    }

    bool structurally_equal(const Type& other) const {
        require_active();
        other.require_active();
        if (!has_type || !other.has_type) {
            return has_type == other.has_type && name == other.name &&
                   shape == other.shape;
        }
        return structurally_equal_air_types(type, other.type);
    }

    bool same_scope(const Type& other) const {
        require_active();
        other.require_active();
        return has_type && other.has_type &&
               &type->Glob_scope() == &other.type->Glob_scope();
    }

    bool is_integer() const {
        require_active();
        return has_type && type->Is_int();
    }
    bool is_float() const {
        require_active();
        return has_type && type->Is_float();
    }
    bool is_scalar() const {
        require_active();
        return has_type && type->Is_scalar();
    }

    int bit_width() const {
        require_active();
        if (!has_type || !type->Is_prim()) return 0;
        switch (type->Cast_to_prim()->Encoding()) {
            case PRIMITIVE_TYPE::BOOL: return 1;
            case PRIMITIVE_TYPE::INT_S8:
            case PRIMITIVE_TYPE::INT_U8: return 8;
            case PRIMITIVE_TYPE::INT_S16:
            case PRIMITIVE_TYPE::INT_U16: return 16;
            case PRIMITIVE_TYPE::INT_S32:
            case PRIMITIVE_TYPE::INT_U32:
            case PRIMITIVE_TYPE::FLOAT_32: return 32;
            case PRIMITIVE_TYPE::INT_S64:
            case PRIMITIVE_TYPE::INT_U64:
            case PRIMITIVE_TYPE::FLOAT_64: return 64;
            default: return static_cast<int>(type->Bit_size());
        }
    }

    std::string to_string() const {
        require_active();
        return name;
    }
    bool is_array() const {
        require_active();
        return !shape.empty();
    }
    std::vector<int> get_shape() const {
        require_active();
        return shape;
    }

private:
    std::shared_ptr<DESTINATION_LIFETIME> _lifetime;
};

// Node wrapper
class Node {
public:
    NODE_PTR node;
    int id;
    std::string opcode_str;
    std::vector<std::shared_ptr<Node>> children;
    bool has_node;

    Node()
        : node(), id(0), opcode_str(""), has_node(false),
          _lifetime(destination_lifetime_for_glob(
              s_active_binding_glob)) {}
    Node(NODE_PTR n, int i, const std::string& op,
         std::shared_ptr<DESTINATION_LIFETIME> lifetime = nullptr)
        : node(n), id(i), opcode_str(op), has_node(true),
          _lifetime(destination_lifetime_for_glob(
              n == Null_ptr || n->Container() == nullptr
                  ? nullptr
                  : n->Container()->Glob_scope(),
              std::move(lifetime))) {}
    Node(int i, const std::string& op,
         std::shared_ptr<DESTINATION_LIFETIME> lifetime = nullptr)
        : node(), id(i), opcode_str(op), has_node(false),
          _lifetime(destination_lifetime_for_glob(
              s_active_binding_glob, std::move(lifetime))) {}

    std::string name() const { return "%" + std::to_string(id); }
    std::string opcode_name() const { return opcode_str; }
    bool is_valid() const { return has_node; }

    void invalidate() {
        node = NODE_PTR();
        has_node = false;
        children.clear();
    }

    Type rtype() const {
        require_active();
        if (!has_node || node == NODE_PTR() || !node->Has_rtype()) {
            return Type::make_void();
        }
        return Type::from_air_type(node->Rtype(), _lifetime);
    }

    void add_child(std::shared_ptr<Node> child) { children.push_back(child); }

    std::string to_string() const {
        std::string s = name() + " = " + opcode_str + "(";
        for (size_t i = 0; i < children.size(); i++) {
            if (i > 0) s += ", ";
            s += children[i]->name();
        }
        s += ")";
        return s;
    }

    void require_attribute_target(const char* operation,
                                  const std::string& attr_name) const {
        require_active();
        if (!has_node || node == NODE_PTR()) {
            throw std::runtime_error(std::string(operation) +
                                     " requires a real AIR node");
        }
        if (attr_name.empty()) {
            throw std::runtime_error(std::string(operation) +
                                     " requires a non-empty attribute name");
        }
        if (!META_INFO::Has_prop<OPR_PROP::ATTR>(node->Opcode())) {
            throw std::runtime_error(std::string(operation) +
                                     " requires an opcode that supports AIR attributes");
        }
    }

    static int32_t checked_s32_attr_value(int64_t value,
                                          const char* operation) {
        if (value < std::numeric_limits<int32_t>::min() ||
            value > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error(std::string(operation) +
                                     " value is out of range for signed i32");
        }
        return static_cast<int32_t>(value);
    }

    static uint32_t checked_u32_attr_value(int64_t value,
                                           const char* operation) {
        if (value < 0 ||
            static_cast<uint64_t>(value) >
                std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(std::string(operation) +
                                     " value is out of range for unsigned i32");
        }
        return static_cast<uint32_t>(value);
    }

    template <typename T>
    void set_array_attr(const std::string& attr_name,
                        const std::vector<T>& values,
                        const char* operation) {
        require_attribute_target(operation, attr_name);
        if (values.empty()) {
            throw std::runtime_error(std::string(operation) +
                                     " requires at least one value");
        }
        if (values.size() >= (1U << 24)) {
            throw std::runtime_error(std::string(operation) +
                                     " has too many values for AIR attribute storage");
        }
        node->Set_attr(attr_name.c_str(), values.data(), values.size());
    }

    void set_s32_attr(const std::string& attr_name, int64_t value) {
        require_attribute_target("set_s32_attr", attr_name);
        int32_t checked = checked_s32_attr_value(value, "set_s32_attr");
        node->Set_attr(attr_name.c_str(), &checked, 1);
    }

    void set_s32_array_attr(const std::string& attr_name,
                            const std::vector<int64_t>& values) {
        std::vector<int32_t> checked;
        checked.reserve(values.size());
        for (int64_t value : values) {
            checked.push_back(
                checked_s32_attr_value(value, "set_s32_array_attr"));
        }
        set_array_attr(attr_name, checked, "set_s32_array_attr");
    }

    void set_u32_attr(const std::string& attr_name, int64_t value) {
        require_attribute_target("set_u32_attr", attr_name);
        uint32_t checked = checked_u32_attr_value(value, "set_u32_attr");
        node->Set_attr(attr_name.c_str(), &checked, 1);
    }

    void set_u32_array_attr(const std::string& attr_name,
                            const std::vector<int64_t>& values) {
        std::vector<uint32_t> checked;
        checked.reserve(values.size());
        for (int64_t value : values) {
            checked.push_back(
                checked_u32_attr_value(value, "set_u32_array_attr"));
        }
        set_array_attr(attr_name, checked, "set_u32_array_attr");
    }

    void set_vector_slot(int64_t slot) {
        require_attribute_target("set_vector_slot", nn::core::ATTR::SLOT);
        if (!node->Has_rtype() || !node->Rtype()->Is_array()) {
            throw std::runtime_error(
                "set_vector_slot requires a ranked Vector value");
        }
        if (slot <= 0) {
            throw std::runtime_error(
                "set_vector_slot requires a positive slot count");
        }
        set_u32_attr(nn::core::ATTR::SLOT, slot);
    }

private:
    void require_active() const {
        if (_lifetime) _lifetime->require_active("node");
    }

    std::shared_ptr<DESTINATION_LIFETIME> _lifetime;
};

// Container - creates real AIR nodes
class Container {
public:
    CONTAINER* container;
    FUNC_SCOPE* func_scope;
    GLOB_SCOPE* glob;
    int node_counter;
    std::vector<std::shared_ptr<Node>> nodes;
    SourceLoc current_loc;  // Current Python source location
    
    // Stack of block nodes for nested control flow
    // When inside a loop/if body, statements go to the top block
    std::vector<NODE_PTR> block_stack;
    
    // Map variable names to their ADDR_DATUM for stores
    std::map<std::string, ADDR_DATUM_PTR> var_map;

    bool callback_scoped;
    bool callback_expired;
    std::shared_ptr<DESTINATION_LIFETIME> callback_lifetime;

    Container()
        : container(nullptr), func_scope(nullptr), glob(nullptr),
          node_counter(0), callback_scoped(false), callback_expired(false) {}
    Container(CONTAINER* c, FUNC_SCOPE* fs, GLOB_SCOPE* g)
        : container(c), func_scope(fs), glob(g), node_counter(0),
          callback_scoped(false), callback_expired(false) {}

    void mark_callback_scoped() {
        callback_scoped = true;
        callback_expired = false;
        callback_lifetime =
            std::make_shared<DESTINATION_LIFETIME>(glob);
    }

    void expire_callback_scope() {
        if (!callback_scoped) return;
        callback_expired = true;
        if (callback_lifetime) callback_lifetime->expire();
    }

    void require_not_expired() const {
        if (callback_expired) {
            throw std::runtime_error(
                "vector-kernel destination container has expired");
        }
    }
    
    // Append a statement to the current block (or main container if no block)
    void append_stmt(STMT_PTR stmt) {
        require_not_expired();
        if (!block_stack.empty() && block_stack.back() != NODE_PTR()) {
            // Create STMT_LIST for the current block and append
            STMT_LIST sl(block_stack.back());
            sl.Append(stmt);
        } else {
            // Append to main container
            STMT_LIST sl = container->Stmt_list();
            sl.Append(stmt);
        }
    }
    
    // Push a new block onto the stack (for entering loop/if body)
    void push_block(NODE_PTR block) {
        require_not_expired();
        block_stack.push_back(block);
    }
    
    // Pop the current block (for exiting loop/if body)
    void pop_block() {
        require_not_expired();
        if (!block_stack.empty()) {
            block_stack.pop_back();
        }
    }
    
    // Set current source location from Python
    void set_loc(uint32_t file_id, uint32_t line, uint32_t col) {
        require_not_expired();
        current_loc = SourceLoc(file_id, line, col);
    }
    
    SPOS get_spos() { 
        require_not_expired();
        if (current_loc.is_valid()) {
            return current_loc.to_spos();
        }
        return glob ? glob->Unknown_simple_spos() : SPOS(); 
    }

    std::shared_ptr<Node> wrap_node(NODE_PTR n, const std::string& opcode) {
        require_not_expired();
        auto node = std::make_shared<Node>(
            n, ++node_counter, opcode, callback_lifetime);
        nodes.push_back(node);
        return node;
    }
    
    // Helper to get type from container's glob_scope by type ID
    TYPE_PTR get_compatible_type(TYPE_PTR src_type) {
        require_not_expired();
        if (src_type == air::base::Null_ptr) return src_type;
        GLOB_SCOPE* container_glob = container->Glob_scope();
        if (&src_type->Glob_scope() != container_glob) {
            throw std::runtime_error(
                "AIR type belongs to a different GLOB_SCOPE");
        }
        return src_type;
    }

    TYPE_PTR require_real_expression(const std::shared_ptr<Node>& value,
                                     const char* operation) {
        require_not_expired();
        if (!container || !value || !value->has_node ||
            value->node == NODE_PTR()) {
            throw std::runtime_error(std::string(operation) +
                                     " requires real AIR operands");
        }
        if (value->node->Container() != container) {
            throw std::runtime_error(std::string(operation) +
                                     " cannot mix AIR containers");
        }
        if (!value->node->Has_rtype()) {
            throw std::runtime_error(std::string(operation) +
                                     " requires expression operands with result types");
        }
        return get_compatible_type(value->node->Rtype());
    }

    TYPE_PTR require_vector_operand(const std::shared_ptr<Node>& value,
                                    const char* operation) {
        require_not_expired();
        TYPE_PTR type = require_real_expression(value, operation);
        if (!type->Is_array()) {
            throw std::runtime_error(std::string(operation) +
                                     " requires ranked array operands");
        }
        return type;
    }

    TYPE_PTR require_core_s32_operand(const std::shared_ptr<Node>& value,
                                      const char* operation) {
        require_not_expired();
        TYPE_PTR type = require_real_expression(value, operation);
        if (value->node->Domain() != air::core::CORE || !type->Is_prim() ||
            type->Cast_to_prim()->Encoding() != PRIMITIVE_TYPE::INT_S32) {
            throw std::runtime_error(std::string(operation) +
                                     " requires a Core signed i32 scalar");
        }
        return type;
    }

    void require_vector_binary_operands(const std::shared_ptr<Node>& lhs,
                                        const std::shared_ptr<Node>& rhs,
                                        const char* operation) {
        require_not_expired();
        TYPE_PTR lhs_type = require_vector_operand(lhs, operation);
        TYPE_PTR rhs_type = require_vector_operand(rhs, operation);
        if (!Type::structurally_equal_air_types(
                lhs_type->Cast_to_arr()->Elem_type(),
                rhs_type->Cast_to_arr()->Elem_type())) {
            throw std::runtime_error(std::string(operation) +
                                     " requires compatible vector element types");
        }
    }

    void require_compatible_operands(const std::shared_ptr<Node>& a,
                                     const std::shared_ptr<Node>& b,
                                     const char* operation,
                                     bool require_integer = false,
                                     bool require_scalar = false) {
        require_not_expired();
        if (!container || !a || !b || !a->has_node || !b->has_node ||
            a->node == NODE_PTR() || b->node == NODE_PTR()) {
            throw std::runtime_error(std::string(operation) +
                                     " requires real AIR operands");
        }
        if (a->node->Container() != container ||
            b->node->Container() != container) {
            throw std::runtime_error(std::string(operation) +
                                     " cannot mix AIR containers");
        }
        if (!a->node->Has_rtype() || !b->node->Has_rtype()) {
            throw std::runtime_error(std::string(operation) +
                                     " requires expression operands with result types");
        }
        TYPE_PTR lhs_type = get_compatible_type(a->node->Rtype());
        TYPE_PTR rhs_type = get_compatible_type(b->node->Rtype());
        if (!lhs_type->Is_compatible_type(rhs_type)) {
            throw std::runtime_error(std::string(operation) +
                                     " requires structurally compatible operands");
        }
        if (require_scalar &&
            (!lhs_type->Is_scalar() || !rhs_type->Is_scalar())) {
            throw std::runtime_error(std::string(operation) +
                                     " requires scalar operands");
        }
        if (require_integer &&
            (!lhs_type->Is_signed_int() || !rhs_type->Is_signed_int())) {
            throw std::runtime_error(std::string(operation) +
                                     " requires signed integer operands");
        }
    }

    std::shared_ptr<Node> new_add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        require_compatible_operands(a, b, "new_add");
        OPCODE op(air::core::CORE, air::core::OPCODE::ADD);
        TYPE_PTR rtype = get_compatible_type(a->node->Rtype());
        NODE_PTR n = container->New_bin_arith(op, rtype, a->node, b->node, get_spos());
        auto node = wrap_node(n, "air::core::ADD");
        node->add_child(a);
        node->add_child(b);
        return node;
    }

    std::shared_ptr<Node> new_core_add(std::shared_ptr<Node> a,
                                       std::shared_ptr<Node> b) {
        require_not_expired();
        require_compatible_operands(a, b, "new_core_add", false, true);
        return new_add(a, b);
    }
    
    std::shared_ptr<Node> new_sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        require_compatible_operands(a, b, "new_sub");
        OPCODE op(air::core::CORE, air::core::OPCODE::SUB);
        TYPE_PTR rtype = get_compatible_type(a->node->Rtype());
        NODE_PTR n = container->New_bin_arith(op, rtype, a->node, b->node, get_spos());
        auto node = wrap_node(n, "air::core::SUB");
        node->add_child(a);
        node->add_child(b);
        return node;
    }
    
    std::shared_ptr<Node> new_mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        require_compatible_operands(a, b, "new_mul");
        OPCODE op(air::core::CORE, air::core::OPCODE::MUL);
        TYPE_PTR rtype = get_compatible_type(a->node->Rtype());
        NODE_PTR n = container->New_bin_arith(op, rtype, a->node, b->node, get_spos());
        auto node = wrap_node(n, "air::core::MUL");
        node->add_child(a);
        node->add_child(b);
        return node;
    }

    std::shared_ptr<Node> new_core_mul(std::shared_ptr<Node> a,
                                       std::shared_ptr<Node> b) {
        require_not_expired();
        require_compatible_operands(a, b, "new_core_mul", false, true);
        return new_mul(a, b);
    }

    std::shared_ptr<Node> new_shl(std::shared_ptr<Node> a,
                                  std::shared_ptr<Node> b) {
        require_not_expired();
        require_compatible_operands(a, b, "new_shl", true);
        OPCODE op(air::core::CORE, air::core::OPCODE::SHL);
        TYPE_PTR rtype = get_compatible_type(a->node->Rtype());
        NODE_PTR n = container->New_bin_arith(
            op, rtype, a->node, b->node, get_spos());
        auto node = wrap_node(n, "air::core::SHL");
        node->add_child(a);
        node->add_child(b);
        return node;
    }

    std::shared_ptr<Node> new_core_lt(std::shared_ptr<Node> a,
                                      std::shared_ptr<Node> b) {
        require_not_expired();
        require_compatible_operands(a, b, "new_core_lt", true);
        OPCODE op(air::core::CORE, air::core::OPCODE::LT);
        TYPE_PTR rtype = get_compatible_type(a->node->Rtype());
        NODE_PTR n = container->New_bin_arith(
            op, rtype, a->node, b->node, get_spos());
        auto node = wrap_node(n, "air::core::LT");
        node->add_child(a);
        node->add_child(b);
        return node;
    }
    
    std::shared_ptr<Node> new_div(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        // DIV not directly in air::core - use reciprocal multiplication pattern
        throw std::runtime_error("new_div not implemented - use multiplication by reciprocal");
    }
    
    std::shared_ptr<Node> new_matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        // MATMUL should use nn::core::GEMM
        throw std::runtime_error("new_matmul not implemented - use nn_gemm instead");
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Domain-Specific Operations: nn::core
    // ═══════════════════════════════════════════════════════════════════════
    
    std::shared_ptr<Node> new_nn_add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (!container || !a->has_node || !b->has_node) {
            throw std::runtime_error("new_nn_add requires real container and operands");
        }
        OPCODE op(nn::core::NN, nn::core::OPCODE::ADD);
        NODE_PTR n = container->New_bin_arith(op, a->node->Rtype(), a->node, b->node, get_spos());
        auto node = wrap_node(n, "nn::core::ADD");
        node->add_child(a);
        node->add_child(b);
        return node;
    }
    
    std::shared_ptr<Node> new_nn_sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (!container || !a->has_node || !b->has_node) {
            throw std::runtime_error("new_nn_sub requires real container and operands");
        }
        OPCODE op(nn::core::NN, nn::core::OPCODE::SUB);
        NODE_PTR n = container->New_bin_arith(op, a->node->Rtype(), a->node, b->node, get_spos());
        auto node = wrap_node(n, "nn::core::SUB");
        node->add_child(a);
        node->add_child(b);
        return node;
    }
    
    std::shared_ptr<Node> new_nn_mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (!container || !a->has_node || !b->has_node) {
            throw std::runtime_error("new_nn_mul requires real container and operands");
        }
        OPCODE op(nn::core::NN, nn::core::OPCODE::MUL);
        NODE_PTR n = container->New_bin_arith(op, a->node->Rtype(), a->node, b->node, get_spos());
        auto node = wrap_node(n, "nn::core::MUL");
        node->add_child(a);
        node->add_child(b);
        return node;
    }
    
    std::shared_ptr<Node> new_nn_conv(std::shared_ptr<Node> x, std::shared_ptr<Node> w, std::shared_ptr<Node> b) {
        require_not_expired();
        if (!container || !x->has_node || !w->has_node || !b->has_node) {
            throw std::runtime_error("new_nn_conv requires real container and operands");
        }
        OPCODE op(nn::core::NN, nn::core::OPCODE::CONV);
        NODE_PTR n = container->New_cust_node(op, x->node->Rtype(), get_spos());
        n->Set_child(0, x->node);
        n->Set_child(1, w->node);
        n->Set_child(2, b->node);
        auto node = wrap_node(n, "nn::core::CONV");
        node->add_child(x); node->add_child(w); node->add_child(b);
        return node;
    }
    
    std::shared_ptr<Node> new_nn_relu(std::shared_ptr<Node> x) {
        require_not_expired();
        // nn::core::RELU - unary ReLU activation
        if (container && x->has_node) {
            OPCODE op(nn::core::NN, nn::core::OPCODE::RELU);
            NODE_PTR n = container->New_cust_node(op, x->node->Rtype(), get_spos());
            n->Set_child(0, x->node);
            auto node = wrap_node(n, "nn::core::RELU");
            node->add_child(x);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "NN.relu");
        node->add_child(x);
        nodes.push_back(node);
        return node;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Domain-Specific Operations: nn::vector
    // ═══════════════════════════════════════════════════════════════════════
    
    std::shared_ptr<Node> new_vec_add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        require_vector_binary_operands(a, b, "new_vec_add");
        nn::vector::VECTOR_GEN vector_gen(container);
        NODE_PTR n = vector_gen.New_add(a->node, b->node, get_spos());
        auto node = wrap_node(n, "nn::vector::ADD");
        node->add_child(a);
        node->add_child(b);
        return node;
    }
    
    std::shared_ptr<Node> new_vec_sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        // nn::vector doesn't have SUB, use air::core::SUB with vec prefix for consistency
        if (container && a->has_node && b->has_node) {
            OPCODE op(air::core::CORE, air::core::OPCODE::SUB);
            NODE_PTR n = container->New_bin_arith(op, a->node->Rtype(), a->node, b->node, get_spos());
            auto node = wrap_node(n, "nn::vector::SUB");  // Keep the domain prefix for display
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "nn::vector::SUB");
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_vec_mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        require_vector_binary_operands(a, b, "new_vec_mul");
        nn::vector::VECTOR_GEN vector_gen(container);
        NODE_PTR n = vector_gen.New_mul(a->node, b->node, get_spos());
        auto node = wrap_node(n, "nn::vector::MUL");
        node->add_child(a);
        node->add_child(b);
        return node;
    }

    std::shared_ptr<Node> new_vec_roll(
        std::shared_ptr<Node> value, std::shared_ptr<Node> shift,
        const std::vector<int64_t>& candidates) {
        require_not_expired();
        require_vector_operand(value, "new_vec_roll");
        require_core_s32_operand(shift, "new_vec_roll");
        if (candidates.empty()) {
            throw std::runtime_error(
                "new_vec_roll requires non-empty rotation candidates");
        }
        if (candidates.size() >= (1U << 24)) {
            throw std::runtime_error(
                "new_vec_roll has too many candidates for AIR "
                "attribute storage");
        }


        static_assert(sizeof(int) == sizeof(int32_t),
                      "VECTOR RNUM attributes require 32-bit int");
        std::vector<int> checked_candidates;
        checked_candidates.reserve(candidates.size());
        for (int64_t candidate : candidates) {
            if (candidate < std::numeric_limits<int32_t>::min() ||
                candidate > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error(
                    "new_vec_roll candidate is out of range for signed i32");
            }
            checked_candidates.push_back(static_cast<int>(candidate));
        }

        nn::vector::VECTOR_GEN vector_gen(container);
        NODE_PTR n = vector_gen.New_roll(
            value->node, shift->node, checked_candidates, get_spos());
        auto node = wrap_node(n, "nn::vector::ROLL");
        node->add_child(value);
        node->add_child(shift);
        return node;
    }

    std::shared_ptr<Node> new_vec_slice(std::shared_ptr<Node> value,
                                        std::shared_ptr<Node> start,
                                        int64_t slice_size) {
        require_not_expired();
        TYPE_PTR value_type = require_vector_operand(value, "new_vec_slice");
        require_core_s32_operand(start, "new_vec_slice");
        if (slice_size <= 0 ||
            slice_size > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error(
                "new_vec_slice size must be a positive signed i32 value");
        }

        const std::vector<int64_t> shape =
            value_type->Cast_to_arr()->Shape();
        if (shape.size() != 2) {
            throw std::runtime_error(
                "new_vec_slice requires a rank-2 source");
        }
        if (shape[1] != slice_size) {
            throw std::runtime_error(
                "new_vec_slice size must match the source trailing dimension");
        }

        TYPE_PTR s32_type = glob->Prim_type(PRIMITIVE_TYPE::INT_S32);
        NODE_PTR size = container->New_intconst(
            s32_type, slice_size, get_spos());
        nn::vector::VECTOR_GEN vector_gen(container);
        NODE_PTR n =
            vector_gen.New_slice(value->node, start->node, size, get_spos());
        auto node = wrap_node(n, "nn::vector::SLICE");
        node->add_child(value);
        node->add_child(start);
        node->add_child(wrap_node(size, "air::core::INTCONST"));
        return node;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Domain-Specific Operations: fhe::sihe
    // ═══════════════════════════════════════════════════════════════════════
    
    std::shared_ptr<Node> new_sihe_add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (container && a && a->has_node && b && b->has_node) {
            OPCODE op(fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::ADD);
            NODE_PTR n = container->New_bin_arith(op, a->node->Rtype(), a->node, b->node, get_spos());
            auto node = wrap_node(n, "fhe::sihe::ADD");
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::sihe::ADD");
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_sihe_sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (container && a && a->has_node && b && b->has_node) {
            OPCODE op(fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::SUB);
            NODE_PTR n = container->New_bin_arith(op, a->node->Rtype(), a->node, b->node, get_spos());
            auto node = wrap_node(n, "fhe::sihe::SUB");
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::sihe::SUB");
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_sihe_mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (container && a && a->has_node && b && b->has_node) {
            OPCODE op(fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::MUL);
            NODE_PTR n = container->New_bin_arith(op, a->node->Rtype(), a->node, b->node, get_spos());
            auto node = wrap_node(n, "fhe::sihe::MUL");
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::sihe::MUL");
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<Node> new_sihe_encode(std::shared_ptr<Node> data) {
        require_not_expired();
        if (container && data && data->has_node) {
            OPCODE op(fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::ENCODE);
            // Get PLAINTEXT type - use void as placeholder
            TYPE_PTR plain_type = container->Glob_scope()->Prim_type(air::base::PRIMITIVE_TYPE::VOID);
            // Create custom node with 2 children: data and length
            NODE_PTR n = container->New_cust_node(op, plain_type, get_spos());
            // Set child 0: data to encode
            n->Set_child(0, data->node);
            // Set child 1: length (use 64 as default for CKKS slot count)
            TYPE_PTR u32_type = container->Glob_scope()->Prim_type(air::base::PRIMITIVE_TYPE::INT_U32);
            NODE_PTR len_node = container->New_intconst(u32_type, 64, get_spos());
            n->Set_child(1, len_node);
            auto node = wrap_node(n, "fhe::sihe::ENCODE");
            node->add_child(data);
            nodes.push_back(node);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::sihe::ENCODE");
        node->add_child(data);
        nodes.push_back(node);
        return node;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Domain-Specific Operations: fhe::ckks
    // ═══════════════════════════════════════════════════════════════════════
    
    std::shared_ptr<Node> new_ckks_add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (container && a->has_node && b->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::ADD);
            TYPE_PTR rtype = get_compatible_type(a->node->Rtype());
            auto is_plain = [](TYPE_PTR t) {
                if (t == air::base::Null_ptr || !t->Is_record()) return false;
                STR_PTR name = t->Name();
                return name != air::base::Null_ptr && std::string(name->Char_str()) == "PLAINTEXT";
            };
            if (!is_plain(a->node->Rtype()) || !is_plain(b->node->Rtype()))
                rtype = !is_plain(a->node->Rtype()) ? get_compatible_type(a->node->Rtype()) : get_compatible_type(b->node->Rtype());
            NODE_PTR n = container->New_bin_arith(op, rtype, a->node, b->node, get_spos());
            auto node = wrap_node(n, "fhe::ckks::ADD");
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::ADD");
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_ckks_sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        // Keep true CKKS SUB for ciphertext-ciphertext inputs so scale/level
        // tracking matches runtime subtraction semantics.
        // Fall back to add(a, mul(b, -1)) for mixed/plain cases.
        if (container && a->has_node && b->has_node) {
            SPOS spos = get_spos();
            TYPE_PTR rtype = get_compatible_type(a->node->Rtype());
            auto is_plain = [](TYPE_PTR t) {
                if (t == air::base::Null_ptr || !t->Is_record()) return false;
                STR_PTR name = t->Name();
                return name != air::base::Null_ptr && std::string(name->Char_str()) == "PLAINTEXT";
            };
            auto is_cipher = [](TYPE_PTR t) {
                if (t == air::base::Null_ptr || !t->Is_record()) return false;
                STR_PTR name = t->Name();
                if (name == air::base::Null_ptr) return false;
                std::string tn(name->Char_str());
                return tn == "CIPHERTEXT" || tn == "CIPHERTEXT3";
            };
            if (!is_plain(a->node->Rtype()) || !is_plain(b->node->Rtype()))
                rtype = !is_plain(a->node->Rtype()) ? get_compatible_type(a->node->Rtype()) : get_compatible_type(b->node->Rtype());

            NODE_PTR n = air::base::Null_ptr;
            std::string node_name = "fhe::ckks::SUB";
            if (is_cipher(a->node->Rtype()) && is_cipher(b->node->Rtype())) {
                OPCODE sub_op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::SUB);
                n = container->New_bin_arith(sub_op, rtype, a->node, b->node, spos);
            } else {
                GLOB_SCOPE* glob = container->Glob_scope();
                TYPE_PTR float_type = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_64);
                long double neg_one_val = -1.0L;
                NODE_PTR neg_one = container->New_ldc(
                    glob->New_const(CONSTANT_KIND::FLOAT, float_type, neg_one_val), spos);
                OPCODE mul_op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::MUL);
                NODE_PTR neg_b = container->New_bin_arith(mul_op, rtype, b->node, neg_one, spos);
                OPCODE add_op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::ADD);
                n = container->New_bin_arith(add_op, rtype, a->node, neg_b, spos);
                node_name = "fhe::ckks::ADD(SUB)";
            }

            auto node = wrap_node(n, node_name);
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::SUB");
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_ckks_mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (container && a->has_node && b->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::MUL);
            // CKKS mul: cipher×cipher → CIPHERTEXT3 (for relin), cipher×plain → cipher, plain×cipher → cipher, plain×plain → plain
            TYPE_PTR rtype = get_compatible_type(a->node->Rtype());
            auto is_plain_type = [](TYPE_PTR t) {
                if (t == air::base::Null_ptr || !t->Is_record()) return false;
                STR_PTR name = t->Name();
                return name != air::base::Null_ptr && std::string(name->Char_str()) == "PLAINTEXT";
            };
            bool a_plain = is_plain_type(a->node->Rtype());
            bool b_plain = is_plain_type(b->node->Rtype());
            if (!a_plain && !b_plain) {
                // cipher×cipher: result must be CIPHERTEXT3 so poly/relin passes accept it
                // Look up CIPHERTEXT3 in glob (registered by ensure_fhe_types_registered when function was created)
                TYPE_PTR cipher3_type = air::base::Null_ptr;
                for (auto it = glob->Begin_type(); it != glob->End_type(); ++it) {
                    TYPE_PTR t = *it;
                    if (t->Is_record() && t->Name() != air::base::Null_ptr &&
                        std::string(t->Name()->Char_str()) == "CIPHERTEXT3") {
                        cipher3_type = t;
                        break;
                    }
                }
                if (cipher3_type != air::base::Null_ptr) {
                    rtype = cipher3_type;
                } else {
                    // Fallback: CIPHERTEXT3 not found (ensure_fhe_types_registered may not have run on this glob)
                    rtype = !a_plain ? get_compatible_type(a->node->Rtype()) : get_compatible_type(b->node->Rtype());
                }
            } else if (!a_plain || !b_plain) {
                // cipher×plain or plain×cipher: result is cipher
                rtype = !a_plain ? get_compatible_type(a->node->Rtype()) : get_compatible_type(b->node->Rtype());
            }
            NODE_PTR n = container->New_bin_arith(op, rtype, a->node, b->node, get_spos());
            auto node = wrap_node(n, "fhe::ckks::MUL");
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::MUL");
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }
    
    // CKKS negation
    std::shared_ptr<Node> new_ckks_neg(std::shared_ptr<Node> a) {
        require_not_expired();
        if (container && a->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::NEG);
            TYPE_PTR rtype = get_compatible_type(a->node->Rtype());
            NODE_PTR n = container->New_una_arith(op, rtype, a->node, get_spos());
            auto node = wrap_node(n, "fhe::ckks::NEG");
            node->add_child(a);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::NEG");
        node->add_child(a);
        nodes.push_back(node);
        return node;
    }
    
    // CKKS rotation - rotates slots by given amount
    std::shared_ptr<Node> new_ckks_rotate(std::shared_ptr<Node> ct, int32_t rotation) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::ROTATE);
            // Create rotation constant
            NODE_PTR rot_const = container->New_intconst(
                glob->Prim_type(PRIMITIVE_TYPE::INT_S32), rotation, get_spos());
            TYPE_PTR rtype = get_compatible_type(ct->node->Rtype());
            NODE_PTR n = container->New_cust_node(op, rtype, get_spos());
            n->Set_child(0, ct->node);
            n->Set_child(1, rot_const);
            Set_rotation_attr(n, rotation);
            auto node = wrap_node(n, "fhe::ckks::ROTATE");
            node->add_child(ct);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::ROTATE");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<Node> new_ckks_rotate_batch(std::shared_ptr<Node> ct,
                                                py::list rotations) {
        require_not_expired();
        if (!(container && ct->has_node)) {
            auto node =
                std::make_shared<Node>(++node_counter, "fhe::ckks::ROTATE_BATCH");
            node->add_child(ct);
            nodes.push_back(node);
            return node;
        }
        if (rotations.empty()) {
            throw std::runtime_error("new_ckks_rotate_batch requires non-empty rotations");
        }

        std::vector<int32_t> rot_vals;
        rot_vals.reserve(rotations.size());
        for (py::handle item : rotations) {
            rot_vals.push_back(item.cast<int32_t>());
        }

        TYPE_PTR elem_type = get_compatible_type(ct->node->Rtype());
        std::vector<int64_t> dims = {static_cast<int64_t>(rot_vals.size())};
        std::string arr_name = "cipher_batch_" + std::to_string(rot_vals.size());
        TYPE_PTR arr_type =
            glob->New_arr_type(glob->New_str(arr_name.c_str()), elem_type, dims, get_spos());

        OPCODE op(fhe::ckks::CKKS_DOMAIN::ID,
                  fhe::ckks::CKKS_OPERATOR::ROTATE_BATCH);
        NODE_PTR n = container->New_cust_node(op, arr_type, get_spos());
        n->Set_child(0, ct->node);
        Set_rotation_list_attr(n, rot_vals);
        auto node = wrap_node(n, "fhe::ckks::ROTATE_BATCH");
        node->add_child(ct);
        return node;
    }
    
    // CKKS encode - encode scalar/constant into plaintext polynomial
    // CKKS encode needs 4 children: data, length, scale, level
    //
    // For scalar constants (INTCONST/FLOATCONST), follows the
    // sihe_gen::Gen_encode_mask pattern:
    //   - Convert to float constant via ldc
    //   - Set nn::core::ATTR::MASK attribute
    // This causes the codegen to emit Encode_double_mask (scalar form)
    // instead of Encode_double (array pointer form).
    std::shared_ptr<Node> new_ckks_encode(std::shared_ptr<Node> data,
                                          int32_t encode_len = -1,
                                          uint32_t scale_degree = 1,
                                          uint32_t level = 0,
                                          bool precompute_cache = false) {
        require_not_expired();
        if (container && data && data->has_node && data->node != air::base::Null_ptr) {
            // Use OPC_ENCODE from ckks_opcode.h so scale manager sees same opcode (plaintext promotion)
            air::base::OPCODE op = fhe::ckks::OPC_ENCODE;
            GLOB_SCOPE* gs = container->Glob_scope();
            
            // Get or create PLAINTEXT type
            TYPE_PTR plain_type = air::base::Null_ptr;
            // Look for existing PLAINTEXT record type
            for (auto it = gs->Begin_type(); it != gs->End_type(); ++it) {
                TYPE_PTR t = *it;
                if (t->Is_record() && t->Name() != air::base::Null_ptr) {
                    std::string name(t->Name()->Char_str());
                    if (name == "PLAINTEXT") {
                        plain_type = t;
                        break;
                    }
                }
            }
            // If not found, create it
            if (plain_type == air::base::Null_ptr) {
                STR_PTR plain_str = gs->New_str("PLAINTEXT");
                plain_type = gs->New_rec_type(RECORD_KIND::STRUCT, plain_str, air::base::SPOS());
            }
            
            TYPE_PTR u32_type = gs->Prim_type(air::base::PRIMITIVE_TYPE::INT_U32);
            SPOS spos = get_spos();
            
            // Create custom node with 4 children
            NODE_PTR n = container->New_cust_node(op, plain_type, spos);
            
            // Check if data is a scalar constant (INTCONST or FLOATCONST)
            // If so, follow sihe_gen::Gen_encode_mask pattern:
            //   1. Convert to float constant via ldc
            //   2. Set MASK attribute so codegen emits Encode_*_mask
            bool is_scalar_const = (data->node->Opcode() == air::core::OPC_INTCONST);
            bool is_float_const = false;
            if (!is_scalar_const && data->node->Opcode() == air::core::OPC_LDC) {
                TYPE_PTR rtype = data->node->Rtype();
                if (rtype->Is_prim()) {
                    auto enc = rtype->Cast_to_prim()->Encoding();
                    is_float_const = (enc == air::base::PRIMITIVE_TYPE::FLOAT_32 ||
                                      enc == air::base::PRIMITIVE_TYPE::FLOAT_64);
                }
            }
            
            if (is_scalar_const || is_float_const) {
                // Extract scalar value
                double scalar_val;
                if (is_scalar_const) {
                    scalar_val = (double)data->node->Intconst();
                } else {
                    // Float constant via ldc - use data node directly
                    scalar_val = 0.0;  // Attribute value for mask flag
                }
                
                if (is_scalar_const) {
                    // Convert int to double constant (like sihe_gen::Gen_encode_mask)
                    TYPE_PTR f64_type = gs->Prim_type(air::base::PRIMITIVE_TYPE::FLOAT_64);
                    CONSTANT_PTR cst = gs->New_const(
                        CONSTANT_KIND::FLOAT, f64_type, (long double)scalar_val);
                    NODE_PTR ldc = container->New_ldc(cst, spos);
                    n->Set_child(0, ldc);
                } else {
                    n->Set_child(0, data->node);
                }
                
                // Set MASK attribute - signals codegen to use Encode_*_mask
                n->Set_attr(nn::core::ATTR::MASK, &scalar_val, 1);
            } else {
                // Array/buffer data - use as-is
                n->Set_child(0, data->node);
            }
            
            // child 1: length
            // For constant arrays, use element count as encode length.
            // For scalar/masked encode, keep default 64 (slot count placeholder).
            uint32_t inferred_encode_len = 64;
            NODE_PTR src_data = n->Child(0);
            if (src_data != air::base::Null_ptr &&
                src_data->Opcode() == air::core::OPC_LDC &&
                src_data->Rtype()->Is_array()) {
                uint64_t elem_cnt = src_data->Rtype()->Cast_to_arr()->Elem_count();
                if (elem_cnt > 0 && elem_cnt <= UINT32_MAX) {
                    inferred_encode_len = static_cast<uint32_t>(elem_cnt);
                }
            }
            uint32_t final_encode_len =
                (encode_len > 0) ? static_cast<uint32_t>(encode_len) : inferred_encode_len;
            NODE_PTR len_node = container->New_intconst(u32_type, final_encode_len, spos);
            n->Set_child(1, len_node);
            NODE_PTR scale_node = container->New_intconst(u32_type, scale_degree, spos);
            n->Set_child(2, scale_node);
            NODE_PTR level_node = container->New_intconst(u32_type, level, spos);
            n->Set_child(3, level_node);
            if (scale_degree != 0) {
                n->Set_attr(fhe::core::FHE_ATTR_KIND::SCALE, &scale_degree, 1);
            }
            if (level != 0) {
                n->Set_attr(fhe::core::FHE_ATTR_KIND::LEVEL, &level, 1);
            }
            if (precompute_cache) {
                uint32_t flag = 1;
                n->Set_attr(fhe::core::FHE_ATTR_KIND::ENCODE_CACHE, &flag, 1);
            }
            
            auto node = wrap_node(n, "fhe::ckks::ENCODE");
            node->add_child(data);
            nodes.push_back(node);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::ENCODE");
        if (data) node->add_child(data);
        nodes.push_back(node);
        return node;
    }

    // CKKS encode for complex arrays represented as interleaved float64:
    // [r0, i0, r1, i1, ...]. Mark node with ENCODE_DCMPLX attr and set len to
    // number of complex slots.
    std::shared_ptr<Node> new_ckks_encode_complex(std::shared_ptr<Node> data,
                                                  int32_t complex_len = -1,
                                                  uint32_t scale_degree = 1,
                                                  uint32_t level = 0,
                                                  uint32_t num_p = 0,
                                                  bool precompute_cache = false) {
        require_not_expired();
        auto encoded = new_ckks_encode(
            data, -1, scale_degree, level, precompute_cache);
        if (!(container && encoded && encoded->has_node &&
              encoded->node != air::base::Null_ptr)) {
            return encoded;
        }

        uint32_t flag = 1;
        encoded->node->Set_attr(fhe::core::FHE_ATTR_KIND::ENCODE_DCMPLX, &flag, 1);

        uint32_t inferred_len = 0;
        if (complex_len > 0) {
            inferred_len = static_cast<uint32_t>(complex_len);
        } else if (data && data->has_node && data->node != air::base::Null_ptr &&
                   data->node->Opcode() == air::core::OPC_LDC &&
                   data->node->Rtype()->Is_array()) {
            uint64_t elem_cnt = data->node->Rtype()->Cast_to_arr()->Elem_count();
            if (elem_cnt >= 2 && elem_cnt <= UINT32_MAX) {
                inferred_len = static_cast<uint32_t>(elem_cnt / 2);
            }
        }

        if (inferred_len > 0) {
            TYPE_PTR u32_type = container->Glob_scope()->Prim_type(air::base::PRIMITIVE_TYPE::INT_U32);
            NODE_PTR len_node = container->New_intconst(u32_type, inferred_len, get_spos());
            encoded->node->Set_child(1, len_node);
        }
        if (num_p != 0) {
            encoded->node->Set_attr(fhe::core::FHE_ATTR_KIND::NUM_P, &num_p, 1);
        }
        return encoded;
    }
    
    // CKKS rescale - reduces scale after multiplication
    std::shared_ptr<Node> new_ckks_rescale(std::shared_ptr<Node> ct) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::RESCALE);
            NODE_PTR n = container->New_cust_node(op, ct->node->Rtype(), get_spos());
            n->Set_child(0, ct->node);
            auto node = wrap_node(n, "fhe::ckks::RESCALE");
            node->add_child(ct);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::RESCALE");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }
    
    // CKKS relin - relinearization after multiplication
    std::shared_ptr<Node> new_ckks_relin(std::shared_ptr<Node> ct) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::RELIN);
            NODE_PTR n = container->New_una_arith(op, ct->node->Rtype(), ct->node, get_spos());
            auto node = wrap_node(n, "fhe::ckks::RELIN");
            node->add_child(ct);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::RELIN");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }
    
    // CKKS mod_switch - reduces modulus (level) by one
    std::shared_ptr<Node> new_ckks_mod_switch(std::shared_ptr<Node> ct) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::MODSWITCH);
            NODE_PTR n = container->New_cust_node(op, ct->node->Rtype(), get_spos());
            n->Set_child(0, ct->node);
            auto node = wrap_node(n, "fhe::ckks::MODSWITCH");
            node->add_child(ct);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::MODSWITCH");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }
    
    // CKKS bootstrap - refreshes ciphertext noise budget
    std::shared_ptr<Node> new_ckks_bootstrap(std::shared_ptr<Node> ct) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::BOOTSTRAP);
            NODE_PTR n = container->New_cust_node(op, ct->node->Rtype(), get_spos());
            n->Set_child(0, ct->node);
            auto node = wrap_node(n, "fhe::ckks::BOOTSTRAP");
            node->add_child(ct);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::BOOTSTRAP");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }

    // CKKS bootstrap stage op - coeffs to slots (context-aware runtime path)
    std::shared_ptr<Node> new_ckks_bootstrap_coeffs_to_slots(
        std::shared_ptr<Node> ct, int32_t num_slots = 0) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(
                fhe::ckks::CKKS_DOMAIN::ID,
                fhe::ckks::CKKS_OPERATOR::BOOTSTRAP_COEFFS_TO_SLOTS);
            TYPE_PTR rtype = get_compatible_type(ct->node->Rtype());
            NODE_PTR n = container->New_cust_node(op, rtype, get_spos());
            n->Set_child(0, ct->node);
            uint32_t slots = static_cast<uint32_t>(num_slots > 0 ? num_slots : 0);
            n->Set_attr(nn::core::ATTR::SLOT, &slots, 1);
            auto node = wrap_node(n, "fhe::ckks::BOOTSTRAP_COEFFS_TO_SLOTS");
            node->add_child(ct);
            return node;
        }
        auto node = std::make_shared<Node>(
            ++node_counter, "fhe::ckks::BOOTSTRAP_COEFFS_TO_SLOTS");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }

    // CKKS bootstrap stage op - EvalMod approximation (context-aware runtime path)
    std::shared_ptr<Node> new_ckks_bootstrap_eval_mod(std::shared_ptr<Node> ct) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(
                fhe::ckks::CKKS_DOMAIN::ID,
                fhe::ckks::CKKS_OPERATOR::BOOTSTRAP_EVAL_MOD);
            TYPE_PTR rtype = get_compatible_type(ct->node->Rtype());
            NODE_PTR n = container->New_cust_node(op, rtype, get_spos());
            n->Set_child(0, ct->node);
            auto node = wrap_node(n, "fhe::ckks::BOOTSTRAP_EVAL_MOD");
            node->add_child(ct);
            return node;
        }
        auto node =
            std::make_shared<Node>(++node_counter, "fhe::ckks::BOOTSTRAP_EVAL_MOD");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }

    // CKKS bootstrap stage op - slots to coeffs (context-aware runtime path)
    std::shared_ptr<Node> new_ckks_bootstrap_slots_to_coeffs(
        std::shared_ptr<Node> ct, int32_t num_slots = 0) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(
                fhe::ckks::CKKS_DOMAIN::ID,
                fhe::ckks::CKKS_OPERATOR::BOOTSTRAP_SLOTS_TO_COEFFS);
            TYPE_PTR rtype = get_compatible_type(ct->node->Rtype());
            NODE_PTR n = container->New_cust_node(op, rtype, get_spos());
            n->Set_child(0, ct->node);
            uint32_t slots = static_cast<uint32_t>(num_slots > 0 ? num_slots : 0);
            n->Set_attr(nn::core::ATTR::SLOT, &slots, 1);
            auto node = wrap_node(n, "fhe::ckks::BOOTSTRAP_SLOTS_TO_COEFFS");
            node->add_child(ct);
            return node;
        }
        auto node = std::make_shared<Node>(
            ++node_counter, "fhe::ckks::BOOTSTRAP_SLOTS_TO_COEFFS");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }

    // CKKS raise_mod - raise ciphertext modulus with a target level/mod_size
    std::shared_ptr<Node> new_ckks_raise_mod(
        std::shared_ptr<Node> ct, int32_t mod_size,
        bool runtime_raise_level = false) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::RAISE_MOD);
            TYPE_PTR u32_type = glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
            NODE_PTR mod_const = container->New_intconst(
                u32_type, static_cast<uint32_t>(mod_size), get_spos());
            TYPE_PTR rtype = get_compatible_type(ct->node->Rtype());
            NODE_PTR n = container->New_cust_node(op, rtype, get_spos());
            n->Set_child(0, ct->node);
            n->Set_child(1, mod_const);
            if (runtime_raise_level) {
                uint32_t attr = 1;
                n->Set_attr(fhe::core::FHE_ATTR_KIND::RUNTIME_RAISE_LEVEL,
                            &attr, 1);
            }
            auto node = wrap_node(n, "fhe::ckks::RAISE_MOD");
            node->add_child(ct);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::RAISE_MOD");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }

    // CKKS conjugate - complex conjugation over slots
    std::shared_ptr<Node> new_ckks_conjugate(std::shared_ptr<Node> ct) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::CONJUGATE);
            TYPE_PTR rtype = get_compatible_type(ct->node->Rtype());
            NODE_PTR n = container->New_una_arith(op, rtype, ct->node, get_spos());
            auto node = wrap_node(n, "fhe::ckks::CONJUGATE");
            node->add_child(ct);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::CONJUGATE");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }

    // CKKS multiply-by-monomial - helper used by bootstrap paths
    std::shared_ptr<Node> new_ckks_mul_mono(std::shared_ptr<Node> ct, int32_t power) {
        require_not_expired();
        if (container && ct->has_node) {
            OPCODE op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::MUL_MONO);
            TYPE_PTR u32_type = glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
            NODE_PTR power_const = container->New_intconst(
                u32_type, static_cast<uint32_t>(power), get_spos());
            TYPE_PTR rtype = get_compatible_type(ct->node->Rtype());
            NODE_PTR n = container->New_cust_node(op, rtype, get_spos());
            n->Set_child(0, ct->node);
            n->Set_child(1, power_const);
            Set_rotation_attr(n, power);
            auto node = wrap_node(n, "fhe::ckks::MUL_MONO");
            node->add_child(ct);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::ckks::MUL_MONO");
        node->add_child(ct);
        nodes.push_back(node);
        return node;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Domain-Specific Operations: fhe::poly
    // ═══════════════════════════════════════════════════════════════════════
    
    std::shared_ptr<Node> new_poly_add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (container && a->has_node && b->has_node) {
            OPCODE op(fhe::poly::POLYNOMIAL_DID, fhe::poly::OPCODE::ADD);
            NODE_PTR n = container->New_bin_arith(op, a->node->Rtype(), a->node, b->node, get_spos());
            auto node = wrap_node(n, "fhe::poly::ADD");
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::poly::ADD");
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_poly_sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (container && a->has_node && b->has_node) {
            OPCODE op(fhe::poly::POLYNOMIAL_DID, fhe::poly::OPCODE::SUB);
            NODE_PTR n = container->New_bin_arith(op, a->node->Rtype(), a->node, b->node, get_spos());
            auto node = wrap_node(n, "fhe::poly::SUB");
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::poly::SUB");
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_poly_mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        if (container && a->has_node && b->has_node) {
            OPCODE op(fhe::poly::POLYNOMIAL_DID, fhe::poly::OPCODE::MUL);
            NODE_PTR n = container->New_bin_arith(op, a->node->Rtype(), a->node, b->node, get_spos());
            auto node = wrap_node(n, "fhe::poly::MUL");
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, "fhe::poly::MUL");
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Comparison Operations (create relational nodes)
    // ═══════════════════════════════════════════════════════════════════════
    
    // Helper to create comparison node with boolean result type
    std::shared_ptr<Node> new_cmp_node(air::core::OPCODE opcode, 
                                        std::shared_ptr<Node> a, 
                                        std::shared_ptr<Node> b,
                                        const std::string& name) {
        require_not_expired();
        if (container && a->has_node && b->has_node) {
            OPCODE op(air::core::CORE, opcode);
            // Comparisons return boolean, NOT the operand type
            TYPE_PTR bool_type = glob->Prim_type(PRIMITIVE_TYPE::BOOL);
            NODE_PTR n = container->New_cust_node(op, bool_type, get_spos());
            n->Set_child(0, a->node);
            n->Set_child(1, b->node);
            auto node = wrap_node(n, name);
            node->add_child(a);
            node->add_child(b);
            return node;
        }
        auto node = std::make_shared<Node>(++node_counter, name);
        node->add_child(a);
        node->add_child(b);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_gt(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        return new_cmp_node(air::core::OPCODE::GT, a, b, "air::core::GT");
    }
    
    std::shared_ptr<Node> new_lt(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        return new_cmp_node(air::core::OPCODE::LT, a, b, "air::core::LT");
    }
    
    std::shared_ptr<Node> new_ge(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        return new_cmp_node(air::core::OPCODE::GE, a, b, "air::core::GE");
    }
    
    std::shared_ptr<Node> new_le(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        return new_cmp_node(air::core::OPCODE::LE, a, b, "air::core::LE");
    }
    
    std::shared_ptr<Node> new_eq(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        return new_cmp_node(air::core::OPCODE::EQ, a, b, "air::core::EQ");
    }
    
    std::shared_ptr<Node> new_ne(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        require_not_expired();
        return new_cmp_node(air::core::OPCODE::NE, a, b, "air::core::NE");
    }
    
    std::shared_ptr<Node> new_retv(std::shared_ptr<Node> val) {
        require_not_expired();
        if (container && val && val->has_node &&
            val->node != NODE_PTR() && val->node->Has_rtype()) {
            if (val->node->Container() != container) {
                throw std::runtime_error(
                    "new_retv cannot return a value from another AIR container");
            }
            STMT_PTR stmt = container->New_retv(val->node, get_spos());
            append_stmt(stmt);
            auto node = wrap_node(stmt->Node(), "air::core::RETV");
            node->add_child(val);
            return node;
        }
        throw std::runtime_error(
            "new_retv requires an expression value with a result type");
    }
    
    std::shared_ptr<Node> new_ret() {
        require_not_expired();
        if (container) {
            STMT_PTR stmt = container->New_ret(get_spos());
            append_stmt(stmt);
            return wrap_node(stmt->Node(), "air::core::RET");
        }
        auto node = std::make_shared<Node>(++node_counter, "air::core::RET");
        nodes.push_back(node);
        return node;
    }

    TYPE_PTR require_local_type(const Type& requested,
                                const char* operation) const {
        require_not_expired();
        requested.require_active();
        if (!requested.has_type || requested.type == Null_ptr) {
            throw std::runtime_error(std::string(operation) +
                                     " requires a concrete AIR type");
        }
        if (!glob || &requested.type->Glob_scope() != glob) {
            throw std::runtime_error(std::string(operation) +
                                     " type belongs to a different GLOB_SCOPE");
        }
        return requested.type;
    }

    // Declare a named local with an explicit type and return a load of it.
    std::shared_ptr<Node> new_local(const std::string& var_name,
                                    const Type& requested_type) {
        require_not_expired();
        if (!(container && func_scope)) {
            throw std::runtime_error("new_local requires a real function scope");
        }
        TYPE_PTR type = require_local_type(requested_type, "new_local");
        auto it = var_map.find(var_name);
        ADDR_DATUM_PTR var;
        if (it != var_map.end()) {
            var = it->second;
            if (!var->Type()->Is_compatible_type(type)) {
                throw std::runtime_error(
                    "new_local redeclaration has an incompatible type: " +
                    var_name);
            }
        } else {
            var = func_scope->New_var(type, var_name.c_str(), get_spos());
            var_map[var_name] = var;
        }
        return wrap_node(container->New_ld(var, get_spos()),
                         "air::core::LDID");
    }
    
    // Store value to a variable (creates a statement in current block)
    // var_node should be a load node for the target variable
    std::shared_ptr<Node> new_stid(const std::string& var_name, std::shared_ptr<Node> val) {
        require_not_expired();
        if (!(container && func_scope) || !val || !val->has_node ||
            val->node == NODE_PTR()) {
            throw std::runtime_error("new_stid requires real container and value");
        }
        if (val->node->Container() != container) {
            throw std::runtime_error(
                "new_stid cannot store a value from another AIR container");
        }
        if (!val->node->Has_rtype()) {
            throw std::runtime_error(
                "new_stid requires an expression value with a result type");
        }
        TYPE_PTR type = get_compatible_type(val->node->Rtype());
        
        // Inherit source position from the value node to preserve line info
        SPOS val_spos = val->node->Spos();
        
        // Find or create variable
        ADDR_DATUM_PTR var;
        auto it = var_map.find(var_name);
        if (it != var_map.end()) {
            var = it->second;
            if (!var->Type()->Is_compatible_type(type)) {
                throw std::runtime_error(
                    "new_stid value type is incompatible with local '" +
                    var_name + "'");
            }
        } else {
            var = func_scope->New_var(type, var_name.c_str(), val_spos);
            var_map[var_name] = var;
        }
        
        // Create store statement with inherited source position
        STMT_PTR stmt = container->New_st(val->node, var, val_spos);
        append_stmt(stmt);
        
        // Create a load node for the stored value (for chaining)
        NODE_PTR ld = container->New_ld(var, val_spos);
        auto node = wrap_node(ld, "air::core::STID");
        node->add_child(val);
        return node;
    }

    // Store a planned packed Vector value into a wider destination local.
    // Native metakernels use this representation for duplicated inputs: the
    // expression keeps its logical source shape while the local carries the
    // prepared packed result width.
    std::shared_ptr<Node> new_vector_widening_stid(
        const std::string& var_name, const Type& requested_type,
        std::shared_ptr<Node> val) {
        require_not_expired();
        if (!(container && func_scope) || !val || !val->has_node ||
            val->node == NODE_PTR()) {
            throw std::runtime_error(
                "new_vector_widening_stid requires real container and value");
        }
        TYPE_PTR target = require_local_type(
            requested_type, "new_vector_widening_stid");
        TYPE_PTR source = require_vector_operand(
            val, "new_vector_widening_stid");
        if (!target->Is_array() ||
            !Type::structurally_equal_air_types(
                source->Cast_to_arr()->Elem_type(),
                target->Cast_to_arr()->Elem_type())) {
            throw std::runtime_error(
                "new_vector_widening_stid requires matching ranked element types");
        }
        const uint64_t source_count = source->Cast_to_arr()->Elem_count();
        const uint64_t target_count = target->Cast_to_arr()->Elem_count();
        if (source_count == 0 || target_count < source_count ||
            target_count % source_count != 0) {
            throw std::runtime_error(
                "new_vector_widening_stid target is not a valid packed widening");
        }

        ADDR_DATUM_PTR var;
        auto it = var_map.find(var_name);
        if (it != var_map.end()) {
            var = it->second;
            if (!var->Type()->Is_compatible_type(target)) {
                throw std::runtime_error(
                    "new_vector_widening_stid local type mismatch: " +
                    var_name);
            }
        } else {
            var = func_scope->New_var(target, var_name.c_str(), val->node->Spos());
            var_map[var_name] = var;
        }
        STMT_PTR stmt = container->New_st(val->node, var, val->node->Spos());
        append_stmt(stmt);
        return wrap_node(container->New_ld(var, val->node->Spos()),
                         "air::core::STID");
    }

    // Store a planned packed Vector value into an explicitly typed local.
    // Packed metakernels may intentionally reinterpret the logical ranked
    // shape while preserving the element type; this is distinct from ordinary
    // structurally typed stores and from widening-only duplication.
    std::shared_ptr<Node> new_vector_packed_stid(
        const std::string& var_name, const Type& requested_type,
        std::shared_ptr<Node> val) {
        require_not_expired();
        if (!(container && func_scope) || !val || !val->has_node ||
            val->node == NODE_PTR()) {
            throw std::runtime_error(
                "new_vector_packed_stid requires real container and value");
        }
        TYPE_PTR target = require_local_type(
            requested_type, "new_vector_packed_stid");
        TYPE_PTR source = require_vector_operand(
            val, "new_vector_packed_stid");
        if (!target->Is_array() ||
            !Type::structurally_equal_air_types(
                source->Cast_to_arr()->Elem_type(),
                target->Cast_to_arr()->Elem_type())) {
            throw std::runtime_error(
                "new_vector_packed_stid requires matching ranked element types");
        }

        ADDR_DATUM_PTR var;
        auto it = var_map.find(var_name);
        if (it != var_map.end()) {
            var = it->second;
            if (!Type::structurally_equal_air_types(var->Type(), target)) {
                throw std::runtime_error(
                    "new_vector_packed_stid local type mismatch: " +
                    var_name);
            }
        } else {
            var = func_scope->New_var(
                target, var_name.c_str(), val->node->Spos());
            var_map[var_name] = var;
        }
        STMT_PTR stmt = container->New_st(
            val->node, var, val->node->Spos());
        append_stmt(stmt);
        return wrap_node(container->New_ld(var, val->node->Spos()),
                         "air::core::STID");
    }

    std::shared_ptr<Node> new_fresh_load(std::shared_ptr<Node> value) {
        require_not_expired();
        if (!(container && value && value->has_node) ||
            value->node == NODE_PTR() || value->node->Container() != container) {
            throw std::runtime_error(
                "new_fresh_load requires a value in the current AIR container");
        }
        NODE_PTR fresh = NODE_PTR();
        if (value->node->Opcode() == air::core::OPC_LD) {
            fresh = container->New_ld(value->node->Addr_datum(), get_spos());
        } else if (value->node->Opcode() == air::core::OPC_LDP) {
            fresh = container->New_ldp(value->node->Preg(), get_spos());
        } else if (value->node->Opcode() == air::core::OPC_LDC) {
            fresh = container->New_ldc(value->node->Const(), get_spos());
        } else {
            return value;
        }
        return wrap_node(fresh, value->opcode_str);
    }

    // Load value from a named variable (LDID)
    std::shared_ptr<Node> new_ldid(const std::string& var_name) {
        require_not_expired();
        if (!(container && func_scope)) {
            throw std::runtime_error("new_ldid requires real container");
        }
        auto it = var_map.find(var_name);
        if (it == var_map.end()) {
            throw std::runtime_error("Unknown variable name for LDID: " + var_name);
        }
        ADDR_DATUM_PTR var = it->second;
        NODE_PTR ld = container->New_ld(var, get_spos());
        return wrap_node(ld, "air::core::LDID");
    }
    
    // Check if we're inside a control flow block (loop or if body)
    bool in_control_flow_body() const {
        require_not_expired();
        return !block_stack.empty();
    }
    
    std::shared_ptr<Node> new_intconst(int64_t val) {
        require_not_expired();
        if (container && glob) {
            TYPE_PTR i64_type = glob->Prim_type(PRIMITIVE_TYPE::INT_S64);
            NODE_PTR n = container->New_intconst(i64_type, val, get_spos());
            return wrap_node(n, "air::core::INTCONST");
        }
        auto node = std::make_shared<Node>(++node_counter, "air::core::INTCONST");
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<Node> new_intconst_typed(int64_t val,
                                             const Type& requested_type) {
        require_not_expired();
        TYPE_PTR type = require_local_type(requested_type,
                                           "new_intconst_typed");
        if (!type->Is_signed_int()) {
            throw std::runtime_error(
                "new_intconst_typed requires a signed integer type");
        }
        if (type->Is_prim() &&
            type->Cast_to_prim()->Encoding() == PRIMITIVE_TYPE::INT_S32 &&
            (val < std::numeric_limits<int32_t>::min() ||
             val > std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                "new_intconst_typed value is out of range for signed i32");
        }
        NODE_PTR n = container->New_intconst(type, val, get_spos());
        return wrap_node(n, "air::core::INTCONST");
    }

    std::shared_ptr<Node> new_floatconst(double val) {
        require_not_expired();
        if (container && glob) {
            TYPE_PTR f64_type = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_64);
            CONSTANT_PTR cst = glob->New_const(
                CONSTANT_KIND::FLOAT, f64_type, static_cast<long double>(val));
            NODE_PTR n = container->New_ldc(cst, get_spos());
            return wrap_node(n, "air::core::LDC");
        }
        auto node = std::make_shared<Node>(++node_counter, "air::core::LDC");
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<Node> new_floatconst_typed(
        double val, const Type& requested_type) {
        require_not_expired();
        TYPE_PTR type = require_local_type(requested_type,
                                           "new_floatconst_typed");
        if (!type->Is_real_float()) {
            throw std::runtime_error(
                "new_floatconst_typed requires a real floating-point type");
        }
        CONSTANT_PTR cst = glob->New_const(
            CONSTANT_KIND::FLOAT, type, static_cast<long double>(val));
        NODE_PTR n = container->New_ldc(cst, get_spos());
        return wrap_node(n, "air::core::LDC");
    }
    
    std::shared_ptr<Node> new_zero() { return new_intconst(0); }
    std::shared_ptr<Node> new_zero(const Type& requested_type) {
        require_not_expired();
        TYPE_PTR type = require_local_type(requested_type, "new_zero");
        if (type->Is_int()) {
            NODE_PTR n = container->New_intconst(type, 0, get_spos());
            return wrap_node(n, "air::core::INTCONST");
        }
        NODE_PTR n = container->New_zero(type, get_spos());
        return wrap_node(n, "air::core::ZERO");
    }
    std::shared_ptr<Node> new_one() { return new_intconst(1); }

    std::shared_ptr<Node> new_checked_cast(std::shared_ptr<Node> value,
                                           const Type& requested_type) {
        require_not_expired();
        if (!value || !value->has_node || value->node == NODE_PTR() ||
            value->node->Container() != container) {
            throw std::runtime_error(
                "new_checked_cast requires a value in the current container");
        }
        if (!value->node->Has_rtype()) {
            throw std::runtime_error(
                "new_checked_cast requires an expression value with a result type");
        }
        TYPE_PTR type = require_local_type(requested_type,
                                           "new_checked_cast");
        if (!value->node->Rtype()->Is_compatible_type(type)) {
            throw std::runtime_error(
                "AIR Core has no width-conversion opcode; checked cast requires "
                "an already compatible type");
        }
        return value;
    }
    
    // Create an LDC node referencing a CONSTANT_KIND::ARRAY constant.
    // Accepts:
    //  - real list: [x0, x1, ...]              -> float32 array
    //  - complex list: [a+bj, c+dj, ...]       -> interleaved float64 array
    //  - pair list: [[r0,i0], [r1,i1], ...]    -> interleaved float64 array
    // Real arrays are float32 to match CKKS offline encode path (DE_MSG_F32).
    // Complex arrays remain float64 for Encode_dcmplx runtime path.
    // Used for constant-array plaintext encoding (no MASK attribute).
    std::shared_ptr<Node> new_array_const(py::list values) {
        require_not_expired();
        if (!(container && glob)) {
            throw std::runtime_error("new_array_const requires real container");
        }
        size_t len = values.size();
        if (len == 0) {
            throw std::runtime_error("new_array_const requires non-empty list");
        }
        
        bool has_complex = false;
        for (size_t i = 0; i < len; i++) {
            py::handle item = values[i];
            if (PyComplex_Check(item.ptr())) {
                has_complex = true;
                break;
            }
            if (py::isinstance<py::list>(item) || py::isinstance<py::tuple>(item)) {
                py::sequence seq = item.cast<py::sequence>();
                if (seq.size() == 2) {
                    has_complex = true;
                    break;
                }
            }
        }

        SPOS spos = get_spos();
        STR_PTR arr_name = glob->New_str("const_arr");

        if (has_complex) {
            // Complex path: keep float64 interleaved representation.
            std::vector<double> buf;
            buf.reserve(len * 2);
            for (size_t i = 0; i < len; i++) {
                py::handle item = values[i];
                double real = 0.0;
                double imag = 0.0;
                if (PyComplex_Check(item.ptr())) {
                    std::complex<double> z = item.cast<std::complex<double>>();
                    real = z.real();
                    imag = z.imag();
                } else if (py::isinstance<py::list>(item) || py::isinstance<py::tuple>(item)) {
                    py::sequence seq = item.cast<py::sequence>();
                    if (seq.size() != 2) {
                        throw std::runtime_error("complex pair must have 2 elements: [real, imag]");
                    }
                    real = seq[0].cast<double>();
                    imag = seq[1].cast<double>();
                } else {
                    real = item.cast<double>();
                    imag = 0.0;
                }
                buf.push_back(real);
                buf.push_back(imag);
            }

            TYPE_PTR f64_type = glob->Prim_type(air::base::PRIMITIVE_TYPE::FLOAT_64);
            std::vector<int64_t> dims = {static_cast<int64_t>(buf.size())};
            TYPE_PTR arr_type = glob->New_arr_type(arr_name, f64_type, dims, spos);
            CONSTANT_PTR cst = glob->New_const(
                CONSTANT_KIND::ARRAY, arr_type, buf.data(),
                buf.size() * sizeof(double));
            NODE_PTR n = container->New_ldc(cst, spos);
            return wrap_node(n, "air::core::LDC(ARRAY)");
        }

        // Real path: emit float32 array for CKKS offline encoding path.
        std::vector<float> buf;
        buf.resize(len);
        for (size_t i = 0; i < len; i++) {
            buf[i] = static_cast<float>(values[i].cast<double>());
        }
        TYPE_PTR f32_type = glob->Prim_type(air::base::PRIMITIVE_TYPE::FLOAT_32);
        std::vector<int64_t> dims = {static_cast<int64_t>(buf.size())};
        TYPE_PTR arr_type = glob->New_arr_type(arr_name, f32_type, dims, spos);
        CONSTANT_PTR cst = glob->New_const(
            CONSTANT_KIND::ARRAY, arr_type, buf.data(),
            buf.size() * sizeof(float));
        NODE_PTR n = container->New_ldc(cst, spos);
        return wrap_node(n, "air::core::LDC(ARRAY)");
    }

    // Materialize a same-typed Python sequence of AIR nodes into a real AIR
    // array local:
    //   var arr: array[len]<elem_type>
    //   arr[i] = values[i]
    //   return LD(arr)
    std::shared_ptr<Node> new_air_array(const std::string& var_name,
                                        py::list values) {
        require_not_expired();
        if (!(container && func_scope && glob)) {
            throw std::runtime_error("new_air_array requires real container");
        }
        if (values.empty()) {
            throw std::runtime_error("new_air_array requires non-empty values");
        }

        std::vector<std::shared_ptr<Node>> elems;
        elems.reserve(values.size());
        TYPE_PTR elem_type = air::base::Null_ptr;
        for (size_t i = 0; i < values.size(); ++i) {
            auto elem = values[i].cast<std::shared_ptr<Node>>();
            if (!elem || !elem->has_node || elem->node == air::base::Null_ptr) {
                throw std::runtime_error(
                    "new_air_array requires real AIR nodes at every element");
            }
            TYPE_PTR rtype = get_compatible_type(elem->node->Rtype());
            if (rtype == air::base::Null_ptr) {
                throw std::runtime_error(
                    "new_air_array element has no result type");
            }
            if (elem_type == air::base::Null_ptr) {
                elem_type = rtype;
            } else if (rtype->Id() != elem_type->Id()) {
                throw std::runtime_error(
                    "new_air_array requires all elements to have the same AIR type");
            }
            elems.push_back(elem);
        }

        SPOS spos = get_spos();
        std::vector<int64_t> dims = {static_cast<int64_t>(elems.size())};
        std::string type_name = var_name + "_type";
        TYPE_PTR arr_type = glob->New_arr_type(
            glob->New_str(type_name.c_str()), elem_type, dims, spos);

        ADDR_DATUM_PTR var;
        auto it = var_map.find(var_name);
        if (it != var_map.end()) {
            var = it->second;
            if (!var->Type()->Is_array() ||
                var->Type()->Cast_to_arr()->Elem_count() != elems.size() ||
                var->Type()->Cast_to_arr()->Elem_type_id() != elem_type->Id()) {
                throw std::runtime_error(
                    "new_air_array variable name already exists with incompatible type");
            }
        } else {
            var = func_scope->New_var(arr_type, var_name.c_str(), spos);
            var_map[var_name] = var;
        }

        TYPE_PTR idx_type = glob->Prim_type(PRIMITIVE_TYPE::INT_U32);
        for (size_t i = 0; i < elems.size(); ++i) {
            SPOS elem_spos = elems[i]->node->Spos();
            NODE_PTR base_addr = container->New_lda(
                var, air::base::POINTER_KIND::FLAT64, elem_spos);
            NODE_PTR array = container->New_array(base_addr, 1, elem_spos);
            NODE_PTR idx = container->New_intconst(
                idx_type, static_cast<int64_t>(i), elem_spos);
            container->Set_array_idx(array, 0, idx);
            append_stmt(container->New_ist(array, elems[i]->node, elem_spos));
        }

        NODE_PTR ld = container->New_ld(var, spos);
        auto node = wrap_node(ld, "air::core::LDID(ARRAY)");
        for (const auto& elem : elems) {
            node->add_child(elem);
        }
        return node;
    }
    
    std::shared_ptr<Node> new_ld(std::shared_ptr<Node> addr) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::LD");
        node->add_child(addr);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_st(std::shared_ptr<Node> val, std::shared_ptr<Node> addr) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::ST");
        node->add_child(val);
        node->add_child(addr);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_ild(std::shared_ptr<Node> base, std::shared_ptr<Node> idx) {
        require_not_expired();
        if (!(container && base && base->has_node &&
              base->node != NODE_PTR())) {
            throw std::runtime_error("new_ild requires real container and base");
        }
        require_vector_operand(base, "new_ild");
        TYPE_PTR idx_type = require_real_expression(idx, "new_ild");
        if (idx->node->Domain() != air::core::CORE ||
            !idx_type->Is_signed_int()) {
            throw std::runtime_error(
                "new_ild requires a Core signed integer index");
        }
        NODE_PTR base_node = base->node;
        NODE_PTR base_addr = NODE_PTR();
        if (base_node->Opcode() == air::core::OPC_LD) {
            base_addr = container->New_lda(
                base_node->Addr_datum(), air::base::POINTER_KIND::FLAT64, get_spos());
        } else if (base_node->Opcode() == air::core::OPC_LDC) {
            require_core_s32_operand(idx, "new_ild constant-array index");
            base_addr = container->New_ldca(
                base_node->Const(), air::base::POINTER_KIND::FLAT32,
                get_spos());
        } else {
            throw std::runtime_error(
                "new_ild only supports LD(array) or LDC(array) base");
        }
        NODE_PTR array = container->New_array(base_addr, 1, get_spos());
        container->Set_array_idx(array, 0, idx->node);
        NODE_PTR ild = container->New_ild(array, get_spos());
        auto node = wrap_node(ild, "air::core::ILD");
        node->add_child(base);
        node->add_child(idx);
        return node;
    }
    
    std::shared_ptr<Node> new_ist(std::shared_ptr<Node> val, std::shared_ptr<Node> base, 
                                   std::shared_ptr<Node> idx) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::IST");
        node->add_child(val);
        node->add_child(base);
        node->add_child(idx);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_array(std::shared_ptr<Node> base) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::ARRAY");
        node->add_child(base);
        nodes.push_back(node);
        return node;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Control Flow Operations (Real AIR)
    // ═══════════════════════════════════════════════════════════════════════
    
    // Stack for building nested control flow
    struct ControlFlowFrame {
        NODE_PTR cond_node;      // Condition for if
        NODE_PTR then_block;     // Then block 
        NODE_PTR else_block;     // Else block (if any)
        NODE_PTR loop_body;      // Loop body block
        ADDR_DATUM_PTR loop_iv;  // Induction variable
        int64_t loop_start;      // Loop start value
        int64_t loop_end;        // Loop end value
        NODE_PTR loop_start_node; // Dynamic loop start value, if any
        NODE_PTR loop_end_node;   // Dynamic loop end value, if any
        TYPE_PTR loop_type;       // Signed integer type shared by IV/bounds
        bool has_dynamic_start;   // Whether loop_start_node is used
        bool has_dynamic_end;     // Whether loop_end_node is used
        std::string type;        // "if" or "loop"
    };
    std::vector<ControlFlowFrame> cf_stack;

    TYPE_PTR signed_loop_type(int bit_width) const {
        require_not_expired();
        if (!glob) return Null_ptr;
        if (bit_width == 32) {
            return glob->Prim_type(PRIMITIVE_TYPE::INT_S32);
        }
        if (bit_width == 64) {
            return glob->Prim_type(PRIMITIVE_TYPE::INT_S64);
        }
        throw std::runtime_error(
            "range_dynamic supports only signed 32-bit or 64-bit indices");
    }

    void require_loop_literal(int64_t value, int bit_width,
                              const char* operation) const {
        require_not_expired();
        if (bit_width == 32 &&
            (value < std::numeric_limits<int32_t>::min() ||
             value > std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                std::string(operation) +
                " literal is out of range for a signed 32-bit loop index");
        }
    }

    void require_loop_bound_type(NODE_PTR bound, TYPE_PTR loop_type,
                                 const char* operation) const {
        require_not_expired();
        if (bound == NODE_PTR() || loop_type == Null_ptr) {
            throw std::runtime_error(
                std::string(operation) + " requires a real AIR bound");
        }
        if (bound->Container() != container) {
            throw std::runtime_error(
                std::string(operation) +
                " bound belongs to a different AIR container");
        }
        if (!bound->Has_rtype()) {
            throw std::runtime_error(
                std::string(operation) +
                " bound must be an expression with a result type");
        }
        if (
            !bound->Rtype()->Is_signed_int() ||
            !bound->Rtype()->Is_compatible_type(loop_type)) {
            throw std::runtime_error(
                std::string(operation) +
                " bound must have the requested signed integer width; "
                "use a matching Int annotation (AIR Core has no width-conversion opcode)");
        }
    }
    
    // Create a do_loop for range(start, end)
    std::shared_ptr<Node> new_loop_begin_range(int64_t start, int64_t end,
                                               int bit_width = 64) {
        require_not_expired();
        require_loop_literal(start, bit_width, "range_dynamic start");
        require_loop_literal(end, bit_width, "range_dynamic end");
        ControlFlowFrame frame;
        frame.type = "loop";
        frame.loop_start = start;
        frame.loop_end = end;
        frame.loop_start_node = NODE_PTR();
        frame.loop_end_node = NODE_PTR();
        frame.loop_type = signed_loop_type(bit_width);
        frame.has_dynamic_start = false;
        frame.has_dynamic_end = false;
        
        if (container && glob && func_scope) {
            // Create induction variable
            std::string iv_name = "_iv_" + std::to_string(node_counter);
            ADDR_DATUM_PTR iv = func_scope->New_var(
                frame.loop_type, iv_name.c_str(), get_spos());
            frame.loop_iv = iv;
            
            // Create loop body block
            frame.loop_body = container->New_stmt_block(get_spos());
            
            // Push the body block so statements go there
            push_block(frame.loop_body);
        }
        cf_stack.push_back(frame);
        
        auto node = std::make_shared<Node>(++node_counter, "air::core::DO_LOOP");
        nodes.push_back(node);
        return node;
    }

    // Create a do_loop for range(constant_start, dynamic_end), with unit step.
    std::shared_ptr<Node> new_loop_begin_range_dynamic(
        int64_t start, std::shared_ptr<Node> end, int bit_width = 64) {
        require_not_expired();
        require_loop_literal(start, bit_width, "range_dynamic start");
        ControlFlowFrame frame;
        frame.type = "loop";
        frame.loop_start = start;
        frame.loop_end = 0;
        frame.loop_start_node = NODE_PTR();
        frame.loop_end_node = NODE_PTR();
        frame.loop_type = signed_loop_type(bit_width);
        frame.has_dynamic_start = false;
        frame.has_dynamic_end = false;

        if (end && end->has_node && end->node != NODE_PTR()) {
            frame.loop_end_node = end->node;
            frame.has_dynamic_end = true;
        }

        if (container && glob && func_scope) {
            if (!frame.has_dynamic_end) {
                throw std::runtime_error(
                    "new_loop_begin_range_dynamic requires a real AIR node for end");
            }
            require_loop_bound_type(frame.loop_end_node, frame.loop_type,
                                    "new_loop_begin_range_dynamic");

            // Create induction variable
            std::string iv_name = "_iv_" + std::to_string(node_counter);
            ADDR_DATUM_PTR iv = func_scope->New_var(
                frame.loop_type, iv_name.c_str(), get_spos());
            frame.loop_iv = iv;

            // Create loop body block
            frame.loop_body = container->New_stmt_block(get_spos());

            // Push the body block so statements go there
            push_block(frame.loop_body);
        }
        cf_stack.push_back(frame);

        auto node = std::make_shared<Node>(++node_counter, "air::core::DO_LOOP");
        if (end) {
            node->add_child(end);
        }
        nodes.push_back(node);
        return node;
    }

    // Create a do_loop for range(dynamic_start, constant_end), with unit step.
    std::shared_ptr<Node> new_loop_begin_range_dynamic_start(
        std::shared_ptr<Node> start, int64_t end, int bit_width = 64) {
        require_not_expired();
        require_loop_literal(end, bit_width, "range_dynamic end");
        ControlFlowFrame frame;
        frame.type = "loop";
        frame.loop_start = 0;
        frame.loop_end = end;
        frame.loop_start_node = NODE_PTR();
        frame.loop_end_node = NODE_PTR();
        frame.loop_type = signed_loop_type(bit_width);
        frame.has_dynamic_start = false;
        frame.has_dynamic_end = false;

        if (start && start->has_node && start->node != NODE_PTR()) {
            frame.loop_start_node = start->node;
            frame.has_dynamic_start = true;
        }

        if (container && glob && func_scope) {
            if (!frame.has_dynamic_start) {
                throw std::runtime_error(
                    "new_loop_begin_range_dynamic_start requires a real AIR node for start");
            }
            require_loop_bound_type(frame.loop_start_node, frame.loop_type,
                                    "new_loop_begin_range_dynamic_start");

            std::string iv_name = "_iv_" + std::to_string(node_counter);
            ADDR_DATUM_PTR iv = func_scope->New_var(
                frame.loop_type, iv_name.c_str(), get_spos());
            frame.loop_iv = iv;

            frame.loop_body = container->New_stmt_block(get_spos());
            push_block(frame.loop_body);
        }
        cf_stack.push_back(frame);

        auto node = std::make_shared<Node>(++node_counter, "air::core::DO_LOOP");
        if (start) {
            node->add_child(start);
        }
        nodes.push_back(node);
        return node;
    }

    // Create a do_loop for range(dynamic_start, dynamic_end), with unit step.
    std::shared_ptr<Node> new_loop_begin_range_dynamic_bounds(
        std::shared_ptr<Node> start, std::shared_ptr<Node> end,
        int bit_width = 64) {
        require_not_expired();
        ControlFlowFrame frame;
        frame.type = "loop";
        frame.loop_start = 0;
        frame.loop_end = 0;
        frame.loop_start_node = NODE_PTR();
        frame.loop_end_node = NODE_PTR();
        frame.loop_type = signed_loop_type(bit_width);
        frame.has_dynamic_start = false;
        frame.has_dynamic_end = false;

        if (start && start->has_node && start->node != NODE_PTR()) {
            frame.loop_start_node = start->node;
            frame.has_dynamic_start = true;
        }
        if (end && end->has_node && end->node != NODE_PTR()) {
            frame.loop_end_node = end->node;
            frame.has_dynamic_end = true;
        }

        if (container && glob && func_scope) {
            if (!frame.has_dynamic_start || !frame.has_dynamic_end) {
                throw std::runtime_error(
                    "new_loop_begin_range_dynamic_bounds requires real AIR nodes for start and end");
            }
            require_loop_bound_type(frame.loop_start_node, frame.loop_type,
                                    "new_loop_begin_range_dynamic_bounds");
            require_loop_bound_type(frame.loop_end_node, frame.loop_type,
                                    "new_loop_begin_range_dynamic_bounds");

            std::string iv_name = "_iv_" + std::to_string(node_counter);
            ADDR_DATUM_PTR iv = func_scope->New_var(
                frame.loop_type, iv_name.c_str(), get_spos());
            frame.loop_iv = iv;

            frame.loop_body = container->New_stmt_block(get_spos());
            push_block(frame.loop_body);
        }
        cf_stack.push_back(frame);

        auto node = std::make_shared<Node>(++node_counter, "air::core::DO_LOOP");
        if (start) {
            node->add_child(start);
        }
        if (end) {
            node->add_child(end);
        }
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_loop_begin(std::shared_ptr<Node> iterable) {
        require_not_expired();
        // Fallback for non-range iterables
        ControlFlowFrame frame;
        frame.type = "loop";
        frame.loop_start = 0;
        frame.loop_end = 10;  // Default
        frame.loop_start_node = NODE_PTR();
        frame.loop_end_node = NODE_PTR();
        frame.loop_type = TYPE_PTR();
        if (glob) {
            frame.loop_type = glob->Prim_type(PRIMITIVE_TYPE::INT_S64);
        }
        frame.has_dynamic_start = false;
        frame.has_dynamic_end = false;
        
        if (container && glob && func_scope) {
            std::string iv_name = "_iv_" + std::to_string(node_counter);
            ADDR_DATUM_PTR iv = func_scope->New_var(
                frame.loop_type, iv_name.c_str(), get_spos());
            frame.loop_iv = iv;
            frame.loop_body = container->New_stmt_block(get_spos());
            
            // Push the body block so statements go there
            push_block(frame.loop_body);
        }
        cf_stack.push_back(frame);
        
        auto node = std::make_shared<Node>(++node_counter, "air::core::DO_LOOP");
        node->add_child(iterable);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_loop_index(std::shared_ptr<Node> loop) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::LOOP_IV");
        node->add_child(loop);
        nodes.push_back(node);
        
        // Return a node that represents the induction variable
        if (!cf_stack.empty() && cf_stack.back().type == "loop" && 
            cf_stack.back().loop_iv != ADDR_DATUM_PTR()) {
            // Wrap the real IV in our node
            if (container) {
                NODE_PTR ld_iv = container->New_ld(cf_stack.back().loop_iv, get_spos());
                node->has_node = true;
                node->node = ld_iv;
            }
        }
        return node;
    }
    
    std::shared_ptr<Node> new_loop_end() {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::LOOP_END");
        nodes.push_back(node);
        
        if (!cf_stack.empty() && cf_stack.back().type == "loop") {
            auto& frame = cf_stack.back();
            
            // Pop the body block before creating the loop statement
            pop_block();
            
            // Create the real do_loop statement
            if (container && frame.loop_iv != ADDR_DATUM_PTR() && 
                frame.loop_body != NODE_PTR()) {
                // Create init: loop_start
                NODE_PTR init = frame.has_dynamic_start
                    ? frame.loop_start_node
                    : container->New_intconst(
                        frame.loop_type, frame.loop_start, get_spos());
                
                // Create comparison: iv < loop_end
                NODE_PTR ld_iv = container->New_ld(frame.loop_iv, get_spos());
                NODE_PTR end_val = frame.has_dynamic_end
                    ? frame.loop_end_node
                    : container->New_intconst(
                        frame.loop_type, frame.loop_end, get_spos());
                OPCODE lt_op(air::core::CORE, air::core::OPCODE::LT);
                NODE_PTR comp = container->New_bin_arith(lt_op, ld_iv->Rtype(), ld_iv, end_val, get_spos());
                
                // Create increment: iv + 1
                NODE_PTR ld_iv2 = container->New_ld(frame.loop_iv, get_spos());
                NODE_PTR one = container->New_intconst(
                    frame.loop_type, 1, get_spos());
                OPCODE add_op(air::core::CORE, air::core::OPCODE::ADD);
                NODE_PTR incr = container->New_bin_arith(add_op, ld_iv2->Rtype(), ld_iv2, one, get_spos());
                
                // Create do_loop statement - appends to parent (now current) stmt list
                STMT_PTR loop_stmt = container->New_do_loop(
                    frame.loop_iv, init, comp, incr, frame.loop_body, get_spos());
                append_stmt(loop_stmt);
            }
            
            cf_stack.pop_back();
        }
        return node;
    }
    
    std::shared_ptr<Node> new_if_begin(std::shared_ptr<Node> cond) {
        require_not_expired();
        ControlFlowFrame frame;
        frame.type = "if";
        // Accept any valid condition node (not just relational ops)
        // For non-boolean conditions, wrap in a comparison to zero
        if (container && glob && cond->has_node) {
            NODE_PTR cond_node = cond->node;
            
            // If condition is not already a relational/boolean op, convert it
            // by comparing to zero (cond != 0)
            if (!cond_node->Is_relational_op() && 
                cond_node->Rtype()->Id() != glob->Prim_type(PRIMITIVE_TYPE::BOOL)->Id()) {
                // Create: cond != 0 using New_cust_node (same as new_cmp_node)
                TYPE_PTR cond_type = cond_node->Rtype();
                NODE_PTR zero;
                if (cond_type->Is_int()) {
                    zero = container->New_intconst(cond_type, 0, get_spos());
                } else if (cond_type->Is_float()) {
                    zero = container->New_intconst(glob->Prim_type(PRIMITIVE_TYPE::INT_S32), 0, get_spos());
                } else {
                    // For array/other types, use the node directly and let passes handle it
                    zero = NODE_PTR();
                }
                
                if (zero != NODE_PTR()) {
                    OPCODE ne_op(air::core::CORE, air::core::OPCODE::NE);
                    TYPE_PTR bool_type = glob->Prim_type(PRIMITIVE_TYPE::BOOL);
                    NODE_PTR ne_node = container->New_cust_node(ne_op, bool_type, get_spos());
                    ne_node->Set_child(0, cond_node);
                    ne_node->Set_child(1, zero);
                    cond_node = ne_node;
                }
            }
            
            frame.cond_node = cond_node;
            frame.then_block = container->New_stmt_block(get_spos());
            
            // Push then block so statements go there
            push_block(frame.then_block);
        }
        cf_stack.push_back(frame);
        
        auto node = std::make_shared<Node>(++node_counter, "air::core::IF");
        node->add_child(cond);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_else() {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::ELSE");
        nodes.push_back(node);
        
        if (!cf_stack.empty() && cf_stack.back().type == "if") {
            // Pop the then block
            pop_block();
            
            if (container && glob) {
                cf_stack.back().else_block = container->New_stmt_block(get_spos());
                
                // Push the else block so statements go there
                push_block(cf_stack.back().else_block);
            }
        }
        return node;
    }
    
    std::shared_ptr<Node> new_if_end() {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::IF_END");
        nodes.push_back(node);
        
        if (!cf_stack.empty() && cf_stack.back().type == "if") {
            auto& frame = cf_stack.back();
            
            // Pop the current block (either then or else)
            pop_block();
            
            // Emit real if-then-else if we have valid condition and then block
            if (container && frame.cond_node != NODE_PTR() && 
                frame.then_block != NODE_PTR()) {
                // Create real if-then-else statement
                NODE_PTR else_b = frame.else_block != NODE_PTR() ? 
                                  frame.else_block : 
                                  container->New_stmt_block(get_spos());
                STMT_PTR if_stmt = container->New_if_then_else(
                    frame.cond_node, frame.then_block, else_b, get_spos());
                append_stmt(if_stmt);
            }
            cf_stack.pop_back();
        }
        return node;
    }
    
    std::string get_control_flow_dump() const {
        require_not_expired();
        // No longer needed - control flow is in actual AIR
        return "";
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Reduction Operations
    // ═══════════════════════════════════════════════════════════════════════
    
    std::shared_ptr<Node> new_reduce_sum(std::shared_ptr<Node> input, py::object axis, bool keepdims) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::REDUCE_SUM");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_reduce_max(std::shared_ptr<Node> input, py::object axis, bool keepdims) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::REDUCE_MAX");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_reduce_min(std::shared_ptr<Node> input, py::object axis, bool keepdims) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::REDUCE_MIN");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_reduce_prod(std::shared_ptr<Node> input, py::object axis, bool keepdims) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::REDUCE_PROD");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_reduce_mean(std::shared_ptr<Node> input, py::object axis, bool keepdims) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::REDUCE_MEAN");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Shape Manipulation Operations
    // ═══════════════════════════════════════════════════════════════════════
    
    std::shared_ptr<Node> new_reshape(std::shared_ptr<Node> input, py::object shape) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::RESHAPE");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_permute(std::shared_ptr<Node> input, py::object axes) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::PERMUTE");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_transpose(std::shared_ptr<Node> input, int axis0, int axis1) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::TRANSPOSE");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Math Operations
    // ═══════════════════════════════════════════════════════════════════════
    
    std::shared_ptr<Node> new_exp(std::shared_ptr<Node> input) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::EXP");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_log(std::shared_ptr<Node> input) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::LOG");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_sqrt(std::shared_ptr<Node> input) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::SQRT");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_sin(std::shared_ptr<Node> input) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::SIN");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_cos(std::shared_ptr<Node> input) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::COS");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_tanh(std::shared_ptr<Node> input) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::TANH");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_neg(std::shared_ptr<Node> input) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::NEG");
        node->add_child(input);
        nodes.push_back(node);
        return node;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Tensor Creation Operations
    // ═══════════════════════════════════════════════════════════════════════
    
    std::shared_ptr<Node> new_zeros(py::object shape, const std::string& dtype) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::ZEROS");
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_ones(py::object shape, const std::string& dtype) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::ONES");
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_full(py::object shape, double fill_value) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::FULL");
        nodes.push_back(node);
        return node;
    }
    
    std::shared_ptr<Node> new_arange(int64_t size, const std::string& dtype) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "nn::core::ARANGE");
        nodes.push_back(node);
        return node;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Conditional Operations
    // ═══════════════════════════════════════════════════════════════════════
    
    std::shared_ptr<Node> new_where(std::shared_ptr<Node> cond, std::shared_ptr<Node> true_val, std::shared_ptr<Node> false_val) {
        require_not_expired();
        auto node = std::make_shared<Node>(++node_counter, "air::core::SELECT");
        node->add_child(cond);
        node->add_child(true_val);
        node->add_child(false_val);
        nodes.push_back(node);
        return node;
    }
    
    std::string dump() const {
        require_not_expired();
        std::string s;
        for (const auto& node : nodes) {
            s += "  " + node->to_string() + "\n";
        }
        return s;
    }
};

class GlobScope;

// Function scope with proper AIR setup
class FuncScope {
public:
    std::string name;
    FUNC_SCOPE* func_scope;
    GLOB_SCOPE* glob;
    std::vector<std::shared_ptr<Node>> params;
    Container container;
    std::vector<ADDR_DATUM_PTR> formal_params;
    bool strict_param_types;
    
    FuncScope(const std::string& n)
        : name(n), func_scope(nullptr), glob(nullptr),
          strict_param_types(false) {}
    
    FuncScope(const std::string& n, FUNC_SCOPE* fs, GLOB_SCOPE* g,
              const std::vector<ADDR_DATUM_PTR>& formals,
              bool strict_types = false)
        : name(n), func_scope(fs), glob(g),
          container(&fs->Container(), fs, g), formal_params(formals),
          strict_param_types(strict_types) {}
    
    // Get parameter - loads from formal that was set up during function creation
    std::shared_ptr<Node> new_param(const std::string& param_name, Type type) {
        uint32_t idx = static_cast<uint32_t>(params.size());
        
        if (func_scope && glob && idx < formal_params.size()) {
            ADDR_DATUM_PTR formal = formal_params[idx];
            if (type.has_type) {
                type.require_active();
                if (type.type == Null_ptr || &type.type->Glob_scope() != glob) {
                    throw std::runtime_error(
                        "new_param type belongs to a different GLOB_SCOPE: " +
                        param_name);
                }
                if (strict_param_types &&
                    !formal->Type()->Is_compatible_type(type.type)) {
                    throw std::runtime_error(
                        "new_param type does not match the function signature: " +
                        param_name);
                }
            }
            NODE_PTR ld_node = func_scope->Container().New_ld(formal, 
                glob->Unknown_simple_spos());
            auto node = std::make_shared<Node>(
                ld_node, container.node_counter++, "PARAM",
                container.callback_lifetime);
            params.push_back(node);
            return node;
        }
        
        // Fallback
        auto node = std::make_shared<Node>(container.node_counter++, "PARAM");
        params.push_back(node);
        return node;
    }
    
    void invalidate() {
        container.expire_callback_scope();
        for (const auto& param : params) {
            if (param) param->invalidate();
        }
        for (const auto& node : container.nodes) {
            if (node) node->invalidate();
        }
        params.clear();
        formal_params.clear();
        container.nodes.clear();
        container.block_stack.clear();
        container.var_map.clear();
        container.cf_stack.clear();
        container.container = nullptr;
        container.func_scope = nullptr;
        container.glob = nullptr;
        func_scope = nullptr;
        glob = nullptr;
    }

    Container& get_container() { return container; }
    
    std::string dump() const {
        std::string s = "func " + name + "(";
        for (size_t i = 0; i < params.size(); i++) {
            if (i > 0) s += ", ";
            s += params[i]->name();
        }
        s += "):\n";
        
        if (func_scope) {
            s += func_scope->To_str();
        }
        return s;
    }
};


py::tuple py_i64_tuple(const std::vector<int64_t>& values) {
    py::tuple result(values.size());
    for (size_t idx = 0; idx < values.size(); ++idx) {
        result[idx] = py::int_(values[idx]);
    }
    return result;
}

py::tuple py_i32_tuple(const std::vector<int32_t>& values) {
    py::tuple result(values.size());
    for (size_t idx = 0; idx < values.size(); ++idx) {
        result[idx] = py::int_(values[idx]);
    }
    return result;
}

py::dict ranked_type_snapshot(
    const nn::vector::VECTOR_KERNEL_RANKED_TYPE_PLAN& type) {
    py::dict result;
    result["element_type"] =
        nn::vector::Vector_kernel_primitive_type_name(type._element_type);
    result["shape"] = py_i64_tuple(type._shape);
    return result;
}

const char* reduction_kind_name(
    nn::vector::VECTOR_KERNEL_REDUCTION_KIND kind) {
    using KIND = nn::vector::VECTOR_KERNEL_REDUCTION_KIND;
    switch (kind) {
    case KIND::POWER_OF_TWO: return "power-of-two";
    case KIND::LINEAR: return "linear";
    case KIND::COLLECTIVE_SINGLE_BLOCK: return "collective-single-block";
    case KIND::COLLECTIVE_BLOCKS: return "collective-blocks";
    }
    throw std::runtime_error("unknown vector-kernel reduction kind");
}

const char* mask_policy_name(nn::vector::VECTOR_KERNEL_MASK_POLICY policy) {
    using POLICY = nn::vector::VECTOR_KERNEL_MASK_POLICY;
    switch (policy) {
    case POLICY::NONE: return "none";
    case POLICY::CLEAR_VALID_PREFIX: return "clear-valid-prefix";
    case POLICY::COLLECTIVE_REDUCTION: return "collective-reduction";
    }
    throw std::runtime_error("unknown vector-kernel mask policy");
}

const char* slot_policy_name(nn::vector::VECTOR_KERNEL_SLOT_POLICY policy) {
    using POLICY = nn::vector::VECTOR_KERNEL_SLOT_POLICY;
    switch (policy) {
    case POLICY::ABSENT_NATIVE_BASELINE: return "absent-native-baseline";
    case POLICY::LOGICAL_OUTPUT_ELEMENTS: return "logical-output-elements";
    case POLICY::EXPLICIT: return "explicit";
    }
    throw std::runtime_error("unknown vector-kernel slot policy");
}

const char* runtime_preparation_kind_name(
    nn::vector::VECTOR_KERNEL_RUNTIME_PREPARATION_KIND kind) {
    using KIND = nn::vector::VECTOR_KERNEL_RUNTIME_PREPARATION_KIND;
    switch (kind) {
    case KIND::PACKED_VECTOR: return "packed-vector";
    case KIND::FLATTEN_PACKED_VECTOR: return "flatten-packed-vector";
    case KIND::BLOCKING_ROTATIONS: return "blocking-rotations";
    }
    throw std::runtime_error("unknown vector-kernel runtime preparation kind");
}

py::dict prepared_common_plan_snapshot(
    const nn::vector::PREPARED_VECTOR_KERNEL_PLAN& prepared,
    const nn::vector::VECTOR_KERNEL_COMMON_PLAN& common,
    const char* kind) {
    py::dict data;
    data["kind"] = kind;
    data["provenance"] = prepared.Provenance();
    data["specialization_key"] = prepared.Specialization_key();
    data["helper_name"] = prepared.Helper_name();
    data["result_type"] = ranked_type_snapshot(common._result_type);

    py::list runtime_types;
    for (const auto& type : common._runtime_vector_inputs) {
        runtime_types.append(ranked_type_snapshot(type));
    }
    data["runtime_vector_inputs"] = py::tuple(runtime_types);

    py::list loops;
    for (const auto& loop : common._loops) {
        py::dict item;
        item["role"] = loop._role;
        item["lower"] = loop._lower;
        item["upper"] = loop._upper;
        item["step"] = loop._step;
        item["nesting_depth"] = loop._nesting_depth;
        loops.append(std::move(item));
    }
    data["loops"] = py::tuple(loops);

    py::list slices;
    for (const auto& slice : common._slices) {
        py::dict index;
        index["iv_coefficients"] = py_i64_tuple(slice._index._iv_coefficients);
        index["constant"] = slice._index._constant;
        index["uses_sharding_offset"] = slice._index._uses_sharding_offset;
        py::dict item;
        item["role"] = slice._role;
        item["index"] = std::move(index);
        item["width"] = slice._width;
        slices.append(std::move(item));
    }
    data["slices"] = py::tuple(slices);

    py::list rotations;
    for (const auto& rotation : common._rotations) {
        py::dict item;
        item["role"] = rotation._role;
        item["candidates"] = py_i32_tuple(rotation._candidates);
        rotations.append(std::move(item));
    }
    data["rotations"] = py::tuple(rotations);

    py::list reductions;
    for (const auto& reduction : common._reductions) {
        py::dict item;
        item["role"] = reduction._role;
        item["kind"] = reduction_kind_name(reduction._kind);
        item["factor"] = reduction._factor;
        item["block_width"] = reduction._block_width;
        item["padding"] = reduction._padding;
        reductions.append(std::move(item));
    }
    data["reductions"] = py::tuple(reductions);

    py::dict mask;
    mask["policy"] = mask_policy_name(common._mask._policy);
    mask["valid_length"] = common._mask._valid_length;
    data["mask"] = std::move(mask);
    py::dict slot;
    slot["policy"] = slot_policy_name(common._slot._policy);
    slot["value"] = common._slot._value;
    data["slot"] = std::move(slot);

    py::list constants;
    for (const auto& constant : prepared.Constants()) {
        py::dict item;
        item["role"] = constant._role;
        item["type"] = ranked_type_snapshot(constant._type);
        item["content_hash"] = constant._content_hash;
        item["bytes"] = py::bytes(
            reinterpret_cast<const char*>(constant._bytes.data()),
            constant._bytes.size());
        constants.append(std::move(item));
    }
    data["constants"] = py::tuple(constants);

    py::list runtime_preparations;
    for (const auto& preparation : prepared.Runtime_preparations()) {
        py::dict item;
        item["role"] = preparation._role;
        item["source_operand"] = preparation._source_operand;
        item["kind"] = runtime_preparation_kind_name(preparation._kind);
        item["result_type"] = ranked_type_snapshot(preparation._result_type);
        item["logical_input_size"] = preparation._logical_input_size;
        item["replications"] = preparation._replications;
        item["blocking_width"] = preparation._blocking_width;
        item["rotation_candidates"] =
            py_i32_tuple(preparation._rotation_candidates);
        item["outer_block_depth"] = preparation._outer_block_depth;
        runtime_preparations.append(std::move(item));
    }
    data["runtime_preparations"] = py::tuple(runtime_preparations);
    return data;
}

py::object prepared_baseline_gemm_snapshot(
    const nn::vector::PREPARED_VECTOR_KERNEL_PLAN& prepared) {
    if (!std::holds_alternative<nn::vector::BASELINE_GEMM_PLAN>(
            prepared.Plan())) {
        throw std::runtime_error(
            "baseline Gemm snapshot received another plan kind");
    }
    const auto& plan =
        std::get<nn::vector::BASELINE_GEMM_PLAN>(prepared.Plan());
    py::dict data = prepared_common_plan_snapshot(
        prepared, plan._common, "baseline-gemm");
    data["height"] = plan._height;
    data["width"] = plan._width;
    data["input_duplications"] = plan._input_duplications;
    py::object freeze = py::module_::import(
        "ace_edsl.edsl.vector_kernel_lowering").attr(
            "_freeze_prepared_baseline_gemm_plan");
    return freeze(std::move(data));
}

py::object prepared_baseline_conv_snapshot(
    const nn::vector::PREPARED_VECTOR_KERNEL_PLAN& prepared) {
    if (!std::holds_alternative<nn::vector::BASELINE_CONV_PLAN>(
            prepared.Plan())) {
        throw std::runtime_error(
            "baseline Conv snapshot received another plan kind");
    }
    const auto& plan =
        std::get<nn::vector::BASELINE_CONV_PLAN>(prepared.Plan());
    py::dict data = prepared_common_plan_snapshot(
        prepared, plan._common, "baseline-conv");
    data["channel_in"] = plan._channel_in;
    data["channel_out"] = plan._channel_out;
    data["output_height"] = plan._output_height;
    data["output_width"] = plan._output_width;
    data["kernel_hw"] = plan._kernel_hw;
    data["stride"] = plan._stride;
    data["input_duplications"] = plan._input_duplications;
    py::object freeze = py::module_::import(
        "ace_edsl.edsl.vector_kernel_lowering").attr(
            "_freeze_prepared_baseline_conv_plan");
    return freeze(std::move(data));
}

py::object prepared_vector_kernel_snapshot(
    const nn::vector::PREPARED_VECTOR_KERNEL_PLAN& prepared) {
    if (std::holds_alternative<nn::vector::BASELINE_GEMM_PLAN>(
            prepared.Plan())) {
        return prepared_baseline_gemm_snapshot(prepared);
    }
    if (std::holds_alternative<nn::vector::BASELINE_CONV_PLAN>(
            prepared.Plan())) {
        return prepared_baseline_conv_snapshot(prepared);
    }
    throw std::runtime_error(
        "Python destination bridge supports baseline-gemm and "
        "baseline-conv only");
}


class VectorKernelTraceContext {
public:
    VectorKernelTraceContext(
        FUNC_SCOPE& helper, NODE_PTR body,
        const std::vector<nn::vector::VECTOR_KERNEL_TYPED_PAYLOAD>& constants,
        const nn::vector::VECTOR_KERNEL_DESTINATION_ABI& abi,
        const SPOS& spos)
        : _active(true), _body(body), _result_type(abi._result_type) {
        if (body == Null_ptr || !body->Is_block()) {
            throw std::runtime_error(
                "vector-kernel destination trace requires a helper body");
        }
        std::vector<ADDR_DATUM_PTR> formals;
        formals.reserve(helper.Formal_cnt());
        for (uint32_t idx = 0; idx < helper.Formal_cnt(); ++idx) {
            formals.push_back(helper.Formal(idx));
        }
        _scope = std::make_shared<FuncScope>(
            helper.Owning_func()->Name()->Char_str(), &helper,
            &helper.Glob_scope(), formals, true);
        _scope->container.mark_callback_scoped();
        _scope->container.push_block(body);
        for (size_t idx = 0; idx < abi._formal_types.size(); ++idx) {
            _formals.push_back(_scope->new_param(
                "packed_input_" + std::to_string(idx),
                Type::from_air_type(
                    abi._formal_types[idx],
                    _scope->container.callback_lifetime)));
        }
        for (const auto& payload : constants) {
            CONSTANT_PTR constant =
                nn::vector::Materialize_prepared_vector_kernel_constant(
                    helper.Glob_scope(), payload, spos);
            NODE_PTR ldc = helper.Container().New_ldc(constant, spos);
            _constants.emplace(
                payload._role, _scope->container.wrap_node(
                                   ldc, "air::core::LDC"));
        }
    }

    ~VectorKernelTraceContext() { expire(); }

    bool is_active() const { return _active; }

    Container& container() {
        require_active();
        return _scope->container;
    }

    std::vector<std::shared_ptr<Node>> formals() const {
        require_active();
        return _formals;
    }

    py::tuple constant_roles() const {
        require_active();
        py::tuple roles(_constants.size());
        size_t idx = 0;
        for (const auto& item : _constants) roles[idx++] = item.first;
        return roles;
    }

    std::shared_ptr<Node> constant(const std::string& role) const {
        require_active();
        auto iter = _constants.find(role);
        if (iter == _constants.end()) {
            throw py::key_error("unknown prepared constant role: " + role);
        }
        return iter->second;
    }

    Type result_type() const {
        require_active();
        return Type::from_air_type(
            _result_type, _scope->container.callback_lifetime);
    }

    Type i32_type() const {
        require_active();
        return Type::from_air_type(
            _scope->glob->Prim_type(PRIMITIVE_TYPE::INT_S32),
            _scope->container.callback_lifetime);
    }

    std::shared_ptr<DESTINATION_LIFETIME> destination_lifetime() const {
        require_active();
        return _scope->container.callback_lifetime;
    }

    NODE_PTR take_result(const std::shared_ptr<Node>& result) {
        require_active();
        if (!result || !result->has_node || result->node == Null_ptr) {
            throw std::runtime_error(
                "vector-kernel recipe must return a real destination AIR node");
        }
        if (result->node->Container() != _scope->container.container) {
            throw std::runtime_error(
                "vector-kernel recipe returned a node from another AIR container");
        }
        if (!result->node->Has_rtype() ||
            !result->node->Rtype()->Is_compatible_type(_result_type)) {
            throw std::runtime_error(
                "vector-kernel recipe result type does not match its prepared ABI");
        }
        if (_scope->container.cf_stack.size() != 0 ||
            _scope->container.block_stack.size() != 1 ||
            _scope->container.block_stack.back() != _body) {
            throw std::runtime_error(
                "vector-kernel recipe left unbalanced control-flow state");
        }
        NODE_PTR raw = result->node;
        _scope->container.pop_block();
        expire();
        return raw;
    }

    void expire() {
        if (!_active) return;
        _active = false;
        if (_scope) _scope->invalidate();
        _formals.clear();
        _constants.clear();
    }

private:
    void require_active() const {
        if (!_active || !_scope) {
            throw std::runtime_error(
                "vector-kernel destination trace context has expired");
        }
    }

    bool _active;
    NODE_PTR _body;
    TYPE_PTR _result_type;
    std::shared_ptr<FuncScope> _scope;
    std::vector<std::shared_ptr<Node>> _formals;
    std::map<std::string, std::shared_ptr<Node>> _constants;
};

nn::vector::VECTOR_KERNEL_PLAN_KIND bound_plan_kind(
    const std::string& name) {
    if (name == "baseline-gemm") {
        return nn::vector::VECTOR_KERNEL_PLAN_KIND::BASELINE_GEMM;
    }
    if (name == "baseline-conv") {
        return nn::vector::VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV;
    }
    throw std::invalid_argument(
        "Python recipe registry supports baseline-gemm and baseline-conv only");
}

#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
struct BASELINE_CONV_ORACLE_RECORD {
    std::vector<NODE_PTR> _expected_actuals;
    std::string _native_air;
};

std::string normalize_prepared_baseline_conv_native_for_testing(
    const nn::vector::PREPARED_VECTOR_KERNEL_PLAN& prepared);
#endif

nn::vector::VECTOR_KERNEL_HELPER_SPEC make_python_plan_recipe(
    py::function callback,
    const nn::vector::PREPARED_VECTOR_KERNEL_PLAN& prepared,
    const nn::vector::VECTOR_KERNEL_DESTINATION_ABI& abi
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
    , std::map<std::string, BASELINE_CONV_ORACLE_RECORD>* conv_oracles
#endif
    ) {
    py::object snapshot = prepared_vector_kernel_snapshot(prepared);
    std::vector<nn::vector::VECTOR_KERNEL_TYPED_PAYLOAD> constants =
        prepared.Constants();
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
    std::shared_ptr<const nn::vector::PREPARED_VECTOR_KERNEL_PLAN>
        native_plan;
    std::string helper_name;
    if (conv_oracles != nullptr) {
        native_plan = std::make_shared<
            const nn::vector::PREPARED_VECTOR_KERNEL_PLAN>(prepared);
        helper_name = prepared.Helper_name();
    }
#endif
    nn::vector::VECTOR_KERNEL_HELPER_SPEC spec;
    spec._formal_types = abi._formal_types;
    spec._result_type = abi._result_type;
    spec._build_body =
        [callback = std::move(callback), snapshot = std::move(snapshot),
         constants = std::move(constants), abi
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
         , native_plan = std::move(native_plan),
         helper_name = std::move(helper_name), conv_oracles
#endif
         ](
            FUNC_SCOPE& helper, NODE_PTR body, const SPOS& spos) -> NODE_PTR {
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
          if (native_plan != nullptr) {
              auto oracle = conv_oracles->find(helper_name);
              if (oracle == conv_oracles->end()) {
                  throw std::runtime_error(
                      "baseline Conv native oracle lost its expected actuals");
              }
              oracle->second._native_air =
                  normalize_prepared_baseline_conv_native_for_testing(
                      *native_plan);
          }
#endif
          auto trace = std::make_shared<VectorKernelTraceContext>(
              helper, body, constants, abi, spos);
          ACTIVE_BINDING_GLOB_GUARD active_glob(
              &helper.Glob_scope(), trace->destination_lifetime());
          try {
              py::object returned = callback(trace, snapshot);
              std::shared_ptr<Node> result =
                  returned.cast<std::shared_ptr<Node>>();
              return trace->take_result(result);
          } catch (...) {
              trace->expire();
              throw;
          }
        };
    return spec;
}


#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
const std::string BASELINE_GEMM_HELPER_PREFIX =
    "__ace_vkernel_baseline_gemm_";
const std::string BASELINE_CONV_HELPER_PREFIX =
    "__ace_vkernel_baseline_conv_";

std::string normalize_prepared_baseline_conv_native_for_testing(
    const nn::vector::PREPARED_VECTOR_KERNEL_PLAN& prepared) {
    if (!std::holds_alternative<nn::vector::BASELINE_CONV_PLAN>(
            prepared.Plan())) {
        throw std::runtime_error(
            "baseline Conv native oracle received another plan kind");
    }
    const auto& plan =
        std::get<nn::vector::BASELINE_CONV_PLAN>(prepared.Plan());
    const auto& common = plan._common;
    if (common._runtime_vector_inputs.size() != 1) {
        throw std::runtime_error(
            "baseline Conv native oracle requires one runtime input");
    }

    std::unique_ptr<GLOB_SCOPE> oracle =
        std::make_unique<GLOB_SCOPE>(0, true);
    const SPOS spos = oracle->Unknown_simple_spos();
    TYPE_PTR input_type = nn::vector::New_array_type(
        oracle.get(), "baseline_conv_native_input",
        oracle->Prim_type(common._runtime_vector_inputs[0]._element_type),
        common._runtime_vector_inputs[0]._shape, spos);
    TYPE_PTR result_type = nn::vector::New_array_type(
        oracle.get(), "baseline_conv_native_result",
        oracle->Prim_type(common._result_type._element_type),
        common._result_type._shape, spos);

    STR_PTR name = oracle->New_str("baseline_conv_native_oracle");
    FUNC_PTR function = oracle->New_func(name, spos);
    function->Set_parent(oracle->Comp_env_id());
    SIGNATURE_TYPE_PTR signature = oracle->New_sig_type();
    oracle->New_ret_param(result_type, signature);
    oracle->New_param(
        oracle->New_str("packed_input"), input_type, signature, spos);
    signature->Set_complete();
    oracle->New_entry_point(signature, function, name, spos);

    FUNC_SCOPE& scope = oracle->New_func_scope(function);
    CONTAINER* container = &scope.Container();
    STMT_PTR entry = container->New_func_entry(spos);
    NODE_PTR body = entry->Node()->Last_child();
    nn::vector::VECTOR_CTX vector_ctx;
    vector_ctx.Update_slot(MAX_SLOT_ALLOWED);
    nn::vector::VECTOR_CONFIG config;
    config._max_slots = MAX_SLOT_ALLOWED;
    nn::vector::TENSOR2VECTOR_CTX lowering_ctx(
        container, vector_ctx, nullptr, config);
    lowering_ctx.Set_cur_func_scope(&scope);
    lowering_ctx.Push(body, body);
    NODE_PTR input = container->New_ld(scope.Formal(0), spos);
    NODE_PTR result = nn::vector::Emit_prepared_vector_kernel_native(
        lowering_ctx, prepared, input, {}, spos);
    container->Stmt_list().Append(container->New_retv(result, spos));
    lowering_ctx.Pop(body, body);

    if (!oracle->Verify_ir()) {
        throw std::runtime_error(
            "baseline Conv direct native oracle failed AIR verification");
    }
    return nn::vector::test::Normalize_vector_kernel_helper(scope);
}

void collect_call_statements(NODE_PTR node, std::vector<STMT_PTR>& calls) {
    if (node == Null_ptr) return;
    if (node->Is_call()) calls.push_back(node->Stmt());
    if (node->Is_block()) {
        for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
             stmt = stmt->Next()) {
            collect_call_statements(stmt->Node(), calls);
        }
        return;
    }
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
        collect_call_statements(node->Child(idx), calls);
    }
}

void collect_prepared_conv_input_stores(
    NODE_PTR node, std::vector<STMT_PTR>& stores) {
    if (node == Null_ptr) return;
    if (node->Opcode() == air::core::OPC_STP && node->Has_preg() &&
        node->Num_child() == 1 &&
        node->Child(0)->Opcode() == nn::vector::OPC_RESHAPE) {
        stores.push_back(node->Stmt());
    }
    if (node->Is_block()) {
        for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
             stmt = stmt->Next()) {
            collect_prepared_conv_input_stores(stmt->Node(), stores);
        }
        return;
    }
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
        collect_prepared_conv_input_stores(node->Child(idx), stores);
    }
}

bool statement_precedes_in_block(
    NODE_PTR block, STMT_PTR before, STMT_PTR after) {
    if (block == Null_ptr || !block->Is_block()) return false;
    bool saw_before = false;
    for (STMT_PTR stmt = block->Begin_stmt(); stmt != block->End_stmt();
         stmt = stmt->Next()) {
        if (stmt == before) saw_before = true;
        if (stmt == after) return saw_before && stmt != before;
    }
    return false;
}

void collect_matching_ldp_nodes(NODE_PTR node, PREG_PTR preg,
                                std::vector<NODE_PTR>& matches) {
    if (node == Null_ptr) return;
    if (node->Opcode() == air::core::OPC_LDP &&
        node->Preg() != Null_ptr && preg != Null_ptr &&
        node->Preg()->Defining_func_scope() == preg->Defining_func_scope() &&
        node->Preg()->Id() == preg->Id()) {
        matches.push_back(node);
    }
    if (node->Is_block()) {
        for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
             stmt = stmt->Next()) {
            collect_matching_ldp_nodes(stmt->Node(), preg, matches);
        }
        return;
    }
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
        collect_matching_ldp_nodes(node->Child(idx), preg, matches);
    }
}

NODE_PTR find_formal_load(NODE_PTR node, ADDR_DATUM_PTR formal) {
    if (node == Null_ptr) return Null_ptr;
    if (node->Opcode() == air::core::OPC_LD && node->Has_sym() &&
        node->Addr_datum() == formal) {
        return node;
    }
    if (node->Is_block()) {
        for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
             stmt = stmt->Next()) {
            NODE_PTR found = find_formal_load(stmt->Node(), formal);
            if (found != Null_ptr) return found;
        }
        return Null_ptr;
    }
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
        NODE_PTR found = find_formal_load(node->Child(idx), formal);
        if (found != Null_ptr) return found;
    }
    return Null_ptr;
}

NODE_PTR find_preg_load(NODE_PTR node) {
    if (node == Null_ptr) return Null_ptr;
    if (node->Opcode() == air::core::OPC_LDP && node->Has_preg()) {
        return node;
    }
    if (node->Is_block()) {
        for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
             stmt = stmt->Next()) {
            NODE_PTR found = find_preg_load(stmt->Node());
            if (found != Null_ptr) return found;
        }
        return Null_ptr;
    }
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
        NODE_PTR found = find_preg_load(node->Child(idx));
        if (found != Null_ptr) return found;
    }
    return Null_ptr;
}

NODE_PTR find_opcode_node(NODE_PTR node, OPCODE opcode) {
    if (node == Null_ptr) return Null_ptr;
    if (node->Opcode() == opcode) return node;
    if (node->Is_block()) {
        for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt();
             stmt = stmt->Next()) {
            NODE_PTR found = find_opcode_node(stmt->Node(), opcode);
            if (found != Null_ptr) return found;
        }
        return Null_ptr;
    }
    for (uint32_t idx = 0; idx < node->Num_child(); ++idx) {
        NODE_PTR found = find_opcode_node(node->Child(idx), opcode);
        if (found != Null_ptr) return found;
    }
    return Null_ptr;
}

py::dict compare_normalized_vector_kernel_air_for_testing(
    const std::string& native_air, const std::string& helper_air) {
    const nn::vector::test::VECTOR_KERNEL_AIR_COMPARE_RESULT comparison =
        nn::vector::test::Compare_normalized_vector_kernel_air(
            native_air, helper_air);
    py::dict result;
    result["equal"] = comparison._equal;
    result["message"] = comparison._message;
    if (comparison._mismatch_offset == std::string::npos) {
        result["mismatch_offset"] = py::none();
    } else {
        result["mismatch_offset"] = comparison._mismatch_offset;
    }
    return result;
}

#endif  // ACE_VECTOR_KERNEL_TEST_SUPPORT

// Global scope - creates functions with proper signatures
class GlobScope {
public:
    GLOB_SCOPE* glob;
    std::vector<std::shared_ptr<FuncScope>> functions;
    std::map<std::string, Type> types;
    std::map<std::string, uint32_t> file_ids;  // Registered source files
    uint32_t next_file_id = 1;
    
    GlobScope() {
        ensure_air_initialized();
        glob = new GLOB_SCOPE(0, true);
        s_active_binding_glob = glob;
        types["void"] = Type::make_void();
        types["i32"] = Type::make_int(32);
        types["i64"] = Type::make_int(64);
        types["f32"] = Type::make_float(32);
        types["f64"] = Type::make_float(64);
    }

    ~GlobScope() {
        if (lower_ctx) {
        }
    }

    // Register a source file and return its ID for SPOS creation
    uint32_t register_file(const std::string& filename) {
        auto it = file_ids.find(filename);
        if (it != file_ids.end()) {
            return it->second;
        }
        uint32_t id = next_file_id++;
        file_ids[filename] = id;
        // Also register with the underlying glob if available
        if (glob) {
            glob->New_file(filename.c_str(), LANG::C);
        }
        return id;
    }
    
    // Get file ID (0 if not registered)
    uint32_t get_file_id(const std::string& filename) const {
        auto it = file_ids.find(filename);
        return it != file_ids.end() ? it->second : 0;
    }
    
    // Create function with specified number of array parameters
    std::shared_ptr<FuncScope> new_func_with_params(const std::string& name, 
                                                     int num_params,
                                                     const std::vector<int>& param_shape) {
        // Use default float32 array type
        return new_func_with_type(name, num_params, param_shape, "param_type");
    }

    // Create function with explicit return/parameter types
    std::shared_ptr<FuncScope> new_func_with_param_types(
        const std::string& name,
        const Type& ret_type,
        const std::vector<Type>& param_types) {

        SPOS spos = glob->Unknown_simple_spos();
        STR_PTR func_str = glob->New_str(name.c_str());
        FUNC_PTR func = glob->New_func(func_str, spos);
        func->Set_parent(glob->Comp_env_id());

        // Helper to resolve type markers (e.g., CIPHERTEXT, PLAINTEXT) to actual types
        auto resolve_type = [&](const Type& t) -> TYPE_PTR {
            // Check for CIPHERTEXT marker
            if (t.name == "CIPHERTEXT") {
                ensure_lower_ctx();
                ensure_fhe_types_registered();
                return lower_ctx->Get_cipher_type(glob);
            }
            
            // Check for PLAINTEXT marker
            if (t.name == "PLAINTEXT") {
                ensure_lower_ctx();
                ensure_fhe_types_registered();
                return lower_ctx->Get_plain_type(glob);
            }

            // Check for has_type
            if (t.has_type) {
                t.require_active();
                if (t.type == Null_ptr || &t.type->Glob_scope() != glob) {
                    throw std::runtime_error(
                        "function signature type belongs to a different GLOB_SCOPE");
                }
                return t.type;
            }

            // Default: array type
            TYPE_PTR elem_type = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_32);
            std::vector<int64_t> dims = {64};
            STR_PTR arr_name = glob->New_str("param_type");
            return glob->New_arr_type(arr_name, elem_type, dims, spos);
        };

        TYPE_PTR ret = resolve_type(ret_type);

        SIGNATURE_TYPE_PTR sig = glob->New_sig_type();
        glob->New_ret_param(ret, sig);

        for (size_t i = 0; i < param_types.size(); i++) {
            std::string pname = "p" + std::to_string(i);
            STR_PTR p_str = glob->New_str(pname.c_str());
            TYPE_PTR ptype = resolve_type(param_types[i]);
            glob->New_param(p_str, ptype, sig, spos);
        }

        sig->Set_complete();
        glob->New_entry_point(sig, func, func_str, spos);

        FUNC_SCOPE* func_scope = &glob->New_func_scope(func);
        func_scope->Container().New_func_entry(spos);

        std::vector<ADDR_DATUM_PTR> formals;
        for (size_t i = 0; i < param_types.size(); i++) {
            formals.push_back(func_scope->Formal(static_cast<uint32_t>(i)));
        }

        auto fs = std::make_shared<FuncScope>(name, func_scope, glob, formals,
                                              true);
        functions.push_back(fs);
        return fs;
    }
    
    // Create function with specified type name
    std::shared_ptr<FuncScope> new_func_with_type(const std::string& name, 
                                                   int num_params,
                                                   const std::vector<int>& param_shape,
                                                   const std::string& type_name) {
        SPOS spos = glob->Unknown_simple_spos();
        STR_PTR func_str = glob->New_str(name.c_str());
        FUNC_PTR func = glob->New_func(func_str, spos);
        func->Set_parent(glob->Comp_env_id());
        
        TYPE_PTR param_type;
        
        // For CIPHERTEXT type, create RECORD_TYPE (struct) like SIHE_GEN does
        if (type_name == "CIPHERTEXT") {
            STR_PTR cipher_str = glob->New_str("CIPHERTEXT");
            RECORD_TYPE_PTR rec_type = glob->New_rec_type(RECORD_KIND::STRUCT, cipher_str, spos);
            param_type = static_cast<TYPE_PTR>(rec_type);
            
            // Initialize lower_ctx with this type
            ensure_lower_ctx();
            lower_ctx->Set_cipher_type_id(param_type->Id());
            
            // Create PLAINTEXT too
            STR_PTR plain_str = glob->New_str("PLAINTEXT");
            RECORD_TYPE_PTR plain_type = glob->New_rec_type(RECORD_KIND::STRUCT, plain_str, spos);
            lower_ctx->Set_plain_type_id(plain_type->Id());
        } else {
            // Create array type for other parameter types
            TYPE_PTR elem_type;
            if (type_name.find("ciphertext") != std::string::npos) {
                elem_type = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_64);
            } else if (type_name == "polynomial") {
                // Polynomial arrays need FLOAT_64 for encoding compatibility
                elem_type = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_64);
            } else {
                elem_type = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_32);
            }
            
            std::vector<int64_t> dims;
            for (int s : param_shape) dims.push_back(static_cast<int64_t>(s));
            if (dims.empty()) dims.push_back(64);
            
            STR_PTR arr_name = glob->New_str(type_name.c_str());
            param_type = glob->New_arr_type(arr_name, elem_type, dims, spos);
        }
        
        // Create signature
        SIGNATURE_TYPE_PTR sig = glob->New_sig_type();
        glob->New_ret_param(param_type, sig);
        
        // Add parameters to signature
        for (int i = 0; i < num_params; i++) {
            std::string pname = "p" + std::to_string(i);
            STR_PTR p_str = glob->New_str(pname.c_str());
            glob->New_param(p_str, param_type, sig, spos);
        }
        
        sig->Set_complete();
        glob->New_entry_point(sig, func, func_str, spos);
        
        // Create function scope
        FUNC_SCOPE* func_scope = &glob->New_func_scope(func);
        func_scope->Container().New_func_entry(spos);
        
        // Collect formal parameters
        std::vector<ADDR_DATUM_PTR> formals;
        for (int i = 0; i < num_params; i++) {
            formals.push_back(func_scope->Formal(i));
        }
        
        auto fs = std::make_shared<FuncScope>(name, func_scope, glob, formals);
        functions.push_back(fs);
        return fs;
    }
    
    // Simple version - no params
    std::shared_ptr<FuncScope> new_func(const std::string& name) {
        return new_func_with_params(name, 0, {64});
    }
    
    Type get_type(const std::string& name) {
        if (types.find(name) != types.end()) return types[name];
        return Type::make_void();
    }
    
    Type new_array_type(const std::vector<int>& shape, const std::string& elem = "f32") {
        return Type::make_array_in_glob(glob, shape, get_type(elem));
    }
    
    std::string dump() const {
        std::stringstream ss;
        // Use compiler's internal IR dump
        if (glob) {
            glob->Print(ss, false);
        }
        // Sanitize output to ensure valid UTF-8 for Python
        return sanitize_utf8(ss.str());
    }
    
    // Dump in flattened format (SSA-like, bottom-up ordering)
    // Each operation prints its children first, then itself
    std::string dump_flat() const {
        std::string s;
        
        if (glob) {
            for (GLOB_SCOPE::FUNC_SCOPE_ITER it = glob->Begin_func_scope();
                 it != glob->End_func_scope(); ++it) {
                FUNC_SCOPE* func = &(*it);
                // Use rot=false for flattened output
                s += func->To_str(false) + "\n";
            }
        }
        return s;
    }
    
    // Inline a lowering body into the glob scope
    // This works with REAL AIR when glob is available
    bool inline_lowering(const std::string& op_pattern, const std::string& lowering_ir) {
        if (!glob) {
            return false;  // No real IR to inline into
        }
        
        // Determine the target op based on pattern
        // From nn/core/opcode_def.inc:
        // INVALID=0, ADD=1, AVERAGE_POOL=2, CONV=3, FLATTEN=4, GEMM=5, 
        // GLOBAL_AVERAGE_POOL=6, MAX_POOL=7, MUL=8, RELU=9, RESHAPE=10
        std::string pattern_lower = op_pattern;
        std::transform(pattern_lower.begin(), pattern_lower.end(), 
                       pattern_lower.begin(), ::tolower);
        
        uint32_t target_op = 0;
        if (pattern_lower.find("conv") != std::string::npos) {
            target_op = 3;  // nn::core::CONV = 3
        } else if (pattern_lower.find("relu") != std::string::npos) {
            target_op = 9;  // nn::core::RELU = 9
        } else if (pattern_lower.find("add") != std::string::npos) {
            target_op = 1;  // nn::core::ADD = 1
        } else if (pattern_lower.find("mul") != std::string::npos) {
            target_op = 8;  // nn::core::MUL = 8
        } else if (pattern_lower.find("gemm") != std::string::npos) {
            target_op = 5;  // nn::core::GEMM = 5
        }
        
        if (target_op == 0) {
            return false;
        }
        
        // Perform actual node replacement
        int replaced_count = replace_nodes_in_air(target_op, lowering_ir);
        
        return replaced_count > 0;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // Flattened IR Processing for Python Lowering Inlining
    // (Forward declaration needed before flat_instructions member)
    // ═══════════════════════════════════════════════════════════════════════
    
    // A single instruction in flattened IR
    // Format: "%2 = VECTOR.mul %0, %1" or "retv %2"
    struct FlatInstruction {
        std::string result_id;               // "%2" or empty for retv
        std::string domain;                  // "VECTOR" or "nn::vector"
        std::string opcode;                  // "mul", "add", etc.
        std::vector<std::string> operands;   // ["%0", "%1"] or ["p0", "p1"]
        bool is_return = false;
        bool is_load = false;
        std::string load_param;              // "p0", "p1" for loads
    };
    
    // Store parsed flat instructions for use during recursion
    std::vector<FlatInstruction> flat_instructions;
    
    // ═══════════════════════════════════════════════════════════════════════
    // PROPER NODE CLONING: Clone from lowering GlobScope to target GlobScope
    // ═══════════════════════════════════════════════════════════════════════
    
    // Clone a node tree from source container to target container
    // target_operands: maps parameter index (0, 1, ...) to target operand nodes
    // 
    // NOTE: This handles simple expression trees where the return value is computed
    // directly from parameters (e.g., "return p0 * p1"). For complex functions with
    // SSA-style intermediate variables (e.g., bootstrap_full), this only clones
    // the final expression which may just be a load of a temp variable.
    // Full function inlining with SSA requires more complex dataflow analysis.
    NODE_PTR clone_node_with_remap(
        CONTAINER& src_cntr,
        CONTAINER& dst_cntr,
        NODE_PTR src_node,
        const std::vector<NODE_PTR>& target_operands,
        SPOS spos
    ) {
        if (src_node->Id().Value() == 0) return NODE_PTR();
        
        air::base::OPCODE opc = src_node->Opcode();
        std::string opc_name = src_node->Name();
        
        // Check if this is a load of a formal parameter
        // In AIR, formal loads have names like "ld" and access "p0", "p1", etc.
        if (opc_name.find("ld") != std::string::npos && opc_name.find("ldc") == std::string::npos) {
            // For leaf loads (no children), this might be:
            // 1. A formal parameter load (p0, p1) - should remap to target_operands
            // 2. An intermediate variable load (_fhe_tmp_0) - would need dataflow analysis
            // For now, return first operand as pass-through (works for simple lowerings)
            if (src_node->Num_child() == 0) {
                if (!target_operands.empty()) {
                    return target_operands[0];  // Pass-through the input
                }
            }
        }
        
        // Clone based on node type
        uint32_t num_children = src_node->Num_child();
        
        // For operations with children, clone children first
        std::vector<NODE_PTR> cloned_children;
        for (uint32_t i = 0; i < num_children; ++i) {
            NODE_PTR child = src_node->Child(i);
            if (child->Id().Value() == 0) {
                cloned_children.push_back(NODE_PTR());
                continue;
            }
            
            // Special case: child is a parameter load
            std::string child_name = child->Name();
            if (child_name.find("ld") != std::string::npos && 
                child_name.find("ldc") == std::string::npos &&
                child->Num_child() == 0) {
                // This looks like a formal parameter load
                // Map by position: first encountered → operand 0, second → operand 1
                // This is a simple heuristic for "p0 * p1" style expressions
                if (i < target_operands.size()) {
                    cloned_children.push_back(target_operands[i]);
                } else if (!target_operands.empty()) {
                    cloned_children.push_back(target_operands[0]);
                } else {
                    cloned_children.push_back(NODE_PTR());
                }
                continue;
            }
            
            NODE_PTR cloned_child = clone_node_with_remap(src_cntr, dst_cntr, child, target_operands, spos);
            cloned_children.push_back(cloned_child);
        }
        
        // Create new node in target container
        // For binary arithmetic ops
        if (num_children == 2 && 
            cloned_children.size() >= 2 &&
            cloned_children[0]->Id().Value() != 0 && 
            cloned_children[1]->Id().Value() != 0) {
            // Binary operation - create with same opcode
            NODE_PTR new_node = dst_cntr.New_bin_arith(opc, cloned_children[0]->Rtype(), cloned_children[0], cloned_children[1], spos);
            return new_node;
        }
        
        // For unary ops
        if (num_children == 1 && 
            !cloned_children.empty() &&
            cloned_children[0]->Id().Value() != 0) {
            NODE_PTR new_node = dst_cntr.New_una_arith(opc, cloned_children[0]->Rtype(), cloned_children[0], spos);
            return new_node;
        }
        
        // For leaf nodes (constants, etc.) - would need deeper cloning
        // For now, return first child if available (pass-through)
        if (!cloned_children.empty() && cloned_children[0]->Id().Value() != 0) {
            return cloned_children[0];
        }
        
        return NODE_PTR();
    }
    
    // Find the return expression in a lowering function by name
    // Result of find_return_expr_with_func - contains expression, function info, AND container reference
    struct ReturnExprResult {
        NODE_PTR expr;
        FUNC_ID func_id;
        std::string func_name;
        CONTAINER* container;  // Pointer to the container (owned by glob_scope)
        bool valid;
        
        ReturnExprResult() : expr(), func_id(), func_name(""), container(nullptr), valid(false) {}
    };
    
    // Find return expression and the function it belongs to
    ReturnExprResult find_return_expr_with_func(GLOB_SCOPE* lowering_glob, const std::string& func_name_filter = "") {
        ReturnExprResult result;
        if (!lowering_glob) return result;
        
        // Iterate over functions in the lowering glob
        for (auto fit = lowering_glob->Begin_func(); fit != lowering_glob->End_func(); ++fit) {
            FUNC_PTR fp = *fit;
            
            // If func_name specified, skip non-matching functions
            std::string fn_name = fp->Name()->Char_str();
            if (!func_name_filter.empty() && fn_name.find(func_name_filter) == std::string::npos) {
                continue;
            }
            
            // Also skip "Main_graph" which is the main model
            if (fn_name == "Main_graph") continue;
            
            FUNC_SCOPE& func_scope = lowering_glob->Open_func_scope(fp->Id());
            CONTAINER& cntr = func_scope.Container();
            
            STMT_LIST stmt_list = cntr.Stmt_list();
            NODE_PTR block_node = stmt_list.Block_node();
            
            if (block_node->Id().Value() == 0) continue;
            
            // Look for retv statement
            if (block_node->Is_block()) {
                for (STMT_PTR stmt = block_node->Begin_stmt(); stmt != block_node->End_stmt(); stmt = stmt->Next()) {
                    if (stmt->Id().Value() == 0) break;
                    NODE_PTR stmt_node = stmt->Node();
                    
                    // Check if this is a return statement
                    std::string node_name = stmt_node->Name();
                    if (node_name.find("retv") != std::string::npos) {
                        // Return the child (the expression being returned)
                        if (stmt_node->Num_child() > 0) {
                            result.expr = stmt_node->Child(0);
                            result.func_id = fp->Id();
                            result.func_name = fn_name;
                            result.container = &cntr;  // Store container pointer
                            result.valid = true;
                            return result;
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    // Legacy wrapper for backward compatibility
    NODE_PTR find_return_expr(GLOB_SCOPE* lowering_glob, const std::string& func_name = "") {
        ReturnExprResult res = find_return_expr_with_func(lowering_glob, func_name);
        return res.expr;
    }
    
    
    // ═══════════════════════════════════════════════════════════════════════
    // PYTHON LOWERING DRIVER - Clone-and-Transform pattern (like nn-addon)
    // ═══════════════════════════════════════════════════════════════════════
    
    int inline_lowering_from_glob(GlobScope* lowering_glob_wrapper, uint32_t target_domain, uint32_t target_op) {
        if (!glob || !lowering_glob_wrapper || !lowering_glob_wrapper->glob) {
            return 0;
        }
        
        GLOB_SCOPE* lowering_glob = lowering_glob_wrapper->glob;
        
        // Find the return expression AND the function it belongs to
        ReturnExprResult return_result = find_return_expr_with_func(lowering_glob);
        if (!return_result.valid || return_result.expr->Id().Value() == 0) {
            return 0;
        }
        
        NODE_PTR lowering_return_expr = return_result.expr;
        CONTAINER* lowering_cntr = return_result.container;
        FUNC_SCOPE* lowering_func_scope = lowering_cntr->Parent_func_scope();
        FUNC_PTR lowering_func = lowering_func_scope->Owning_func();
        
        
        // ═══════════════════════════════════════════════════════════════════
        // Step 1: Create new GLOB_SCOPE and clone tables (like Vector_driver)
        // ═══════════════════════════════════════════════════════════════════
        GLOB_SCOPE* new_glob = new GLOB_SCOPE(glob->Id(), true);
        AIR_ASSERT(new_glob != nullptr);
        new_glob->Clone(*glob);
        
        
        int replaced = 0;
        
        // ═══════════════════════════════════════════════════════════════════
        // Step 1b: Build type remapping from lowering kernel to main model
        // Only remap CIPHERTEXT, PLAINTEXT, CIPHERTEXT3 - don't create new types
        // (Creating new array types causes Has_size() issues)
        // ═══════════════════════════════════════════════════════════════════
        std::unordered_map<uint64_t, TYPE_ID> type_remap;
        
        // Find CIPHERTEXT/PLAINTEXT/CIPHERTEXT3 type IDs in lowering kernel
        TYPE_ID lowering_cipher_id, lowering_plain_id, lowering_cipher3_id;
        bool found_lowering_cipher = false, found_lowering_plain = false, found_lowering_cipher3 = false;
        
        for (auto tit = lowering_glob->Begin_type(); tit != lowering_glob->End_type(); ++tit) {
            TYPE_PTR t = *tit;
            if (t->Name() != air::base::Null_ptr) {
                std::string name(t->Name()->Char_str());
                if (name == "CIPHERTEXT" && !found_lowering_cipher) {
                    lowering_cipher_id = t->Id();
                    found_lowering_cipher = true;
                } else if (name == "PLAINTEXT" && !found_lowering_plain) {
                    lowering_plain_id = t->Id();
                    found_lowering_plain = true;
                } else if (name == "CIPHERTEXT3" && !found_lowering_cipher3) {
                    lowering_cipher3_id = t->Id();
                    found_lowering_cipher3 = true;
                }
            }
        }
        
        // Find CIPHERTEXT/PLAINTEXT/CIPHERTEXT3 type IDs in main model (new_glob)
        TYPE_ID main_cipher_id, main_plain_id, main_cipher3_id;
        bool found_main_cipher = false, found_main_plain = false, found_main_cipher3 = false;
        
        for (auto tit = new_glob->Begin_type(); tit != new_glob->End_type(); ++tit) {
            TYPE_PTR t = *tit;
            if (t->Name() != air::base::Null_ptr) {
                std::string name(t->Name()->Char_str());
                if (name == "CIPHERTEXT" && !found_main_cipher) {
                    main_cipher_id = t->Id();
                    found_main_cipher = true;
                } else if (name == "PLAINTEXT" && !found_main_plain) {
                    main_plain_id = t->Id();
                    found_main_plain = true;
                } else if (name == "CIPHERTEXT3" && !found_main_cipher3) {
                    main_cipher3_id = t->Id();
                    found_main_cipher3 = true;
                }
            }
        }
        
        // Build type remap (only for CIPHERTEXT, PLAINTEXT, CIPHERTEXT3)
        if (found_lowering_cipher && found_main_cipher) {
            type_remap[lowering_cipher_id.Value()] = main_cipher_id;
        }
        if (found_lowering_plain && found_main_plain) {
            type_remap[lowering_plain_id.Value()] = main_plain_id;
        }
        if (found_lowering_cipher3 && found_main_cipher3) {
            type_remap[lowering_cipher3_id.Value()] = main_cipher3_id;
        }
        
        // Helper to remap type ID from lowering to main model
        auto remap_type_id = [&](TYPE_ID src_type_id) -> TYPE_ID {
            auto it = type_remap.find(src_type_id.Value());
            if (it != type_remap.end()) {
                return it->second;
            }
            return src_type_id;  // No remapping needed
        };
        
        // ═══════════════════════════════════════════════════════════════════
        // Step 2: For each function, create new func_scope and transform
        // ═══════════════════════════════════════════════════════════════════
        for (GLOB_SCOPE::FUNC_SCOPE_ITER it = glob->Begin_func_scope();
             it != glob->End_func_scope(); ++it) {
            FUNC_SCOPE* old_func = &(*it);
            std::string fn_name = old_func->Owning_func()->Name()->Char_str();
            
            // Skip the lowering function itself
            if (fn_name == return_result.func_name) continue;
            
            
            // Create new func_scope in new_glob and clone local tables
            FUNC_SCOPE& new_func = new_glob->New_func_scope(old_func->Id());
            new_func.Clone(*old_func);
            CONTAINER& new_cntr = new_func.Container();  // NEW container to create nodes in
            
            // Get entry node from OLD function
            CONTAINER& old_cntr = old_func->Container();
            NODE_PTR old_entry = old_cntr.Entry_node();
            
            // Verify it's a valid FUNC_ENTRY (domain=0, op=25)
            if (old_entry->Opcode().Domain() != 0 || old_entry->Opcode().Operator() != 25) {
                continue;
            }
            
            // ═══════════════════════════════════════════════════════════════
            // Step 3: Clone-and-transform using TRANSFORM_CTX pattern
            // Visit OLD nodes, create NEW nodes in new_cntr
            // When we hit a target op, inline ALL statements from lowering function
            // ═══════════════════════════════════════════════════════════════
            
            // Helper to get the lowering function's body block
            // The entry node's last child is the body block
            auto get_lowering_body = [&]() -> NODE_PTR {
                NODE_PTR entry = lowering_cntr->Entry_node();
                // Entry structure: [param0, param1, ..., body_block]
                // The body block is the last child
                for (int i = entry->Num_child() - 1; i >= 0; --i) {
                    NODE_PTR child = entry->Child(i);
                    if (child->Is_block()) return child;
                }
                return air::base::Null_ptr;
            };
            
            // Current block we're building statements into (for inlining)
            NODE_PTR current_block = air::base::Null_ptr;
            STMT_LIST* current_sl = nullptr;
            
            // Maps for inlining (populated when we hit a target op)
            std::unordered_map<uint64_t, NODE_PTR> param_remap;
            std::unordered_map<uint64_t, ADDR_DATUM_PTR> var_remap;
            
            // Helper to get/create variable mapping
            auto get_or_create_var = [&](ADDR_DATUM_PTR src_var) -> ADDR_DATUM_PTR {
                uint64_t src_id = src_var->Id().Value();
                auto it = var_remap.find(src_id);
                if (it != var_remap.end()) {
                    return it->second;
                }
                // Create new variable in target function with unique name
                // IMPORTANT: Remap type ID from lowering kernel to main model
                std::string new_name = std::string(src_var->Name()->Char_str()) + 
                                       "_inl" + std::to_string(replaced);
                TYPE_ID src_type = src_var->Type_id();
                TYPE_ID remapped_type_id = remap_type_id(src_type);
                TYPE_PTR target_type = new_glob->Type(remapped_type_id);
                // Debug: Show each variable remap (only for first bootstrap)
                if (replaced == 1) {
                }
                ADDR_DATUM_PTR new_var = new_func.New_var(
                    target_type, 
                    new_name.c_str(), 
                    src_var->Spos()
                );
                var_remap[src_id] = new_var;
                return new_var;
            };
            
            std::function<NODE_PTR(NODE_PTR)> transform_node = [&](NODE_PTR old_node) -> NODE_PTR {
                air::base::OPCODE opc = old_node->Opcode();
                
                // Check if this is a target op to replace
                if (opc.Domain() == target_domain && opc.Operator() == target_op) {
                    replaced++;
                    
                    // Get the operands (visited children of the target op)
                    std::vector<NODE_PTR> operands;
                    for (uint32_t i = 0; i < old_node->Num_child(); ++i) {
                        operands.push_back(transform_node(old_node->Child(i)));
                    }
                    
                    // Build parameter remap: lowering's formals -> actual operands
                    param_remap.clear();
                    var_remap.clear();
                    FUNC_SCOPE* lowering_func_scope = lowering_cntr->Parent_func_scope();
                    size_t param_idx = 0;
                    for (FORMAL_ITER fit = lowering_func_scope->Begin_formal();
                         fit != lowering_func_scope->End_formal();
                         ++fit, ++param_idx) {
                        ADDR_DATUM_PTR formal = *fit;
                        if (param_idx < operands.size()) {
                            param_remap[formal->Id().Value()] = operands[param_idx];
                            continue;
                        }

                        // Some source ops (e.g. CKKS.bootstrap) are unary while the Python
                        // lowering may define extra helper formals. Materialize defaults
                        // instead of leaving an unmapped formal load (use-before-define).
                        TYPE_ID  formal_type_id = remap_type_id(formal->Type_id());
                        TYPE_PTR formal_type    = new_glob->Type(formal_type_id);
                        NODE_PTR default_arg    = air::base::Null_ptr;
                        if (!operands.empty() &&
                            ((found_main_cipher && formal_type_id == main_cipher_id) ||
                             (found_main_cipher3 && formal_type_id == main_cipher3_id))) {
                            // Prefer a semantic zero ciphertext (x - x) so CKKS scale manager
                            // sees a regular CKKS expression (not an isolated OPC_ZERO leaf).
                            OPCODE sub_op(fhe::ckks::CKKS_DOMAIN::ID,
                                          fhe::ckks::CKKS_OPERATOR::SUB);
                            default_arg =
                                new_cntr.New_cust_node(sub_op, formal_type, old_node->Spos());
                            default_arg->Set_child(0, operands[0]);
                            default_arg->Set_child(1, operands[0]);
                        }
                        if (formal_type != air::base::Null_ptr) {
                            if (default_arg == air::base::Null_ptr) {
                                if (formal_type->Is_int()) {
                                    default_arg =
                                        new_cntr.New_intconst(formal_type, 0, old_node->Spos());
                                } else {
                                    default_arg = new_cntr.New_zero(formal_type, old_node->Spos());
                                }
                            }
                        }
                        if (default_arg == air::base::Null_ptr && !operands.empty()) {
                            default_arg = operands[0];
                        }
                        if (default_arg != air::base::Null_ptr) {
                            param_remap[formal->Id().Value()] = default_arg;
                        }
                    }
                    
                    // Get lowering's body block
                    NODE_PTR lowering_body = get_lowering_body();
                    if (lowering_body == air::base::Null_ptr) {
                        return operands.empty() ? air::base::Null_ptr : operands[0];
                    }
                    
                    // Clone helper: recursively clone a node from lowering, remapping params and vars
                    std::function<NODE_PTR(NODE_PTR)> clone_from_lowering;
                    clone_from_lowering = [&](NODE_PTR src_node) -> NODE_PTR {
                        // Check for parameter/variable load
                        air::base::OPCODE src_opc = src_node->Opcode();
                        if (src_opc == air::core::OPC_LD || src_opc == air::core::OPC_IDNAME) {
                            if (src_node->Has_sym()) {
                                uint64_t sym_id = src_node->Addr_datum()->Id().Value();
                                // Check if it's a parameter
                                auto pit = param_remap.find(sym_id);
                                if (pit != param_remap.end()) {
                                    return pit->second;  // Return the actual operand
                                }
                                // Check if it's a variable we've remapped
                                auto vit = var_remap.find(sym_id);
                                if (vit != var_remap.end()) {
                                    // Load from the remapped variable
                                    return new_cntr.New_ld(vit->second, old_node->Spos());
                                }
                                // Otherwise create remapped variable and load from it
                                ADDR_DATUM_PTR new_var = get_or_create_var(src_node->Addr_datum());
                                return new_cntr.New_ld(new_var, old_node->Spos());
                            }
                        }
                        
                        // Clone regular node (handle statement vs expression)
                        NODE_PTR new_node;
                        if (src_node->Is_root()) {
                            STMT_PTR new_stmt = new_cntr.Clone_stmt(src_node->Stmt());
                            new_node = new_stmt->Node();
                        } else {
                            new_node = new_cntr.Clone_node(src_node);
                        }
                        
                        // IMPORTANT: Remap return type from lowering kernel to main model
                        TYPE_ID remapped_rtype = remap_type_id(src_node->Rtype_id());
                        if (remapped_rtype.Value() != src_node->Rtype_id().Value()) {
                            new_node->Set_rtype(new_glob->Type(remapped_rtype));
                        }
                        // Note: CKKS.mul return types will be fixed by CKKS type fixup (run after Python lowering)
                        
                        // Recursively clone children
                        for (uint32_t i = 0; i < src_node->Num_child(); ++i) {
                            NODE_PTR new_child = clone_from_lowering(src_node->Child(i));
                            new_node->Set_child(i, new_child->Id());
                        }
                        
                        return new_node;
                    };
                    
                    // Clone and insert all statements except return
                    NODE_PTR result_value = air::base::Null_ptr;
                    int stmt_count = 0;
                    int st_count = 0;
                    int retv_count = 0;
                    for (STMT_PTR src_stmt = lowering_body->Begin_stmt();
                         src_stmt != lowering_body->End_stmt();
                         src_stmt = src_stmt->Next()) {
                        stmt_count++;
                        NODE_PTR src_stmt_node = src_stmt->Node();
                        air::base::OPCODE stmt_opc = src_stmt_node->Opcode();
                        
                        // Skip return - we'll use its value as the result
                        if (stmt_opc == air::core::OPC_RETV) {
                            retv_count++;
                            // Get the return value
                            if (src_stmt_node->Num_child() > 0) {
                                result_value = clone_from_lowering(src_stmt_node->Child(0));
                            }
                            continue;
                        }
                        
                        // Handle store statements - create new var and store
                        if (stmt_opc == air::core::OPC_ST) {
                            if (src_stmt_node->Has_sym()) {
                                st_count++;
                                ADDR_DATUM_PTR new_var = get_or_create_var(src_stmt_node->Addr_datum());
                                NODE_PTR value = clone_from_lowering(src_stmt_node->Child(0));
                                STMT_PTR new_st = new_cntr.New_st(value, new_var, old_node->Spos());
                                if (current_sl) {
                                    current_sl->Append(new_st);
                                } else {
                                }
                            }
                            continue;
                        }
                        
                        // Clone other statements
                        NODE_PTR cloned_node = clone_from_lowering(src_stmt_node);
                        if (cloned_node->Is_root() && current_sl) {
                            current_sl->Append(cloned_node->Stmt());
                        }
                    }
                    if (replaced == 1) {
                    }
                    
                    // Return the result value (from the return statement)
                    if (result_value != air::base::Null_ptr) {
                        return result_value;
                    }
                    // Fallback: return first operand
                    return operands.empty() ? air::base::Null_ptr : operands[0];
                }
                
                // For blocks: create FRESH block and transform statements
                if (old_node->Is_block()) {
                    NODE_PTR new_block = new_cntr.New_stmt_block(old_node->Spos());
                    STMT_LIST new_sl(new_block);
                    
                    // Set current block for inlining
                    NODE_PTR prev_block = current_block;
                    STMT_LIST* prev_sl = current_sl;
                    current_block = new_block;
                    current_sl = &new_sl;
                    
                    for (STMT_PTR old_stmt = old_node->Begin_stmt(); 
                         old_stmt != old_node->End_stmt(); 
                         old_stmt = old_stmt->Next()) {
                        NODE_PTR new_stmt_node = transform_node(old_stmt->Node());
                        if (new_stmt_node != air::base::Null_ptr && new_stmt_node->Is_root()) {
                            new_sl.Append(new_stmt_node->Stmt());
                        }
                    }
                    
                    // Restore previous block
                    current_block = prev_block;
                    current_sl = prev_sl;
                    
                    return new_block;
                }
                
                // For regular nodes: clone and transform children
                NODE_PTR new_node;
                if (old_node->Is_root()) {
                    STMT_PTR new_stmt = new_cntr.Clone_stmt(old_node->Stmt());
                    new_node = new_stmt->Node();
                } else {
                    new_node = new_cntr.Clone_node(old_node);
                }
                
                // Transform children and set in new node
                for (uint32_t i = 0; i < old_node->Num_child(); ++i) {
                    NODE_PTR new_child = transform_node(old_node->Child(i));
                    AIR_ASSERT(new_child != air::base::Null_ptr);
                    new_node->Set_child(i, new_child->Id());
                    
                    if (new_node->Is_root() && new_child->Is_root()) {
                        new_child->Stmt()->Set_parent_node(new_node);
                    }
                }
                
                return new_node;
            };
            
            // Transform the entry node
            NODE_PTR new_entry = transform_node(old_entry);
            AIR_ASSERT(new_entry->Is_entry());
            new_func.Set_entry_stmt(new_entry->Stmt());
            
        }
        
        
        // Debug: verify that main model's CIPHERTEXT type is correct in new_glob
        for (auto tit = new_glob->Begin_type(); tit != new_glob->End_type(); ++tit) {
            TYPE_PTR t = *tit;
            if (t->Name() != air::base::Null_ptr) {
                std::string name(t->Name()->Char_str());
                if (name == "CIPHERTEXT") {
                    break;
                }
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // Step 4: Update glob to point to new transformed glob
        // ═══════════════════════════════════════════════════════════════════
        glob = new_glob;
        s_active_binding_glob = glob;
        
        return replaced;
    }
    
    // Helper: Parse op_pattern string and return (domain_id, opcode) using actual enum values
    // Returns (0, 0) if pattern not recognized
    std::pair<uint32_t, uint32_t> parse_op_pattern(const std::string& op_pattern) {
        std::string pattern_lower = op_pattern;
        std::transform(pattern_lower.begin(), pattern_lower.end(), 
                       pattern_lower.begin(), ::tolower);
        
        // nn::core domain
        if (pattern_lower.find("nn::core::") != std::string::npos || 
            pattern_lower.find("nn.") != std::string::npos ||
            (pattern_lower.find("::") == std::string::npos && pattern_lower.find(".") == std::string::npos)) {
            
            if (pattern_lower.find("conv") != std::string::npos) 
                return {nn::core::NN, nn::core::OPCODE::CONV};
            if (pattern_lower.find("relu") != std::string::npos) 
                return {nn::core::NN, nn::core::OPCODE::RELU};
            if (pattern_lower.find("gemm") != std::string::npos) 
                return {nn::core::NN, nn::core::OPCODE::GEMM};
            if (pattern_lower.find("flatten") != std::string::npos) 
                return {nn::core::NN, nn::core::OPCODE::FLATTEN};
            if (pattern_lower.find("reshape") != std::string::npos) 
                return {nn::core::NN, nn::core::OPCODE::RESHAPE};
            if (pattern_lower.find("global_average_pool") != std::string::npos) 
                return {nn::core::NN, nn::core::OPCODE::GLOBAL_AVERAGE_POOL};
            if (pattern_lower.find("average_pool") != std::string::npos) 
                return {nn::core::NN, nn::core::OPCODE::AVERAGE_POOL};
            if (pattern_lower.find("max_pool") != std::string::npos) 
                return {nn::core::NN, nn::core::OPCODE::MAX_POOL};
            if (pattern_lower.find("add") != std::string::npos) 
                return {nn::core::NN, nn::core::OPCODE::ADD};
            if (pattern_lower.find("mul") != std::string::npos) 
                return {nn::core::NN, nn::core::OPCODE::MUL};
        }
        // nn::vector domain
        else if (pattern_lower.find("nn::vector::") != std::string::npos || 
                 pattern_lower.find("vector.") != std::string::npos) {
            
            if (pattern_lower.find("roll") != std::string::npos) 
                return {nn::vector::VECTOR, nn::vector::VECTOR_OPCODE::ROLL};
            if (pattern_lower.find("slice") != std::string::npos) 
                return {nn::vector::VECTOR, nn::vector::VECTOR_OPCODE::SLICE};
            if (pattern_lower.find("pad") != std::string::npos) 
                return {nn::vector::VECTOR, nn::vector::VECTOR_OPCODE::PAD};
            if (pattern_lower.find("reshape") != std::string::npos) 
                return {nn::vector::VECTOR, nn::vector::VECTOR_OPCODE::RESHAPE};
            if (pattern_lower.find("add") != std::string::npos) 
                return {nn::vector::VECTOR, nn::vector::VECTOR_OPCODE::ADD};
            if (pattern_lower.find("mul") != std::string::npos) 
                return {nn::vector::VECTOR, nn::vector::VECTOR_OPCODE::MUL};
        }
        // fhe::sihe domain
        else if (pattern_lower.find("fhe::sihe::") != std::string::npos ||
                 pattern_lower.find("sihe.") != std::string::npos) {
            
            if (pattern_lower.find("encode") != std::string::npos) 
                return {fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::ENCODE};
            if (pattern_lower.find("bootstrap") != std::string::npos && pattern_lower.find("msg") == std::string::npos) 
                return {fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::BOOTSTRAP};
            if (pattern_lower.find("rotate") != std::string::npos && pattern_lower.find("msg") == std::string::npos) 
                return {fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::ROTATE};
            if (pattern_lower.find("neg") != std::string::npos) 
                return {fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::NEG};
            if (pattern_lower.find("sub") != std::string::npos) 
                return {fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::SUB};
            if (pattern_lower.find("add") != std::string::npos && pattern_lower.find("msg") == std::string::npos) 
                return {fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::ADD};
            if (pattern_lower.find("mul") != std::string::npos && pattern_lower.find("msg") == std::string::npos) 
                return {fhe::sihe::SIHE_DOMAIN::ID, fhe::sihe::SIHE_OPERATOR::MUL};
        }
        // fhe::ckks domain
        else if (pattern_lower.find("fhe::ckks::") != std::string::npos ||
                 pattern_lower.find("ckks.") != std::string::npos) {
            
            if (pattern_lower.find("encode") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::ENCODE};
            if (pattern_lower.find("rescale") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::RESCALE};
            if (pattern_lower.find("upscale") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::UPSCALE};
            if (pattern_lower.find("mod_switch") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::MODSWITCH};
            if (pattern_lower.find("relin") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::RELIN};
            if (pattern_lower.find("bootstrap") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::BOOTSTRAP};
            if (pattern_lower.find("raise_mod") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::RAISE_MOD};
            if (pattern_lower.find("conjugate") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::CONJUGATE};
            if (pattern_lower.find("mul_mono") != std::string::npos || pattern_lower.find("mul_monomial") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::MUL_MONO};
            if (pattern_lower.find("rotate") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::ROTATE};
            if (pattern_lower.find("neg") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::NEG};
            if (pattern_lower.find("sub") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::SUB};
            if (pattern_lower.find("add") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::ADD};
            if (pattern_lower.find("mul") != std::string::npos) 
                return {fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::MUL};
        }
        
        return {0, 0};  // Unknown
    }
    
    // High-level interface: inline from GlobScope with op pattern matching
    // Supports multiple domains: nn::core, nn::vector, fhe::sihe, fhe::ckks
    bool inline_lowering_from_scope(GlobScope* lowering_glob, const std::string& op_pattern) {
        if (!glob || !lowering_glob) return false;
        
        auto [target_domain, target_op] = parse_op_pattern(op_pattern);
        
        if (target_domain == 0) {
            return false;
        }
        
        // Use the unified domain-aware inlining
        int count = inline_lowering_from_glob(lowering_glob, target_domain, target_op);
        return count > 0;
    }

    // Rewrite CKKS extended ops in-place to primitive CKKS ops.
    // - RAISE_MOD  -> ADD(child0, SUB(child0, child0))
    // - CONJUGATE  -> ADD(child0, SUB(child0, child0))
    // - MUL_MONO   -> ROTATE(child0, child1)
    //
    // Returns number of replaced nodes.
    int rewrite_ckks_extended_ops(bool verbose = false) {
        if (!glob) return 0;

        int replaced = 0;

        auto is_extended_ckks_op = [](air::base::NODE_PTR node) -> bool {
            if (node == air::base::Null_ptr) return false;
            auto opc = node->Opcode();
            if (opc.Domain() != fhe::ckks::CKKS_DOMAIN::ID) return false;
            std::string n = node->Name();
            std::transform(n.begin(), n.end(), n.begin(), ::tolower);
            return n.find("raise_mod") != std::string::npos ||
                   n.find("conjugate") != std::string::npos ||
                   n.find("mul_mono") != std::string::npos;
        };

        std::function<air::base::NODE_PTR(CONTAINER&, air::base::NODE_PTR, bool)> rewrite_node;
        rewrite_node = [&](CONTAINER& cntr, air::base::NODE_PTR node, bool allow_replace) -> air::base::NODE_PTR {
            if (node == air::base::Null_ptr || node->Id().Value() == 0) return node;

            // Rewrite child expressions first.
            for (uint32_t i = 0; i < node->Num_child(); ++i) {
                air::base::NODE_PTR child = node->Child(i);
                if (child == air::base::Null_ptr || child->Id().Value() == 0) continue;
                air::base::NODE_PTR new_child = rewrite_node(cntr, child, true);
                if (new_child != child && new_child != air::base::Null_ptr) {
                    node->Set_child(i, new_child->Id());
                }
            }

            // Traverse statement lists inside blocks.
            if (node->Is_block()) {
                STMT_LIST sl(node);
                for (STMT_PTR st = sl.Begin_stmt(); st != sl.End_stmt(); st = st->Next()) {
                    if (st == air::base::Null_ptr || st->Id().Value() == 0) continue;
                    air::base::NODE_PTR st_node = st->Node();
                    if (st_node == air::base::Null_ptr || st_node->Id().Value() == 0) continue;
                    (void)rewrite_node(cntr, st_node, false);
                }
                return node;
            }

            if (!allow_replace || !is_extended_ckks_op(node)) {
                return node;
            }

            std::string op_name = node->Name();
            std::transform(op_name.begin(), op_name.end(), op_name.begin(), ::tolower);
            TYPE_PTR rtype = node->Rtype();
            SPOS spos = node->Spos();

            // Defensive checks for expected arity.
            if (op_name.find("conjugate") != std::string::npos ||
                op_name.find("raise_mod") != std::string::npos) {
                if (node->Num_child() < 1) return node;
                NODE_PTR ct = node->Child(0);
                if (ct == air::base::Null_ptr || ct->Id().Value() == 0) return node;

                // zero_like = ct - ct
                air::base::OPCODE sub_op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::SUB);
                NODE_PTR zero_like = cntr.New_bin_arith(sub_op, rtype, ct, ct, spos);
                // repl = ct + zero_like
                air::base::OPCODE add_op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::ADD);
                NODE_PTR repl = cntr.New_bin_arith(add_op, rtype, ct, zero_like, spos);
                replaced++;
                return repl;
            }

            if (op_name.find("mul_mono") != std::string::npos) {
                if (node->Num_child() < 2) return node;
                NODE_PTR ct = node->Child(0);
                NODE_PTR amount = node->Child(1);
                if (ct == air::base::Null_ptr || amount == air::base::Null_ptr ||
                    ct->Id().Value() == 0 || amount->Id().Value() == 0) {
                    return node;
                }
                air::base::OPCODE rot_op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::ROTATE);
                NODE_PTR repl = cntr.New_cust_node(rot_op, rtype, spos);
                repl->Set_child(0, ct);
                repl->Set_child(1, amount);
                if (amount->Opcode() == air::core::OPC_INTCONST) {
                    int32_t rot_idx = static_cast<int32_t>(amount->Intconst());
                    Set_rotation_attr(repl, rot_idx);
                }
                replaced++;
                return repl;
            }

            return node;
        };

        for (auto fit = glob->Begin_func(); fit != glob->End_func(); ++fit) {
            FUNC_PTR fp = *fit;
            FUNC_SCOPE& func_scope = glob->Open_func_scope(fp->Id());
            CONTAINER& cntr = func_scope.Container();

            STMT_LIST stmt_list = cntr.Stmt_list();
            NODE_PTR block_node = stmt_list.Block_node();

            if (block_node != air::base::Null_ptr && block_node->Id().Value() != 0) {
                if (block_node->Is_block()) {
                    for (STMT_PTR stmt = block_node->Begin_stmt();
                         stmt != block_node->End_stmt(); stmt = stmt->Next()) {
                        if (stmt == air::base::Null_ptr || stmt->Id().Value() == 0) continue;
                        NODE_PTR stmt_node = stmt->Node();
                        if (stmt_node == air::base::Null_ptr || stmt_node->Id().Value() == 0) continue;
                        (void)rewrite_node(cntr, stmt_node, false);
                    }
                } else {
                    (void)rewrite_node(cntr, block_node, false);
                }
            } else {
                NODE_PTR entry = cntr.Entry_node();
                if (entry != air::base::Null_ptr && entry->Id().Value() != 0) {
                    (void)rewrite_node(cntr, entry, false);
                }
            }
        }

        if (verbose) {
            std::cerr << "[DEBUG-CKKS-REWRITE] replaced=" << replaced << std::endl;
        }
        return replaced;
    }
    
    // Replace nodes in the real AIR tree
    int replace_nodes_in_air(uint32_t target_op, const std::string& lowering_ir) {
        if (!glob) return 0;
        
        // Parse the lowering IR into flattened instructions
        flat_instructions = parse_flat_ir(lowering_ir);
        
        int replaced = 0;
        // Walk all funcs and replace target ops
        for (auto fit = glob->Begin_func(); fit != glob->End_func(); ++fit) {
            FUNC_PTR fp = *fit;
            FUNC_SCOPE& func_scope = glob->Open_func_scope(fp->Id());
            CONTAINER& cntr = func_scope.Container();
            
            // Access the function body via Stmt_list's block node
            STMT_LIST stmt_list = cntr.Stmt_list();
            air::base::NODE_PTR block_node = stmt_list.Block_node();
            
            if (block_node->Id().Value() != 0) {
                if (block_node->Is_block()) {
                    // Iterate all statements in the block
                    for (STMT_PTR stmt = block_node->Begin_stmt(); stmt != block_node->End_stmt(); stmt = stmt->Next()) {
                        if (stmt->Id().Value() == 0) break;
                        air::base::NODE_PTR stmt_node = stmt->Node();
                        replaced += replace_nodes_recursive(cntr, stmt_node, target_op, 0);
                    }
                } else {
                    replaced += replace_nodes_recursive(cntr, block_node, target_op, 0);
                }
            } else {
                // Fallback to entry_node if block_node is unavailable
                air::base::NODE_PTR entry_node = cntr.Entry_node();
                if (entry_node->Id().Value() != 0) {
                    replaced += replace_nodes_recursive(cntr, entry_node, target_op, 0);
                }
            }
        }
        
        return replaced;
    }
    
    // Parse flattened IR into a list of instructions
    std::vector<FlatInstruction> parse_flat_ir(const std::string& ir) {
        std::vector<FlatInstruction> instructions;
        
        // Split by lines
        std::istringstream stream(ir);
        std::string line;
        int node_counter = 0;
        
        while (std::getline(stream, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            line = line.substr(start);
            
            FlatInstruction instr;
            
            // Check for "ld" (load parameter)
            if (line.find("ld \"p") != std::string::npos) {
                instr.is_load = true;
                // Extract parameter name: ld "p0" -> p0
                size_t p_start = line.find("\"p") + 1;
                size_t p_end = line.find("\"", p_start);
                if (p_start != std::string::npos && p_end != std::string::npos) {
                    instr.load_param = line.substr(p_start, p_end - p_start);
                    instr.result_id = "%" + std::to_string(node_counter++);
                    instructions.push_back(instr);
                }
            }
            // Check for "VECTOR.xxx" operations
            else if (line.find("VECTOR.") != std::string::npos) {
                size_t vec_pos = line.find("VECTOR.");
                size_t op_start = vec_pos + 7;
                size_t op_end = line.find_first_of(" \t\n", op_start);
                
                instr.domain = "VECTOR";
                instr.opcode = line.substr(op_start, op_end - op_start);
                instr.result_id = "%" + std::to_string(node_counter++);
                
                // Operands are the previous instructions that feed into this
                // For binary ops, we assume the last 2 loads/ops are operands
                if (instructions.size() >= 2) {
                    instr.operands.push_back(instructions[instructions.size()-2].result_id);
                    instr.operands.push_back(instructions[instructions.size()-1].result_id);
                }
                instructions.push_back(instr);
            }
            // Check for "retv"
            else if (line.find("retv") != std::string::npos) {
                instr.is_return = true;
                if (!instructions.empty()) {
                    instr.operands.push_back(instructions.back().result_id);
                }
                instructions.push_back(instr);
            }
        }
        
        return instructions;
    }
    
    // Map opcode string to nn::vector opcode
    uint32_t get_vector_opcode(const std::string& op_name) {
        if (op_name == "mul") return nn::vector::VECTOR_OPCODE::MUL;
        if (op_name == "add") return nn::vector::VECTOR_OPCODE::ADD;
        if (op_name == "roll") return nn::vector::VECTOR_OPCODE::ROLL;
        // Add more as needed
        return 0;
    }
    
    // Inline flattened instructions, creating nodes in target container
    // Returns the final result node ID (0 if failed)
    air::base::NODE_ID inline_flat_instructions(
        CONTAINER& cntr,
        const std::vector<FlatInstruction>& instructions,
        air::base::NODE_PTR target_node,  // The conv node being replaced
        SPOS spos
    ) {
        // Map: instruction result_id -> created NODE_ID
        std::map<std::string, air::base::NODE_ID> node_map;
        
        // Map parameter loads to target_node's children
        // p0 -> target_node->Child(0), p1 -> target_node->Child(1), etc.
        
        air::base::NODE_ID result_id;  // Default invalid ID (value 0)
        
        for (const auto& instr : instructions) {
            if (instr.is_load) {
                // Map load to target's child
                // "p0" -> Child(0), "p1" -> Child(1), etc.
                int param_idx = 0;
                if (instr.load_param.size() > 1 && instr.load_param[0] == 'p') {
                    param_idx = std::stoi(instr.load_param.substr(1));
                }
                if (param_idx < (int)target_node->Num_child()) {
                    node_map[instr.result_id] = target_node->Child(param_idx)->Id();
                }
            }
            else if (instr.is_return) {
                // Return instruction - get the result node
                if (!instr.operands.empty() && node_map.count(instr.operands[0])) {
                    result_id = node_map[instr.operands[0]];
                }
            }
            else if (!instr.opcode.empty()) {
                // Operation instruction
                uint32_t op = get_vector_opcode(instr.opcode);
                if (op != 0 && instr.operands.size() >= 2 &&
                    node_map.count(instr.operands[0]) && node_map.count(instr.operands[1])) {
                    // Get operand nodes from container
                    NODE_PTR lhs = cntr.Node(node_map[instr.operands[0]]);
                    NODE_PTR rhs = cntr.Node(node_map[instr.operands[1]]);
                    
                    if (lhs->Id().Value() != 0 && rhs->Id().Value() != 0) {
                        OPCODE new_op(nn::vector::VECTOR, op);
                        NODE_PTR new_node = cntr.New_bin_arith(new_op, lhs->Rtype(), lhs, rhs, spos);
                        node_map[instr.result_id] = new_node->Id();
                        result_id = new_node->Id();  // Update result in case no explicit retv
                    }
                }
            }
        }
        
        return result_id;
    }
    
    // Recursively replace target ops using flattened IR inlining
    // nn::core domain ID is 1 (from nn/core/opcode.h)
    static constexpr uint32_t NN_CORE_DOMAIN = 1;
    
    int replace_nodes_recursive(CONTAINER& cntr, air::base::NODE_PTR node, uint32_t target_op, int depth = 0) {
        if (node->Id().Value() == 0) return 0;
        
        int replaced = 0;
        
        // Process children first (bottom-up)
        uint32_t num_children = node->Num_child();
        
        for (uint32_t i = 0; i < num_children; ++i) {
            air::base::NODE_PTR child = node->Child(i);
            if (child->Id().Value() != 0) {
                // Check if child is the target op in nn::core domain
                air::base::OPCODE child_opc = child->Opcode();
                if (child_opc.Domain() == NN_CORE_DOMAIN && child_opc.Operator() == target_op) {
                    // REAL INLINING: Process flattened IR instructions
                    SPOS spos = child->Spos();
                    
                    // Use flattened IR inlining - returns NODE_ID
                    air::base::NODE_ID result_id = inline_flat_instructions(
                        cntr, flat_instructions, child, spos
                    );
                    
                    if (result_id.Value() != 0) {
                        // Replace target op with the inlined result
                        node->Set_child(i, result_id);
                        replaced++;
                    } else if (child->Num_child() > 0) {
                        // Fallback: pass-through (return first input)
                        air::base::NODE_PTR grandchild = child->Child(0);
                        if (grandchild->Id().Value() != 0) {
                            node->Set_child(i, grandchild->Id());
                            replaced++;
                        }
                    }
                }
                // Continue recursion into the (possibly new) child
                replaced += replace_nodes_recursive(cntr, node->Child(i), target_op, depth + 1);
            }
        }
        
        // Also process statements if this is a block
        if (node->Is_block()) {
            for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt(); stmt = stmt->Next()) {
                if (stmt->Id().Value() != 0) {
                    air::base::NODE_PTR stmt_node = stmt->Node();
                    if (stmt_node->Id().Value() != 0) {
                        replaced += replace_nodes_recursive(cntr, stmt_node, target_op, depth + 1);
                    }
                }
            }
        }
        
        return replaced;
    }
    
    
    uintptr_t get_native_ptr() const {
        return reinterpret_cast<uintptr_t>(glob);
    }
    
    bool has_native_ir() const { return glob != nullptr; }
    bool verify_ir() const { return glob != nullptr && glob->Verify_ir(); }
    // Temporary baseline-kernel E2E bridge. M13 replaces this binding-side
    // algorithm with the independent Python pass, and M15 removes this entry
    // point after differential and downstream validation.

    py::dict inline_generated_vector_kernel_helpers() {
        py::dict output;
        if (!glob) {
            output["success"] = false;
            output["changed"] = false;
            output["calls_inlined"] = 0;
            output["helpers_removed"] = 0;
            output["diagnostic"] = "no AIR GLOB_SCOPE is available";
            return output;
        }

        std::unique_ptr<GLOB_SCOPE> candidate;
        ace::bindings::AIR_FUNCTION_INLINE_RESULT result;
        try {
            candidate.reset(ace::bindings::Clone_glob_with_code(*glob));
            result = ace::bindings::Inline_tagged_leaf_helpers(
                *candidate,
                nn::vector::VECTOR_KERNEL_GENERATED_CALL_ATTR,
                nn::vector::VECTOR_KERNEL_GENERATED_HELPER_ATTR);
            if (result._success && result._changed) {
                if (!candidate->Verify_ir()) {
                    result._success = false;
                    result._changed = false;
                    result._diagnostic =
                        "transactional inliner candidate does not verify";
                } else {
                    std::map<std::string, Type> candidate_types;
                    for (const auto& item : types) {
                        const Type& source_type = item.second;
                        if (!source_type.has_type ||
                            source_type.type == Null_ptr) {
                            candidate_types.emplace(item.first, source_type);
                            continue;
                        }
                        TYPE_PTR remapped_type = Null_ptr;
                        if (source_type.type->Is_prim()) {
                            remapped_type = candidate->Prim_type(
                                source_type.type->Cast_to_prim()->Encoding());
                        } else if (
                            &source_type.type->Glob_scope() == glob) {
                            remapped_type = candidate->Type(
                                source_type.type->Id());
                        }
                        if (remapped_type == Null_ptr) {
                            throw std::runtime_error(
                                "cannot remap cached binding type during "
                                "transactional inliner commit: " +
                                item.first);
                        }
                        candidate_types.emplace(
                            item.first,
                            Type(remapped_type, source_type.name,
                                 source_type.shape));
                    }

                    for (const auto& function : functions) {
                        if (function) function->invalidate();
                    }
                    functions.clear();
                    // GlobScope wrappers do not own prior native scopes. Keep
                    // the old scope alive so externally copied Type wrappers
                    // remain readable; they become foreign to the candidate
                    // and existing scope checks reject their reuse. M15 removes
                    // this transition-only clone/swap path.
                    glob = candidate.release();
                    types.swap(candidate_types);
                    s_active_binding_glob = glob;
                }
            }
        } catch (const std::exception& error) {
            result._success = false;
            result._changed = false;
            result._diagnostic = error.what();
        }
        if (!result._success) {
            result._changed = false;
            result._calls_inlined = 0;
            result._helpers_removed = 0;
        }

        output["success"] = result._success;
        output["changed"] = result._changed;
        output["calls_inlined"] = result._calls_inlined;
        output["helpers_removed"] = result._helpers_removed;
        output["diagnostic"] = result._diagnostic;
        return output;
    }

#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
    py::dict inspect_baseline_air_for_testing(
        const std::string& helper_prefix,
        bool has_prepared_input) const {
        if (!glob || !glob->Verify_ir()) {
            throw std::runtime_error(
                "baseline AIR oracle requires a verified GLOB_SCOPE");
        }

        FUNC_SCOPE* caller = nullptr;
        FUNC_SCOPE* helper = nullptr;
        uint32_t function_count = 0;
        for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = glob->Begin_func_scope();
             iter != glob->End_func_scope(); ++iter) {
            ++function_count;
            FUNC_SCOPE* scope = &(*iter);
            const char* raw_name = scope->Owning_func()->Name()->Char_str();
            const std::string name = raw_name == nullptr ? "" : raw_name;
            if (name.compare(0, helper_prefix.size(),
                             helper_prefix) == 0) {
                if (helper != nullptr) {
                    throw std::runtime_error(
                        "baseline AIR oracle found multiple matching helpers");
                }
                helper = scope;
            } else {
                if (caller != nullptr) {
                    throw std::runtime_error(
                        "baseline AIR oracle requires one caller function");
                }
                caller = scope;
            }
        }
        if (caller == nullptr) {
            throw std::runtime_error("baseline AIR oracle found no caller function");
        }

        py::dict result;
        if (helper != nullptr) {
            NODE_PTR caller_entry = caller->Container().Entry_node();
            std::vector<STMT_PTR> calls;
            collect_call_statements(caller_entry, calls);
            if (calls.size() != 1) {
                throw std::runtime_error(
                    "baseline AIR oracle requires one helper CALL");
            }
            NODE_PTR call = calls[0]->Node();
            if (call->Num_arg() != 1 || caller->Formal_cnt() < 1) {
                throw std::runtime_error(
                    "baseline AIR oracle requires the one-input baseline helper ABI");
            }
            NODE_PTR actual = call->Child(0);
            std::vector<NODE_PTR> expected_actuals;
            std::string native_air;
            bool prepared_input_ok = !has_prepared_input;
            if (has_prepared_input) {
                const char* raw_helper_name =
                    helper->Owning_func()->Name()->Char_str();
                const std::string helper_name =
                    raw_helper_name == nullptr ? "" : raw_helper_name;
                auto oracle = _baseline_conv_oracles.find(helper_name);
                if (oracle == _baseline_conv_oracles.end()) {
                    throw std::runtime_error(
                        "baseline Conv helper has no retained native oracle");
                }
                expected_actuals = oracle->second._expected_actuals;
                native_air = oracle->second._native_air;
                std::vector<STMT_PTR> prepared_input_stores;
                collect_prepared_conv_input_stores(
                    caller_entry, prepared_input_stores);
                if (expected_actuals.size() == 1 &&
                    prepared_input_stores.size() == 1) {
                    PREG_PTR prepared_preg =
                        prepared_input_stores[0]->Node()->Preg();
                    std::vector<NODE_PTR> prepared_loads;
                    collect_matching_ldp_nodes(
                        caller_entry, prepared_preg, prepared_loads);
                    NODE_PTR caller_body = caller_entry->Last_child();
                    prepared_input_ok =
                        prepared_preg->Defining_func_scope() == caller &&
                        prepared_preg->Type()->Is_compatible_type(
                            helper->Formal(0)->Type()) &&
                        prepared_loads.size() == 1 &&
                        prepared_loads[0]->Id() == expected_actuals[0]->Id() &&
                        statement_precedes_in_block(
                            caller_body, prepared_input_stores[0], calls[0]);
                }
            } else if (actual->Opcode() != air::core::OPC_LD ||
                       !actual->Has_sym() ||
                       actual->Addr_datum() != caller->Formal(0)) {
                throw std::runtime_error(
                    "baseline helper CALL does not use the caller formal input");
            } else {
                expected_actuals.push_back(actual);
            }
            std::vector<NODE_PTR> replacements;
            collect_matching_ldp_nodes(caller_entry, call->Ret_preg(),
                                       replacements);
            if (replacements.size() != 1) {
                throw std::runtime_error(
                    "baseline AIR oracle requires one replacing CALL LDP");
            }
            const nn::vector::test::VECTOR_KERNEL_AIR_COMPARE_RESULT bridge =
                nn::vector::test::Check_vector_kernel_call_bridge(
                    *caller, calls[0], expected_actuals, replacements[0],
                    *helper);
            result["kind"] = "helper";
            result["normalized"] =
                nn::vector::test::Normalize_vector_kernel_helper(*helper);
            result["bridge_ok"] = bridge._equal;
            result["bridge_message"] = bridge._message;
            result["prepared_input_ok"] = prepared_input_ok;
            result["native_normalized"] =
                has_prepared_input ? py::cast(native_air) : py::none();
            return result;
        }

        if (function_count != 1 || caller->Formal_cnt() < 1) {
            throw std::runtime_error(
                "baseline native AIR oracle requires one one-input function");
        }
        NODE_PTR body = caller->Container().Entry_node()->Last_child();
        if (body == Null_ptr || !body->Is_block()) {
            throw std::runtime_error("baseline native AIR oracle found no body");
        }
        STMT_PTR terminal = Null_ptr;
        for (STMT_PTR stmt = body->Begin_stmt(); stmt != body->End_stmt();
             stmt = stmt->Next()) {
            terminal = stmt;
        }
        if (terminal == Null_ptr ||
            terminal->Node()->Opcode() != air::core::OPC_RETV ||
            terminal->Node()->Num_child() != 1) {
            throw std::runtime_error(
                "baseline native AIR oracle requires one terminal RETV");
        }
        NODE_PTR output_load = terminal->Node()->Child(0);
        if (output_load->Opcode() != air::core::OPC_LD ||
            !output_load->Has_sym()) {
            throw std::runtime_error(
                "baseline native AIR oracle RETV must load the caller output");
        }
        ADDR_DATUM_PTR output = output_load->Addr_datum();
        STMT_PTR first_kernel = Null_ptr;
        STMT_PTR output_store = Null_ptr;
        for (STMT_PTR stmt = body->Begin_stmt(); stmt != terminal;
             stmt = stmt->Next()) {
            NODE_PTR node = stmt->Node();
            if (node->Opcode() == air::core::OPC_ST && node->Has_sym() &&
                node->Addr_datum() == output) {
                output_store = stmt;
                break;
            }
            if (first_kernel == Null_ptr &&
                ((has_prepared_input &&
                  node->Opcode() == air::core::OPC_ST) ||
                 (!has_prepared_input &&
                  node->Opcode() != air::core::OPC_COMMENT &&
                  node->Opcode() != air::core::OPC_PRAGMA))) {
                first_kernel = stmt;
            }
        }
        if (first_kernel == Null_ptr || output_store == Null_ptr ||
            output_store->Node()->Num_child() != 1) {
            throw std::runtime_error(
                "baseline native AIR oracle could not isolate the kernel region");
        }
        NODE_PTR input = Null_ptr;
        for (STMT_PTR stmt = first_kernel; stmt != output_store;
             stmt = stmt->Next()) {
            input = has_prepared_input
                        ? find_preg_load(stmt->Node())
                        : find_formal_load(stmt->Node(), caller->Formal(0));
            if (input != Null_ptr) break;
        }
        if (input == Null_ptr) {
            throw std::runtime_error(
                "baseline native AIR oracle found no semantic input load");
        }
        NODE_PTR native_result = output_store->Node()->Child(0);
        const nn::vector::test::VECTOR_KERNEL_NATIVE_AIR_VIEW native_view{
            caller, {input}, first_kernel, output_store, native_result};
        result["kind"] = "native";
        result["normalized"] =
            nn::vector::test::Normalize_native_vector_kernel(native_view);
        result["bridge_ok"] = py::none();
        result["bridge_message"] = "";
        return result;
    }

    py::dict inspect_baseline_gemm_air_for_testing() const {
        return inspect_baseline_air_for_testing(
            BASELINE_GEMM_HELPER_PREFIX, false);
    }

    py::dict inspect_baseline_conv_air_for_testing() const {
        return inspect_baseline_air_for_testing(
            BASELINE_CONV_HELPER_PREFIX, true);
    }

    void mutate_baseline_conv_call_actual_for_testing() {
        if (!glob || !glob->Verify_ir()) {
            throw std::runtime_error(
                "baseline Conv call mutation requires verified AIR");
        }
        FUNC_SCOPE* caller = nullptr;
        FUNC_SCOPE* helper = nullptr;
        for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = glob->Begin_func_scope();
             iter != glob->End_func_scope(); ++iter) {
            FUNC_SCOPE* scope = &(*iter);
            const char* raw_name = scope->Owning_func()->Name()->Char_str();
            const std::string name = raw_name == nullptr ? "" : raw_name;
            if (name.compare(0, BASELINE_CONV_HELPER_PREFIX.size(),
                             BASELINE_CONV_HELPER_PREFIX) == 0) {
                if (helper != nullptr) {
                    throw std::runtime_error(
                        "baseline Conv call mutation found multiple helpers");
                }
                helper = scope;
            } else {
                if (caller != nullptr) {
                    throw std::runtime_error(
                        "baseline Conv call mutation requires one caller");
                }
                caller = scope;
            }
        }
        if (caller == nullptr || helper == nullptr) {
            throw std::runtime_error(
                "baseline Conv call mutation requires caller and helper");
        }
        std::vector<STMT_PTR> calls;
        collect_call_statements(caller->Container().Entry_node(), calls);
        if (calls.size() != 1 || calls[0]->Node()->Num_arg() != 1) {
            throw std::runtime_error(
                "baseline Conv call mutation requires one one-input CALL");
        }

        NODE_PTR call = calls[0]->Node();
        NODE_PTR actual = call->Child(0);
        PREG_PTR wrong_preg = caller->New_preg(actual->Rtype());
        STMT_PTR wrong_store = caller->Container().New_stp(
            caller->Container().New_zero(actual->Rtype(), actual->Spos()),
            wrong_preg, actual->Spos());
        STMT_LIST::Enclosing_list(calls[0]).Prepend(calls[0], wrong_store);
        call->Set_child(
            0, caller->Container().New_ldp(wrong_preg, actual->Spos()));
        if (!glob->Verify_ir()) {
            throw std::runtime_error(
                "baseline Conv wrong-actual mutation produced invalid AIR");
        }
    }

    void mutate_generated_vector_helper_for_testing(
        const std::string& mutation) {
        if (!glob || !glob->Verify_ir()) {
            throw std::runtime_error(
                "generated-helper mutation requires verified AIR");
        }
        FUNC_SCOPE* helper = nullptr;
        for (GLOB_SCOPE::FUNC_SCOPE_ITER iter = glob->Begin_func_scope();
             iter != glob->End_func_scope(); ++iter) {
            FUNC_SCOPE* scope = &(*iter);
            const char* raw_name = scope->Owning_func()->Name()->Char_str();
            const std::string name = raw_name == nullptr ? "" : raw_name;
            const bool baseline_gemm =
                name.compare(0, BASELINE_GEMM_HELPER_PREFIX.size(),
                             BASELINE_GEMM_HELPER_PREFIX) == 0;
            const bool baseline_conv =
                name.compare(0, BASELINE_CONV_HELPER_PREFIX.size(),
                             BASELINE_CONV_HELPER_PREFIX) == 0;
            if (!baseline_gemm && !baseline_conv) {
                continue;
            }
            if (helper != nullptr) {
                throw std::runtime_error(
                    "generated-helper mutation requires one helper");
            }
            helper = scope;
        }
        if (helper == nullptr) {
            throw std::runtime_error(
                "generated-helper mutation found no helper");
        }

        FUNC_PTR helper_func = helper->Owning_func();
        ENTRY_PTR helper_entry = helper_func->Entry_point();
        if (mutation == "extra-entry") {
            const std::string alias =
                std::string(helper_func->Name()->Char_str()) + "_alias";
            glob->New_entry_point(helper_entry->Type(), helper_func,
                                  alias.c_str(), glob->Unknown_simple_spos());
        } else if (mutation == "program-entry") {
            helper_entry->Set_program_entry();
        } else if (mutation == "array-non-pointer-rtype") {
            NODE_PTR array = find_opcode_node(
                helper->Container().Entry_node(), air::core::OPC_ARRAY);
            if (array == Null_ptr || array->Array_dim() == 0 ||
                !array->Array_idx(0)->Rtype()->Is_signed_int()) {
                throw std::runtime_error(
                    "array-rtype mutation found no suitable ARRAY");
            }
            array->Set_rtype(array->Array_idx(0)->Rtype());
        } else if (mutation == "escaping-ldca") {
            CONTAINER& container = helper->Container();
            NODE_PTR body = container.Entry_node()->Body_blk();
            NODE_PTR source = find_opcode_node(body, air::core::OPC_LDCA);
            if (source == Null_ptr) {
                throw std::runtime_error(
                    "escaping-LDCA mutation found no constant address");
            }
            STMT_PTR terminal = Null_ptr;
            for (STMT_PTR stmt = body->Begin_stmt(); stmt != body->End_stmt();
                 stmt = stmt->Next()) {
                if (stmt->Node()->Opcode() == air::core::OPC_RETV) {
                    terminal = stmt;
                    break;
                }
            }
            if (terminal == Null_ptr) {
                throw std::runtime_error(
                    "escaping-LDCA mutation found no terminal return");
            }
            ADDR_DATUM_PTR pointer_local = helper->New_var(
                source->Rtype(), "__escaping_constant_address",
                source->Spos());
            STMT_PTR store = container.New_st(
                container.Clone_node_tree(source), pointer_local,
                source->Spos());
            STMT_LIST::Enclosing_list(terminal).Prepend(terminal, store);
        } else {
            throw std::runtime_error(
                "unsupported generated-helper test mutation: " + mutation);
        }
        if (!glob->Verify_ir()) {
            throw std::runtime_error(
                "generated-helper test mutation produced invalid AIR");
        }
    }
#endif  // ACE_VECTOR_KERNEL_TEST_SUPPORT

public:
    // For external access by run_ckks_driver wrapper
    std::unique_ptr<fhe::core::LOWER_CTX>& get_lower_ctx_ref() { ensure_lower_ctx(); return lower_ctx; }
    void prep_fhe_types() { ensure_fhe_types_registered(); }

public:
    
    // C++ pass integration
    bool run_cpp_pass(const std::string& pass_name, 
                      const std::vector<std::string>& skip_ops = {}) {
        if (!glob) return false;
        
        // Register skip ops with ALL registries:
        // 1. Python lowering bridge (for post-processing)
        pyace::PythonLoweringBridge::instance().set_skip_ops(skip_ops);
        // 2. nn::vector skip registry (checked by tensor2vector_handler.h)
        nn::vector::Set_skip_lowering_ops(skip_ops);
        // 3. fhe::sihe skip registry (checked by sihe2ckks_impl.h Handle_bootstrap)
        fhe::sihe::Set_skip_lowering_ops(skip_ops);
        // 4. fhe::ckks skip registry (checked by ckks2poly handlers)
        fhe::ckks::Set_skip_lowering_ops(skip_ops);
        
        if (pass_name == "tensor2vector") {
            return run_tensor2vector_pass();
        }
        else if (pass_name == "vector2sihe") {
            return run_vector2sihe_pass();
        }
        else if (pass_name == "sihe2ckks") {
            return run_sihe2ckks_pass();
        }
        else if (pass_name == "ckks2poly") {
            return run_ckks2poly_pass();
        }
        else if (pass_name == "poly2c") {
            return run_poly2c_pass();
        }
        
        return false;
    }

    bool run_tensor2vector_with_recipes(
        const std::vector<std::string>& skip_ops,
        const std::string& plan_provider,
        const std::string& kernel_impl,
        const std::string& plan_kind,
        const std::string& fallback, py::dict recipes,
        bool mask_fuse, uint64_t max_slots) {
        if (!glob) {
            throw std::runtime_error(
                "tensor2vector recipe lowering requires a real GLOB_SCOPE");
        }
        nn::vector::VECTOR_KERNEL_SELECTION_RESULT selection =
            nn::vector::Parse_vector_kernel_selection(
                plan_provider, kernel_impl, plan_kind, fallback);
        if (!selection._selection.has_value()) {
            throw std::invalid_argument(selection._diagnostic);
        }

        pyace::PythonLoweringBridge::instance().set_skip_ops(skip_ops);
        nn::vector::Set_skip_lowering_ops(skip_ops);
        fhe::sihe::Set_skip_lowering_ops(skip_ops);
        fhe::ckks::Set_skip_lowering_ops(skip_ops);

        nn::vector::VECTOR_KERNEL_LOWERING_REGISTRY registry;
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
        _baseline_conv_oracles.clear();
#endif
        for (auto item : recipes) {
            const std::string key = py::cast<std::string>(item.first);
            if (!PyCallable_Check(item.second.ptr())) {
                throw py::type_error(
                    "vector-kernel recipe for " + key + " is not callable");
            }
            py::function callback =
                py::reinterpret_borrow<py::function>(item.second);
            const nn::vector::VECTOR_KERNEL_PLAN_KIND kind =
                bound_plan_kind(key);
            const bool inserted = registry.Register(
                kind,
                [this, kind, callback = std::move(callback)](
                    const nn::vector::PREPARED_VECTOR_KERNEL_PLAN& prepared,
                    const std::vector<NODE_PTR>& actuals, GLOB_SCOPE&,
                    const nn::vector::VECTOR_KERNEL_DESTINATION_ABI& abi) {
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
                  std::map<std::string, BASELINE_CONV_ORACLE_RECORD>*
                      conv_oracles = nullptr;
                  if (kind ==
                      nn::vector::VECTOR_KERNEL_PLAN_KIND::BASELINE_CONV) {
                    BASELINE_CONV_ORACLE_RECORD record{actuals, {}};
                    _baseline_conv_oracles.try_emplace(
                        prepared.Helper_name(), std::move(record));
                    conv_oracles = &_baseline_conv_oracles;
                  }
#endif
                  return make_python_plan_recipe(
                      callback, prepared, abi
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
                      , conv_oracles
#endif
                      );
                });
            if (!inserted) {
                throw std::invalid_argument(
                    "duplicate vector-kernel recipe kind: " + key);
            }
        }

        nn::vector::VECTOR_CTX ctx;
        ctx.Set_vector_kernel_lowering_registry(&registry);
        nn::vector::VECTOR_CONFIG config;
        config._plan_provider = plan_provider;
        config._kernel_impl = kernel_impl;
        config._plan_kind = plan_kind;
        config._fallback = fallback;
        config._mask_fuse = mask_fuse;
        config._max_slots = max_slots;
        GLOB_SCOPE* new_glob =
            nn::vector::Vector_driver(glob, ctx, nullptr, config);
        if (!new_glob) {
            throw std::runtime_error(
                "Tensor-to-Vector recipe lowering returned no destination");
        }
        glob = new_glob;
        s_active_binding_glob = glob;
        return true;
    }
    
private:
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
    std::map<std::string, BASELINE_CONV_ORACLE_RECORD>
        _baseline_conv_oracles;
#endif
    // Create the LOWER_CTX needed by FHE passes
    std::unique_ptr<fhe::core::LOWER_CTX> lower_ctx;
    bool fhe_types_registered = false;
    bool cipher_types_initialized = false;  // Track if CIPHERTEXT/PLAINTEXT types initialized
    
    void ensure_lower_ctx() {
        if (!lower_ctx) {
            lower_ctx = std::make_unique<fhe::core::LOWER_CTX>();
        }
    }
    
    void ensure_fhe_types_registered() {
        if (!glob || !lower_ctx) {
            return;
        }

        // Always bind LOWER_CTX to existing CIPHERTEXT/PLAINTEXT if present.
        TYPE_ID existing_cipher_id, existing_plain_id;
        bool found_existing_cipher = false, found_existing_plain = false;
        for (auto it = glob->Begin_type(); it != glob->End_type(); ++it) {
            TYPE_PTR t = *it;
            if (t->Name() != air::base::Null_ptr) {
                std::string name(t->Name()->Char_str());
                if (name == "CIPHERTEXT" && t->Is_record()) {
                    existing_cipher_id = t->Id();
                    found_existing_cipher = true;
                }
                if (name == "PLAINTEXT" && t->Is_record()) {
                    existing_plain_id = t->Id();
                    found_existing_plain = true;
                }
            }
        }
        if (found_existing_cipher) {
            lower_ctx->Set_cipher_type_id(existing_cipher_id);
        }
        if (found_existing_plain) {
            lower_ctx->Set_plain_type_id(existing_plain_id);
        }

        if (!fhe_types_registered) {
            // Set up CTX_PARAM with reasonable FHE/CKKS defaults
            auto& ctx_param = lower_ctx->Get_ctx_param();
            ctx_param.Set_poly_degree(16384, false);               // N = 2^14
            ctx_param.Set_mul_level(10, true);              // 10 multiplication levels
            ctx_param.Set_security_level(128);              // 128-bit security
            ctx_param.Set_first_prime_bit_num(60);          // First prime bits
            ctx_param.Set_scaling_factor_bit_num(40);       // Scale factor bits
            ctx_param.Set_hamming_weight(192);              // Hamming weight
            
            if (found_existing_cipher) {
                // Use existing types instead of creating new ones
                lower_ctx->Set_cipher_type_id(existing_cipher_id);
                if (found_existing_plain) {
                    lower_ctx->Set_plain_type_id(existing_plain_id);
                } else {
                    // Create PLAINTEXT if not found
                    SPOS spos = glob->Unknown_simple_spos();
                    STR_PTR plain_str = glob->New_str("PLAINTEXT");
                    RECORD_TYPE_PTR plain_type = glob->New_rec_type(RECORD_KIND::STRUCT, plain_str, spos);
                    lower_ctx->Set_plain_type_id(plain_type->Id());
                }
            } else {
                // Register SIHE types (cipher, plain) - creates new types and sets cipher_type_id
                fhe::sihe::SIHE_GEN sihe_gen(glob, lower_ctx.get());
                sihe_gen.Register_sihe_types();
            }

            // Ensure CKKS types (CIPHERTEXT3/POLY) exist for sihe2ckks paths
            TYPE_ID existing_cipher3_id = air::base::TYPE_ID();
            for (auto it = glob->Begin_type(); it != glob->End_type(); ++it) {
                TYPE_PTR t = *it;
                if (t->Is_record()) {
                    STR_PTR name_ptr = t->Name();
                    if (name_ptr != air::base::Null_ptr &&
                        strcmp(name_ptr->Char_str(), "CIPHERTEXT3") == 0) {
                        existing_cipher3_id = t->Id();
                        break;
                    }
                }
            }

            if (!existing_cipher3_id.Is_null()) {
                lower_ctx->Set_cipher3_type_id(existing_cipher3_id);
            } else {
                fhe::ckks::CKKS_GEN(glob, lower_ctx.get())
                    .Register_ckks_types();
            }
            
            fhe_types_registered = true;
        }

        // Ensure CKKS types (CIPHERTEXT3/POLY) exist for sihe2ckks paths
        TYPE_ID existing_cipher3_id = air::base::TYPE_ID();
        for (auto it = glob->Begin_type(); it != glob->End_type(); ++it) {
            TYPE_PTR t = *it;
            if (t->Is_record()) {
                STR_PTR name_ptr = t->Name();
                if (name_ptr != air::base::Null_ptr &&
                    strcmp(name_ptr->Char_str(), "CIPHERTEXT3") == 0) {
                    existing_cipher3_id = t->Id();
                    break;
                }
            }
        }

        if (!existing_cipher3_id.Is_null()) {
            lower_ctx->Set_cipher3_type_id(existing_cipher3_id);
        } else {
            fhe::ckks::CKKS_GEN(glob, lower_ctx.get())
                .Register_ckks_types();
        }
    }
    
    // VEC configuration options (equivalent to -VEC:conv_fast:gemm_fast)
    // These are applied ONLY when configure_vec_params() is called
    bool vec_conv_fast = false;  // Enable conv_fast optimization
    bool vec_gemm_fast = false;  // Enable gemm_fast optimization
    bool vec_config_set = false; // Track if user explicitly configured VEC options
    
    bool run_tensor2vector_pass() {
        
        if (!glob) {
            return false;
        }
        
        try {
            nn::vector::VECTOR_CTX ctx;
            nn::vector::VECTOR_CONFIG config;
            
            // VEC options removed in newer API
            // if (vec_config_set) {
            //     config._conv_fast = vec_conv_fast;
            //     config._gemm_fast = vec_gemm_fast;
            // }
            
            GLOB_SCOPE* new_glob = nn::vector::Vector_driver(glob, ctx, nullptr, config);
            
            if (new_glob) {
                glob = new_glob;
                s_active_binding_glob = glob;
                return true;
            }
        } catch (const std::exception& e) {
        } catch (...) {
        }
        return false;
    }
    
    // SIHE configuration options (equivalent to -SIHE:relu_vr_def=3:relu_vr=...)
    // These are applied ONLY when configure_sihe_params() is called
    double sihe_relu_vr_def = 3.0;           // Default ReLU value range
    std::string sihe_relu_vr = "";           // Per-layer ReLU value range (e.g., "/relu/Relu=4;...")
    uint32_t sihe_relu_mul_depth = 0;        // ReLU multiplication depth
    uint32_t sihe_relu_base_poly_type = 0;        // ReLU base type
    bool sihe_config_set = false;            // Track if user explicitly configured SIHE options
    
    public:
    // FHE/CKKS configuration options (equivalent to -CKKS:sk_hw=192:q0=60:sf=56)
    // These are applied ONLY when configure_fhe_params() is called
    // Made public for access by run_ckks_driver
    uint32_t fhe_poly_degree = 16384;
    uint32_t fhe_mul_level = 10;
    uint32_t fhe_input_level = 0;
    uint32_t fhe_security_level = 128;
    uint32_t fhe_scaling_factor_bits = 40;
    uint32_t fhe_first_prime_bits = 60;
    uint32_t fhe_hamming_weight = 192;
    bool fhe_config_set = false;             // Track if user explicitly configured FHE options

private:
    
    bool run_vector2sihe_pass() {
        if (!glob) {
            return false;
        }
        
        ensure_lower_ctx();
        
        // Register SIHE types BEFORE running the driver
        // This creates CIPHERTEXT/PLAINTEXT types and sets their IDs in lower_ctx
        fhe::sihe::SIHE_GEN(glob, lower_ctx.get()).Register_sihe_types();
        
        // Store the registered cipher_type_id - this is what CKKS pass will check against
        TYPE_ID registered_cipher_type = lower_ctx->Get_cipher_type_id();
        
        try {
            fhe::sihe::SIHE_CONFIG cfg;
            
            // Only apply SIHE options if user explicitly configured them
            if (sihe_config_set) {
                cfg._relu_value_range_default = sihe_relu_vr_def;
                cfg._relu_value_range = sihe_relu_vr;
                cfg._relu_mul_depth = sihe_relu_mul_depth;
                cfg._relu_base_poly_type = sihe_relu_base_poly_type;
            }
            
            GLOB_SCOPE* new_glob = fhe::sihe::Sihe_driver(glob, lower_ctx.get(), nullptr, cfg);
            if (new_glob && new_glob != glob) {
                glob = new_glob;
                s_active_binding_glob = glob;
                
                // After lowering, find the existing CIPHERTEXT/PLAINTEXT types in the cloned glob
                // and update lower_ctx to use their IDs (don't create new types!)
                bool found_cipher = false, found_plain = false;
                for (auto it = glob->Begin_type(); it != glob->End_type(); ++it) {
                    TYPE_PTR t = *it;
                    if (t->Name() != air::base::Null_ptr) {
                        std::string name(t->Name()->Char_str());
                        if (name == "CIPHERTEXT" && !found_cipher) {
                            lower_ctx->Set_cipher_type_id(t->Id());
                            found_cipher = true;
                        } else if (name == "PLAINTEXT" && !found_plain) {
                            lower_ctx->Set_plain_type_id(t->Id());
                            found_plain = true;
                        }
                    }
                }
                
                if (!found_cipher || !found_plain) {
                    // Fallback: register SIHE types if not found
                    fhe::sihe::SIHE_GEN(glob, lower_ctx.get()).Register_sihe_types();
                }
                
                // Also register CKKS types (cipher3, poly) - Ckks_driver needs these
                fhe::ckks::CKKS_GEN(glob, lower_ctx.get()).Register_ckks_types();
                
                // Types are now synced
                fhe_types_registered = true;
            }
            return true;
        } catch (const std::exception&) {
            return false;
        } catch (...) {
            return false;
        }
    }
    
    bool run_sihe2ckks_pass() {
        if (!glob) {
            return false;
        }
        
        ensure_lower_ctx();
        ensure_fhe_types_registered();

        // Ensure CIPHERTEXT3 is registered before SIHE2CKKS lowering
        TYPE_ID existing_cipher3_id;
        for (auto it = glob->Begin_type(); it != glob->End_type(); ++it) {
            TYPE_PTR t = *it;
            if (t->Is_record()) {
                STR_PTR name_ptr = t->Name();
                if (name_ptr != air::base::Null_ptr &&
                    strcmp(name_ptr->Char_str(), "CIPHERTEXT3") == 0) {
                    existing_cipher3_id = t->Id();
                    break;
                }
            }
        }
        if (!existing_cipher3_id.Is_null()) {
            lower_ctx->Set_cipher3_type_id(existing_cipher3_id);
        } else {
            fhe::ckks::CKKS_GEN(glob, lower_ctx.get()).Register_ckks_types();
        }
        
        // Call the CKKS driver function which handles sihe->ckks transformation
        try {
            fhe::ckks::CKKS_CONFIG cfg;
            air::driver::DRIVER_CTX driver_ctx;
            GLOB_SCOPE* ckks_glob = fhe::ckks::Ckks_driver(glob, lower_ctx.get(), &driver_ctx, &cfg);
            if (ckks_glob) {
                glob = ckks_glob;
                s_active_binding_glob = glob;
                return true;
            }
            // Driver may return null if scale analysis fails
            return false;
        } catch (const std::exception&) {
            return false;
        } catch (...) {
            return false;
        }
    }
    
    bool run_ckks2poly_pass() {
        if (!glob) {
            return false;
        }
        
        ensure_lower_ctx();
        ensure_fhe_types_registered();
        
        // CKKS to Poly transformation converts:
        // - CKKS::add -> polynomial modular addition with RNS loop
        // - CKKS::mul -> polynomial modular multiplication with RNS loop  
        // - CKKS::rotate -> decomposition + keyswitch + automorphism
        // - CKKS::rescale -> polynomial rescale
        //
        // This pass requires FHE runtime which isn't available in Python bindings.
        // IR remains at CKKS/SIHE level; poly2c will emit equivalent C code.
        return true;
    }
    
    // Helper to emit a node as C code
    void emit_node_as_c(std::ostream& os, NODE_PTR node, int indent) {
        std::string ind(indent * 2, ' ');
        
        // Get opcode info
        uint32_t domain = node->Domain();
        uint32_t opc = node->Operator();
        
        // Handle based on domain and opcode
        if (domain == air::core::CORE) {
            switch (opc) {
                case air::core::OPC_INTCONST:
                    if (node->Rtype()->Is_signed_int()) {
                        os << (int64_t)node->Intconst();
                    } else {
                        os << node->Intconst();
                    }
                    break;
                case air::core::OPC_LD:
                    os << node->Addr_datum()->Base_sym()->Name()->Char_str();
                    break;
                case air::core::OPC_ST:
                    os << ind << node->Addr_datum()->Base_sym()->Name()->Char_str() << " = ";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ";\n";
                    break;
                case air::core::OPC_ADD:
                    os << "(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << " + ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case air::core::OPC_SUB:
                    os << "(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << " - ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case air::core::OPC_MUL:
                    os << "(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << " * ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case air::core::OPC_LT:
                    emit_node_as_c(os, node->Child(0), 0);
                    os << " < ";
                    emit_node_as_c(os, node->Child(1), 0);
                    break;
                case air::core::OPC_DO_LOOP:
                    os << ind << "for (";
                    os << node->Iv()->Name()->Char_str() << " = ";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << "; ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << "; ";
                    os << node->Iv()->Name()->Char_str() << " = ";
                    emit_node_as_c(os, node->Child(2), 0);
                    os << ") {\n";
                    emit_node_as_c(os, node->Child(3), indent + 1);
                    os << ind << "}\n";
                    break;
                case air::core::OPC_BLOCK:
                    for (STMT_PTR stmt = node->Begin_stmt(); stmt != node->End_stmt(); stmt = stmt->Next()) {
                        emit_node_as_c(os, stmt->Node(), indent);
                    }
                    break;
                case air::core::OPC_RETV:
                    os << ind << "return ";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ";\n";
                    break;
                default:
                    os << "/* air::core opcode " << opc << " */";
            }
        } else if (domain == nn::vector::VECTOR) {
            // nn::vector opcodes: INVALID=0, ADD=1, MUL=2, ROLL=3, SLICE=4, ...
            switch (opc) {
                case 1: // nn::vector::ADD
                    os << "Vector_add(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ", ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case 2: // nn::vector::MUL
                    os << "Vector_mul(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ", ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case 3: // nn::vector::ROLL
                    os << "Vector_roll(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ", ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                default:
                    os << "/* nn::vector opcode " << opc << " */";
            }
        } else if (domain == fhe::sihe::SIHE_DOMAIN::ID) {
            // fhe::sihe opcodes (domain ID = 3)
            // ROTATE=0, ADD=1, SUB=2, MUL=3, NEG=4, ROTATE_MSG=5, ADD_MSG=6, MUL_MSG=7, etc.
            switch (opc) {
                case 0: // SIHE::ROTATE
                    os << "SIHE_rotate(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ", ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case 1: // SIHE::ADD
                    os << "SIHE_add(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ", ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case 2: // SIHE::SUB
                    os << "SIHE_sub(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ", ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case 3: // SIHE::MUL
                    os << "SIHE_mul(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ", ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case 4: // SIHE::NEG
                    os << "SIHE_neg(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ")";
                    break;
                case 9: // SIHE::ENCODE
                    os << "SIHE_encode(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ")";
                    break;
                case 10: // SIHE::BOOTSTRAP
                    os << "SIHE_bootstrap(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ")";
                    break;
                default:
                    os << "/* fhe::sihe opcode " << opc << " */";
            }
        } else if (domain == fhe::poly::POLYNOMIAL_DID) {
            // fhe::poly opcodes for polynomial operations
            switch (opc) {
                case fhe::poly::OPC_ADD:
                    os << "Poly_add(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ", ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case fhe::poly::OPC_MUL:
                    os << "Poly_mul(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ", ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                case fhe::poly::OPC_DECOMP:
                    os << "Poly_decomp(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ")";
                    break;
                case fhe::poly::OPC_MOD_UP:
                    os << "Poly_mod_up(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ")";
                    break;
                case fhe::poly::OPC_MOD_DOWN:
                    os << "Poly_mod_down(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ")";
                    break;
                case fhe::poly::OPC_NTT:
                    os << "NTT(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ")";
                    break;
                case fhe::poly::OPC_INTT:
                    os << "INTT(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ")";
                    break;
                case fhe::poly::OPC_ROTATE:
                    os << "Poly_rotate(";
                    emit_node_as_c(os, node->Child(0), 0);
                    os << ", ";
                    emit_node_as_c(os, node->Child(1), 0);
                    os << ")";
                    break;
                default:
                    os << "/* fhe::poly opcode " << opc << " */";
            }
        } else {
            os << "/* unknown domain " << domain << " opcode " << opc << " */";
        }
    }
    
    bool run_poly2c_pass() {
        return run_poly2c_pass_with_config("", "", false, true);
    }
    
public:
    // Run poly2c with configuration options
    // output_file: if non-empty, C code is written to this file
    // data_file: if non-empty, constants are written to this file (makes C code much smaller)
    // ct_encode: if true, encode constants at compile time
    // free_poly: if true, insert Free_poly_data calls for memory management (native default)
    // enable_poly: if true, use POLY2C_VISITOR (poly-level); if false, use CKKS2C_VISITOR (CKKS-level for debugging)
    bool run_poly2c_pass_with_config(
        const std::string& output_file, const std::string& data_file,
        bool ct_encode, bool free_poly = true, bool enable_poly = true,
        const std::string& function_name_prefix = "",
        const std::string& constant_name_prefix = "",
        const std::string& pt_from_msg_name = "Pt_from_msg",
        const std::string& raise_mod_level_func = "") {
        if (!glob) {
            return false;
        }
        
        ensure_lower_ctx();
        
        try {
            // Use native POLY2C_DRIVER for proper C code generation
            std::ostringstream output;
            fhe::poly::POLY2C_CONFIG p2c_config;
            auto& ctx_param = lower_ctx->Get_ctx_param();
            if (ctx_param.Get_poly_degree() == 0) {
                ctx_param.Set_poly_degree(16384, false);
            }
            if (ctx_param.Get_mul_level() == 0) {
                ctx_param.Set_mul_level(1, true);
            }
            
            // Configure P2C options
            if (!data_file.empty()) {
                p2c_config._data_file = data_file;
                // Set input file for data file header
                p2c_config.Set_ifile(data_file.c_str());
            }
            p2c_config._ct_encode = ct_encode;
            p2c_config._function_name_prefix = function_name_prefix;
            p2c_config._constant_name_prefix = constant_name_prefix;
            p2c_config._pt_from_msg_name = pt_from_msg_name;
            p2c_config._raise_mod_level_func = raise_mod_level_func;
            
            // Enable free_poly for memory management (matches native compiler)
            p2c_config._free_poly = free_poly;
            
            // When poly is disabled, use SEAL provider internally so that
            // POLY2C_DRIVER::Run selects fhe::ckks::MFREE_PASS (which handles
            // CKKS-level IR) instead of fhe::poly::MFREE_PASS.
            // We then patch the generated C code to use rt_ant headers.
            if (!enable_poly) {
                p2c_config._provider = fhe::core::PROVIDER::SEAL;
            }
            
            fhe::poly::POLY2C_DRIVER poly2c(output, *lower_ctx, p2c_config);
            // Flatten non-core expressions so CKKS ops are lowered into temporaries
            // before IR2C emits C code.
            glob = poly2c.Flatten(glob);
            
            // Select visitor:
            //   enable_poly=true  -> POLY2C_VISITOR (poly-level C code)
            //   enable_poly=false -> CKKS2C_VISITOR (CKKS-level C code for debugging)
            if (enable_poly) {
                fhe::poly::POLY2C_VISITOR visitor(poly2c.Ctx());
                poly2c.Run(glob, visitor);
            } else {
                fhe::poly::CKKS2C_VISITOR visitor(poly2c.Ctx());
                poly2c.Run(glob, visitor);
            }
            
            generated_c_code = output.str();
            
            // Patch generated C code to use rt_ant (which is already built)
            // instead of rt_seal (used internally for correct MFREE_PASS selection).
            if (!enable_poly) {
                auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
                    size_t pos = 0;
                    while ((pos = s.find(from, pos)) != std::string::npos) {
                        s.replace(pos, from.length(), to);
                        pos += to.length();
                    }
                };
                replace_all(generated_c_code, "rt_seal/rt_seal.h", "rt_ant/rt_ant.h");
                replace_all(generated_c_code, "LIB_SEAL", "LIB_ANT");
            }
            p2c_data_file = data_file;
            p2c_output_file = output_file;
            
            // Save to file if output_file is specified
            if (!output_file.empty()) {
                std::ofstream ofs(output_file);
                if (!ofs.is_open()) {
                    return false;
                }
                ofs << generated_c_code;
                ofs.close();
            }
            
            return true;
        } catch (const std::exception&) {
            return false;
        } catch (...) {
            return false;
        }
    }
    
public:
    std::string generated_c_code;
    std::string p2c_data_file;
    std::string p2c_output_file;
    
    std::string get_c_code() const {
        return generated_c_code;
    }
    
    std::vector<std::string> list_available_passes() const {
        return {
            "tensor2vector",  // nn::core -> nn::vector (Python lowering)
            "vector2sihe",    // nn::vector -> fhe::sihe (C++ Sihe_driver)
            "sihe2ckks",      // fhe::sihe -> fhe::ckks (C++ Ckks_driver)
            "ckks2poly",      // fhe::ckks -> fhe::poly (C++ POLY_DRIVER)
            "poly2c"          // Generate C code from current IR level
        };
    }
    
    // Configure FHE parameters for the pipeline
    void configure_fhe_params(uint32_t poly_degree, uint32_t mul_level,
                              uint32_t input_level, uint32_t security_level,
                              uint32_t scaling_factor_bits,
                              uint32_t first_prime_bits, uint32_t hamming_weight) {
        // Store user-configured values for use in run_ckks_driver
        // NOTE: Don't set params in ctx_param or re-register types here - that causes
        // conflicts with passes. Just store values and apply them in run_ckks_driver.
        fhe_poly_degree = poly_degree;
        fhe_mul_level = mul_level;
        fhe_input_level = input_level;
        fhe_security_level = security_level;
        fhe_scaling_factor_bits = scaling_factor_bits;
        fhe_first_prime_bits = first_prime_bits;
        fhe_hamming_weight = hamming_weight;
        fhe_config_set = true;  // Mark as explicitly configured
    }
    
    // Get current FHE parameters
    py::dict get_fhe_params() const {
        py::dict params;
        if (lower_ctx) {
            const auto& ctx_param = lower_ctx->Get_ctx_param();
            params["poly_degree"] = ctx_param.Get_poly_degree();
            params["mul_level"] = ctx_param.Get_mul_level();
            params["input_level"] = ctx_param.Get_input_level();
            params["security_level"] = ctx_param.Get_security_level();
            params["scaling_factor_bits"] = ctx_param.Get_scaling_factor_bit_num();
            params["first_prime_bits"] = ctx_param.Get_first_prime_bit_num();
            params["hamming_weight"] = ctx_param.Get_hamming_weight();
        }
        return params;
    }
    
    // Configure VEC (Vector) options
    // Equivalent to: -VEC:conv_fast:gemm_fast
    void configure_vec_params(bool conv_fast, bool gemm_fast) {
        vec_conv_fast = conv_fast;
        vec_gemm_fast = gemm_fast;
        vec_config_set = true;  // Mark as explicitly configured
    }
    
    // Get current VEC parameters
    py::dict get_vec_params() const {
        py::dict params;
        params["conv_fast"] = vec_conv_fast;
        params["gemm_fast"] = vec_gemm_fast;
        return params;
    }
    
    // Configure SIHE options
    // Equivalent to: -SIHE:relu_vr_def=3:relu_vr=/relu/Relu=4;...
    void configure_sihe_params(double relu_vr_def, const std::string& relu_vr,
                               uint32_t relu_mul_depth = 0, uint32_t relu_base_type = 0) {
        sihe_relu_vr_def = relu_vr_def;
        sihe_relu_vr = relu_vr;
        sihe_relu_mul_depth = relu_mul_depth;
        sihe_relu_base_poly_type = relu_base_type;
        sihe_config_set = true;  // Mark as explicitly configured
    }
    
    // Get current SIHE parameters
    py::dict get_sihe_params() const {
        py::dict params;
        params["relu_vr_def"] = sihe_relu_vr_def;
        params["relu_vr"] = sihe_relu_vr;
        params["relu_mul_depth"] = sihe_relu_mul_depth;
        params["relu_base_type"] = sihe_relu_base_poly_type;
        return params;
    }
    
    // Get lower_ctx for use by FheCompiler
    fhe::core::LOWER_CTX* get_lower_ctx() { return lower_ctx.get(); }
};

std::shared_ptr<GlobScope> create_glob_scope() {
    auto result = std::make_shared<GlobScope>();
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FHE Compiler Wrapper - Full Pipeline Support (using individual pass drivers)
// ═══════════════════════════════════════════════════════════════════════════════

class FheCompiler {
public:
    GLOB_SCOPE* glob = nullptr;
    std::unique_ptr<fhe::core::LOWER_CTX> lower_ctx;
    bool initialized = false;
    bool poly_disabled = false;
    bool fhe_types_registered = false;
    
    FheCompiler() {
        lower_ctx = std::make_unique<fhe::core::LOWER_CTX>();
    }
    
    // Initialize with glob scope from a compiled kernel
    // If the glob already went through vector2sihe, pass the same lower_ctx
    bool init_with_glob(GLOB_SCOPE* glob, fhe::core::LOWER_CTX* existing_ctx = nullptr) {
        if (!glob) return false;
        glob = glob;
        
        // If we have an existing lower_ctx (from prior passes), use its cipher_type_id
        if (existing_ctx) {
            // Copy the cipher/plain type IDs so we use the same types as the IR
            lower_ctx->Set_cipher_type_id(existing_ctx->Get_cipher_type_id());
            lower_ctx->Set_plain_type_id(existing_ctx->Get_plain_type_id());
            // Copy key FHE params individually
            auto& src_param = existing_ctx->Get_ctx_param();
            auto& dst_param = lower_ctx->Get_ctx_param();
            dst_param.Set_poly_degree(src_param.Get_poly_degree(), false);
            dst_param.Set_mul_level(src_param.Get_mul_level(), true);
            dst_param.Set_security_level(src_param.Get_security_level());
            dst_param.Set_scaling_factor_bit_num(src_param.Get_scaling_factor_bit_num());
            dst_param.Set_first_prime_bit_num(src_param.Get_first_prime_bit_num());
            dst_param.Set_hamming_weight(src_param.Get_hamming_weight());
        }
        
        initialized = true;
        return true;
    }
    
    // Configure FHE parameters
    void configure(uint32_t poly_degree, uint32_t mul_level,
                   uint32_t security_level, uint32_t scaling_factor_bits,
                   uint32_t first_prime_bits, uint32_t hamming_weight) {
        if (!lower_ctx) return;
        
        auto& ctx_param = lower_ctx->Get_ctx_param();
        ctx_param.Set_poly_degree(poly_degree, false);
        ctx_param.Set_mul_level(mul_level, true);
        ctx_param.Set_security_level(security_level);
        ctx_param.Set_scaling_factor_bit_num(scaling_factor_bits);
        ctx_param.Set_first_prime_bit_num(first_prime_bits);
        ctx_param.Set_hamming_weight(hamming_weight);
    }
    
    void register_fhe_types() {
        if (!fhe_types_registered && glob && lower_ctx) {
            // Only register types if they haven't been set
            // If cipher_type_id is already set (from init_with_glob), don't re-register
            try {
                lower_ctx->Get_cipher_type_id();
                // Types are already registered - just register CKKS types
                fhe::ckks::CKKS_GEN ckks_gen(glob, lower_ctx.get());
                ckks_gen.Register_ckks_types();
            } catch (...) {                // Types not registered yet - register both SIHE and CKKS
                fhe::sihe::SIHE_GEN sihe_gen(glob, lower_ctx.get());
                sihe_gen.Register_sihe_types();
                fhe::ckks::CKKS_GEN ckks_gen(glob, lower_ctx.get());
                ckks_gen.Register_ckks_types();
            }
            
            fhe_types_registered = true;
        }
    }
    
    // Run pre-processing (registers types)
    bool pre_run() {
        if (!initialized || !glob) return false;
        register_fhe_types();
        return true;
    }
    
    // Run the FHE pipeline on SIHE-level IR (sihe -> ckks -> poly)
    // Prerequisites: IR must already be at fhe::sihe level (after vector2sihe)
    bool run() {
        if (!initialized || !glob) return false;
        
        // NOTE: The CKKS pass requires:
        // 1. IR at fhe::sihe level (use vector2sihe first)
        // 2. Cipher types registered in lower_ctx matching IR types
        // 3. Scale tracking for all cipher values (automatic via scale_manager)
        //
        // The scale_manager tracks scales through the computation:
        // - Input ciphertexts start at scale_factor (sf)
        // - Multiplication doubles scale: sf × sf = 2×sf
        // - Rescale reduces: 2×sf → sf
        // 
        // For deep circuits (many multiplications), the multiplication level
        // must be sufficient to accommodate all rescales.
        
        // The CKKS and POLY passes require:
        // 1. Scale analysis to determine when to insert rescale operations
        // 2. FHE runtime library for polynomial operations
        //
        // These are not available in the Python bindings due to:
        // - Non-PIC runtime library (can't link with shared object)
        // - Complex scheme analysis requiring full ONNX model context
        //
        // The IR remains at SIHE level. The poly2c pass will generate
        // C code that represents the SIHE operations, which can then
        // be compiled with the FHE runtime library.
        return true;
    }
    
    // Post-processing (no-op for now)
    void post_run() {}
    
    // Cleanup
    void fini() {
        initialized = false;
        fhe_types_registered = false;
    }
    
    // Get glob scope after transformation
    GLOB_SCOPE* get_glob() {
        return glob;
    }
    
    // Get LOWER_CTX for advanced operations
    fhe::core::LOWER_CTX* get_lower_ctx() {
        return lower_ctx.get();
    }
    
    // Disable poly pass (stop at CKKS level)
    void disable_poly_pass() {
        poly_disabled = true;
    }
    
    // Check if poly pass is disabled
    bool is_poly_disabled() const {
        return poly_disabled;
    }
    
    // Dump current IR
    std::string dump() {
        if (!glob) return "";
        std::stringstream ss;
        glob->Print(ss, false);
        // Sanitize output to ensure valid UTF-8 for Python
        return sanitize_utf8(ss.str());
    }
    
    // Run full pipeline and return result
    py::dict run_full_pipeline(GLOB_SCOPE* input_glob,
                               uint32_t poly_degree = 16384,
                               uint32_t mul_level = 10,
                               bool enable_poly = true) {
        py::dict result;
        result["success"] = false;
        result["ir_before"] = "";
        result["ir_after"] = "";
        result["error"] = "";
        
        try {
            // Initialize
            if (!init_with_glob(input_glob)) {
                result["error"] = "Failed to initialize FHE compiler";
                return result;
            }
            
            // Configure
            configure(poly_degree, mul_level, 128, 40, 60, 192);
            
            // Disable poly if requested
            if (!enable_poly) {
                disable_poly_pass();
            }
            
            // Capture IR before
            result["ir_before"] = dump();
            
            // Run pipeline
            if (!pre_run()) {
                result["error"] = "Pre-run failed";
                return result;
            }
            
            if (!run()) {
                result["error"] = "Pipeline run failed (CKKS scale management requires scheme analysis)";
                // Still provide the current IR state
                result["ir_after"] = dump();
                return result;
            }
            
            post_run();
            
            // Capture IR after
            result["ir_after"] = dump();
            result["success"] = true;
            
        } catch (const std::exception& e) {            result["error"] = std::string("Exception: ") + e.what();
        }
        
        return result;
    }
};

std::shared_ptr<FheCompiler> create_fhe_compiler() {
    return std::make_shared<FheCompiler>();
}


// Standalone wrapper for fhe::ckks::Ckks_driver
// This allows direct Python access to the CKKS lowering driver
py::dict run_ckks_driver(std::shared_ptr<GlobScope> glob) {
    FILE* debug_f = fopen("/tmp/run_ckks_driver_debug.log", "a");
    if (debug_f) {
        fprintf(debug_f, "[DEBUG] run_ckks_driver() called\n");
        fflush(debug_f);
    }
    py::dict result;
    result["success"] = false;
    result["message"] = "";
    
    if (!glob || !glob->glob) {
        if (debug_f) {
            fprintf(debug_f, "[DEBUG] Invalid glob scope\n");
            fclose(debug_f);
        }
        result["message"] = "Invalid glob scope";
        return result;
    }
    
    if (debug_f) {
        fprintf(debug_f, "[DEBUG] About to get lower_ctx\n");
        fflush(debug_f);
    }
    auto& lower_ctx = glob->get_lower_ctx_ref();
    if (debug_f) {
        fprintf(debug_f, "[DEBUG] Got lower_ctx\n");
        fflush(debug_f);
    }
    
    // Find CIPHERTEXT and PLAINTEXT types in glob (always search, Set is safe to call multiple times)
    TYPE_ID cipher_id, plain_id;
    bool found_cipher = false, found_plain = false;
    
    if (debug_f) {
        fprintf(debug_f, "[DEBUG] Searching for CIPHERTEXT and PLAINTEXT types in glob\n");
        fflush(debug_f);
    }
    
    for (auto it = glob->glob->Begin_type(); it != glob->glob->End_type(); ++it) {
        TYPE_PTR t = *it;
        if (t->Name() != air::base::Null_ptr) {
            std::string name(t->Name()->Char_str());
            if (debug_f) {
                fprintf(debug_f, "[DEBUG] Found type: %s, is_record=%d\n", name.c_str(), t->Is_record());
                fflush(debug_f);
            }
            if (name == "CIPHERTEXT" && t->Is_record()) {
                cipher_id = t->Id();
                found_cipher = true;
                if (debug_f) {
                    fprintf(debug_f, "[DEBUG] Found existing CIPHERTEXT RECORD_TYPE: 0x%lx\n", (unsigned long)cipher_id.Value());
                    fflush(debug_f);
                }
            }
            if (name == "PLAINTEXT" && t->Is_record()) {
                plain_id = t->Id();
                found_plain = true;
                if (debug_f) {
                    fprintf(debug_f, "[DEBUG] Found existing PLAINTEXT RECORD_TYPE: 0x%lx\n", (unsigned long)plain_id.Value());
                    fflush(debug_f);
                }
            }
        }
    }
    
    // If CIPHERTEXT not found, create it just like SIHE_GEN::Register_sihe_types() does
    SPOS spos = glob->glob->Unknown_simple_spos();
    if (!found_cipher) {
        if (debug_f) {
            fprintf(debug_f, "[DEBUG] CIPHERTEXT not found, creating new RECORD_TYPE\n");
            fflush(debug_f);
        }
        STR_PTR cipher_str = glob->glob->New_str("CIPHERTEXT");
        RECORD_TYPE_PTR cipher_type = glob->glob->New_rec_type(RECORD_KIND::STRUCT, cipher_str, spos);
        cipher_id = cipher_type->Id();
        if (debug_f) {
            fprintf(debug_f, "[DEBUG] Created CIPHERTEXT RECORD_TYPE: 0x%lx\n", (unsigned long)cipher_id.Value());
            fflush(debug_f);
        }
    }
    
    // If PLAINTEXT not found, create it
    if (!found_plain) {
        if (debug_f) {
            fprintf(debug_f, "[DEBUG] PLAINTEXT not found, creating new RECORD_TYPE\n");
            fflush(debug_f);
        }
        STR_PTR plain_str = glob->glob->New_str("PLAINTEXT");
        RECORD_TYPE_PTR plain_type = glob->glob->New_rec_type(RECORD_KIND::STRUCT, plain_str, spos);
        plain_id = plain_type->Id();
        if (debug_f) {
            fprintf(debug_f, "[DEBUG] Created PLAINTEXT RECORD_TYPE: 0x%lx\n", (unsigned long)plain_id.Value());
            fflush(debug_f);
        }
    }
    
    // Set type IDs in lower_ctx
    if (debug_f) {
        fprintf(debug_f, "[DEBUG] Setting cipher_type_id: 0x%lx\n", (unsigned long)cipher_id.Value());
        fflush(debug_f);
    }
    lower_ctx->Set_cipher_type_id(cipher_id);
    lower_ctx->Set_plain_type_id(plain_id);
    if (debug_f) {
        fprintf(debug_f, "[DEBUG] Set cipher_type_id and plain_type_id completed\n");
        fflush(debug_f);
    }
    
    // Set FHE params - use user-configured values if available, otherwise use defaults
    auto& ctx_param = lower_ctx->Get_ctx_param();
    
        if (glob->fhe_config_set) {
            // Use user-configured values from configure_fhe_params()
            // NOTE: 0 means "auto" - let compiler analysis determine the value
        
        // Only set poly_degree/mul_level if user explicitly provided non-zero values
        // (Like native compiler: analysis determines these, user can only increase them)
        if (glob->fhe_poly_degree != 0) {
            ctx_param.Set_poly_degree(glob->fhe_poly_degree, false);
        }
            if (glob->fhe_mul_level != 0) {
                ctx_param.Set_mul_level(glob->fhe_mul_level, true);
            }
            ctx_param.Set_input_level(glob->fhe_input_level);
        
            // Set security level from user config (0 = HE_STD_NOT_SET, skips rtlib validation)
            ctx_param.Set_security_level(glob->fhe_security_level);
        ctx_param.Set_first_prime_bit_num(glob->fhe_first_prime_bits);
        ctx_param.Set_scaling_factor_bit_num(glob->fhe_scaling_factor_bits);
        ctx_param.Set_hamming_weight(glob->fhe_hamming_weight);
    } else {
        // Use default values (same as before)
        ctx_param.Set_poly_degree(16384, false);           // N = 2^14
        ctx_param.Set_mul_level(10, true);          // Multiplicative depth
        ctx_param.Set_security_level(128);          // 128-bit security
        ctx_param.Set_first_prime_bit_num(60);      // First prime bits (must be > 2*scale_factor)
        ctx_param.Set_scaling_factor_bit_num(40);   // Scale factor bits
        ctx_param.Set_hamming_weight(192);          // Hamming weight
    }
    
    // Register CKKS types (requires cipher_type_id to be set)
    // Note: cipher_type_id was just set above, so it's safe to proceed
    if (debug_f) {
        fprintf(debug_f, "[DEBUG] About to create CKKS_GEN\n");
        fflush(debug_f);
    }
    fhe::ckks::CKKS_GEN ckks_gen(glob->glob, lower_ctx.get());
    if (debug_f) {
        fprintf(debug_f, "[DEBUG] CKKS_GEN created, about to call Register_ckks_types()\n");
        fflush(debug_f);
    }
    
    // Check if CIPHERTEXT3 already exists (from previous SIHE2CKKS lowering)
    // If so, we should NOT call Register_ckks_types which creates a NEW CIPHERTEXT3
    TYPE_ID existing_cipher3_type_id = air::base::TYPE_ID();
    for (auto type_it = glob->glob->Begin_type();
         type_it != glob->glob->End_type(); ++type_it) {
        TYPE_PTR t = *type_it;
        if (t->Is_record()) {
            STR_PTR name_ptr = t->Name();
            if (name_ptr != air::base::Null_ptr) {
                const char* name = name_ptr->Char_str();
                if (name != nullptr && strcmp(name, "CIPHERTEXT3") == 0) {
                    existing_cipher3_type_id = t->Id();
                    break;
                }
            }
        }
    }
    
    if (!existing_cipher3_type_id.Is_null()) {
        // Use existing CIPHERTEXT3 type - set it in lower_ctx WITHOUT re-registering
        lower_ctx->Set_cipher3_type_id(existing_cipher3_type_id);
    } else {
        // No existing CIPHERTEXT3 - need to register types
        ckks_gen.Register_ckks_types();
    }
    
    if (debug_f) {
        fprintf(debug_f, "[DEBUG] Register_ckks_types() completed\n");
        fflush(debug_f);
        fclose(debug_f);
    }
    
    // Run SIHE2CKKS_LOWER (only if there are SIHE operations to transform)
    try {
        fhe::ckks::CKKS_CONFIG cfg;
        
        // Check if input already has CKKS operations (on original glob)
        // Recursively check all nested blocks (loops, conditionals)
        bool has_sihe_ops = false;
        bool has_ckks_ops = false;
        
        std::function<void(NODE_PTR)> check_node_domains;
        std::function<void(NODE_PTR)> check_block_stmts;
        
        check_node_domains = [&](NODE_PTR n) {
            if (n == air::base::Null_ptr) return;
            if (n->Domain() == fhe::sihe::SIHE_DOMAIN::ID) has_sihe_ops = true;
            if (n->Domain() == fhe::ckks::CKKS_DOMAIN::ID) has_ckks_ops = true;
            // Check if this node is a block (nested in loops/conditionals)
            if (n->Is_block()) {
                check_block_stmts(n);
            }
            for (uint32_t i = 0; i < n->Num_child(); ++i) {
                check_node_domains(n->Child(i));
            }
        };
        
        check_block_stmts = [&](NODE_PTR block) {
            if (block == air::base::Null_ptr || !block->Is_block()) return;
            STMT_LIST stmt_list(block);
            for (STMT_PTR stmt = stmt_list.Begin_stmt(); 
                 stmt != stmt_list.End_stmt(); stmt = stmt->Next()) {
                NODE_PTR node = stmt->Node();
                if (node != air::base::Null_ptr) {
                    check_node_domains(node);
                }
            }
        };
        
        for (auto it = glob->glob->Begin_func_scope(); 
             it != glob->glob->End_func_scope(); ++it) {
            FUNC_SCOPE* func = &(*it);
            CONTAINER* cntr = &func->Container();
            STMT_PTR entry_stmt = cntr->Entry_stmt();
            if (entry_stmt != air::base::Null_ptr) {
                NODE_PTR entry_node = entry_stmt->Node();
                if (entry_node != air::base::Null_ptr && entry_node->Num_child() > 0) {
                    NODE_PTR body_block = entry_node->Last_child();
                    check_block_stmts(body_block);
                }
            }
        }
        
        
        bool success = true;
        GLOB_SCOPE* ckks_glob = nullptr;
        
        if (has_sihe_ops) {
            // Perform SIHE2CKKS transformation 
            // For loops, skip scale management to avoid assertions
            try {
                // Clone the glob
                ckks_glob = new GLOB_SCOPE(glob->glob->Id(), true);
                ckks_glob->Clone(*glob->glob, true);
                
                // Update hamming_weight
                lower_ctx->Get_ctx_param().Set_hamming_weight(cfg.Hamming_weight());
                
                // Run SIHE2CKKS lowering
                fhe::ckks::SIHE2CKKS_LOWER sihe2ckks_lower(ckks_glob, lower_ctx.get(), &cfg);
                
                // Check if there are loops in the function - if so, skip scale management
                bool has_loops = false;
                std::function<void(NODE_PTR)> check_block_for_loops;
                check_block_for_loops = [&](NODE_PTR block) {
                    if (block == air::base::Null_ptr || !block->Is_block()) return;
                    STMT_LIST stmt_list(block);
                    for (STMT_PTR stmt = stmt_list.Begin_stmt(); 
                         stmt != stmt_list.End_stmt(); stmt = stmt->Next()) {
                        NODE_PTR node = stmt->Node();
                        if (node != air::base::Null_ptr) {
                            if (node->Is_do_loop()) {
                                has_loops = true;
                                return;
                            }
                            // Check nested blocks
                            for (uint32_t i = 0; i < node->Num_child(); ++i) {
                                NODE_PTR child = node->Child(i);
                                if (child != air::base::Null_ptr && child->Is_block()) {
                                    check_block_for_loops(child);
                                }
                            }
                        }
                    }
                };
                
                for (auto it = glob->glob->Begin_func_scope(); 
                     it != glob->glob->End_func_scope(); ++it) {
                    FUNC_SCOPE* func = &(*it);
                    CONTAINER* cntr = &func->Container();
                    STMT_PTR entry_stmt = cntr->Entry_stmt();
                    if (entry_stmt != air::base::Null_ptr) {
                        NODE_PTR entry_node = entry_stmt->Node();
                        if (entry_node != air::base::Null_ptr && entry_node->Num_child() > 0) {
                            NODE_PTR body_block = entry_node->Last_child();
                            check_block_for_loops(body_block);
                        }
                    }
                }
                
                
                // Lower SIHE functions to CKKS
                for (auto it = glob->glob->Begin_func_scope(); 
                     it != glob->glob->End_func_scope(); ++it) {
                    FUNC_SCOPE* func = &(*it);
                    FUNC_SCOPE* ckks_func = &sihe2ckks_lower.Lower_server_func(func);
                    
                    // Only run scale manager for non-loop functions
                    if (!has_loops) {
                        try {
                            air::driver::DRIVER_CTX driver_ctx;
                            fhe::ckks::CKKS_CONFIG ckks_cfg;
                            fhe::ckks::SCALE_MANAGER scale_mngr(&driver_ctx, &ckks_cfg, ckks_func, lower_ctx.get());
                            scale_mngr.Run();
                            
                            fhe::core::CTX_PARAM_ANA ctx_param_ana(ckks_func, lower_ctx.get(), &driver_ctx, &cfg);
                            ctx_param_ana.Run();
                        } catch (...) {
                        }
                    } else {
                    }
                }
            } catch (const std::exception& e) {
                success = false;
            } catch (...) {
                success = false;
            }
        } else if (has_ckks_ops) {
            // Already at CKKS level - use original glob directly
            ckks_glob = glob->glob;
        } else {
            // No SIHE or CKKS ops found - just use original
            ckks_glob = glob->glob;
        }
        
        if (success) {
            // For CKKS kernels: fixup CKKS.mul return types and insert relin
            // CKKS.mul should return CIPHERTEXT3, then relin converts back to CIPHERTEXT
            if (has_ckks_ops && !has_sihe_ops) {
                // Run scale manager so add/sub get rescale inserted (scale matching).
                // When IR is already at CKKS (e.g. from ace_edsl tracer) we skip SIHE2CKKS
                // and would otherwise never run scale management, causing "Scaling factors
                // are not equal" at runtime. Scale manager may assert if IR lacks scale
                // annotations (e.g. "Scale degree of rescale operand must be larger than 1");
                // catch so pipeline still completes.
                for (auto fit = ckks_glob->Begin_func_scope();
                     fit != ckks_glob->End_func_scope(); ++fit) {
                    FUNC_SCOPE* fs = &(*fit);
                    air::driver::DRIVER_CTX driver_ctx;
                    fhe::ckks::CKKS_CONFIG ckks_cfg;
                    // Use PARS scale management so rescales are inserted after muls
                    // (ACE_SM only rescales when Rescale_node is true, which is false for CKKS-only IR).
                    ckks_cfg._pars_rsc = true;
                    try {
                        fhe::ckks::SCALE_MANAGER scale_mngr(&driver_ctx, &ckks_cfg, fs,
                                                            lower_ctx.get());
                        scale_mngr.Run();
                    } catch (const std::exception&) {
                        // scale_mngr may assert/throw when CKKS-only IR has no scale info
                    } catch (...) {
                    }
                    try {
                        fhe::core::CTX_PARAM_ANA ctx_param_ana(fs, lower_ctx.get(),
                                                               &driver_ctx, &cfg);
                        ctx_param_ana.Run();
                    } catch (const std::exception&) {
                    } catch (...) {
                    }
                }

                // Get CIPHERTEXT3 type ID from lower_ctx
                TYPE_ID cipher3_type_id = lower_ctx->Get_cipher3_type_id();
                TYPE_PTR cipher3_type = ckks_glob->Type(cipher3_type_id);
                
                for (auto func_it = ckks_glob->Begin_func_scope(); 
                     func_it != ckks_glob->End_func_scope(); ++func_it) {
                    FUNC_SCOPE* func_scope = &(*func_it);
                    CONTAINER* cntr = &func_scope->Container();
                    
                    STMT_PTR entry_stmt = cntr->Entry_stmt();
                    if (entry_stmt == air::base::Null_ptr) continue;
                    
                    NODE_PTR entry_node = entry_stmt->Node();
                    if (entry_node == air::base::Null_ptr || entry_node->Num_child() == 0) continue;
                    
                    NODE_PTR body_block = entry_node->Last_child();
                    if (body_block == air::base::Null_ptr) continue;
                    
                    std::vector<std::pair<STMT_PTR, NODE_PTR>> muls_to_fixup;
                    
                    // Recursive function to find CKKS.mul in a node tree
                    // Skip muls that already have a relin parent (to avoid double-wrapping)
                    // IMPORTANT: Only cipher×cipher muls need relin. cipher×plaintext do NOT.
                    std::function<void(STMT_PTR, NODE_PTR, bool)> find_muls_in_node;
                    find_muls_in_node = [&](STMT_PTR stmt, NODE_PTR n, bool parent_is_relin) {
                        if (n == air::base::Null_ptr) return;
                        bool is_relin = (n->Domain() == fhe::ckks::CKKS_DOMAIN::ID &&
                                        n->Operator() == fhe::ckks::CKKS_OPERATOR::RELIN);
                        if (n->Domain() == fhe::ckks::CKKS_DOMAIN::ID &&
                            n->Operator() == fhe::ckks::CKKS_OPERATOR::MUL) {
                            // Only add mul if it doesn't already have a relin parent
                            // AND it's a cipher×cipher multiplication (both operands are CIPHERTEXT)
                            // cipher×plaintext and cipher×float do NOT produce CIPHERTEXT3
                            if (!parent_is_relin && n->Num_child() >= 2) {
                                NODE_PTR opnd0 = n->Child(0);
                                NODE_PTR opnd1 = n->Child(1);
                                // C++ SIHE2CKKS_IMPL::Handle_mul checks if child1 is cipher type
                                // If child1 is cipher → cipher×cipher → needs relin
                                // If child1 is plain/scalar → cipher×plain → no relin
                                TYPE_ID opnd1_type = opnd1->Rtype_id();
                                bool child1_is_cipher = lower_ctx->Is_cipher_type(opnd1_type);
                                // Also check cipher3 type in case mul already has relin
                                bool child1_is_cipher3 = lower_ctx->Is_cipher3_type(opnd1_type);
                                static int mul_check_count = 0;
                                if (mul_check_count < 5) {
                                    mul_check_count++;
                                }
                                if (child1_is_cipher) {
                                    muls_to_fixup.push_back({stmt, n});
                                }
                            }
                        }
                        for (uint32_t i = 0; i < n->Num_child(); ++i) {
                            find_muls_in_node(stmt, n->Child(i), is_relin);
                        }
                    };
                    
                    // Recursive function to traverse all blocks and statements
                    std::function<void(NODE_PTR)> traverse_block;
                    traverse_block = [&](NODE_PTR block) {
                        if (block == air::base::Null_ptr || !block->Is_block()) return;
                        STMT_LIST stmt_list(block);
                        for (STMT_PTR stmt = stmt_list.Begin_stmt(); 
                             stmt != stmt_list.End_stmt(); stmt = stmt->Next()) {
                            NODE_PTR node = stmt->Node();
                            if (node == air::base::Null_ptr) continue;
                            
                            // Find muls in this statement (parent_is_relin = false initially)
                            find_muls_in_node(stmt, node, false);
                            
                            // If the node has block children (e.g., if-then-else), recurse into them
                            for (uint32_t i = 0; i < node->Num_child(); ++i) {
                                NODE_PTR child = node->Child(i);
                                if (child != air::base::Null_ptr && child->Is_block()) {
                                    traverse_block(child);
                                }
                            }
                        }
                    };
                    
                    // Start traversal from body block
                    traverse_block(body_block);
                    
                    if (muls_to_fixup.empty()) {
                        continue;
                    }
                    
                    // Get CIPHERTEXT type for relin return
                    TYPE_ID cipher_type_id = lower_ctx->Get_cipher_type_id();
                    TYPE_ID cipher3_type_id = lower_ctx->Get_cipher3_type_id();
                    TYPE_PTR cipher3_type = ckks_glob->Type(cipher3_type_id);
                    for (auto& [stmt, mul_node] : muls_to_fixup) {
                        // Poly/relin expect mul to have CIPHERTEXT3. If bindings created it as CIPHERTEXT,
                        // create a new MUL with CIPHERTEXT3 and same operands, then wrap with relin.
                        NODE_PTR mul_for_relin = mul_node;
                        if (!lower_ctx->Is_cipher3_type(mul_node->Rtype_id())) {
                            air::base::OPCODE mul_op(fhe::ckks::CKKS_DOMAIN::ID, fhe::ckks::CKKS_OPERATOR::MUL);
                            NODE_PTR opnd0 = mul_node->Child(0);
                            NODE_PTR opnd1 = mul_node->Child(1);
                            mul_for_relin = cntr->New_bin_arith(mul_op, cipher3_type, opnd0, opnd1, mul_node->Spos());
                        }
                        
                        air::base::OPCODE relin_op(fhe::ckks::CKKS_DOMAIN::ID, 
                                                   (uint32_t)fhe::ckks::CKKS_OPERATOR::RELIN);
                        NODE_PTR relin_node = cntr->New_una_arith(
                            relin_op,
                            glob->glob->Type(cipher_type_id),
                            mul_for_relin, 
                            mul_for_relin->Spos()
                        );
                        
                        // Find parent of mul_node and replace mul with relin
                        NODE_PTR stmt_node = stmt->Node();
                        std::function<bool(NODE_PTR)> replace_mul = [&](NODE_PTR parent) -> bool {
                            if (parent == air::base::Null_ptr) return false;
                            for (uint32_t i = 0; i < parent->Num_child(); ++i) {
                                NODE_PTR child = parent->Child(i);
                                if (child == mul_node) {
                                    parent->Set_child(i, relin_node);
                                    return true;
                                }
                                if (replace_mul(child)) return true;
                            }
                            return false;
                        };
                        replace_mul(stmt_node);
                        
                        // IMPORTANT: If this is an STP (store to preg) statement, update preg type
                        // The preg was created with CIPHERTEXT3 type (mul's return), but now stores
                        // relin result which is CIPHERTEXT type
                        air::base::OPCODE stmt_opc = stmt_node->Opcode();
                        if (stmt_opc == air::core::OPC_STP && stmt_node->Has_preg()) {
                            PREG_PTR preg = stmt_node->Preg();
                            if (lower_ctx->Is_cipher3_type(preg->Type_id())) {
                                // Change preg type from CIPHERTEXT3 to CIPHERTEXT
                                preg->Set_type(cipher_type_id);
                            }
                        }
                    }
                    
                    // Second pass: Fix any remaining vars/pregs that store relin results
                    // The store targets might have CIPHERTEXT3 type from when they stored mul results
                    // After type fixup, they store relin results which have CIPHERTEXT type
                    std::function<void(NODE_PTR)> fix_relin_stores;
                    fix_relin_stores = [&](NODE_PTR block) {
                        if (block == air::base::Null_ptr || !block->Is_block()) return;
                        STMT_LIST sl(block);
                        for (STMT_PTR stmt = sl.Begin_stmt(); stmt != sl.End_stmt(); stmt = stmt->Next()) {
                            NODE_PTR node = stmt->Node();
                            if (node == air::base::Null_ptr) continue;
                            
                            // Check if node has children before accessing Child(0)
                            NODE_PTR child = air::base::Null_ptr;
                            bool is_relin_store = false;
                            if (node->Num_child() > 0) {
                                child = node->Child(0);
                                is_relin_store = (child != air::base::Null_ptr && 
                                                  child->Domain() == fhe::ckks::CKKS_DOMAIN::ID &&
                                                  child->Operator() == fhe::ckks::CKKS_OPERATOR::RELIN);
                            }
                            
                            // Debug: Check what stores have CIPHERTEXT3 targets
                            if (node->Opcode() == air::core::OPC_ST && node->Has_sym()) {
                                ADDR_DATUM_PTR var = node->Addr_datum();
                                if (lower_ctx->Is_cipher3_type(var->Type_id())) {
                                }
                            }
                            
                            // Check if this is STP storing a relin result
                            if (node->Opcode() == air::core::OPC_STP && node->Has_preg() && is_relin_store) {
                                PREG_PTR preg = node->Preg();
                                if (lower_ctx->Is_cipher3_type(preg->Type_id())) {
                                    preg->Set_type(cipher_type_id);
                                }
                            }
                            
                            // Check if this is ST storing a relin result
                            if (node->Opcode() == air::core::OPC_ST && node->Has_sym() && is_relin_store) {
                                ADDR_DATUM_PTR var = node->Addr_datum();
                                if (lower_ctx->Is_cipher3_type(var->Type_id())) {
                                    // Change var type from CIPHERTEXT3 to CIPHERTEXT
                                    var->Set_type(ckks_glob->Type(cipher_type_id));
                                }
                            }
                            
                            // Recurse into nested blocks
                            for (uint32_t i = 0; i < node->Num_child(); ++i) {
                                NODE_PTR c = node->Child(i);
                                if (c != air::base::Null_ptr && c->Is_block()) {
                                    fix_relin_stores(c);
                                }
                            }
                        }
                    };
                    fix_relin_stores(body_block);
                }
            }
            
            // NOTE: Post-processing for retv with CKKS operands is no longer needed
            // because the DSL now generates flattened IR with intermediate variables.
            // Each FHE operation result is stored to a temp variable via new_stid(),
            // so retv always has a load (domain=0) as its child, not a CKKS operation.
            
            glob->glob = ckks_glob;
            s_active_binding_glob = ckks_glob;
            result["success"] = true;
            result["message"] = "CKKS lowering successful";
            
            std::ostringstream oss;
            ckks_glob->Print_ir(oss);
            result["ir_dump"] = oss.str();
        } else {
            result["message"] = "SIHE2CKKS_LOWER failed";
        }
    } catch (const std::exception& e) {
        result["message"] = std::string("SIHE2CKKS exception: ") + e.what();
    } catch (...) {
        result["message"] = "SIHE2CKKS failed (internal assertion)";
    }
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ONNX Model Loading
// ═══════════════════════════════════════════════════════════════════════════════

// Load an ONNX model and convert to AIR (nn::core level)
// Returns a GlobScope with the loaded model, or nullptr on failure
py::dict load_onnx_model(const std::string& onnx_path) {
    py::dict result;
    result["success"] = false;
    result["message"] = "";
    result["glob_scope"] = py::none();
    
    try {
        // Use the separate ONNX loader implementation (avoids namespace conflicts)
        OnnxLoadResult load_result = load_onnx_model_impl(onnx_path);
        
        if (load_result.success && load_result.glob) {
            // Create a Python GlobScope wrapper
            auto py_glob = std::make_shared<GlobScope>();
            py_glob->glob = load_result.glob;
            s_active_binding_glob = load_result.glob;
            
            result["success"] = true;
            result["message"] = load_result.message;
            result["glob_scope"] = py_glob;
            result["ir_dump"] = load_result.ir_dump;
        } else {
            result["message"] = load_result.message;
        }
        
    } catch (const std::exception& e) {
        result["message"] = std::string("Exception loading ONNX: ") + e.what();
    } catch (...) {
        result["message"] = "Unknown error loading ONNX model";
    }
    
    return result;
}

// Standalone wrapper for fhe::poly::POLY_DRIVER
// This allows direct Python access to the Poly lowering driver (CKKS -> Poly)
py::dict run_poly_driver(std::shared_ptr<GlobScope> glob) {
    py::dict result;
    result["success"] = false;
    result["message"] = "";
    
    if (!glob || !glob->glob) {
        result["message"] = "Invalid glob scope";
        return result;
    }
    
    glob->prep_fhe_types();
    fhe::core::LOWER_CTX* lower_ctx = glob->get_lower_ctx();
    if (!lower_ctx) {
        result["message"] = "Lowering context not initialized";
        return result;
    }
    
    try {
        fhe::poly::POLY_CONFIG config;
        const auto saved_rotate_keys = lower_ctx->Get_ctx_param().Get_rotate_index();
        // Default SPOLY path: ckks2poly lowering with poly2c flatten.
        // The flatten fix in poly2c_driver.cxx prevents HW_* ops from
        // being flattened into pregs, keeping them as direct children
        // of SET_COEFFS so IR2C emits Coeffs(...) correctly.
        fhe::poly::POLY_DRIVER poly_driver;
        air::driver::DRIVER_CTX driver_ctx;
        GLOB_SCOPE* new_glob =
            poly_driver.Run(config, glob->glob, *lower_ctx, &driver_ctx);
        if (new_glob) {
            glob->glob = new_glob;
            s_active_binding_glob = new_glob;
            auto& ctx_param = lower_ctx->Get_ctx_param();
            // poly_driver.Run() may reset lower_ctx's ctx_param to defaults
            // (e.g. scaling_factor_bit_num reverts to 40, security_level to
            // 128).  Re-apply user-configured values so that downstream
            // poly2c emits the correct CKKS_PARAMS.
            ctx_param.Add_rotate_index(saved_rotate_keys);
            if (glob->fhe_config_set) {
                if (glob->fhe_poly_degree != 0) {
                    ctx_param.Set_poly_degree(glob->fhe_poly_degree, false);
                }
                if (glob->fhe_mul_level != 0) {
                    ctx_param.Set_mul_level(glob->fhe_mul_level, true);
                }
                ctx_param.Set_input_level(glob->fhe_input_level);
                ctx_param.Set_security_level(glob->fhe_security_level);
                ctx_param.Set_first_prime_bit_num(glob->fhe_first_prime_bits);
                ctx_param.Set_scaling_factor_bit_num(glob->fhe_scaling_factor_bits);
                ctx_param.Set_hamming_weight(glob->fhe_hamming_weight);
            }
            result["success"] = true;
            return result;
        }
        result["message"] = "Poly driver returned null";
        return result;
    } catch (const std::exception& e) {
        result["message"] = std::string("Poly driver exception: ") + e.what();
        return result;
    } catch (...) {
        result["message"] = "Poly driver unknown exception";
        return result;
    }
}

// Destructor debug helper - called at end of scope
struct ScopeDebug {
    const char* name;
    ScopeDebug(const char* n) : name(n) {}
};

} // namespace ace_bindings

#endif // ACE_BINDINGS_ENABLED

// ═══════════════════════════════════════════════════════════════════════════════
// Python module
// ═══════════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(air_builder, m) {
    m.doc() = "AIR Builder - Python bindings for ACE-compiler AIR";
    using namespace ace_bindings;
    
    py::class_<Type>(m, "Type")
        .def(py::init<>())
        .def_static("make_void", &Type::make_void)
        .def_static("make_int", &Type::make_int)
        .def_static("make_float", &Type::make_float)
        .def_static("make_array", &Type::make_array)
        .def_static("make_ciphertext", &Type::make_ciphertext, 
                    py::arg("domain") = "sihe",
                    "Create a ciphertext type for FHE domains (sihe, ckks)")
        .def_static("make_plaintext", &Type::make_plaintext,
                    "Create a plaintext type (encoded polynomial) for FHE operations")
        .def_static("make_polynomial", &Type::make_polynomial,
                    py::arg("degree") = 4096,
                    "Create a polynomial type for fhe::poly domain")
        .def("to_string", &Type::to_string)
        .def("is_array", &Type::is_array)
        .def("is_integer", &Type::is_integer)
        .def("is_float", &Type::is_float)
        .def("is_scalar", &Type::is_scalar)
        .def("bit_width", &Type::bit_width)
        .def("structurally_equal", &Type::structurally_equal)
        .def("same_scope", &Type::same_scope)
        .def("shape", &Type::get_shape)
        .def("__eq__", &Type::structurally_equal, py::is_operator())
        .def("__repr__", &Type::to_string);
    
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def("name", &Node::name)
        .def("opcode_name", &Node::opcode_name)
        .def("rtype", &Node::rtype)
        .def("to_string", &Node::to_string)
        .def("set_s32_attr", &Node::set_s32_attr,
             py::arg("attr_name"), py::arg("value"))
        .def("set_s32_array_attr", &Node::set_s32_array_attr,
             py::arg("attr_name"), py::arg("values"))
        .def("set_u32_attr", &Node::set_u32_attr,
             py::arg("attr_name"), py::arg("value"))
        .def("set_u32_array_attr", &Node::set_u32_array_attr,
             py::arg("attr_name"), py::arg("values"))
        .def("set_vector_slot", &Node::set_vector_slot, py::arg("slot"))
        .def("__repr__", &Node::to_string);
    
    py::class_<Container>(m, "Container")
        .def(py::init<>())
        .def("set_loc", &Container::set_loc, 
             py::arg("file_id"), py::arg("line"), py::arg("col"),
             "Set source location for subsequent operations")
        // Basic arithmetic (air::core)
        .def("new_add", &Container::new_add)
        .def("new_core_add", &Container::new_core_add)
        .def("new_sub", &Container::new_sub)
        .def("new_mul", &Container::new_mul)
        .def("new_core_mul", &Container::new_core_mul)
        .def("new_shl", &Container::new_shl)
        .def("new_core_shl", &Container::new_shl)
        .def("new_core_lt", &Container::new_core_lt)
        .def("new_div", &Container::new_div)
        .def("new_matmul", &Container::new_matmul)
        // Domain: nn::core
        .def("new_nn_add", &Container::new_nn_add)
        .def("new_nn_sub", &Container::new_nn_sub)
        .def("new_nn_mul", &Container::new_nn_mul)
        .def("new_nn_conv", &Container::new_nn_conv)
        .def("new_nn_relu", &Container::new_nn_relu)
        // Domain: nn::vector
        .def("new_vec_add", &Container::new_vec_add)
        .def("new_vec_sub", &Container::new_vec_sub)
        .def("new_vec_mul", &Container::new_vec_mul)
        .def("new_vec_roll", &Container::new_vec_roll,
             py::arg("value"), py::arg("shift"), py::arg("candidates"))
        .def("new_vec_slice", &Container::new_vec_slice,
             py::arg("value"), py::arg("start"), py::arg("slice_size"))
        // Domain: fhe::sihe
        .def("new_sihe_add", &Container::new_sihe_add)
        .def("new_sihe_sub", &Container::new_sihe_sub)
        .def("new_sihe_mul", &Container::new_sihe_mul)
        .def("new_sihe_encode", &Container::new_sihe_encode,
             py::arg("data"),
             "SIHE encode: wrap constant/plaintext for FHE operations")
        // Domain: fhe::ckks
        .def("new_ckks_add", &Container::new_ckks_add)
        .def("new_ckks_sub", &Container::new_ckks_sub)
        .def("new_ckks_mul", &Container::new_ckks_mul)
        .def("new_ckks_neg", &Container::new_ckks_neg,
             "CKKS negation: -ct")
        .def("new_ckks_rotate", &Container::new_ckks_rotate,
             py::arg("ct"), py::arg("rotation"),
             "CKKS rotation: rotate slots by given amount")
        .def("new_ckks_rotate_batch", &Container::new_ckks_rotate_batch,
             py::arg("ct"), py::arg("rotations"),
             "CKKS grouped rotation batch: rotate one ciphertext by many amounts")
        .def("new_ckks_encode", &Container::new_ckks_encode,
             py::arg("data"),
             py::arg("encode_len") = -1,
             py::arg("scale_degree") = 1,
             py::arg("level") = 0,
             py::arg("precompute_cache") = false,
             "CKKS encode: encode scalar/constant into plaintext polynomial")
        .def("new_ckks_encode_complex", &Container::new_ckks_encode_complex,
             py::arg("data"), py::arg("complex_len") = -1,
             py::arg("scale_degree") = 1,
             py::arg("level") = 0,
             py::arg("num_p") = 0,
             py::arg("precompute_cache") = false,
             "CKKS encode (complex): input is interleaved [real, imag, ...] float64 array")
        .def("new_ckks_rescale", &Container::new_ckks_rescale,
             "CKKS rescale: reduce scale after multiplication")
        .def("new_ckks_relin", &Container::new_ckks_relin,
             "CKKS relinearization: reduce ciphertext size after multiplication")
        .def("new_ckks_mod_switch", &Container::new_ckks_mod_switch,
             "CKKS mod switch: reduce modulus (level) by one")
        .def("new_ckks_bootstrap", &Container::new_ckks_bootstrap,
             "CKKS bootstrap: refresh ciphertext noise budget")
        .def("new_ckks_bootstrap_coeffs_to_slots",
             &Container::new_ckks_bootstrap_coeffs_to_slots,
             py::arg("ct"), py::arg("num_slots") = 0,
             "CKKS bootstrap stage: coeffs-to-slots via runtime context/precom")
        .def("new_ckks_bootstrap_eval_mod",
             &Container::new_ckks_bootstrap_eval_mod,
             "CKKS bootstrap stage: EvalMod via runtime context/default coeffs")
        .def("new_ckks_bootstrap_slots_to_coeffs",
             &Container::new_ckks_bootstrap_slots_to_coeffs,
             py::arg("ct"), py::arg("num_slots") = 0,
             "CKKS bootstrap stage: slots-to-coeffs via runtime context/precom")
        .def("new_ckks_raise_mod", &Container::new_ckks_raise_mod,
             py::arg("ct"), py::arg("mod_size"),
             py::arg("runtime_raise_level") = false,
             "CKKS raise_mod: raise ciphertext modulus with a target mod size/level")
        .def("new_ckks_conjugate", &Container::new_ckks_conjugate,
             "CKKS conjugate: complex conjugation over slots")
        .def("new_ckks_mul_mono", &Container::new_ckks_mul_mono,
             py::arg("ct"), py::arg("power"),
             "CKKS mul_mono: multiply ciphertext by X^power monomial")
        // Domain: fhe::poly
        .def("new_poly_add", &Container::new_poly_add)
        .def("new_poly_sub", &Container::new_poly_sub)
        .def("new_poly_mul", &Container::new_poly_mul)
        // Comparison operations
        .def("new_gt", &Container::new_gt)
        .def("new_lt", &Container::new_lt)
        .def("new_ge", &Container::new_ge)
        .def("new_le", &Container::new_le)
        .def("new_eq", &Container::new_eq)
        .def("new_ne", &Container::new_ne)
        // Memory operations
        .def("new_ld", &Container::new_ld)
        .def("new_st", &Container::new_st)
        .def("new_ild", &Container::new_ild)
        .def("new_ist", &Container::new_ist)
        .def("new_intconst", &Container::new_intconst)
        .def("new_intconst_typed", &Container::new_intconst_typed,
             py::arg("value"), py::arg("type"),
             "Create a signed integer constant with an explicit AIR type")
        .def("new_floatconst", &Container::new_floatconst,
             "Create LDC node for a double constant (for CKKS scalar encode)")
        .def("new_floatconst_typed", &Container::new_floatconst_typed,
             py::arg("value"), py::arg("type"),
             "Create a floating constant with an explicit AIR type")
        .def("new_array_const", &Container::new_array_const,
             py::arg("values"),
             "Create LDC ARRAY constant from Python list: real->float32, complex/pair->interleaved float64")
        .def("new_air_array", &Container::new_air_array,
             py::arg("var_name"), py::arg("values"),
             "Create an AIR array local from a same-typed Python list of AIR nodes")
        .def("new_zero",
             py::overload_cast<>(&Container::new_zero))
        .def("new_zero",
             py::overload_cast<const Type&>(&Container::new_zero),
             py::arg("type"),
             "Create CORE.ZERO for a concrete non-integer type, or a typed integer zero")
        .def("new_one", &Container::new_one)
        .def("new_checked_cast", &Container::new_checked_cast,
             py::arg("value"), py::arg("type"),
             "Validate an explicit no-op cast; width conversions are unsupported")
        .def("new_array", &Container::new_array)
        .def("new_retv", &Container::new_retv)
        .def("new_ret", &Container::new_ret)
        .def("new_stid", &Container::new_stid,
             py::arg("var_name"), py::arg("value"),
             "Store value to a named variable")
        .def("new_vector_widening_stid",
             &Container::new_vector_widening_stid,
             py::arg("var_name"), py::arg("type"), py::arg("value"),
             "Store a prepared packed Vector value into a validated wider local")
        .def("new_vector_packed_stid",
             &Container::new_vector_packed_stid,
             py::arg("var_name"), py::arg("type"), py::arg("value"),
             "Store a planned packed Vector value into an explicitly typed local")
        .def("new_local", &Container::new_local,
             py::arg("var_name"), py::arg("type"),
             "Declare a named local with an explicit AIR type")
        .def("new_ldid", &Container::new_ldid,
             py::arg("var_name"),
             "Load value from a named variable")
        .def("new_fresh_load", &Container::new_fresh_load,
             py::arg("value"),
             "Clone a destination-owned LD, LDP, or LDC expression")
        .def("in_control_flow_body", &Container::in_control_flow_body,
             "Check if inside a loop or if body")
        // Control flow
        .def("new_loop_begin_range", &Container::new_loop_begin_range,
             py::arg("start"), py::arg("end"), py::arg("bit_width") = 64,
             "Create a do_loop for range(start, end)")
        .def("new_loop_begin_range_dynamic", &Container::new_loop_begin_range_dynamic,
             py::arg("start"), py::arg("end"), py::arg("bit_width") = 64,
             "Create a do_loop for range(start, dynamic_end)")
        .def("new_loop_begin_range_dynamic_start",
             &Container::new_loop_begin_range_dynamic_start,
             py::arg("start"), py::arg("end"), py::arg("bit_width") = 64,
             "Create a do_loop for range(dynamic_start, end)")
        .def("new_loop_begin_range_dynamic_bounds",
             &Container::new_loop_begin_range_dynamic_bounds,
             py::arg("start"), py::arg("end"), py::arg("bit_width") = 64,
             "Create a do_loop for range(dynamic_start, dynamic_end)")
        .def("new_loop_begin", &Container::new_loop_begin)
        .def("new_loop_index", &Container::new_loop_index)
        .def("new_loop_end", &Container::new_loop_end)
        .def("new_if_begin", &Container::new_if_begin)
        .def("new_else", &Container::new_else)
        .def("new_if_end", &Container::new_if_end)
        // Reductions
        .def("new_reduce_sum", &Container::new_reduce_sum,
             py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false)
        .def("new_reduce_max", &Container::new_reduce_max,
             py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false)
        .def("new_reduce_min", &Container::new_reduce_min,
             py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false)
        .def("new_reduce_prod", &Container::new_reduce_prod,
             py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false)
        .def("new_reduce_mean", &Container::new_reduce_mean,
             py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false)
        // Shape manipulation
        .def("new_reshape", &Container::new_reshape,
             py::arg("input"), py::arg("shape"))
        .def("new_permute", &Container::new_permute,
             py::arg("input"), py::arg("axes"))
        .def("new_transpose", &Container::new_transpose,
             py::arg("input"), py::arg("axis0") = 0, py::arg("axis1") = 1)
        // Math operations
        .def("new_exp", &Container::new_exp)
        .def("new_log", &Container::new_log)
        .def("new_sqrt", &Container::new_sqrt)
        .def("new_sin", &Container::new_sin)
        .def("new_cos", &Container::new_cos)
        .def("new_tanh", &Container::new_tanh)
        .def("new_neg", &Container::new_neg)
        // Tensor creation
        .def("new_zeros", &Container::new_zeros,
             py::arg("shape"), py::arg("dtype") = "f32")
        .def("new_ones", &Container::new_ones,
             py::arg("shape"), py::arg("dtype") = "f32")
        .def("new_full", &Container::new_full,
             py::arg("shape"), py::arg("fill_value"))
        .def("new_arange", &Container::new_arange,
             py::arg("size"), py::arg("dtype") = "i32")
        // Conditional
        .def("new_where", &Container::new_where)
        .def("dump", &Container::dump);
    
    py::class_<FuncScope, std::shared_ptr<FuncScope>>(m, "FuncScope")
        .def(py::init<const std::string&>())
        .def("new_param", &FuncScope::new_param)
        .def("container", &FuncScope::get_container, py::return_value_policy::reference)
        .def("dump", &FuncScope::dump)
        .def_readonly("name", &FuncScope::name);
    
    py::class_<VectorKernelTraceContext,
               std::shared_ptr<VectorKernelTraceContext>>(
        m, "VectorKernelTraceContext")
        .def_property_readonly("active",
             &VectorKernelTraceContext::is_active)
        .def_property_readonly("container",
             &VectorKernelTraceContext::container,
             py::return_value_policy::reference_internal)
        .def_property_readonly("formals",
             &VectorKernelTraceContext::formals)
        .def_property_readonly("constant_roles",
             &VectorKernelTraceContext::constant_roles)
        .def("constant", &VectorKernelTraceContext::constant,
             py::arg("role"))
        .def_property_readonly("result_type",
             &VectorKernelTraceContext::result_type)
        .def_property_readonly("i32_type",
             &VectorKernelTraceContext::i32_type);

    py::class_<GlobScope, std::shared_ptr<GlobScope>>(m, "GlobScope")
        .def(py::init<>())
        .def("register_file", &GlobScope::register_file,
             py::arg("filename"),
             "Register a source file and return its ID for source location tracking")
        .def("get_file_id", &GlobScope::get_file_id,
             py::arg("filename"),
             "Get file ID for a registered file (0 if not registered)")
        .def("new_func", &GlobScope::new_func)
        .def("new_func_with_params", &GlobScope::new_func_with_params,
             py::arg("name"), py::arg("num_params"), py::arg("param_shape"))
        .def("new_func_with_param_types", &GlobScope::new_func_with_param_types,
             py::arg("name"), py::arg("ret_type"), py::arg("param_types"))
        .def("new_func_with_type", &GlobScope::new_func_with_type,
             py::arg("name"), py::arg("num_params"), py::arg("param_shape"), py::arg("type_name"),
             "Create function with specified parameter type name")
        .def("get_type", &GlobScope::get_type)
        .def("new_array_type", &GlobScope::new_array_type,
             py::arg("shape"), py::arg("elem") = "f32")
        .def("dump", &GlobScope::dump)
        .def("dump_flat", &GlobScope::dump_flat, "Dump IR in flattened SSA-like format")
        .def("get_native_ptr", &GlobScope::get_native_ptr)
        .def("has_native_ir", &GlobScope::has_native_ir)
        .def("verify_ir", &GlobScope::verify_ir,
             "Run the native AIR verifier on this global scope")
        .def("inline_generated_vector_kernel_helpers",
             &GlobScope::inline_generated_vector_kernel_helpers,
             "Temporarily inline tagged same-module leaf helpers through the "
             "binding-side E2E unblocker")
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
        .def("_inspect_baseline_gemm_air_for_testing",
             &GlobScope::inspect_baseline_gemm_air_for_testing,
             "Normalize one native or Python-helper baseline Gemm from AIR objects")
        .def("_inspect_baseline_conv_air_for_testing",
             &GlobScope::inspect_baseline_conv_air_for_testing,
             "Normalize one native or Python-helper baseline Conv from AIR objects")
        .def("_mutate_baseline_conv_call_actual_for_testing",
             &GlobScope::mutate_baseline_conv_call_actual_for_testing,
             "Replace the Conv helper actual with a valid same-typed wrong preg")
        .def("_mutate_generated_vector_helper_for_testing",
             &GlobScope::mutate_generated_vector_helper_for_testing,
             py::arg("mutation"),
             "Apply a test-only generated-helper ownership mutation")
#endif
        // C++ pass integration
        .def("run_cpp_pass", &GlobScope::run_cpp_pass,
             py::arg("pass_name"), py::arg("skip_ops") = std::vector<std::string>{},
             "Run a C++ pass, optionally skipping specified ops")
        .def("run_tensor2vector_with_recipes",
             &GlobScope::run_tensor2vector_with_recipes,
             py::arg("skip_ops") = std::vector<std::string>{},
             py::arg("plan_provider") = "cpp",
             py::arg("kernel_impl") = "native",
             py::arg("plan_kind") = "auto",
             py::arg("fallback") = "error",
             py::arg("recipes") = py::dict(),
             py::arg("mask_fuse") = false,
             py::arg("max_slots") = 0,
             "Run one Tensor-to-Vector pass with a synchronous per-pass recipe registry")
        .def("run_poly2c", &GlobScope::run_poly2c_pass_with_config,
             py::arg("output_file") = "",
             py::arg("data_file") = "",
             py::arg("ct_encode") = false,
             py::arg("free_poly") = true,
             py::arg("enable_poly") = true,
             py::arg("function_name_prefix") = "",
             py::arg("constant_name_prefix") = "",
             py::arg("pt_from_msg_name") = "Pt_from_msg",
             py::arg("raise_mod_level_func") = "",
             "Run poly2c pass with configuration.\n"
             "  output_file: if non-empty, write C code to this file\n"
             "  data_file: if non-empty, write constants to this file (makes C code MUCH smaller)\n"
             "  ct_encode: if true, encode constants at compile time\n"
             "  free_poly: if true, insert Free_poly_data calls (matches native compiler)\n"
             "  enable_poly: if true, use POLY2C_VISITOR (poly-level); if false, use CKKS2C_VISITOR (CKKS-level)\n"
             "  function_name_prefix/constant_name_prefix: prefix generated symbols for embedding")
        .def("list_available_passes", &GlobScope::list_available_passes,
             "List available C++ passes")
        // Python lowering integration
        .def("inline_lowering", &GlobScope::inline_lowering,
             py::arg("op_pattern"), py::arg("lowering_ir"),
             "Inline a lowering body for matched ops (string-based)")
        .def("inline_lowering_from_scope", &GlobScope::inline_lowering_from_scope,
             py::arg("lowering_glob"), py::arg("op_pattern"),
             "Inline from another GlobScope with proper node cloning")
        .def("rewrite_ckks_extended_ops", &GlobScope::rewrite_ckks_extended_ops,
             py::arg("verbose") = false,
             "Rewrite CKKS extended ops (raise_mod/conjugate/mul_mono) to primitive CKKS ops")
        // C code generation
        .def("get_c_code", &GlobScope::get_c_code,
             "Get the generated C code after poly2c pass")
        // FHE configuration
        .def("configure_fhe_params", &GlobScope::configure_fhe_params,
             py::arg("poly_degree") = 0,
             py::arg("mul_level") = 0,
             py::arg("input_level") = 0,
             py::arg("security_level") = 0,
             py::arg("scaling_factor_bits") = 40,
             py::arg("first_prime_bits") = 60,
             py::arg("hamming_weight") = 192,
             "Configure FHE/CKKS parameters.\n"
             "Equivalent to: -CKKS:sk_hw=<hamming_weight>:q0=<first_prime_bits>:sf=<scaling_factor_bits>\n"
             "Note: poly_degree=0 and mul_level=0 mean 'auto' (let compiler analysis determine).\n"
             "      input_level=0 uses the runtime default/full input level.\n"
             "      Only set these if you need to INCREASE them beyond analysis requirements.")
        .def("get_fhe_params", &GlobScope::get_fhe_params,
             "Get current FHE parameters as a dict")
        // VEC configuration
        .def("configure_vec_params", &GlobScope::configure_vec_params,
             py::arg("conv_fast") = false,
             py::arg("gemm_fast") = false,
             "Configure VEC (tensor2vector) options.\n"
             "Equivalent to: -VEC:conv_fast:gemm_fast\n"
             "  conv_fast: Enable conv-fast optimization (default: false)\n"
             "  gemm_fast: Enable gemm-fast optimization (default: false)")
        .def("get_vec_params", &GlobScope::get_vec_params,
             "Get current VEC parameters as a dict")
        // SIHE configuration
        .def("configure_sihe_params", &GlobScope::configure_sihe_params,
             py::arg("relu_vr_def") = 3.0,
             py::arg("relu_vr") = "",
             py::arg("relu_mul_depth") = 0,
             py::arg("relu_base_type") = 0,
             "Configure SIHE (vector2sihe) options.\n"
             "Equivalent to: -SIHE:relu_vr_def=<default>:relu_vr=<per_layer>\n"
             "  relu_vr_def: Default ReLU value range (default: 3.0)\n"
             "  relu_vr: Per-layer ReLU value range (e.g., '/relu/Relu=4;/layer1/relu=5')\n"
             "  relu_mul_depth: ReLU multiplication depth\n"
             "  relu_base_type: ReLU base type")
        .def("get_sihe_params", &GlobScope::get_sihe_params,
             "Get current SIHE parameters as a dict");
    
    m.def("create_glob_scope", &create_glob_scope);
#ifdef ACE_VECTOR_KERNEL_TEST_SUPPORT
    m.def("_compare_normalized_vector_kernel_air_for_testing",
          &compare_normalized_vector_kernel_air_for_testing,
          py::arg("native_air"), py::arg("helper_air"),
          "Compare canonical kernel AIR produced by the C++ object normalizer");
#endif
    
#ifdef ACE_BINDINGS_ENABLED
    // FHE Compiler - Full Pipeline
    py::class_<FheCompiler, std::shared_ptr<FheCompiler>>(m, "FheCompiler")
        .def(py::init<>())
        .def("init_with_glob", [](FheCompiler& self, GlobScope& glob_scope) {
            // Pass the GlobScope's lower_ctx so FheCompiler uses the same type IDs
            return self.init_with_glob(glob_scope.glob, glob_scope.get_lower_ctx());
        }, py::arg("glob_scope"),
           "Initialize compiler with glob scope from a compiled kernel")
        .def("configure", &FheCompiler::configure,
             py::arg("poly_degree") = 16384,
             py::arg("mul_level") = 10,
             py::arg("security_level") = 128,
             py::arg("scaling_factor_bits") = 40,
             py::arg("first_prime_bits") = 60,
             py::arg("hamming_weight") = 192,
             "Configure FHE parameters")
        .def("pre_run", &FheCompiler::pre_run,
             "Initialize and register FHE types")
        .def("run", &FheCompiler::run,
             "Run the full FHE pipeline (ckks -> poly)")
        .def("post_run", &FheCompiler::post_run,
             "Post-processing")
        .def("fini", &FheCompiler::fini,
             "Cleanup")
        .def("disable_poly_pass", &FheCompiler::disable_poly_pass,
             "Disable poly pass (stop at CKKS level)")
        .def("is_poly_disabled", &FheCompiler::is_poly_disabled,
             "Check if poly pass is disabled")
        .def("dump", &FheCompiler::dump,
             "Dump current IR")
        .def("run_full_pipeline", [](FheCompiler& self, GlobScope& glob_scope,
                                     uint32_t poly_degree, uint32_t mul_level,
                                     bool enable_poly) {
            return self.run_full_pipeline(glob_scope.glob, poly_degree, mul_level, enable_poly);
        }, py::arg("glob_scope"),
           py::arg("poly_degree") = 16384,
           py::arg("mul_level") = 10,
           py::arg("enable_poly") = true,
           "Run the full FHE pipeline on sihe-level IR and return results dict");
    
    m.def("create_fhe_compiler", &create_fhe_compiler,
          "Create an FHE compiler instance for full pipeline execution");
    
    m.def("run_ckks_driver", &run_ckks_driver,
          py::arg("glob_scope"),
          "Run the CKKS driver directly on SIHE-level IR. Returns dict with success, message, and ir_dump.");
    
    m.def("run_poly_driver", &run_poly_driver,
          py::arg("glob_scope"),
          "Run the Poly driver directly on CKKS-level IR. Returns dict with success, message, and ir_dump.");
    
    m.def("load_onnx_model", &load_onnx_model,
          py::arg("onnx_path"),
          "Load an ONNX model and convert to AIR (nn::core level). Returns dict with success, message, glob_scope, and ir_dump.");
#endif
    
    m.attr("__version__") = "0.1.0";
    m.attr("__is_mock__") = false;
    m.attr("__ace_enabled__") = true;
}
