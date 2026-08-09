//-*-c-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef RTLIB_RT_PHANTOM_RT_DEF_H
#define RTLIB_RT_PHANTOM_RT_DEF_H

#include <cstddef>
#include <cstdint>

// NOLINTBEGIN (readability-identifier-naming)
#include "ciphertext.h"
#include "plaintext.h"
//! @brief Forward declaration of phantom types
namespace phantom {}  // namespace phantom

//! @brief Define CIPHERTEXT/CIPHER/PLAINTEXT/PLAIN for rt APIs
typedef PhantomCiphertext  CIPHERTEXT;
typedef PhantomCiphertext  CIPHERTEXT3;
typedef PhantomCiphertext* CIPHER;
typedef PhantomCiphertext* CIPHER3;
typedef PhantomPlaintext   PLAINTEXT;
typedef PhantomPlaintext*  PLAIN;

// NOLINTEND (readability-identifier-naming)

#define CIPHER_DEFINED 1
#define PLAIN_DEFINED  1

enum PHANTOM_PACKING_CONVENTION : uint32_t {
  PHANTOM_PACKING_FULL = 1,
};

enum PHANTOM_RESOURCE_FLAG : uint64_t {
  PHANTOM_RESOURCE_RELIN_KEY     = UINT64_C(1) << 0,
  PHANTOM_RESOURCE_ROTATION_KEYS = UINT64_C(1) << 1,
  PHANTOM_RESOURCE_CONJUGATION_KEY = UINT64_C(1) << 2,
  PHANTOM_RESOURCE_ROTATE_BATCH  = UINT64_C(1) << 3,
  PHANTOM_RESOURCE_RAISE_MOD     = UINT64_C(1) << 4,
  PHANTOM_RESOURCE_MONOMIALS     = UINT64_C(1) << 5,
  PHANTOM_RESOURCE_COMPLEX_PLAINTEXT = UINT64_C(1) << 6,
  PHANTOM_RESOURCE_NATIVE_BOOTSTRAP_PRECOMPUTE = UINT64_C(1) << 7,
};

enum PHANTOM_CONSTANT_ELEMENT_TYPE : uint32_t {
  PHANTOM_CONSTANT_COMPLEX_F64 = 1,
};

//! @brief Compiler-emitted, immutable inputs for Phantom CKKS construction.
typedef struct {
  uint32_t        _schema_version;
  uint32_t        _packing;
  uint32_t        _poly_degree;
  uint32_t        _logical_slots;
  size_t          _data_q_count;
  const uint32_t* _data_q_bit_sizes;
  size_t          _special_p_count;
  const uint32_t* _special_p_bit_sizes;
  uint32_t        _input_level;
  uint32_t        _q_part_count;
  uint32_t        _hamming_weight;
  uint32_t        _security_level;
  uint32_t        _first_modulus_bits;
  uint32_t        _scaling_modulus_bits;
  uint32_t        _resource_schema_version;
} PHANTOM_CONTEXT_MANIFEST;

//! @brief Non-overlapping key requirements derived from post-CKKS AIR.
typedef struct {
  uint32_t       _schema_version;
  uint32_t       _context_schema_version;
  uint64_t       _flags;
  size_t         _rotation_count;
  const int32_t* _rotation_steps;
  size_t         _rotation_batch_count;
  const size_t*  _rotation_batch_offsets;
  const int32_t* _rotation_batch_steps;
  size_t          _monomial_count;
  const uint32_t* _monomial_powers;
} PHANTOM_RESOURCE_MANIFEST;

//! @brief One compiler-emitted ordinary-Q complex plaintext constant.
typedef struct {
  uint32_t      _entry_id;
  uint64_t      _constant_id;
  uint32_t      _element_type;
  size_t        _slot_count;
  uint32_t      _ace_level;
  uint32_t      _chain_index;
  int32_t       _scale_degree;
  double        _raw_scale;
  const char*   _symbol;
  const char*   _payload_sha256;
  const char*   _cache_key_sha256;
  size_t        _interleaved_value_count;
  const double* _interleaved_values;
} PHANTOM_CONSTANT_ENTRY;

//! @brief Immutable constants bound to the compiler-emitted context manifest.
typedef struct {
  uint32_t                      _schema_version;
  uint32_t                      _context_schema_version;
  uint32_t                      _resource_schema_version;
  const char*                   _context_manifest_sha256;
  size_t                        _entry_count;
  const PHANTOM_CONSTANT_ENTRY* _entries;
} PHANTOM_CONSTANT_MANIFEST;

//! @brief Provider-observed setup and immutable plaintext-cache costs.
typedef struct {
  double   _context_and_key_setup_seconds;
  double   _plaintext_cache_setup_seconds;
  uint64_t _context_and_key_device_bytes;
  uint64_t _plaintext_cache_device_bytes;
  uint64_t _plaintext_cache_logical_device_bytes;
  uint64_t _plaintext_cache_host_bytes;
  size_t   _plaintext_cache_entries;
} PHANTOM_SETUP_METRICS;

#endif  // RTLIB_RT_OPENFHE_RT_DEF_H
