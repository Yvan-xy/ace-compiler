//-*-c++-*-
//=============================================================================
//
// Copyright (c) Ant Group Co., Ltd
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef CKKS_OPCODE
  #define CKKS_OPCODE(NAME, name, kid_num, fld_num, property)
#endif

CKKS_OPCODE(ROTATE, rotate, 2, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(ROTATE_BATCH, rotate_batch, 1, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(ADD,    add,    2, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(SUB,    sub,    2, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(MUL,    mul,    2, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(NEG,    neg,    1, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)

// encode opcode transform vector to plaintext:
CKKS_OPCODE(ENCODE,     encode,     4, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)

// scale management opcodes:
CKKS_OPCODE(RESCALE,    rescale,    1, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(UPSCALE,    upscale,    2, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(MODSWITCH,  modswitch,  1, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(RELIN,      relin,      1, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(BOOTSTRAP,  bootstrap,  1, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)

// opcodes get scale/level/batch_size of ciphertext
CKKS_OPCODE(SCALE,      scale,      1, 0, PROP_EXPR | PROP_TYPE)
CKKS_OPCODE(LEVEL,      level,      1, 0, PROP_EXPR | PROP_TYPE)
CKKS_OPCODE(BATCH_SIZE, batch_size, 1, 0, PROP_EXPR | PROP_TYPE)
CKKS_OPCODE(RAISE_MOD,  raise_mod,  2, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(MUL_MONO,   mul_mono,   2, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(CONJUGATE,  conjugate,  1, 0, PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(BOOTSTRAP_COEFFS_TO_SLOTS, bootstrap_coeffs_to_slots, 1, 0,
            PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(BOOTSTRAP_EVAL_MOD, bootstrap_eval_mod, 1, 0,
            PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)
CKKS_OPCODE(BOOTSTRAP_SLOTS_TO_COEFFS, bootstrap_slots_to_coeffs, 1, 0,
            PROP_EXPR | PROP_TYPE | PROP_ATTR | PROP_LIB_CALL)

// Compiler-only semantic description of one BSGS diagonal linear transform.
// This operation must be lowered to POLY primitives before code generation.
CKKS_OPCODE(LINEAR_TRANSFORM, linear_transform, 2, 0,
            PROP_EXPR | PROP_TYPE | PROP_ATTR)

// free opcode to free ciphertext
CKKS_OPCODE(FREE,       free,       1, 0, PROP_STMT | PROP_TYPE | PROP_LIB_CALL)

// Compiler-only structured scheduling markers.  They delimit independent
// statement regions while keeping every CKKS/POLY operation visible to the
// normal lowering and optimization passes.
CKKS_OPCODE(PARALLEL_SECTIONS_BEGIN, parallel_sections_begin, 0, 0,
            PROP_SCF | PROP_STMT | PROP_ATTR)
CKKS_OPCODE(PARALLEL_SECTION_BEGIN, parallel_section_begin, 0, 0,
            PROP_SCF | PROP_STMT)
CKKS_OPCODE(PARALLEL_SECTION_END, parallel_section_end, 0, 0,
            PROP_SCF | PROP_STMT)
CKKS_OPCODE(PARALLEL_SECTIONS_END, parallel_sections_end, 0, 0,
            PROP_SCF | PROP_STMT)

#undef CKKS_OPCODE
