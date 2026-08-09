#ifndef RTLIB_RT_PHANTOM_PHANTOM_API_H
#define RTLIB_RT_PHANTOM_PHANTOM_API_H

//! @brief phantom_api.h
//! Define phantom API can be used by rtlib

#include "common/tensor.h"
#include "rt_def.h"

#include <complex>
#include <vector>

typedef double SCALE_T;
typedef int    LEVEL_T;
typedef std::complex<double> DCMPLX;

//! @brief Sole compiler-generated Phantom context/resource/constant authorities.
extern "C" const PHANTOM_CONTEXT_MANIFEST* Get_phantom_context_manifest();
extern "C" const PHANTOM_RESOURCE_MANIFEST* Get_phantom_resource_manifest();
extern "C" const PHANTOM_CONSTANT_MANIFEST* Get_phantom_constant_manifest();

//! @brief Phantom API for context management

void       Phantom_prepare_input(TENSOR* input, const char* name);
void       Phantom_set_output_data(const char* name, size_t idx, CIPHER data);
CIPHERTEXT Phantom_get_input_data(const char* name, size_t idx);
void Phantom_encode_float(PLAIN plain, float* input, size_t len, SCALE_T scale,
                          LEVEL_T level);
void Phantom_encode_double(PLAIN plain, const double* input, size_t len,
                           SCALE_T scale, LEVEL_T level);
void Phantom_encode_dcmplx(PLAIN plain, const DCMPLX* input, size_t len,
                           SCALE_T scale, LEVEL_T level);
void Phantom_encode_manifest_constant(PLAIN plain, uint32_t entry_id);
void Phantom_load_cached_constant(PLAIN plain, uint32_t entry_id);
PHANTOM_SETUP_METRICS Phantom_get_setup_metrics();

void Phantom_encode_float_cst_lvl(PLAIN plain, float* input, size_t len,
                                  SCALE_T scale, int level);
void Phantom_encode_float_mask(PLAIN plain, float input, size_t len,
                               SCALE_T scale, LEVEL_T level);
void Phantom_encode_double_mask(PLAIN plain, double input, size_t len,
                                SCALE_T scale, LEVEL_T level);
void Phantom_encode_float_mask_cst_lvl(PLAIN plain, float input, size_t len,
                                       SCALE_T scale, int level);
void Phantom_decode_float(PLAIN plain, std::vector<double>& output);
void Phantom_decode_dcmplx(PLAIN plain,
                           std::vector<std::complex<double>>& output);
void Phantom_encrypt_plain(CIPHER cipher, PLAIN plain);
void Phantom_decrypt_dcmplx(
    CIPHER cipher, std::vector<std::complex<double>>& output);

double* Seal_handle_output(const char* name);

//! @brief Phantom API for evaluation
void Phantom_add_ciph(CIPHER res, CIPHER op1, CIPHER op2);
void Phantom_add_const(CIPHER res, CIPHER op1, double op2);
void Phantom_add_plain(CIPHER res, CIPHER op1, PLAIN op2);
void Phantom_sub_ciph(CIPHER res, CIPHER op1, CIPHER op2);
void Phantom_sub_const(CIPHER res, CIPHER op1, double op2);
void Phantom_sub_plain(CIPHER res, CIPHER op1, PLAIN op2);
void Phantom_mul_ciph(CIPHER res, CIPHER op1, CIPHER op2);
void Phantom_mul_ciph_const(CIPHER res, CIPHER op1, double op2);
void Phantom_mul_plain(CIPHER res, CIPHER op1, PLAIN op2);
void Phantom_rotate(CIPHER res, CIPHER op, int step);
void Phantom_conjugate(CIPHER res, CIPHER op);
void Phantom_rotate_batch(CIPHER outputs, CIPHER op, const int32_t* steps,
                          size_t count);
void Phantom_raise_mod(CIPHER res, CIPHER op, uint32_t target_q_count);
void Phantom_mul_mono(CIPHER res, CIPHER op, uint32_t power);
void Phantom_rescale(CIPHER res, CIPHER op);
void Phantom_mod_switch(CIPHER res, CIPHER op);
void Phantom_relin(CIPHER res, CIPHER3 op);
void Phantom_copy(CIPHER res, CIPHER op);
void Phantom_zero(CIPHER res);
void Phantom_register_ciph_lifetime(CIPHER value);
void Phantom_register_plain_lifetime(PLAIN value);
void Phantom_register_ciph_array_lifetime(CIPHER values, size_t count);
void Phantom_register_plain_array_lifetime(PLAIN values, size_t count);
void Phantom_free_ciph(CIPHER res);
void Phantom_free_ciph_array(CIPHER3 res, size_t size);
void Phantom_free_plain(PLAIN res);

SCALE_T Phantom_raw_scale(CIPHER value);
SCALE_T Phantom_scale_degree(CIPHER value);
LEVEL_T Phantom_ace_level(CIPHER value);
LEVEL_T Phantom_active_q_count(CIPHER value);
LEVEL_T Phantom_chain_index(CIPHER value);
size_t  Phantom_slots(CIPHER value);
size_t  Phantom_ciphertext_size(CIPHER value);
bool    Phantom_is_ntt(CIPHER value);

SCALE_T Phantom_plain_raw_scale(PLAIN value);
SCALE_T Phantom_plain_scale_degree(PLAIN value);
LEVEL_T Phantom_plain_ace_level(PLAIN value);
LEVEL_T Phantom_plain_active_q_count(PLAIN value);
LEVEL_T Phantom_plain_chain_index(PLAIN value);
size_t  Phantom_plain_slots(PLAIN value);
bool    Phantom_plain_is_ntt(PLAIN value);

#endif
