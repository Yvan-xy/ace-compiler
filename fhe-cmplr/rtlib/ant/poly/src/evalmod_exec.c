// Region-local CPU execution. No global scheduling mode; legacy entry points
// and callers outside the compiler-owned region retain their old behavior.
#include "poly/evalmod_exec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hal/hal.h"
#include "rns_poly_impl.h"
#ifdef _OPENMP
#include <omp.h>
#endif

enum { POINTWISE, DECOMP, MODDOWN, RESCALE };
typedef void (*CHUNK_FN)(size_t, size_t, unsigned, void*);

void Evalmod_exec_init(EVALMOD_EXEC* e) {
  memset(e, 0, sizeof(*e));
  e->threads = 1;
#ifdef _OPENMP
  if (!omp_in_parallel()) {
    e->threads = omp_get_max_threads();
    if (e->threads > omp_get_thread_limit())
      e->threads = omp_get_thread_limit();
  }
#endif
  const char* flag = getenv("ACE_EVALMOD_DIAGNOSTICS");
  e->diagnostics   = flag && strcmp(flag, "1") == 0;
}
void Evalmod_exec_report(EVALMOD_EXEC* e) {
  if (!e->diagnostics) return;
  printf(
      "EVALMOD_EXEC={\"threads\":%d,\"calls\":[%lu,%lu,%lu,%lu],"
      "\"jobs\":[%lu,%lu,%lu,%lu],\"fallback\":[%lu,%lu,%lu,%lu],"
      "\"thread_mask\":%lu,\"max_active_jobs\":%u}\n",
      e->threads, e->calls[0], e->calls[1], e->calls[2], e->calls[3],
      e->jobs[0], e->jobs[1], e->jobs[2], e->jobs[3], e->fallback[0],
      e->fallback[1], e->fallback[2], e->fallback[3], e->thread_mask,
      e->max_active_jobs);
  fflush(stdout);
}
static void Count(EVALMOD_EXEC* e, unsigned kind, bool fallback) {
  if (!e->diagnostics) return;
  __atomic_fetch_add(&e->calls[kind], 1, __ATOMIC_RELAXED);
  if (fallback) __atomic_fetch_add(&e->fallback[kind], 1, __ATOMIC_RELAXED);
}
static unsigned Chunks(EVALMOD_EXEC* e, size_t n, size_t grain) {
  if (!n) return 0;
  size_t jobs = (n + grain - 1) / grain;
  if (jobs > (size_t)e->threads) jobs = e->threads;
#ifdef _OPENMP
  if (!omp_in_parallel()) jobs = 1;
#else
  jobs = 1;
#endif
  return (unsigned)jobs;
}
static void Invoke(EVALMOD_EXEC* e, unsigned kind, CHUNK_FN fn, void* arg,
                   size_t begin, size_t end, unsigned job) {
  if (e->diagnostics) {
    __atomic_fetch_add(&e->jobs[kind], 1, __ATOMIC_RELAXED);
    unsigned active = __atomic_add_fetch(&e->active_jobs, 1, __ATOMIC_RELAXED);
    unsigned old    = __atomic_load_n(&e->max_active_jobs, __ATOMIC_RELAXED);
    while (old < active && !__atomic_compare_exchange_n(
                               &e->max_active_jobs, &old, active, false,
                               __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    }
#ifdef _OPENMP
    unsigned tid = omp_get_thread_num();
    if (tid < 64)
      __atomic_fetch_or(&e->thread_mask, UINT64_C(1) << tid, __ATOMIC_RELAXED);
#endif
  }
  // Leaf callbacks do not create tasks or wait. Their existing threadprivate
  // RTM stacks therefore never span a task scheduling point. Parent operations
  // below deliberately do not use stack-based RTM timing across taskgroups.
  fn(begin, end, job, arg);
  if (e->diagnostics) __atomic_fetch_sub(&e->active_jobs, 1, __ATOMIC_RELAXED);
}
static void Run(EVALMOD_EXEC* e, unsigned kind, size_t n, size_t grain,
                CHUNK_FN fn, void* arg) {
  unsigned jobs = Chunks(e, n, grain);
  if (!jobs) return;
  if (jobs == 1) {
    Invoke(e, kind, fn, arg, 0, n, 0);
    return;
  }
#ifdef _OPENMP
#pragma omp taskgroup
  {
    for (unsigned j = 0; j < jobs; ++j) {
      size_t begin = n * j / jobs, end = n * (j + 1) / jobs;
#pragma omp task firstprivate(e, kind, fn, arg, begin, end, j)
      Invoke(e, kind, fn, arg, begin, end, j);
    }
  }
#endif
}
static bool Separate(const void* a, size_t as, const void* b, size_t bs) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x && y && (x < y ? y - x >= as : x - y >= bs);
}
static bool SameOrSeparate(const void* a, const void* b, size_t n) {
  return a == b || Separate(a, n, b, n);
}
static void Fft(int64_t* dst, int64_t* src, size_t n, CRT_PRIME* prime,
                bool inv) {
  VALUE_LIST in, out;
  Init_i64_value_list_no_copy(&in, n, src);
  Init_i64_value_list_no_copy(&out, n, dst);
  if (inv)
    Ftt_inv(&out, Get_ntt(prime), &in);
  else
    Ftt_fwd(&out, Get_ntt(prime), &in);
}

typedef struct {
  int64_t *dst, *a, *b;
  MODULUS* mod;
  unsigned op;
} HW_JOB;
static void Hw_job(size_t b, size_t e, unsigned job, void* raw) {
  (void)job;
  HW_JOB* s = raw;
  if (s->op == 0)
    Hw_modmul(s->dst + b, s->a + b, s->b + b, s->mod, e - b);
  else if (s->op == 1)
    Hw_modadd(s->dst + b, s->a + b, s->b + b, s->mod, e - b);
  else
    Hw_modsub(s->dst + b, s->a + b, s->b + b, s->mod, e - b);
}
static int64_t* Hw(int64_t* dst, int64_t* a, int64_t* b, MODULUS* mod,
                   uint32_t n, EVALMOD_EXEC* e, unsigned op) {
  HW_JOB s    = {dst, a, b, mod, op};
  bool   safe = SameOrSeparate(dst, a, (size_t)n * 8) &&
              SameOrSeparate(dst, b, (size_t)n * 8);
  Count(e, POINTWISE, !safe);
  if (!safe)
    Hw_job(0, n, 0, &s);
  else
    Run(e, POINTWISE, n, 16384, Hw_job, &s);
  return dst;
}
int64_t* Evalmod_hw_modmul(int64_t* d, int64_t* a, int64_t* b, MODULUS* m,
                           uint32_t n, EVALMOD_EXEC* e) {
  return Hw(d, a, b, m, n, e, 0);
}
int64_t* Evalmod_hw_modadd(int64_t* d, int64_t* a, int64_t* b, MODULUS* m,
                           uint32_t n, EVALMOD_EXEC* e) {
  return Hw(d, a, b, m, n, e, 1);
}
int64_t* Evalmod_hw_modsub(int64_t* d, int64_t* a, int64_t* b, MODULUS* m,
                           uint32_t n, EVALMOD_EXEC* e) {
  return Hw(d, a, b, m, n, e, 2);
}

typedef struct {
  POLY         dst, src;
  size_t       n, offset, part_len, compl_len;
  int64_t*     coeff;
  VL_CRTPRIME *part_primes, *compl_primes;
  VALUE_LIST * inv, *hat;
} DECOMP_JOB;
static void Decomp_input(size_t b, size_t e, unsigned job, void* raw) {
  (void)job;
  DECOMP_JOB* s = raw;
  for (size_t i = b; i < e; ++i) {
    int64_t* copy = s->dst->_data + (s->offset + i) * s->n;
    memcpy(copy, s->src->_data + (s->offset + i) * s->n, s->n * 8);
    Fft(s->coeff + i * s->n, copy, s->n, Get_vlprime_at(s->part_primes, i),
        true);
  }
}
static void Decomp_convert(size_t b, size_t e, unsigned job, void* raw) {
  (void)job;
  DECOMP_JOB* s = raw;
  INT128_T    sum[s->compl_len];
  for (size_t n = b; n < e; ++n) {
    memset(sum, 0, sizeof(sum));
    for (size_t i = 0; i < s->part_len; ++i) {
      int64_t val = Mul_int64_mod_barret(
          s->coeff[i * s->n + n], Get_i64_value_at(s->inv, i),
          Get_modulus(Get_vlprime_at(s->part_primes, i)));
      VALUE_LIST* row = VL_VALUE_AT(s->hat, i);
      for (size_t j = 0; j < s->compl_len; ++j)
        sum[j] += (INT128_T)val * Get_i64_value_at(row, j);
    }
    for (size_t j = 0; j < s->compl_len; ++j) {
      size_t out                    = j < s->offset ? j : j + s->part_len;
      s->dst->_data[out * s->n + n] = Mod_barrett_128(
          sum[j], Get_modulus(Get_vlprime_at(s->compl_primes, j)));
    }
  }
}
static void Decomp_output(size_t b, size_t e, unsigned job, void* raw) {
  (void)job;
  DECOMP_JOB* s = raw;
  for (size_t j = b; j < e; ++j) {
    size_t   out  = j < s->offset ? j : j + s->part_len;
    int64_t* data = s->dst->_data + out * s->n;
    Fft(data, data, s->n, Get_vlprime_at(s->compl_primes, j), false);
  }
}
POLY Evalmod_decomp(POLY dst, POLY src, uint32_t part, EVALMOD_EXEC* e) {
  bool safe = Is_ntt(src) && Num_p(src) == 0 &&
              Num_alloc(dst) == Get_num_pq(dst) &&
              Separate(dst->_data, Get_poly_mem_size(dst), src->_data,
                       Get_poly_mem_size(src));
  Count(e, DECOMP, !safe);
  if (!safe) return Decomp_modup(dst, src, part);
  CRT_CONTEXT* crt   = Get_crt_context();
  CRT_PRIMES*  parts = Get_qpart(crt);
  size_t       q = Poly_level(src), width = Get_per_part_size(parts);
  VL_CRTPRIME* primes = Get_vl_value_at(Get_primes(parts), part);
  size_t       len =
      part == Num_decomp(src) - 1 ? q - width * part : LIST_LEN(primes);
  DECOMP_JOB s = {
      .dst          = dst,
      .src          = src,
      .n            = Get_rdgree(src),
      .offset       = width * part,
      .part_len     = len,
      .compl_len    = Get_num_pq(dst) - len,
      .part_primes  = primes,
      .compl_primes = Get_qpart_compl_at(Get_qpart_compl(crt), q - 1, part),
      .inv          = VL_L2_VALUE_AT(Get_qlhatinvmodq(parts), part, len - 1),
      .hat          = VL_L2_VALUE_AT(Get_qlhatmodp(parts), q - 1, part)};
  IS_TRUE(LIST_LEN(s.compl_primes) == s.compl_len,
          "decomp complement mismatch");
  s.coeff = malloc(len * s.n * 8);
  IS_TRUE(s.coeff, "decomp allocation failed");
  Run(e, DECOMP, len, 1, Decomp_input, &s);
  Run(e, DECOMP, s.n, 256, Decomp_convert, &s);
  Run(e, DECOMP, s.compl_len, 1, Decomp_output, &s);
  Set_is_ntt(dst, true);
  free(s.coeff);
  return dst;
}

typedef struct {
  POLY        dst, src;
  size_t      n, p, q;
  CRT_PRIMES *ps, *qs;
  int64_t*    prep;
} DOWN_JOB;
static void Down_input(size_t b, size_t e, unsigned job, void* raw) {
  (void)job;
  DOWN_JOB* s = raw;
  for (size_t p = b; p < e; ++p) {
    int64_t* data = s->prep + p * s->n;
    Fft(data, Get_p_coeffs(s->src) + p * s->n, s->n, Get_prime_at(s->ps, p),
        true);
    int64_t  mod  = Get_modulus_val(Get_prime_at(s->ps, p));
    int64_t  inv  = Get_i64_value_at(Get_phatinvmodp(s->ps), p);
    uint64_t prec = Get_i64_value_at(Get_phatinvmodp_prec(s->ps), p);
    for (size_t n = 0; n < s->n; ++n)
      data[n] = Fast_mul_const_with_mod(data[n], inv, prec, mod);
  }
}
static void Down_output(size_t b, size_t e, unsigned job, void* raw) {
  (void)job;
  DOWN_JOB* s = raw;
  for (size_t q = b; q < e; ++q) {
    CRT_PRIME*  prime = Get_prime_at(s->qs, q);
    MODULUS*    mod   = Get_modulus(prime);
    VALUE_LIST* row   = Get_phatmodq_at(s->ps, q);
    int64_t*    dst   = s->dst->_data + q * s->n;
    for (size_t n = 0; n < s->n; ++n) {
      INT128_T sum = 0;
      for (size_t p = 0; p < s->p; ++p)
        sum += (INT128_T)s->prep[p * s->n + n] * Get_i64_value_at(row, p);
      dst[n] = Mod_barrett_128(sum, mod);
    }
    Fft(dst, dst, s->n, prime, false);
    int64_t pinv = Get_i64_value_at(Get_pinvmodq(s->ps), q);
    for (size_t n = 0; n < s->n; ++n)
      dst[n] =
          Mul_int64_mod_barret(Sub_int64_with_mod(s->src->_data[q * s->n + n],
                                                  dst[n], Get_mod_val(mod)),
                               pinv, mod);
  }
}
POLY Evalmod_mod_down(POLY dst, POLY src, EVALMOD_EXEC* e) {
  bool safe = Is_ntt(src) && Num_p(dst) == 0 &&
              Num_alloc(src) == Get_num_pq(src) &&
              Separate(dst->_data, Get_poly_mem_size(dst), src->_data,
                       Get_poly_mem_size(src));
  Count(e, MODDOWN, !safe);
  if (!safe) return Mod_down(dst, src);
  CRT_CONTEXT* crt = Get_crt_context();
  DOWN_JOB     s   = {
      dst,        src, Get_rdgree(src), Num_p(src), Poly_level(src), Get_p(crt),
      Get_q(crt), NULL};
  IS_TRUE(s.p == Get_primes_cnt(s.ps) && Poly_level(dst) == s.q,
          "moddown shape mismatch");
  s.prep = malloc(s.n * s.p * 8);
  IS_TRUE(s.prep, "moddown allocation failed");
  Run(e, MODDOWN, s.p, 1, Down_input, &s);
  Run(e, MODDOWN, s.q, 1, Down_output, &s);
  Set_is_ntt(dst, true);
  free(s.prep);
  return dst;
}

typedef struct {
  POLY        dst, src;
  size_t      n, q;
  CRT_PRIMES* qs;
  int64_t *   last, *scratch;
} RESCALE_JOB;
static void Rescale_output(size_t b, size_t e, unsigned job, void* raw) {
  RESCALE_JOB* s       = raw;
  int64_t*     tmp     = s->scratch + job * s->n;
  CRT_PRIME*   last    = Get_prime_at(s->qs, s->q - 1);
  int64_t      lastmod = Get_modulus_val(last);
  VL_I64*      inv     = Get_ql_inv_mod_qi_at(s->qs, s->q - 2);
  VL_I64*      invprec = Get_ql_inv_mod_qi_prec_at(s->qs, s->q - 2);
  VL_I64*      corr    = Get_ql_ql_inv_mod_ql_div_ql_mod_qi_at(s->qs, s->q - 2);
  VL_I64*      corrprec =
      Get_ql_ql_inv_mod_ql_div_ql_mod_qi_prec_at(s->qs, s->q - 2);
  for (size_t q = b; q < e; ++q) {
    CRT_PRIME* prime = Get_prime_at(s->qs, q);
    int64_t    mod   = Get_modulus_val(prime);
    for (size_t n = 0; n < s->n; ++n)
      tmp[n] = Fast_mul_const_with_mod(Switch_modulus(s->last[n], lastmod, mod),
                                       Get_i64_value_at(corr, q),
                                       Get_i64_value_at(corrprec, q), mod);
    Fft(tmp, tmp, s->n, prime, false);
    int64_t* dst = s->dst->_data + q * s->n;
    int64_t* src = s->src->_data + q * s->n;
    for (size_t n = 0; n < s->n; ++n)
      dst[n] = Add_int64_with_mod(
          Fast_mul_const_with_mod(src[n], Get_i64_value_at(inv, q),
                                  Get_i64_value_at(invprec, q), mod),
          tmp[n], mod);
  }
}
POLY Evalmod_rescale(POLY dst, POLY src, EVALMOD_EXEC* e) {
  bool safe = Is_ntt(src) && Num_p(src) == 0 && Num_p(dst) == 0 &&
              SameOrSeparate(dst->_data, src->_data,
                             Get_rdgree(src) * Poly_level(src) * 8);
  Count(e, RESCALE, !safe);
  if (!safe) return Rescale(dst, src);
  RESCALE_JOB s = {
      dst,  src, Get_rdgree(src), Poly_level(src), Get_q(Get_crt_context()),
      NULL, NULL};
  IS_TRUE(s.q >= 2 && Poly_level(dst) == s.q, "rescale shape mismatch");
  unsigned jobs = Chunks(e, s.q - 1, 1);
  s.last        = malloc(s.n * 8);
  s.scratch     = malloc(jobs * s.n * 8);
  IS_TRUE(s.last && s.scratch, "rescale allocation failed");
  Fft(s.last, src->_data + (s.q - 1) * s.n, s.n, Get_prime_at(s.qs, s.q - 1),
      true);
  Run(e, RESCALE, s.q - 1, 1, Rescale_output, &s);
  Set_is_ntt(dst, true);
  Modswitch(dst, dst);
  free(s.last);
  free(s.scratch);
  return dst;
}
