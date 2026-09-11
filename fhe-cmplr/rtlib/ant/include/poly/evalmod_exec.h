// Experimental execution context for compiler-owned CPU arithmetic regions.
#ifndef RTLIB_ANT_EVALMOD_EXEC_H
#define RTLIB_ANT_EVALMOD_EXEC_H
#include "poly/rns_poly.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int      threads;
  bool     diagnostics;
  uint64_t calls[4];  // pointwise, decomposition, moddown, rescale
  uint64_t jobs[4];
  uint64_t fallback[4];
  uint64_t thread_mask;
  unsigned active_jobs, max_active_jobs;
} EVALMOD_EXEC;
void     Evalmod_exec_init(EVALMOD_EXEC* exec);
void     Evalmod_exec_report(EVALMOD_EXEC* exec);
POLY     Evalmod_decomp(POLY res, POLY src, uint32_t part, EVALMOD_EXEC* exec);
POLY     Evalmod_mod_down(POLY res, POLY src, EVALMOD_EXEC* exec);
POLY     Evalmod_rescale(POLY res, POLY src, EVALMOD_EXEC* exec);
int64_t* Evalmod_hw_modmul(int64_t*, int64_t*, int64_t*, MODULUS*, uint32_t,
                           EVALMOD_EXEC*);
int64_t* Evalmod_hw_modadd(int64_t*, int64_t*, int64_t*, MODULUS*, uint32_t,
                           EVALMOD_EXEC*);
int64_t* Evalmod_hw_modsub(int64_t*, int64_t*, int64_t*, MODULUS*, uint32_t,
                           EVALMOD_EXEC*);
#ifdef __cplusplus
}
#endif
#endif
