// Optional link-time observer for correctness smoke runs, not performance data.
#include "ckks/cipher.h"
#include "util/ntt.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

namespace {
std::atomic<bool> active{false};
std::atomic<unsigned long> entries{0}, limbs{0}, parallel_limbs{0};
std::atomic<int> max_team{0};
void Observe() {
  if (!active.load(std::memory_order_relaxed)) return;
  ++limbs;
  int team = omp_get_num_threads();
  if (team > 1) ++parallel_limbs;
  int old = max_team.load();
  while (old < team && !max_team.compare_exchange_weak(old, team)) {}
}
void Report() {
  std::printf("CPU_NTT_PROBE={\"new_entry_calls\":%lu,\"ntt_limb_calls\":%lu,"
              "\"parallel_limb_calls\":%lu,\"max_team\":%d}\n",
              entries.load(), limbs.load(), parallel_limbs.load(), max_team.load());
}
struct Register { Register() { std::atexit(Report); } } registration;
}
extern "C" POLY __real_Decomp_modup_with_ntt_threads(POLY, POLY, uint32_t, uint32_t);
extern "C" POLY __wrap_Decomp_modup_with_ntt_threads(POLY dst, POLY src,
                                                    uint32_t part, uint32_t threads) {
  ++entries;
  active = true;
  POLY out = __real_Decomp_modup_with_ntt_threads(dst, src, part, threads);
  active = false;
  return out;
}
extern "C" void __real_Ftt_fwd(VALUE_LIST*, NTT_CONTEXT*, VALUE_LIST*);
extern "C" void __real_Ftt_inv(VALUE_LIST*, NTT_CONTEXT*, VALUE_LIST*);
extern "C" void __wrap_Ftt_fwd(VALUE_LIST* out, NTT_CONTEXT* ctx, VALUE_LIST* in) {
  Observe(); __real_Ftt_fwd(out, ctx, in);
}
extern "C" void __wrap_Ftt_inv(VALUE_LIST* out, NTT_CONTEXT* ctx, VALUE_LIST* in) {
  Observe(); __real_Ftt_inv(out, ctx, in);
}
