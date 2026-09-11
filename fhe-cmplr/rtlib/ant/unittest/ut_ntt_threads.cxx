// Exact arithmetic and execution-path checks for the CPU NTT experiment.
#include "helper.h"
#include "common/rt_api.h"
#include "context/ckks_context.h"
#include "rns_poly_impl.h"
#include <atomic>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(ACE_NTT_TEST_WRAP) && defined(_OPENMP)
namespace {
std::atomic<bool> observe_ntt{false};
std::atomic<int> observed_team{0}, observed_level{0}, observed_calls{0};
void ObserveNtt() {
  if (!observe_ntt.load()) return;
  ++observed_calls;
  for (auto pair : {std::make_pair(&observed_team, omp_get_num_threads()),
                    std::make_pair(&observed_level, omp_get_level())}) {
    int old = pair.first->load();
    while (old < pair.second && !pair.first->compare_exchange_weak(old, pair.second)) {}
  }
}
void ResetObservation() { observed_team = 0; observed_level = 0; observed_calls = 0; observe_ntt = true; }
}
extern "C" void __real_Ftt_fwd(VALUE_LIST*, NTT_CONTEXT*, VALUE_LIST*);
extern "C" void __real_Ftt_inv(VALUE_LIST*, NTT_CONTEXT*, VALUE_LIST*);
extern "C" void __wrap_Ftt_fwd(VALUE_LIST* out, NTT_CONTEXT* ctx, VALUE_LIST* in) {
  ObserveNtt(); __real_Ftt_fwd(out, ctx, in);
}
extern "C" void __wrap_Ftt_inv(VALUE_LIST* out, NTT_CONTEXT* ctx, VALUE_LIST* in) {
  ObserveNtt(); __real_Ftt_inv(out, ctx, in);
}
#endif

namespace {
class TEST_NTT_THREADS : public ::testing::Test {
protected:
  static constexpr unsigned N = 64, Q = 8;
  std::string previous_disable;
  bool had_disable;
#ifdef _OPENMP
  int old_threads, old_dynamic, old_levels;
#endif
  void SetUp() override {
    const char* old = std::getenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM");
    had_disable = old != nullptr;
    if (old) previous_disable = old;
    setenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM", "1", 1);
#ifdef _OPENMP
    old_threads = omp_get_max_threads(); old_dynamic = omp_get_dynamic();
    old_levels = omp_get_max_active_levels();
    omp_set_num_threads(4); omp_set_dynamic(0); omp_set_max_active_levels(2);
#endif
    int32_t unused = 0;
    Set_context_params(N, Q, 50, 40, 3, 1, 8, 0, &unused);
    Prepare_context();
  }
  void TearDown() override {
#if defined(ACE_NTT_TEST_WRAP) && defined(_OPENMP)
    observe_ntt = false;
#endif
    Finalize_context();
#ifdef _OPENMP
    omp_set_num_threads(old_threads); omp_set_dynamic(old_dynamic);
    omp_set_max_active_levels(old_levels);
#endif
    if (had_disable) setenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM", previous_disable.c_str(), 1);
    else unsetenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM");
  }
  POLYNOMIAL Make(unsigned q, unsigned p = 0, bool ntt = false) {
    POLYNOMIAL out{};
    Alloc_poly_data(&out, N, q, p);
    for (unsigned i = 0; i < q * N; ++i) out._data[i] = (i * 17 + 3) % 101;
    Set_is_ntt(&out, ntt);
    return out;
  }
  VL_CRTPRIME* Primes() { return Get_q_primes(Get_crt_context()); }
  void Same(POLY expected, POLY actual) {
    EXPECT_EQ(Poly_level(expected), Poly_level(actual));
    EXPECT_EQ(Num_p(expected), Num_p(actual));
    EXPECT_EQ(Is_ntt(expected), Is_ntt(actual));
    EXPECT_EQ(std::memcmp(expected->_data, actual->_data,
                         Get_num_pq(expected) * N * sizeof(int64_t)), 0);
  }
};

TEST_F(TEST_NTT_THREADS, ExactRoundTripAndRepeatedCalls) {
  for (unsigned threads : {0U, 1U, 2U, 3U, 16U}) {
    auto original = Make(Q), expected = Make(Q), actual = Make(Q), decoded = Make(Q);
    Conv_poly2ntt_inplace_with_primes(&expected, Primes());
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      std::memcpy(actual._data, original._data, Q * N * sizeof(int64_t));
      Set_is_ntt(&actual, false);
      Conv_poly2ntt_inplace_with_primes_threads(&actual, Primes(), threads);
      Same(&expected, &actual);
      Conv_ntt2poly_with_primes_threads(&decoded, &actual, Primes(), threads);
      Same(&original, &decoded);
      Conv_ntt2poly_with_primes_threads(&actual, &actual, Primes(), threads);
      Same(&original, &actual);
    }
    for (auto* p : {&original, &expected, &actual, &decoded}) Free_poly_data(p);
  }
}

TEST_F(TEST_NTT_THREADS, EmptyAndPrimeListPrefix) {
  auto whole = Make(Q), original = Make(Q);
  POLYNOMIAL prefix{}, empty{};
  Extract_poly(&prefix, &whole, 0, 2);
  Conv_poly2ntt_inplace_with_primes_threads(&prefix, Primes(), 16);
  Conv_ntt2poly_with_primes_threads(&prefix, &prefix, Primes(), 16);
  Same(&original, &whole); // Includes the untouched tail of the allocation.
  Extract_poly(&empty, &whole, 0, 0);
  Set_is_ntt(&empty, false);
  Conv_poly2ntt_inplace_with_primes_threads(&empty, Primes(), 16);
  EXPECT_TRUE(Is_ntt(&empty));
  Conv_ntt2poly_with_primes_threads(&empty, &empty, Primes(), 16);
  EXPECT_FALSE(Is_ntt(&empty));
  Same(&original, &whole);
  Free_poly_data(&whole); Free_poly_data(&original);
}

TEST_F(TEST_NTT_THREADS, DecompositionMatchesLegacyAllPartitions) {
  for (unsigned q : {1U, 2U, 5U, 7U, 8U}) {
    auto input = Make(q, 0, true), snapshot = Make(q, 0, true);
    for (unsigned part = 0; part < Num_decomp(&input); ++part) {
      auto expected = Make(q, Get_p_cnt());
      Decomp_modup(&expected, &input, part);
      for (unsigned threads : {0U, 1U, 2U, 16U}) {
        auto actual = Make(q, Get_p_cnt());
        Decomp_modup_with_ntt_threads(&actual, &input, part, threads);
        Same(&expected, &actual);
        Same(&snapshot, &input);
        Free_poly_data(&actual);
      }
      Free_poly_data(&expected);
    }
    Free_poly_data(&input); Free_poly_data(&snapshot);
  }
}

TEST_F(TEST_NTT_THREADS, CoefficientInputKeepsSerialBaseConversion) {
  auto input = Make(Q), expected = Make(Q, Get_p_cnt()), actual = Make(Q, Get_p_cnt());
  Decomp_modup(&expected, &input, 2);
  Decomp_modup_with_ntt_threads(&actual, &input, 2, 16);
  Same(&expected, &actual);
  EXPECT_FALSE(Is_ntt(&actual));
  Free_poly_data(&input); Free_poly_data(&expected); Free_poly_data(&actual);
}

#ifdef _OPENMP
TEST_F(TEST_NTT_THREADS, TimingAggregationKeepsConcurrentCounts) {
  const char* previous = std::getenv("RTLIB_TIMING_OUTPUT");
  bool had_previous = previous != nullptr;
  std::string saved = previous ? previous : "";
  setenv("RTLIB_TIMING_OUTPUT", "stdout", 1);
  auto count = []() {
    testing::internal::CaptureStdout();
    Report_rtlib_timing();
    std::istringstream lines(testing::internal::GetCapturedStdout());
    std::string line, name;
    while (std::getline(lines, line)) {
      std::istringstream fields(line);
      fields >> name;
      if (name == "NTT") {
        uint64_t value = 0;
        fields >> value;
        return value;
      }
    }
    return uint64_t{0};
  };
  uint64_t before = count();
#pragma omp parallel for num_threads(4)
  for (int i = 0; i < 32000; ++i) Append_rtlib_timing(RTM_NTT, 0, 1000, 0);
  uint64_t after = count();
  if (had_previous) setenv("RTLIB_TIMING_OUTPUT", saved.c_str(), 1);
  else unsetenv("RTLIB_TIMING_OUTPUT");
  EXPECT_EQ(after - before, 32000U);
}
#endif

#if defined(ACE_NTT_TEST_WRAP) && defined(_OPENMP)
TEST_F(TEST_NTT_THREADS, WorkersObeyBudgetAndAvoidNestedTeams) {
  auto input = Make(Q);
  ResetObservation();
  Conv_poly2ntt_inplace_with_primes_threads(&input, Primes(), 3);
  EXPECT_EQ(observed_team.load(), std::min(3, omp_get_thread_limit()));
  EXPECT_EQ(observed_calls.load(), Q);
  Conv_ntt2poly_with_primes_threads(&input, &input, Primes(), 1);
  ResetObservation();
  Conv_poly2ntt_inplace_with_primes_threads(&input, Primes(), 16);
  EXPECT_EQ(observed_team.load(), std::min(4, omp_get_thread_limit()));
  ResetObservation();
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    Conv_ntt2poly_with_primes_threads(&input, &input, Primes(), 16);
  }
  EXPECT_EQ(observed_level.load(), 1); // No nested (even serialized) region.
  EXPECT_EQ(observed_calls.load(), Q);
  observe_ntt = false;
  Free_poly_data(&input);
}

TEST_F(TEST_NTT_THREADS, DecompEntryReallyReachesParallelNtt) {
  auto input = Make(Q, 0, true), output = Make(Q, Get_p_cnt());
  ResetObservation();
  Decomp_modup_with_ntt_threads(&output, &input, 2, 3);
  EXPECT_GT(observed_calls.load(), 0);
  if (omp_get_thread_limit() > 1) EXPECT_GT(observed_team.load(), 1);
  EXPECT_LE(observed_team.load(), 3);
  EXPECT_EQ(observed_level.load(), omp_get_thread_limit() > 1 ? 1 : 0);
  ResetObservation();
  Decomp_modup_with_ntt_threads(&output, &input, 2, 1);
  EXPECT_EQ(observed_team.load(), 1);
  EXPECT_EQ(observed_level.load(), 0);
  observe_ntt = false;
  Free_poly_data(&input); Free_poly_data(&output);
}

TEST_F(TEST_NTT_THREADS, OverlappingViewsFallBackToLegacyOrder) {
  auto expected = Make(3, 0, true), actual = Make(3, 0, true);
  POLYNOMIAL in0{}, out0{}, in1{}, out1{};
  Extract_poly(&in0, &expected, 0, 2); Extract_poly(&out0, &expected, 1, 2);
  Extract_poly(&in1, &actual, 0, 2); Extract_poly(&out1, &actual, 1, 2);
  Conv_ntt2poly_with_primes(&out0, &in0, Primes());
  ResetObservation();
  Conv_ntt2poly_with_primes_threads(&out1, &in1, Primes(), 16);
  EXPECT_EQ(observed_team.load(), 1);
  EXPECT_EQ(observed_level.load(), 0);
  Same(&expected, &actual);
  observe_ntt = false;
  Free_poly_data(&expected); Free_poly_data(&actual);
}
#endif
} // namespace
