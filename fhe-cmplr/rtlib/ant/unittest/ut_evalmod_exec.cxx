#include <cstring>
#include <functional>
#include <vector>

#include "common/rt_api.h"
#include "context/ckks_context.h"
#include "hal/hal.h"
#include "helper.h"
#include "poly/evalmod_exec.h"
#include "rns_poly_impl.h"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
class TEST_EVALMOD_EXEC : public ::testing::Test {
protected:
  EVALMOD_EXEC exec{};
  std::string  old_disable;
  bool         had_disable;
#ifdef _OPENMP
  int old_threads, old_dynamic;
#endif
  void SetUp() override {
    const char* old = getenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM");
    had_disable     = old != nullptr;
    if (old) old_disable = old;
    setenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM", "1", 1);
#ifdef _OPENMP
    old_threads = omp_get_max_threads();
    old_dynamic = omp_get_dynamic();
    omp_set_num_threads(4);
    omp_set_dynamic(0);
#endif
    int32_t unused = 0;
    Set_context_params(64, 8, 50, 40, 3, 1, 8, 0, &unused);
    Prepare_context();
    Evalmod_exec_init(&exec);
    exec.diagnostics = true;
  }
  void TearDown() override {
    Finalize_context();
#ifdef _OPENMP
    omp_set_num_threads(old_threads);
    omp_set_dynamic(old_dynamic);
#endif
    if (had_disable)
      setenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM", old_disable.c_str(), 1);
    else
      unsetenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM");
  }
  POLYNOMIAL Make(unsigned q, unsigned p = 0, bool ntt = true) {
    POLYNOMIAL r{};
    Alloc_poly_data(&r, 64, q, p);
    for (size_t i = 0; i < (q + p) * 64; ++i) r._data[i] = (i * 17 + 9) % 113;
    Set_is_ntt(&r, ntt);
    return r;
  }
  void Same(POLY a, POLY b) {
    EXPECT_EQ(Poly_level(a), Poly_level(b));
    EXPECT_EQ(Num_p(a), Num_p(b));
    EXPECT_EQ(Is_ntt(a), Is_ntt(b));
    EXPECT_EQ(memcmp(a->_data, b->_data, Get_num_pq(a) * 64 * 8), 0);
  }
  void Execute(const std::function<void()>& fn) {
#ifdef _OPENMP
#pragma omp parallel num_threads(exec.threads)
    {
#pragma omp single
      fn();
    }
#else
    fn();
#endif
  }
};
TEST_F(TEST_EVALMOD_EXEC, ExactDecompositionAllPartitions) {
  for (unsigned q : {1U, 2U, 5U, 7U, 8U}) {
    auto src = Make(q), snapshot = Make(q);
    for (unsigned part = 0; part < Num_decomp(&src); ++part) {
      auto expected = Make(q, Get_p_cnt()), actual = Make(q, Get_p_cnt());
      Decomp_modup(&expected, &src, part);
      Execute([&] { Evalmod_decomp(&actual, &src, part, &exec); });
      Same(&expected, &actual);
      Same(&snapshot, &src);
      Free_poly_data(&expected);
      Free_poly_data(&actual);
    }
    Free_poly_data(&src);
    Free_poly_data(&snapshot);
  }
  EXPECT_EQ(exec.fallback[1], 0U);
}
TEST_F(TEST_EVALMOD_EXEC, ExactModDown) {
  for (unsigned q : {1U, 2U, 5U, 7U, 8U}) {
    auto src = Make(q, Get_p_cnt()), snapshot = Make(q, Get_p_cnt()),
         expected = Make(q), actual = Make(q);
    Mod_down(&expected, &src);
    Execute([&] { Evalmod_mod_down(&actual, &src, &exec); });
    Same(&expected, &actual);
    Same(&snapshot, &src);
    for (auto* p : {&src, &snapshot, &expected, &actual}) Free_poly_data(p);
  }
  EXPECT_EQ(exec.fallback[2], 0U);
}
TEST_F(TEST_EVALMOD_EXEC, ExactRescaleInPlaceAndOutOfPlace) {
  for (unsigned q : {2U, 5U, 7U, 8U}) {
    auto src = Make(q), snapshot = Make(q), expected = Make(q),
         actual = Make(q), inplace = Make(q);
    Rescale(&expected, &src);
    Execute([&] { Evalmod_rescale(&actual, &src, &exec); });
    Same(&expected, &actual);
    Same(&snapshot, &src);
    Execute([&] { Evalmod_rescale(&inplace, &inplace, &exec); });
    Same(&expected, &inplace);
    for (auto* p : {&src, &snapshot, &expected, &actual, &inplace})
      Free_poly_data(p);
  }
  EXPECT_EQ(exec.fallback[3], 0U);
}
TEST_F(TEST_EVALMOD_EXEC, PointwiseTasksAndAliasing) {
  constexpr unsigned   n = 65536;
  std::vector<int64_t> a(n), b(n), expected(n), actual(n);
  for (unsigned i = 0; i < n; ++i) {
    a[i] = i * 3 + 1;
    b[i] = i * 5 + 2;
  }
  MODULUS* mod = Get_modulus(Get_prime_at(Get_q(Get_crt_context()), 0));
  Hw_modmul(expected.data(), a.data(), b.data(), mod, n);
  Execute([&] {
    Evalmod_hw_modmul(actual.data(), a.data(), b.data(), mod, n, &exec);
  });
  EXPECT_EQ(actual, expected);
  Hw_modadd(expected.data(), a.data(), b.data(), mod, n);
  actual = a;
  Execute([&] {
    Evalmod_hw_modadd(actual.data(), actual.data(), b.data(), mod, n, &exec);
  });
  EXPECT_EQ(actual, expected);
  Hw_modsub(expected.data(), a.data(), b.data(), mod, n);
  Execute([&] {
    Evalmod_hw_modsub(actual.data(), a.data(), b.data(), mod, n, &exec);
  });
  EXPECT_EQ(actual, expected);
  EXPECT_EQ(exec.fallback[0], 0U);
#ifdef _OPENMP
  if (exec.threads > 1) EXPECT_GT(exec.max_active_jobs, 1U);
  EXPECT_LE(exec.max_active_jobs, (unsigned)exec.threads);
#endif
}
TEST_F(TEST_EVALMOD_EXEC, TwoBranchesShareTeamAndScratchIsPrivate) {
  auto a = Make(8, Get_p_cnt()), b = Make(8, Get_p_cnt());
  auto refa = Make(8), refb = Make(8), outa = Make(8), outb = Make(8);
  Mod_down(&refa, &a);
  Rescale(&refa, &refa);
  Mod_down(&refb, &b);
  Rescale(&refb, &refb);
  for (unsigned repeat = 0; repeat < 4; ++repeat) {
    Set_poly_level(&outa, 8);
    Set_poly_level(&outb, 8);
    Execute([&] {
#ifdef _OPENMP
#pragma omp taskgroup
      {
#pragma omp task shared(outa, a)
        {
          Evalmod_mod_down(&outa, &a, &exec);
          Evalmod_rescale(&outa, &outa, &exec);
        }
#pragma omp task shared(outb, b)
        {
          Evalmod_mod_down(&outb, &b, &exec);
          Evalmod_rescale(&outb, &outb, &exec);
        }
      }
#else
      Evalmod_mod_down(&outa, &a, &exec);
      Evalmod_rescale(&outa, &outa, &exec);
      Evalmod_mod_down(&outb, &b, &exec);
      Evalmod_rescale(&outb, &outb, &exec);
#endif
    });
    Same(&refa, &outa);
    Same(&refb, &outb);
  }
  EXPECT_EQ(exec.active_jobs, 0U);
  for (auto* p : {&a, &b, &refa, &refb, &outa, &outb}) Free_poly_data(p);
}
TEST_F(TEST_EVALMOD_EXEC, CoefficientFallbackAndOutsideTeam) {
  auto src = Make(8, 0, false), a = Make(8, Get_p_cnt()),
       b = Make(8, Get_p_cnt());
  Decomp_modup(&a, &src, 0);
  Execute([&] { Evalmod_decomp(&b, &src, 0, &exec); });
  Same(&a, &b);
  EXPECT_EQ(exec.fallback[1], 1U);
  auto r = Make(8), s = Make(8), out = Make(8);
  Rescale(&out, &r);
  Evalmod_rescale(&s, &s, &exec);
  Same(&out, &s);
  for (auto* p : {&src, &a, &b, &r, &s, &out}) Free_poly_data(p);
}
}  // namespace
