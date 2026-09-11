#include "common.h"
#include "ckks/cipher.h"
#include "ckks/ciphertext.h"
#include "ckks/key_gen.h"
#include "common/rt_api.h"
#include "rt_ant/rt_ant.h"
#include <sys/resource.h>

extern "C" CIPHERTEXT dsl_bts_bootstrap_full(CIPHERTEXT, CIPHERTEXT);
extern "C" uint32_t dsl_bts_raise_level();
extern "C" {
CKKS_PARAMS* Get_context_params() {
  static CKKS_PARAMS* p = nullptr;
  if (!p) {
    p = static_cast<CKKS_PARAMS*>(std::calloc(1, sizeof(CKKS_PARAMS)));
    p->_provider = LIB_ANT;
    p->_poly_degree = bench::N;
    p->_mul_depth = bench::Q - 1;
    p->_input_level = bench::InputQ;
    p->_first_mod_size = 60;
    p->_scaling_mod_size = bench::ScaleBits;
    p->_num_q_parts = 3;
    p->_hamming_weight = 192;
  }
  return p;
}
int Get_input_count() { return 0; }
int Get_output_count() { return 0; }
RT_DATA_INFO* Get_rt_data_info() { return nullptr; }
DATA_SCHEME* Get_encode_scheme(int) { return nullptr; }
DATA_SCHEME* Get_decode_scheme(int) { return nullptr; }
}

int main(int argc, char** argv) try {
  unsigned repeats = bench::Repetitions(argc, argv);
  setenv("RTLIB_DISABLE_BOOTSTRAP_PRECOM", "1", 1);
  Prepare_context();
  auto* crt = Get_crt_context();
  bench::Require(Get_primes_cnt(Get_q(crt)) == bench::Q &&
                 Get_primes_cnt(Get_p(crt)) == bench::P,
                 "ANT Q/P chain count mismatch");
  bench::Require(dsl_bts_raise_level() == bench::Q, "DSL must raise to full Q");
  auto* generated = Get_extra_context_params();
  bench::Require(generated && generated->_poly_degree == bench::N &&
                 generated->_mul_depth == bench::Q - 1 &&
                 generated->_input_level == bench::InputQ,
                 "generated context mismatch");
  for (unsigned i = 0; i < bench::Q + bench::P; ++i) {
    bool is_q = i < bench::Q;
    auto value = static_cast<uint64_t>(Get_modulus_val(Get_prime_at(
        is_q ? Get_q(crt) : Get_p(crt), is_q ? i : i - bench::Q)));
    bench::Require(std::abs(std::log2(static_cast<double>(value)) -
                   (is_q && i > 0 ? 56 : 60)) < 0.001, "ANT prime size mismatch");
  }
  auto* kg = reinterpret_cast<CKKS_KEY_GENERATOR*>(Keygen());
  bench::Require(Insert_rot_map(kg, 2 * bench::N - 1) != nullptr,
                 "missing conjugation key");
  auto x = bench::Input();
  PLAIN pt = Alloc_plaintext();
  CIPHER input = Alloc_ciphertext();
  Encode_dcmplx(pt, x.data(), x.size(), 1, bench::InputQ);
  Encrypt(input, pt);
  Free_plaintext(pt);
  bench::Require(Level(input) == bench::InputQ && Sc_degree(input) == 1,
                 "ANT input state mismatch");
  bench::CheckScale(input->_scaling_factor);
  for (unsigned sample = 0; sample <= repeats; ++sample) {
    CIPHERTEXT copy{};
    Copy_ciphertext(&copy, input);
    rusage usage_before{}, usage_after{};
    bench::Require(getrusage(RUSAGE_SELF, &usage_before) == 0, "getrusage failed");
    double start = bench::Now();
    CIPHERTEXT output = dsl_bts_bootstrap_full(copy, copy);
    unsigned raw_q = Level(&output), raw_sf = Sc_degree(&output);
    while (Sc_degree(&output) > 1) Rescale_ciph(&output, &output);
    bench::Require(Level(&output) >= bench::OutputQ, "ANT insufficient output Q");
    while (Level(&output) > bench::OutputQ) Modswitch_ciph(&output);
    double seconds = bench::Now() - start;
    bench::Require(getrusage(RUSAGE_SELF, &usage_after) == 0, "getrusage failed");
    bench::Require(Level(&output) == bench::OutputQ && Sc_degree(&output) == 1,
                   "ANT output state mismatch");
    bench::CheckScale(output._scaling_factor);
    auto* decoded = Get_msg_with_imag(&output);
    bench::Require(decoded && Get_slots(&output) == bench::Slots, "ANT decode failed");
    std::vector<std::complex<double>> y(decoded, decoded + bench::Slots);
    std::free(decoded);
    double error = bench::Error(x, y);
    bench::Result("dsl_cpu", sample, seconds, error, raw_q, raw_sf,
                  output._scaling_factor);
    auto tv = [](timeval t) { return t.tv_sec + t.tv_usec * 1e-6; };
    double system_seconds = tv(usage_after.ru_stime) - tv(usage_before.ru_stime);
    double cpu_seconds = tv(usage_after.ru_utime) - tv(usage_before.ru_utime) + system_seconds;
    std::printf("CPU_BTS_RESOURCE={\"sample\":%u,\"cpu_seconds\":%.9f,"
                "\"system_seconds\":%.9f,\"minor_faults\":%ld,"
                "\"major_faults\":%ld,\"process_peak_rss_kib\":%ld}\n",
                sample, cpu_seconds, system_seconds,
                usage_after.ru_minflt - usage_before.ru_minflt,
                usage_after.ru_majflt - usage_before.ru_majflt, usage_after.ru_maxrss);
    std::fflush(stdout);
    Zero_ciph(&output);
    Free_poly_data(Get_c0(&copy));
    Free_poly_data(Get_c1(&copy));
  }
  Free_ciphertext(input);
  Finalize_context();
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ANT benchmark failed: %s\n", e.what());
  return 1;
}
