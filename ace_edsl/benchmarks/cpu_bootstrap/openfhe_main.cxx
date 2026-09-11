#include "common.h"
#include "openfhe.h"

using namespace lbcrypto;

int main(int argc, char** argv) try {
  unsigned repeats = bench::Repetitions(argc, argv);
  CCParams<CryptoContextCKKSRNS> p;
  p.SetSecurityLevel(HEStd_NotSet);
  p.SetRingDim(bench::N);
  p.SetBatchSize(bench::Slots);
  p.SetSecretKeyDist(SPARSE_TERNARY);
  p.SetMultiplicativeDepth(bench::Q - 1);
  p.SetFirstModSize(60);
  p.SetScalingModSize(bench::ScaleBits);
  p.SetScalingTechnique(FIXEDMANUAL);
  p.SetKeySwitchTechnique(HYBRID);
  p.SetNumLargeDigits(3);
  auto cc = GenCryptoContext(p);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  cc->Enable(ADVANCEDSHE);
  cc->Enable(FHE);
  auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
  const auto& qs = cp->GetElementParams()->GetParams();
  const auto& ps = cp->GetParamsP()->GetParams();
  bench::Require(qs.size() == bench::Q && ps.size() == bench::P,
                 "OpenFHE Q/P chain count mismatch");
  for (unsigned i = 0; i < qs.size(); ++i)
    bench::Require(std::abs(std::log2(qs[i]->GetModulus().ConvertToDouble()) - (i ? 56 : 60)) < 0.001,
                   "OpenFHE Q prime size mismatch");
  for (const auto& prime : ps)
    bench::Require(prime->GetModulus().GetMSB() == 60, "OpenFHE P prime size mismatch");
  // Explicit correction 60-56=4: no additional attenuation relative to DSL.
  // The default correction is different and would change the precision workload.
  cc->EvalBootstrapSetup({3, 3}, {16, 16}, bench::Slots, 4);
  auto keys = cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);
  cc->EvalBootstrapKeyGen(keys.secretKey, bench::Slots);
  auto x = bench::Input();
  auto pt = cc->MakeCKKSPackedPlaintext(x, 1, bench::Q - bench::InputQ,
                                      nullptr, bench::Slots);
  auto input = cc->Encrypt(keys.publicKey, pt);
  bench::Require(input->GetElements()[0].GetNumOfElements() == bench::InputQ &&
                 input->GetNoiseScaleDeg() == 1, "OpenFHE input state mismatch");
  bench::CheckScale(input->GetScalingFactor());
  for (unsigned sample = 0; sample <= repeats; ++sample) {
    auto copy = input->Clone();
    double start = bench::Now();
    auto output = cc->EvalBootstrap(copy, 1);
    unsigned raw_q = output->GetElements()[0].GetNumOfElements();
    unsigned raw_sf = output->GetNoiseScaleDeg();
    while (output->GetNoiseScaleDeg() > 1) cc->ModReduceInPlace(output);
    unsigned q = output->GetElements()[0].GetNumOfElements();
    bench::Require(q >= bench::OutputQ, "OpenFHE insufficient output Q");
    if (q > bench::OutputQ) cc->LevelReduceInPlace(output, nullptr, q - bench::OutputQ);
    double seconds = bench::Now() - start;
    bench::Require(output->GetElements()[0].GetNumOfElements() == bench::OutputQ &&
                   output->GetNoiseScaleDeg() == 1, "OpenFHE output state mismatch");
    bench::CheckScale(output->GetScalingFactor());
    Plaintext decoded;
    cc->Decrypt(keys.secretKey, output, &decoded);
    decoded->SetLength(bench::Slots);
    auto y = decoded->GetCKKSPackedValue();
    double error = bench::Error(x, y);
    bench::Result("openfhe", sample, seconds, error, raw_q, raw_sf,
                  output->GetScalingFactor());
  }
  cc->ClearStaticMapsAndVectors();
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "OpenFHE benchmark failed: %s\n", e.what());
  return 1;
}
