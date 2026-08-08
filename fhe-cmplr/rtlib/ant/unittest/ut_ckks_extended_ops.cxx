// Exact and decoded tests for the retained ANT CKKS operations.

#include "ckks/cipher.h"
#include "ckks/ciphertext.h"
#include "common/rt_api.h"
#include "context/ckks_context.h"
#include "helper.h"
#include "rns_poly_impl.h"
#include "util/crt.h"
#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

namespace {

using Complex = std::complex<double>;

class TEST_CKKS_EXTENDED_OPS : public ::testing::Test {
protected:
  void SetUp() override {
    _degree = 32;
    _slots = _degree / 2;
    // ANT still represents conjugation internally with 2N-1, but the
    // ordinary rotation list itself contains only unique, nonzero steps.
    _rotation_steps = {
        5, -7, static_cast<std::int32_t>(2U * _degree - 1U)};
    Set_context_params(_degree, 3, 33, 30, 0, 1, 0, _rotation_steps.size(),
                       _rotation_steps.data());
    Prepare_context();
    _full_q_count = Get_q_cnt();
    ASSERT_GE(_full_q_count, 2U);
  }

  void TearDown() override { Finalize_context(); }

  std::uint64_t Modulus(std::size_t index) const {
    return static_cast<std::uint64_t>(
        Get_modulus_val(Get_prime_at(Get_q(Get_crt_context()), index)));
  }

  static std::uint64_t ReduceSigned(std::int64_t value, std::uint64_t modulus) {
    if (value >= 0)
      return static_cast<std::uint64_t>(value) % modulus;
    const std::uint64_t magnitude = static_cast<std::uint64_t>(-value);
    const std::uint64_t reduced = magnitude % modulus;
    return reduced == 0 ? 0 : modulus - reduced;
  }

  CIPHERTEXT MakeCoefficientCipher(std::size_t q_count) const {
    CIPHERTEXT cipher{};
    Init_ciphertext(&cipher, _degree, q_count, 0, std::ldexp(1.0, 30), 1,
                    _slots);
    Set_is_ntt(Get_c0(&cipher), false);
    Set_is_ntt(Get_c1(&cipher), false);
    return cipher;
  }

  std::vector<std::int64_t> FillSignedPattern(CIPHERTEXT *cipher) const {
    std::vector<std::int64_t> signed_values(2U * _degree);
    for (std::size_t component = 0; component < 2; ++component) {
      POLYNOMIAL *polynomial = component == 0 ? Get_c0(cipher) : Get_c1(cipher);
      for (std::size_t coefficient = 0; coefficient < _degree; ++coefficient) {
        const std::int64_t value =
            static_cast<std::int64_t>((coefficient * 7U + component * 3U) %
                                      19U) -
            9;
        signed_values[component * _degree + coefficient] = value;
        for (std::size_t modulus_index = 0;
             modulus_index < Get_ciph_prime_cnt(cipher); ++modulus_index) {
          Set_coeff_at(polynomial,
                       static_cast<std::int64_t>(
                           ReduceSigned(value, Modulus(modulus_index))),
                       modulus_index * _degree + coefficient);
        }
      }
    }
    return signed_values;
  }

  std::vector<std::int64_t> Snapshot(CIPHERTEXT *cipher) const {
    const std::size_t component_length = _degree * Get_ciph_prime_cnt(cipher);
    std::vector<std::int64_t> result(2U * component_length);
    std::memcpy(result.data(), Get_poly_coeffs(Get_c0(cipher)),
                component_length * sizeof(std::int64_t));
    std::memcpy(result.data() + component_length,
                Get_poly_coeffs(Get_c1(cipher)),
                component_length * sizeof(std::int64_t));
    return result;
  }

  void ExpectMetadataEqual(CIPHERTEXT *left, CIPHERTEXT *right) const {
    EXPECT_EQ(Get_ciph_degree(left), Get_ciph_degree(right));
    EXPECT_EQ(Get_ciph_prime_cnt(left), Get_ciph_prime_cnt(right));
    EXPECT_EQ(Get_ciph_prime_p_cnt(left), Get_ciph_prime_p_cnt(right));
    EXPECT_EQ(Get_ciph_sfactor(left), Get_ciph_sfactor(right));
    EXPECT_EQ(Get_ciph_sf_degree(left), Get_ciph_sf_degree(right));
    EXPECT_EQ(Get_ciph_slots(left), Get_ciph_slots(right));
    EXPECT_EQ(Is_ntt(Get_c0(left)), Is_ntt(Get_c0(right)));
    EXPECT_EQ(Is_ntt(Get_c1(left)), Is_ntt(Get_c1(right)));
  }

  void ExpectNegacyclic(CIPHERTEXT *result,
                        const std::vector<std::int64_t> &signed_values,
                        std::uint64_t power) const {
    const std::uint64_t normalized = power % (2U * _degree);
    for (std::size_t component = 0; component < 2; ++component) {
      POLYNOMIAL *polynomial = component == 0 ? Get_c0(result) : Get_c1(result);
      for (std::size_t modulus_index = 0; modulus_index < _full_q_count;
           ++modulus_index) {
        const std::uint64_t modulus = Modulus(modulus_index);
        std::vector<std::uint64_t> expected(_degree, 0);
        for (std::size_t coefficient = 0; coefficient < _degree;
             ++coefficient) {
          const std::uint64_t destination_unreduced = coefficient + normalized;
          const std::size_t destination = destination_unreduced % _degree;
          const bool negate = (destination_unreduced / _degree) % 2U != 0;
          std::uint64_t value = ReduceSigned(
              signed_values[component * _degree + coefficient], modulus);
          if (negate && value != 0)
            value = modulus - value;
          expected[destination] = value;
        }
        for (std::size_t coefficient = 0; coefficient < _degree;
             ++coefficient) {
          EXPECT_EQ(static_cast<std::uint64_t>(Get_coeff_at(
                        polynomial, modulus_index * _degree + coefficient)),
                    expected[coefficient])
              << "component=" << component << " modulus=" << modulus_index
              << " coefficient=" << coefficient << " power=" << power;
        }
      }
    }
  }

  std::vector<Complex> TestValues() const {
    std::vector<Complex> result;
    result.reserve(_slots);
    for (std::size_t index = 0; index < _slots; ++index) {
      result.emplace_back(
          (static_cast<double>((index * 7U) % 23U) - 11.0) / 16.0,
          (static_cast<double>((index * 5U + 3U) % 19U) - 9.0) / 16.0);
    }
    return result;
  }

  CIPHER EncryptValues(const std::vector<Complex> &values) const {
    PLAIN plain = Alloc_plaintext();
    std::vector<DCMPLX> input(values.begin(), values.end());
    Encode_dcmplx(plain, input.data(), input.size(), 1, 1);
    CIPHER cipher = Alloc_ciphertext();
    Encrypt(cipher, plain);
    Free_plaintext(plain);
    return cipher;
  }

  std::vector<Complex> DecodeValues(CIPHER cipher) const {
    DCMPLX *raw = Get_msg_with_imag(cipher);
    EXPECT_NE(raw, nullptr);
    std::vector<Complex> result(raw, raw + _slots);
    std::free(raw);
    return result;
  }

  void ExpectApprox(const std::vector<Complex> &actual,
                    const std::vector<Complex> &expected,
                    double tolerance = 0.005) const {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
      EXPECT_LE(std::abs(actual[index] - expected[index]), tolerance)
          << "slot=" << index;
    }
  }

  std::uint32_t _degree = 0;
  std::uint32_t _slots = 0;
  std::size_t _full_q_count = 0;
  std::array<std::int32_t, 3> _rotation_steps{};
};

TEST_F(TEST_CKKS_EXTENDED_OPS, CenteredRaiseIsExactAndOutOfPlace) {
  CIPHERTEXT source = MakeCoefficientCipher(1);
  const std::uint64_t q0 = Modulus(0);
  const std::array<std::uint64_t, 6> edge_values = {
      0, 1, q0 / 2U, q0 / 2U + 1U, q0 - 1U, q0 - 2U};
  for (std::size_t component = 0; component < 2; ++component) {
    POLYNOMIAL *polynomial = component == 0 ? Get_c0(&source) : Get_c1(&source);
    for (std::size_t coefficient = 0; coefficient < _degree; ++coefficient) {
      Set_coeff_at(
          polynomial,
          static_cast<std::int64_t>(
              edge_values[(coefficient + component) % edge_values.size()]),
          coefficient);
    }
  }
  const auto source_before = Snapshot(&source);
  CIPHERTEXT result{};
  Raise_mod(&result, &source, static_cast<std::uint32_t>(_full_q_count));
  EXPECT_EQ(Snapshot(&source), source_before);
  EXPECT_FALSE(Is_ntt(Get_c0(&source)));
  EXPECT_EQ(Get_ciph_prime_cnt(&result), _full_q_count);
  EXPECT_TRUE(Is_ntt(Get_c0(&result)));
  EXPECT_EQ(Get_ciph_sfactor(&result), Get_ciph_sfactor(&source));
  EXPECT_EQ(Get_ciph_sf_degree(&result), Get_ciph_sf_degree(&source));

  Conv_ntt2poly_inplace(Get_c0(&result));
  Conv_ntt2poly_inplace(Get_c1(&result));
  const auto coefficient_raised = Snapshot(&result);
  for (std::size_t component = 0; component < 2; ++component) {
    POLYNOMIAL *source_poly =
        component == 0 ? Get_c0(&source) : Get_c1(&source);
    POLYNOMIAL *result_poly =
        component == 0 ? Get_c0(&result) : Get_c1(&result);
    for (std::size_t coefficient = 0; coefficient < _degree; ++coefficient) {
      const std::uint64_t residue =
          static_cast<std::uint64_t>(Get_coeff_at(source_poly, coefficient));
      const std::int64_t centered =
          residue <= q0 / 2U ? static_cast<std::int64_t>(residue)
                             : -static_cast<std::int64_t>(q0 - residue);
      for (std::size_t modulus_index = 0; modulus_index < _full_q_count;
           ++modulus_index) {
        EXPECT_EQ(static_cast<std::uint64_t>(Get_coeff_at(
                      result_poly, modulus_index * _degree + coefficient)),
                  ReduceSigned(centered, Modulus(modulus_index)));
      }
    }
  }

  CIPHERTEXT ntt_source{};
  Copy_ciphertext(&ntt_source, &source);
  Conv_poly2ntt_inplace(Get_c0(&ntt_source));
  Conv_poly2ntt_inplace(Get_c1(&ntt_source));
  const auto ntt_source_before = Snapshot(&ntt_source);
  CIPHERTEXT ntt_result{};
  Raise_mod(&ntt_result, &ntt_source,
            static_cast<std::uint32_t>(_full_q_count));
  EXPECT_EQ(Snapshot(&ntt_source), ntt_source_before);
  EXPECT_TRUE(Is_ntt(Get_c0(&ntt_source)));
  EXPECT_TRUE(Is_ntt(Get_c0(&ntt_result)));
  Conv_ntt2poly_inplace(Get_c0(&ntt_result));
  Conv_ntt2poly_inplace(Get_c1(&ntt_result));
  EXPECT_EQ(Snapshot(&ntt_result), coefficient_raised);

  Zero_ciph(&ntt_result);
  Zero_ciph(&ntt_source);
  Zero_ciph(&result);
  Zero_ciph(&source);
}

TEST_F(TEST_CKKS_EXTENDED_OPS,
       MonomialIsExactForCoefficientAndNttFormsAndNormalizesPower) {
  const std::array<std::uint64_t, 6> powers = {0U,
                                               _degree / 2U,
                                               _degree,
                                               3U * _degree / 2U,
                                               2U * _degree - 1U,
                                               2U * _degree + 1U};
  for (std::uint64_t power : powers) {
    CIPHERTEXT source = MakeCoefficientCipher(_full_q_count);
    const auto signed_values = FillSignedPattern(&source);
    CIPHERTEXT coefficient_result{};
    Mul_mono_ciph(&coefficient_result, &source,
                  static_cast<std::uint32_t>(power));
    // ANT's Mul_poly converts coefficient operands to NTT in place.  This is
    // an observable legacy behavior of Mul_mono_ciph, so do not claim that a
    // coefficient-form source is byte-for-byte preserved.
    EXPECT_TRUE(Is_ntt(Get_c0(&source)));
    EXPECT_TRUE(Is_ntt(Get_c1(&source)));
    ExpectMetadataEqual(&coefficient_result, &source);
    Conv_ntt2poly_inplace(Get_c0(&coefficient_result));
    Conv_ntt2poly_inplace(Get_c1(&coefficient_result));
    ExpectNegacyclic(&coefficient_result, signed_values, power);

    CIPHERTEXT ntt_source = MakeCoefficientCipher(_full_q_count);
    FillSignedPattern(&ntt_source);
    Conv_poly2ntt_inplace(Get_c0(&ntt_source));
    Conv_poly2ntt_inplace(Get_c1(&ntt_source));
    const auto ntt_source_before = Snapshot(&ntt_source);
    CIPHERTEXT ntt_result{};
    Mul_mono_ciph(&ntt_result, &ntt_source, static_cast<std::uint32_t>(power));
    EXPECT_EQ(Snapshot(&ntt_source), ntt_source_before);
    EXPECT_TRUE(Is_ntt(Get_c0(&ntt_result)));
    Conv_ntt2poly_inplace(Get_c0(&ntt_result));
    Conv_ntt2poly_inplace(Get_c1(&ntt_result));
    ExpectNegacyclic(&ntt_result, signed_values, power);

    Zero_ciph(&ntt_result);
    Zero_ciph(&ntt_source);
    Zero_ciph(&coefficient_result);
    Zero_ciph(&source);
  }
}

TEST_F(TEST_CKKS_EXTENDED_OPS, MonomialIdentityNegationInverseAndInPlace) {
  CIPHERTEXT source = MakeCoefficientCipher(_full_q_count);
  const auto signed_values = FillSignedPattern(&source);
  CIPHERTEXT identity{};
  Mul_mono_ciph(&identity, &source, 0);
  EXPECT_EQ(Snapshot(&identity), Snapshot(&source));

  CIPHERTEXT negated{};
  Mul_mono_ciph(&negated, &source, _degree);
  ExpectNegacyclic(&negated, signed_values, _degree);

  CIPHERTEXT inverse_first{};
  CIPHERTEXT inverse_result{};
  Mul_mono_ciph(&inverse_first, &source, 2U * _degree - 1U);
  Mul_mono_ciph(&inverse_result, &inverse_first, 1U);
  EXPECT_EQ(Snapshot(&inverse_result), Snapshot(&source));

  CIPHERTEXT in_place{};
  Copy_ciphertext(&in_place, &source);
  Mul_mono_ciph(&in_place, &in_place, _degree);
  ExpectNegacyclic(&in_place, signed_values, _degree);

  Zero_ciph(&in_place);
  Zero_ciph(&inverse_result);
  Zero_ciph(&inverse_first);
  Zero_ciph(&negated);
  Zero_ciph(&identity);
  Zero_ciph(&source);
}

TEST_F(TEST_CKKS_EXTENDED_OPS,
       ConjugateAndOrderedBatchMatchIndependentDecodedValues) {
  const auto values = TestValues();
  CIPHER source = EncryptValues(values);
  const auto source_before = Snapshot(source);

  CIPHER conjugated = Alloc_ciphertext();
  Conjugate_ciph(conjugated, source);
  std::vector<Complex> expected_conjugate(values);
  for (Complex &value : expected_conjugate)
    value = std::conj(value);
  ExpectApprox(DecodeValues(conjugated), expected_conjugate);
  EXPECT_EQ(Snapshot(source), source_before);

  CIPHER twice = Alloc_ciphertext();
  Conjugate_ciph(twice, conjugated);
  ExpectApprox(DecodeValues(twice), values);

  CIPHER in_place = Alloc_ciphertext();
  Copy_ciphertext(in_place, source);
  Conjugate_ciph(in_place, in_place);
  ExpectApprox(DecodeValues(in_place), expected_conjugate);

  const std::array<std::int32_t, 4> steps = {5, 0, -7, 5};
  CIPHER outputs =
      static_cast<CIPHER>(std::calloc(steps.size(), sizeof(CIPHERTEXT)));
  ASSERT_NE(outputs, nullptr);
  Rotate_batch_ciph(outputs, source, steps.data(), steps.size());
  std::set<const std::int64_t *> buffers;
  for (std::size_t output = 0; output < steps.size(); ++output) {
    EXPECT_TRUE(
        buffers.insert(Get_poly_coeffs(Get_c0(&outputs[output]))).second);
    std::vector<Complex> expected(_slots);
    const std::int64_t normalized =
        (static_cast<std::int64_t>(steps[output]) % _slots + _slots) % _slots;
    for (std::size_t slot = 0; slot < _slots; ++slot) {
      expected[slot] = values[(slot + normalized) % _slots];
    }
    ExpectApprox(DecodeValues(&outputs[output]), expected);
  }
  EXPECT_EQ(Snapshot(&outputs[1]), Snapshot(source));
  EXPECT_EQ(Snapshot(&outputs[0]), Snapshot(&outputs[3]));
  EXPECT_EQ(Snapshot(source), source_before);

  Free_ciph_poly(outputs, steps.size());
  std::free(outputs);
  Free_ciphertext(in_place);
  Free_ciphertext(twice);
  Free_ciphertext(conjugated);
  Free_ciphertext(source);
}

} // namespace
