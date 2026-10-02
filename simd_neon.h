#ifndef SIMD_NEON_H
#define SIMD_NEON_H

/*
 * Native NEON kernels for aarch64 (e.g. Raspberry Pi 4 / Cortex-A72).
 *
 * On non-x86 targets the AVX-512 code of labrados is compiled through SIMDe,
 * which emulates many of the intrinsics (16-bit high multiplies, mask
 * operations, VNNI dot products, compress stores, 64-bit arithmetic shifts,
 * cross-lane permutes) lane by lane. The hot kernels therefore have native
 * NEON versions guarded by LABRADOS_NEON. Every NEON kernel computes results
 * that are bit-identical to the AVX-512 code for all inputs (all integer
 * arithmetic below wraps exactly like the corresponding AVX-512 instruction).
 *
 * Define LABRADOS_FORCE_AVX512 (or JL_FORCE_AVX512) to compile the original
 * AVX-512/SIMDe code paths instead (used by the equivalence tests).
 */

#if defined(__aarch64__) && !defined(LABRADOS_FORCE_AVX512) && !defined(JL_FORCE_AVX512) \
    && !(defined(__x86_64__) && !defined(LAZER_PORTABLE))
#define LABRADOS_NEON 1
#include <arm_neon.h>

/* exact _mm_mulhi_epi16 for arbitrary a, b */
static inline int16x8_t neon_mulhi_s16(int16x8_t a, int16x8_t b) {
  int32x4_t lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
  int32x4_t hi = vmull_high_s16(a, b);
  return vuzp2q_s16(vreinterpretq_s16_s32(lo), vreinterpretq_s16_s32(hi));
}

/* exact _mm_mulhi_epi16 if no lane pair is (-32768,-32768), e.g. if all lanes of
 * c are != -32768: sqdmulh = floor(2ab/2^16) (no saturation), >>1 = floor(ab/2^16) */
static inline int16x8_t neon_mulhi_s16_c(int16x8_t a, int16x8_t c) {
  return vshrq_n_s16(vqdmulhq_s16(a, c), 1);
}

/* exact _mm_mulhi_epu16 */
static inline int16x8_t neon_mulhi_u16(int16x8_t a, int16x8_t b) {
  uint16x8_t ua = vreinterpretq_u16_s16(a), ub = vreinterpretq_u16_s16(b);
  uint32x4_t lo = vmull_u16(vget_low_u16(ua), vget_low_u16(ub));
  uint32x4_t hi = vmull_high_u16(ua, ub);
  return vreinterpretq_s16_u16(vuzp2q_u16(vreinterpretq_u16_u32(lo), vreinterpretq_u16_u32(hi)));
}

/* Montgomery reduction of 8 int32 values (x0: coeffs 0-3, x1: coeffs 4-7) to int16,
 * exactly as the AVX-512 code: hi16(x) - mulhi16(lo16(x)*pinv, p) (16-bit wrapping).
 * Requires p != -32768 (p is a 13-14 bit prime). */
static inline int16x8_t neon_montred32(int32x4_t x0, int32x4_t x1, int16x8_t pinv, int16x8_t p) {
  int16x8_t lo = vuzp1q_s16(vreinterpretq_s16_s32(x0), vreinterpretq_s16_s32(x1));
  int16x8_t hi = vuzp2q_s16(vreinterpretq_s16_s32(x0), vreinterpretq_s16_s32(x1));
  return vsubq_s16(hi, neon_mulhi_s16_c(vmulq_s16(lo, pinv), p));
}

#endif

#endif
