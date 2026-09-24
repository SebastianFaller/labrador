#ifndef SIMD_H
#define SIMD_H

/*
 * SIMD abstraction for labrador.
 *
 * On x86-64 the native AVX-512 intrinsics are used. On all other targets
 * (e.g. aarch64 / Raspberry Pi 4), or when LAZER_PORTABLE is defined, the
 * AVX-512 intrinsics are provided by SIMDe (https://github.com/simd-everywhere/simde),
 * which maps them to NEON or portable C. Build with
 *   -I<simde dir> -DSIMDE_ENABLE_NATIVE_ALIASES
 * The few intrinsics that SIMDe does not implement are provided below.
 */

#if defined(__x86_64__) && !defined(LAZER_PORTABLE)

#include <immintrin.h>

#else

#include <stdint.h>
#include <string.h>

#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif
#include <simde/x86/avx512.h>
#include "../aes-ct64.h"  /* constant-time bitsliced AES (no AES hardware) */

typedef union {
  uint8_t u8[64];
  uint16_t u16[32];
  int16_t i16[32];
  uint32_t u32[16];
  int64_t i64[8];
  uint64_t u64[8];
} simd_lanes512;

typedef union {
  uint8_t u8[32];
  uint32_t u32[8];
} simd_lanes256;

typedef union {
  uint8_t u8[16];
  uint32_t u32[4];
} simd_lanes128;

#define SIMD_LOAD512(dst, src) memcpy(&(dst), &(src), 64)
#define SIMD_LOAD256(dst, src) memcpy(&(dst), &(src), 32)
#define SIMD_LOAD128(dst, src) memcpy(&(dst), &(src), 16)

/*
 * VAES: one AES round on each of the four 128-bit lanes, computed
 * constant-time with the bitsliced BearSSL code (all 4 lanes at once).
 * SubBytes and ShiftRows commute, so the order matches AESENC.
 */
static inline simde__m512i simd_aes_round4(simde__m512i a, simde__m512i k, int last) {
  uint8_t buf[64], key[64];
  uint32_t w[16];
  uint64_t q[8];
  int i;

  memcpy(buf, &a, 64);
  memcpy(key, &k, 64);
  for(i=0;i<16;i++)
    w[i] = _aes_dec32le(buf + 4*i);
  for(i=0;i<4;i++)
    _aes_ct64_interleave_in(&q[i], &q[i+4], w + 4*i);
  _aes_ct64_ortho(q);
  _aes_ct64_bitslice_sbox(q);
  _aes_ct64_shift_rows(q);
  if(!last)
    _aes_ct64_mix_columns(q);
  _aes_ct64_ortho(q);
  for(i=0;i<4;i++)
    _aes_ct64_interleave_out(w + 4*i, q[i], q[i+4]);
  for(i=0;i<16;i++)
    _aes_enc32le(buf + 4*i, w[i]);
  for(i=0;i<64;i++)
    buf[i] ^= key[i];
  memcpy(&a, buf, 64);
  return a;
}

static inline simde__m512i simd_mm512_aesenc_epi128(simde__m512i a, simde__m512i k) {
  return simd_aes_round4(a, k, 0);
}

static inline simde__m512i simd_mm512_aesenclast_epi128(simde__m512i a, simde__m512i k) {
  return simd_aes_round4(a, k, 1);
}

/* AES-NI key expansion assist (Intel SDM definition), constant-time */
static inline simde__m128i simd_mm_aeskeygenassist_si128(simde__m128i a, int rcon) {
  simd_lanes128 x, r;
  uint32_t x1, x3;

  SIMD_LOAD128(x, a);
  x1 = _aes_ct64_sub_word(x.u32[1]);
  x3 = _aes_ct64_sub_word(x.u32[3]);
  r.u32[0] = x1;
  r.u32[1] = ((x1 >> 8) | (x1 << 24)) ^ (uint32_t)rcon;
  r.u32[2] = x3;
  r.u32[3] = ((x3 >> 8) | (x3 << 24)) ^ (uint32_t)rcon;
  SIMD_LOAD128(a, r);
  return a;
}

/* per 128-bit lane: (a:b) >> 8*imm */
static inline simde__m512i simd_mm512_alignr_epi8(simde__m512i a, simde__m512i b, int imm) {
  simd_lanes512 x, y, r;
  int i, j, k;

  SIMD_LOAD512(x, a);
  SIMD_LOAD512(y, b);
  for(i=0;i<4;i++) {
    for(j=0;j<16;j++) {
      k = j + imm;
      if(k < 16)      r.u8[16*i+j] = y.u8[16*i+k];
      else if(k < 32) r.u8[16*i+j] = x.u8[16*i+k-16];
      else            r.u8[16*i+j] = 0;
    }
  }
  SIMD_LOAD512(a, r);
  return a;
}

static inline simde__m512i simd_mm512_broadcast_i64x2(simde__m128i a) {
  return simde_mm512_broadcast_i32x4(a);
}

static inline simde__m512i simd_mm512_cvtepu32_epi64(simde__m256i a) {
  simd_lanes256 x;
  simd_lanes512 r;
  simde__m512i v;
  int i;

  SIMD_LOAD256(x, a);
  for(i=0;i<8;i++)
    r.u64[i] = x.u32[i];
  SIMD_LOAD512(v, r);
  return v;
}

static inline simde__m512i simd_mm512_cvtepu8_epi64(simde__m128i a) {
  simd_lanes128 x;
  simd_lanes512 r;
  simde__m512i v;
  int i;

  SIMD_LOAD128(x, a);
  for(i=0;i<8;i++)
    r.u64[i] = x.u8[i];
  SIMD_LOAD512(v, r);
  return v;
}

static inline simde__m512i simd_mm512_mask_sub_epi16(simde__m512i src, simde__mmask32 k, simde__m512i a, simde__m512i b) {
  return simde_mm512_mask_mov_epi16(src, k, simde_mm512_sub_epi16(a, b));
}

static inline simde__m512i simd_mm512_maskz_sub_epi16(simde__mmask32 k, simde__m512i a, simde__m512i b) {
  return simde_mm512_maskz_mov_epi16(k, simde_mm512_sub_epi16(a, b));
}

static inline simde__m512 simd_mm512_moveldup_ps(simde__m512 a) {
  simd_lanes512 x;
  int i;

  SIMD_LOAD512(x, a);
  for(i=0;i<16;i+=2)
    x.u32[i+1] = x.u32[i];  /* bit copy: used on integer data */
  SIMD_LOAD512(a, x);
  return a;
}

static inline simde__m512i simd_mm512_mulhi_epu16(simde__m512i a, simde__m512i b) {
  simd_lanes512 x, y;
  int i;

  SIMD_LOAD512(x, a);
  SIMD_LOAD512(y, b);
  for(i=0;i<32;i++)
    x.u16[i] = (uint16_t)(((uint32_t)x.u16[i] * y.u16[i]) >> 16);
  SIMD_LOAD512(a, x);
  return a;
}

static inline simde__m512i simd_mm512_srai_epi64(simde__m512i a, unsigned int imm) {
  simd_lanes512 x;
  int i;

  if(imm > 63) imm = 63;
  SIMD_LOAD512(x, a);
  for(i=0;i<8;i++)
    x.i64[i] >>= imm;
  SIMD_LOAD512(a, x);
  return a;
}

/* mask register conversions and popcount */
static inline simde__mmask32 simd_cvtu32_mask32(uint32_t a) { return (simde__mmask32)a; }
static inline uint32_t simd_cvtmask32_u32(simde__mmask32 a) { return (uint32_t)a; }
static inline simde__mmask64 simd_cvtu64_mask64(uint64_t a) { return (simde__mmask64)a; }
static inline int simd_popcnt32(uint32_t a) { return __builtin_popcount(a); }

#undef _cvtu32_mask32
#undef _cvtmask32_u32
#undef _cvtu64_mask64
#undef _popcnt32
#define _cvtu32_mask32(a) simd_cvtu32_mask32(a)
#define _cvtmask32_u32(a) simd_cvtmask32_u32(a)
#define _cvtu64_mask64(a) simd_cvtu64_mask64(a)
#define _popcnt32(a) simd_popcnt32(a)

#undef _mm512_aesenc_epi128
#undef _mm512_aesenclast_epi128
#undef _mm_aeskeygenassist_si128
#undef _mm512_alignr_epi8
#undef _mm512_broadcast_i64x2
#undef _mm512_cvtepu32_epi64
#undef _mm512_cvtepu8_epi64
#undef _mm512_mask_sub_epi16
#undef _mm512_maskz_sub_epi16
#undef _mm512_moveldup_ps
#undef _mm512_mulhi_epu16
#undef _mm512_srai_epi64
#define _mm512_aesenc_epi128(a, k) simd_mm512_aesenc_epi128(a, k)
#define _mm512_aesenclast_epi128(a, k) simd_mm512_aesenclast_epi128(a, k)
#define _mm_aeskeygenassist_si128(a, imm) simd_mm_aeskeygenassist_si128(a, imm)
#define _mm512_alignr_epi8(a, b, imm) simd_mm512_alignr_epi8(a, b, imm)
#define _mm512_broadcast_i64x2(a) simd_mm512_broadcast_i64x2(a)
#define _mm512_cvtepu32_epi64(a) simd_mm512_cvtepu32_epi64(a)
#define _mm512_cvtepu8_epi64(a) simd_mm512_cvtepu8_epi64(a)
#define _mm512_mask_sub_epi16(src, k, a, b) simd_mm512_mask_sub_epi16(src, k, a, b)
#define _mm512_maskz_sub_epi16(k, a, b) simd_mm512_maskz_sub_epi16(k, a, b)
#define _mm512_moveldup_ps(a) simd_mm512_moveldup_ps(a)
#define _mm512_mulhi_epu16(a, b) simd_mm512_mulhi_epu16(a, b)
#define _mm512_srai_epi64(a, imm) simd_mm512_srai_epi64(a, imm)

#endif

#endif
