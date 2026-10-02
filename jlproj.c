#include <stdalign.h>
#include <stdint.h>
#include "simd.h"
#include "simd_neon.h"
#include <stdlib.h>
#include <string.h>
#include "data.h"
#include "polz.h"
#include "jlproj.h"

/* Substracting 2 iid random variables with P(-1)=0.5, P(1)=0.5
 * gives random variable with P(-2)=0.25, P(0)=0.5, P(2)=0.25.
 * Divide by 2 to obtain the correct distribution:
 * P(-1)=0.25, P(0)=0.5, P(1)=0.25
 *
 * m1 = mat1
 * m2 = mat2 // interpreted as negative
 * r += ((m1 - (-m2)) / 2) * p
 */
void polyvec_jlproj_add_bin1(
  int32_t r[256], 
  const poly *p, 
  size_t len, 
  const uint8_t *mat1,
  const uint8_t *mat2
) 
{
  size_t i;
  polyvec_jlproj_add(r, p, len, mat1);
  polyvec_jlproj_add(r, p, len, mat2);
  for (i = 0; i < 256; i++)
    r[i] >>= 1;
}

/* matrix format (32x16):
 * 4 bits -> nibble: 8 row stride
 * 2 nibbles -> byte: 8 col stride
 * 4 bytes -> dword: 1 row stride
 * 2 dwords -> qword: 4 row stride
 * 8 qwords -> zvec: 1 col stride
 */

#ifdef LABRADOS_NEON
/* NEON version of jlproj_add64 (out[row] += sum_col (+-1) in[col], mod 2^32).
 *
 * Bit (row,col) of the matrix (row in [0,256), col in [0,64)) lives in byte
 *   512*(col/16) + 64*(row/32) + 8*(col%8) + (row%8),  bit 4*((col%16)/8) + (row%32)/8,
 * and a set bit means -in[col]. As in the AVX-512 code, the 64 columns are grouped
 * into 16 groups {c, c+16, c+32, c+48}; for every group a 16-entry table holds all
 * signed sums of the 4 (secret) coefficients, indexed by a nibble of (public) matrix
 * bits. The 16-entry int32 tables are split into byte planes so that a single TBL
 * looks up 16 rows at once: T = p0 + 2^8*p1 + 2^16*p2 with p0,p1 the two low bytes
 * and p2 = T>>16 in [-2,1] (|T| <= 2^17). The planes are accumulated separately
 * (16 tables: sum p0,p1 <= 4080 fits uint16, sum p2 in [-32,16] fits int8) and
 * recombined, which gives exactly out + sum_t T_t mod 2^32, i.e. the same result as
 * the wrapping 32-bit additions of the AVX-512 code.
 * Constant time: the table contents are secret, but they are only accessed through
 * TBL register lookups and loads at fixed addresses; the lookup indices are public
 * matrix bits. No branches depend on data. */
static void jlproj_add64(int32_t out[256], const int16_t in[64], const uint8_t mat[256*64/8]) {
  size_t c,j,q,rb,h;
  uint8_t pl[16][3][16] __attribute__((aligned(16)));  // [table][plane][index]
  const int32x4_t sgn0 = {1,-1,1,-1};
  const int32x4_t sgn1 = {1,1,-1,-1};

  /* tables: T_c[idx] = sum_j (bit j of idx ? -1 : 1)*in[16*j+c] = E[idx&3] + G[idx>>2] */
  for(c=0;c<16;c++) {
    const int32_t a0 = in[c], a1 = in[16+c], a2 = in[32+c], a3 = in[48+c];
    const int32_t g[4] = {a2+a3, a3-a2, a2-a3, -a2-a3};
    const int32x4_t e = vmlaq_n_s32(vmulq_n_s32(sgn0,a0),sgn1,a1);
    int32x4_t t[4];
    int16x8_t lo01, lo23, hi01, hi23;
    for(j=0;j<4;j++)
      t[j] = vaddq_s32(e,vdupq_n_s32(g[j]));
    lo01 = vcombine_s16(vmovn_s32(t[0]),vmovn_s32(t[1]));
    lo23 = vcombine_s16(vmovn_s32(t[2]),vmovn_s32(t[3]));
    hi01 = vcombine_s16(vshrn_n_s32(t[0],16),vshrn_n_s32(t[1],16));
    hi23 = vcombine_s16(vshrn_n_s32(t[2],16),vshrn_n_s32(t[3],16));
    vst1q_u8(pl[c][0],vreinterpretq_u8_s8(vcombine_s8(vmovn_s16(lo01),vmovn_s16(lo23))));
    vst1q_u8(pl[c][1],vreinterpretq_u8_s8(vcombine_s8(vshrn_n_s16(lo01,8),vshrn_n_s16(lo23,8))));
    vst1q_u8(pl[c][2],vreinterpretq_u8_s8(vcombine_s8(vmovn_s16(hi01),vmovn_s16(hi23))));
  }

  for(rb=0;rb<8;rb++) {
    /* idx[k][q]: byte b holds in its low nibble the index of table 2q+b/8 and in its
     * high nibble the index of table 2q+b/8+8, for row 32*rb+8*k+b%8.
     * Nibble bit j comes from column block j (4x4 bit transpose). */
    uint8x16_t idx[4][4];
    const uint8x16_t m55 = vdupq_n_u8(0x55), m33 = vdupq_n_u8(0x33);
    for(q=0;q<4;q++) {
      uint8x16_t s = vld1q_u8(&mat[64*rb+16*q+512*0]);
      uint8x16_t t = vld1q_u8(&mat[64*rb+16*q+512*1]);
      uint8x16_t u = vld1q_u8(&mat[64*rb+16*q+512*2]);
      uint8x16_t v = vld1q_u8(&mat[64*rb+16*q+512*3]);
      uint8x16_t ste = vbslq_u8(m55,s,vshlq_n_u8(t,1));
      uint8x16_t sto = vbslq_u8(m55,vshrq_n_u8(s,1),t);
      uint8x16_t uve = vbslq_u8(m55,u,vshlq_n_u8(v,1));
      uint8x16_t uvo = vbslq_u8(m55,vshrq_n_u8(u,1),v);
      idx[0][q] = vbslq_u8(m33,ste,vshlq_n_u8(uve,2));
      idx[1][q] = vbslq_u8(m33,sto,vshlq_n_u8(uvo,2));
      idx[2][q] = vbslq_u8(m33,vshrq_n_u8(ste,2),uve);
      idx[3][q] = vbslq_u8(m33,vshrq_n_u8(sto,2),uvo);
    }

    for(h=0;h<2;h++) {  // rows 32*rb+16*h+[0,16)
      uint16x8_t a0l = vdupq_n_u16(0), a0h = a0l, a1l = a0l, a1h = a0l;
      int8x16_t a2 = vdupq_n_s8(0);
      int32_t *o = &out[32*rb+16*h];
      int32x4_t o0,o1,o2,o3;
      int16x8_t s2l, s2h;

#define LOOKUP(C,IX) do {                                         \
        uint8x16_t b0_ = vqtbl1q_u8(vld1q_u8(pl[C][0]),IX);       \
        uint8x16_t b1_ = vqtbl1q_u8(vld1q_u8(pl[C][1]),IX);       \
        uint8x16_t b2_ = vqtbl1q_u8(vld1q_u8(pl[C][2]),IX);       \
        a0l = vaddw_u8(a0l,vget_low_u8(b0_));                     \
        a0h = vaddw_high_u8(a0h,b0_);                             \
        a1l = vaddw_u8(a1l,vget_low_u8(b1_));                     \
        a1h = vaddw_high_u8(a1h,b1_);                             \
        a2 = vaddq_s8(a2,vreinterpretq_s8_u8(b2_));               \
      } while(0)

      for(q=0;q<4;q++) {
        const uint64x2_t x0 = vreinterpretq_u64_u8(idx[2*h][q]);
        const uint64x2_t x1 = vreinterpretq_u64_u8(idx[2*h+1][q]);
        const uint8x16_t za = vreinterpretq_u8_u64(vzip1q_u64(x0,x1));  // tables 2q, 2q+8
        const uint8x16_t zb = vreinterpretq_u8_u64(vzip2q_u64(x0,x1));  // tables 2q+1, 2q+9
        const uint8x16_t m0f = vdupq_n_u8(0x0F);
        LOOKUP(2*q,  vandq_u8(za,m0f));
        LOOKUP(2*q+8,vshrq_n_u8(za,4));
        LOOKUP(2*q+1,vandq_u8(zb,m0f));
        LOOKUP(2*q+9,vshrq_n_u8(zb,4));
      }
#undef LOOKUP

      s2l = vmovl_s8(vget_low_s8(a2));
      s2h = vmovl_high_s8(a2);
      o0 = vld1q_s32(&o[0]);
      o1 = vld1q_s32(&o[4]);
      o2 = vld1q_s32(&o[8]);
      o3 = vld1q_s32(&o[12]);
      o0 = vaddq_s32(o0,vreinterpretq_s32_u32(vaddw_u16(vshll_n_u16(vget_low_u16(a1l),8),vget_low_u16(a0l))));
      o1 = vaddq_s32(o1,vreinterpretq_s32_u32(vaddw_high_u16(vshll_high_n_u16(a1l,8),a0l)));
      o2 = vaddq_s32(o2,vreinterpretq_s32_u32(vaddw_u16(vshll_n_u16(vget_low_u16(a1h),8),vget_low_u16(a0h))));
      o3 = vaddq_s32(o3,vreinterpretq_s32_u32(vaddw_high_u16(vshll_high_n_u16(a1h,8),a0h)));
      o0 = vaddq_s32(o0,vshll_n_s16(vget_low_s16(s2l),16));
      o1 = vaddq_s32(o1,vshll_high_n_s16(s2l,16));
      o2 = vaddq_s32(o2,vshll_n_s16(vget_low_s16(s2h),16));
      o3 = vaddq_s32(o3,vshll_high_n_s16(s2h,16));
      vst1q_s32(&o[0],o0);
      vst1q_s32(&o[4],o1);
      vst1q_s32(&o[8],o2);
      vst1q_s32(&o[12],o3);
    }
  }
}
#else
static void jlproj_add64(int32_t out[256], const int16_t in[64], const uint8_t mat[256*64/8]) {
  int i;
  __m512i a,b,c,d;
  __m512i e,f,g,h;
  __m512i k,l,m,n;
  __m512i o,p,q,r;
  __m512i s,t,u,v;
  __m512i w,x,y,z;
  const __m512i zero = _mm512_setzero_si512();
  const __m512i stencil1 = _mm512_set1_epi8(0x11);
  const __m512i stencil2 = _mm512_set1_epi8(0x22);
  const __m512i stencil4 = _mm512_set1_epi8(0x44);
  const __m512i stencil8 = _mm512_set1_epi8((char)0x88);
  const __m512i vpermi2bidx = _mm512_set_epi8(95,87,79,71,
                                              94,86,78,70,
                                              93,85,77,69,
                                              92,84,76,68,
                                              91,83,75,67,
                                              90,82,74,66,
                                              89,81,73,65,
                                              88,80,72,64,
                                              31,23,15, 7,
                                              30,22,14, 6,
                                              29,21,13, 5,
                                              28,20,12, 4,
                                              27,19,11, 3,
                                              26,18,10, 2,
                                              25,17, 9, 1,
                                              24,16, 8, 0);

  /* Prepare 16 vectors containing all 16 signed combinations of 4 polynomial coeffs that are 16 apart */
  a = _mm512_cvtepi16_epi32(_mm256_load_si256((__m256i*)&in[48]));
  b = _mm512_cvtepi16_epi32(_mm256_load_si256((__m256i*)&in[32]));

  e = _mm512_add_epi32(a,b);     // ++
  f = _mm512_sub_epi32(a,b);     // +-
  g = _mm512_sub_epi32(b,a);     // -+
  h = _mm512_sub_epi32(zero,e);  // --

  a = _mm512_shuffle_i64x2(e,f,0x44);  // ++,+-
  b = _mm512_shuffle_i64x2(e,f,0xEE);  // ++,+-
  c = _mm512_shuffle_i64x2(g,h,0x44);  // -+,--
  d = _mm512_shuffle_i64x2(g,h,0xEE);  // -+,--

  e = _mm512_shuffle_i64x2(a,c,0x88);  // ++,+-,-+,--
  f = _mm512_shuffle_i64x2(a,c,0xDD);  // ++,+-,-+,--
  g = _mm512_shuffle_i64x2(b,d,0x88);  // ++,+-,-+,--
  h = _mm512_shuffle_i64x2(b,d,0xDD);  // ++,+-,-+,--

  a = _mm512_cvtepi16_epi32(_mm256_load_si256((__m256i*)&in[16]));
  b = _mm512_cvtepi16_epi32(_mm256_load_si256((__m256i*)&in[ 0]));

  c = _mm512_add_epi32(a,b);  // ++
  d = _mm512_sub_epi32(a,b);  // +-

  k = _mm512_shuffle_i64x2(c,c,0x00);  // ++,++,++,++
  l = _mm512_shuffle_i64x2(c,c,0x55);  // ++,++,++,++
  m = _mm512_shuffle_i64x2(c,c,0xAA);  // ++,++,++,++
  n = _mm512_shuffle_i64x2(c,c,0xFF);  // ++,++,++,++
  o = _mm512_shuffle_i64x2(d,d,0x00);  // +-,+-,+-,+-
  p = _mm512_shuffle_i64x2(d,d,0x55);  // +-,+-,+-,+-
  q = _mm512_shuffle_i64x2(d,d,0xAA);  // +-,+-,+-,+-
  r = _mm512_shuffle_i64x2(d,d,0xFF);  // +-,+-,+-,+-

  s = _mm512_add_epi32(e,k);  // ++++,+-++,-+++,--++
  t = _mm512_add_epi32(e,o);  // +++-,+-+-,-++-,--+-
  u = _mm512_sub_epi32(e,o);  // ++-+,+--+,-+-+,---+
  v = _mm512_sub_epi32(e,k);  // ++--,+---,-+--,----
  w = _mm512_add_epi32(f,l);  // ++++,+-++,-+++,--++
  x = _mm512_add_epi32(f,p);  // +++-,+-+-,-++-,--+-
  y = _mm512_sub_epi32(f,p);  // ++-+,+--+,-+-+,---+
  z = _mm512_sub_epi32(f,l);  // ++--,+---,-+--,----
  a = _mm512_add_epi32(g,m);  // ++++,+-++,-+++,--++
  b = _mm512_add_epi32(g,q);  // +++-,+-+-,-++-,--+-
  c = _mm512_sub_epi32(g,q);  // ++-+,+--+,-+-+,---+
  d = _mm512_sub_epi32(g,m);  // ++--,+---,-+--,----
  e = _mm512_add_epi32(h,n);  // ++++,+-++,-+++,--++
  f = _mm512_add_epi32(h,r);  // +++-,+-+-,-++-,--+-
  g = _mm512_sub_epi32(h,r);  // ++-+,+--+,-+-+,---+
  h = _mm512_sub_epi32(h,n);  // ++--,+---,-+--,----

  k = _mm512_unpacklo_epi32(s,t);  // ++++,+++-,+-++,+-+-,-+++,-++-,--++,--+-
  l = _mm512_unpacklo_epi32(u,v);  // ++-+,++--,+--+,+---,-+-+,-+--,---+,----
  m = _mm512_unpackhi_epi32(s,t);  // ++++,+++-,+-++,+-+-,-+++,-++-,--++,--+-
  n = _mm512_unpackhi_epi32(u,v);  // ++-+,++--,+--+,+---,-+-+,-+--,---+,----
  o = _mm512_unpacklo_epi32(w,x);  // ++++,+++-,+-++,+-+-,-+++,-++-,--++,--+-
  p = _mm512_unpacklo_epi32(y,z);  // ++-+,++--,+--+,+---,-+-+,-+--,---+,----
  q = _mm512_unpackhi_epi32(w,x);  // ++++,+++-,+-++,+-+-,-+++,-++-,--++,--+-
  r = _mm512_unpackhi_epi32(y,z);  // ++-+,++--,+--+,+---,-+-+,-+--,---+,----
  s = _mm512_unpacklo_epi32(a,b);  // ++++,+++-,+-++,+-+-,-+++,-++-,--++,--+-
  t = _mm512_unpacklo_epi32(c,d);  // ++-+,++--,+--+,+---,-+-+,-+--,---+,----
  u = _mm512_unpackhi_epi32(a,b);  // ++++,+++-,+-++,+-+-,-+++,-++-,--++,--+-
  v = _mm512_unpackhi_epi32(c,d);  // ++-+,++--,+--+,+---,-+-+,-+--,---+,----
  w = _mm512_unpacklo_epi32(e,f);  // ++++,+++-,+-++,+-+-,-+++,-++-,--++,--+-
  x = _mm512_unpacklo_epi32(g,h);  // ++-+,++--,+--+,+---,-+-+,-+--,---+,----
  y = _mm512_unpackhi_epi32(e,f);  // ++++,+++-,+-++,+-+-,-+++,-++-,--++,--+-
  z = _mm512_unpackhi_epi32(g,h);  // ++-+,++--,+--+,+---,-+-+,-+--,---+,----

  a = _mm512_unpacklo_epi64(k,l);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  b = _mm512_unpackhi_epi64(k,l);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  c = _mm512_unpacklo_epi64(m,n);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  d = _mm512_unpackhi_epi64(m,n);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  e = _mm512_unpacklo_epi64(o,p);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  f = _mm512_unpackhi_epi64(o,p);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  g = _mm512_unpacklo_epi64(q,r);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  h = _mm512_unpackhi_epi64(q,r);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  k = _mm512_unpacklo_epi64(s,t);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  l = _mm512_unpackhi_epi64(s,t);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  m = _mm512_unpacklo_epi64(u,v);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  n = _mm512_unpackhi_epi64(u,v);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  o = _mm512_unpacklo_epi64(w,x);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  p = _mm512_unpackhi_epi64(w,x);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  q = _mm512_unpacklo_epi64(y,z);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----
  r = _mm512_unpackhi_epi64(y,z);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---,-+++,-++-,-+-+,-+--,--++,--+-,---+,----

  for(i=0;i<256/32;i++) {
    s = _mm512_load_si512((__m512i*)&mat[32*16/8*i+256*16/8*0]);
    t = _mm512_load_si512((__m512i*)&mat[32*16/8*i+256*16/8*1]);
    u = _mm512_load_si512((__m512i*)&mat[32*16/8*i+256*16/8*2]);
    v = _mm512_load_si512((__m512i*)&mat[32*16/8*i+256*16/8*3]);

    x = _mm512_slli_epi32(t,1);
    y = _mm512_slli_epi32(u,2);
    z = _mm512_slli_epi32(v,3);
    w = _mm512_and_si512(s,stencil1);
    x = _mm512_and_si512(x,stencil2);
    y = _mm512_and_si512(y,stencil4);
    z = _mm512_and_si512(z,stencil8);
    w = _mm512_add_epi8(w,x);
    x = _mm512_add_epi8(y,z);
    w = _mm512_add_epi8(w,x);

    x = _mm512_srli_epi32(s,1);
    x = _mm512_and_si512(x,stencil1);
    y = _mm512_and_si512(t,stencil2);
    x = _mm512_add_epi8(x,y);
    y = _mm512_slli_epi32(u,1);
    z = _mm512_slli_epi32(v,2);
    y = _mm512_and_si512(y,stencil4);
    z = _mm512_and_si512(z,stencil8);
    y = _mm512_add_epi8(y,z);
    x = _mm512_add_epi8(x,y);

    y = _mm512_permutex2var_epi8(w,vpermi2bidx,x);
    z = _mm512_add_epi8(vpermi2bidx,_mm512_set1_epi8(32));
    z = _mm512_permutex2var_epi8(w,z,x);

#define BLOCK(T0,T1)                  \
  x = _mm512_permutexvar_epi32(y,T0); \
  w = _mm512_add_epi32(w,x);          \
  x = _mm512_permutexvar_epi32(z,T1); \
  y = _mm512_srli_epi32(y,4);         \
  z = _mm512_srli_epi32(z,4);         \
  w = _mm512_add_epi32(w,x)

    w = _mm512_loadu_si512((__m512i*)&out[32*i+ 0]);
    BLOCK(a,e);
    BLOCK(k,o);
    BLOCK(b,f);
    BLOCK(l,p);
    BLOCK(c,g);
    BLOCK(m,q);
    BLOCK(d,h);
    BLOCK(n,r);
    _mm512_storeu_si512((__m512i*)&out[32*i+ 0],w);

    w = _mm512_srli_epi32(s,2);
    x = _mm512_srli_epi32(t,1);
    z = _mm512_slli_epi32(v,1);
    w = _mm512_and_si512(w,stencil1);
    x = _mm512_and_si512(x,stencil2);
    y = _mm512_and_si512(u,stencil4);
    z = _mm512_and_si512(z,stencil8);
    w = _mm512_add_epi8(w,x);
    x = _mm512_add_epi8(y,z);
    w = _mm512_add_epi8(w,x);

    x = _mm512_srli_epi32(s,3);
    y = _mm512_srli_epi32(t,2);
    x = _mm512_and_si512(x,stencil1);
    y = _mm512_and_si512(y,stencil2);
    x = _mm512_add_epi8(x,y);
    y = _mm512_srli_epi32(u,1);
    y = _mm512_and_si512(y,stencil4);
    z = _mm512_and_si512(v,stencil8);
    y = _mm512_add_epi8(y,z);
    x = _mm512_add_epi8(x,y);

    y = _mm512_permutex2var_epi8(w,vpermi2bidx,x);
    z = _mm512_add_epi8(vpermi2bidx,_mm512_set1_epi8(32));
    z = _mm512_permutex2var_epi8(w,z,x);

    w = _mm512_loadu_si512((__m512i*)&out[32*i+16]);
    BLOCK(a,e);
    BLOCK(k,o);
    BLOCK(b,f);
    BLOCK(l,p);
    BLOCK(c,g);
    BLOCK(m,q);
    BLOCK(d,h);
    BLOCK(n,r);
    _mm512_storeu_si512((__m512i*)&out[32*i+16],w);

#undef BLOCK
  }
}
#endif

void poly_jlproj_add(int32_t out[256], const poly in, const uint8_t mat[256*N/8]) {
  size_t i;

  for(i=0;i<N/64;i++)
    jlproj_add64(out,&in->c[64*i],&mat[2048*i]);
}

void polyvec_jlproj_add(int32_t r[256], const poly *p, size_t len, const uint8_t *mat) {
  size_t i;

  for(i=0;i<len;i++)
    poly_jlproj_add(r,p[i],&mat[256*N/8*i]);
}

static void jlproj_collapsmat32(int64_t *row, const uint8_t *mat, size_t len, const int64_t alpha[32]) {
  size_t i;
  __m512i a,b,c,d;
  __m512i e,f,g,h;
  __m512i k,l,m,n;
  __m512i o,p,q,r;
  __m512i s,t,u,v;
  __m512i w,x,y,z;
  const __m512i zero = _mm512_setzero_si512();

  /* Prepare 8 pairs of vectors containing all 16 signed combinations of 4 challenge coeffs that are 8 apart */
  a = _mm512_load_si512((__m512i*)&alpha[24]);
  b = _mm512_load_si512((__m512i*)&alpha[16]);

  e = _mm512_add_epi64(a,b);     // ++
  f = _mm512_sub_epi64(a,b);     // +-
  g = _mm512_sub_epi64(b,a);     // -+
  h = _mm512_sub_epi64(zero,e);  // --

  s = _mm512_shuffle_i64x2(e,f,0x00);  // ++,++,+-,+-
  t = _mm512_shuffle_i64x2(e,f,0x55);  // ++,++,+-,+-
  u = _mm512_shuffle_i64x2(e,f,0xAA);  // ++,++,+-,+-
  v = _mm512_shuffle_i64x2(e,f,0xFF);  // ++,++,+-,+-
  w = _mm512_shuffle_i64x2(g,h,0x00);  // -+,-+,--,--
  x = _mm512_shuffle_i64x2(g,h,0x55);  // -+,-+,--,--
  y = _mm512_shuffle_i64x2(g,h,0xAA);  // -+,-+,--,--
  z = _mm512_shuffle_i64x2(g,h,0xFF);  // -+,-+,--,--

  a = _mm512_load_si512((__m512i*)&alpha[ 8]);
  b = _mm512_load_si512((__m512i*)&alpha[ 0]);

  c = _mm512_add_epi64(a,b);  // ++
  d = _mm512_sub_epi64(b,a);  // -+

  k = _mm512_shuffle_i64x2(c,d,0x44);  // ++,-+
  l = _mm512_shuffle_i64x2(c,d,0xEE);  // ++,-+
  a = _mm512_shuffle_i64x2(k,k,0x88);  // ++,-+,++,-+
  b = _mm512_shuffle_i64x2(k,k,0xDD);  // ++,-+,++,-+
  c = _mm512_shuffle_i64x2(l,l,0x88);  // ++,-+,++,-+
  d = _mm512_shuffle_i64x2(l,l,0xDD);  // ++,-+,++,-+
  e = _mm512_shuffle_i64x2(k,k,0x22);  // -+,++,-+,++
  f = _mm512_shuffle_i64x2(k,k,0x77);  // -+,++,-+,++
  g = _mm512_shuffle_i64x2(l,l,0x22);  // -+,++,-+,++
  h = _mm512_shuffle_i64x2(l,l,0x77);  // -+,++,-+,++

  k = _mm512_add_epi64(s,a);  // ++++,++-+,+-++,+--+
  l = _mm512_add_epi64(t,b);  // ++++,++-+,+-++,+--+
  m = _mm512_add_epi64(u,c);  // ++++,++-+,+-++,+--+
  n = _mm512_add_epi64(v,d);  // ++++,++-+,+-++,+--+
  o = _mm512_sub_epi64(s,e);  // +++-,++--,+-+-,+---
  p = _mm512_sub_epi64(t,f);  // +++-,++--,+-+-,+---
  q = _mm512_sub_epi64(u,g);  // +++-,++--,+-+-,+---
  r = _mm512_sub_epi64(v,h);  // +++-,++--,+-+-,+---
  s = _mm512_add_epi64(w,a);  // -+++,-+-+,--++,+--+
  t = _mm512_add_epi64(x,b);  // -+++,-+-+,--++,+--+
  u = _mm512_add_epi64(y,c);  // -+++,-+-+,--++,+--+
  v = _mm512_add_epi64(z,d);  // -+++,-+-+,--++,+--+
  w = _mm512_sub_epi64(w,e);  // -++-,-+--,--+-,----
  x = _mm512_sub_epi64(x,f);  // -++-,-+--,--+-,----
  y = _mm512_sub_epi64(y,g);  // -++-,-+--,--+-,----
  z = _mm512_sub_epi64(z,h);  // -++-,-+--,--+-,----

  a = _mm512_unpacklo_epi64(k,o);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  b = _mm512_unpackhi_epi64(k,o);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  c = _mm512_unpacklo_epi64(l,p);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  d = _mm512_unpackhi_epi64(l,p);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  e = _mm512_unpacklo_epi64(m,q);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  f = _mm512_unpackhi_epi64(m,q);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  g = _mm512_unpacklo_epi64(n,r);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  h = _mm512_unpackhi_epi64(n,r);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  k = _mm512_unpacklo_epi64(s,w);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  l = _mm512_unpackhi_epi64(s,w);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  m = _mm512_unpacklo_epi64(t,x);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  n = _mm512_unpackhi_epi64(t,x);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  o = _mm512_unpacklo_epi64(u,y);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  p = _mm512_unpackhi_epi64(u,y);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  q = _mm512_unpacklo_epi64(v,z);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  r = _mm512_unpackhi_epi64(v,z);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----

#if 0
  const __m125i revidx = _mm512_set_epi64(0,1,2,3,4,5,6,7);

  a = _mm512_loadu_si256((__m512i*)&alpha[ 0]));
  b = _mm512_loadu_si256((__m512i*)&alpha[ 8]));
  c = _mm512_loadu_si256((__m512i*)&alpha[16]));
  d = _mm512_loadu_si256((__m512i*)&alpha[24]));

  e = _mm512_add_epi64(a,b);  // ++ (a+b)
  f = _mm512_sub_epi64(a,b);  // +- (a-b)
  g = _mm512_add_epi64(c,d);  // ++ (c+d)
  h = _mm512_sub_epi64(c,d);  // +- (c-d)

  k = _mm512_add_epi64(e,g);  // ++++
  l = _mm512_add_epi64(e,h);  // +++-
  m = _mm512_sub_epi64(e,h);  // ++-+
  n = _mm512_sub_epi64(e,g);  // ++--
  o = _mm512_add_epi64(f,g);  // +-++
  p = _mm512_add_epi64(f,h);  // +-+-
  q = _mm512_sub_epi64(f,h);  // +--+
  r = _mm512_sub_epi64(f,g);  // +---

  a = _mm512_shuffle_i64x2(k,m,0x44);  // ++++,++-+
  b = _mm512_shuffle_i64x2(k,m,0xEE);  // ++++,++-+
  c = _mm512_shuffle_i64x2(l,n,0x44);  // +++-,++--
  d = _mm512_shuffle_i64x2(l,n,0xEE);  // +++-,++--
  e = _mm512_shuffle_i64x2(o,q,0x44);  // +-++,+--+
  f = _mm512_shuffle_i64x2(o,q,0xEE);  // +-++,+--+
  g = _mm512_shuffle_i64x2(p,r,0x44);  // +-+-,+---
  h = _mm512_shuffle_i64x2(p,r,0xEE);  // +-+-,+---

  k = _mm512_shuffle_i64x2(a,e,0x88);  // ++++,++-+,+-++,+--+
  l = _mm512_shuffle_i64x2(a,e,0xDD);  // ++++,++-+,+-++,+--+
  m = _mm512_shuffle_i64x2(b,f,0x88);  // ++++,++-+,+-++,+--+
  n = _mm512_shuffle_i64x2(b,f,0xDD);  // ++++,++-+,+-++,+--+
  o = _mm512_shuffle_i64x2(c,g,0x88);  // +++-,++--,+-+-,+---
  p = _mm512_shuffle_i64x2(c,g,0xDD);  // +++-,++--,+-+-,+---
  q = _mm512_shuffle_i64x2(d,h,0x88);  // +++-,++--,+-+-,+---
  r = _mm512_shuffle_i64x2(d,h,0xDD);  // +++-,++--,+-+-,+---

  a = _mm512_unpacklo_epi64(k,o);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  b = _mm512_unpackhi_epi64(k,o);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  c = _mm512_unpacklo_epi64(l,p);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  d = _mm512_unpackhi_epi64(l,p);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  e = _mm512_unpacklo_epi64(m,q);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  f = _mm512_unpackhi_epi64(m,q);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  g = _mm512_unpacklo_epi64(n,r);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---
  h = _mm512_unpackhi_epi64(n,r);  // ++++,+++-,++-+,++--,+-++,+-+-,+--+,+---

  k = _mm512_sub_epi64(zero,a);  // ----,---+,--+-,--++,-+--,-+-+,-++-,-+++
  l = _mm512_sub_epi64(zero,b);  // ----,---+,--+-,--++,-+--,-+-+,-++-,-+++
  m = _mm512_sub_epi64(zero,c);  // ----,---+,--+-,--++,-+--,-+-+,-++-,-+++
  n = _mm512_sub_epi64(zero,d);  // ----,---+,--+-,--++,-+--,-+-+,-++-,-+++
  o = _mm512_sub_epi64(zero,e);  // ----,---+,--+-,--++,-+--,-+-+,-++-,-+++
  p = _mm512_sub_epi64(zero,f);  // ----,---+,--+-,--++,-+--,-+-+,-++-,-+++
  q = _mm512_sub_epi64(zero,g);  // ----,---+,--+-,--++,-+--,-+-+,-++-,-+++
  r = _mm512_sub_epi64(zero,h);  // ----,---+,--+-,--++,-+--,-+-+,-++-,-+++

  k = _mm512_permutexvar_epi64(revidx,k);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  l = _mm512_permutexvar_epi64(revidx,l);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  m = _mm512_permutexvar_epi64(revidx,m);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  n = _mm512_permutexvar_epi64(revidx,n);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  o = _mm512_permutexvar_epi64(revidx,o);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  p = _mm512_permutexvar_epi64(revidx,p);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  q = _mm512_permutexvar_epi64(revidx,q);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
  r = _mm512_permutexvar_epi64(revidx,r);  // -+++,-++-,-+-+,-+--,--++,--+-,---+,----
#endif

#if !defined(JL_FORCE_AVX512) && !(defined(__x86_64__) && !defined(LAZER_PORTABLE))
  /* Non-x86: SIMDe emulates _mm512_permutex2var_epi64 lane by lane, which made
   * this loop ~50% of all prover/verifier cycles. Each permutex2var is just a
   * 16-entry table lookup indexed by a 4-bit nibble of the (public) JL matrix,
   * so do exactly that in scalar code. Wrapping uint64 adds give results
   * identical to _mm512_add_epi64. */
  (void)u; (void)v;
  {
    alignas(64) uint64_t tab[8][16];
    size_t j;
    _mm512_store_si512((__m512i*)&tab[0][0],a); _mm512_store_si512((__m512i*)&tab[0][8],k);
    _mm512_store_si512((__m512i*)&tab[1][0],b); _mm512_store_si512((__m512i*)&tab[1][8],l);
    _mm512_store_si512((__m512i*)&tab[2][0],c); _mm512_store_si512((__m512i*)&tab[2][8],m);
    _mm512_store_si512((__m512i*)&tab[3][0],d); _mm512_store_si512((__m512i*)&tab[3][8],n);
    _mm512_store_si512((__m512i*)&tab[4][0],e); _mm512_store_si512((__m512i*)&tab[4][8],o);
    _mm512_store_si512((__m512i*)&tab[5][0],f); _mm512_store_si512((__m512i*)&tab[5][8],p);
    _mm512_store_si512((__m512i*)&tab[6][0],g); _mm512_store_si512((__m512i*)&tab[6][8],q);
    _mm512_store_si512((__m512i*)&tab[7][0],h); _mm512_store_si512((__m512i*)&tab[7][8],r);

    for(i=0;i<len/32;i++) {
      const uint8_t *mi = &mat[256*32/8*i];
      uint64_t *rw = (uint64_t*)&row[32*i];
      for(j=0;j<8;j++) {
        uint64_t sv, tv, w0, x0, y0, z0;
        memcpy(&sv, mi + 8*j, 8);
        memcpy(&tv, mi + 256*16/8 + 8*j, 8);
        w0 = rw[j]; x0 = rw[8+j]; y0 = rw[16+j]; z0 = rw[24+j];
#define NBLOCK(G)                                   \
        w0 += tab[G][sv & 15]; y0 += tab[G][tv & 15]; \
        sv >>= 4; tv >>= 4;                          \
        x0 += tab[G][sv & 15]; z0 += tab[G][tv & 15]; \
        sv >>= 4; tv >>= 4
        NBLOCK(0); NBLOCK(1); NBLOCK(2); NBLOCK(3);
        NBLOCK(4); NBLOCK(5); NBLOCK(6); NBLOCK(7);
#undef NBLOCK
        rw[j] = w0; rw[8+j] = x0; rw[16+j] = y0; rw[24+j] = z0;
      }
    }
  }
  return;
#endif

#define BLOCK(T0,T1)                      \
  u = _mm512_permutex2var_epi64(T0,s,T1); \
  v = _mm512_permutex2var_epi64(T0,t,T1); \
  s = _mm512_srli_epi64(s,4);             \
  t = _mm512_srli_epi64(t,4);             \
  w = _mm512_add_epi64(w,u);              \
  y = _mm512_add_epi64(y,v);              \
  u = _mm512_permutex2var_epi64(T0,s,T1); \
  v = _mm512_permutex2var_epi64(T0,t,T1); \
  s = _mm512_srli_epi64(s,4);             \
  t = _mm512_srli_epi64(t,4);             \
  x = _mm512_add_epi64(x,u);              \
  z = _mm512_add_epi64(z,v)

  for(i=0;i<len/32;i++) {
    s = _mm512_load_si512((__m512i*)&mat[256*32/8*i+       0]);
    t = _mm512_load_si512((__m512i*)&mat[256*32/8*i+256*16/8]);
    w = _mm512_load_si512((__m512i*)&row[32*i+ 0]);
    x = _mm512_load_si512((__m512i*)&row[32*i+ 8]);
    y = _mm512_load_si512((__m512i*)&row[32*i+16]);
    z = _mm512_load_si512((__m512i*)&row[32*i+24]);
    BLOCK(a,k);
    BLOCK(b,l);
    BLOCK(c,m);
    BLOCK(d,n);
    BLOCK(e,o);
    BLOCK(f,p);
    BLOCK(g,q);
    BLOCK(h,r);
    _mm512_store_si512((__m512i*)&row[32*i+ 0],w);
    _mm512_store_si512((__m512i*)&row[32*i+ 8],x);
    _mm512_store_si512((__m512i*)&row[32*i+16],y);
    _mm512_store_si512((__m512i*)&row[32*i+24],z);
  }

#undef BLOCK
}

void jlproj_expand_challenge(int64_t alpha[256], const uint8_t buf[256*QBYTES]) {
  int i;
  __m512i f;

#if QBYTES == 3
  const __mmask64 mask = _cvtu64_mask64(0x0B0B0B0B0B0B0B0Bull);
  const __m512i vpermbidx = _mm512_set_epi8(-1,-1,-1,-1,-1,23,22,21,
                                            -1,-1,-1,-1,-1,20,19,18,
                                            -1,-1,-1,-1,-1,17,16,15,
                                            -1,-1,-1,-1,-1,14,13,12,
                                            -1,-1,-1,-1,-1,11,10, 9,
                                            -1,-1,-1,-1,-1, 8, 7, 6,
                                            -1,-1,-1,-1,-1, 5, 4, 3,
                                            -1,-1,-1,-1,-1, 2, 1, 0);
  for(i=0;i<256/8;i++) {
    f = _mm512_loadu_si512((__m512i*)&buf[24*i]);
    f = _mm512_maskz_permutexvar_epi8(mask,vpermbidx,f);
    _mm512_store_si512((__m512i*)&alpha[8*i],f);
  }
#elif QBYTES == 4
  for(i=0;i<256/8;i++) {
    f = _mm512_cvtepu32_epi64(_mm256_load_si256((__m256i*)&buf[32*i]));
    _mm512_store_si512((__m512i*)&alpha[8*i],f);
  }
#elif QBYTES == 5
  const __mmask64 mask = _cvtu64_mask64(0x1F1F1F1F1F1F1F1Full);
  const __m512i vpermbidx = _mm512_set_epi8(-1,-1,-1,39,38,37,36,35,
                                            -1,-1,-1,34,33,32,31,30,
                                            -1,-1,-1,29,28,27,26,25,
                                            -1,-1,-1,24,23,22,21,20,
                                            -1,-1,-1,19,18,17,16,15,
                                            -1,-1,-1,14,13,12,11,10,
                                            -1,-1,-1, 9, 8, 7, 6, 5,
                                            -1,-1,-1, 4, 3, 2, 1, 0);
  for(i=0;i<256/8;i++) {
    f = _mm512_loadu_si512((__m512i*)&buf[40*i]);
    f = _mm512_maskz_permutexvar_epi8(mask,vpermbidx,f);
    _mm512_store_si512((__m512i*)&alpha[8*i],f);
  }
#elif QBYTES == 6
  const __mmask64 mask = _cvtu64_mask64(0x3F3F3F3F3F3F3F3Full);
  const __m512i vpermbidx = _mm512_set_epi8(-1,-1,47,46,45,44,43,42,
                                            -1,-1,41,40,39,38,37,36,
                                            -1,-1,35,34,33,32,31,30,
                                            -1,-1,29,28,27,26,25,24,
                                            -1,-1,23,22,21,20,19,18,
                                            -1,-1,17,16,15,14,13,12,
                                            -1,-1,11,10, 9, 8, 7, 6,
                                            -1,-1, 5, 4, 3, 2, 1, 0);
  for(i=0;i<256/8;i++) {
    f = _mm512_loadu_si512((__m512i*)&buf[48*i]);
    f = _mm512_maskz_permutexvar_epi8(mask,vpermbidx,f);
    _mm512_store_si512((__m512i*)&alpha[8*i],f);
  }
#else
#error
#endif
}

#ifdef LABRADOS_NEON
/* NEON: decomposition of the 64-bit row sums into L=3 limbs, identical to the
 * AVX-512 code (srai_epi64, mul_epi32 with the low 32 bits of x>>LOGQ, mask
 * compress-stores of the low 16 bits of each lane):
 *   f = (x & (2^LOGQ-1)) + (int64)(int32)(x>>LOGQ)*QOFF,
 *   limb0 = f & 0x3FFF, limb1 = (f>>14) & 0x3FFF, limb2 = (int16)(f>>28).
 * |f| < 2^38 + 2^31*2^8 < 2^40 for every int64 x, so f>>14 fits in 32 bits and the
 * rest of the computation is done exactly in 32-bit lanes. */
static void jl_decompose(polz *t, const int64_t *raw, size_t cnt) {
  size_t i,j;
  const int64x2_t mask = vdupq_n_s64(((int64_t)1 << LOGQ) - 1);
  const int32x4_t qoff = vdupq_n_s32(QOFF);
  const int16x8_t mask14 = vdupq_n_s16(0x3FFF);
  _Static_assert(L == 3, "jl_decompose assumes 3 limbs");

  for(i=0;i<cnt;i++) {
    for(j=0;j<N/8;j++) {
      const int64_t *x = &raw[N*i+8*j];
      int64x2_t x0 = vld1q_s64(x+0), x1 = vld1q_s64(x+2), x2 = vld1q_s64(x+4), x3 = vld1q_s64(x+6);
      int32x4_t g01 = vcombine_s32(vmovn_s64(vshrq_n_s64(x0,LOGQ)),vmovn_s64(vshrq_n_s64(x1,LOGQ)));
      int32x4_t g23 = vcombine_s32(vmovn_s64(vshrq_n_s64(x2,LOGQ)),vmovn_s64(vshrq_n_s64(x3,LOGQ)));
      int64x2_t f0 = vmlal_s32(vandq_s64(x0,mask),vget_low_s32(g01),vget_low_s32(qoff));
      int64x2_t f1 = vmlal_high_s32(vandq_s64(x1,mask),g01,qoff);
      int64x2_t f2 = vmlal_s32(vandq_s64(x2,mask),vget_low_s32(g23),vget_low_s32(qoff));
      int64x2_t f3 = vmlal_high_s32(vandq_s64(x3,mask),g23,qoff);
      int32x4_t l01 = vuzp1q_s32(vreinterpretq_s32_s64(f0),vreinterpretq_s32_s64(f1));
      int32x4_t l23 = vuzp1q_s32(vreinterpretq_s32_s64(f2),vreinterpretq_s32_s64(f3));
      int32x4_t h01 = vcombine_s32(vshrn_n_s64(f0,14),vshrn_n_s64(f1,14));
      int32x4_t h23 = vcombine_s32(vshrn_n_s64(f2,14),vshrn_n_s64(f3,14));
      int16x8_t limb0 = vandq_s16(vuzp1q_s16(vreinterpretq_s16_s32(l01),vreinterpretq_s16_s32(l23)),mask14);
      int16x8_t limb1 = vandq_s16(vuzp1q_s16(vreinterpretq_s16_s32(h01),vreinterpretq_s16_s32(h23)),mask14);
      int16x8_t limb2 = vuzp1q_s16(vreinterpretq_s16_s32(vshrq_n_s32(h01,14)),vreinterpretq_s16_s32(vshrq_n_s32(h23,14)));
      vst1q_s16(&t[i]->limbs[0]->c[8*j],limb0);
      vst1q_s16(&t[i]->limbs[1]->c[8*j],limb1);
      vst1q_s16(&t[i]->limbs[2]->c[8*j],limb2);
    }
  }
}

void polxvec_jlproj_collapsmat(polxvec r, const uint8_t *mat, const int64_t alpha[256]) {
  size_t i;
  alignas(64) int64_t raw[16*N];
  polz t[16];
  size_t rpos = 0;
  size_t len = r->len;

  while(len >= 16) {
    memset(raw,0,sizeof(int64_t)*16*N);
    for(i=0;i<256/32;i++)
      jlproj_collapsmat32(raw,&mat[32*16/8*i],16*N,&alpha[32*i]);
    jl_decompose(t,raw,16);
    polzvec_sigmam1(t,t,16);
    polzvec_topolxvec(r,t,rpos,1,16);
    rpos += 16;
    mat += 256*16*N/8;
    len -= 16;
  }

  if(len) {
    memset(raw,0,sizeof(int64_t)*len*N);
    for(i=0;i<256/32;i++)
      jlproj_collapsmat32(raw,&mat[32*16/8*i],len*N,&alpha[32*i]);
    jl_decompose(t,raw,len);
    polzvec_sigmam1(t,t,len);
    polzvec_topolxvec(r,t,rpos,1,len);
  }
}
#else
void polxvec_jlproj_collapsmat(polxvec r, const uint8_t *mat, const int64_t alpha[256]) {
  size_t i,j,k;
  __m512i f,g;
  const __m512i qoff = _mm512_set1_epi64(QOFF);
  const __m512i mask = _mm512_set1_epi64(((uint64_t)1 << LOGQ) - 1);
  const __m512i mask14 = _mm512_set1_epi64((1 << 14) - 1);
  const __mmask32 oneinfour = _cvtu32_mask32(0x11111111);
  alignas(64) int64_t raw[16*N];
  polz t[16];
  size_t rpos = 0;
  size_t len = r->len;

  while(len >= 16) {
    for(i=0;i<16*N;i++)
      raw[i] = 0;
    for(i=0;i<256/32;i++)
      jlproj_collapsmat32(raw,&mat[32*16/8*i],16*N,&alpha[32*i]);

    for(i=0;i<16;i++) {
      for(j=0;j<N/8;j++) {
        f = _mm512_load_si512((__m512i*)&raw[N*i+8*j]);
        g = _mm512_srai_epi64(f,LOGQ);
        f = _mm512_and_si512(f,mask);
        g = _mm512_mul_epi32(g,qoff);
        f = _mm512_add_epi64(f,g);
        for(k=0;k<L-1;k++) {
          g = _mm512_and_si512(f,mask14);
          f = _mm512_srai_epi64(f,14);
          _mm512_mask_compressstoreu_epi16(&t[i]->limbs[k]->c[8*j],oneinfour,g);
        }
        _mm512_mask_compressstoreu_epi16(&t[i]->limbs[L-1]->c[8*j],oneinfour,f);
      }
    }
    polzvec_sigmam1(t,t,16);
    polzvec_topolxvec(r,t,rpos,1,16);
    rpos += 16;
    mat += 256*16*N/8;
    len -= 16;
  }

  if(len) {
    for(i=0;i<len*N;i++)
      raw[i] = 0;
    for(i=0;i<256/32;i++)
      jlproj_collapsmat32(raw,&mat[32*16/8*i],len*N,&alpha[32*i]);

    for(i=0;i<len;i++) {
      for(j=0;j<N/8;j++) {
        f = _mm512_load_si512((__m512i*)&raw[N*i+8*j]);
        g = _mm512_srai_epi64(f,LOGQ);
        f = _mm512_and_si512(f,mask);
        g = _mm512_mul_epi32(g,qoff);
        f = _mm512_add_epi64(f,g);
        for(k=0;k<L-1;k++) {
          g = _mm512_and_si512(f,mask14);
          f = _mm512_srai_epi64(f,14);
          _mm512_mask_compressstoreu_epi16(&t[i]->limbs[k]->c[8*j],oneinfour,g);
        }
        _mm512_mask_compressstoreu_epi16(&t[i]->limbs[L-1]->c[8*j],oneinfour,f);
      }
    }
    polzvec_sigmam1(t,t,len);
    polzvec_topolxvec(r,t,rpos,1,len);
  }
}
#endif

static inline int64_t _mm512_hsum_epi64(__m512i a) {
  __m256i t;
  __m128i u;
  t = _mm256_add_epi64(_mm512_castsi512_si256(a),_mm512_extracti64x4_epi64(a,1));
  u = _mm_add_epi64(_mm256_castsi256_si128(t),_mm256_extracti64x2_epi64(t,1));
  u = _mm_add_epi64(u,_mm_unpackhi_epi64(u,u));
  return _mm_cvtsi128_si64(u);
}

int64_t jlproj_collapsproj(const int32_t p[256], const int64_t alpha[256]) {
  int i;
  int64_t r,t;
  __m512i f,g,h,s;
  const __m512i mask = _mm512_set1_epi64(((uint64_t)1 << LOGQ) - 1);
  const __m512i qoff = _mm512_set1_epi64(QOFF);

  s = _mm512_setzero_si512();
  for(i=0;i<256/8;i++) {
    f = _mm512_cvtepi32_epi64(_mm256_loadu_si256((__m256i*)&p[8*i]));
    g = _mm512_load_si512((__m512i*)&alpha[8*i]);
    h = _mm512_mul_epi32(f,g);
    s = _mm512_add_epi64(s,h);
#if LOGQ >= 32
    h = _mm512_srli_epi32(g,31);
    g = _mm512_srli_epi64(g,32);
    g = _mm512_add_epi64(g,h);
    f = _mm512_mul_epi32(f,g);
    g = _mm512_srai_epi64(f,LOGQ-32);  // 2^31
    f = _mm512_slli_epi64(f,32);
    f = _mm512_and_si512(f,mask);
    s = _mm512_add_epi64(s,f);
    g = _mm512_mul_epi32(g,qoff);
    s = _mm512_add_epi64(s,g);
#endif
    f = _mm512_srai_epi64(s,LOGQ);
    s = _mm512_and_si512(s,mask);
    f = _mm512_mul_epi32(f,qoff);
    s = _mm512_add_epi64(s,f);
  }

  r = _mm512_hsum_epi64(s);
  t = r >> LOGQ;
  r &= ((int64_t)1 << LOGQ) - 1;
  r += t*QOFF;
  return r;
}

uint64_t jlproj_normsq(const int32_t p[256]) {
  int i;
  __m512i f,g,s;

  s = _mm512_setzero_si512();
  for(i=0;i<256/16;i++) {
    f = _mm512_loadu_si512(&p[16*i]);
    g = _mm512_mul_epi32(f,f);
    f = _mm512_srli_epi64(f,32);
    f = _mm512_mul_epi32(f,f);
    f = _mm512_add_epi64(f,g);
    s = _mm512_add_epi64(f,s);
  }

  return _mm512_hsum_epi64(s);
}
