#ifndef AESCTR_H
#define AESCTR_H

#include <stddef.h>
#include <stdint.h>
#include "simd.h"
#include "parallel.h"

#define AES128CTR_NAMESPACE(s) aes128ctr_avx2_##s
#define AES256CTR_NAMESPACE(s) aes256ctr_avx2_##s

#define AES128CTR_BLOCKBYTES 512
#define AES256CTR_BLOCKBYTES 512

/* without AES hardware: round keys are also cached in bitsliced form */
#if !defined(__x86_64__) || defined(LAZER_PORTABLE)
#define AESCTR_BSKEYS 1
#endif

typedef struct {
  __m128i rkeys[11];
  __m128i n;
#ifdef AESCTR_BSKEYS
  __m128i bsk[11*8];
#endif
} aes128ctr_ctx;

typedef struct {
  __m128i rkeys[15];
  __m128i n;
#ifdef AESCTR_BSKEYS
  __m128i bsk[15*8];
#endif
} aes256ctr_ctx;

#define aes128ctr_init AES128CTR_NAMESPACE(init)
void aes128ctr_init(aes128ctr_ctx *state, const uint8_t key[16], uint64_t nonce);

#define aes128ctr_select AES128CTR_NAMESPACE(select)
void aes128ctr_select(aes128ctr_ctx *state, uint64_t nonce);

#define aes128ctr_squeezeblocks AES128CTR_NAMESPACE(squeezeblocks)
void aes128ctr_squeezeblocks(uint8_t *out, size_t nblocks, aes128ctr_ctx *state);

#ifdef LAB_PARALLEL
/* asynchronous (multi-threaded) squeeze, see aesctr.c */
typedef struct {
  uint8_t *out;
  __m128i n;
  const __m128i *keys;
  int rounds;
  int async;
} aes128ctr_async;

#define aes128ctr_squeezeblocks_start AES128CTR_NAMESPACE(squeezeblocks_start)
void aes128ctr_squeezeblocks_start(aes128ctr_async *h, uint8_t *out, size_t nblocks, aes128ctr_ctx *state);

#define aes128ctr_squeezeblocks_finish AES128CTR_NAMESPACE(squeezeblocks_finish)
void aes128ctr_squeezeblocks_finish(aes128ctr_async *h);
#endif /* LAB_PARALLEL */
#define aes256ctr_init AES256CTR_NAMESPACE(init)
void aes256ctr_init(aes256ctr_ctx *state, const uint8_t key[32], uint64_t nonce);

#define aes256ctr_select AES256CTR_NAMESPACE(select)
void aes256ctr_select(aes256ctr_ctx *state, uint64_t nonce);

#define aes256ctr_squeezeblocks AES256CTR_NAMESPACE(squeezeblocks)
void aes256ctr_squeezeblocks(uint8_t *out, size_t nblocks, aes256ctr_ctx *state);

#endif
