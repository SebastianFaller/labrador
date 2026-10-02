#ifndef AESCTR_H
#define AESCTR_H

#include <stddef.h>
#include <stdint.h>
#include "simd.h"

#define AES128CTR_BLOCKBYTES 512
#define AES256CTR_BLOCKBYTES 512

#ifdef LAB_PRG_CHACHA20
/*
 * -DLAB_PRG_CHACHA20: the aes128ctr_* / aes256ctr_* API below produces a
 * ChaCha20 keystream (RFC 8439 block function, 20 rounds, see
 * ../chacha20.h) instead of AES-CTR. NOT output-compatible with the AES
 * build. Same semantics as the AES version:
 *   key:   aes256ctr: the 32-byte key;
 *          aes128ctr: key[16] || "LAB-CHACHA20-K16" (16 ASCII bytes)
 *   nonce: 64-bit nonce in ChaCha20 state words 14, 15 (little endian)
 *   ctr:   64-bit block counter in state words 12, 13, starting at 0;
 *          one squeezeblock = 512 bytes = 8 ChaCha20 blocks
 *   init/select: set the nonce and reset the counter to 0.
 * I.e. output = RFC 8439 ChaCha20 keystream with nonce 00000000||le64(nonce)
 * from counter 0, consumed sequentially.
 */
#define AES128CTR_NAMESPACE(s) aes128ctr_chacha20_##s
#define AES256CTR_NAMESPACE(s) aes256ctr_chacha20_##s

typedef struct {
  uint32_t s[16];
} aes128ctr_ctx;

typedef struct {
  uint32_t s[16];
} aes256ctr_ctx;

/* the multi-threaded squeeze API degenerates to a synchronous squeeze */
typedef struct {
  int async;
} aes128ctr_async;

#else /* AES-CTR */

#define AES128CTR_NAMESPACE(s) aes128ctr_avx2_##s
#define AES256CTR_NAMESPACE(s) aes256ctr_avx2_##s

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

/* asynchronous (multi-threaded) squeeze, see aesctr.c */
typedef struct {
  uint8_t *out;
  __m128i n;
  const __m128i *keys;
  int rounds;
  int async;
} aes128ctr_async;

#endif /* LAB_PRG_CHACHA20 */

#define aes128ctr_init AES128CTR_NAMESPACE(init)
void aes128ctr_init(aes128ctr_ctx *state, const uint8_t key[16], uint64_t nonce);

#define aes128ctr_select AES128CTR_NAMESPACE(select)
void aes128ctr_select(aes128ctr_ctx *state, uint64_t nonce);

#define aes128ctr_squeezeblocks AES128CTR_NAMESPACE(squeezeblocks)
void aes128ctr_squeezeblocks(uint8_t *out, size_t nblocks, aes128ctr_ctx *state);


#define aes128ctr_squeezeblocks_start AES128CTR_NAMESPACE(squeezeblocks_start)
void aes128ctr_squeezeblocks_start(aes128ctr_async *h, uint8_t *out, size_t nblocks, aes128ctr_ctx *state);

#define aes128ctr_squeezeblocks_finish AES128CTR_NAMESPACE(squeezeblocks_finish)
void aes128ctr_squeezeblocks_finish(aes128ctr_async *h);

#define aes256ctr_init AES256CTR_NAMESPACE(init)
void aes256ctr_init(aes256ctr_ctx *state, const uint8_t key[32], uint64_t nonce);

#define aes256ctr_select AES256CTR_NAMESPACE(select)
void aes256ctr_select(aes256ctr_ctx *state, uint64_t nonce);

#define aes256ctr_squeezeblocks AES256CTR_NAMESPACE(squeezeblocks)
void aes256ctr_squeezeblocks(uint8_t *out, size_t nblocks, aes256ctr_ctx *state);

#endif
