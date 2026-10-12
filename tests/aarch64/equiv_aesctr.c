/* Equivalence test (and microbenchmark) for the AES-CTR PRG in aesctr.c:
 * the NEON bitsliced AES and the multi-threaded squeeze against the scalar
 * bitsliced AES. Also checks every first block against textbook FIPS-197 AES
 * (block i = AES_k(nonce || 64-bit little-endian counter)). */
#include "aesctr.c"
#include "kt.h"
#include "ref_aes.h"

static int check_ref(const uint8_t *key, int nk, uint64_t nonce, const uint8_t *out) {
  uint8_t in[16], exp[16], w[240];
  memset(in, 0, 16);
  for(int i = 0; i < 8; i++) in[i] = (uint8_t)(nonce >> 8*i);  /* counter 0 */
  ref_expand(w, key, nk);
  ref_encrypt(exp, in, w, nk + 6);
  return memcmp(exp, out, 16) == 0;
}

int main(int argc, char **argv) {
  int bench = kt_init(argc, argv);
  static uint8_t out[600 * AES128CTR_BLOCKBYTES];
  uint8_t key[32];
  aes128ctr_ctx s128;
  aes256ctr_ctx s256;
  size_t i, n;
  int t, bad = 0;

  ref_init_sbox();
  for(t = 0; t < 200; t++) {
    uint64_t nonce = (t < 2) ? (uint64_t)-t : kt_rnd();
    n = (t < 20) ? (size_t)t : (size_t)kt_range(1, 600);  /* covers the threaded sizes */
    for(i = 0; i < 32; i++) key[i] = (uint8_t)kt_rnd();

    aes128ctr_init(&s128, key, nonce);
    aes128ctr_squeezeblocks(out, n, &s128); kt_emit(out, n * AES128CTR_BLOCKBYTES);
    if(n && !check_ref(key, 4, nonce, out)) bad++;
    aes128ctr_squeezeblocks(out, 3, &s128); kt_emit(out, 3 * AES128CTR_BLOCKBYTES);
    aes128ctr_select(&s128, nonce + 1);
    aes128ctr_squeezeblocks(out, 2, &s128); kt_emit(out, 2 * AES128CTR_BLOCKBYTES);

    aes256ctr_init(&s256, key, nonce);
    aes256ctr_squeezeblocks(out, n, &s256); kt_emit(out, n * AES256CTR_BLOCKBYTES);
    if(n && !check_ref(key, 8, nonce, out)) bad++;
    aes256ctr_select(&s256, nonce ^ 5);
    aes256ctr_squeezeblocks(out, 2, &s256); kt_emit(out, 2 * AES256CTR_BLOCKBYTES);
  }
  if(bad) { printf("FAIL: %d first blocks differ from FIPS-197 AES\n", bad); return 1; }

  if(bench) {
    aes128ctr_init(&s128, key, 0);
    aes256ctr_init(&s256, key, 0);
    KT_BENCH("aes128ctr_squeezeblocks(1 block = 512 B)", 2000, { aes128ctr_squeezeblocks(out, 1, &s128); CLOBBER(); });
    KT_BENCH("aes256ctr_squeezeblocks(1 block = 512 B)", 2000, { aes256ctr_squeezeblocks(out, 1, &s256); CLOBBER(); });
    KT_BENCH("aes128ctr_squeezeblocks(256 blocks)", 10, { aes128ctr_squeezeblocks(out, 256, &s128); CLOBBER(); });
  }
  kt_done();
  return 0;
}
