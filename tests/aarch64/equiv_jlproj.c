/* Equivalence test (and microbenchmark) for jlproj.c: NEON kernels, the JL
 * table lookup in jlproj_collapsmat32 and the multi-threaded loops */
#include "jlproj.c"
#include "kt.h"

static uint64_t polxvec_digest(polxvec r) {
  /* emit all CRT projections of all elements (and the widths) */
  for(size_t i = 0; i < r->len; i++) {
    for(int k = 0; k < K; k++) kt_emit(r->proj[k][i*r->stride]->c, sizeof(poly));
    if(r->widths) kt_emit(&r->widths[i*r->stride], sizeof(double));
  }
  return 0;
}

int main(int argc, char **argv) {
  int bench = kt_init(argc, argv);
  size_t i, t;
  const size_t maxlen = 40;
  uint8_t *mat = malloc(256 * N / 8 * maxlen);
  poly *p = malloc(sizeof(poly) * maxlen);
  int32_t out[256];
  int64_t alpha[256];
  uint8_t buf[256 * QBYTES];

  /* 1) polyvec_jlproj_add (secret polynomials x public matrix) */
  for(t = 0; t < 400; t++) {
    size_t len = (t < 4) ? t : (size_t)kt_range(1, 8);
    for(i = 0; i < 256 * N / 8 * len; i++) mat[i] = (uint8_t)kt_rnd();
    if(t == 4) memset(mat, 0xFF, 256 * N / 8 * len);
    for(i = 0; i < len; i++) for(int j = 0; j < N; j++)
      p[i]->c[j] = (t % 3 == 0) ? (int16_t)kt_range(-2, 2) : kt_i16();
    for(i = 0; i < 256; i++) out[i] = (t & 1) ? (int32_t)kt_rnd() : 0;
    polyvec_jlproj_add(out, p, len, mat);
    kt_emit(out, sizeof out);
    if(t % 50 == 0) {
      polyvec_jlproj_add_bin1(out, p, len, mat, mat + 7);
      kt_emit(out, sizeof out);
    }
  }

  /* 2) polxvec_jlproj_collapsmat (public matrix x public challenge) */
  for(t = 0; t < 12; t++) {
    size_t lens[] = {37, 16, 5, 1, 17, 32};
    size_t len = lens[t % 6];
    polxvec r;
    for(i = 0; i < 256 * N / 8 * len; i++) mat[i] = (uint8_t)kt_rnd();
    for(i = 0; i < sizeof buf; i++) buf[i] = (uint8_t)kt_rnd();
    if(t == 6) memset(buf, 0xFF, sizeof buf);
    if(t == 7) memset(mat, 0xFF, 256 * N / 8 * len);
    jlproj_expand_challenge(alpha, buf);
    kt_emit(alpha, sizeof alpha);
    polxvec_init(r, len, 1);
    polxvec_jlproj_collapsmat(r, mat, alpha);
    polxvec_digest(r);
    polxvec_free(r);
  }

  /* 3) raw jlproj_collapsmat32 kernel: random data, full 64-bit alpha range */
  {
    static alignas(64) uint8_t m3[16 * N * 32 + 4096];
    static alignas(64) int64_t row[16 * N];
    const size_t lens[] = {32, 64, 16 * N, 5 * N, 3 * N + 32};

    for(t = 0; t < 40; t++) {
      size_t len = lens[t % 5];
      for(i = 0; i < sizeof m3; i++) m3[i] = (uint8_t)kt_rnd();
      for(i = 0; i < 32; i++) alpha[i] = (int64_t)kt_rnd();
      for(i = 0; i < len; i++) row[i] = (int64_t)kt_rnd();
      jlproj_collapsmat32(row, m3 + 64 * (t % 8), len, alpha);
      kt_emit(row, len * 8);
    }
  }

  if(bench) {
    polxvec r;
    for(i = 0; i < 256 * N / 8 * maxlen; i++) mat[i] = (uint8_t)kt_rnd();
    for(i = 0; i < maxlen; i++) for(int j = 0; j < N; j++) p[i]->c[j] = (int16_t)kt_range(-20, 20);
    KT_BENCH("poly_jlproj_add", 2000, { poly_jlproj_add(out, p[0], mat); CLOBBER(); });
    for(i = 0; i < sizeof buf; i++) buf[i] = (uint8_t)kt_rnd();
    jlproj_expand_challenge(alpha, buf);
    polxvec_init(r, 37, 1);
    KT_BENCH("polxvec_jlproj_collapsmat(len=37)", 3, { polxvec_jlproj_collapsmat(r, mat, alpha); CLOBBER(); });
    polxvec_free(r);
  }
  kt_done();
  free(mat); free(p);
  return 0;
}
