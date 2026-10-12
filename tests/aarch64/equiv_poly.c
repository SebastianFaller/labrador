/* Equivalence test (and microbenchmark) for the NEON kernels in poly.c */
#include "poly.c"
#include "kt.h"
#define MAXV 300

static poly A[MAXV], B[MAXV], R[MAXV];

static void rnd_poly(poly a) { for(int i = 0; i < N; i++) a->c[i] = kt_i16(); }
/* coefficients in [-bound,bound] */
static void rnd_poly_b(poly a, int bound) { for(int i = 0; i < N; i++) a->c[i] = (int16_t)kt_range(-bound, bound); }

int main(int argc, char **argv) {
  int bench = kt_init(argc, argv);
  int pi, t;
  size_t i;

  for(pi = 0; pi < K; pi++) {
    const pdata_ptr pr = primes[pi];
    /* sanity of the prime constants the NEON code relies on */
    if(!(pr->p > 0 && pr->v > 0 && pr->p != INT16_MIN) || (int16_t)(pr->p*pr->pinv) != 1) {
      printf("FAIL prime constants %d\n", pi); return 1;
    }
    for(t = 0; t < 200; t++) {
      rnd_poly(A[0]); rnd_poly(B[0]); rnd_poly(R[0]);
      *R[1] = *A[0]; poly_reduce(R[1], pr); kt_emit(R[1], sizeof(poly));
      *R[1] = *A[0]; poly_caddp(R[1], pr); kt_emit(R[1], sizeof(poly));
      *R[1] = *R[0]; poly_quot_add(R[1], A[0], pr); kt_emit(R[1], sizeof(poly));
      poly_pointwise(R[1], A[0], B[0], pr); kt_emit(R[1], sizeof(poly));
      *R[1] = *R[0]; poly_pointwise_add(R[1], A[0], B[0], pr); kt_emit(R[1], sizeof(poly));
      {
        int16_t svals[] = {INT16_MIN, INT16_MAX, 0, 1, -1, pr->t, pr->u, pr->f, pr->s, pr->mont};
        int16_t s = (t < 10) ? svals[t] : kt_i16();
        poly_scale(R[1], A[0], s, pr); kt_emit(R[1], sizeof(poly));
      }
    }
    for(t = 0; t < 40; t++) {
      size_t len = (t < 8) ? (size_t)t : (size_t)kt_range(0, 40);
      ssize_t sr = kt_range(1, 3), sb = kt_range(1, 3);
      for(i = 0; i < MAXV; i++) { rnd_poly(B[i]); rnd_poly(R[i]); }
      rnd_poly(A[0]);
      polyvec_poly_pointwise(R, A[0], B, sr, sb, len, pr); kt_emit(R, sizeof R);
      polyvec_poly_pointwise_add(R, A[0], B, sr, sb, len, pr); kt_emit(R, sizeof R);
    }
    for(t = 0; t < 60; t++) {
      size_t len = (t < 6) ? (size_t)t : (t < 20) ? (size_t)kt_range(90, 140) : (size_t)kt_range(0, 140);
      ssize_t sa = kt_range(1, 2), sb = kt_range(1, 2);
      for(i = 0; i < MAXV; i++) {
        if(t & 1) { rnd_poly(A[i]); rnd_poly(B[i]); }
        else { rnd_poly_b(A[i], pr->p); rnd_poly_b(B[i], pr->p); }  // typical (reduced) inputs
      }
      polyvec_sprod_pointwise(R[0], A, B, sa, sb, len, pr); kt_emit(R[0], sizeof(poly));
    }

    for(t = 0; t < 120; t++) {
      size_t degs[] = {2, 4, 8, 2, 4, 1};
      size_t deg = degs[t % 6];
      size_t clen = (deg == 1) ? 1 : (size_t)kt_range(deg/2+1, deg);
      size_t blen = (t < 12) ? (size_t)(t/6 + 1) : (size_t)kt_range(1, 40);
      ssize_t sa = kt_range(1, 2), sb = kt_range(1, 2), sc = kt_range(1, 2);
      for(i = 0; i < MAXV; i++) {
        if(t & 1) { rnd_poly(A[i]); rnd_poly(B[i]); }
        else { rnd_poly_b(A[i], pr->p); rnd_poly_b(B[i], pr->p); }
        rnd_poly(R[i]);
      }
      polyvec_sprod_extension(R, A, B, sc, sa, sb, deg, clen, blen, pr); kt_emit(R, sizeof(poly)*20);
    }
    for(t = 0; t < 20; t++) {
      size_t len = kt_range(0, 30);
      ssize_t st = kt_range(1, 3);
      for(i = 0; i < MAXV; i++) rnd_poly(R[i]);
      polyvec_setzero(R, st, len); kt_emit(R, sizeof(poly)*100);
    }
  }

  if(bench) {
    const pdata_ptr pr = primes[0];
    for(i = 0; i < MAXV; i++) { rnd_poly_b(A[i], pr->p); rnd_poly_b(B[i], pr->p); rnd_poly_b(R[i], pr->p); }
    KT_BENCH("poly_reduce", 2000, { poly_reduce(R[0], pr); CLOBBER(); });
    KT_BENCH("poly_caddp", 2000, { poly_caddp(R[0], pr); CLOBBER(); });
    KT_BENCH("poly_quot_add", 2000, { poly_quot_add(R[0], A[0], pr); CLOBBER(); });
    KT_BENCH("poly_scale", 2000, { poly_scale(R[0], A[0], 1234, pr); CLOBBER(); });
    KT_BENCH("poly_pointwise", 2000, { poly_pointwise(R[0], A[0], B[0], pr); CLOBBER(); });
    KT_BENCH("poly_pointwise_add", 2000, { poly_pointwise_add(R[0], A[0], B[0], pr); CLOBBER(); });
    KT_BENCH("polyvec_poly_pointwise(len=32)", 500, { polyvec_poly_pointwise(R, A[0], B, 1, 1, 32, pr); CLOBBER(); });
    KT_BENCH("polyvec_poly_pointwise_add(len=32)", 500, { polyvec_poly_pointwise_add(R, A[0], B, 1, 1, 32, pr); CLOBBER(); });
    KT_BENCH("polyvec_sprod_pointwise(len=64)", 300, { polyvec_sprod_pointwise(R[0], A, B, 1, 1, 64, pr); CLOBBER(); });
    KT_BENCH("polyvec_sprod_pointwise(len=7)", 2000, { polyvec_sprod_pointwise(R[0], A, B, 1, 1, 7, pr); CLOBBER(); });
    KT_BENCH("polyvec_sprod_extension(deg=2,clen=2,blen=32)", 100, { polyvec_sprod_extension(R, A, B, 1, 1, 1, 2, 2, 32, pr); CLOBBER(); });
    KT_BENCH("polyvec_sprod_extension(deg=4,clen=4,blen=32)", 100, { polyvec_sprod_extension(R, A, B, 1, 1, 1, 4, 4, 32, pr); CLOBBER(); });
    KT_BENCH("polyvec_sprod_extension(deg=8,clen=5,blen=37)", 100, { polyvec_sprod_extension(R, A, B, 1, 1, 1, 8, 5, 37, pr); CLOBBER(); });
    KT_BENCH("polyvec_add(len=32)", 500, { polyvec_add(R, A, B, 1, 1, 1, 32); CLOBBER(); });
    KT_BENCH("polyvec_setzero(len=32)", 500, { polyvec_setzero(R, 1, 32); CLOBBER(); });
  }
  kt_done();
  return 0;
}
