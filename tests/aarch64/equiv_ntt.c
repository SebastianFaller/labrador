/* Equivalence test (and microbenchmark) for the NEON NTT in ntt.c:
 * poly_ntt, poly_invntt and poly_nttunpack on the full int16 input range */
#include "ntt.c"
#include "kt.h"

static void rnd_poly(poly a) { for(int i = 0; i < N; i++) a->c[i] = kt_i16(); }

int main(int argc, char **argv) {
  int bench = kt_init(argc, argv);
  int pi, t, i;
  poly a, r;

  for(pi = 0; pi < K; pi++) {
    const pdata_ptr pr = primes[pi];
    for(t = 0; t < 5000; t++) {
      if(t < 4)  /* constant inputs: extremes and zero */
        for(i = 0; i < N; i++) a->c[i] = (int16_t[]){INT16_MIN, INT16_MAX, 0, -1}[t];
      else
        rnd_poly(a);
      poly_ntt(r, a, pr); kt_emit(r, sizeof(poly));
      poly_invntt(r, a, pr); kt_emit(r, sizeof(poly));
      *r = *a; poly_nttunpack(r, pr); kt_emit(r, sizeof(poly));
    }
  }

  if(bench) {
    rnd_poly(a);
    KT_BENCH("poly_ntt", 20000, { poly_ntt(r, a, primes[0]); CLOBBER(); });
    KT_BENCH("poly_invntt", 20000, { poly_invntt(r, a, primes[0]); CLOBBER(); });
    KT_BENCH("poly_nttunpack", 20000, { poly_nttunpack(r, primes[0]); CLOBBER(); });
  }
  kt_done();
  return 0;
}
