/* Equivalence test (and microbenchmark) for the NEON kernels in polz.c */
#include "polz.c"
#include "kt.h"

static void rnd_poly(poly a) { for(int i = 0; i < N; i++) a->c[i] = kt_i16(); }
static void rnd_polz(polz a) { for(int k = 0; k < L; k++) rnd_poly(a->limbs[k]); }
/* canonical-ish polz: limbs in [0,2^14) (top limb small), as produced by the library */
static void rnd_polz_canon(polz a) {
  for(int k = 0; k < L; k++) for(int i = 0; i < N; i++)
    a->limbs[k]->c[i] = (int16_t)(kt_rnd() & ((k < L-1) ? 0x3FFF : ((1 << (LOGQ-14*(L-1))) - 1)));
}

int main(int argc, char **argv) {
  int bench = kt_init(argc, argv);
  int t, pi;
  polz z, z2;
  poly p;
  zz a;

  for(t = 0; t < 3000; t++) {
    if(t & 1) rnd_polz(z); else rnd_polz_canon(z);
    *z2 = *z; polz_reduce(z2); kt_emit(z2, sizeof(polz));
    *z2 = *z; polz_center(z2); kt_emit(z2, sizeof(polz));
    for(pi = 0; pi < K; pi++) { polz_topoly_montgomery(p, z, primes[pi]); kt_emit(p, sizeof(poly)); }
    rnd_poly(p);
    for(int k = 0; k < L; k++) a->limbs[k] = (t & 2) ? kt_i16() : (int16_t)(kt_rnd() & 0x3FFF);
    if(t < 2*K) *a = (t & 1) ? *modulus->xvec[t/2] : *modulus->pmq;
    zz_poly_mul(z2, a, p); kt_emit(z2, sizeof(polz));
    *z2 = *z; zz_poly_fma(z2, a, p); kt_emit(z2, sizeof(polz));
  }
  if(bench) {
    rnd_polz_canon(z); rnd_poly(p); *a = *modulus->xvec[1];
    KT_BENCH("polz_reduce", 20000, { polz_reduce(z); CLOBBER(); });
    KT_BENCH("polz_center", 20000, { *z2 = *z; polz_center(z2); CLOBBER(); });
    KT_BENCH("polz_topoly_montgomery", 20000, { polz_topoly_montgomery(p, z, primes[1]); CLOBBER(); });
    KT_BENCH("zz_poly_mul", 20000, { zz_poly_mul(z2, a, p); CLOBBER(); });
    KT_BENCH("zz_poly_fma", 20000, { zz_poly_fma(z2, a, p); CLOBBER(); });
    KT_BENCH("polz_copy(baseline_for_center)", 20000, { *z2 = *z; CLOBBER(); });
  }
  kt_done();
  return 0;
}
