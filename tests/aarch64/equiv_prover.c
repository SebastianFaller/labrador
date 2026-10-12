/*
 * End-to-end equivalence test: deterministic Labrador proofs (prove + reduce)
 * for several instance shapes. Writes a SHAKE128 hash of the prover's output
 * (statement hash, output witness, JL projection) and the verification result
 * per instance. The ref and opt builds, and opt with any LAB_THREADS, must
 * produce identical files; every proof must verify.
 */
#include <stdio.h>
#include <string.h>
#include "labrador.h"
#include "proofsystem.h"
#include "polx.h"
#include "test_proofsystem_setup.h"
#include "comkey.h"
#include "fips202.h"
#include "kt.h"

/* deterministic replacement for randombytes.c */
static shake128incctx rb;
void randombytes(uint8_t *out, size_t outlen) { shake128_inc_squeeze(out, outlen, &rb); }

static int run(unsigned seedno, size_t r, size_t len, size_t ncnst) {
  size_t nn, iwtbits, owtbits, pibits, nonce = 0, i;
  uint8_t seed[16] = {0}, h[16];
  char line[160];
  polxvec sxl, *sxq;
  statement ist, ostp, ostv;
  witness iwt, owt;
  lab_params pp;
  lab_proof pi;
  shake128incctx hs;
  int ok = 0;

  seed[0] = (uint8_t)seedno;
  shake128_inc_init(&rb);
  shake128_inc_absorb(&rb, seed, 16);
  shake128_inc_finalize(&rb);

  witness_init(iwt, r, r);
  ps_witness_set(iwt, sxl, &sxq, &nn, &iwtbits, len, seed, &nonce);
  comkey_init(nn);
  statement_init(ist, iwt->r, iwt->r);
  ps_statement_set(ist, iwt, sxl, sxq, nn, ncnst, seed, &nonce);
  if(!verify(ist, iwt) || lab_params_gen(pp, &pibits, &owtbits, ist, 0, 0, 0, 1, 1, JL_L2_SLACK)) {
    printf("FAIL seed=%u r=%zu len=%zu ncnst=%zu: no valid instance\n", seedno, r, len, ncnst);
    return 0;
  }

  ldr_prove(pi, ostp, owt, ist, iwt, pp);
  shake128_inc_init(&hs);
  shake128_inc_absorb(&hs, ostp->h, 16);
  for(i = 0; i < owt->r; i++)
    shake128_inc_absorb(&hs, (const uint8_t *)owt->s[i], owt->n[i]*sizeof(poly));
  shake128_inc_absorb(&hs, (const uint8_t *)pi->p, sizeof(pi->p));
  shake128_inc_finalize(&hs);
  shake128_inc_squeeze(h, 16, &hs);

  if(ldr_reduce(ostv, ist, pi, pp) == 0) {
    ok = verify(ostv, owt) && !memcmp(ostv->h, ostp->h, 16);
    statement_free(ostv);
  }
  snprintf(line, sizeof line, "seed=%u r=%zu len=%zu ncnst=%zu verify=%d hash=", seedno, r, len, ncnst, ok);
  kt_emit(line, strlen(line));
  for(i = 0; i < 16; i++) { snprintf(line, sizeof line, "%02x", h[i]); kt_emit(line, 2); }
  kt_emit("\n", 1);
  if(!ok) printf("FAIL seed=%u r=%zu len=%zu ncnst=%zu: proof does not verify\n", seedno, r, len, ncnst);

  lab_proof_free(pi);
  lab_params_free(pp);
  statement_free(ostp);
  witness_free(owt);
  free(sxq);
  polxvec_free(sxl);
  statement_free(ist);
  witness_free(iwt);
  comkey_free();
  return ok;
}

int main(int argc, char **argv) {
  static const size_t R[]   = {5,    2,    1,     5};
  static const size_t LEN[] = {1000, 2000, 10000, 10000};
  static const size_t NC[]  = {5,    5,    5,     2};
  unsigned s;
  size_t c;
  int ok = 1;

  kt_init(argc, argv);
  for(c = 0; c < sizeof(R)/sizeof(R[0]); c++)
    for(s = 1; s <= 2; s++)
      ok &= run(s, R[c], LEN[c], NC[c]);
  kt_done();
  return !ok;
}
