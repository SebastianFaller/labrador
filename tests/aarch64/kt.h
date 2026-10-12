/*
 * Helpers shared by the aarch64 equivalence tests (see README-aarch64.md).
 *
 * Every test is compiled twice: "ref" with -DLABRADOS_FORCE_AVX512
 * -DLAB_NO_THREADS (the original AVX-512 code through SIMDe, serial) and "opt"
 * without (the aarch64 code: NEON kernels, multi-threading). Both binaries
 * write all results to the file given as argv[1]; the files must be
 * byte-identical. With argv[2] == "bench" a test also prints timings.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define CLOBBER() __asm__ volatile("" ::: "memory")

static uint64_t kt_s = 0x9E3779B97F4A7C15ULL;
static uint64_t kt_rnd(void) { kt_s ^= kt_s << 13; kt_s ^= kt_s >> 7; kt_s ^= kt_s << 17; return kt_s; }
/* random int16 with a bias towards edge values */
static int16_t kt_i16(void) {
  uint64_t r = kt_rnd();
  switch(r & 15) {
    case 0: return INT16_MIN;
    case 1: return INT16_MAX;
    case 2: return (int16_t)((r >> 8) % 7) - 3;
    case 3: return -1;
    default: return (int16_t)(r >> 16);
  }
}
/* random value in [lo,hi] */
static int64_t kt_range(int64_t lo, int64_t hi) { return lo + (int64_t)(kt_rnd() % (uint64_t)(hi - lo + 1)); }

static FILE *kt_out;
static void kt_emit(const void *p, size_t n) {
  if(fwrite(p, 1, n, kt_out) != n) { perror("fwrite"); exit(1); }
}

static double kt_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

/* run body reps times per round, 31 rounds, print the best time per rep */
#define KT_BENCH(name, reps, body) do {                                   \
    double best_ = 1e300;                                                 \
    for(int r_ = 0; r_ < 31; r_++) {                                      \
      double t0_ = kt_now();                                              \
      for(long i_ = 0; i_ < (reps); i_++) { body; }                       \
      double t_ = (kt_now() - t0_) / (reps);                              \
      if(t_ < best_) best_ = t_;                                          \
    }                                                                     \
    printf("BENCH %-48s %12.1f ns\n", name, 1e9 * best_);                 \
  } while(0)

static int kt_init(int argc, char **argv) {
  if(argc < 2) { fprintf(stderr, "usage: %s outfile [bench]\n", argv[0]); exit(1); }
  kt_out = fopen(argv[1], "wb");
  if(!kt_out) { perror("fopen"); exit(1); }
  return argc > 2 && !strcmp(argv[2], "bench");
}

static void kt_done(void) {
  if(fclose(kt_out)) { perror("fclose"); exit(1); }
}
