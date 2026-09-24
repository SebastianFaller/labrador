#ifndef CPUCYCLES_H
#define CPUCYCLES_H

#include <stdint.h>

// #define USE_RDPMC

#if defined(__x86_64__)

#ifdef USE_RDPMC  /* Needs echo 2 > /sys/devices/cpu/rdpmc */

static inline uint64_t cpucycles(void) {
  const uint32_t ecx = (1U << 30) + 1;
  uint64_t result;

  __asm__ volatile ("rdpmc; shlq $32,%%rdx; orq %%rdx,%%rax"
    : "=a" (result) : "c" (ecx) : "rdx");

  return result;
}

#else

static inline uint64_t cpucycles(void) {
  uint64_t result;

  __asm__ volatile ("rdtsc; shlq $32,%%rdx; orq %%rdx,%%rax"
    : "=a" (result) : : "%rdx");

  return result;
}

#endif

#elif defined(__aarch64__)

/* virtual counter (fixed frequency, not core cycles; 54 MHz on the Pi 4) */
static inline uint64_t cpucycles(void) {
  uint64_t result;

  __asm__ volatile ("isb; mrs %0, cntvct_el0" : "=r" (result));

  return result;
}

#else

#include <time.h>

static inline uint64_t cpucycles(void) {
  struct timespec ts;

  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec*1000000000ULL + (uint64_t)ts.tv_nsec;
}

#endif

uint64_t cpucycles_overhead(void);
void warmup(void);

#endif
