#ifndef LAB_PARALLEL_H
#define LAB_PARALLEL_H

/*
 * Multi-threading for labrados on aarch64 (e.g. Raspberry Pi 4).
 *
 * LAB_PARALLEL is defined on aarch64 only (and not with -DLAB_NO_THREADS).
 * Every multi-threaded code path is under #ifdef LAB_PARALLEL, with the
 * original serial code in the #else branch, so on x86-64 this header expands
 * to nothing and the original code is compiled unchanged.
 *
 * With LAB_PARALLEL, a minimal persistent thread pool runs data-parallel loops:
 *
 *   lab_parallel_for(n, grain, fn, arg)
 *
 * splits the index range [0,n) into at most T contiguous sub-ranges whose
 * boundaries are multiples of `grain` (except the end n) and calls
 * fn(arg, begin, end) once per sub-range, concurrently on T threads (the
 * calling thread takes the first sub-range).  It returns after all calls have
 * finished.  T = number of threads (env LAB_THREADS, default 4, 1 = serial).
 *
 * The loop runs serially, as a single call fn(arg, 0, n), when
 *   - T == 1, or n <= grain (only one sub-range), or
 *   - it is called from inside a parallel region (no nested parallelism), or
 *   - another thread is already using the pool (re-entrancy from multiple
 *     application threads).
 *
 * Callers must only parallelise loops whose iterations write disjoint memory
 * and whose result does not depend on how the range is split.  Splitting
 * never depends on data, only on (n, grain, T), so it leaks nothing secret,
 * and proofs are bit-identical for every T.
 *
 * The implementation is compiled into exactly one translation unit (polx.c)
 * via LAB_PARALLEL_IMPL (see the end of this file), so the library's source
 * list stays unchanged.
 */

#if defined(__aarch64__) && !defined(LAB_NO_THREADS)
#define LAB_PARALLEL 1
#endif

/*
 * LAB_SERIAL(f) starts the definition of the serial kernel of a vector function
 * f that has a parallel wrapper: with LAB_PARALLEL it is the static helper
 * f_ser (the public f, defined further down in the same file, splits the work
 * and calls f_ser on each part); without, it is the public function f itself.
 */
#ifdef LAB_PARALLEL
#define LAB_SERIAL(f) static void f##_ser
#else
#define LAB_SERIAL(f) void f
#endif

#ifdef LAB_PARALLEL

#include <stddef.h>

typedef void (*lab_par_fn)(void *arg, size_t begin, size_t end);

void lab_parallel_for(size_t n, size_t grain, lab_par_fn fn, void *arg);
int lab_parallel_nthreads(void);

/*
 * Asynchronous variant: splits [0,n) over the worker threads only (not the
 * caller) and returns immediately with 1; the caller must then call
 * lab_parallel_wait() before it uses the results (and it cannot use the pool
 * in between: its lab_parallel_for calls run serially).  If the pool is not
 * available, fn(arg,0,n) runs synchronously and 0 is returned (no wait needed).
 */
int lab_parallel_async(size_t n, size_t grain, lab_par_fn fn, void *arg);
void lab_parallel_wait(void);

#endif /* LAB_PARALLEL */

#endif /* LAB_PARALLEL_H */

/*
 * The thread pool itself, compiled into the one file that defines
 * LAB_PARALLEL_IMPL before including this header (outside the include guard,
 * since that file may already have included the header via another one).
 */
#if defined(LAB_PARALLEL) && defined(LAB_PARALLEL_IMPL) && !defined(LAB_PARALLEL_IMPL_DONE)
#define LAB_PARALLEL_IMPL_DONE

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

#define LAB_MAXTHREADS 16
#define LAB_SPIN 20000   /* spin iterations before a worker sleeps */

#define LAB_CPU_RELAX() __asm__ __volatile__("yield" ::: "memory")

static struct {
  int nthreads;                 /* 0 = not initialised */
  pthread_t tid[LAB_MAXTHREADS];
  pthread_mutex_t mtx;          /* protects sleeping/waking */
  pthread_cond_t cv;
  pthread_mutex_t busy;         /* held by the thread that owns the pool */
  _Atomic uint64_t gen;         /* job generation counter */
  _Atomic int pending;          /* workers that have not finished the job */
  _Atomic int sleepers;
  /* current job */
  lab_par_fn fn;
  void *arg;
  size_t bounds[LAB_MAXTHREADS+1];
  int nparts;
  int base;                     /* thread id that runs part 0 (0: caller) */
  int dyn;                      /* dynamic scheduling (async jobs) */
  _Atomic size_t next;          /* next chunk (dynamic scheduling) */
  size_t nchunks, grain, n;
} lab_pool = { .mtx = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER,
               .busy = PTHREAD_MUTEX_INITIALIZER };

static pthread_once_t lab_pool_once = PTHREAD_ONCE_INIT;
static _Thread_local int lab_in_parallel = 0;

/* grab chunks of the current dynamic job until none is left */
static void lab_run_dynamic(void) {
  size_t c, b, e;

  while((c = atomic_fetch_add_explicit(&lab_pool.next, 1, memory_order_relaxed)) < lab_pool.nchunks) {
    b = c * lab_pool.grain;
    e = b + lab_pool.grain < lab_pool.n ? b + lab_pool.grain : lab_pool.n;
    lab_pool.fn(lab_pool.arg, b, e);
  }
}

static void *lab_worker(void *p) {
  const int id = (int)(intptr_t)p;
  uint64_t seen = 0, g;
  long spin;

  lab_in_parallel = 1;  /* workers never start nested regions */
  for(;;) {
    /* wait for a new generation: spin first, then sleep */
    for(spin = 0; (g = atomic_load_explicit(&lab_pool.gen, memory_order_acquire)) == seen; spin++) {
      if(spin < LAB_SPIN) { LAB_CPU_RELAX(); continue; }
      pthread_mutex_lock(&lab_pool.mtx);
      atomic_fetch_add(&lab_pool.sleepers, 1);
      while(atomic_load(&lab_pool.gen) == seen)  /* seq_cst: pairs with sleepers */
        pthread_cond_wait(&lab_pool.cv, &lab_pool.mtx);
      atomic_fetch_sub(&lab_pool.sleepers, 1);
      pthread_mutex_unlock(&lab_pool.mtx);
    }
    seen = g;
    if(lab_pool.dyn)
      lab_run_dynamic();
    else {
      const int part = id - lab_pool.base;
      if(part >= 0 && part < lab_pool.nparts && lab_pool.bounds[part] < lab_pool.bounds[part+1])
        lab_pool.fn(lab_pool.arg, lab_pool.bounds[part], lab_pool.bounds[part+1]);
    }
    atomic_fetch_sub_explicit(&lab_pool.pending, 1, memory_order_acq_rel);
  }
  return NULL;
}

static void lab_pool_atfork_child(void) {
  /* worker threads do not survive fork(): run serially in the child */
  lab_pool.nthreads = 1;
  pthread_mutex_init(&lab_pool.busy, NULL);
  pthread_mutex_init(&lab_pool.mtx, NULL);
  pthread_cond_init(&lab_pool.cv, NULL);
}

static void lab_pool_init(void) {
  const char *s = getenv("LAB_THREADS");
  long t = 4, i;
  pthread_attr_t attr;

  if(s && *s) t = strtol(s, NULL, 10);
  if(t < 1) t = 1;
  if(t > LAB_MAXTHREADS) t = LAB_MAXTHREADS;
  lab_pool.nthreads = (int)t;
  if(t == 1) return;

  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, (size_t)16 << 20);  /* VLAs in callees */
  for(i = 1; i < t; i++) {
    if(pthread_create(&lab_pool.tid[i], &attr, lab_worker, (void *)(intptr_t)i)) {
      /* could not start all threads: never use the pool */
      lab_pool.nthreads = 1;
      break;
    }
    pthread_detach(lab_pool.tid[i]);
  }
  pthread_attr_destroy(&attr);
  if(lab_pool.nthreads > 1)
    pthread_atfork(NULL, NULL, lab_pool_atfork_child);
  /* if only some workers could be started, they stay asleep forever */
}

int lab_parallel_nthreads(void) {
  pthread_once(&lab_pool_once, lab_pool_init);
  return lab_pool.nthreads;
}

#ifdef LAB_PAR_STATS
#include <stdio.h>
static size_t lab_stat_calls, lab_stat_par;
static void lab_stat_print(void) { fprintf(stderr, "lab_parallel_for: %zu calls, %zu parallel\n", lab_stat_calls, lab_stat_par); }
#endif

/* split [0,n) into t parts at multiples of grain; publish and wake workers */
static void lab_dispatch(size_t n, size_t grain, size_t chunks, size_t t, int base,
                         lab_par_fn fn, void *arg) {
  size_t i;

  /* contiguous split, chunk counts differ by at most one */
  for(i = 0; i <= t; i++) {
    size_t b = (chunks * i / t) * grain;
    lab_pool.bounds[i] = b < n ? b : n;
  }
  for(i = t + 1; i <= (size_t)LAB_MAXTHREADS; i++)
    lab_pool.bounds[i] = n;
  lab_pool.nparts = (int)t;
  lab_pool.base = base;
  lab_pool.dyn = 0;
  lab_pool.fn = fn;
  lab_pool.arg = arg;
  atomic_store_explicit(&lab_pool.pending, lab_pool.nthreads - 1, memory_order_relaxed);

  /* publish the job and wake sleeping workers */
  atomic_fetch_add(&lab_pool.gen, 1);  /* seq_cst, see worker */
  if(atomic_load(&lab_pool.sleepers)) {
    pthread_mutex_lock(&lab_pool.mtx);
    pthread_cond_broadcast(&lab_pool.cv);
    pthread_mutex_unlock(&lab_pool.mtx);
  }
}

static void lab_join(void) {
  while(atomic_load_explicit(&lab_pool.pending, memory_order_acquire) > 0)
    LAB_CPU_RELAX();
  pthread_mutex_unlock(&lab_pool.busy);
}

void lab_parallel_for(size_t n, size_t grain, lab_par_fn fn, void *arg) {
  size_t chunks, t;
#ifdef LAB_PAR_STATS
  if(!lab_stat_calls++) atexit(lab_stat_print);
#endif

  if(!n) return;
  if(!grain) grain = 1;
  chunks = (n + grain - 1) / grain;
  if(lab_in_parallel || chunks < 2 || lab_parallel_nthreads() < 2
     || pthread_mutex_trylock(&lab_pool.busy)) {
    fn(arg, 0, n);
    return;
  }
#ifdef LAB_PAR_STATS
  lab_stat_par++;
#endif

  t = (size_t)lab_pool.nthreads;
  if(t > chunks) t = chunks;
  lab_dispatch(n, grain, chunks, t, 0, fn, arg);

  lab_in_parallel = 1;
  fn(arg, lab_pool.bounds[0], lab_pool.bounds[1]);
  lab_in_parallel = 0;

  lab_join();
}

static _Thread_local int lab_async_pending = 0;

int lab_parallel_async(size_t n, size_t grain, lab_par_fn fn, void *arg) {
  size_t chunks;

  if(!n) return 0;
  if(!grain) grain = 1;
  chunks = (n + grain - 1) / grain;
  if(lab_in_parallel || lab_async_pending || lab_parallel_nthreads() < 2
     || pthread_mutex_trylock(&lab_pool.busy)) {
    fn(arg, 0, n);
    return 0;
  }
  /* dynamic scheduling: workers grab chunks; the caller helps in
     lab_parallel_wait() with whatever is left at that point */
  lab_pool.dyn = 1;
  lab_pool.nchunks = chunks;
  lab_pool.grain = grain;
  lab_pool.n = n;
  lab_pool.fn = fn;
  lab_pool.arg = arg;
  atomic_store_explicit(&lab_pool.next, 0, memory_order_relaxed);
  atomic_store_explicit(&lab_pool.pending, lab_pool.nthreads - 1, memory_order_relaxed);
  atomic_fetch_add(&lab_pool.gen, 1);  /* seq_cst, see worker */
  if(atomic_load(&lab_pool.sleepers)) {
    pthread_mutex_lock(&lab_pool.mtx);
    pthread_cond_broadcast(&lab_pool.cv);
    pthread_mutex_unlock(&lab_pool.mtx);
  }
  lab_async_pending = 1;
  return 1;
}

void lab_parallel_wait(void) {
  if(!lab_async_pending) return;
  lab_async_pending = 0;
  lab_in_parallel = 1;
  lab_run_dynamic();
  lab_in_parallel = 0;
  lab_join();
}

#endif /* LAB_PARALLEL_IMPL */
