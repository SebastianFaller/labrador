# Equivalence tests for the aarch64 code paths of labrados (README-aarch64.md).
#
# Every test is built twice from the same sources:
#   ref: -DLABRADOS_FORCE_AVX512 -DLAB_NO_THREADS, i.e. the original AVX-512
#        code (through SIMDe) and serial loops, as without the aarch64 changes;
#   opt: the aarch64 code (NEON kernels, NEON AES, JL table lookup, threads).
# `make check` runs both and requires byte-identical outputs; the prover test
# also runs opt with LAB_THREADS=1 and 4. Outputs (up to ~430 MB per LOGQ) are
# deleted once they compare equal and kept in build/ if they differ. `make bench` prints kernel timings.
#
# Run on aarch64, from lazer/src/labrados/tests/aarch64 (SIMDe is unpacked by
# lazer's top-level make), or set SIMDE_DIR.

CC ?= cc
SIMDE_DIR ?= ../../../../third_party/simde-d4d85e3
LOGQS ?= 32 36 38
MARCH ?= -march=native -mtune=native

CFLAGS = -std=gnu2x -O3 $(MARCH) -fwrapv -ffast-math -DNDEBUG -pthread \
  -Wall -Wextra -Wno-unused-function -Wno-unused-parameter \
  -I. -I../.. -I$(SIMDE_DIR) -DSIMDE_ENABLE_NATIVE_ALIASES
REF = -DLABRADOS_FORCE_AVX512 -DLAB_NO_THREADS
LIBS = -lm -pthread

LIB = aesctr.c comkey.c constraints.c data.c fips202.c gaussian.c jlproj.c \
  ntt.c polx.c poly.c polz.c proofsystem.c randombytes.c
PROVER = labrador.c labrador_core.c timing.c test_proofsystem_setup.c \
  test_constraints_setup.c
KERNELS = poly polz jlproj ntt aesctr

# sources linked into test T (the one it includes, and randombytes.c for the
# prover test, which brings its own deterministic randombytes)
src = $(addprefix ../../,$(filter-out $(1).c randombytes.c,$(LIB)) \
  $(if $(filter prover,$(1)),$(PROVER),randombytes.c))

.PHONY: all check bench clean
all: $(foreach q,$(LOGQS),$(foreach t,$(KERNELS) prover,build/$(t)_$(q)_ref build/$(t)_$(q)_opt))

build:
	mkdir -p build

.SECONDEXPANSION:
build/%_ref: equiv_$$(word 1,$$(subst _, ,$$*)).c kt.h $$(wildcard ../../*.[ch]) | build
	$(CC) $(CFLAGS) $(CPPFLAGS) $(REF) -DLOGQ=$(word 2,$(subst _, ,$*)) -o $@ $< \
	  $(call src,$(word 1,$(subst _, ,$*))) $(LIBS)

build/%_opt: equiv_$$(word 1,$$(subst _, ,$$*)).c kt.h $$(wildcard ../../*.[ch]) | build
	$(CC) $(CFLAGS) $(CPPFLAGS) -DLOGQ=$(word 2,$(subst _, ,$*)) -o $@ $< \
	  $(call src,$(word 1,$(subst _, ,$*))) $(LIBS)

check: all
	@set -e; for q in $(LOGQS); do \
	  for t in $(KERNELS); do \
	    build/$${t}_$${q}_ref build/$${t}_$${q}.ref; \
	    build/$${t}_$${q}_opt build/$${t}_$${q}.opt; \
	    cmp build/$${t}_$${q}.ref build/$${t}_$${q}.opt; \
	    echo "OK  $$t LOGQ=$$q ($$(wc -c < build/$${t}_$${q}.ref | tr -d " ") bytes identical)"; \
	    rm -f build/$${t}_$${q}.ref build/$${t}_$${q}.opt; \
	  done; \
	  build/prover_$${q}_ref build/prover_$${q}.ref; \
	  for n in 1 4; do \
	    LAB_THREADS=$$n build/prover_$${q}_opt build/prover_$${q}.opt$$n; \
	    cmp build/prover_$${q}.ref build/prover_$${q}.opt$$n; \
	  done; \
	  echo "OK  prover LOGQ=$$q ($$(wc -l < build/prover_$${q}.ref | tr -d ' ') proofs identical, LAB_THREADS=1,4)"; \
	  rm -f build/prover_$${q}.ref build/prover_$${q}.opt1 build/prover_$${q}.opt4; \
	done; echo "ALL OK"

bench: all
	@for t in $(KERNELS); do for v in ref opt; do \
	  echo "== $$t ($$v, LOGQ=38)"; build/$${t}_38_$$v /dev/null bench; done; done

clean:
	rm -rf build
