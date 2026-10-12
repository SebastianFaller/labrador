# labrados on aarch64 (Raspberry Pi 4)

labrados is written in AVX-512 intrinsics. On aarch64 it is compiled through
[SIMDe](https://github.com/simd-everywhere/simde) (`simd.h`), which emulates
each 512-bit intrinsic in generic C or NEON. That works, but on a Raspberry
Pi 4 (Cortex-A72, no AES instructions) the Labrador proof in device-binding
took 45–66x longer than the x86-64 numbers in the paper. This branch adds native aarch64 code for the hot spots and a small
thread pool.

**Proofs are bit-identical.** Every change computes exactly the same bytes as
the original code, for all inputs, so proofs made on aarch64 verify on x86-64
and the other way round.

**x86-64 is unchanged.** All aarch64 code sits behind two macros that are
never defined on x86-64, and the original code is kept as it was in the
`#else` branches. An x86-64 build compiles the same code as upstream.

| macro | defined in | when | covers |
|---|---|---|---|
| `LABRADOS_NEON` | `simd_neon.h` | aarch64 with NEON, unless `-DLABRADOS_FORCE_AVX512` | NEON kernels, NEON NTT, JL table lookup, NEON AES |
| `LAB_PARALLEL` | `parallel.h` | aarch64, unless `-DLAB_NO_THREADS` | multi-threading |

## What changed

The speedups are single-call microbenchmarks on a Pi 4 at LOGQ=38. "old" is
the original AVX-512 code through SIMDe.

| change | file(s) | old → new | how it stays exact |
|---|---|---|---|
| JL collapse as a table lookup | `jlproj.c` (`jlproj_collapsmat32`) | 17.05 → 2.01 ms (8.5x) | `_mm512_permutex2var_epi64` picks one of 16 precomputed sums per 4-bit nibble, so the code indexes a 16-entry table directly. Wrapping `uint64` adds equal `_mm512_add_epi64`. |
| NEON bitsliced AES-CTR | `aesctr.[ch]`, `../aes-neonbs.h` | AES-128: 28.6 → 12.1 cycles/byte (2.4x) | The same Boyar–Peralta S-box circuit, 8 blocks per pass, bitsliced round keys cached in the context. Checked against FIPS-197 and the scalar bitsliced AES. |
| NEON NTT | `ntt.c` | `poly_ntt` 2.8 → 0.95 µs (2.9x), `poly_invntt` 3.2x | The same int16 operation sequence for each coefficient; only the data movement differs. The comments in `ntt.c` prove that the NEON Montgomery sequence gives the same integer, not just a congruent one. |
| NEON kernels | `poly.c`, `polz.c`, `jlproj.c`, `simd_neon.h` | `jlproj_add64` 6.2x, `polyvec_sprod_pointwise` 4.2x, `polyvec_sprod_extension` 3.5–7x, `poly_caddp` 7x, `poly_reduce` 2.1x, `polz_center` 2.2x, `zz_poly_fma` 1.5x, ... | Each kernel wraps exactly like the AVX-512 instruction it replaces. |
| Multi-threading | `parallel.h`, `polx.c`, `polz.c`, `jlproj.c`, `constraints.c`, `proofsystem.c`, `gaussian.c`, `aesctr.c` | Labrador prove and verify about 1.4x on 4 cores (median) | See below. |

Together with the lazer changes on the same branch, these make
device-binding's precomputation (dominated by this proof) 3.3x faster and
the proof's verification 4.6x faster on the Pi (Falcon-512 version of the
protocol; see lazer's `docs/RPI4.md`).

### Multi-threading

`parallel.h` provides a persistent pthread pool, created on first use.
`LAB_THREADS` sets the number of threads (default 4; `LAB_THREADS=1` starts no
threads). These loops are split up:
- NTT/INTT and element-wise vector loops, by element range;
- inner products, by CRT prime (never along a summation, so the order of
  additions is unchanged);
- the JL collapse, in blocks of 16 output polynomials;
- the JL projection, with per-thread partial `int32` sums, which wrap mod
  2^32 and therefore add up to exactly the serial result;
- AES-CTR expansion, by counter range (the same byte stream);
- the Gaussian sampler's keystream, generated ahead of time while the sampler
  itself stays serial;
- constraint aggregation, in independent column blocks.

These stay serial:
- the order in which randomness is consumed;
- transcript hashing;
- the floating-point width bookkeeping.

The proofs are therefore identical for every thread count. A function that
has a parallel version keeps its serial body as the static helper `f_ser`
(`LAB_SERIAL(f)` in `parallel.h`); without `LAB_PARALLEL` that body is the
public function `f` again.

### Side channels

None of the changes adds a secret-dependent branch or memory access:
- table lookups and the multi-threaded work split are indexed only by public
  data (the JL matrix, the challenge, vector lengths);
- `jlproj_add64` multiplies the *secret* witness using register lookups
  (`TBL`), which are indexed only by public matrix bits;
- the AES is bitsliced (no S-box tables);
- NEON integer instructions on the Cortex-A72 have data-independent timing.

## Equivalence tests

`tests/aarch64` builds each test twice:
- `ref`, with `-DLABRADOS_FORCE_AVX512 -DLAB_NO_THREADS`, which is the
  original code;
- `opt`, which is the aarch64 code.

The outputs must be byte-identical:

| test | what it compares |
|---|---|
| `equiv_poly` | the NEON kernels of `poly.c`, edge-biased random int16 inputs |
| `equiv_polz` | the NEON kernels of `polz.c` |
| `equiv_jlproj` | JL projection, collapse and the raw table-lookup kernel |
| `equiv_ntt` | `poly_ntt`, `poly_invntt`, `poly_nttunpack` over the full int16 range |
| `equiv_aesctr` | AES-128/256-CTR streams of 0–600 blocks (including the threaded sizes); the first blocks against textbook FIPS-197 AES |
| `equiv_prover` | 8 deterministic Labrador proofs (prove, reduce, verify); `opt` runs with `LAB_THREADS=1` and `4` |

Run on aarch64 after lazer's top-level `make` has unpacked SIMDe:

```sh
cd lazer/src/labrados/tests/aarch64
make check          # LOGQ 32, 36 and 38; ends with "ALL OK"
make bench          # kernel timings, ref vs opt
```
