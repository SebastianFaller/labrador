/* Textbook (FIPS-197) AES-128/256 block encryption, test reference only
 * (NOT constant-time). */
#ifndef REF_AES_H
#define REF_AES_H
#include <stdint.h>
#include <string.h>

static uint8_t ref_sbox[256];

static uint8_t ref_xt (uint8_t x) { return (uint8_t)((x << 1) ^ ((x >> 7) * 0x1b)); }
static uint8_t ref_mul (uint8_t a, uint8_t b)
{ uint8_t r = 0; while (b) { if (b & 1) r ^= a; a = ref_xt (a); b >>= 1; } return r; }

static void ref_init_sbox (void)
{
  for (int x = 0; x < 256; x++)
    {
      uint8_t inv = 0;
      if (x)
        for (int y = 1; y < 256; y++)
          if (ref_mul ((uint8_t)x, (uint8_t)y) == 1) { inv = (uint8_t)y; break; }
      uint8_t s = inv, r = inv;
      for (int i = 0; i < 4; i++) { r = (uint8_t)((r << 1) | (r >> 7)); s ^= r; }
      ref_sbox[x] = s ^ 0x63;
    }
}

/* nk = 4 (AES-128) or 8 (AES-256); w = 4*(nr+1) words as bytes */
static void ref_expand (uint8_t w[240], const uint8_t *key, int nk)
{
  int nr = nk + 6, i;
  uint8_t rc = 1;
  memcpy (w, key, 4 * nk);
  for (i = nk; i < 4 * (nr + 1); i++)
    {
      uint8_t t[4];
      memcpy (t, w + 4 * (i - 1), 4);
      if (i % nk == 0)
        {
          uint8_t u = t[0];
          t[0] = ref_sbox[t[1]] ^ rc; t[1] = ref_sbox[t[2]]; t[2] = ref_sbox[t[3]]; t[3] = ref_sbox[u];
          rc = ref_xt (rc);
        }
      else if (nk > 6 && i % nk == 4)
        for (int j = 0; j < 4; j++) t[j] = ref_sbox[t[j]];
      for (int j = 0; j < 4; j++) w[4 * i + j] = w[4 * (i - nk) + j] ^ t[j];
    }
}

static void ref_encrypt (uint8_t out[16], const uint8_t in[16], const uint8_t *w, int nr)
{
  uint8_t s[16], t[16];
  int r, i, c;
  for (i = 0; i < 16; i++) s[i] = in[i] ^ w[i];
  for (r = 1; r <= nr; r++)
    {
      for (i = 0; i < 16; i++) s[i] = ref_sbox[s[i]];
      for (i = 0; i < 16; i++) t[i] = s[(i + 4 * (i % 4)) % 16]; /* ShiftRows */
      if (r != nr)
        for (c = 0; c < 4; c++)
          {
            uint8_t *a = t + 4 * c, b0 = a[0], b1 = a[1], b2 = a[2], b3 = a[3];
            a[0] = ref_xt (b0) ^ ref_xt (b1) ^ b1 ^ b2 ^ b3;
            a[1] = b0 ^ ref_xt (b1) ^ ref_xt (b2) ^ b2 ^ b3;
            a[2] = b0 ^ b1 ^ ref_xt (b2) ^ ref_xt (b3) ^ b3;
            a[3] = ref_xt (b0) ^ b0 ^ b1 ^ b2 ^ ref_xt (b3);
          }
      for (i = 0; i < 16; i++) s[i] = t[i] ^ w[16 * r + i];
    }
  memcpy (out, s, 16);
}
#endif
