// Copyright 2026, University of Maryland, All Rights Reserved
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <assert.h>
#include "utils.h"
#include "comm.h"   // allreduce_sum_real: the full-field dot/norm helpers
                    // below return global values under MPI (no-op at 1 rank)
#include <complex.h>

// --- LIME container ----------------------------------------------------------
// Layout follows the published LIME version 1.2 specification.
// A LIME file is a flat chain of records.  Each is a 144-byte big-endian header
//   magic 0x456789ab | u16 version | u16 flags | u64 payload length |
//   128-byte NUL-padded type string
// followed by the payload, padded up to a multiple of 8.  The gauge field is
// the record typed "ildg-binary-data"; the others hold variable-length XML
// metadata, so the payload sits at a different offset in every file and the
// chain must be walked, never assumed.  Each hop's length comes from that
// record's own header.
#define LIME_MAGIC    0x456789abUL
#define LIME_HDR      144
#define LIME_NOT_LIME (-1L)   // no magic at byte 0: a raw pre-extracted payload
#define LIME_ABSENT   (-2L)   // a LIME chain, but no record of the wanted type
#define LIME_MALFORMED (-3L)  // a LIME header or record extends beyond the file

// Byte offset of the named record's payload, with its length in *len, or one
// of the negative codes above.
static long lime_find(FILE * f, const char * want, long * len) {
  unsigned char h[LIME_HDR];
  struct stat st;
  if (fstat(fileno(f), &st) != 0 || st.st_size < 0) return LIME_MALFORMED;
  const off_t file_size = st.st_size;
  off_t off = 0;
  int  seen = 0;                          // have we accepted any header yet?
  while (off <= file_size && file_size - off >= LIME_HDR
      && fseeko(f, off, SEEK_SET) == 0
      && fread(h, 1, LIME_HDR, f) == LIME_HDR) {
    const unsigned long magic = ((unsigned long) h[0] << 24)
                              | ((unsigned long) h[1] << 16)
                              | ((unsigned long) h[2] <<  8)
                              | ((unsigned long) h[3]);
    if (magic != LIME_MAGIC) break;
    unsigned long long n = 0;             // payload length, big-endian u64
    for (int i = 0; i < 8; i ++) n = (n << 8) | h[8 + i];
    const unsigned long long pad = (8 - n % 8) % 8;
    const unsigned long long remaining =
      (unsigned long long) (file_size - off - LIME_HDR);
    if (n > (unsigned long long) LONG_MAX || n > remaining
        || pad > remaining - n)
      return LIME_MALFORMED;
    seen = 1;
    h[LIME_HDR - 1] = '\0';              // type is NUL-padded; force-terminate
    if (!strcmp((char *) h + 16, want)) {
      if (off > LONG_MAX - LIME_HDR) return LIME_MALFORMED;
      *len = (long) n;
      return (long) (off + LIME_HDR);
    }
    // The outer % 8 is load-bearing: a payload whose length is already a
    // multiple of 8 carries no padding at all, not 8 bytes of it.
    off += LIME_HDR + (off_t) n + (off_t) pad;
  }
  return seen ? LIME_ABSENT : LIME_NOT_LIME;
}

// Value of an <tag>integer</tag> element, or -1 if the tag is absent.
static int xml_int(const char * xml, const char * tag) {
  char open[32];
  snprintf(open, sizeof open, "<%s>", tag);
  const char * p = strstr(xml, open);
  return p ? atoi(p + strlen(open)) : -1;
}

// First character of a <tag>text</tag> element, or 0 if the tag is absent.
static char xml_char(const char * xml, const char * tag) {
  char open[32];
  snprintf(open, sizeof open, "<%s>", tag);
  const char * p = strstr(xml, open);
  return p ? p[strlen(open)] : '\0';
}

// Value of a <tag>hexadecimal</tag> element, or 0 if the tag is absent.  The
// SciDAC checksum words are written as bare hex, with no 0x prefix.
static uint32_t xml_hex(const char * xml, const char * tag) {
  char open[32];
  snprintf(open, sizeof open, "<%s>", tag);
  const char * p = strstr(xml, open);
  return p ? (uint32_t) strtoul(p + strlen(open), NULL, 16) : 0;
}

// Up to n whitespace-separated integers from <tag>a b c d</tag>; returns how
// many were parsed.  SciDAC writes the lattice dimensions this way, as one
// element, where ILDG uses a separate element per direction.
static int xml_ints(const char * xml, const char * tag, int * out, int n) {
  char open[32];
  snprintf(open, sizeof open, "<%s>", tag);
  const char * p = strstr(xml, open);
  if (!p) return 0;
  p += strlen(open);
  int got = 0;
  while (got < n) {
    char * end;
    const long v = strtol(p, &end, 10);
    if (end == p) break;
    out[got ++] = (int) v;
    p = end;
  }
  return got;
}

// Copy a metadata record's text into buf, NUL-terminated.  0 if the record is
// absent or too big to be metadata.
static int record_text(FILE * f, const char * type, char * buf, size_t cap) {
  long len = 0;
  const long off = lime_find(f, type, &len);
  if (off < 0 || len <= 0 || (size_t) len >= cap) return 0;
  if (fseek(f, off, SEEK_SET) != 0) return 0;
  buf[fread(buf, 1, (size_t) len, f)] = '\0';
  return 1;
}

static void check_dims(const char * path, int lx, int ly, int lz, int lt) {
  if (lx <= 0) return;
  if (lx != Lx || ly != Ly || lz != Lz_glob || lt != Lt_glob) {
    comm_die("read_gauge: %s holds a %d x %d x %d x %d lattice but this build is "
             "%d x %d x %d x %d -- pass a matching configuration, or rebuild at "
             "the geometry in global_params.h.",
             path, lx, ly, lz, lt, Lx, Ly, Lz_glob, Lt_glob);
  }
}

// Validate every self-description the file carries against this build, and
// return the payload's floating-point precision in bits.  Without this the only
// test is that the file holds at least gauge_size elements, so a configuration
// for a larger lattice reads cleanly and solves on garbage.  A file may carry
// the SciDAC records, the ILDG ones, or both; each is checked when present.
static int check_gauge_metadata(FILE * f, const char * path) {
  char xml[4096];
  int prec = 0;                           // bits; 0 until something states it

  if (record_text(f, "scidac-private-file-xml", xml, sizeof xml)) {
    // Only volfmt 0 (single file) stores the volume in the lexicographic order
    // assumed below; multifile and partition volumes use internal site order.
    const int vf = xml_int(xml, "volfmt");
    if (vf > 0) {
      comm_die("read_gauge: %s has <volfmt>%d</volfmt>; only single-file (0) "
               "volumes are stored in the site order this reader assumes.",
               path, vf);
    }
    int d[4];
    if (xml_ints(xml, "dims", d, 4) == 4)
      check_dims(path, d[0], d[1], d[2], d[3]);
  }

  if (record_text(f, "scidac-private-record-xml", xml, sizeof xml)) {
    const char p = xml_char(xml, "precision");         // 'D' or 'F'
    if      (p == 'D') prec = 64;
    else if (p == 'F') prec = 32;
    const int nc = xml_int(xml, "colors");
    const int dc = xml_int(xml, "datacount");         // link matrices per site
    const int ts = xml_int(xml, "typesize");          // bytes in one matrix
    if (nc > 0 && nc != Nc) {
      comm_die("read_gauge: %s is an Nc=%d field, this build is Nc=%d.",
               path, nc, Nc);
    }
    if (dc > 0 && dc != 4) {
      comm_die("read_gauge: %s has <datacount>%d</datacount>; a gauge field has "
               "4 link matrices per site.", path, dc);
    }
    if (ts > 0 && prec > 0 && ts != Nc * Nc * 2 * (prec / 8)) {
      comm_die("read_gauge: %s has <typesize>%d</typesize>, inconsistent with "
               "precision %c.", path, ts, p);
    }
  }

  if (record_text(f, "ildg-format", xml, sizeof xml)) {
    const int p = xml_int(xml, "precision");        // 32 or 64, as a bit count
    if (p == 32 || p == 64) prec = p;
    check_dims(path, xml_int(xml, "lx"), xml_int(xml, "ly"),
                     xml_int(xml, "lz"), xml_int(xml, "lt"));
  }

  return prec ? prec : 64;                // ILDG's default when nothing says
}

// --- SciDAC checksum --------------------------------------------------------
// The scidac-checksum record carries two 32-bit words covering the binary
// record's bytes exactly as they sit in the file -- big-endian, before any
// swapping.  Each site's bytes go through CRC-32, and the result is folded into
// both words, rotated left by that site's global lexicographic rank mod 29 and
// mod 31:
//   suma ^= rotl(crc, rank % 29)        sumb ^= rotl(crc, rank % 31)
// Two coprime rotation periods make the pair sensitive to sites being permuted,
// which one CRC over the whole payload would not be.  Both folds are xor, so
// sites may be visited in any order and each rank can accumulate its own slab
// independently; see allreduce_xor_u32.
// The algorithm is not written down in the record itself; it was determined
// by reproducing the stored words of configurations from two independent
// producers, at both precisions.
static uint32_t crc_tab[256];
static void crc32_init(void) {
  for (int i = 0; i < 256; i ++) {
    uint32_t c = (uint32_t) i;
    for (int k = 0; k < 8; k ++)
      c = (c & 1) ? (c >> 1) ^ 0xedb88320U : (c >> 1);
    crc_tab[i] = c;
  }
}

// CRC-32 with the IEEE polynomial, the one zlib computes.  This is not the
// polynomial behind the SSE4.2 crc32 instruction (that is Castagnoli), so the
// hardware instruction cannot be substituted here however tempting it looks.
static uint32_t crc32_bytes(const unsigned char * p, long n) {
  uint32_t c = 0xffffffffU;
  for (long i = 0; i < n; i ++) c = crc_tab[(c ^ p[i]) & 0xff] ^ (c >> 8);
  return c ^ 0xffffffffU;
}

// Rotate left.  n == 0 is a separate case on purpose: shifting a 32-bit value
// by 32 is undefined in C, and x86 masks the count so the naive expression
// happens to give the right answer -- it would pass every test here and break
// somewhere else.
static inline uint32_t rotl32(uint32_t v, int n) {
  return n ? ((v << n) | (v >> (32 - n))) : v;
}

// Fold nsite consecutive sites of ssz bytes each, the first of them having
// global lexicographic rank rank0, into a running checksum pair.
static void checksum_accum(const unsigned char * p, long nsite, long ssz,
                           long rank0, uint32_t * suma, uint32_t * sumb) {
  uint32_t a = 0, b = 0;
  #pragma omp parallel for reduction(^ : a, b)
  for (long s = 0; s < nsite; s ++) {
    const uint32_t c = crc32_bytes(p + s * ssz, ssz);
    const long     r = rank0 + s;
    a ^= rotl32(c, (int) (r % 29));
    b ^= rotl32(c, (int) (r % 31));
  }
  *suma ^= a;
  *sumb ^= b;
}

// The file's own checksum words.  0 if it carries no scidac-checksum record --
// an extracted payload never does, and the record is optional in a container.
static int stored_checksum(FILE * f, uint32_t * suma, uint32_t * sumb) {
  char xml[4096];
  if (!record_text(f, "scidac-checksum", xml, sizeof xml)) return 0;
  if (!strstr(xml, "<suma>") || !strstr(xml, "<sumb>")) return 0;
  *suma = xml_hex(xml, "suma");
  *sumb = xml_hex(xml, "sumb");
  return 1;
}

// read a gauge field in ILDG binary format
// dimensions of gauge will be Lx * Ly * Lz * Lt * 4 * Nc * Nc * 2
// The path may be either a LIME container or a payload already extracted from
// one; the bytes below the container header are identical, so only the starting
// offset differs and the read path is shared.
void read_gauge(REAL * U, char * path) {
  FILE * file = fopen(path, "rb");
  if (file == NULL) comm_die("read_gauge: cannot open %s", path);

  // The gauge payload is typed "ildg-binary-data" in an ILDG-compatible file
  // and "scidac-binary-data" in a plain SciDAC one.  The two records hold
  // identical bytes for a four-SU(3)-matrix-per-site field, so only the name
  // is tried twice.
  long rec_len = 0;
  int  esz = (int) sizeof(double);      // bytes per real in the file
  uint32_t file_sum[2] = { 0, 0 };      // what the scidac-checksum record says
  uint32_t data_sum[2] = { 0, 0 };      // what the bytes we read actually give
  int      have_sum = 0;                // does the file carry that record?
  long base = lime_find(file, "ildg-binary-data", &rec_len);
  // A malformed chain skips the fallback: only LIME_ABSENT means "walked it,
  // record not there", so the second name is worth trying.
  if (base == LIME_ABSENT)
    base = lime_find(file, "scidac-binary-data", &rec_len);
  if (base == LIME_MALFORMED)
    comm_die("read_gauge: %s has a malformed LIME record or length.", path);
  if (base == LIME_ABSENT) {
    comm_die("read_gauge: %s is a LIME file with no ildg-binary-data or "
             "scidac-binary-data record.", path);
  } else if (base == LIME_NOT_LIME) {
    base = 0;                      // extracted payload: field starts at byte 0
  } else {
    esz = check_gauge_metadata(file, path) / 8;
    const long want = gauge_size_glob * esz;
    if (rec_len != want) {
      comm_die("read_gauge: %s binary record holds %ld bytes, expected %ld.",
               path, rec_len, want);
    }
    // Take the stored checksum now, before the payload seeks below move the
    // file position.  A Chroma-written file puts this record *after* the binary
    // data, so reaching it means walking the whole chain -- but that is only
    // header hops, not a pass over the payload.
    have_sum = stored_checksum(file, &file_sum[0], &file_sum[1]);
    if (have_sum) crc32_init();
  }

  const long ssz = 4L * Nc * Nc * 2 * esz;  // bytes per gauge site in the file

  void * U_temp = malloc((size_t) esz * gauge_size);
#if FERM_GHOSTS
  // Decomposed build: the file holds the global config in lexicographic
  // T,Z,Y,X order (gauge is not even/odd, so it maps straight onto U's lex
  // layout).  x,y are never decomposed, so for one local t the whole local
  // plane (z in [oZ,oZ+Lz), all y,x) is a contiguous run in the file --
  // one fread per local t scatters this rank's slab.
  const long site_e     = 4L * Nc * Nc * 2;              // reals per gauge site
  const long plane_e    = (long) Lz * Ly * Lx * site_e;  // local t-plane reals
  const int  oZ = qc_grid.origin[2], oT = qc_grid.origin[3];
  for (int t = 0; t < Lt; t ++) {
    // global (tg,oZ,0,0)
    long gsite = (((long)(t + oT) * Lz_glob + oZ) * Ly) * Lx;
    // Each rank reads only its own slab, so a failure here can be one rank's
    // alone -- comm_die reports it from whichever rank saw it.
    if (fseek(file, base + gsite * site_e * esz, SEEK_SET) != 0)
      comm_die("read_gauge: cannot seek in %s: %s", path, strerror(errno));
    size_t got = fread((char *) U_temp + (long) t * plane_e * esz, esz, 
                       plane_e, file);
    if (got != (size_t) plane_e) {
      if (ferror(file))
        comm_die("read_gauge: I/O error reading %s: %s", path, strerror(errno));
      else
        comm_die("read_gauge: %s is truncated (expected %ld elements in the "
                 "t-plane at global t=%d, got %zu).", path, plane_e, t + oT,
                 got);
    }
    // One local t-plane is a contiguous run of the file, hence of global site
    // ranks, so the whole plane folds in with gsite as its first rank.
    if (have_sum)
      checksum_accum((const unsigned char *) U_temp + (long) t * plane_e * esz,
                     (long) Lz * Ly * Lx, ssz, gsite, &data_sum[0],
                     &data_sum[1]);
  }
#else
  if (fseek(file, base, SEEK_SET) != 0)
    comm_die("read_gauge: cannot seek in %s: %s", path, strerror(errno));
  size_t items_read = fread(U_temp, esz, gauge_size, file);
  if (items_read != gauge_size) {
    if (ferror(file))
      comm_die("read_gauge: I/O error reading %s: %s", path, strerror(errno));
    else
      comm_die("read_gauge: %s is truncated (expected %ld elements, got %zu).",
               path, (long) gauge_size, items_read);
  }
  if (have_sum)
    checksum_accum((const unsigned char *) U_temp, (long) Lx * Ly * Lz * Lt,
                   ssz, 0, &data_sum[0], &data_sum[1]);
#endif

  // Verify before the swap below: the checksum covers the file's own byte
  // order, and that swap runs in place, so this is the last moment the bytes
  // still look the way the checksum was computed over.
  if (have_sum) {
    allreduce_xor_u32(data_sum, 2);   // ranks cover the lattice disjointly
    if (data_sum[0] != file_sum[0] || data_sum[1] != file_sum[1]) {
      comm_die("read_gauge: %s fails its scidac-checksum -- the record says "
               "%08x/%08x, the bytes give %08x/%08x.  The configuration is "
               "corrupt or truncated.",
               path, file_sum[0], file_sum[1], data_sum[0], data_sum[1]);
    }
  }
  // ILDG and SciDAC both write big-endian; Intel processors are little-endian,
  // so reverse each element before widening it into this build's working
  // precision.  1L * forces the count to long: the element count overflows int
  // at production volumes, and gauge_size expands to a bare product with no
  // cast of its own.
  const long ng = 1L * gauge_size;
  if (esz == (int) sizeof(double)) {
    u_int64_t * U_int = (u_int64_t *) U_temp;
    #pragma omp parallel for
    for (long i = 0; i < ng; i ++) U_int[i] = __bswap_64(U_int[i]);
    const double * src = (const double *) U_temp;
    #pragma omp parallel for
    for (long i = 0; i < ng; i ++) U[i] = src[i];
  } else {
    u_int32_t * U_int = (u_int32_t *) U_temp;
    #pragma omp parallel for
    for (long i = 0; i < ng; i ++) U_int[i] = __bswap_32(U_int[i]);
    const float * src = (const float *) U_temp;
    #pragma omp parallel for
    for (long i = 0; i < ng; i ++) U[i] = src[i];
  }
  free(U_temp);
  fclose(file);
}

// erase 3x3 complex matrix
static void zero_matrix(REAL * A) {
  for (int i = 0; i < Nc * Nc * 2; i ++) A[i] = 0;
}

// Index of a gauge site in the (possibly z/t-extended) array Ug used by
// compute_Fmunu.  x,y are never decomposed.  With ghosts, interior local (t,z)
// sits at (t+DECOMP_T, z+DECOMP_Z) of an array padded by one site in each
// decomposed dim; passing z=-1/Lz or t=-1/Lt lands cleanly in the ghost layer.
static inline long fmunu_gidx(int t, int z, int y, int x) {
#if FERM_GHOSTS
  return (long) (t + DECOMP_T) * ((long) Lz + 2 * DECOMP_Z) * Ly * Lx
       + (long) (z + DECOMP_Z) * Ly * Lx + (long) y * Lx + x;
#else
  // Ug is an alias of U at one rank, so this is a gauge-array index
  return site_full(x, y, z, t);
#endif
}
// One step (sign s = +1/-1) in direction dim (0=x,1=y,2=z,3=t) from local
// (t,z,y,x).  x,y always wrap locally; a decomposed z/t indexes the ghost
// layer (no wrap), an undecomposed one wraps locally -- matching D's stencil
// convention.
static inline long fmunu_nbr(int t, int z, int y, int x, int dim, int s) {
  if (dim == 0)      x = (x + s + Lx) % Lx;
  else if (dim == 1) y = (y + s + Ly) % Ly;
  else if (dim == 2) z = DECOMP_Z ? z + s : (z + s + Lz) % Lz;
  else               t = DECOMP_T ? t + s : (t + s + Lt) % Lt;
  return fmunu_gidx(t, z, y, x);
}

// this is not optimized since it's only called once
// Dimensions of U: Lt * Lz * Ly * Lx * 4 * Nc * Nc * 2
// Dimensions of Fmunu: Lt * Lz * Ly * Lx * 2 (block) * 6 * 6 * 2 (complex)
void compute_Fmunu(REAL * U, REAL * Fmunu) {
  // The clover leaves step one site into the rank-face neighbors (and the
  // (z,t) corner), so under decomposition read links from a gauge array
  // extended by one ghost layer in each decomposed dim.  Ug == U /
  // fmunu_gidx == loc at 1 rank.
  REAL * Ug = U;
#if FERM_GHOSTS
  const long lsz = (long) 4 * Nc * Nc * 2;            // REALs per gauge site
  const long Lzx = (long) Lz + 2 * DECOMP_Z, Ltx = (long) Lt + 2 * DECOMP_T;
  Ug = malloc(sizeof(REAL) * Ltx * Lzx * Ly * Lx * lsz);
  #pragma omp parallel for collapse(2)
  for (int t = 0; t < Lt; t ++)
    for (int z = 0; z < Lz; z ++)
      for (int y = 0; y < Ly; y ++)
        for (int x = 0; x < Lx; x ++)
          memcpy(Ug + fmunu_gidx(t, z, y, x) * lsz,
                 U + site_full(x, y, z, t) * lsz, sizeof(REAL) * lsz);
  halo_gauge_ext(Ug, sizeof(REAL) * lsz);
#endif
  #pragma omp parallel for
  for (int t = 0; t < Lt; t ++) {
    for (int z = 0; z < Lz; z ++) {
      for (int y = 0; y < Ly; y ++) {
        for (int x = 0; x < Lx; x ++) {
          long loc = site_full(x, y, z, t);                     // local output
          long gc = fmunu_gidx(t, z, y, x);                     // gauge center
          complex REAL * Fmunu_block = calloc(2 * 6 * 6, sizeof(complex REAL));
          for (int mu = 0; mu < 3; mu ++) {
            for (int nu = mu + 1; nu < 4; nu ++) {
              long loc_mu_plus  = fmunu_nbr(t, z, y, x, mu, +1);
              long loc_mu_minus = fmunu_nbr(t, z, y, x, mu, -1);
              long loc_nu_plus  = fmunu_nbr(t, z, y, x, nu, +1);
              long loc_nu_minus = fmunu_nbr(t, z, y, x, nu, -1);
              long loc_munu;
              int munu = nu; // specifies which of the 6 (mu, nu) pairings
              if (mu == 0)      munu -= 1;
              else if (mu == 1) munu += 1;
              else /* mu == 2 */ munu += 2;

              REAL * temp1 = calloc(Nc * Nc * 2, sizeof(REAL));
              REAL * temp2 = calloc(Nc * Nc * 2, sizeof(REAL));
              REAL * temp3 = calloc(Nc * Nc * 2, sizeof(REAL));
              REAL * temp4 = calloc(Nc * Nc * 2, sizeof(REAL));
              REAL * Udag = calloc(Nc * Nc * 2, sizeof(REAL));

              // mu > 0, nu > 0 plaquette
              matmul(Ug + (gc * 4 + mu) * Nc * Nc * 2,
                     Ug + (loc_mu_plus * 4 + nu) * Nc * Nc * 2, temp1, 0);
              matmul(temp1, Ug + (loc_nu_plus * 4 + mu) * Nc * Nc * 2,
                     temp2, 1);
              matmul(temp2, Ug + (gc * 4 + nu) * Nc * Nc * 2, temp3, 1);

              // mu > 0, nu < 0 plaquette
              zero_matrix(temp1);
              zero_matrix(temp2);
              loc_munu = loc_mu_plus + loc_nu_minus - gc;
              // initialize Udag to U_nu_minus, nu direction
              for (int a = 0; a < Nc; a ++) {
                for (int b = 0; b < Nc; b ++) {
                  Udag[((a * Nc) + b) * 2]
                    = Ug[(loc_nu_minus * 4 + nu) * Nc * Nc * 2 + ((b * Nc)
                        + a) * 2];
                  Udag[((a * Nc) + b) * 2 + 1]
                    = -Ug[(loc_nu_minus * 4 + nu) * Nc * Nc * 2 + ((b * Nc)
                         + a) * 2 + 1];
                }
              }
              matmul(Udag, Ug + (loc_nu_minus * 4 + mu) * Nc * Nc * 2,
                     temp1, 0);
              matmul(temp1, Ug + (loc_munu * 4 + nu) * Nc * Nc * 2, temp2, 0);
              matmul(temp2, Ug + (gc * 4 + mu) * Nc * Nc * 2, temp3, 1);

              // mu < 0, nu < 0 plaquette
              zero_matrix(temp1);
              zero_matrix(temp2);
              loc_munu = loc_mu_minus + loc_nu_minus - gc;
              // initialize Udag to U_mu_mimus, mu direction
              for (int a = 0; a < Nc; a ++) {
                for (int b = 0; b < Nc; b ++) {
                  Udag[((a * Nc) + b) * 2]
                    = Ug[(loc_mu_minus * 4 + mu) * Nc * Nc * 2 + ((b * Nc)
                        + a) * 2];
                  Udag[((a * Nc) + b) * 2 + 1]
                    = -Ug[(loc_mu_minus * 4 + mu) * Nc * Nc * 2 + ((b * Nc)
                         + a) * 2 + 1];
                }
              }
              matmul(Udag, Ug + (loc_munu * 4 + nu) * Nc * Nc * 2, temp1, 1);
              matmul(temp1, Ug + (loc_munu * 4 + mu) * Nc * Nc * 2, temp2, 0);
              matmul(temp2, Ug + (loc_nu_minus * 4 + nu) * Nc * Nc * 2,
                     temp3, 0);

              // mu < 0, nu > 0 plaquette
              zero_matrix(temp1);
              zero_matrix(temp2);
              loc_munu = loc_mu_minus + loc_nu_plus - gc;
              matmul(Ug + (gc * 4 + nu) * Nc * Nc * 2,
                     Ug + (loc_munu * 4 + mu) * Nc * Nc * 2, temp1, 1);
              matmul(temp1, Ug + (loc_mu_minus * 4 + nu) * Nc * Nc * 2,
                     temp2, 1);
              matmul(temp2, Ug + (loc_mu_minus * 4 + mu) * Nc * Nc * 2,
                     temp3, 0);


              // compute temp4 = temp3 - temp3:adjoint()
              for (int a = 0; a < Nc; a ++) {
                for (int b = 0; b < Nc; b ++) {
                  temp4[((a * Nc) + b) * 2] 
                    = temp3[((a * Nc) + b) * 2] - temp3[((b * Nc) + a) * 2];
                  temp4[((a * Nc) + b) * 2 + 1] 
                    = temp3[((a * Nc) + b) * 2 + 1] 
                     + temp3[((b * Nc) + a) * 2 + 1];
                }
              }
              // multiplicative factors for each munu combination
              complex REAL factors [6][4]
                = {{-1j,  1j, -1j,  1j},   // mu = 0, nu = 1
                   { -1,   1,  -1,   1},   // mu = 0, nu = 2
                   { 1j,  1j, -1j, -1j},   // mu = 0, nu = 3
                   {-1j, -1j, -1j, -1j},   // mu = 1, nu = 2
                   { -1,   1,   1,  -1},   // mu = 1, nu = 3
                   { 1j, -1j, -1j,  1j}};  // mu = 2, nu = 3
              // column of F matrix to insert block into
              // this is either 0 or 1 (left/right 3x3 block within 6x6 block)
              int columns [6][4]
                = {{0, 1, 0, 1},
                   {1, 0, 1, 0},
                   {1, 0, 1, 0},
                   {1, 0, 1, 0},
                   {1, 0, 1, 0},
                   {0, 1, 0, 1}};
              for (int a = 0; a < Nc; a ++) {
                for (int b = 0; b < Nc; b ++) {
                  complex REAL entry = ((complex REAL *) temp4)[a * Nc + b];
                  for (int i = 0; i < 4; i ++) {
                    // location to place entry in matrix
                    int block = i / 2;
                    int row = i % 2;
                    int index = (block * 6 + 3 * row + a) * 6 
                                + 3 * columns[munu][i] + b;
                    Fmunu_block[index] += factors[munu][i] * entry;
                  }
                }
              }
              free(temp1);
              free(temp2);
              free(temp3);
              free(temp4);
              free(Udag);
            }
          }
          // copy Fmunu_block into Fmunu
          // put real entries in upper triangle and imaginaries in lower
          for (int block = 0; block < 2; block ++) {
            for (int a = 0; a < 6; a ++) {
              for (int b = 0; b <= a; b ++) {
                // do imaginary first so it gets overridden on diagonal
                Fmunu[((loc * 2 + block) * 6 + a) * 6 + b]
                  = cimag(Fmunu_block[(block * 6 + b) * 6 + a]);
                Fmunu[((loc * 2 + block) * 6 + b) * 6 + a]
                  = creal(Fmunu_block[(block * 6 + b) * 6 + a]);
              }
            }
          }
          free(Fmunu_block);
        }
      }
    }
  }
#if FERM_GHOSTS
  free(Ug);
#endif
}

// In-place-safe inverse of one dense N x N interleaved-complex matrix, computed
// in double precision (Gauss-Jordan with partial pivoting) for robustness, then
// stored back as REAL.  src is read fully into a scratch buffer before dst is
// written, so src == dst is safe.  Works for any invertible matrix (not just
// Hermitian positive-definite) and needs no external linear-algebra library.
void invert_dense_block(const REAL * src, REAL * dst, int N) {
  double complex * A = malloc((size_t) N * N * sizeof(double complex));
  double complex * B = malloc((size_t) N * N * sizeof(double complex));
  for (int i = 0; i < N * N; i ++)
    A[i] = (double) src[2 * i] + I * (double) src[2 * i + 1];
  for (int r = 0; r < N; r ++)
    for (int c = 0; c < N; c ++)
      B[r * N + c] = (r == c) ? 1.0 : 0.0;
  for (int col = 0; col < N; col ++) {
    int piv = col; double best = cabs(A[col * N + col]);
    for (int r = col + 1; r < N; r ++) {
      double v = cabs(A[r * N + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (piv != col)
      for (int c = 0; c < N; c ++) {
        double complex t = A[col * N + c];
        A[col * N + c] = A[piv * N + c];
        A[piv * N + c] = t;
        t = B[col * N + c];
        B[col * N + c] = B[piv * N + c];
        B[piv * N + c] = t;
      }
    double complex inv = 1.0 / A[col * N + col];
    for (int c = 0; c < N; c ++) {
      A[col * N + c] *= inv;
      B[col * N + c] *= inv;
    }
    for (int r = 0; r < N; r ++) {
      if (r == col) continue;
      double complex f = A[r * N + col];
      if (f == 0.0) continue;
      for (int c = 0; c < N; c ++) {
        A[r * N + c] -= f * A[col * N + c];
        B[r * N + c] -= f * B[col * N + c];
      }
    }
  }
  for (int i = 0; i < N * N; i ++) {
    dst[2 * i]     = (REAL) creal(B[i]);
    dst[2 * i + 1] = (REAL) cimag(B[i]);
  }
  free(A); free(B);
}

// 3x3 complex matrix multiplication (A x B = C)
// dag specifies if second entry should be conjugate transposed
void matmul (REAL * A, REAL * B, REAL * C, int dag) {
  if (dag) {
    // C[i][k] += A[i][j] * conj(B[k][j])
    for (int i = 0; i < Nc; i ++) {
      for (int j = 0; j < Nc; j ++) {
        for (int k = 0; k < Nc; k ++) {
          C[2*(Nc*i + k)] += A[2*(Nc*i + j)] * B[2*(Nc*k + j)] 
                            + A[2*(Nc*i + j) + 1] * B[2*(Nc*k + j) + 1];
          C[2*(Nc*i + k) + 1] += A[2*(Nc*i + j) + 1] * B[2*(Nc*k + j)] 
                                - A[2*(Nc*i + j)] * B[2*(Nc*k + j) + 1];
        }
      }
    }
  } else {
    // C[i][k] += A[i][j] * B[j][k]
    for (int i = 0; i < Nc; i ++) {
      for (int j = 0; j < Nc; j ++) {
        for (int k = 0; k < Nc; k ++) {
          C[2*(Nc*i + k)] += A[2*(Nc*i + j)] * B[2*(Nc*j + k)] 
                            - A[2*(Nc*i + j) + 1] * B[2*(Nc*j + k) + 1];
          C[2*(Nc*i + k) + 1] += A[2*(Nc*i + j) + 1] * B[2*(Nc*j + k)] 
                                + A[2*(Nc*i + j)] * B[2*(Nc*j + k) + 1];
        }
      }
    }
  }
}

// copy a gauge field
void copy_gauge(REAL * output, REAL * input) {
  #pragma omp parallel for
  for (int i = 0; i < gauge_size; i ++)
    output[i] = input[i];
}


// multi-RHS versions of above functions
void rescale_multi_ferm(REAL * ferm, REAL c, long num_ferm) {
  #pragma omp parallel for
  for (long i = 0; i < ferm_size * num_ferm; i ++)
    ferm[i] *= c;
}

void copy_multi_ferm(REAL * output, REAL * input, long num_ferm) {
  #pragma omp parallel for
  for (long i = 0; i < ferm_size * num_ferm; i ++)
    output[i] = input[i];
}

// multi-RHS, half-fermion versions of above functions
REAL multi_half_ferm_norm(REAL * ferm, long num_ferm) {
  REAL norm = 0;
  #pragma omp parallel for reduction(+:norm)
  for (long i = 0; i < half_ferm_size * num_ferm; i ++)
    norm += ferm[i] * ferm[i];
  allreduce_sum_real(&norm, 1);
  return norm;
}

// Per-rhs squared norms: norms[k] = ||f_k||^2, allreduced.  The rhs index is
// the fastest one -- num_rhs reals then num_rhs imags per complex element --
// so a group of num_rhs consecutive reals holds one value from each rhs and
// the accumulator is exactly one vector register wide, with no horizontal
// fold at the end.  Same traffic and flop count as multi_half_ferm_norm above;
// it keeps the lanes apart instead of summing them.
void multi_half_ferm_norm_rhs(REAL * ferm, long num_ferm, REAL * norms) {
  for (int k = 0; k < num_rhs; k ++) norms[k] = 0;
  const long ngroup = half_ferm_size * num_ferm / num_rhs;
  #pragma omp parallel for reduction(+:norms[:num_rhs])
  for (long g = 0; g < ngroup; g ++)
    for (int k = 0; k < num_rhs; k ++)
      norms[k] += ferm[g * num_rhs + k] * ferm[g * num_rhs + k];
  allreduce_sum_real(norms, num_rhs);
}

void zero_multi_half_ferm(REAL * output, long num_ferm) {
  #pragma omp parallel for
  for (long i = 0; i < half_ferm_size * num_ferm; i ++)
    output[i] = 0;
}

void copy_multi_half_ferm(REAL * output, REAL * input, long num_ferm) {
  #pragma omp parallel for
  for (long i = 0; i < half_ferm_size * num_ferm; i ++)
    output[i] = input[i];
}

complex REAL cdot_ferm(REAL * a, REAL * b, long length) {
  REAL sum_re = 0;
  REAL sum_im = 0;
  #pragma omp parallel for reduction(+:sum_re, sum_im)
  for (long i = 0; i < length / (2 * num_rhs); i ++) {
    REAL * shifted_a = a + 2 * i * num_rhs;
    REAL * shifted_b = b + 2 * i * num_rhs;
    for (int j = 0; j < num_rhs; j ++)
      DOTC_KERNEL(sum_re, sum_im, shifted_a, shifted_b, j);
  }
  REAL sums[2] = { sum_re, sum_im };
  allreduce_sum_real(sums, 2);
  return (sums[0] + 1j * sums[1]);
}

void caxpy_ferm(REAL * output, REAL * input, complex REAL a, long length) {
  REAL a_re = creal(a);
  REAL a_im = cimag(a);
  #pragma omp parallel for
  for (long i = 0; i < length / (2 * num_rhs); i ++) {
    REAL * shifted_input  = input  + 2 * i * num_rhs;
    REAL * shifted_output = output + 2 * i * num_rhs;
    for (int j = 0; j < num_rhs; j ++)
      CAXPY_KERNEL(shifted_output, shifted_input, a_re, a_im, j);
  }
}
