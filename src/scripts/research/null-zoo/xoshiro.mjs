// xoshiro128** seeded by splitmix32: the generator the Null Zoo paper's robustness run used in place
// of v0's xorshift32. One 32-bit seed expands into four state words; u() is a 53-bit uniform in
// (0, 1) built from two outputs, gauss() a Box-Muller normal. Deterministic given the seed.
export function xoshiro(seed) {
  let z = seed >>> 0;
  const split = () => {
    z = (z + 0x9e3779b9) >>> 0;
    let x = z;
    x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
    x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
    return (x ^ (x >>> 16)) >>> 0;
  };
  const s = [split(), split(), split(), split()];
  const rotl = (x, k) => ((x << k) | (x >>> (32 - k))) >>> 0;
  const next = () => {
    const result = Math.imul(rotl(Math.imul(s[1], 5) >>> 0, 7), 9) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] = (s[2] ^ t) >>> 0; s[3] = rotl(s[3], 11);
    s[0] >>>= 0; s[1] >>>= 0; s[2] >>>= 0;
    return result;
  };
  const u = () => ((next() >>> 5) * 67108864 + (next() >>> 6) + 0.5) / 9007199254740992;
  let spare = null;
  const gauss = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    const r = Math.sqrt(-2 * Math.log(u()));
    const a = 2 * Math.PI * u();
    spare = r * Math.sin(a);
    return r * Math.cos(a);
  };
  return { u, gauss, next };
}
