// Seeds of the confirmation run's JavaScript part, and a check that none equals a top-level seed of any earlier
// xoshiro run in this repository: v0 (src/scripts/research/null-zoo/run.mjs, from 1000), the extended run
// (analysis/extend.mjs, from 20260929), the bootstrap-resolution run (from 20261000, plus 1000003 * i per draw count),
// v1 and v1b (base 20261004 / 20261005 + cell * 1000003 + chunk * 7919), the autocorrelation levels (20261105 +
// i * 1000003) and the snooping-fix check (20261107 + design * 1000003 + chunk * 7919). The ranges below are generous
// supersets of what those runs drew.
export const confirmSeed = (base, exp, cell, chunk) => (base + exp * 100000007 + cell * 1000033 + chunk * 7927) >>> 0;

export function earlierSeeds() {
  const s = new Set();
  for (let i = 0; i < 200; i++) { s.add(1000 + i); s.add(20260929 + i); }
  for (let i = 0; i < 200; i++) for (let d = 0; d <= 20; d++) s.add((20261000 + i + 1000003 * d) >>> 0);
  for (const b of [20261004, 20261005]) for (let c = 0; c < 64; c++) for (let k = 0; k < 400; k++) s.add((b + c * 1000003 + k * 7919) >>> 0);
  for (let i = 0; i < 10; i++) s.add((20261105 + i * 1000003) >>> 0);
  for (let d = 0; d < 20; d++) for (let k = 0; k < 400; k++) s.add((20261107 + d * 1000003 + k * 7919) >>> 0);
  return s;
}

export function assertDisjoint(design, experiments) {
  const old = earlierSeeds();
  const mine = new Set();
  for (const [exp, cells] of Object.entries(experiments)) {
    const rounds = Number(exp) === 1 ? design.joint.rounds : design.single.rounds;
    cells.forEach((_, cell) => { for (let c = 0; c < rounds; c++) {
      const seed = confirmSeed(design.baseSeed, Number(exp), cell, c);
      if (old.has(seed)) throw new Error(`seed ${seed} was used by an earlier run`);
      if (mine.has(seed)) throw new Error(`seed ${seed} repeats within the confirmation run`);
      mine.add(seed);
    } });
  }
  return mine.size;
}
