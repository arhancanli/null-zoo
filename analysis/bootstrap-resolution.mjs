// Paired check of the bootstrap's resolution: the same searches are scored by the bootstrap with
// 400 and with 1,999 resamples, and by the Sidak t test for reference. The searches come from one
// generator stream and the resampling from separate ones, so the only difference between the two
// bootstrap columns is the number of resamples.
//   node analysis/bootstrap-resolution.mjs [reps] > analysis/bootstrap-resolution.json
import { drawSearch } from "../src/scripts/research/null-zoo/families.mjs";
import { DESIGN } from "../src/scripts/research/null-zoo/run.mjs";
import { VALIDATORS, summarize } from "../src/scripts/research/null-zoo/validators.mjs";
import { xoshiro } from "./extend.mjs";

const FAMILIES = ["iid_normal", "student_t4", "skew_negative"];
const DRAWS = [400, 1999];

export function pairedCell(family, { reps, skill, seed }) {
  const data = xoshiro(seed);
  const resample = Object.fromEntries(DRAWS.map((b, i) => [b, xoshiro(seed + 1000003 * (i + 1))]));
  const reject = { luck_trials: 0, ...Object.fromEntries(DRAWS.map((b) => [`bootstrap_${b}`, 0])) };
  for (let rep = 0; rep < reps; rep++) {
    const search = drawSearch(family, { ...DESIGN, skill }, data);
    const s = summarize(search, DESIGN.periodsPerYear);
    if (VALIDATORS.luck_trials.p(search, s) <= DESIGN.level) reject.luck_trials++;
    for (const b of DRAWS) if (VALIDATORS.bootstrap_best.p(search, s, resample[b], b) <= DESIGN.level) reject[`bootstrap_${b}`]++;
  }
  return reject;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const reps = Number(process.argv[2] ?? 10000);
  const rows = [];
  let seed = 20261000; // disjoint from the seeds of the other runs
  for (const family of FAMILIES) {
    rows.push({ family, seeds: [seed, seed + 1], size: pairedCell(family, { reps, skill: 0, seed: seed++ }), power: pairedCell(family, { reps, skill: DESIGN.skill, seed: seed++ }) });
    process.stderr.write(`${family} `);
  }
  process.stderr.write("\n");
  console.log(JSON.stringify({ schema: "canli.null-zoo.bootstrap-resolution.v1", design: DESIGN, reps, draws: DRAWS, rows }, null, 1));
}
