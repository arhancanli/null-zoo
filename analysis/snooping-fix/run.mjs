// Unregistered check (2026-11-07): the author's data-snooping implementation before and after its fix.
//
// v1 and v1b scored Hansen's SPA and the stepwise test with js/snooping-core.js as copied into src/ (canlicapital
// commit ea398598): every resample divided by the Politis-Romano standard deviation of the sample, held fixed, and
// p = #(T* > T) / B. canlicapital PR 562 (commit 4c7e429e, merged 3 November 2026) replaced both: each resample is
// divided by its own standard deviation (a bootstrap-t, as in Romano and Wolf 2005), ties are counted, and StepM's
// critical value is floored at zero. The fixed core is copied unchanged into analysis/snooping-fix/js/.
//
// This script runs both versions on the same new skill-less searches and the same bootstrap seed per search: the
// nine v1 families and Experiment C's poor-alternative design (one strategy with mean zero, nineteen with a true
// annualized Sharpe ratio of -3). 20 strategies, 504 days, 1,000 resamples, mean block length 8, as in v1.
//   node analysis/snooping-fix/run.mjs [searches per design] [workers] > analysis/snooping-fix/snooping-fix.json
// Chunks of searches are seeded separately and merged in order, so the output does not depend on the workers.
import { Worker, isMainThread, parentPort, workerData } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { snoopingTests as oldTests } from "../../src/js/snooping-core.js";
import { snoopingTests as newTests } from "./js/snooping-core.js";
import { xoshiro } from "../../src/scripts/research/null-zoo/xoshiro.mjs";
import { FAMILIES_V1, drawSearchV1 } from "../../src/scripts/research/null-zoo/v1/families.mjs";

export const DESIGN = Object.freeze({
  trials: 20, observations: 504, periodsPerYear: 252, reps: 1000, chunk: 100, baseSeed: 20261107,
  levels: Object.freeze([0.01, 0.05, 0.1]), poorSharpe: -3,
  designs: Object.freeze([...FAMILIES_V1, "poor_alternatives"]),
});
const TESTS = ["spa_consistent", "spa_upper", "stepm_any", "reality_check"];
const VERSIONS = { old: oldTests, new: newTests };
const seedFrom = (r) => (r.next() >>> 1) || 1;
const chunkSeed = (d, c) => (DESIGN.baseSeed + d * 1000003 + c * 7919) >>> 0;

function draw(design, r) {
  const base = { trials: DESIGN.trials, observations: DESIGN.observations, periodsPerYear: DESIGN.periodsPerYear, skill: 0 };
  if (design !== "poor_alternatives") return drawSearchV1(design, base, r);
  const search = drawSearchV1("iid_normal", base, r);
  const drift = DESIGN.poorSharpe / Math.sqrt(DESIGN.periodsPerYear);
  for (let k = 1; k < search.length; k++) for (let t = 0; t < search[k].length; t++) search[k][t] += drift;
  return search;
}

function asMatrix(search) {
  const k = search.length, n = search[0].length, d = new Float64Array(n * k);
  for (let j = 0; j < k; j++) for (let t = 0; t < n; t++) d[t * k + j] = search[j][t];
  return { d, n, k };
}

/** Rejection counts of both versions at every level for one chunk of searches. */
export function runChunk(d, c, count) {
  const r = xoshiro(chunkSeed(d, c));
  const counts = Object.fromEntries(Object.keys(VERSIONS).flatMap((v) => TESTS.map((t) => [`${v}.${t}`, DESIGN.levels.map(() => 0)])));
  let disagree = 0;
  for (let i = 0; i < count; i++) {
    const { d: m, n, k } = asMatrix(draw(DESIGN.designs[d], r));
    const seed = seedFrom(r);
    const res = {};
    for (const [v, f] of Object.entries(VERSIONS)) {
      const out = f(m, n, k, { reps: DESIGN.reps, seed, alpha: 0.05 });
      res[v] = out;
      const p = { spa_consistent: out.spa.p_consistent, spa_upper: out.spa.p_upper, reality_check: out.reality_check.p_value };
      DESIGN.levels.forEach((a, j) => {
        for (const t of ["spa_consistent", "spa_upper", "reality_check"]) if (p[t] <= a) counts[`${v}.${t}`][j]++;
      });
      // StepM is run at alpha 0.05 only; its count is stored at the 5% position.
      if (out.stepm.superior.length) counts[`${v}.stepm_any`][1]++;
    }
    if ((res.old.spa.p_consistent <= 0.05) !== (res.new.spa.p_consistent <= 0.05)) disagree++;
  }
  return { d, c, counts, disagree };
}

function assemble(searches, results) {
  const chunks = searches / DESIGN.chunk;
  const out = {
    schema: "canli.null-zoo.snooping-fix-check", registered: false,
    what: "Size of the author's SPA, StepM and Reality Check before (canlicapital ea398598, as in v1) and after (canlicapital 4c7e429e, PR 562) re-studentizing every resample and counting ties, on the same skill-less searches and bootstrap seeds",
    cores: { old: "src/js/snooping-core.js (canlicapital ea3985981ec873bd2df09850e8045ffcbc685bb1)", new: "analysis/snooping-fix/js/snooping-core.js (canlicapital 4c7e429e760206cf95debccf19ad164bf01b366b)" },
    design: DESIGN, searches_per_design: searches, block: Math.round(Math.cbrt(DESIGN.observations)),
    monte_carlo_se_at_5_percent: Number(Math.sqrt(0.05 * 0.95 / searches).toFixed(6)),
    note: "stepm_any is at alpha 0.05 only (null at 1% and 10%); the Reality Check is unchanged by the fix and serves as a check that both versions saw the same draws",
    size: {}, spa_consistent_disagreements: {},
  };
  DESIGN.designs.forEach((name, d) => {
    const tot = {};
    let dis = 0;
    for (const res of results.filter((x) => x.d === d).sort((a, b) => a.c - b.c)) {
      dis += res.disagree;
      for (const [key, arr] of Object.entries(res.counts)) tot[key] = (tot[key] ?? DESIGN.levels.map(() => 0)).map((x, j) => x + arr[j]);
    }
    if (results.filter((x) => x.d === d).length !== chunks) throw new Error(`missing chunks for ${name}`);
    out.size[name] = Object.fromEntries(Object.entries(tot).map(([key, arr]) => [key, Object.fromEntries(DESIGN.levels.map((a, j) => [String(a), key.endsWith("stepm_any") && j !== 1 ? null : arr[j] / searches]))]));
    out.spa_consistent_disagreements[name] = dis;
  });
  return out;
}

if (!isMainThread) {
  for (const job of workerData.jobs) parentPort.postMessage(runChunk(job.d, job.c, DESIGN.chunk));
} else if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const searches = Number(process.argv[2] ?? 4000);
  const workers = Number(process.argv[3] ?? 4);
  if (searches % DESIGN.chunk) throw new RangeError(`searches must be a multiple of ${DESIGN.chunk}`);
  const jobs = DESIGN.designs.flatMap((_, d) => Array.from({ length: searches / DESIGN.chunk }, (_, c) => ({ d, c })));
  const started = Date.now();
  const results = [];
  await Promise.all(Array.from({ length: workers }, (_, w) => new Promise((resolve, reject) => {
    const mine = jobs.filter((_, i) => i % workers === w);
    const worker = new Worker(fileURLToPath(import.meta.url), { workerData: { jobs: mine } });
    worker.on("message", (m) => { results.push(m); process.stderr.write(`\r${results.length}/${jobs.length} chunks`); });
    worker.on("error", reject);
    worker.on("exit", resolve);
  })));
  const out = assemble(searches, results);
  process.stderr.write(`\ndone in ${Math.round((Date.now() - started) / 1000)} s\n`);
  process.stdout.write(`${JSON.stringify(out, null, 1)}\n`);
}
