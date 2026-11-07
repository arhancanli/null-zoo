// Null Zoo v1: the size and power of every validator, and of the candidates for audit_backtest's
// calibrated headline test, on nine return families (v0's eight plus block_cluster) at four levels
// of skill, with the xoshiro128** generator. The bar the candidates must meet is pre-registered in
// config/research/null-zoo-v1-prereg.json and was committed before this run.
//
// The work is split into chunks of searches with fixed seeds, spread over worker threads, and merged
// in chunk order, so the output is byte-identical whatever the number of workers.
//   node scripts/research/null-zoo/v1/run.mjs [reps per cell] [workers] [base seed] > config/research/null-zoo-v1.json
// The base seed defaults to v1's (20261004); the confirmation run v1b uses 20261005.
import { Worker, isMainThread, parentPort } from "node:worker_threads";
import { availableParallelism } from "node:os";
import { fileURLToPath } from "node:url";
import { summarize } from "../validators.mjs";
import { xoshiro } from "../xoshiro.mjs";
import { BLOCK_CLUSTER, FAMILIES_V1, drawSearchV1 } from "./families.mjs";
import { DECISION_ONLY, SIDES, V1_SETTINGS, V1_VALIDATORS, pValuesV1 } from "./validators.mjs";

export const DESIGN_V1 = Object.freeze({
  trials: 20, observations: 504, periodsPerYear: 252, levels: Object.freeze([0.01, 0.05, 0.1]),
  skills: Object.freeze([0, 1, 2, 3]), chunk: 250, baseSeed: 20261004,
});
// p-values in [0, 0.2) are kept at 0.0002 resolution for size-adjusted power; the rest share one bin.
export const HIST = Object.freeze({ bins: 1000, max: 0.2 });

export function cellList(design = DESIGN_V1) {
  const cells = [];
  FAMILIES_V1.forEach((family, f) => design.skills.forEach((skill, s) => cells.push({ family, skill, index: f * design.skills.length + s })));
  return cells;
}

export const chunkSeed = (design, cell, chunk) => (design.baseSeed + cell.index * 1000003 + chunk * 7919) >>> 0;

/** Runs `count` searches of one cell from one chunk's seed; returns integer counts and float sums. */
export function runChunk(cell, chunk, count, design = DESIGN_V1, settings = V1_SETTINGS) {
  const r = xoshiro(chunkSeed(design, cell, chunk));
  const rejections = Object.fromEntries(V1_VALIDATORS.map((v) => [v, design.levels.map(() => 0)]));
  const hist = Object.fromEntries(V1_VALIDATORS.map((v) => [v, new Array(HIST.bins + 1).fill(0)]));
  let bestIsSkilled = 0, lagBest = 0, skewBest = 0;
  for (let rep = 0; rep < count; rep++) {
    const search = drawSearchV1(cell.family, { ...design, skill: cell.skill }, r);
    const s = summarize(search, design.periodsPerYear);
    if (s.best === 0) bestIsSkilled++;
    const b = search[s.best];
    const m = s.stats[s.best].mean;
    let c = 0, v = 0;
    for (let i = 0; i < b.length; i++) { const d = b[i] - m; v += d * d; if (i > 0) c += d * (b[i - 1] - m); }
    lagBest += c / v;
    skewBest += s.stats[s.best].skew;
    const ps = pValuesV1(search, s, r, settings);
    for (const name of V1_VALIDATORS) {
      const p = ps[name];
      if (!(p >= 0 && p <= 1)) throw new RangeError(`${name} gave p = ${p} in ${cell.family} skill ${cell.skill}`);
      design.levels.forEach((a, i) => { if (p <= a) rejections[name][i]++; });
      hist[name][p >= HIST.max ? HIST.bins : Math.floor((p / HIST.max) * HIST.bins)]++;
    }
  }
  return { rejections, hist, bestIsSkilled, lagBest, skewBest, count };
}

export function jobList(reps, design = DESIGN_V1) {
  const jobs = [];
  for (const cell of cellList(design)) {
    for (let chunk = 0; chunk * design.chunk < reps; chunk++) jobs.push({ id: jobs.length, cell, chunk, count: Math.min(design.chunk, reps - chunk * design.chunk), baseSeed: design.baseSeed });
  }
  return jobs;
}

/** Merges chunk results in job order and renders the output object. */
export function assemble(reps, jobs, results, design = DESIGN_V1, settings = V1_SETTINGS) {
  const cells = cellList(design).map((cell) => {
    const mine = jobs.filter((j) => j.cell.index === cell.index).map((j) => results[j.id]);
    const total = (pick) => mine.reduce((a, res) => a + pick(res), 0);
    const rejections = Object.fromEntries(V1_VALIDATORS.map((name) => [name, Object.fromEntries(design.levels
      .filter((a) => !DECISION_ONLY.includes(name) || a === 0.05)
      .map((a) => [String(a), total((res) => res.rejections[name][design.levels.indexOf(a)])]))]));
    const hist = Object.fromEntries(V1_VALIDATORS.filter((name) => !DECISION_ONLY.includes(name)).map((name) => {
      const merged = new Array(HIST.bins + 1).fill(0);
      for (const res of mine) res.hist[name].forEach((n, i) => { merged[i] += n; });
      // Sparse: [bin, count] for the non-empty bins only.
      return [name, merged.flatMap((n, i) => (n ? [[i, n]] : []))];
    }));
    return {
      family: cell.family, skill: cell.skill, searches: total((res) => res.count),
      seeds: mine.length ? [chunkSeed(design, cell, 0), chunkSeed(design, cell, mine.length - 1)] : [],
      best_is_skilled_trial: total((res) => res.bestIsSkilled),
      mean_lag1_of_best: Number((total((res) => res.lagBest) / reps).toFixed(6)),
      mean_skew_of_best: Number((total((res) => res.skewBest) / reps).toFixed(6)),
      rejections, p_histogram: hist,
    };
  });
  return {
    schema: "canli.null-zoo.v1",
    generator: "xoshiro128** seeded by splitmix32; one seed per chunk of searches",
    design, settings, reps_per_cell: reps,
    families: { names: FAMILIES_V1, block_cluster: BLOCK_CLUSTER },
    validators: Object.fromEntries(V1_VALIDATORS.map((v) => [v, { sides: SIDES[v], decision_only: DECISION_ONLY.includes(v) }])),
    histogram: { bins: HIST.bins, max: HIST.max, note: "bin i holds p in [i*max/bins, (i+1)*max/bins); the last bin holds p >= max" },
    cells,
  };
}

async function main() {
  const reps = Number(process.argv[2] ?? 10000);
  const workers = Number(process.argv[3] ?? Math.max(1, availableParallelism() - 2));
  const design = { ...DESIGN_V1, baseSeed: Number(process.argv[4] ?? DESIGN_V1.baseSeed) };
  const jobs = jobList(reps, design);
  const results = new Array(jobs.length);
  const started = Date.now();
  let done = 0;
  process.stderr.write(`null-zoo v1: START ${new Date().toISOString()} reps=${reps} cells=${cellList().length} jobs=${jobs.length} workers=${workers}\n`);
  await new Promise((resolve, reject) => {
    let nextJob = 0;
    let live = 0;
    const spawn = () => {
      const w = new Worker(fileURLToPath(import.meta.url));
      live++;
      const feed = () => { if (nextJob < jobs.length) w.postMessage(jobs[nextJob++]); else { w.terminate(); } };
      w.on("message", ({ id, res }) => {
        results[id] = res;
        done++;
        if (done % 20 === 0 || done === jobs.length) {
          const s = (Date.now() - started) / 1000;
          process.stderr.write(`null-zoo v1: ${done}/${jobs.length} chunks, ${s.toFixed(0)} s, about ${((s / done) * (jobs.length - done) / 60).toFixed(1)} min left\n`);
        }
        feed();
      });
      w.on("error", reject);
      w.on("exit", () => { live--; if (live === 0) (done === jobs.length ? resolve() : reject(new Error(`stopped at ${done}/${jobs.length}`))); });
      feed();
    };
    for (let i = 0; i < Math.min(workers, jobs.length); i++) spawn();
  });
  process.stdout.write(`${JSON.stringify(assemble(reps, jobs, results, design))}\n`);
  process.stderr.write(`null-zoo v1: DONE ${new Date().toISOString()} in ${((Date.now() - started) / 60000).toFixed(1)} min\n`);
}

if (!isMainThread) {
  parentPort.on("message", (job) => parentPort.postMessage({ id: job.id, res: runChunk(job.cell, job.chunk, job.count, { ...DESIGN_V1, baseSeed: job.baseSeed }) }));
} else if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => { process.stderr.write(`null-zoo v1: FAILED ${e.stack ?? e}\n`); process.exit(1); });
}
