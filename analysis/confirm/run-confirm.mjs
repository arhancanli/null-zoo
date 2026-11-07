// Pre-registered confirmation run of 8 November 2026, JavaScript part (analysis/confirm/prereg-confirm.json).
//
// Experiment S/L, joint tests (exp 1): the author's data-snooping implementation as v1 scored it (src/js/snooping-core.js,
//   canlicapital ea398598: fixed Politis-Romano studentizer, strict inequality) and as corrected (analysis/snooping-fix/js/,
//   canlicapital 4c7e429e, PR 562: every resample re-studentized by its own standard deviation, ties counted, StepM's
//   critical value floored at zero), on the same skill-less searches and bootstrap seed per search: the nine v1 families
//   (drawSearchV1) and independent AR(1) strategies at phi = 0.02, 0.05, 0.1 (drawAr1Search). Block 8, 1,000 resamples.
// Experiment L, single-series tests (exp 2): the four tests of ar1-levels.mjs at phi = 0.02, 0.05, 0.1.
// Experiment C, correlated strategies (exp 3): x_k = sqrt(rho) f + sqrt(1 - rho) e_k with f and e_k independent
//   unit-variance AR(1) series of coefficient phi (drawAr1Search), so every strategy is AR(1) with coefficient phi and
//   every pair has correlation rho; rho = 0.3, 0.6 and phi = 0, 0.05, 0.1; the same four tests.
// T = 504, N = 20, 252 periods a year. Seeds: (20261108 + exp * 100000007 + cell * 1000033 + chunk * 7927) >>> 0,
// checked against every earlier xoshiro run's seeds by seeds.mjs. Jobs are dispatched round by round; if the run
// reaches 85 minutes it stops and keeps, for each experiment, only the rounds finished in every cell.
//   node analysis/confirm/run-confirm.mjs [workers] [scale] [base] > analysis/confirm/results-js.json
// scale (default 1) divides every experiment's number of rounds, and base replaces the seed base: smoke tests only.
import { Worker, isMainThread, parentPort } from "node:worker_threads";
import { fileURLToPath } from "node:url";
import { snoopingTests as oldTests } from "../../src/js/snooping-core.js";
import { snoopingTests as newTests } from "../snooping-fix/js/snooping-core.js";
import { xoshiro } from "../../src/scripts/research/null-zoo/xoshiro.mjs";
import { FAMILIES_V1, drawSearchV1 } from "../../src/scripts/research/null-zoo/v1/families.mjs";
import { AR1_TESTS, drawAr1Search } from "../../src/scripts/research/null-zoo/v1/ar1-levels.mjs";
import { summarize } from "../../src/scripts/research/null-zoo/validators.mjs";
import { assertDisjoint, confirmSeed } from "./seeds.mjs";

export const DESIGN = Object.freeze({
  trials: 20, observations: 504, periodsPerYear: 252, reps: 1000, baseSeed: 20261108, deadlineMinutes: 85,
  levels: Object.freeze([0.01, 0.05, 0.1]), phis: Object.freeze([0.02, 0.05, 0.1]),
  rhos: Object.freeze([0.3, 0.6]), corrPhis: Object.freeze([0, 0.05, 0.1]),
  joint: { chunk: 100, rounds: 30 }, single: { chunk: 1000, rounds: 10 },
});
const base = { trials: DESIGN.trials, observations: DESIGN.observations, periodsPerYear: DESIGN.periodsPerYear, skill: 0 };
export const EXPERIMENTS = {
  1: [...FAMILIES_V1.map((f) => ({ label: `S.${f}`, family: f })), ...DESIGN.phis.map((phi) => ({ label: `L.phi${phi}`, phi }))],
  2: DESIGN.phis.map((phi) => ({ label: `L.phi${phi}`, phi, rho: 0 })),
  3: DESIGN.rhos.flatMap((rho) => DESIGN.corrPhis.map((phi) => ({ label: `C.rho${rho}.phi${phi}`, phi, rho }))),
};
const JOINT = ["spa_consistent", "spa_upper", "stepm_any", "reality_check"];
const VERSIONS = { v1: oldTests, corrected: newTests };
const seedFrom = (r) => (r.next() >>> 1) || 1;

/** N strategies, each AR(1) with coefficient phi and unit variance, pairwise correlation rho (rho = 0: independent). */
export function drawCorrelatedAr1(phi, rho, r) {
  const own = drawAr1Search(phi, base, r);
  if (rho === 0) return own;
  const [f] = drawAr1Search(phi, { ...base, trials: 1 }, r);
  const a = Math.sqrt(rho), b = Math.sqrt(1 - rho);
  return own.map((xs) => xs.map((x, t) => a * f[t] + b * x));
}

function asMatrix(search) {
  const k = search.length, n = search[0].length, d = new Float64Array(n * k);
  for (let j = 0; j < k; j++) for (let t = 0; t < n; t++) d[t * k + j] = search[j][t];
  return d;
}

export function runJob({ exp, cell, chunk, baseSeed }) {
  const spec = EXPERIMENTS[exp][cell];
  const r = xoshiro(confirmSeed(baseSeed, exp, cell, chunk));
  const counts = {};
  const add = (name, p) => {
    const row = (counts[name] ??= DESIGN.levels.map(() => 0));
    DESIGN.levels.forEach((a, j) => { if (p <= a) row[j]++; });
  };
  if (exp === 1) {
    for (let i = 0; i < DESIGN.joint.chunk; i++) {
      const search = spec.family ? drawSearchV1(spec.family, base, r) : drawAr1Search(spec.phi, base, r);
      const d = asMatrix(search);
      const seed = seedFrom(r);
      for (const [v, f] of Object.entries(VERSIONS)) {
        const out = f(d, DESIGN.observations, DESIGN.trials, { reps: DESIGN.reps, seed, alpha: 0.05 });
        add(`${v}.spa_consistent`, out.spa.p_consistent);
        add(`${v}.spa_upper`, out.spa.p_upper);
        add(`${v}.reality_check`, out.reality_check.p_value);
        // StepM runs at alpha 0.05 only: "p" 0 if it declares any strategy superior, 1 otherwise; read at 5%.
        add(`${v}.stepm_any`, out.stepm.superior.length ? 0.05 : 1);
      }
    }
  } else {
    for (let i = 0; i < DESIGN.single.chunk; i++) {
      const search = drawCorrelatedAr1(spec.phi, spec.rho, r);
      const s = summarize(search, DESIGN.periodsPerYear);
      for (const [name, test] of Object.entries(AR1_TESTS)) {
        const p = test(search, s);
        if (!(p >= 0 && p <= 1)) throw new RangeError(`${name} gave p = ${p}`);
        add(name, p);
      }
    }
  }
  return { exp, cell, chunk, counts };
}

if (!isMainThread) {
  parentPort.on("message", (job) => parentPort.postMessage(runJob(job)));
} else if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const workers = Number(process.argv[2] ?? 12);
  const scale = Number(process.argv[3] ?? 1);
  const baseSeed = Number(process.argv[4] ?? DESIGN.baseSeed);
  if (baseSeed === DESIGN.baseSeed) assertDisjoint(DESIGN, EXPERIMENTS);
  const rounds = { 1: DESIGN.joint.rounds / scale, 2: DESIGN.single.rounds / scale, 3: DESIGN.single.rounds / scale };
  const maxRounds = Math.max(...Object.values(rounds));
  const jobs = [];
  for (let c = 0; c < maxRounds; c++) for (const exp of [1, 2, 3]) if (c < rounds[exp]) EXPERIMENTS[exp].forEach((_, cell) => jobs.push({ exp, cell, chunk: c, baseSeed }));
  const started = Date.now();
  process.stderr.write(`confirm (js): START ${new Date().toISOString()} jobs=${jobs.length} workers=${workers}\n`);
  const done = new Map();
  let next = 0, stopped = false;
  await new Promise((resolve, reject) => {
    const pool = Array.from({ length: workers }, () => new Worker(fileURLToPath(import.meta.url)));
    let busy = 0;
    const finish = () => { pool.forEach((w) => w.terminate()); resolve(); };
    const feed = (w) => {
      if (next < jobs.length && !stopped) { busy++; w.postMessage(jobs[next++]); }
      else if (busy === 0) finish();
    };
    for (const w of pool) {
      w.on("message", (m) => {
        busy--;
        done.set(`${m.exp}/${m.cell}/${m.chunk}`, m.counts);
        if (done.size % 24 === 0) process.stderr.write(`confirm (js): ${done.size}/${jobs.length} jobs, ${((Date.now() - started) / 60000).toFixed(1)} min\n`);
        if (Date.now() - started > DESIGN.deadlineMinutes * 60000) stopped = true;
        if (stopped && busy === 0) finish(); else feed(w);
      });
      w.on("error", reject);
      feed(w);
    }
  });
  const out = {
    schema: "canli.null-zoo.confirm.results-js", registered: true, prereg: "analysis/confirm/prereg-confirm.json",
    generator: "xoshiro128** seeded by splitmix32", design: DESIGN, base_seed: baseSeed, scale,
    stopped_at_deadline: stopped, minutes: Number(((Date.now() - started) / 60000).toFixed(1)),
    cores: { v1: "src/js/snooping-core.js (canlicapital ea398598)", corrected: "analysis/snooping-fix/js/snooping-core.js (canlicapital 4c7e429e, PR 562)" },
    experiments: {},
  };
  for (const exp of [1, 2, 3]) {
    let complete = 0;
    while (complete < rounds[exp] && EXPERIMENTS[exp].every((_, cell) => done.has(`${exp}/${cell}/${complete}`))) complete++;
    const per = exp === 1 ? DESIGN.joint.chunk : DESIGN.single.chunk;
    const cells = {};
    EXPERIMENTS[exp].forEach((spec, cell) => {
      const tot = {};
      for (let c = 0; c < complete; c++) for (const [name, row] of Object.entries(done.get(`${exp}/${cell}/${c}`))) tot[name] = (tot[name] ?? row.map(() => 0)).map((x, j) => x + row[j]);
      cells[spec.label] = { ...spec, searches: complete * per, rejections: tot };
    });
    out.experiments[exp] = { rounds_planned: rounds[exp], rounds_complete: complete, searches_per_cell: complete * per, cells };
  }
  process.stderr.write(`confirm (js): DONE in ${out.minutes} min\n`);
  process.stdout.write(`${JSON.stringify(out, null, 1)}\n`);
}
