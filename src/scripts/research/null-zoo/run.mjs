// The Null Zoo: scores each validator's size (how often it calls a skill-less search's best trial
// skilled) and power (how often it finds a trial with real skill) across return families with
// known ground truth. Deterministic: every cell has its own seed, recorded in the output.
//   node scripts/research/null-zoo/run.mjs [reps] > config/research/null-zoo-v0.json
import { FAMILIES, drawSearch, rng } from "./families.mjs";
import { VALIDATORS, summarize } from "./validators.mjs";

export const DESIGN = Object.freeze({ trials: 20, observations: 504, periodsPerYear: 252, skill: 2, level: 0.05 });

export function scoreCell(family, { reps, skill, seed, design = DESIGN, validators = Object.keys(VALIDATORS) }) {
  const r = rng(seed);
  const rejections = Object.fromEntries(validators.map((v) => [v, 0]));
  let bestIsSkilled = 0;
  for (let rep = 0; rep < reps; rep++) {
    const search = drawSearch(family, { ...design, skill }, r);
    const summary = summarize(search, design.periodsPerYear);
    if (summary.best === 0) bestIsSkilled++;
    for (const v of validators) if (VALIDATORS[v].p(search, summary, r) <= design.level) rejections[v]++;
  }
  return { rates: Object.fromEntries(validators.map((v) => [v, rejections[v] / reps])), best_is_skilled_trial: bestIsSkilled / reps };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const reps = Number(process.argv[2] ?? 1000);
  const rows = [];
  let seed = 1000;
  for (const family of FAMILIES) {
    const size = scoreCell(family, { reps, skill: 0, seed: seed++ });
    const power = scoreCell(family, { reps, skill: DESIGN.skill, seed: seed++ });
    rows.push({ family, size: size.rates, power: power.rates, power_best_is_skilled_trial: power.best_is_skilled_trial, seeds: [seed - 2, seed - 1] });
    process.stderr.write(`${family} `);
  }
  process.stderr.write("\n");
  console.log(JSON.stringify({
    schema: "canli.null-zoo.v0",
    design: DESIGN,
    reps,
    validators: Object.fromEntries(Object.entries(VALIDATORS).map(([k, v]) => [k, { sides: v.sides }])),
    rows,
  }, null, 1));
}
