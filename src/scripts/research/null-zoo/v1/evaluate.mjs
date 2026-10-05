// Reads a Null Zoo v1 run and the pre-registration, and derives everything published from them:
// each validator's size and power per family, size-adjusted power, the ceiling, and the headline
// test each audit path uses, chosen by the pre-registered rule. Nothing is typed by hand.
//   node scripts/research/null-zoo/v1/evaluate.mjs [run] [prereg] [evaluation out] [sizes module out]
// defaults: config/research/null-zoo-v1.json, config/research/null-zoo-v1-prereg.json,
// config/research/null-zoo-v1-evaluation.json and js/null-zoo-v1-sizes.js.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");

const SIZE_BAR = 0.06;
const POWER_BAR = 0.45;

export function evaluate(run, prereg) {
  const families = run.families.names;
  const cell = (family, skill) => run.cells.find((c) => c.family === family && c.skill === skill);
  const rate = (family, skill, v, level = "0.05") => cell(family, skill).rejections[v][level] / cell(family, skill).searches;
  const se = (p, n) => Math.sqrt((p * (1 - p)) / n);
  const validators = Object.keys(run.validators);
  const cumulative = (family, skill, v) => {
    const hist = cell(family, skill).p_histogram[v];
    const n = cell(family, skill).searches;
    // Cumulative share at or below each bin's upper edge.
    const out = new Float64Array(run.histogram.bins + 1);
    for (const [bin, count] of hist) out[bin] += count;
    let acc = 0;
    for (let i = 0; i < out.length; i++) { acc += out[i]; out[i] = acc / n; }
    return out;
  };
  // Size-adjusted power: the largest histogram edge at which this validator's null rejection rate in
  // the same family is still at most 5%, applied to the skilled cell.
  const sizeAdjusted = (family, skill, v) => {
    if (run.validators[v].decision_only) return null;
    const nullCum = cumulative(family, 0, v);
    let edge = -1;
    for (let i = 0; i < run.histogram.bins; i++) if (nullCum[i] <= 0.05) edge = i; else break;
    if (edge < 0) return 0;
    return cumulative(family, skill, v)[edge];
  };
  const table = Object.fromEntries(validators.map((v) => [v, {
    size: Object.fromEntries(families.map((f) => [f, rate(f, 0, v)])),
    power: Object.fromEntries(families.map((f) => [f, Object.fromEntries(run.design.skills.filter((s) => s > 0).map((s) => [String(s), rate(f, s, v)]))])),
    size_adjusted_power_skill_2: Object.fromEntries(families.map((f) => [f, sizeAdjusted(f, 2, v)])),
  }]));
  const n = run.reps_per_cell;
  const passesSize = (v, f) => table[v].size[f] <= SIZE_BAR;
  const powerIid = (v) => table[v].power.iid_normal["2"];
  const ceiling = powerIid(prereg.ceiling.validator);
  const choose = (candidates) => {
    const scored = candidates.map((v, order) => ({
      v, order,
      sizeFamilies: families.filter((f) => passesSize(v, f)),
      power: powerIid(v),
    }));
    const full = scored.find((s) => s.sizeFamilies.length === families.length && s.power >= POWER_BAR);
    const pick = full ?? [...scored].filter((s) => s.power >= POWER_BAR).sort((a, b) => b.sizeFamilies.length - a.sizeFamilies.length || a.order - b.order)[0] ?? null;
    if (!pick) return { test: null, reason: "no candidate meets the power bar", candidates: scored.map(({ v, sizeFamilies, power }) => ({ test: v, size_passes: sizeFamilies.length, power_iid_skill_2: power })) };
    const failures = families.filter((f) => !passesSize(pick.v, f)).map((f) => ({ family: f, size: table[pick.v].size[f] }));
    return {
      test: pick.v,
      meets_every_bar: failures.length === 0,
      size_by_family: table[pick.v].size,
      worst: families.reduce((w, f) => (table[pick.v].size[f] > table[pick.v].size[w] ? f : w), families[0]),
      power_iid_skill_2: pick.power,
      failures,
      candidates: scored.map(({ v, sizeFamilies, power }) => ({ test: v, size_passes: sizeFamilies.length, power_iid_skill_2: power })),
    };
  };
  return {
    schema: "canli.null-zoo.v1.evaluation",
    run: { reps_per_cell: n, generator: run.generator, settings: run.settings, monte_carlo_se_at_5_percent: se(0.05, n) },
    bar: { size: SIZE_BAR, power: POWER_BAR, from: "config/research/null-zoo-v1-prereg.json" },
    ceiling: { validator: prereg.ceiling.validator, power_iid_skill_2: ceiling, power_bar_reachable: ceiling >= POWER_BAR },
    headline: {
      with_variants: choose(prereg.candidates.with_variants),
      single_series: choose(prereg.candidates.single_series),
    },
    table,
  };
}

export function sizesModule(evaluation, evaluationPath = "config/research/null-zoo-v1-evaluation.json") {
  const pick = (h) => (h.test ? { test: h.test, size_by_family: h.size_by_family, worst: h.worst, meets_every_bar: h.meets_every_bar, power_iid_skill_2: h.power_iid_skill_2 } : null);
  const body = {
    source: `https://github.com/arhancanli/canlicapital/blob/main/${evaluationPath} (${evaluation.run.reps_per_cell.toLocaleString("en-US")} searches per cell; bar pre-registered in ${evaluation.prereg ?? "config/research/null-zoo-v1-prereg.json"})`,
    level: 0.05,
    with_variants: pick(evaluation.headline.with_variants),
    single_series: pick(evaluation.headline.single_series),
  };
  return [
    "// Generated by scripts/research/null-zoo/v1/evaluate.mjs from the Null Zoo v1 run. Do not edit.",
    "// The measured false-positive rate (size) of audit_backtest's headline tests on each return family.",
    `export const NULL_ZOO_V1 = Object.freeze(${JSON.stringify(body, null, 2)});`,
    "",
  ].join("\n");
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [runPath, preregPath, evalOut, sizesOut] = process.argv.slice(2);
  const run = JSON.parse(readFileSync(resolve(ROOT, runPath ?? "config/research/null-zoo-v1.json"), "utf8"));
  const prereg = JSON.parse(readFileSync(resolve(ROOT, preregPath ?? "config/research/null-zoo-v1-prereg.json"), "utf8"));
  const evaluation = evaluate(run, prereg);
  evaluation.prereg = preregPath ?? "config/research/null-zoo-v1-prereg.json";
  const evaluationPath = evalOut ?? "config/research/null-zoo-v1-evaluation.json";
  writeFileSync(resolve(ROOT, evaluationPath), `${JSON.stringify(evaluation, null, 1)}\n`);
  writeFileSync(resolve(ROOT, sizesOut ?? "js/null-zoo-v1-sizes.js"), sizesModule(evaluation, evaluationPath));
  const fmt = (x) => (x === null ? "  -  " : x.toFixed(3));
  const fams = run.families.names;
  console.log(`ceiling (oracle) power on iid_normal at skill 2: ${fmt(evaluation.ceiling.power_iid_skill_2)}`);
  console.log(`size at 5% (rows) by family (columns): ${fams.join(" ")}`);
  for (const [v, t] of Object.entries(evaluation.table)) console.log(`${v.padEnd(32)} ${fams.map((f) => fmt(t.size[f])).join(" ")}   power iid@2 ${fmt(t.power.iid_normal["2"])}`);
  for (const [path, h] of Object.entries(evaluation.headline)) console.log(`HEADLINE ${path}: ${h.test} (every bar met: ${h.meets_every_bar}; failures ${JSON.stringify(h.failures ?? [])})`);
}
