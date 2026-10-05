// Builds every table and every number quoted in the paper from the result files, so no figure in
// the text is typed by hand. The paper reads a number with \nz{key}; LaTeX stops with an error if
// a key is missing, so a number cannot silently go stale or be invented.
//   node analysis/tables.mjs  ->  paper/generated/{values.tex,tables.tex,appendix-tables.tex}
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { expectedMaxStandardNormal, normalCdf, normalPpf } from "../src/js/dsr-core.js";
import { studentTUpper } from "../src/js/student-t.js";

const here = new URL("..", import.meta.url).pathname;
const read = (p) => JSON.parse(readFileSync(`${here}${p}`, "utf8"));

const V1 = read("src/config/research/null-zoo-v1-evaluation.json");
const V1B = read("src/config/research/null-zoo-v1b-evaluation.json");
const V1RAW = read("src/config/research/null-zoo-v1.json");
const V1BRAW = read("src/config/research/null-zoo-v1b.json");
const V1PRE = read("src/config/research/null-zoo-v1-prereg.json");
const V2 = read("analysis/v2/evaluation-v2.json");
const V2RUN = read("analysis/v2/results-v2.json");
const V2ARCH = read("analysis/v2/arch-check-v2.json");
const EXT = read("analysis/null-zoo-extended.json");
const BR = read("analysis/bootstrap-resolution.json");
const EXPLORE = read("analysis/v2/explore-floor-v2.json");
const V0PATH = `${here}src/config/research/null-zoo-v0.json`;
const V0 = JSON.parse(readFileSync(V0PATH, "utf8"));

const LEVEL = 0.05;
const N_TRIALS = V1RAW.design.trials;
const T_OBS = V1RAW.design.observations;
const nV1 = V1.run.reps_per_cell;

export const FAMILIES = ["iid_normal", "student_t4", "skew_negative", "skew_positive", "garch", "ar1", "regimes", "correlated_trials", "block_cluster"];
export const FAMILY_LABEL = {
  iid_normal: "IID normal",
  student_t4: "Student $t_4$",
  skew_negative: "Skew $-1.32$",
  skew_positive: "Skew $+1.32$",
  garch: "GARCH(1,1)",
  ar1: "AR(1), $\\phi=0.2$",
  regimes: "Two vol.\\ regimes",
  correlated_trials: "Correlated, $\\rho=0.5$",
  block_cluster: "Clusters, $\\rho=0.8$",
};
const NON_AR = FAMILIES.filter((f) => f !== "ar1");

const se = (p, n) => Math.sqrt((p * (1 - p)) / n);
const z99 = normalPpf(0.995);
const band = (n) => z99 * se(LEVEL, n);
const pct = (x, d = 1) => (100 * x).toFixed(d);
const nf = (x) => x.toLocaleString("en-US");
const range = (xs, d = 1) => `${pct(Math.min(...xs), d)}--${pct(Math.max(...xs), d)}`;

// ---- values ------------------------------------------------------------------------------------
const values = new Map();
const def = (key, value) => {
  if (values.has(key)) throw new Error(`duplicate value ${key}`);
  if (value === undefined || value === null || (typeof value === "number" && !Number.isFinite(value))) throw new Error(`bad value ${key}: ${value}`);
  values.set(key, String(value));
};

def("trials", N_TRIALS);
def("obs", T_OBS);
def("reps.v1", nf(nV1));
def("reps.v0", nf(V0.reps));
def("reps.ext", nf(EXT.reps));
def("se5.10000", pct(se(LEVEL, 10000), 2));
def("se5.5000", pct(se(LEVEL, 5000), 2));
def("se5.2000", pct(se(LEVEL, 2000), 2));
def("band99.10000", `${pct(LEVEL - band(10000), 2)}--${pct(LEVEL + band(10000), 2)}`);
def("band99.5000", `${pct(LEVEL - band(5000), 2)}--${pct(LEVEL + band(5000), 2)}`);
def("v1.spa_reps", nf(V1.run.settings.spaReps));
def("v1.boot_t_reps", nf(V1.run.settings.bootstrapTReps));
def("v1.boot_best_draws", nf(V1.run.settings.bootstrapBestDraws));
def("v1.bar_size", pct(V1.bar.size, 0));
def("v1.bar_power", pct(V1.bar.power, 0));
def("v1.base_seed", V1RAW.design.baseSeed);
def("v1b.base_seed", read("src/config/research/null-zoo-v1b-prereg.json").design.base_seed);
def("v2.base_seed", V2RUN.base_seed);
def("v2.reps", nf(V2RUN.bootstrap_reps));

// Size-adjusted power is undefined when the smallest attainable p-value already occurs in more than
// 5% of the null cell (v1's evaluate.mjs then stores 0). Detect that from the raw p-value histograms.
function adjUndefined(raw, v, f) {
  const cell = raw.cells.find((c) => c.family === f && c.skill === 0);
  const hist = cell.p_histogram[v];
  if (!hist || !hist.length) return false;
  const [firstBin, count] = hist.reduce((a, b) => (b[0] < a[0] ? b : a));
  return count / cell.searches > LEVEL && firstBin < raw.histogram.bins;
}
const adjText = (ev, raw, v, f) => {
  const adj = ev.table[v].size_adjusted_power_skill_2?.[f];
  if (adj === undefined) return undefined;
  if (adj === null || (adj === 0 && adjUndefined(raw, v, f))) return "n/a";
  return pct(adj);
};
// v1 and v1b: every validator in every family.
for (const [run, ev, raw] of [["v1", V1, V1RAW], ["v1b", V1B, V1BRAW]]) {
  for (const [v, row] of Object.entries(ev.table)) {
    for (const f of FAMILIES) {
      def(`${run}.size.${v}.${f}`, pct(row.size[f]));
      for (const s of ["1", "2", "3"]) if (row.power?.[f]?.[s] !== undefined) def(`${run}.power${s}.${v}.${f}`, pct(row.power[f][s]));
      const adj = adjText(ev, raw, v, f);
      if (adj !== undefined) def(`${run}.adj2.${v}.${f}`, adj);
    }
    const sizes = FAMILIES.map((f) => row.size[f]);
    def(`${run}.size_range.${v}`, range(sizes));
    def(`${run}.size_range_nonar.${v}`, range(NON_AR.map((f) => row.size[f])));
    def(`${run}.fails6.${v}`, FAMILIES.filter((f) => row.size[f] > 0.06).length);
  }
  for (const c of raw.cells) {
    def(`${run}.best_skilled${c.skill}.${c.family}`, pct(c.best_is_skilled_trial / c.searches));
    if (c.skill === 0) def(`${run}.skew_best.${c.family}`, c.mean_skew_of_best.toFixed(2));
  }
  def(`${run}.oracle_power2`, pct(ev.ceiling.power_iid_skill_2));
}
def("v1.validator_count", Object.keys(V1.table).filter((v) => !v.endsWith("_one_sided")).length);
def("v1.headline.single", V1.headline.single_series.test.replaceAll("_", "\\_"));
def("v1b.headline.joint.fails", V1B.headline.with_variants.failures.map((x) => FAMILY_LABEL[x.family]).join(", "));
// Both runs, joint tests: the largest and smallest size over the two runs.
for (const v of ["spa_consistent", "spa_upper", "stepm_any", "reality_check"]) {
  const xs = FAMILIES.flatMap((f) => [V1.table[v].size[f], V1B.table[v].size[f]]);
  def(`both.size_range.${v}`, range(xs));
  def(`both.size_range_nonar.${v}`, range(NON_AR.flatMap((f) => [V1.table[v].size[f], V1B.table[v].size[f]])));
}
// StepM and SPA_c: largest difference in size across the two runs and nine families.
def("both.stepm_spa_maxdiff", pct(Math.max(...FAMILIES.flatMap((f) => [V1, V1B].map((e) => Math.abs(e.table.stepm_any.size[f] - e.table.spa_consistent.size[f])))), 2));

// v2 Python experiments.
const A = V2.A;
for (const f of FAMILIES) {
  for (const [t, x] of Object.entries(A[f].size)) def(`v2.A.size.${t}.${f}`, pct(x));
  for (const [t, x] of Object.entries(A[f].power)) def(`v2.A.power.${t}.${f}`, pct(x));
  for (const [t, x] of Object.entries(A[f].size_adjusted_power)) def(`v2.A.adj.${t}.${f}`, pct(x));
}
for (const t of Object.keys(A.iid_normal.size)) {
  def(`v2.A.size_range.${t}`, range(FAMILIES.map((f) => A[f].size[t])));
  def(`v2.A.size_range_nonar.${t}`, range(NON_AR.map((f) => A[f].size[t])));
  def(`v2.A.fails6.${t}`, FAMILIES.filter((f) => A[f].size[t] > 0.06).length);
  def(`v2.A.fails65_nonar.${t}`, NON_AR.filter((f) => A[f].size[t] > 0.065).length);
}
for (const [fam, designs] of Object.entries(V2.B)) {
  for (const [design, cell] of Object.entries(designs)) {
    for (const [t, x] of Object.entries(cell.size)) def(`v2.B.size.${t}.${fam}.${design}`, pct(x));
    if (!values.has(`v2.B.block.${design}`)) def(`v2.B.block.${design}`, cell.block);
  }
}
for (const [design, cell] of Object.entries(V2.C)) {
  for (const [t, x] of Object.entries(cell.size)) def(`v2.C.size.${t}.${design}`, pct(x));
  for (const [t, x] of Object.entries(cell.power)) def(`v2.C.power.${t}.${design}`, pct(x));
  for (const [t, x] of Object.entries(cell.size_adjusted_power)) def(`v2.C.adj.${t}.${design}`, pct(x));
  def(`v2.C.best_t.${design}`, pct(cell.best_t_is_skilled));
  def(`v2.C.best_mean.${design}`, pct(cell.best_mean_is_skilled));
}
for (const [fam, cell] of Object.entries(V2.D)) {
  def(`v2.D.searches.${fam}`, nf(cell.searches));
  for (const [k, x] of Object.entries(cell.size)) def(`v2.D.size.${k}.${fam}`, pct(x));
  for (const [lab, x] of Object.entries(cell.flag_changes_p)) def(`v2.D.flagchanges.${lab}.${fam}`, x);
  for (const [lab, kinds] of Object.entries(cell.mean_abs_p_difference_arch_vs_ours)) for (const [kind, x] of Object.entries(kinds)) def(`v2.D.meanabs.${lab}.${kind}.${fam}`, x.toFixed(3));
}
def("v2.D.arch", V2ARCH.arch);
def("v2.D.default_block", V2ARCH.default_block);
def("v2.D.total_searches", nf(Object.values(V2.D).reduce((a, c) => a + c.searches, 0)));
def("v2.D.total_flag_changes", Object.values(V2.D).reduce((a, c) => a + Object.values(c.flag_changes_p).reduce((x, y) => x + y, 0), 0));
const PRED = Object.keys(V2.predictions).sort();
for (const id of PRED) def(`v2.${id}.held`, V2.predictions[id].held ? "held" : "did not hold");
def("v2.predictions.held_count", PRED.filter((id) => V2.predictions[id].held).length);
def("v2.predictions.count", PRED.length);
def("v2.total_searches", nf(V2RUN.cells.reduce((a, c) => a + c.searches, 0)));
def("v2.D.total_pairs", nf(2 * Object.values(V2.D).reduce((a, c) => a + c.searches, 0)));
{
  const diffs = Object.values(V2.D).flatMap((c) => Object.values(c.mean_abs_p_difference_arch_vs_ours).flatMap((k) => Object.values(k)));
  def("v2.D.meanabs_range", `${Math.min(...diffs).toFixed(3)}--${Math.max(...diffs).toFixed(3)}`);
}
// Exploratory (not pre-registered) check of the tie convention, fresh seed.
if (!EXPLORE.exploratory) throw new Error("explore-floor-v2.json must be marked exploratory");
def("explore.seed", EXPLORE.seed);
def("explore.searches", nf(EXPLORE.searches));
def("explore.share_T_zero", pct(EXPLORE.share_T_zero));
def("explore.share_all_below", pct(EXPLORE.share_T_zero_and_all_below_threshold));
for (const [k, x] of Object.entries(EXPLORE.size)) def(`explore.size.${k}`, pct(x));
for (const [k, x] of Object.entries(EXPLORE.size_from_searches_with_T_zero)) def(`explore.size_T_zero.${k}`, pct(x));
// P1 detail: the largest absolute difference between the Python and JavaScript sizes.
def("v2.P2.pass_count", Object.values(V2.predictions.P2.detail).filter((x) => x <= 0.06).length);
def("v2.P2.family_count", Object.keys(V2.predictions.P2.detail).length);
for (const t of ["spa_c_boot_t", "rc"]) def(`v2.B.range.${t}`, range(Object.values(V2.B).flatMap((designs) => Object.values(designs).map((c) => c.size[t]))));
def("v2.P1.maxdiff", pct(Math.max(...V2.predictions.P1.detail.map((r) => Math.abs(r.difference))), 2));

// Bootstrap resolution (paired run, v0 design).
def("br.reps", nf(BR.reps));
for (const r of BR.rows) for (const arm of ["size", "power"]) for (const [k, kk] of [["luck_trials", "let"], ["bootstrap_400", "b400"], ["bootstrap_1999", "b1999"]]) def(`br.${arm}.${kk}.${r.family}`, pct(r[arm][k] / BR.reps, arm === "size" ? 2 : 1));
const sidakP1 = -Math.expm1(Math.log1p(-LEVEL) / N_TRIALS);
const minBootP = (b) => -Math.expm1(N_TRIALS * Math.log1p(-1 / (b + 1)));
def("boot.min_p.400", minBootP(400).toFixed(4));
def("boot.min_p.1999", minBootP(1999).toFixed(4));
def("boot.max_hits.1999", Math.floor((1999 + 1) * sidakP1) - 1);
def("sidak.p1", sidakP1.toFixed(6));
def("bonf.p1", (LEVEL / N_TRIALS).toFixed(6));

// DSR analytics: Sidak critical value, the expected-maximum benchmark and the implied hurdles. The
// cross-trial dispersion diagnostics come from the v0-design robustness run (analysis/null-zoo-extended.json).
function exactExpectedMax(n) {
  let s = 0;
  const h = 1e-3;
  for (let x = -10; x <= 10; x += h) s += x * n * normalCdf(x) ** (n - 1) * (Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI)) * h;
  return s;
}
const sidakZ = normalPpf(1 - sidakP1);
const approxMax = expectedMaxStandardNormal(N_TRIALS);
const oneZ = normalPpf(1 - LEVEL);
const extRow = Object.fromEntries(EXT.rows.map((r) => [r.family, r]));
const sdNull = extRow.iid_normal.diagnostics.null.cross_trial_sd_over_iid_se;
const sdSkill = extRow.iid_normal.diagnostics.skill.cross_trial_sd_over_iid_se;
def("dsr.sidak_crit", sidakZ.toFixed(2));
def("dsr.exp_max", approxMax.toFixed(2));
def("dsr.exp_max_exact", exactExpectedMax(N_TRIALS).toFixed(2));
def("dsr.single_crit", oneZ.toFixed(3));
def("dsr.sd_ratio_null", sdNull.toFixed(2));
def("dsr.sd_ratio_skill", sdSkill.toFixed(2));
def("dsr.hurdle_null", (sdNull * approxMax + oneZ).toFixed(2));
def("dsr.hurdle_skill", (sdSkill * approxMax + oneZ).toFixed(2));
def("dsr.sd_ratio_crossover", ((sidakZ - oneZ) / approxMax).toFixed(2));
def("dsr.sd_ratio_null_corr", extRow.correlated_trials.diagnostics.null.cross_trial_sd_over_iid_se.toFixed(2));
def("dsr.hurdle_null_corr", (extRow.correlated_trials.diagnostics.null.cross_trial_sd_over_iid_se * approxMax + oneZ).toFixed(2));
def("v1.dsr_fail_to_accept.iid_normal", pct(1 - V1.table.deflated_sharpe.power.iid_normal["2"]));
def("v1.dsr_fail_to_accept_range", range(FAMILIES.map((f) => 1 - V1.table.deflated_sharpe.power[f]["2"])));
def("v1.dsr_adj_gap_range", range(FAMILIES.map((f) => V1.table.luck_trials.size_adjusted_power_skill_2[f] - V1.table.deflated_sharpe.size_adjusted_power_skill_2[f])));
// Lo factor: size shift against LET in the eight families without autocorrelation (v1).
const loShift = NON_AR.map((f) => V1.table.luck_trials_lo.size[f] - V1.table.luck_trials.size[f]);
def("v1.lo_shift_range", `${pct(Math.min(...loShift), 2)}--${pct(Math.max(...loShift), 2)}`);
// LET against the one-sided haircut: they can disagree only when p1 lies between the two cutoffs.
def("v1.let_hc_maxdiff", pct(Math.max(...FAMILIES.map((f) => Math.abs(V1.table.luck_trials.size[f] - V1.table.haircut_bonferroni_one_sided.size[f]))), 2));

// A back-of-the-envelope account of the studentized SPA's excess size under i.i.d. normal returns.
// The Politis-Romano estimate w^2 = g0 + 2 sum kappa_i g_i has, to first order, relative variance
// v = (2 + 4 sum_i kappa_i^2 (T - i) / T) / T, so sqrt(T) m / w behaves like Student t with
// nu = 2 / v degrees of freedom. The bootstrap holds w fixed, so its critical value is close to the
// normal Sidak value z*. The predicted size of the maximum over N independent strategies is then
// 1 - (1 - P(t_nu > z*))^N. It ignores the demeaning and the bootstrap's own noise.
const zStar = normalPpf(1 - sidakP1);
def("approx.z_star", zStar.toFixed(2));
for (const b of [1, 8, 22]) {
  const p = 1 / b;
  let s = 0;
  for (let i = 1; i < T_OBS; i++) {
    const kappa = (1 - i / T_OBS) * (1 - p) ** i + (i / T_OBS) * (1 - p) ** (T_OBS - i);
    s += kappa * kappa * (T_OBS - i) / T_OBS;
  }
  const v = (2 + 4 * s) / T_OBS;
  const nu = 2 / v;
  def(`approx.nu.b${b}`, Math.round(nu));
  def(`approx.cv_w.b${b}`, pct(Math.sqrt(v) / 2, 1));
  const predicted = 1 - (1 - studentTUpper(zStar, nu)) ** N_TRIALS;
  def(`approx.size.b${b}`, pct(predicted));
  // Share of the measured excess over 5% (experiment A, i.i.d. normal) that the approximation accounts for.
  const measured = { 1: V2.A.iid_normal.size.spa_c_sd, 8: V2.A.iid_normal.size.spa_c, 22: V2.A.iid_normal.size.spa_c_b22 }[b];
  def(`approx.share.b${b}`, pct((predicted - LEVEL) / (measured - LEVEL), 0));
}

// Provenance.
def("v0.sha", createHash("sha256").update(readFileSync(V0PATH)).digest("hex"));
def("src.commit.v0", readFileSync(`${here}src/SOURCE_COMMIT`, "utf8").trim().slice(0, 12));
def("src.commit.v1", readFileSync(`${here}src/SOURCE_COMMIT_V1`, "utf8").trim().slice(0, 12));
const git = (...a) => execFileSync("git", ["-C", here, ...a], { encoding: "utf8" }).trim();
def("v2.prereg.commit", git("log", "--diff-filter=A", "--format=%H", "--", "analysis/v2/prereg-v2.json").split("\n").pop().slice(0, 12));
def("v1.prereg.candidates", V1PRE.candidates.with_variants.join(", ").replaceAll("_", "\\_"));

// ---- tables ------------------------------------------------------------------------------------
function sizeCell(p, n, d = 1) {
  const s = pct(p, d);
  if (p > LEVEL + band(n)) return `\\textbf{${s}}`;
  if (p < LEVEL - band(n)) return `\\textit{${s}}`;
  return s;
}
function table({ caption, label, rows, cols, cell, note, rowLabel = (r) => FAMILY_LABEL[r], head = "Family", placement = "t", colspec, tight = false, groups }) {
  let t = `\\begin{table}[${placement}]\n\\centering\n${tight ? "\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}" : "\\small"}\n\\caption{${caption}}\n\\label{${label}}\n`;
  t += `\\begin{tabular}{${colspec ?? `l${"r".repeat(cols.length)}`}}\n\\toprule\n`;
  if (groups) t += ` & ${groups.map(([label, span]) => `\\multicolumn{${span}}{c}{${label}}`).join(" & ")} \\\\\n${(() => { let c = 2; return groups.map(([, span]) => { const r = `\\cmidrule(lr){${c}-${c + span - 1}}`; c += span; return r; }).join(""); })()}\n`;
  t += `${head} & ${cols.map(([, l]) => l).join(" & ")} \\\\\n\\midrule\n`;
  for (const r of rows) t += `${rowLabel(r)} & ${cols.map(([k]) => cell(r, k)).join(" & ")} \\\\\n`;
  t += `\\bottomrule\n\\end{tabular}\n`;
  if (note) t += `\\par\\smallskip\\begin{minipage}{0.97\\linewidth}\\footnotesize ${note}\\end{minipage}\n`;
  return `${t}\\end{table}\n\n`;
}
const BAND_NOTE = (n) => `Bold: above the 99\\% Monte Carlo band around 5\\% (${pct(LEVEL - band(n), 2)}--${pct(LEVEL + band(n), 2)} for ${nf(n)} searches); italic: below it.`;

const SINGLE = [
  ["luck_trials", "LET"], ["luck_trials_lo", "LET-Lo"], ["luck_trials_nonnormal_se", "LET-NN"], ["haircut_bonferroni_one_sided", "HC"],
  ["deflated_sharpe", "DSR"], ["bootstrap_best", "Boot"], ["hac_t_sidak", "HAC-$t$"], ["bootstrap_t_sidak", "Boot-$t$"],
];
const JOINT = [["reality_check", "RC"], ["spa_consistent", "SPA$_c$"], ["spa_upper", "SPA$_u$"], ["stepm_any", "StepM"]];
const MECH = [
  ["spa_c", "SPA$_c$"], ["spa_c_sd", "SPA$_c$-SD"], ["spa_c_boot_t", "SPA$_c$-BT"], ["rc", "RC"], ["spa_c_unstud", "SPA$_c$-U"],
  ["spa_c_b22", "SPA$_c$ (22)"], ["spa_c_unstud_b22", "SPA$_c$-U (22)"],
];

let main = "";
main += table({
  caption: `Size of the single-series corrections: percentage of skill-less searches (${N_TRIALS} strategies, ${T_OBS} daily returns) whose best strategy is called skilled at the nominal 5\\% level. Null Zoo v1, ${nf(nV1)} searches per cell.`,
  label: "tab:size-single", rows: FAMILIES, cols: SINGLE,
  cell: (f, v) => sizeCell(V1.table[v].size[f], nV1),
  note: `${BAND_NOTE(nV1)} HC is the Bonferroni haircut read one-sided (two-sided $p$ halved); Boot uses ${nf(V1.run.settings.bootstrapBestDraws)} resamples and Boot-$t$ ${nf(V1.run.settings.bootstrapTReps)}.`,
});
main += table({
  caption: `Size of the joint resampling tests in Null Zoo v1 and in its confirmation run v1b (fresh seeds), ${nf(nV1)} searches per cell each, ${nf(V1.run.settings.spaReps)} stationary-bootstrap resamples.`,
  label: "tab:size-joint", rows: FAMILIES,
  cols: [["reality_check:v1", "RC v1"], ["reality_check:v1b", "RC v1b"], ["spa_consistent:v1", "SPA$_c$ v1"], ["spa_consistent:v1b", "SPA$_c$ v1b"], ["spa_upper:v1", "SPA$_u$ v1"], ["stepm_any:v1", "StepM v1"]],
  cell: (f, k) => { const [v, r] = k.split(":"); return sizeCell((r === "v1" ? V1 : V1B).table[v].size[f], nV1); },
  note: `${BAND_NOTE(nV1)} StepM is scored as a decision (at least one strategy declared superior at 5\\%).`,
});
main += table({
  caption: `Why the studentized SPA test is oversized: size of variants of the consistent SPA test on the same searches (Null Zoo v2, experiment A; independent Python implementation, ${nf(A.iid_normal.searches)} searches per cell, block length 8 unless marked 22).`,
  label: "tab:mechanism", rows: FAMILIES, cols: MECH,
  cell: (f, t) => sizeCell(A[f].size[t], A[f].searches),
  note: `${BAND_NOTE(A.iid_normal.searches)} SPA$_c$: Hansen's statistic studentized by a fixed Politis--Romano long-run variance. SPA$_c$-SD: studentized by the sample standard deviation instead. SPA$_c$-BT: every resample re-studentized by its own standard deviation (bootstrap-$t$). SPA$_c$-U: not studentized, the computation arch~${V2ARCH.arch} performs. (22): block length 22, arch's default for ${T_OBS} periods.`,
});
main += table({
  caption: `Power when one strategy has a true annualized Sharpe ratio of 2 (raw / size-adjusted), Null Zoo v1. The oracle test knows which strategy is skilled and tests it alone; no valid best-of-${N_TRIALS} test can beat it.`,
  label: "tab:power", rows: FAMILIES, tight: true,
  cols: [["oracle_hac_t", "Oracle"], ["luck_trials", "LET"], ["luck_trials_lo", "LET-Lo"], ["deflated_sharpe", "DSR"], ["bootstrap_best", "Boot"], ["hac_t_sidak", "HAC-$t$"], ["reality_check", "RC"], ["spa_consistent", "SPA$_c$"]],
  cell: (f, v) => `${pct(V1.table[v].power[f]["2"])}\\,/\\,${adjText(V1, V1RAW, v, f)}`,
  note: `Size-adjusted power is power at the cutoff at which the same test rejects at most 5\\% of the matching null cell; a user who does not know the return family cannot apply it, but it shows how well each statistic ranks skilled searches above skill-less ones. n/a: the test's smallest attainable $p$-value already occurs in more than 5\\% of the null cell, so no cutoff attains the level. Monte Carlo standard errors are at most ${pct(se(0.5, nV1), 1)} points.`,
});
{
  const designs = ["n252_k20", "n504_k20", "n1260_k20", "n2520_k20", "n504_k5", "n504_k100"];
  const lab = (d) => { const c = V2.B.iid_normal[d]; return `$T=${nf(c.n)}$, $k=${c.k}$ ($b=${c.block}$)`; };
  const T = [["spa_c", "SPA$_c$"], ["spa_c_sd", "SPA$_c$-SD"], ["spa_c_boot_t", "SPA$_c$-BT"], ["rc", "RC"]];
  main += table({
    caption: `Size against the number of periods $T$ and strategies $k$ (Null Zoo v2, experiment B, ${nf(V2.B.iid_normal.n504_k20.searches)} null searches per cell; block length $b=\\mathrm{round}(T^{1/3})$).`,
    label: "tab:grid", rows: designs, tight: true, groups: [["IID normal", 4], ["Skew $-1.32$", 4]], rowLabel: lab, head: "Design", colspec: `l${"r".repeat(8)}`,
    cols: [...T.map(([t, l]) => [`iid_normal:${t}`, `${l}`]), ...T.map(([t, l]) => [`skew_negative:${t}`, `${l}`])],
    cell: (d, k) => { const [fam, t] = k.split(":"); const c = V2.B[fam][d]; return sizeCell(c.size[t], c.searches); },
    note: `${BAND_NOTE(V2.B.iid_normal.n504_k20.searches)}`,
  });
}
{
  const T = [["rc", "RC"], ["spa_c_unstud", "SPA$_c$-U"], ["spa_c", "SPA$_c$"], ["spa_u", "SPA$_u$"], ["spa_c_sd", "SPA$_c$-SD"], ["spa_c_boot_t", "SPA$_c$-BT"], ["sidak_t", "LET"]];
  const C = V2.C;
  main += table({
    caption: `Where studentizing and recentring matter (Null Zoo v2, experiment C; i.i.d.\\ normal returns, ${T_OBS} periods, ${N_TRIALS} strategies, ${nf(C.poor_alternatives.searches)} searches per cell). Size / power / size-adjusted power, in percent.`,
    label: "tab:fair", rows: T.map(([t]) => t), rowLabel: (t) => T.find(([k]) => k === t)[1], head: "Test",
    cols: [["heterogeneous_volatility", "Unequal volatilities"], ["poor_alternatives", "Poor alternatives"]],
    cell: (t, d) => `${sizeCell(C[d].size[t], C[d].searches)}\\,/\\,${pct(C[d].power[t])}\\,/\\,${pct(C[d].size_adjusted_power[t])}`,
    note: `Unequal volatilities: volatilities from 0.5 to 2, all Sharpe ratios 0 under the null; under the alternative the least volatile strategy has a Sharpe ratio of 2. Poor alternatives: unit volatility, ${N_TRIALS - 1} strategies with a Sharpe ratio of $-3$; strategy 0 has 0 (null) or 2 (alternative). LET is the \\v{S}id\\'ak-adjusted $t$ test on the best Sharpe ratio.`,
  });
}

// Appendix tables.
let app = "";
app += table({
  caption: `Size of the remaining v1 validators, and of every joint test in the confirmation run v1b (percent, ${nf(nV1)} searches per cell).`,
  label: "tab:size-more", rows: FAMILIES,
  cols: [["haircut_bonferroni:v1", "HC 2s"], ["haircut_bonferroni_lo_one_sided:v1", "HC-Lo"], ["haircut_holm_one_sided:v1", "Holm"], ["oracle_hac_t:v1", "Oracle"], ["spa_upper:v1b", "SPA$_u$ v1b"], ["stepm_any:v1b", "StepM v1b"]],
  cell: (f, k) => { const [v, r] = k.split(":"); return sizeCell((r === "v1" ? V1 : V1B).table[v].size[f], nV1); },
  note: `${BAND_NOTE(nV1)} HC 2s: the haircut with the two-sided $p$-values its authors define, so its nominal one-sided level is 2.5\\%.`,
});
for (const s of ["1", "3"]) {
  app += table({
    caption: `Raw power at a true annualized Sharpe ratio of ${s} (percent), Null Zoo v1.`,
    label: `tab:power-${s}`, rows: FAMILIES,
    cols: [["oracle_hac_t", "Oracle"], ["luck_trials", "LET"], ["luck_trials_lo", "LET-Lo"], ["deflated_sharpe", "DSR"], ["bootstrap_best", "Boot"], ["hac_t_sidak", "HAC-$t$"], ["reality_check", "RC"], ["spa_consistent", "SPA$_c$"]],
    cell: (f, v) => pct(V1.table[v].power[f][s]),
  });
}
app += table({
  caption: `Power (raw / size-adjusted) of the SPA variants at a true annualized Sharpe ratio of 2, Null Zoo v2 experiment A.`,
  label: "tab:mechanism-power", rows: FAMILIES, tight: true, cols: MECH.slice(0, 5),
  cell: (f, t) => `${pct(A[f].power[t])}\\,/\\,${pct(A[f].size_adjusted_power[t])}`,
});
app += table({
  caption: `arch ${V2ARCH.arch} on Null Zoo searches (experiment D, ${nf(V2ARCH.searches_per_family)} null searches per family): size of arch's consistent and upper $p$-values and of this paper's implementation on the same searches, at block length 8 and at arch's default (${V2ARCH.default_block}).`,
  label: "tab:arch", rows: Object.keys(V2.D), tight: true,
  cols: [["arch_b8_consistent", "arch$_c$ (8)"], ["ours_b8_spa_c_unstud", "SPA$_c$-U (8)"], ["arch_b8_upper", "arch$_u$ (8)"], ["ours_b8_rc", "RC (8)"], ["arch_default_consistent", `arch$_c$ (${V2ARCH.default_block})`], ["ours_default_spa_c_unstud", `SPA$_c$-U (${V2ARCH.default_block})`], ["ours_default_spa_c", `SPA$_c$ (${V2ARCH.default_block})`]],
  cell: (f, k) => sizeCell(V2.D[f].size[k], V2.D[f].searches),
  note: `${BAND_NOTE(V2ARCH.searches_per_family)} Searches on which arch's $p$-values differed between \\texttt{studentize=True} and \\texttt{studentize=False}: ${Object.values(V2.D).reduce((a, c) => a + Object.values(c.flag_changes_p).reduce((x, y) => x + y, 0), 0)} of ${nf(Object.values(V2.D).reduce((a, c) => a + c.searches, 0) * 2)} search-and-block pairs.`,
});
app += table({
  caption: `Bootstrap resolution: the same ${nf(BR.reps)} searches per cell scored by the single-series bootstrap with 400 and with 1,999 resamples (v0 design, robustness generator). Size and power in percent.`,
  label: "tab:bootres", rows: BR.rows.map((r) => r.family), tight: true, rowLabel: (f) => FAMILY_LABEL[f],
  cols: [["size:luck_trials", "LET size"], ["size:bootstrap_400", "Boot$_{400}$ size"], ["size:bootstrap_1999", "Boot$_{1999}$ size"], ["power:luck_trials", "LET power"], ["power:bootstrap_400", "Boot$_{400}$ power"], ["power:bootstrap_1999", "Boot$_{1999}$ power"]],
  cell: (f, k) => { const [arm, v] = k.split(":"); const r = BR.rows.find((x) => x.family === f); return pct(r[arm][v] / BR.reps, arm === "size" ? 2 : 1); },
});
{
  const rows = PRED;
  let t = `\\begin{table}[t]\n\\centering\n\\small\n\\caption{The nine predictions registered before the v2 run (analysis/v2/prereg-v2.json, commit \\texttt{${values.get("v2.prereg.commit")}}) and whether each held, as scored by analysis/v2/evaluate\\_v2.py.}\n\\label{tab:predictions}\n\\begin{tabular}{lp{0.72\\linewidth}l}\n\\toprule\nID & Prediction & Outcome \\\\\n\\midrule\n`;
  for (const id of rows) t += `${id} & ${V2.predictions[id].statement.replaceAll("_", "\\_").replaceAll("%", "\\%")} & ${V2.predictions[id].held ? "held" : "\\textbf{did not hold}"} \\\\\n`;
  t += `\\bottomrule\n\\end{tabular}\n\\end{table}\n\n`;
  app += t;
}

// ---- write -------------------------------------------------------------------------------------
let out = `% Generated by analysis/tables.mjs from the result files. Do not edit.\n`;
for (const [k, v] of values) out += `\\nzdef{${k}}{${v}}\n`;
mkdirSync(`${here}paper/generated`, { recursive: true });
writeFileSync(`${here}paper/generated/values.tex`, out);
writeFileSync(`${here}paper/generated/tables.tex`, main);
writeFileSync(`${here}paper/generated/appendix-tables.tex`, app);
console.error(`wrote ${values.size} values and ${main.length + app.length} characters of tables`);
