// Builds every table and every number quoted in the paper from the result files, so no figure in
// the text is typed by hand. The paper reads a number with \nz{key}; LaTeX stops with an error if
// a key is missing, so a number cannot silently go stale or be invented.
//   node analysis/tables.mjs  ->  paper/generated/{values,tables-single,tables-snooping,tables-empirical,appendix-tables}.tex
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
const ARCHMERGED = read("analysis/v2/arch-merged-check.json");
const ANOM = read("analysis/anomalies/anomalies-results.json");
const EXT = read("analysis/null-zoo-extended.json");
const BR = read("analysis/bootstrap-resolution.json");
const EXPLORE = read("analysis/v2/explore-floor-v2.json");
const V2B = read("analysis/v2b/evaluation-v2b.json");
const V2BRUN = read("analysis/v2b/results-v2b.json");
const V2BPRE = read("analysis/v2b/prereg-v2b.json");
const EMP = read("analysis/empirical/empirical-results.json");
const PROV = read("analysis/provenance.json");
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
// Family names for running text: "the correlated family", joined with "and".
const FAMILY_NAME = {
  iid_normal: "i.i.d.\\ normal", student_t4: "Student-$t_4$", skew_negative: "negative-skew", skew_positive: "positive-skew",
  garch: "GARCH", ar1: "AR(1)", regimes: "regime-switching", correlated_trials: "correlated", block_cluster: "cluster",
};
const familyText = (fs) => (fs.length ? `the ${fs.map((f) => FAMILY_NAME[f]).join(" and ")} ${fs.length > 1 ? "families" : "family"}` : "no family");
// One display name per test (v1 validator keys, then the v2/v2b variants).
const V1_LABEL = {
  luck_trials: "LET", luck_trials_lo: "LET-Lo", luck_trials_nonnormal_se: "LET-NN", haircut_bonferroni_one_sided: "HL",
  haircut_bonferroni_lo_one_sided: "HL-Lo", haircut_holm_one_sided: "Holm", deflated_sharpe: "DSR", bootstrap_best: "Boot",
  hac_t_sidak: "HAC-$t$", bootstrap_t_sidak: "SB-$t$", oracle_hac_t: "Oracle", reality_check: "RC", spa_consistent: "SPA$_c$",
  spa_upper: "SPA$_u$", stepm_any: "Step-SPA",
};
const V2_LABEL = {
  rc: "RC", spa_c_unstud: "SPA$_c$-U", spa_c: "SPA$_c$", spa_u: "SPA$_u$", spa_c_sd: "SPA$_c$-SD", spa_c_boot_t: "SPA$_c$-BT",
  spa_c_strict: "SPA$_c$ ($>$)", spa_c_ge: "SPA$_c$", spa_c_boot_t_ge: "SPA$_c$-BT", spa_c_nb_ge: "SPA$_c$-NB", sidak_t: "LET",
  spa_c_b22: "SPA$_c$ (22)", spa_c_unstud_b22: "SPA$_c$-U (22)", spa_c_ge_b22: "SPA$_c$ (22)", rc_b22: "RC (22)",
};

const se = (p, n) => Math.sqrt((p * (1 - p)) / n);
const z99 = normalPpf(0.995);
const band = (n) => z99 * se(LEVEL, n);
const pct = (x, d = 1) => (100 * x).toFixed(d);
// Thousands separators as {,} so that numbers set in math mode do not get a space after the comma.
const nf = (x) => x.toLocaleString("en-US").replaceAll(",", "{,}");
const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const range = (xs, d = 1) => `${pct(Math.min(...xs), d)}--${pct(Math.max(...xs), d)}`;

// ---- values ------------------------------------------------------------------------------------
const values = new Map();
const def = (key, value, same = false) => {
  if (values.has(key) && !(same && values.get(key) === String(value))) throw new Error(`duplicate value ${key}`);
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
def("v2b.base_seed", V2BRUN.base_seed);
def("v2b.reps", nf(V2BRUN.bootstrap_reps));
def("v2b.numpy", V2BRUN.numpy);
def("v2.numpy", V2RUN.numpy);
if (V2BRUN.scale !== 1) throw new Error("results-v2b.json is not the full-size run");

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
    def(`${run}.above_band_nonar.${v}`, familyText(NON_AR.filter((f) => row.size[f] > LEVEL + band(nV1))));
  }
  for (const c of raw.cells) {
    def(`${run}.best_skilled${c.skill}.${c.family}`, pct(c.best_is_skilled_trial / c.searches));
    if (c.skill === 0) def(`${run}.skew_best.${c.family}`, c.mean_skew_of_best.toFixed(2));
  }
  def(`${run}.oracle_power2`, pct(ev.ceiling.power_iid_skill_2));
}
def("v1.validator_count", Object.keys(V1.table).filter((v) => !v.endsWith("_one_sided")).length);
def("v1.validator_count_no_oracle", Object.keys(V1.table).filter((v) => !v.endsWith("_one_sided") && v !== "oracle_hac_t").length);
if (V1B.headline.single_series.test !== V1.headline.single_series.test) throw new Error("v1 and v1b chose different single-series defaults");
def("v1.headline.single", V1_LABEL[V1.headline.single_series.test]);
def("v1.headline.single.fails", familyText(V1.headline.single_series.failures.map((x) => x.family)));
def("v1b.headline.single.fails", familyText(V1B.headline.single_series.failures.map((x) => x.family)));
// The text says the single-series default failed only under negative skew, in both runs.
for (const ev of [V1, V1B]) if (ev.headline.single_series.failures.map((x) => x.family).join() !== "skew_negative") throw new Error("single-series default: failures changed");
// The joint tests' p-value rule, p = #{T* > T} / B with rejection at p <= 0.05: for an exact bootstrap the
// number of resamples above the observed statistic is uniform on 0..B, so the rule rejects with
// probability (floor(0.05 B) + 1) / (B + 1).
def("prule.exact", pct((Math.floor(LEVEL * V1.run.settings.spaReps + 1e-9) + 1) / (V1.run.settings.spaReps + 1), 2));
// The DSR's shortfall against the Sidak test's size-adjusted power, split into threshold and statistic.
{
  const t = V1.table;
  const letAdj = t.luck_trials.size_adjusted_power_skill_2.iid_normal;
  const dsrAdj = t.deflated_sharpe.size_adjusted_power_skill_2.iid_normal;
  const dsrRaw = t.deflated_sharpe.power.iid_normal["2"];
  def("v1.dsr_gap.total", pct(letAdj - dsrRaw));
  def("v1.dsr_gap.threshold", pct(dsrAdj - dsrRaw));
  def("v1.dsr_gap.statistic", pct(letAdj - dsrAdj));
}
def("v1b.headline.joint.fails", familyText(V1B.headline.with_variants.failures.map((x) => x.family)));
// Both runs, joint tests: the largest and smallest size over the two runs.
for (const v of ["spa_consistent", "spa_upper", "stepm_any", "reality_check"]) {
  const xs = FAMILIES.flatMap((f) => [V1.table[v].size[f], V1B.table[v].size[f]]);
  def(`both.size_range.${v}`, range(xs));
  def(`both.size_max.${v}`, pct(Math.max(...xs)));
  def(`both.size_range_nonar.${v}`, range(NON_AR.flatMap((f) => [V1.table[v].size[f], V1B.table[v].size[f]])));
}
// StepM and SPA_c: largest difference in size across the two runs and nine families.
def("both.fails6_min.spa_consistent", WORDS[Math.min(...[V1, V1B].map((ev) => FAMILIES.filter((f) => ev.table.spa_consistent.size[f] > 0.06).length))]);
// "the Reality Check exceeded 6% only under autocorrelation", in both runs
for (const ev of [V1, V1B]) if (FAMILIES.filter((f) => ev.table.reality_check.size[f] > 0.06).join() !== "ar1") throw new Error("RC above 6% outside AR(1)");
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
// Unregistered check: arch's own code after pull request 871 was merged (studentize=True, the default).
def("archm.version", ARCHMERGED.arch);
def("archm.commit", ARCHMERGED.arch_merge_commit.slice(0, 7));
def("archm.searches", nf(ARCHMERGED.searches_per_family));
for (const [fam, cell] of Object.entries(ARCHMERGED.families)) for (const [k, x] of Object.entries(cell.rejections_at_5pct)) def(`archm.size.${k}.${fam}`, pct(x / cell.searches));
// The text says arch's p-values were identical under both flag values on every pair.
if (values.get("v2.D.total_flag_changes") !== "0") throw new Error("arch: the studentize flag changed some p-values");
const PRED = Object.keys(V2.predictions).sort();
for (const id of PRED) def(`v2.${id}.held`, V2.predictions[id].held ? "held" : "did not hold");
def("v2.predictions.held_count", PRED.filter((id) => V2.predictions[id].held).length);
def("v2.predictions.count", PRED.length);
def("v2.total_searches", nf(V2RUN.cells.reduce((a, c) => a + c.searches, 0)));
def("v2.D.total_pairs", nf(2 * Object.values(V2.D).reduce((a, c) => a + c.searches, 0)));
{
  const diffs = Object.values(V2.D).flatMap((c) => Object.values(c.mean_abs_p_difference_arch_vs_ours).flatMap((k) => Object.values(k)));
  def("v2.D.meanabs_range", `${Math.min(...diffs).toFixed(3)}--${Math.max(...diffs).toFixed(3)}`);
  // Expected mean absolute difference between two independent bootstrap p-values of B resamples each
  // when p is uniform: E|p1 - p2| = sqrt(2/pi) sqrt(2 p (1 - p) / B), averaged over p (E sqrt(p(1-p)) = pi/8).
  def("v2.D.meanabs_expected", (Math.sqrt(2 / Math.PI) * Math.sqrt(2 / V2RUN.bootstrap_reps) * Math.PI / 8).toFixed(3));
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
for (const t of ["spa_c_boot_t", "rc"]) {
  const cellsB = Object.values(V2.B).flatMap((designs) => Object.values(designs));
  def(`v2.B.range.${t}`, range(cellsB.map((c) => c.size[t])));
  def(`v2.B.above_band.${t}`, cellsB.filter((c) => c.size[t] > LEVEL + band(c.searches)).length);
  def(`v2.B.outside_band.${t}`, cellsB.filter((c) => Math.abs(c.size[t] - LEVEL) > band(c.searches)).length);
  def(`v2.B.cells`, cellsB.length, true);
}
def("v2.P1.maxdiff", pct(Math.max(...V2.predictions.P1.detail.map((r) => Math.abs(r.difference))), 2));

// v2b: number of strategies and block length (E), block re-studentization in the nine families (F),
// where the skilled strategy sits among unequal volatilities and poor alternatives with ties counted (C2).
const E = V2B.E;
const E_K = [1, 2, 5, 10, 20, 50, 100];
const E_B = [1, 8, 22];
for (const k of E_K) for (const b of E_B) for (const [t, x] of Object.entries(E[`k${k}`][`b${b}`].size)) def(`v2b.E.size.${t}.k${k}.b${b}`, pct(x));
def("v2b.E.searches", nf(E.k1.b1.searches));
{
  const cellsE = E_K.flatMap((k) => E_B.map((b) => E[`k${k}`][`b${b}`]));
  def("v2b.E.rc_unstud_maxdiff", pct(Math.max(...cellsE.map((c) => Math.abs(c.size.rc - c.size.spa_c_unstud))), 1));
  for (const t of ["rc", "spa_c_boot_t_ge", "spa_c_nb_ge", "spa_c_ge"]) {
    for (const b of E_B) def(`v2b.E.range.${t}.b${b}`, range(E_K.map((k) => E[`k${k}`][`b${b}`].size[t])));
    def(`v2b.E.range_k20plus.${t}.b8`, range([20, 50, 100].map((k) => E[`k${k}`].b8.size[t])));
  }
}
const F = V2B.F;
for (const f of FAMILIES) {
  for (const [t, x] of Object.entries(F[f].size)) def(`v2b.F.size.${t}.${f}`, pct(x));
  for (const [t, x] of Object.entries(F[f].power)) def(`v2b.F.power.${t}.${f}`, pct(x));
  for (const [t, x] of Object.entries(F[f].size_adjusted_power)) def(`v2b.F.adj.${t}.${f}`, pct(x));
}
for (const t of Object.keys(F.iid_normal.size)) {
  def(`v2b.F.size_range.${t}`, range(FAMILIES.map((f) => F[f].size[t])));
  def(`v2b.F.size_range_nonar.${t}`, range(NON_AR.map((f) => F[f].size[t])));
  def(`v2b.F.fails65_nonar.${t}`, NON_AR.filter((f) => F[f].size[t] > 0.065).length);
  def(`v2b.F.outside_band.${t}`, FAMILIES.filter((f) => Math.abs(F[f].size[t] - LEVEL) > band(F[f].searches_null)).length);
}
def("v2b.F.searches_null", nf(F.iid_normal.searches_null));
for (const t of Object.keys(F.iid_normal.size)) {
  const n = NON_AR.filter((f) => F[f].size[t] < LEVEL - band(F[f].searches_null)).length;
  def(`v2b.F.below_band_nonar.${t}`, n);
  def(`v2b.F.below_band_nonar_text.${t}`, n === NON_AR.length ? `all ${WORDS[n]}` : `${WORDS[n]} of the ${WORDS[NON_AR.length]}`);
}
// Experiment E: ranges used in the text, and checks behind its typed wording.
for (const t of ["rc", "spa_c_boot_t_ge"]) {
  def(`v2b.E.range_k20minus.${t}.b22`, range([1, 2, 5, 10, 20].map((k) => E[`k${k}`].b22.size[t])));
  // "within the band at block length 8 for every number of strategies"
  if (E_K.some((k) => Math.abs(E[`k${k}`].b8.size[t] - LEVEL) > band(E[`k${k}`].b8.searches))) throw new Error(`E: ${t} left the band at block 8`);
  // "liberal at block length 22 with up to 20 strategies"
  if ([1, 2, 5, 10, 20].some((k) => E[`k${k}`].b22.size[t] <= LEVEL + band(E[`k${k}`].b22.searches))) throw new Error(`E: ${t} not above the band at block 22`);
}
// The Reality Check and the bootstrap-t in the correlated and cluster families, across v1, v1b, v2 and v2b.
{
  const xs = [];
  for (const f of ["correlated_trials", "block_cluster"]) {
    xs.push(V1.table.reality_check.size[f], V1B.table.reality_check.size[f], V2.A[f].size.rc, V2.A[f].size.spa_c_boot_t, F[f].size.rc, F[f].size.spa_c_boot_t_ge);
  }
  def("corrclus.range", range(xs));
}
def("v2b.P12.diff", pct(V2B.predictions.P12.detail.rc_k1_b8 - V2B.predictions.P12.detail.rc_k100_b8, 2));
def("v2b.P12.shortfall", pct(0.005 - (V2B.predictions.P12.detail.rc_k1_b8 - V2B.predictions.P12.detail.rc_k100_b8), 2));
def("v2b.F.searches_skill", nf(F.iid_normal.searches_skill));
const C2 = V2B.C2;
for (const [t, x] of Object.entries(C2.het_null.size)) def(`v2b.C2.size.${t}.het_null`, pct(x));
for (const where of ["low", "median", "high"]) {
  const c = C2[`het_skill_${where}`];
  for (const [t, x] of Object.entries(c.power)) def(`v2b.C2.power.${t}.${where}`, pct(x));
  for (const [t, x] of Object.entries(c.size_adjusted_power)) def(`v2b.C2.adj.${t}.${where}`, pct(x));
  def(`v2b.C2.best_mean.${where}`, pct(c.best_mean_is_skilled));
  def(`v2b.C2.best_t.${where}`, pct(c.best_t_is_skilled));
}
for (const [t, x] of Object.entries(C2.poor.size)) def(`v2b.C2.size.${t}.poor`, pct(x));
for (const [t, x] of Object.entries(C2.poor.power)) def(`v2b.C2.power.${t}.poor`, pct(x));
for (const [t, x] of Object.entries(C2.poor.size_adjusted_power)) def(`v2b.C2.adj.${t}.poor`, pct(x));
def("v2b.C2.searches", nf(C2.het_null.searches));
const PRED_B = Object.keys(V2B.predictions).sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
for (const id of PRED_B) def(`v2b.${id}.held`, V2B.predictions[id].held ? "held" : "did not hold");
def("v2b.predictions.held_count", PRED_B.filter((id) => V2B.predictions[id].held).length);
def("v2b.predictions.count", PRED_B.length);
if (PRED_B.length !== V2BPRE.predictions.length) throw new Error("v2b evaluation does not score every registered prediction");
def("prereg.total", PRED.length + PRED_B.length);
def("prereg.held", PRED.filter((id) => V2.predictions[id].held).length + PRED_B.filter((id) => V2B.predictions[id].held).length);
def("v2b.total_searches", nf(V2BRUN.cells.reduce((a, c) => a + c.searches, 0)));
def("v2b.cells", V2BRUN.cells.length);

// The empirical illustration (v2b, part G).
const EMP_TESTS = ["rc", "spa_c_unstud", "spa_c_ge", "spa_c_boot_t_ge", "spa_c_nb_ge", "spa_c_unstud_b22", "spa_c_ge_b22", "sidak_t"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const ym = (d) => `${MONTHS[Number(d.slice(4, 6)) - 1]} ${d.slice(0, 4)}`;
const ymd = (d) => `${Number(d.slice(6, 8))} ${["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"][Number(d.slice(4, 6)) - 1]} ${d.slice(0, 4)}`;
const span = (label) => { const [a, b] = label.split(" ")[0].split("-"); return `${ym(a)}--${ym(b)}`; };
const EW = EMP.windows.filter((w) => w.p !== null);
if (EW.length !== EMP.windows.length) throw new Error("an empirical window has no p-values");
def("emp.windows", EW.length);
def("emp.window_len", EMP.window);
def("emp.rules", EMP.rules.lengths.length);
def("emp.rule_min", EMP.rules.lengths[0]);
def("emp.rule_max", EMP.rules.lengths.at(-1));
def("emp.reps", nf(EMP.bootstrap_reps));
def("emp.seed", EMP.seed);
def("emp.first_day", ymd(EMP.full_sample.label.split("-")[0]));
def("emp.last_day", ymd(EMP.full_sample.label.split(" ")[0].split("-")[1]));
def("emp.first_window", span(EW[0].label));
def("emp.last_window", span(EW.at(-1).label));
def("emp.full.periods", nf(EMP.full_sample.periods));
def("emp.full.block", EMP.full_sample.block);
def("emp.reduced_windows", EW.filter((w) => w.strategies_used < EMP.rules.lengths.length).length);
def("emp.min_strategies", Math.min(...EW.map((w) => w.strategies_used)));
for (const t of [...EMP_TESTS, "spa_c_strict"]) {
  def(`emp.reject.${t}`, EW.filter((w) => w.p[t] <= LEVEL).length);
  def(`emp.full.p.${t}`, EMP.full_sample.p[t].toFixed(3));
}
def("emp.any_reject", EW.filter((w) => EMP_TESTS.some((t) => w.p[t] <= LEVEL)).length);
def("emp.all_reject", EW.filter((w) => EMP_TESTS.every((t) => w.p[t] <= LEVEL)).length);
def("emp.joint_any_reject", EW.filter((w) => EMP_TESTS.filter((t) => t !== "sidak_t").some((t) => w.p[t] <= LEVEL)).length);
def("emp.split_windows", EW.filter((w) => { const r = EMP_TESTS.map((t) => w.p[t] <= LEVEL); return r.some(Boolean) && !r.every(Boolean); }).length);
def("emp.strict_equals_ge", EW.every((w) => (w.p.spa_c_strict <= LEVEL) === (w.p.spa_c_ge <= LEVEL)) ? "yes" : "no");
for (const [pair, d] of Object.entries(EMP.summary.disagreements)) def(`emp.disagree.${pair.replace("|", "-")}`, d.only_first + d.only_second);
// Windows in which test a rejects and test b does not, with the two p-values, for the text.
for (const a of [...EMP_TESTS, "spa_c_strict"]) {
  for (const b of [...EMP_TESTS, "spa_c_strict"]) {
    if (a === b) continue;
    const ws = EW.filter((w) => w.p[a] <= LEVEL && w.p[b] > LEVEL);
    def(`emp.only.${a}.not.${b}`, ws.length);
    def(`emp.only.${a}.not.${b}.list`, ws.length ? ws.map((w) => `${span(w.label)} (${w.p[a].toFixed(3)} against ${w.p[b].toFixed(3)})`).join(" and ") : "none");
  }
}
// Windows in which the joint tests at block length 8 all agree, and the range of their full-sample p-values.
{
  const joint = EMP_TESTS.filter((t) => t !== "sidak_t").map((t) => EMP.full_sample.p[t]);
  def("emp.full.p.min_joint", Math.min(...joint).toFixed(3));
  def("emp.full.p.max_joint", Math.max(...joint).toFixed(3));
  def("emp.full.joint_count", WORDS[joint.length]);
}
def("emp.fixed_alone", WORDS[EW.filter((w) => w.p.spa_c_ge <= LEVEL && ["rc", "spa_c_boot_t_ge", "spa_c_nb_ge"].every((t) => w.p[t] > LEVEL)).length]);
def("emp.none_reject", EW.filter((w) => EMP_TESTS.every((t) => w.p[t] > LEVEL)).length);

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
// nu = 2 / v degrees of freedom. Because the autocovariances are computed from demeaned data, w^2 is
// also biased downward: E[w^2] / sigma^2 = (T - 1) / T - 2 sum_i kappa_i (T - i) / T^2. The bootstrap
// holds w fixed, so its critical value is close to the normal Sidak value z*, and its conditional
// variance of a mean is w^2 itself, so it cannot reproduce the bias either. The predicted size of the
// maximum over N independent strategies is then 1 - (1 - P(t_nu > z* sqrt(E[w^2] / sigma^2)))^N. It
// ignores the variability of the bootstrap's own critical value.
const zStar = normalPpf(1 - sidakP1);
def("approx.z_star", zStar.toFixed(2));
for (const b of [1, 8, 22]) {
  const p = 1 / b;
  let s2 = 0;
  let s1 = 0;
  for (let i = 1; i < T_OBS; i++) {
    const kappa = (1 - i / T_OBS) * (1 - p) ** i + (i / T_OBS) * (1 - p) ** (T_OBS - i);
    s2 += kappa * kappa * (T_OBS - i) / T_OBS;
    s1 += kappa * (T_OBS - i);
  }
  const v = (2 + 4 * s2) / T_OBS;
  const nu = 2 / v;
  const ratio = (T_OBS - 1) / T_OBS - 2 * s1 / T_OBS ** 2;
  def(`approx.nu.b${b}`, Math.round(nu));
  def(`approx.cv_w.b${b}`, pct(Math.sqrt(v) / 2, 1));
  def(`approx.bias.b${b}`, pct(1 - ratio, 1));
  const predicted = 1 - (1 - studentTUpper(zStar * Math.sqrt(ratio), nu)) ** N_TRIALS;
  const noBias = 1 - (1 - studentTUpper(zStar, nu)) ** N_TRIALS;
  def(`approx.size.b${b}`, pct(predicted));
  def(`approx.size_nobias.b${b}`, pct(noBias));
  // Share of the measured excess over 5% (experiment A, i.i.d. normal) that the approximation accounts for.
  const measured = { 1: V2.A.iid_normal.size.spa_c_sd, 8: V2.A.iid_normal.size.spa_c, 22: V2.A.iid_normal.size.spa_c_b22 }[b];
  def(`approx.share.b${b}`, pct((predicted - LEVEL) / (measured - LEVEL), 0));
  def(`approx.share_nobias.b${b}`, pct((noBias - LEVEL) / (measured - LEVEL), 0));
}

// How often, under v1's null (every mean zero), every strategy falls below the consistent threshold, so
// that a floored statistic with a strict inequality could reject: a normal approximation to the
// t-statistics, equicorrelated within a block. Worst case over the v1 families.
{
  const c = -Math.sqrt(2 * Math.log(Math.log(T_OBS)));
  const allBelow = (rho, k) => {
    let s = 0;
    const h = 1e-3;
    for (let z = -12; z <= 12; z += h) s += normalCdf((c - Math.sqrt(rho) * z) / Math.sqrt(1 - rho)) ** k * Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI) * h;
    return s;
  };
  const worst = Math.max(normalCdf(c) ** N_TRIALS, allBelow(0.5, N_TRIALS), allBelow(0.8, 5) ** 4);
  def("ties.p_all_below.max_per_100000", (worst * 1e5).toFixed(1));
  def("ties.p_below.k1", pct(normalCdf(c), 1));
}

// Provenance.
def("v0.sha", createHash("sha256").update(readFileSync(V0PATH)).digest("hex"));
def("src.commit.v0", readFileSync(`${here}src/SOURCE_COMMIT`, "utf8").trim().slice(0, 12));
def("src.commit.v1", readFileSync(`${here}src/SOURCE_COMMIT_V1`, "utf8").trim().slice(0, 12));
const git = (...a) => execFileSync("git", ["-C", here, ...a], { encoding: "utf8" }).trim();
def("v2.prereg.commit", git("log", "--diff-filter=A", "--format=%H", "--", "analysis/v2/prereg-v2.json").split("\n").pop().slice(0, 12));
def("v2b.prereg.commit", git("log", "--diff-filter=A", "--format=%H", "--", "analysis/v2b/prereg-v2b.json").split("\n").pop().slice(0, 12));
for (const run of ["v2", "v2b"]) if (!PROV[run].prereg_commit.startsWith(values.get(`${run}.prereg.commit`))) throw new Error(`provenance.json disagrees with git on the ${run} pre-registration`);
for (const run of ["v1", "v1b", "v2", "v2b"]) for (const [k, x] of Object.entries(PROV[run])) if (k !== "repository") def(`prov.${run}.${k}`, x);
for (const [run, text] of Object.entries(PROV.reruns)) {
  def(`prov.rerun.${run}.sha`, text.match(/[0-9a-f]{64}/)[0]);
  def(`prov.rerun.${run}.workers`, text.match(/with (\d+) worker/)[1]);
  def(`prov.rerun.${run}.published_workers`, text.match(/run used (\d+)\)/)[1]);
}
def("emp.zip_sha", EMP.data.zip_sha256);
def("emp.downloaded", EMP.data.downloaded_utc);
def("v1.prereg.candidates", V1PRE.candidates.with_variants.join(", ").replaceAll("_", "\\_"));

// ---- tables ------------------------------------------------------------------------------------
function sizeCell(p, n, d = 1) {
  const s = pct(p, d);
  if (p > LEVEL + band(n)) return `\\textbf{${s}}`;
  if (p < LEVEL - band(n)) return `\\textit{${s}}`;
  return s;
}
const groupHeader = (groups, first = 2) => {
  let c = first;
  const rules = groups.map(([label, span]) => { const r = label ? `\\cmidrule(lr){${c}-${c + span - 1}}` : ""; c += span; return r; }).join("");
  return ` & ${groups.map(([label, span]) => `\\multicolumn{${span}}{c}{${label}}`).join(" & ")} \\\\\n${rules}\n`;
};
function tabular({ rows, cols, cell, rowLabel = (r) => FAMILY_LABEL[r], head = "Family", colspec, groups }) {
  let t = `\\begin{tabular}{${colspec ?? `l${"r".repeat(cols.length)}`}}\n\\toprule\n`;
  if (groups) t += groupHeader(groups);
  t += `${head} & ${cols.map(([, l]) => l).join(" & ")} \\\\\n\\midrule\n`;
  for (const r of rows) t += `${rowLabel(r)} & ${cols.map(([k]) => cell(r, k)).join(" & ")} \\\\\n`;
  return `${t}\\bottomrule\n\\end{tabular}\n`;
}
const tableNote = (note) => (note ? `\\par\\smallskip\\begin{minipage}{0.97\\linewidth}\\footnotesize ${note}\\end{minipage}\n` : "");
function table({ caption, label, note, placement = "t", tight = false, ...spec }) {
  let t = `\\begin{table}[${placement}]\n\\centering\n${tight ? "\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}" : "\\small"}\n\\caption{${caption}}\n\\label{${label}}\n`;
  t += tabular(spec);
  return `${t}${tableNote(note)}\\end{table}\n\n`;
}
// Two stacked panels sharing one caption.
function panels({ caption, label, note, placement = "t", tight = false, parts }) {
  let t = `\\begin{table}[${placement}]\n\\centering\n${tight ? "\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}" : "\\small"}\n\\caption{${caption}}\n\\label{${label}}\n`;
  t += parts.map(({ title, ...spec }) => `\\par\\textit{${title}}\\par\\smallskip\n${tabular(spec)}\\par\n`).join("\\medskip\n");
  return `${t}${tableNote(note)}\\end{table}\n\n`;
}
const BAND_NOTE = (n) => `Bold: above the 99\\% Monte Carlo band around 5\\% (${pct(LEVEL - band(n), 2)}--${pct(LEVEL + band(n), 2)} for ${nf(n)} searches); italic: below it.`;
const lbl = (keys, labels) => keys.map((k) => [k, labels[k]]);

const SINGLE = lbl(["luck_trials", "luck_trials_lo", "luck_trials_nonnormal_se", "haircut_bonferroni_one_sided", "deflated_sharpe", "bootstrap_best", "hac_t_sidak", "bootstrap_t_sidak"], V1_LABEL);
const POWER = lbl(["oracle_hac_t", "luck_trials", "luck_trials_lo", "deflated_sharpe", "bootstrap_best", "hac_t_sidak", "reality_check", "spa_consistent"], V1_LABEL);
const MECH = lbl(["spa_c", "spa_c_sd", "spa_c_boot_t", "rc", "spa_c_unstud", "spa_c_b22", "spa_c_unstud_b22"], V2_LABEL);

// Section 3: single-series corrections.
let single = "";
single += table({
  caption: `Size of the single-series corrections: percentage of skill-less searches (${N_TRIALS} strategies, ${T_OBS} daily returns) whose best strategy is called skilled at the nominal 5\\% level. Null Zoo v1, ${nf(nV1)} searches per cell.`,
  label: "tab:size-single", rows: FAMILIES, cols: SINGLE,
  cell: (f, v) => sizeCell(V1.table[v].size[f], nV1),
  note: `${BAND_NOTE(nV1)} HL is the Harvey--Liu haircut read one-sided (two-sided $p$ halved); Boot uses ${nf(V1.run.settings.bootstrapBestDraws)} resamples and SB-$t$ ${nf(V1.run.settings.bootstrapTReps)}.`,
});
single += panels({
  caption: `Power when one strategy has a true annualized Sharpe ratio of 2, Null Zoo v1 (percent). The oracle knows which strategy is skilled and tests it alone; its power is a natural benchmark, and no best-of-${N_TRIALS} test in the table exceeds it.`,
  label: "tab:power",
  parts: [
    { title: "Raw power", rows: FAMILIES, cols: POWER, cell: (f, v) => pct(V1.table[v].power[f]["2"]) },
    { title: "Size-adjusted power", rows: FAMILIES, cols: POWER, cell: (f, v) => adjText(V1, V1RAW, v, f) },
  ],
  note: `Size-adjusted power is power at the cutoff at which the same test rejects at most 5\\% of the matching null cell; a user who does not know the return family cannot apply it, but it shows how well each statistic ranks skilled searches above skill-less ones. n/a: the test's smallest attainable $p$-value already occurs in more than 5\\% of the null cell, so no cutoff attains the level. The Monte Carlo standard error of raw power is at most ${pct(se(0.5, nV1), 1)} points. Size-adjusted power also carries the error of the estimated cutoff, and the joint tests' $p$-values lie on a grid of $1/${nf(V1.run.settings.spaReps)}$, so their achieved null rejection rate at the cutoff can be below 5\\%.`,
});

// Section 4: data-snooping tests.
let snoop = "";
snoop += table({
  caption: `Size of the joint resampling tests in Null Zoo v1 and in its confirmation run v1b (fresh seeds), ${nf(nV1)} searches per cell each, ${nf(V1.run.settings.spaReps)} stationary-bootstrap resamples with mean block length 8.`,
  label: "tab:size-joint", rows: FAMILIES,
  cols: [["reality_check:v1", "RC v1"], ["reality_check:v1b", "RC v1b"], ["spa_consistent:v1", "SPA$_c$ v1"], ["spa_consistent:v1b", "SPA$_c$ v1b"], ["spa_upper:v1", "SPA$_u$ v1"], ["stepm_any:v1", "Step-SPA v1"]],
  cell: (f, k) => { const [v, r] = k.split(":"); return sizeCell((r === "v1" ? V1 : V1B).table[v].size[f], nV1); },
  note: `${BAND_NOTE(nV1)} Step-SPA is scored as a decision (at least one strategy declared superior at 5\\%).`,
});
snoop += table({
  caption: `Why the studentized SPA test is oversized: size of variants of the consistent SPA test on the same searches (Null Zoo v2, experiment A; second implementation, ${nf(A.iid_normal.searches)} searches per cell, mean block length 8 unless marked 22).`,
  label: "tab:mechanism", rows: FAMILIES, cols: MECH, tight: true,
  cell: (f, t) => sizeCell(A[f].size[t], A[f].searches),
  note: `${BAND_NOTE(A.iid_normal.searches)} SPA$_c$: Hansen's statistic studentized by a fixed Politis--Romano long-run variance. SPA$_c$-SD: studentized by the sample standard deviation instead. SPA$_c$-BT: every resample re-studentized by its own standard deviation (bootstrap-$t$). SPA$_c$-U: neither studentized nor floored at zero, the computation arch~${V2ARCH.arch} performs. (22): mean block length 22, arch's default for ${T_OBS} periods.`,
});
{
  const designs = ["n252_k20", "n504_k20", "n1260_k20", "n2520_k20", "n504_k5", "n504_k100"];
  const lab = (d) => { const c = V2.B.iid_normal[d]; return `$T=${nf(c.n)}$, $k=${c.k}$ ($b=${c.block}$)`; };
  const T = lbl(["spa_c", "spa_c_sd", "spa_c_boot_t", "rc"], V2_LABEL);
  snoop += table({
    caption: `Size against the number of periods $T$ and strategies $k$ (Null Zoo v2, experiment B, ${nf(V2.B.iid_normal.n504_k20.searches)} null searches per cell; mean block length $b=\\mathrm{round}(T^{1/3})$).`,
    label: "tab:grid", rows: designs, tight: true, groups: [["IID normal", 4], ["Skew $-1.32$", 4]], rowLabel: lab, head: "Design", colspec: `l${"r".repeat(8)}`,
    cols: [...T.map(([t, l]) => [`iid_normal:${t}`, l]), ...T.map(([t, l]) => [`skew_negative:${t}`, l])],
    cell: (d, k) => { const [fam, t] = k.split(":"); const c = V2.B[fam][d]; return sizeCell(c.size[t], c.searches); },
    note: `${BAND_NOTE(V2.B.iid_normal.n504_k20.searches)}`,
  });
}
let supp = "";
supp += table({
  caption: `Size against the number of strategies $k$ and the mean block length (1, 8 or 22), Null Zoo v2b, experiment E: i.i.d.\\ normal returns, $T=${T_OBS}$, ${nf(E.k1.b1.searches)} null searches per cell.`,
  label: "tab:k-block", rows: E_K, tight: true, head: "", rowLabel: (k) => `$k=${k}$`,
  groups: [["RC", 3], ["SPA$_c$", 3], ["SPA$_c$-BT", 3], ["SPA$_c$-NB", 3]],
  cols: ["rc", "spa_c_ge", "spa_c_boot_t_ge", "spa_c_nb_ge"].flatMap((t) => E_B.map((b) => [`${t}:${b}`, `${b}`])),
  cell: (k, key) => { const [t, b] = key.split(":"); const c = E[`k${k}`][`b${b}`]; return sizeCell(c.size[t], c.searches); },
  note: `${BAND_NOTE(E.k1.b1.searches)} Floored tests count ties. At block length 1 the Politis--Romano variance is the sample variance, so SPA$_c$ is then SPA$_c$-SD. SPA$_c$-NB re-studentizes every resample by its natural block variance. The Reality Check and SPA$_c$-U differ by at most ${values.get("v2b.E.rc_unstud_maxdiff")} points in any cell; SPA$_c$-U and the strict-inequality SPA$_c$ are in Table~\\ref{tab:k-block-more}.`,
});
supp += table({
  caption: `Re-studentizing every resample, by its own standard deviation (SPA$_c$-BT) or by its natural block variance (SPA$_c$-NB), in the nine families: Null Zoo v2b, experiment F (mean block length 8; ${nf(F.iid_normal.searches_null)} null and ${nf(F.iid_normal.searches_skill)} skill searches per family; size and size-adjusted power at a Sharpe ratio of 2, percent).`,
  label: "tab:block-student", rows: FAMILIES, tight: true, groups: [["Size", 4], ["Size-adjusted power", 3]], colspec: `l${"r".repeat(7)}`,
  cols: [["s:rc", "RC"], ["s:spa_c_ge", "SPA$_c$"], ["s:spa_c_boot_t_ge", "SPA$_c$-BT"], ["s:spa_c_nb_ge", "SPA$_c$-NB"], ["a:rc", "RC"], ["a:spa_c_boot_t_ge", "SPA$_c$-BT"], ["a:spa_c_nb_ge", "SPA$_c$-NB"]],
  cell: (f, key) => { const [m, t] = key.split(":"); return m === "s" ? sizeCell(F[f].size[t], F[f].searches_null) : pct(F[f].size_adjusted_power[t]); },
  note: `${BAND_NOTE(F.iid_normal.searches_null)} Floored tests count ties. Raw power and the other variants are in Table~\\ref{tab:block-student-power}.`,
});
{
  const T = lbl(["rc", "spa_c_unstud", "spa_c_strict", "spa_c_ge", "spa_c_boot_t_ge", "spa_c_nb_ge", "sidak_t"], V2_LABEL);
  const pw = (c, t) => `${pct(c.power[t])}\\,/\\,${pct(c.size_adjusted_power[t])}`;
  supp += table({
    caption: `Where studentizing and recentring matter, with ties counted (Null Zoo v2b, experiment C2; i.i.d.\\ normal returns, ${T_OBS} periods, ${N_TRIALS} strategies, ${nf(C2.het_null.searches)} searches per cell; percent).`,
    label: "tab:fair2", rows: T.map(([t]) => t), rowLabel: (t) => T.find(([k]) => k === t)[1], head: "Test", tight: true,
    groups: [["Unequal volatilities", 4], ["Poor alternatives", 2]], colspec: "lrrrrrr",
    cols: [["het_null", "Null"], ["low", "Skill low"], ["median", "Skill median"], ["high", "Skill high"], ["poor_size", "Null"], ["poor_power", "Skill"]],
    cell: (t, d) => {
      if (d === "het_null") return sizeCell(C2.het_null.size[t], C2.het_null.searches);
      if (d === "poor_size") return sizeCell(C2.poor.size[t], C2.poor.searches);
      if (d === "poor_power") return pw(C2.poor, t);
      return pw(C2[`het_skill_${d}`], t);
    },
    note: `${BAND_NOTE(C2.het_null.searches)} Null columns: size; skill columns: raw\\,/\\,size-adjusted power. Unequal volatilities: $0.5\\cdot4^{j/19}$ for strategy $j=0,\\dots,19$ (0.5 to 2), all Sharpe ratios 0 under the null; under the alternative strategy 0 (low, the least volatile), 10 (median) or 19 (high, the most volatile) has a Sharpe ratio of 2. Poor alternatives: unit volatility, ${N_TRIALS - 1} strategies with a Sharpe ratio of $-3$, strategy 0 at 0 (null) or 2. SPA$_c$ ($>$) uses Hansen's strict inequality; the other floored tests count ties. LET is the \\v{S}id\\'ak-adjusted $t$ test on the best Sharpe ratio.`,
  });
}

// Section 5: the empirical illustration.
const EMP_LABEL = { rc: "RC", spa_c_unstud: "SPA$_c$-U", spa_c_ge: "SPA$_c$", spa_c_boot_t_ge: "SPA$_c$-BT", spa_c_nb_ge: "SPA$_c$-NB", spa_c_unstud_b22: "SPA$_c$-U", spa_c_ge_b22: "SPA$_c$", sidak_t: "LET" };
const pCell = (p) => (p <= LEVEL ? `\\textbf{${p.toFixed(3)}}` : p.toFixed(3));
function empTable({ caption, label, windows, placement = "t", note, summary, font = "\\footnotesize" }) {
  let t = `\\begin{table}[${placement}]\n\\centering\n${font}\n\\setlength{\\tabcolsep}{3.5pt}\n\\caption{${caption}}\n\\label{${label}}\n`;
  t += `\\begin{tabular}{l${"r".repeat(EMP_TESTS.length)}}\n\\toprule\n`;
  t += groupHeader([["Mean block length 8", 5], ["22", 2], ["", 1]]);
  t += `Window & ${EMP_TESTS.map((k) => EMP_LABEL[k]).join(" & ")} \\\\\n\\midrule\n`;
  for (const w of windows) t += `${span(w.label)}${w.strategies_used < EMP.rules.lengths.length ? `$^{${w.strategies_used}}$` : ""} & ${EMP_TESTS.map((k) => pCell(w.p[k])).join(" & ")} \\\\\n`;
  if (summary) {
    t += `\\midrule\nWindows rejected & ${EMP_TESTS.map((k) => EW.filter((w) => w.p[k] <= LEVEL).length).join(" & ")} \\\\\n`;
    t += `Full sample$^{\\dagger}$ & ${EMP_TESTS.map((k) => pCell(EMP.full_sample.p[k])).join(" & ")} \\\\\n`;
  }
  t += `\\bottomrule\n\\end{tabular}\n`;
  return `${t}${tableNote(note)}\\end{table}\n\n`;
}
const EMP_NOTE = `$p$-values with ${nf(EMP.bootstrap_reps)} stationary-bootstrap resamples; bold: at most 0.05. LET is the \\v{S}id\\'ak-adjusted $t$ test on the best rule's Sharpe ratio, with no resampling. Floored tests count ties. A superscript gives the number of rules used when some rules never left the market in a window and so had no excess return.`;
let empirical = empTable({
  caption: `Twenty moving-average timing rules against buy-and-hold on the US market, in the ${EW.filter((w) => EMP_TESTS.some((t) => w.p[t] <= LEVEL)).length} of ${EW.length} non-overlapping two-year windows in which at least one test rejects at 5\\% (Null Zoo v2b, part G; $p$-values).`,
  label: "tab:empirical", windows: EW.filter((w) => EMP_TESTS.some((t) => w.p[t] <= LEVEL)), summary: true,
  note: `${EMP_NOTE} Windows rejected: of all ${EW.length} windows. $^{\\dagger}$${values.get("emp.first_day")} to ${values.get("emp.last_day")}, ${nf(EMP.full_sample.periods)} days, mean block length ${EMP.full_sample.block} for the first five tests. Every window is in Table~\\ref{tab:empirical-all}.`,
});

// Anomalies illustration (analysis/anomalies, pre-registered): stepwise tests on published predictors.
const ANOM_TESTS = [["fixed_q10", "Fixed, $q=10$"], ["fixed", "Fixed, $q=5$"], ["boot_t", "Bootstrap-$t$"], ["rc", "Reality Check"]];
const ANOM_TOT = Object.fromEntries(ANOM_TESTS.map(([t]) => [t, ANOM.windows.reduce((a, w) => a + w.tests[t].superior, 0)]));
def("anom.windows", ANOM.windows.length);
def("anom.reps", nf(ANOM.reps));
def("anom.k_min", Math.min(...ANOM.windows.map((w) => w.predictors)));
def("anom.k_max", Math.max(...ANOM.windows.map((w) => w.predictors)));
for (const [t] of ANOM_TESTS) def(`anom.total.${t}`, ANOM_TOT[t]);
def("anom.excess_fixed_pct", pct(ANOM_TOT.fixed / ANOM_TOT.boot_t - 1, 0));
def("anom.excess_q10_pct", pct(ANOM_TOT.fixed_q10 / ANOM_TOT.boot_t - 1, 0));
def("anom.windows_fixed_ge_boot", ANOM.windows.filter((w) => w.tests.fixed.superior >= w.tests.boot_t.superior).length);
for (const w of ANOM.windows) for (const [t] of ANOM_TESTS) def(`anom.${w.start.slice(0, 4)}.${t}`, w.tests[t].superior);
const ANOM_PRED = { A1: ANOM_TOT.fixed > ANOM_TOT.boot_t, A2: ANOM_TOT.fixed_q10 >= ANOM_TOT.fixed, A3: values.get("anom.windows_fixed_ge_boot") >= 5 };
def("anom.pred_held", Object.values(ANOM_PRED).filter(Boolean).length);
def("prereg.total_all", Number(values.get("prereg.total")) + Object.keys(ANOM_PRED).length);
def("prereg.held_all", Number(values.get("prereg.held")) + Object.values(ANOM_PRED).filter(Boolean).length);
empirical += table({
  caption: `Published cross-sectional predictors declared superior to zero by Romano--Wolf stepwise tests at 5\\%, in ten-year windows of monthly long-short returns (pre-registered illustration; counts of predictors).`,
  label: "tab:anomalies", rows: ANOM.windows, head: "Window",
  rowLabel: (w) => `${w.start.slice(0, 4)}--${w.end.slice(0, 4)} ($k=${w.predictors}$)`,
  cols: ANOM_TESTS, cell: (w, t) => String(w.tests[t].superior),
  note: `Data: Open Source Asset Pricing release 2.0.0 \\citep{chen2022open}, every predictor with a return in all 120 months of the window. Stationary bootstrap with ${nf(ANOM.reps)} resamples shared by all tests; mean block length $q=5$ except where stated. Fixed: Hansen's studentization, the Politis--Romano standard deviation held fixed in every resample (at $q=10$, arch's default, this is the computation arch merged in October 2026). Bootstrap-$t$: each resample studentized by its own standard deviation. Totals over the windows: ${ANOM_TESTS.map(([t, l]) => `${l} ${ANOM_TOT[t]}`).join(", ")}.`,
});

// Appendix tables.
let app = supp; // tables moved from the main text to the online appendix come first
app += table({
  caption: `Size of the remaining v1 validators, and of the other joint tests in the confirmation run v1b (percent, ${nf(nV1)} searches per cell).`,
  label: "tab:size-more", rows: FAMILIES,
  cols: [["haircut_bonferroni:v1", "HL 2s"], ["haircut_bonferroni_lo_one_sided:v1", "HL-Lo"], ["haircut_holm_one_sided:v1", "Holm"], ["oracle_hac_t:v1", "Oracle"], ["spa_upper:v1b", "SPA$_u$ v1b"], ["stepm_any:v1b", "Step-SPA v1b"]],
  cell: (f, k) => { const [v, r] = k.split(":"); const x = (r === "v1" ? V1 : V1B).table[v].size[f]; return v === "haircut_bonferroni" ? pct(x) : sizeCell(x, nV1); },
  note: `${BAND_NOTE(nV1)} HL 2s: the haircut with the two-sided $p$-values its authors define; its nominal one-sided level is 2.5\\%, so it is not flagged.`,
});
for (const sr of ["1", "3"]) {
  app += table({
    caption: `Raw power at a true annualized Sharpe ratio of ${sr} (percent), Null Zoo v1.`,
    label: `tab:power-${sr}`, rows: FAMILIES, cols: POWER,
    cell: (f, v) => pct(V1.table[v].power[f][sr]),
  });
}
app += table({
  caption: `Power (raw\\,/\\,size-adjusted) of the SPA variants at a true annualized Sharpe ratio of 2, Null Zoo v2 experiment A.`,
  label: "tab:mechanism-power", rows: FAMILIES, tight: true, cols: MECH.slice(0, 5),
  cell: (f, t) => `${pct(A[f].power[t])}\\,/\\,${pct(A[f].size_adjusted_power[t])}`,
});
{
  const T = lbl(["rc", "spa_c_unstud", "spa_c", "spa_u", "spa_c_sd", "spa_c_boot_t", "sidak_t"], V2_LABEL);
  const C = V2.C;
  app += table({
    caption: `Unequal volatilities with the skill on the least volatile strategy, and poor alternatives, with the strict inequality (Null Zoo v2, experiment C; i.i.d.\\ normal returns, ${T_OBS} periods, ${N_TRIALS} strategies, ${nf(C.poor_alternatives.searches)} searches per cell). Size\\,/\\,power\\,/\\,size-adjusted power, in percent.`,
    label: "tab:fair", rows: T.map(([t]) => t), rowLabel: (t) => T.find(([k]) => k === t)[1], head: "Test",
    cols: [["heterogeneous_volatility", "Unequal volatilities"], ["poor_alternatives", "Poor alternatives"]],
    cell: (t, d) => `${sizeCell(C[d].size[t], C[d].searches)}\\,/\\,${pct(C[d].power[t])}\\,/\\,${pct(C[d].size_adjusted_power[t])}`,
    note: `The designs of Table~\\ref{tab:fair2}, before ties were counted: every floored test here uses the strict inequality $p=\\#\\{T^*>T\\}/B$.`,
  });
}
app += table({
  caption: `arch ${V2ARCH.arch} on Null Zoo searches (experiment D, ${nf(V2ARCH.searches_per_family)} null searches per family): size of arch's consistent and upper $p$-values and of this paper's implementation on the same searches, at mean block length 8 and at arch's default (${V2ARCH.default_block}).`,
  label: "tab:arch", rows: Object.keys(V2.D), tight: true,
  cols: [["arch_b8_consistent", "arch$_c$ (8)"], ["ours_b8_spa_c_unstud", "SPA$_c$-U (8)"], ["arch_b8_upper", "arch$_u$ (8)"], ["ours_b8_rc", "RC (8)"], ["arch_default_consistent", `arch$_c$ (${V2ARCH.default_block})`], ["ours_default_spa_c_unstud", `SPA$_c$-U (${V2ARCH.default_block})`], ["ours_default_spa_c", `SPA$_c$ (${V2ARCH.default_block})`]],
  cell: (f, k) => sizeCell(V2.D[f].size[k], V2.D[f].searches),
  note: `${BAND_NOTE(V2ARCH.searches_per_family)} Searches on which arch's $p$-values differed between \\texttt{studentize=True} and \\texttt{studentize=False}: ${Object.values(V2.D).reduce((a, c) => a + Object.values(c.flag_changes_p).reduce((x, y) => x + y, 0), 0)} of ${nf(Object.values(V2.D).reduce((a, c) => a + c.searches, 0) * 2)} search-and-block pairs.`,
});
app += table({
  caption: `Bootstrap resolution: the same ${nf(BR.reps)} searches per cell scored by the single-series bootstrap with 400 and with 1,999 resamples (v0 design, robustness generator; not pre-registered). Size and power in percent.`,
  label: "tab:bootres", rows: BR.rows.map((r) => r.family), tight: true, rowLabel: (f) => FAMILY_LABEL[f],
  cols: [["size:luck_trials", "LET size"], ["size:bootstrap_400", "Boot$_{400}$ size"], ["size:bootstrap_1999", "Boot$_{1999}$ size"], ["power:luck_trials", "LET power"], ["power:bootstrap_400", "Boot$_{400}$ power"], ["power:bootstrap_1999", "Boot$_{1999}$ power"]],
  cell: (f, k) => { const [arm, v] = k.split(":"); const r = BR.rows.find((x) => x.family === f); return pct(r[arm][v] / BR.reps, arm === "size" ? 2 : 1); },
});
app += table({
  caption: `The two variants of experiment E not shown in Table~\\ref{tab:k-block}: arch's computation (SPA$_c$-U) and Hansen's fixed-variance test with the strict inequality (SPA$_c$ ($>$)); size in percent.`,
  label: "tab:k-block-more", rows: E_K, tight: true, head: "", rowLabel: (k) => `$k=${k}$`,
  groups: [["SPA$_c$-U", 3], ["SPA$_c$ ($>$)", 3]],
  cols: ["spa_c_unstud", "spa_c_strict"].flatMap((t) => E_B.map((b) => [`${t}:${b}`, `${b}`])),
  cell: (k, key) => { const [t, b] = key.split(":"); const c = E[`k${k}`][`b${b}`]; return sizeCell(c.size[t], c.searches); },
  note: `${BAND_NOTE(E.k1.b1.searches)}`,
});
{
  const T = lbl(["rc", "spa_c_unstud", "spa_c_strict", "spa_c_ge", "spa_c_boot_t_ge", "spa_c_nb_ge"], V2_LABEL);
  app += table({
    caption: `Experiment F (Null Zoo v2b): size and raw\\,/\\,size-adjusted power at a Sharpe ratio of 2 of every variant (percent).`,
    label: "tab:block-student-power", rows: FAMILIES, tight: true, groups: [["Size", 6], ["Raw\\,/\\,size-adjusted power", 3]], colspec: `l${"r".repeat(9)}`,
    cols: [...T.map(([t, l]) => [`s:${t}`, l]), ["p:rc", "RC"], ["p:spa_c_boot_t_ge", "SPA$_c$-BT"], ["p:spa_c_nb_ge", "SPA$_c$-NB"]],
    cell: (f, key) => { const [m, t] = key.split(":"); return m === "s" ? sizeCell(F[f].size[t], F[f].searches_null) : `${pct(F[f].power[t])}\\,/\\,${pct(F[f].size_adjusted_power[t])}`; },
    note: `${BAND_NOTE(F.iid_normal.searches_null)}`,
  });
}
app += empTable({
  caption: `Every window of the empirical illustration ($p$-values; Null Zoo v2b, part G).`,
  label: "tab:empirical-all", windows: EW, placement: "p", font: "\\scriptsize", note: "As in Table~\\ref{tab:empirical}.",
});
const NAME_NOTE = (names) => `Code names: ${names.map((k) => `\\texttt{${k.replaceAll("_", "\\_")}} = ${V2_LABEL[k]}`).join("; ")}.`;
function predictionTable({ caption, label, ids, preds, names }) {
  let t = `\\begin{table}[t]\n\\centering\n\\small\n\\caption{${caption}}\n\\label{${label}}\n\\begin{tabular}{lp{0.72\\linewidth}l}\n\\toprule\nID & Prediction & Outcome \\\\\n\\midrule\n`;
  for (const id of ids) t += `${id} & ${preds[id].statement.replaceAll("_", "\\_").replaceAll("%", "\\%")} & ${preds[id].held ? "held" : "\\textbf{did not hold}"} \\\\\n`;
  t += `\\bottomrule\n\\end{tabular}\n`;
  return `${t}${tableNote(NAME_NOTE(names))}\\end{table}\n\n`;
}
app += predictionTable({
  caption: `The nine predictions registered before the v2 run (analysis/v2/prereg-v2.json, commit \\texttt{${values.get("v2.prereg.commit")}}) and whether each held, as scored by analysis/v2/evaluate\\_v2.py.`,
  label: "tab:predictions", ids: PRED, preds: V2.predictions,
  names: ["spa_c", "spa_c_sd", "spa_c_boot_t", "rc", "spa_c_unstud", "spa_c_b22", "spa_u"],
});
app += predictionTable({
  caption: `The seven predictions registered before the v2b run (analysis/v2b/prereg-v2b.json, commit \\texttt{${values.get("v2b.prereg.commit")}}) and whether each held, as scored by analysis/v2b/evaluate\\_v2b.py.`,
  label: "tab:predictions-v2b", ids: PRED_B, preds: V2B.predictions,
  names: ["rc", "spa_c_ge", "spa_c_strict", "spa_c_boot_t_ge", "spa_c_nb_ge"],
});

// ---- write -------------------------------------------------------------------------------------
let out = `% Generated by analysis/tables.mjs from the result files. Do not edit.\n`;
for (const [k, v] of values) out += `\\nzdef{${k}}{${v}}\n`;
mkdirSync(`${here}paper/generated`, { recursive: true });
writeFileSync(`${here}paper/generated/values.tex`, out);
writeFileSync(`${here}paper/generated/tables-single.tex`, single);
writeFileSync(`${here}paper/generated/tables-snooping.tex`, snoop);
writeFileSync(`${here}paper/generated/tables-empirical.tex`, empirical);
writeFileSync(`${here}paper/generated/appendix-tables.tex`, app);
console.error(`wrote ${values.size} values and ${single.length + snoop.length + empirical.length + app.length} characters of tables`);
