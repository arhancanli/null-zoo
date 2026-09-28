// Builds every table and every number quoted in the paper from the result files, so no figure in
// the text is typed by hand.
//   node analysis/tables.mjs   ->  paper/generated/{tables.tex,appendix-tables.tex,numbers.tex}
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { expectedMaxStandardNormal, normalCdf, normalPpf } from "../src/js/dsr-core.js";

const here = new URL("..", import.meta.url).pathname;
const v0Path = `${here}src/config/research/null-zoo-v0.json`;
const v0 = JSON.parse(readFileSync(v0Path, "utf8"));
const ext = JSON.parse(readFileSync(`${here}analysis/null-zoo-extended.json`, "utf8"));
const ext1 = JSON.parse(readFileSync(`${here}analysis/null-zoo-extended-v1.json`, "utf8"));
const LEVEL = v0.design.level;
const N_TRIALS = v0.design.trials;
const nV0 = v0.reps;
const nExt = ext.reps;

export const FAMILY_LABEL = {
  iid_normal: "IID normal",
  student_t4: "Student $t_4$",
  skew_negative: "Skew $-1.32$",
  skew_positive: "Skew $+1.32$",
  garch: "GARCH(1,1)",
  ar1: "AR(1), $\\phi=0.2$",
  regimes: "Two vol.\\ regimes",
  correlated_trials: "Correlated, $\\rho=0.5$",
};
// Main-table columns. In the robustness run the haircut columns are read one-sided.
const COLS = [
  ["luck_trials", "LET"],
  ["luck_trials_lo", "LET-Lo"],
  ["luck_trials_nonnormal_se", "LET-NN"],
  ["haircut_bonferroni", "HC"],
  ["haircut_bonferroni_lo", "HC-Lo"],
  ["deflated_sharpe", "DSR"],
  ["bootstrap_best", "Boot"],
];
const extKey = (v) => (v.startsWith("haircut_") ? `${v}_one_sided` : v);
const FAMILIES = v0.rows.map((r) => r.family);
const extRow = Object.fromEntries(ext.rows.map((r) => [r.family, r]));
const ext1Row = Object.fromEntries(ext1.rows.map((r) => [r.family, r]));

const se = (p, n) => Math.sqrt((p * (1 - p)) / n);
const z99 = normalPpf(0.995);
const z95 = normalPpf(0.975);
const band = (n, z = z99) => z * se(LEVEL, n);
const pct = (x, d = 1) => (100 * x).toFixed(d);

function sizeCell(p, n) {
  const s = pct(p, n >= 10000 ? 2 : 1);
  if (p > LEVEL + band(n)) return `\\textbf{${s}}`;
  if (p < LEVEL - band(n)) return `\\textit{${s}}`;
  return s;
}

function table({ caption, label, cell, note, cols = COLS }) {
  let t = `\\begin{table}[t]\n\\centering\n\\small\n\\caption{${caption}}\n\\label{${label}}\n`;
  t += `\\begin{tabular}{l${"r".repeat(cols.length)}}\n\\toprule\n`;
  t += `Family & ${cols.map(([, l]) => l).join(" & ")} \\\\\n\\midrule\n`;
  for (const f of FAMILIES) t += `${FAMILY_LABEL[f]} & ${cols.map(([v]) => cell(f, v)).join(" & ")} \\\\\n`;
  t += `\\bottomrule\n\\end{tabular}\n`;
  if (note) t += `\\par\\smallskip\\begin{minipage}{0.97\\linewidth}\\footnotesize ${note}\\end{minipage}\n`;
  return `${t}\\end{table}\n\n`;
}

const v0Row = Object.fromEntries(v0.rows.map((r) => [r.family, r]));
const nf = (x) => x.toLocaleString("en-US");

let main = "";
main += table({
  caption: `Size: percentage of skill-less searches (${N_TRIALS} strategies, 504 daily returns) whose best strategy is called skilled at the nominal 5\\% level. Published v0 run, ${nf(nV0)} searches per cell; HC two-sided as \\citet{harvey2015backtesting} define it; bootstrap with 400 resamples.`,
  label: "tab:size-v0",
  cell: (f, v) => sizeCell(v0Row[f].size[v], nV0),
  note: `Bold: above the 99\\% Monte Carlo band around 5\\% (${pct(LEVEL - band(nV0))}--${pct(LEVEL + band(nV0))}); italic: below it. A two-sided $p\\le 0.05$ for a positive Sharpe ratio is a one-sided test at 2.5\\%, so the HC columns are not comparable with the others here; Table~\\ref{tab:size-ext} reads them one-sided.`,
});
main += table({
  caption: `Size in the robustness run: xoshiro128** generator, ${nf(nExt)} searches per cell, HC read one-sided (two-sided $p$ halved), bootstrap with ${nf(ext.bootstrap_draws)} resamples.`,
  label: "tab:size-ext",
  cell: (f, v) => sizeCell(extRow[f].size[extKey(v)] / nExt, nExt),
  note: `99\\% Monte Carlo band around 5\\%: ${pct(LEVEL - band(nExt), 2)}--${pct(LEVEL + band(nExt), 2)}. The standard error of any rate $\\hat p$ is $\\sqrt{\\hat p(1-\\hat p)/${nf(nExt)}}$, at most ${pct(se(0.5, nExt), 2)} percentage points.`,
});
main += table({
  caption: `Power: percentage of searches in which one strategy has a true annualized Sharpe ratio of 2 and the validator rejects. Robustness run, as in Table~\\ref{tab:size-ext}.`,
  label: "tab:power-ext",
  cell: (f, v) => pct(extRow[f].power[extKey(v)] / nExt),
  note: `Raw power rewards a liberal test; compare columns with Table~\\ref{tab:power-adj}. Monte Carlo standard errors are at most ${pct(se(0.5, nExt), 2)} percentage points.`,
});
main += table({
  caption: `Size-adjusted power: power at the cutoff at which each validator rejects at most 5\\% of the skill-less searches in the same family (its own null arm). Robustness run.`,
  label: "tab:power-adj",
  cell: (f, v) => { const a = extRow[f].size_adjusted[extKey(v)]; return a.cutoff === null ? "n/a" : pct(a.power / nExt); },
  note: `n/a: the validator's smallest attainable $p$-value already occurs in more than 5\\% of the null arm, so no cutoff attains the level. Size-adjusted power is not available to a user, who does not know the family; it isolates how well each statistic ranks skilled searches above skill-less ones. The bootstrap's $p$-value is discrete, so its attained null rate falls below 5\\% in some families (${FAMILIES.map((f) => { const a = extRow[f].size_adjusted.bootstrap_best; return a.cutoff === null ? null : `${FAMILY_LABEL[f]} ${pct(a.null_rate_count / nExt, 2)}\\%`; }).filter(Boolean).join("; ")}), and its size-adjusted power is then measured at that lower level.`,
});

// Appendix tables.
let app = "";
app += table({
  caption: `Power in the published v0 run (${nf(nV0)} searches per cell; HC two-sided; bootstrap with 400 resamples).`,
  label: "tab:power-v0",
  cell: (f, v) => pct(v0Row[f].power[v]),
});
app += table({
  caption: `Correct discoveries: percentage of searches in which the validator rejects \\emph{and} the selected best strategy is the skilled one. Robustness run.`,
  label: "tab:correct-ext",
  cell: (f, v) => pct(extRow[f].correct[extKey(v)] / nExt),
  note: `The skilled strategy is the in-sample best in the following share of searches: ${FAMILIES.map((f) => `${FAMILY_LABEL[f]} ${pct(extRow[f].best_is_skilled_trial / nExt)}\\%`).join("; ")}.`,
});
{
  const D = [
    ["sdnull", "SD/SE, null"],
    ["sdskill", "SD/SE, skill"],
    ["skewbest", "Skew of best"],
    ["skewothers", "Skew of others"],
  ];
  const val = (f, k) => {
    const d = extRow[f].diagnostics;
    return { sdnull: d.null.cross_trial_sd_over_iid_se, sdskill: d.skill.cross_trial_sd_over_iid_se, skewbest: d.null.mean_skew_best, skewothers: d.null.mean_skew_others }[k].toFixed(2);
  };
  app += table({
    caption: `Diagnostics from the robustness run. SD/SE: mean cross-trial standard deviation of the ${N_TRIALS} annualized Sharpe ratios divided by the i.i.d.\\ standard error $\\sqrt{q/T}$, in the null and skill arms. Skew: mean sample skewness of the selected best series and of the other series, null arm.`,
    label: "tab:diagnostics",
    cols: D,
    cell: val,
  });
}

// ---- Numbers quoted in the text --------------------------------------------------------------
const camel = (s) => s.replace(/(^|_)([a-z0-9])/g, (_, __, c) => c.toUpperCase()).replace(/[0-9]/g, (d) => ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"][d]);
let nums = `% Generated by analysis/tables.mjs from the result files. Do not edit.\n`;
const def = (name, value) => { nums += `\\newcommand{\\${name}}{${value}}\n`; };

def("nzTrials", String(N_TRIALS));
def("nzRepsVzero", nf(nV0));
def("nzRepsExt", nf(nExt));
def("nzBootDraws", nf(ext.bootstrap_draws));
def("nzBandVzeroLo", pct(LEVEL - band(nV0)));
def("nzBandVzeroHi", pct(LEVEL + band(nV0)));
def("nzBandExtLo", pct(LEVEL - band(nExt), 2));
def("nzBandExtHi", pct(LEVEL + band(nExt), 2));
def("nzBandExtNinetyFiveHi", pct(LEVEL + band(nExt, z95), 2));
def("nzSeFiveVzero", pct(se(LEVEL, nV0), 2));
def("nzSeFiveExt", pct(se(LEVEL, nExt), 2));

for (const f of FAMILIES) {
  const e = extRow[f];
  for (const [v] of [...COLS, ["haircut_holm"]]) {
    const key = `${camel(v)}${camel(f)}`;
    if (v0Row[f].size[v] !== undefined) {
      def(`nz${key}SizeVzero`, pct(v0Row[f].size[v]));
      def(`nz${key}PowerVzero`, pct(v0Row[f].power[v]));
    }
    const k = extKey(v);
    def(`nz${key}SizeExt`, pct(e.size[k] / nExt, 2));
    def(`nz${key}PowerExt`, pct(e.power[k] / nExt));
    def(`nz${key}AdjPowerExt`, e.size_adjusted[k].cutoff === null ? "n/a" : pct(e.size_adjusted[k].power / nExt));
    def(`nz${key}CorrectExt`, pct(e.correct[k] / nExt));
    if (k !== v) def(`nz${key}SizeExtTwoSided`, pct(e.size[v] / nExt, 2));
  }
  def(`nzBestSkilled${camel(f)}Ext`, pct(e.best_is_skilled_trial / nExt));
  def(`nzSdRatioNull${camel(f)}`, e.diagnostics.null.cross_trial_sd_over_iid_se.toFixed(2));
  def(`nzSdRatioSkill${camel(f)}`, e.diagnostics.skill.cross_trial_sd_over_iid_se.toFixed(2));
  def(`nzSkewBest${camel(f)}`, e.diagnostics.null.mean_skew_best.toFixed(2));
  def(`nzSkewOthers${camel(f)}`, e.diagnostics.null.mean_skew_others.toFixed(2));
}

// Bootstrap resolution: smallest attainable Sidak-adjusted p-value with B resamples.
const minBootP = (b) => -Math.expm1(N_TRIALS * Math.log1p(-1 / (b + 1)));
def("nzBootMinPFourHundred", minBootP(400).toFixed(4));
def("nzBootMinPExt", minBootP(ext.bootstrap_draws).toFixed(4));
def("nzBootIidPowerFourHundred", pct(ext1Row.iid_normal.power.reject.bootstrap_best / ext1.reps));
def("nzBootSkewNegPowerFourHundred", pct(ext1Row.skew_negative.power.reject.bootstrap_best / ext1.reps));

// LET vs one-sided Bonferroni haircut: they can disagree only when p1 lies in (a, b].
const sidakP1 = -Math.expm1(Math.log1p(-LEVEL) / N_TRIALS);
def("nzBonfPOne", (LEVEL / N_TRIALS).toFixed(6));
def("nzSidakPOne", sidakP1.toFixed(6));
def("nzLetHcMaxDiff", String(Math.max(...ext.rows.flatMap((r) => ["size", "power"].map((a) => Math.abs(r[a].luck_trials - r[a].haircut_bonferroni_one_sided))))));
def("nzHolmBonfMaxDiff", String(Math.max(...ext.rows.flatMap((r) => ["size", "power"].map((a) => Math.abs(r[a].haircut_holm_one_sided - r[a].haircut_bonferroni_one_sided))))));

// Lo correction: its size shift relative to LET in the seven families without autocorrelation.
const loShift = FAMILIES.filter((f) => f !== "ar1").map((f) => (extRow[f].size.luck_trials_lo - extRow[f].size.luck_trials) / nExt);
def("nzLoShiftMin", pct(Math.min(...loShift), 2));
def("nzLoShiftMax", pct(Math.max(...loShift), 2));
const loPowerShift = FAMILIES.filter((f) => f !== "ar1").map((f) => Math.abs(extRow[f].power.luck_trials_lo - extRow[f].power.luck_trials) / nExt);
def("nzLoPowerShiftMax", pct(Math.max(...loPowerShift), 2));

// DSR constants: exact expected maximum of N standard normals (numerical integration), the
// Bailey-Lopez de Prado approximation, the Sidak critical value under independence and normality,
// and the implied DSR hurdles in the null and skill arms (i.i.d. normal).
function exactExpectedMax(n) {
  let s = 0;
  const h = 1e-3;
  for (let x = -10; x <= 10; x += h) {
    const pdf = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    s += x * n * normalCdf(x) ** (n - 1) * pdf * h;
  }
  return s;
}
const sidakZ = normalPpf(1 - sidakP1);
const approxMax = expectedMaxStandardNormal(N_TRIALS);
const oneZ = normalPpf(1 - LEVEL);
def("nzSidakCrit", sidakZ.toFixed(2));
def("nzExpMax", approxMax.toFixed(2));
def("nzExpMaxExact", exactExpectedMax(N_TRIALS).toFixed(2));
def("nzSingleCrit", oneZ.toFixed(3));
const sdNull = extRow.iid_normal.diagnostics.null.cross_trial_sd_over_iid_se;
const sdSkill = extRow.iid_normal.diagnostics.skill.cross_trial_sd_over_iid_se;
def("nzDsrHurdleNull", (sdNull * approxMax + oneZ).toFixed(2));
def("nzDsrHurdleSkill", (sdSkill * approxMax + oneZ).toFixed(2));
def("nzDsrMaxSizeExceptCorr", pct(Math.max(...FAMILIES.filter((f) => f !== "correlated_trials").map((f) => extRow[f].size.deflated_sharpe / nExt)), 2));
def("nzDsrFailToAccept", pct(1 - extRow.iid_normal.power.deflated_sharpe / nExt));

// DSR size-adjusted gap to the Sidak test across families (N1).
const dsrGap = FAMILIES.map((f) => (extRow[f].size_adjusted.luck_trials.power - extRow[f].size_adjusted.deflated_sharpe.power) / nExt);
def("nzDsrAdjGapMin", pct(Math.min(...dsrGap)));
def("nzDsrAdjGapMax", pct(Math.max(...dsrGap)));
// DSR fails to accept, per family range (N10).
const fta = FAMILIES.map((f) => 1 - extRow[f].power.deflated_sharpe / nExt);
def("nzDsrFailToAcceptMin", pct(Math.min(...fta)));
def("nzDsrFailToAcceptMax", pct(Math.max(...fta)));
// SD/SE ratio below which the DSR hurdle falls under the Sidak critical value (N3).
def("nzDsrHurdleNullCorr", (extRow.correlated_trials.diagnostics.null.cross_trial_sd_over_iid_se * approxMax + oneZ).toFixed(2));
def("nzSdRatioCrossover", ((sidakZ - oneZ) / approxMax).toFixed(2));
// First 10,000-replication run (400 resamples), for cells the text compares across runs.
const ext1Rate = (f, v) => pct(ext1Row[f].size.reject[v] / ext1.reps, 2);
def("nzNNSkewPosSizeExtOne", ext1Rate("skew_positive", "luck_trials_nonnormal_se"));
def("nzBootTFourSizeExtOne", ext1Rate("student_t4", "bootstrap_best"));
def("nzLetGarchSizeExtOne", ext1Rate("garch", "luck_trials"));
// Paired bootstrap-resolution run (N2).
{
  const br = JSON.parse(readFileSync(`${here}analysis/bootstrap-resolution.json`, "utf8"));
  const n = br.reps;
  def("nzBrReps", nf(n));
  for (const r of br.rows) {
    const F = camel(r.family);
    for (const arm of ["size", "power"]) {
      const A = arm === "size" ? "Size" : "Power";
      def(`nzBr${F}${A}Let`, pct(r[arm].luck_trials / n, arm === "size" ? 2 : 1));
      def(`nzBr${F}${A}BootFourHundred`, pct(r[arm].bootstrap_400 / n, arm === "size" ? 2 : 1));
      def(`nzBr${F}${A}BootMax`, pct(r[arm].bootstrap_1999 / n, arm === "size" ? 2 : 1));
    }
  }
  // With B resamples the Sidak-adjusted p-value is at most 5% only when h+1 <= (B+1) * p1*.
  def("nzBootMaxHitsExt", String(Math.floor((ext.bootstrap_draws + 1) * sidakP1) - 1));
}
def("nzVzeroSha", createHash("sha256").update(readFileSync(v0Path)).digest("hex"));
def("nzSourceCommit", readFileSync(`${here}src/SOURCE_COMMIT`, "utf8").trim().slice(0, 12));

mkdirSync(`${here}paper/generated`, { recursive: true });
writeFileSync(`${here}paper/generated/tables.tex`, main);
writeFileSync(`${here}paper/generated/appendix-tables.tex`, app);
writeFileSync(`${here}paper/generated/numbers.tex`, nums);
console.error(`wrote tables (${main.length + app.length} chars) and numbers.tex (${nums.split("\n").length} lines)`);
