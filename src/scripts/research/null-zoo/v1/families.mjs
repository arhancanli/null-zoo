// Null Zoo v1 families: the eight v0 families, unchanged, plus block_cluster.
import { FAMILIES, drawSearch } from "../families.mjs";

export const FAMILIES_V1 = Object.freeze([...FAMILIES, "block_cluster"]);

// Four clusters of five trials. Trials in a cluster share a factor (pairwise correlation 0.8) and
// clusters are independent: twenty trials that are about four ideas, the shape a parameter grid
// around a few rules produces. Unit variance, mean zero; `skill` is added to trial 0 only.
export const BLOCK_CLUSTER = Object.freeze({ clusters: 4, correlation: 0.8 });

export function drawSearchV1(family, design, r) {
  if (family !== "block_cluster") return drawSearch(family, design, r);
  const { trials, observations, periodsPerYear, skill = 0 } = design;
  const drift = skill / Math.sqrt(periodsPerYear);
  const size = Math.ceil(trials / BLOCK_CLUSTER.clusters);
  const shared = Math.sqrt(BLOCK_CLUSTER.correlation);
  const own = Math.sqrt(1 - BLOCK_CLUSTER.correlation);
  const out = [];
  let factor = null;
  for (let k = 0; k < trials; k++) {
    if (k % size === 0) factor = Float64Array.from({ length: observations }, () => r.gauss());
    const xs = new Float64Array(observations);
    for (let t = 0; t < observations; t++) xs[t] = shared * factor[t] + own * r.gauss() + (k === 0 ? drift : 0);
    out.push(xs);
  }
  return out;
}
