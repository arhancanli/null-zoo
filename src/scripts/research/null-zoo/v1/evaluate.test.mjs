// What the audit quotes must be derived, never typed: each committed evaluation is exactly what
// evaluate.mjs derives from its committed run and pre-registration, and the size table the server
// ships (js/null-zoo-v1-sizes.js, mirrored into the MCP package) is exactly the confirmation run's.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { evaluate, sizesModule } from "./evaluate.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../..", "..");
const read = (p) => readFileSync(resolve(ROOT, p), "utf8");
const json = (p) => JSON.parse(read(p));

const RUNS = [
  { run: "config/research/null-zoo-v1.json", prereg: "config/research/null-zoo-v1-prereg.json", evaluation: "config/research/null-zoo-v1-evaluation.json" },
  { run: "config/research/null-zoo-v1b.json", prereg: "config/research/null-zoo-v1b-prereg.json", evaluation: "config/research/null-zoo-v1b-evaluation.json" },
];

for (const r of RUNS) {
  test(`${r.evaluation} is exactly what evaluate.mjs derives from ${r.run}`, () => {
    const derived = evaluate(json(r.run), json(r.prereg));
    derived.prereg = r.prereg;
    assert.equal(read(r.evaluation), `${JSON.stringify(derived, null, 1)}\n`);
  });
}

test("the size table the server ships is the confirmation run's, byte for byte, in both copies", () => {
  const confirmation = RUNS[1];
  const expected = sizesModule(json(confirmation.evaluation), confirmation.evaluation);
  assert.equal(read("js/null-zoo-v1-sizes.js"), expected);
  assert.equal(read("mcp/src/local/js/null-zoo-v1-sizes.js"), expected);
});

test("the two runs share no seed: v1b is a fresh sample", () => {
  const seeds = (p) => new Set(json(p).cells.flatMap((c) => c.seeds));
  const v1 = seeds(RUNS[0].run);
  for (const s of seeds(RUNS[1].run)) assert.ok(!v1.has(s), `seed ${s} appears in both runs`);
});
