"""Seed check for the confirmation run (imported by run_confirm.py; also runnable alone).

Every earlier numpy run in this repository seeds SeedSequence([BASE or SEED, ...]) with the module-level constant
it defines. The confirmation run's first entropy word, 20261108, must differ from all of them, so that its streams
share no seed with any earlier run. Raises SystemExit otherwise.
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
NEW = 20261108
used = {}
for path in glob.glob(os.path.join(HERE, "..", "**", "*.py"), recursive=True):
    if os.path.abspath(os.path.dirname(path)) == HERE:
        continue
    for m in re.finditer(r"^(?:BASE|SEED) = (\d+)", open(path).read(), flags=re.M):
        used.setdefault(int(m.group(1)), []).append(os.path.relpath(path, os.path.join(HERE, "..", "..")))
if NEW in used:
    raise SystemExit(f"base seed {NEW} was used by {used[NEW]}")
if __name__ == "__main__":
    for k in sorted(used):
        print(k, used[k])
    print(f"{NEW}: not used by any earlier numpy run")
