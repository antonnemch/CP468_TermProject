#!/usr/bin/env bash
# scripts/run_required_sizes.sh
# Purpose:
#   Run the Min-Conflicts solver for the six required n values from the
#   project specification (10, 100, 1,000, 10,000, 100,000, 1,000,000),
#   capturing runtime metrics into a CSV for inclusion in the final report.
#
# Behavior:
#   - Creates a "results" directory if missing.
#   - Invokes the CLI solver with fixed parameters for reproducibility.
#   - Appends one line per n to benchmarks/results/required_sizes.csv
#
# Usage:
#   bash scripts/run_required_sizes.sh
#
# Output:
#   benchmarks/results/required_sizes.csv  (columns: n, time_sec, steps, restarts, solved)
#
# Example:
#   $ bash scripts/run_required_sizes.sh
#   Solving for n=10 ...
#   Solving for n=100 ...
#   ...
#   Results saved to benchmarks/results/required_sizes.csv

mkdir -p benchmarks/results

echo "n,time_sec,steps,restarts,solved" > benchmarks/results/required_sizes.csv

for n in 10 100 1000 10000 100000 1000000; do
  echo "Solving for n=$n ..."
  python -m nqueens.cli solve --n $n --k-sample 64 --max-steps 300000 --seed 42 \
    --json-out tmp.json > /dev/null
  python - <<'EOF'
import json, csv
data = json.load(open("tmp.json"))
with open("benchmarks/results/required_sizes.csv","a",newline="") as f:
    csv.writer(f).writerow([data["n"],data["time_sec"],data["steps"],data["restarts"],data["solved"]])
EOF
done

rm -f tmp.json
echo "All runs complete. Results in benchmarks/results/required_sizes.csv"
