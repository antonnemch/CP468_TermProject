# scripts/run_large.sh — example large-n runs
mkdir -p logs
python -m nqueens.cli solve --n 100000 --k-sample 64 --max-steps 300000 --seed 42 --json-out logs/run_100k.json
python -m nqueens.cli solve --n 200000 --k-sample 64 --max-steps 600000 --seed 43 --json-out logs/run_200k.json
