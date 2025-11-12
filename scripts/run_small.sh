# scripts/run_small.sh — small-n smoke tests
python -m nqueens.cli solve --n 5000  --k-sample 64 --max-steps 15000 --seed 1
python -m nqueens.cli solve --n 20000 --k-sample 64 --max-steps 60000 --seed 2
python -m nqueens.cli solve-bt --n 200 --time-limit 5
