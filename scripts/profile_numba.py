"""
scripts.profile_numba
Purpose: Microbenchmark the Numba kernel to confirm JIT and throughput.

Flow:
- Warm up JIT compilation
- Time fixed-step kernel loop with modest n
- Print moves/sec and step latency
"""
