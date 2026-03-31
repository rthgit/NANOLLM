#!/usr/bin/env python3
"""Quick script to print clean results table."""
import numpy as np
from nano_link_test import run_experiment, PATTERN_COUNTS, NOISE_LEVELS, NUM_TRIALS
from itertools import product as cartesian

results = []
for n_pat, noise in cartesian(PATTERN_COUNTS, NOISE_LEVELS):
    trials = []
    for trial in range(NUM_TRIALS):
        seed = 1000 * n_pat + int(noise * 100) + trial
        trials.append(run_experiment(n_pat, noise, seed=seed))
    avg = {"n_patterns": n_pat, "noise": noise}
    for key in ["nano_exact_acc", "nano_bit_acc", "nano_cost",
                 "linear_exact_acc", "linear_bit_acc", "linear_cost"]:
        avg[key] = np.mean([t[key] for t in trials])
    results.append(avg)

header = "Patterns | Noise | NL_Exact | NL_Bit  | NL_Cost   | Lin_Exact | Lin_Bit | Lin_Cost   | Delta_Bit"
print(header)
print("-" * len(header))
for r in results:
    d = r["nano_bit_acc"] - r["linear_bit_acc"]
    tag = "WIN" if d > 0.005 else ("TIE" if abs(d) < 0.005 else "LOSS")
    print("{:>8} | {:>4}% | {:>8.4f} | {:>7.4f} | {:>9.0f} | {:>9.4f} | {:>7.4f} | {:>10.0f} | {:>+8.4f} {}".format(
        r["n_patterns"], int(r["noise"]*100),
        r["nano_exact_acc"], r["nano_bit_acc"], r["nano_cost"],
        r["linear_exact_acc"], r["linear_bit_acc"], r["linear_cost"],
        d, tag))

print()
c = [r for r in results if r["n_patterns"] == 1000 and r["noise"] == 0.30][0]
print("=== CRITICAL TEST (1000 patterns, 30% noise) ===")
print("  NanoLink bit acc: {:.4f}  |  Linear bit acc: {:.4f}".format(c["nano_bit_acc"], c["linear_bit_acc"]))
print("  NanoLink cost:    {:.0f}     |  Linear cost:    {:.0f}".format(c["nano_cost"], c["linear_cost"]))
print("  Cost ratio (Lin/Nano): {:.1f}x".format(c["linear_cost"] / max(c["nano_cost"], 1)))

print()
print("=== SCALABILITY ===")
for noise in NOISE_LEVELS:
    s20 = [r for r in results if r["n_patterns"] == 20 and r["noise"] == noise][0]
    s1k = [r for r in results if r["n_patterns"] == 1000 and r["noise"] == noise][0]
    ng = s1k["nano_cost"] / max(s20["nano_cost"], 1)
    lg = s1k["linear_cost"] / max(s20["linear_cost"], 1)
    print("  Noise {:>2}%: 20->1000 growth | Nano: {:.1f}x | Linear: {:.1f}x".format(
        int(noise*100), ng, lg))
