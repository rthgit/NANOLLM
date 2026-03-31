#!/usr/bin/env python3
"""Extract clean Phase 5 results."""
import sys; sys.path.insert(0, ".")
from nano_link_phase5 import *

# Test 1: Expert scaling
scaling = test_expert_scaling()
# Test 2: Sparsity
sparsity = test_sparsity_level()
# Efficiency
efficiency = test_efficiency()
print_summary(scaling, sparsity, efficiency)
