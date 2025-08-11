#!/usr/bin/env python3
"""Test generating a single figure to verify it works"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import just the network topology figure generation
from generate_all_figures_real import generate_network_topology_figure

print("Testing single figure generation...")
generate_network_topology_figure()
print("Done!")