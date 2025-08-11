#!/usr/bin/env python3
"""
Direct comparison between MPS and ADMM algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.admm_algorithm import DecentralizedADMM

def run_comparison():
    """Compare MPS and ADMM algorithms on the same problem"""
    
    print("=" * 60)
    print("MPS vs ADMM Algorithm Comparison")
    print("=" * 60)
    
    # Problem configuration
    problem_params = {
        'n_sensors': 30,
        'n_anchors': 6,
        'd': 2,
        'communication_range': 0.4,
        'noise_factor': 0.05,
        'alpha_admm': 150.0,
        'max_iter': 500,
        'tol': 1e-4
    }
    
    # Generate network
    np.random.seed(42)
    true_positions = {}
    for i in range(problem_params['n_sensors']):
        pos = np.random.normal(0.5, 0.2, 2)
        true_positions[i] = np.clip(pos, 0, 1)
    
    anchor_positions = np.random.uniform(0, 1, (problem_params['n_anchors'], 2))
    
    # Run ADMM
    print("\nRunning ADMM algorithm...")
    admm = DecentralizedADMM(problem_params)
    admm.generate_network(true_positions, anchor_positions)
    
    start_time = time.time()
    admm_results = admm.run_admm()
    admm_time = time.time() - start_time
    
    # Run MPS algorithm
    print("\nRunning MPS algorithm...")
    from core.mps_algorithm import MPSSensorNetwork
    
    mps = MPSSensorNetwork(problem_params)
    mps.generate_network(true_positions, anchor_positions)
    
    start_time = time.time()
    mps_results = mps.run_mps()
    mps_time = time.time() - start_time
    
    # Add timing to results
    mps_results['time'] = mps_time
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS:")
    print("=" * 60)
    
    print("\nMPS Algorithm:")
    print(f"  Iterations: {mps_results['iterations']}")
    print(f"  Final Error: {mps_results['errors'][-1]:.4f}")
    print(f"  Time: {mps_results['time']:.2f}s")
    print(f"  Converged: {mps_results['converged']}")
    
    print("\nADMM Algorithm:")
    print(f"  Iterations: {admm_results['iterations']}")
    if admm_results['errors']:
        print(f"  Final Error: {admm_results['errors'][-1]:.4f}")
    print(f"  Time: {admm_time:.2f}s")
    print(f"  Converged: {admm_results['converged']}")
    
    if admm_results['errors'] and mps_results['errors']:
        ratio = admm_results['errors'][-1] / mps_results['errors'][-1]
        print(f"\nPerformance Ratio:")
        print(f"  MPS is {ratio:.1f}x more accurate than ADMM")
        print(f"  MPS converges {admm_results['iterations']/mps_results['iterations']:.1f}x faster")
    
    # Visualize comparison
    visualize_comparison(mps_results, admm_results)
    
    return mps_results, admm_results

def visualize_comparison(mps_results, admm_results):
    """Create comparison visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convergence comparison
    ax1.set_title('Algorithm Convergence Comparison')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Error (RMSE)')
    
    if mps_results['errors']:
        mps_iters = np.linspace(0, mps_results['iterations'], len(mps_results['errors']))
        ax1.semilogy(mps_iters, mps_results['errors'], 
                    'b-', linewidth=2, label='MPS', marker='o')
    
    if admm_results['errors']:
        admm_iters = np.arange(0, len(admm_results['errors']) * 10, 10)
        ax1.semilogy(admm_iters, admm_results['errors'], 
                    'r--', linewidth=2, label='ADMM', marker='s')
    
    ax1.axhline(y=0.05, color='green', linestyle=':', 
               linewidth=1, alpha=0.5, label='Target Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance metrics bar chart
    ax2.set_title('Performance Metrics')
    
    metrics = ['Iterations', 'Time (s)', 'Final Error']
    mps_values = [
        mps_results['iterations'],
        mps_results['time'],
        mps_results['errors'][-1] * 100 if mps_results['errors'] else 0
    ]
    admm_values = [
        admm_results['iterations'],
        admm_results.get('time', 0),
        admm_results['errors'][-1] * 100 if admm_results['errors'] else 0
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, mps_values, width, label='MPS', alpha=0.8)
    bars2 = ax2.bar(x + width/2, admm_values, width, label='ADMM', alpha=0.8)
    
    ax2.set_ylabel('Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('MPS vs ADMM: Algorithm Comparison')
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150)
    plt.show()
    
    print(f"\nVisualization saved to algorithm_comparison.png")

if __name__ == "__main__":
    mps_results, admm_results = run_comparison()