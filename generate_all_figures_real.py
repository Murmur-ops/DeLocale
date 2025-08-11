#!/usr/bin/env python3
"""
Generate all paper-style figures using REAL data from actual algorithm execution
No mock data - everything computed from actual MPS and ADMM runs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import time
from typing import Dict, List, Tuple
import os
import sys

# Import our actual implementations
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.mps_algorithm import MPSSensorNetwork
from core.admm_algorithm import DecentralizedADMM
from analysis.crlb_analysis import CRLBAnalyzer

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def run_algorithm_comparison(params):
    """Run both algorithms on the same problem and return real results"""
    print(f"  Running comparison with {params['n_sensors']} sensors...")
    
    # Generate network
    np.random.seed(42)
    true_positions = {}
    for i in range(params['n_sensors']):
        pos = np.random.normal(0.5, 0.2, 2)
        true_positions[i] = np.clip(pos, 0, 1)
    
    anchor_positions = np.random.uniform(0, 1, (params['n_anchors'], 2))
    
    # Run MPS
    mps = MPSSensorNetwork(params)
    mps.generate_network(true_positions, anchor_positions)
    mps_start = time.time()
    mps_results = mps.run_mps()
    mps_results['time'] = time.time() - mps_start
    
    # Run ADMM
    admm = DecentralizedADMM(params)
    admm.generate_network(true_positions, anchor_positions)
    admm_start = time.time()
    admm_results = admm.run_admm()
    admm_results['time'] = time.time() - admm_start
    
    # Add CRLB
    crlb = CRLBAnalyzer(
        n_sensors=params['n_sensors'],
        n_anchors=params['n_anchors'],
        communication_range=params['communication_range'],
        d=params.get('d', 2)
    )
    crlb_bound = crlb.compute_crlb(params['noise_factor'])
    
    return {
        'mps': mps_results,
        'admm': admm_results,
        'crlb': crlb_bound,
        'true_positions': true_positions,
        'anchor_positions': anchor_positions
    }

def generate_network_topology_figure():
    """Figure 1: Sensor Network Topology with Real Data"""
    print("\nGenerating Figure 1: Network Topology...")
    
    params = {
        'n_sensors': 30,
        'n_anchors': 6,
        'd': 2,
        'communication_range': 0.4,
        'noise_factor': 0.05,
        'max_iter': 100,
        'tol': 1e-3
    }
    
    results = run_algorithm_comparison(params)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Initial network with communication links
    ax1.set_title('Sensor Network Topology', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    
    # Draw communication range circles
    for i, pos in results['true_positions'].items():
        circle = Circle(pos, params['communication_range'], 
                       fill=False, edgecolor='lightblue', alpha=0.2, linewidth=1)
        ax1.add_patch(circle)
    
    # Draw communication links
    for i, pos_i in results['true_positions'].items():
        for j, pos_j in results['true_positions'].items():
            if i < j:
                dist = np.linalg.norm(pos_i - pos_j)
                if dist <= params['communication_range']:
                    ax1.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                            'gray', alpha=0.3, linewidth=0.5)
    
    # Plot sensors
    for i, pos in results['true_positions'].items():
        ax1.scatter(pos[0], pos[1], c='blue', s=50, zorder=5)
        ax1.text(pos[0]+0.02, pos[1]+0.02, str(i), fontsize=8)
    
    # Plot anchors
    for i, pos in enumerate(results['anchor_positions']):
        ax1.scatter(pos[0], pos[1], c='red', s=150, marker='^', zorder=6)
        ax1.text(pos[0]+0.02, pos[1]+0.02, f'A{i}', fontsize=10, fontweight='bold')
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Communication Range', 'Links', 'Sensors', 'Anchors'], loc='upper right')
    
    # Right: Localization results
    ax2.set_title('Localization Results (MPS)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    
    # Plot error vectors
    for i in results['true_positions']:
        true_pos = results['true_positions'][i]
        est_pos = results['mps']['final_positions'][i]
        error = np.linalg.norm(true_pos - est_pos)
        
        # Error line with color based on magnitude
        color = 'green' if error < 0.1 else 'orange' if error < 0.2 else 'red'
        ax2.plot([true_pos[0], est_pos[0]], [true_pos[1], est_pos[1]], 
                color=color, alpha=0.5, linewidth=1)
    
    # Plot positions
    for i in results['true_positions']:
        true_pos = results['true_positions'][i]
        est_pos = results['mps']['final_positions'][i]
        ax2.scatter(true_pos[0], true_pos[1], c='blue', s=30, alpha=0.5)
        ax2.scatter(est_pos[0], est_pos[1], c='green', s=30, marker='x')
    
    # Plot anchors
    for i, pos in enumerate(results['anchor_positions']):
        ax2.scatter(pos[0], pos[1], c='red', s=150, marker='^')
    
    # Add statistics
    errors = [np.linalg.norm(results['true_positions'][i] - results['mps']['final_positions'][i]) 
              for i in results['true_positions']]
    stats_text = f"RMSE: {np.sqrt(np.mean(np.square(errors))):.4f}\nMax Error: {np.max(errors):.4f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Error', 'True', 'Estimated', 'Anchors'], loc='upper right')
    
    plt.suptitle(f'Real Data: {params["n_sensors"]} Sensors, {params["n_anchors"]} Anchors, Range={params["communication_range"]}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/network_topology_real.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/network_topology_real.png")

def generate_convergence_comparison_figure():
    """Figure 2: Algorithm Convergence Comparison with Real Data"""
    print("\nGenerating Figure 2: Convergence Comparison...")
    
    params = {
        'n_sensors': 50,
        'n_anchors': 8,
        'd': 2,
        'communication_range': 0.3,
        'noise_factor': 0.05,
        'max_iter': 200,
        'tol': 1e-4
    }
    
    results = run_algorithm_comparison(params)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Objective convergence
    ax1.set_title('Objective Function Convergence', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value (log scale)')
    
    if results['mps']['objectives']:
        mps_iters = np.arange(0, len(results['mps']['objectives']) * 10, 10)
        ax1.semilogy(mps_iters, results['mps']['objectives'], 
                    'b-', linewidth=2, label='MPS', marker='o', markersize=4)
    
    if results['admm']['objectives']:
        admm_iters = np.arange(0, len(results['admm']['objectives']) * 10, 10)
        ax1.semilogy(admm_iters, results['admm']['objectives'], 
                    'r--', linewidth=2, label='ADMM', marker='s', markersize=4)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error convergence
    ax2.set_title('Localization Error Convergence', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('RMSE (log scale)')
    
    if results['mps']['errors']:
        mps_iters = np.arange(0, len(results['mps']['errors']) * 10, 10)
        ax2.semilogy(mps_iters, results['mps']['errors'], 
                    'b-', linewidth=2, label=f"MPS (final: {results['mps']['errors'][-1]:.4f})")
    
    if results['admm']['errors']:
        admm_iters = np.arange(0, len(results['admm']['errors']) * 10, 10)
        ax2.semilogy(admm_iters, results['admm']['errors'], 
                    'r--', linewidth=2, label=f"ADMM (final: {results['admm']['errors'][-1]:.4f})")
    
    # Add CRLB bound
    ax2.axhline(y=results['crlb'], color='green', linestyle=':', 
               linewidth=2, label=f'CRLB: {results["crlb"]:.4f}')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Convergence rate analysis
    ax3.set_title('Convergence Rate Analysis', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Relative Change')
    
    if len(results['mps']['objectives']) > 1:
        mps_changes = np.abs(np.diff(results['mps']['objectives'])) / np.abs(results['mps']['objectives'][:-1])
        ax3.semilogy(mps_iters[1:], mps_changes, 'b-', linewidth=2, label='MPS')
    
    if len(results['admm']['objectives']) > 1:
        admm_changes = np.abs(np.diff(results['admm']['objectives'])) / np.abs(results['admm']['objectives'][:-1])
        ax3.semilogy(admm_iters[1:], admm_changes, 'r--', linewidth=2, label='ADMM')
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance metrics
    ax4.axis('off')
    
    metrics_text = f"""Performance Comparison (Real Data):
    
    MPS Algorithm:
      • Iterations: {results['mps']['iterations']}
      • Final Error: {results['mps']['errors'][-1] if results['mps']['errors'] else 'N/A':.4f}
      • Time: {results['mps']['time']:.2f}s
      • Converged: {results['mps']['converged']}
    
    ADMM Algorithm:
      • Iterations: {results['admm']['iterations']}
      • Final Error: {results['admm']['errors'][-1] if results['admm']['errors'] else 'N/A':.4f}
      • Time: {results['admm']['time']:.2f}s
      • Converged: {results['admm']['converged']}
    
    Relative Performance:
      • Error Ratio: {results['admm']['errors'][-1] / results['mps']['errors'][-1] if results['mps']['errors'] and results['admm']['errors'] else 'N/A':.2f}x
      • Speed Ratio: {results['admm']['iterations'] / results['mps']['iterations']:.2f}x
      • CRLB Gap (MPS): {(results['mps']['errors'][-1] - results['crlb']) / results['crlb'] * 100 if results['mps']['errors'] else 'N/A':.1f}%
    """
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.suptitle('Algorithm Convergence Analysis - Real Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/convergence_comparison_real.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/convergence_comparison_real.png")

def generate_matrix_structure_figure():
    """Figure 3: Matrix Structure Visualization"""
    print("\nGenerating Figure 3: Matrix Structures...")
    
    # Generate actual matrix from small network
    params = {
        'n_sensors': 10,
        'n_anchors': 3,
        'd': 2,
        'communication_range': 0.5,
        'noise_factor': 0.05,
        'gamma': 0.999,
        'max_iter': 50
    }
    
    # Create MPS instance to get actual matrix
    mps = MPSSensorNetwork(params)
    mps.generate_network()
    M = mps._create_matrix_M()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 2-Block Matrix Structure
    ax1.set_title('2-Block Matrix M Structure', fontsize=12, fontweight='bold')
    im1 = ax1.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Row Index')
    
    # Add block boundaries
    n = params['n_sensors']
    ax1.axhline(y=n-0.5, color='black', linewidth=2)
    ax1.axvline(x=n-0.5, color='black', linewidth=2)
    ax1.text(n/2, -1, 'Block 1', ha='center', fontsize=10, fontweight='bold')
    ax1.text(3*n/2, -1, 'Block 2', ha='center', fontsize=10, fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Sparsity Pattern
    ax2.set_title('Sparsity Pattern (Non-zero elements)', fontsize=12, fontweight='bold')
    sparsity = (M != 0).astype(float)
    ax2.imshow(sparsity, cmap='Greys', vmin=0, vmax=1)
    ax2.set_xlabel('Column Index')
    ax2.set_ylabel('Row Index')
    
    # Add statistics
    nnz = np.count_nonzero(M)
    total = M.size
    sparsity_ratio = 1 - nnz/total
    ax2.text(0.02, 0.98, f'Sparsity: {sparsity_ratio:.1%}\nNNZ: {nnz}/{total}', 
            transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Eigenvalue Distribution
    ax3.set_title('Eigenvalue Distribution', fontsize=12, fontweight='bold')
    eigenvalues = np.linalg.eigvals(M)
    ax3.scatter(eigenvalues.real, eigenvalues.imag, alpha=0.6, s=50)
    ax3.set_xlabel('Real Part')
    ax3.set_ylabel('Imaginary Part')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.axvline(x=0, color='k', linewidth=0.5)
    
    # Add unit circle for reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.3, label='Unit Circle')
    ax3.legend()
    ax3.set_aspect('equal')
    
    # 4. Row Stochasticity Check
    ax4.set_title('Doubly Stochastic Property', fontsize=12, fontweight='bold')
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    
    x = np.arange(len(row_sums))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, row_sums, width, label='Row Sums', alpha=0.7)
    bars2 = ax4.bar(x + width/2, col_sums, width, label='Column Sums', alpha=0.7)
    
    ax4.axhline(y=1.0, color='red', linestyle='--', label='Target (1.0)')
    ax4.set_xlabel('Index')
    ax4.set_ylabel('Sum')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Check if doubly stochastic
    is_doubly_stochastic = np.allclose(row_sums, 1) and np.allclose(col_sums, 1)
    ax4.text(0.02, 0.98, f'Doubly Stochastic: {is_doubly_stochastic}', 
            transform=ax4.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='green' if is_doubly_stochastic else 'red', 
                     alpha=0.3))
    
    plt.suptitle('Matrix Structure Analysis - Real Implementation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/matrix_structure_real.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/matrix_structure_real.png")

def generate_scalability_analysis_figure():
    """Figure 4: Scalability Analysis with Real Data"""
    print("\nGenerating Figure 4: Scalability Analysis...")
    
    # Test different network sizes
    network_sizes = [10, 20, 30, 40, 50]
    mps_times = []
    mps_iters = []
    mps_errors = []
    admm_times = []
    admm_iters = []
    admm_errors = []
    
    for n in network_sizes:
        print(f"  Testing n={n} sensors...")
        params = {
            'n_sensors': n,
            'n_anchors': max(3, n // 5),
            'd': 2,
            'communication_range': 0.3,
            'noise_factor': 0.05,
            'max_iter': 200,
            'tol': 1e-3
        }
        
        results = run_algorithm_comparison(params)
        
        mps_times.append(results['mps']['time'])
        mps_iters.append(results['mps']['iterations'])
        mps_errors.append(results['mps']['errors'][-1] if results['mps']['errors'] else np.nan)
        
        admm_times.append(results['admm']['time'])
        admm_iters.append(results['admm']['iterations'])
        admm_errors.append(results['admm']['errors'][-1] if results['admm']['errors'] else np.nan)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Computation time vs network size
    ax1.set_title('Computation Time Scaling', fontsize=12, fontweight='bold')
    ax1.plot(network_sizes, mps_times, 'b-o', linewidth=2, label='MPS', markersize=8)
    ax1.plot(network_sizes, admm_times, 'r--s', linewidth=2, label='ADMM', markersize=8)
    ax1.set_xlabel('Number of Sensors')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Iterations vs network size
    ax2.set_title('Iterations to Convergence', fontsize=12, fontweight='bold')
    ax2.plot(network_sizes, mps_iters, 'b-o', linewidth=2, label='MPS', markersize=8)
    ax2.plot(network_sizes, admm_iters, 'r--s', linewidth=2, label='ADMM', markersize=8)
    ax2.set_xlabel('Number of Sensors')
    ax2.set_ylabel('Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final error vs network size
    ax3.set_title('Final Error vs Network Size', fontsize=12, fontweight='bold')
    ax3.plot(network_sizes, mps_errors, 'b-o', linewidth=2, label='MPS', markersize=8)
    ax3.plot(network_sizes, admm_errors, 'r--s', linewidth=2, label='ADMM', markersize=8)
    ax3.set_xlabel('Number of Sensors')
    ax3.set_ylabel('Final RMSE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency ratio
    ax4.set_title('Relative Efficiency (ADMM/MPS)', fontsize=12, fontweight='bold')
    
    time_ratio = np.array(admm_times) / np.array(mps_times)
    iter_ratio = np.array(admm_iters) / np.array(mps_iters)
    error_ratio = np.array(admm_errors) / np.array(mps_errors)
    
    x = np.arange(len(network_sizes))
    width = 0.25
    
    bars1 = ax4.bar(x - width, time_ratio, width, label='Time Ratio', alpha=0.7)
    bars2 = ax4.bar(x, iter_ratio, width, label='Iteration Ratio', alpha=0.7)
    bars3 = ax4.bar(x + width, error_ratio, width, label='Error Ratio', alpha=0.7)
    
    ax4.set_xlabel('Network Size')
    ax4.set_ylabel('Ratio (ADMM/MPS)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(network_sizes)
    ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Scalability Analysis - Real Performance Data', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/scalability_analysis_real.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/scalability_analysis_real.png")

def generate_crlb_comparison_figure():
    """Figure 5: CRLB Comparison with Real Data"""
    print("\nGenerating Figure 5: CRLB Comparison...")
    
    # Test different noise levels
    noise_levels = np.linspace(0.01, 0.15, 8)
    mps_errors = []
    admm_errors = []
    crlb_bounds = []
    
    params = {
        'n_sensors': 30,
        'n_anchors': 6,
        'd': 2,
        'communication_range': 0.4,
        'max_iter': 150,
        'tol': 1e-3
    }
    
    for noise in noise_levels:
        print(f"  Testing noise={noise:.3f}...")
        params['noise_factor'] = noise
        
        results = run_algorithm_comparison(params)
        
        mps_errors.append(results['mps']['errors'][-1] if results['mps']['errors'] else np.nan)
        admm_errors.append(results['admm']['errors'][-1] if results['admm']['errors'] else np.nan)
        crlb_bounds.append(results['crlb'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Error vs noise level
    ax1.set_title('Localization Error vs Noise Level', fontsize=14, fontweight='bold')
    ax1.plot(noise_levels, mps_errors, 'b-o', linewidth=2, label='MPS', markersize=8)
    ax1.plot(noise_levels, admm_errors, 'r--s', linewidth=2, label='ADMM', markersize=8)
    ax1.plot(noise_levels, crlb_bounds, 'g:', linewidth=2, label='CRLB (Lower Bound)', markersize=6)
    ax1.set_xlabel('Noise Factor')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gap from CRLB
    ax2.set_title('Gap from Theoretical Bound (CRLB)', fontsize=14, fontweight='bold')
    
    mps_gap = (np.array(mps_errors) - np.array(crlb_bounds)) / np.array(crlb_bounds) * 100
    admm_gap = (np.array(admm_errors) - np.array(crlb_bounds)) / np.array(crlb_bounds) * 100
    
    ax2.plot(noise_levels, mps_gap, 'b-o', linewidth=2, label='MPS', markersize=8)
    ax2.plot(noise_levels, admm_gap, 'r--s', linewidth=2, label='ADMM', markersize=8)
    ax2.axhline(y=0, color='green', linestyle=':', linewidth=2, alpha=0.5, label='CRLB (0% gap)')
    ax2.set_xlabel('Noise Factor')
    ax2.set_ylabel('Gap from CRLB (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    avg_mps_gap = np.mean(mps_gap)
    avg_admm_gap = np.mean(admm_gap)
    stats_text = f"Average Gap:\nMPS: {avg_mps_gap:.1f}%\nADMM: {avg_admm_gap:.1f}%"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('CRLB Analysis - Real Algorithm Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/crlb_comparison_real.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/crlb_comparison_real.png")

def generate_all_figures():
    """Generate all figures with real data"""
    print("\n" + "="*60)
    print("GENERATING ALL FIGURES WITH REAL DATA")
    print("No mock data - all from actual algorithm execution")
    print("="*60)
    
    # Create output directory
    os.makedirs('figures', exist_ok=True)
    
    # Generate each figure
    generate_network_topology_figure()
    generate_convergence_comparison_figure()
    generate_matrix_structure_figure()
    generate_scalability_analysis_figure()
    generate_crlb_comparison_figure()
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("Location: figures/")
    print("All data from actual MPS and ADMM algorithm execution")
    print("="*60)

if __name__ == "__main__":
    generate_all_figures()