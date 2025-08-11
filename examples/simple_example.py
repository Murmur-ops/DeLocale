#!/usr/bin/env python3
"""
Simple example of using MPS algorithm for sensor network localization
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.admm_algorithm import DecentralizedADMM
import matplotlib.pyplot as plt

def create_sample_network(n_sensors=20, n_anchors=4, seed=42):
    """Create a sample sensor network for testing"""
    np.random.seed(seed)
    
    # Generate true positions
    true_positions = {}
    for i in range(n_sensors):
        pos = np.random.normal(0.5, 0.2, 2)
        true_positions[i] = np.clip(pos, 0, 1)
    
    # Generate anchor positions
    anchor_positions = np.random.uniform(0, 1, (n_anchors, 2))
    
    return true_positions, anchor_positions

def run_simple_example():
    """Run a simple sensor localization example"""
    
    print("=" * 60)
    print("Simple Sensor Network Localization Example")
    print("=" * 60)
    
    # Problem configuration
    problem_params = {
        'n_sensors': 20,
        'n_anchors': 4,
        'd': 2,
        'communication_range': 0.4,
        'noise_factor': 0.05,
        'alpha_admm': 150.0,
        'max_iter': 200,
        'tol': 1e-4
    }
    
    # Create network
    true_positions, anchor_positions = create_sample_network(
        problem_params['n_sensors'],
        problem_params['n_anchors']
    )
    
    # Run ADMM algorithm (as a simple example)
    print("\nRunning ADMM algorithm...")
    admm = DecentralizedADMM(problem_params)
    admm.generate_network(true_positions, anchor_positions)
    
    results = admm.run_admm()
    
    # Print results
    print(f"\nResults:")
    print(f"  Converged: {results['converged']}")
    print(f"  Iterations: {results['iterations']}")
    if results['errors']:
        print(f"  Final Error: {results['errors'][-1]:.4f}")
    if results['objectives']:
        print(f"  Final Objective: {results['objectives'][-1]:.4f}")
    
    # Visualize results
    visualize_results(true_positions, results['final_positions'], 
                     anchor_positions, problem_params)
    
    return results

def visualize_results(true_positions, estimated_positions, 
                      anchor_positions, params):
    """Visualize the localization results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot true network
    ax1.set_title('True Sensor Positions')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    
    # Plot sensors
    for i, pos in true_positions.items():
        ax1.scatter(pos[0], pos[1], c='blue', s=50, alpha=0.6)
        ax1.text(pos[0], pos[1], str(i), fontsize=8)
    
    # Plot anchors
    for i, pos in enumerate(anchor_positions):
        ax1.scatter(pos[0], pos[1], c='red', s=100, marker='^')
        ax1.text(pos[0], pos[1], f'A{i}', fontsize=8)
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Sensors', 'Anchors'])
    
    # Plot estimated network
    ax2.set_title('Estimated Sensor Positions')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    
    # Plot estimated positions with error lines
    for i in true_positions:
        true_pos = true_positions[i]
        if i in estimated_positions:
            est_pos = estimated_positions[i]
            
            # Plot error line
            ax2.plot([true_pos[0], est_pos[0]], 
                    [true_pos[1], est_pos[1]], 
                    'gray', alpha=0.3, linewidth=1)
            
            # Plot true position
            ax2.scatter(true_pos[0], true_pos[1], 
                       c='blue', s=50, alpha=0.3, marker='o')
            
            # Plot estimated position
            ax2.scatter(est_pos[0], est_pos[1], 
                       c='green', s=50, alpha=0.8, marker='x')
    
    # Plot anchors
    for i, pos in enumerate(anchor_positions):
        ax2.scatter(pos[0], pos[1], c='red', s=100, marker='^')
        ax2.text(pos[0], pos[1], f'A{i}', fontsize=8)
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Error', 'True', 'Estimated', 'Anchors'])
    
    plt.suptitle(f'Sensor Network Localization (n={params["n_sensors"]}, anchors={params["n_anchors"]})')
    plt.tight_layout()
    plt.savefig('localization_example.png', dpi=150)
    plt.show()
    
    print(f"\nVisualization saved to localization_example.png")

if __name__ == "__main__":
    results = run_simple_example()