#!/usr/bin/env python3
"""
Matrix-Parametrized Proximal Splitting (MPS) Algorithm
Standalone implementation for sensor network localization
"""

import numpy as np
from scipy.linalg import eigh
import time
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MPSSensorNetwork:
    """
    MPS algorithm for decentralized sensor network localization
    Based on the paper's 2-block matrix design with Sinkhorn-Knopp
    """
    
    def __init__(self, problem_params: dict):
        self.n_sensors = problem_params['n_sensors']
        self.n_anchors = problem_params['n_anchors']
        self.d = problem_params.get('d', 2)
        self.communication_range = problem_params.get('communication_range', 0.3)
        self.noise_factor = problem_params.get('noise_factor', 0.05)
        self.gamma = problem_params.get('gamma', 0.999)
        self.alpha = problem_params.get('alpha_mps', 10.0)
        self.max_iter = problem_params.get('max_iter', 200)
        self.tol = problem_params.get('tol', 1e-4)
        
        self.sensor_positions = {}
        self.anchor_positions = None
        self.distance_measurements = {}
        self.neighbors = {}
        
    def generate_network(self, true_positions=None, anchor_positions=None):
        """Generate or use provided network configuration"""
        
        # Generate or use provided positions
        if true_positions is None:
            np.random.seed(42)
            true_positions = {}
            for i in range(self.n_sensors):
                pos = np.random.normal(0.5, 0.2, self.d)
                true_positions[i] = np.clip(pos, 0, 1)
        
        if anchor_positions is None:
            self.anchor_positions = np.random.uniform(0, 1, (self.n_anchors, self.d))
        else:
            self.anchor_positions = anchor_positions
        
        # Initialize sensor positions with noise
        for i in range(self.n_sensors):
            self.sensor_positions[i] = true_positions[i] + 0.1 * np.random.randn(self.d)
        
        # Generate distance measurements with noise
        self._generate_measurements(true_positions)
        
    def _generate_measurements(self, true_positions):
        """Generate noisy distance measurements"""
        
        # Sensor-to-sensor measurements
        for i in range(self.n_sensors):
            self.neighbors[i] = []
            for j in range(self.n_sensors):
                if i != j:
                    true_dist = np.linalg.norm(true_positions[i] - true_positions[j])
                    if true_dist <= self.communication_range:
                        noisy_dist = true_dist * (1 + self.noise_factor * np.random.randn())
                        self.distance_measurements[(i, j)] = max(0.01, noisy_dist)
                        self.neighbors[i].append(j)
        
        # Sensor-to-anchor measurements
        for i in range(self.n_sensors):
            for k in range(self.n_anchors):
                true_dist = np.linalg.norm(true_positions[i] - self.anchor_positions[k])
                if true_dist <= self.communication_range:
                    noisy_dist = true_dist * (1 + self.noise_factor * np.random.randn())
                    self.distance_measurements[(i, f'anchor_{k}')] = max(0.01, noisy_dist)
    
    def _sinkhorn_knopp(self, A, max_iter=30, tol=1e-6):
        """Sinkhorn-Knopp algorithm for doubly stochastic matrix"""
        n = A.shape[0]
        A = A + 1e-10  # Avoid division by zero
        
        for _ in range(max_iter):
            # Row normalization
            row_sums = A.sum(axis=1, keepdims=True)
            A = A / np.maximum(row_sums, 1e-10)
            
            # Column normalization
            col_sums = A.sum(axis=0, keepdims=True)
            A = A / np.maximum(col_sums, 1e-10)
            
            # Check convergence
            if np.allclose(A.sum(axis=1), 1, atol=tol) and \
               np.allclose(A.sum(axis=0), 1, atol=tol):
                break
        
        return A
    
    def _create_matrix_M(self):
        """Create the 2-block matrix M using Sinkhorn-Knopp"""
        n = self.n_sensors
        
        # Create adjacency-based initial matrix
        A = np.zeros((n, n))
        for i in range(n):
            for j in self.neighbors[i]:
                A[i, j] = 1.0
            A[i, i] = len(self.neighbors[i])  # Diagonal weight
        
        # Normalize and make doubly stochastic
        A = A / (A.sum() + 1e-10)
        M_base = self._sinkhorn_knopp(A)
        
        # Create 2-block structure
        M = np.block([
            [self.gamma * M_base, (1 - self.gamma) * M_base],
            [(1 - self.gamma) * M_base, self.gamma * M_base]
        ])
        
        return M
    
    def _proximal_step(self, X, measured_dist, target_pos, alpha):
        """Proximal operator for distance constraint"""
        direction = X - target_pos
        current_dist = np.linalg.norm(direction)
        
        if current_dist < 1e-10:
            return X
        
        # Soft thresholding towards measured distance
        scale = 1 - alpha * (current_dist - measured_dist) / current_dist
        scale = np.clip(scale, 0, 2)
        
        return target_pos + scale * direction
    
    def run_mps(self):
        """Run the MPS algorithm"""
        
        # Create matrix M
        M = self._create_matrix_M()
        
        # Initialize variables
        Z = np.zeros((2 * self.n_sensors, self.d))
        for i in range(self.n_sensors):
            Z[i] = self.sensor_positions[i]
            Z[i + self.n_sensors] = self.sensor_positions[i]
        
        objectives = []
        errors = []
        iteration_times = []
        
        for iteration in range(self.max_iter):
            iter_start = time.time()
            
            # Store previous Z
            Z_old = Z.copy()
            
            # Proximal updates for each sensor
            for i in range(self.n_sensors):
                # Update based on sensor measurements
                for j in self.neighbors[i]:
                    if (i, j) in self.distance_measurements:
                        measured_dist = self.distance_measurements[(i, j)]
                        Z[i] = self._proximal_step(
                            Z[i], measured_dist, Z[j], 
                            self.alpha / (len(self.neighbors[i]) + 1)
                        )
                
                # Update based on anchor measurements
                for k in range(self.n_anchors):
                    key = (i, f'anchor_{k}')
                    if key in self.distance_measurements:
                        measured_dist = self.distance_measurements[key]
                        Z[i] = self._proximal_step(
                            Z[i], measured_dist, self.anchor_positions[k],
                            self.alpha / (len(self.neighbors[i]) + 1)
                        )
            
            # Matrix multiplication step
            Z = M @ Z
            
            # Update sensor positions (average of two blocks)
            for i in range(self.n_sensors):
                self.sensor_positions[i] = (Z[i] + Z[i + self.n_sensors]) / 2
            
            # Compute metrics every 10 iterations
            if iteration % 10 == 0:
                obj = self._compute_objective()
                objectives.append(obj)
                iteration_times.append(time.time() - iter_start)
                
                logger.info(f"MPS Iteration {iteration}: obj={obj:.6f}")
                
                # Check convergence
                if np.linalg.norm(Z - Z_old) < self.tol:
                    logger.info(f"MPS Converged at iteration {iteration}")
                    break
        
        return {
            'converged': iteration < self.max_iter - 1,
            'iterations': iteration + 1,
            'objectives': objectives,
            'errors': errors,
            'final_positions': dict(self.sensor_positions),
            'iteration_times': iteration_times
        }
    
    def _compute_objective(self):
        """Compute the objective function value"""
        total_error = 0.0
        count = 0
        
        # Sensor-to-sensor distance errors
        for (i, j), measured_dist in self.distance_measurements.items():
            if isinstance(j, int):  # Sensor-to-sensor
                actual_dist = np.linalg.norm(
                    self.sensor_positions[i] - self.sensor_positions[j]
                )
                total_error += (actual_dist - measured_dist) ** 2
                count += 1
        
        # Sensor-to-anchor distance errors
        for i in range(self.n_sensors):
            for k in range(self.n_anchors):
                key = (i, f'anchor_{k}')
                if key in self.distance_measurements:
                    measured_dist = self.distance_measurements[key]
                    actual_dist = np.linalg.norm(
                        self.sensor_positions[i] - self.anchor_positions[k]
                    )
                    total_error += (actual_dist - measured_dist) ** 2
                    count += 1
        
        return total_error / max(count, 1)

def test_mps():
    """Test the MPS implementation"""
    
    problem_params = {
        'n_sensors': 20,
        'n_anchors': 4,
        'd': 2,
        'communication_range': 0.4,
        'noise_factor': 0.05,
        'gamma': 0.999,
        'alpha_mps': 10.0,
        'max_iter': 200,
        'tol': 1e-4
    }
    
    # Create MPS solver
    mps = MPSSensorNetwork(problem_params)
    
    # Generate network
    print("Generating network for MPS...")
    mps.generate_network()
    
    # Run MPS
    print("Running MPS algorithm...")
    start_time = time.time()
    results = mps.run_mps()
    total_time = time.time() - start_time
    
    # Report results
    print(f"\nMPS Results:")
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['iterations']}")
    if results['objectives']:
        print(f"Final objective: {results['objectives'][-1]:.6f}")
    print(f"Total time: {total_time:.2f}s")
    
    return results

if __name__ == "__main__":
    test_mps()