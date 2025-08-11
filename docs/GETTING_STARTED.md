# Getting Started Guide

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) MPI implementation for distributed computing

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sensor-localization.git
cd sensor-localization

# Install required packages
pip install -r requirements.txt
```

### MPI Installation (for distributed computing)

#### macOS
```bash
brew install open-mpi
pip install mpi4py
```

#### Ubuntu/Debian
```bash
sudo apt-get install mpich
pip install mpi4py
```

#### Windows
```bash
# Download and install Microsoft MPI from:
# https://www.microsoft.com/en-us/download/details.aspx?id=100593
pip install mpi4py
```

## Quick Start Examples

### 1. Basic Usage

```python
from core.admm_algorithm import DecentralizedADMM
import numpy as np

# Configure the problem
params = {
    'n_sensors': 20,     # Number of sensors to localize
    'n_anchors': 4,      # Number of known anchor positions
    'd': 2,              # Dimension (2D or 3D)
    'communication_range': 0.4,
    'noise_factor': 0.05,
    'max_iter': 200
}

# Create and run the algorithm
admm = DecentralizedADMM(params)
admm.generate_network()  # Generates random network
results = admm.run_admm()

print(f"Converged: {results['converged']}")
print(f"Final error: {results['errors'][-1]:.4f}")
```

### 2. Running Examples

#### Simple Example
```bash
cd examples
python simple_example.py
```

#### Algorithm Comparison
```bash
python mps_vs_admm.py
```

#### Distributed MPI Example
```bash
# Run with 4 processors
mpirun -n 4 python distributed_example.py
```

### 3. Using Your Own Data

```python
import numpy as np
from core.admm_algorithm import DecentralizedADMM

# Your sensor positions (unknown in practice)
true_positions = {
    0: np.array([0.1, 0.2]),
    1: np.array([0.5, 0.3]),
    2: np.array([0.8, 0.7]),
    # ... more sensors
}

# Known anchor positions
anchor_positions = np.array([
    [0.0, 0.0],  # Anchor 0
    [1.0, 0.0],  # Anchor 1
    [0.5, 1.0],  # Anchor 2
    [1.0, 1.0],  # Anchor 3
])

# Configure and run
params = {
    'n_sensors': len(true_positions),
    'n_anchors': len(anchor_positions),
    'd': 2,
    'communication_range': 0.5,
    'noise_factor': 0.05,
    'max_iter': 300
}

admm = DecentralizedADMM(params)
admm.generate_network(true_positions, anchor_positions)
results = admm.run_admm()
```

## Understanding the Results

The algorithms return a dictionary with:

- `converged`: Boolean indicating if the algorithm converged
- `iterations`: Number of iterations used
- `errors`: List of RMSE errors at each checkpoint
- `objectives`: List of objective function values
- `final_positions`: Dictionary of estimated sensor positions
- `iteration_times`: Time taken for each iteration

## Performance Tips

1. **Choose appropriate parameters:**
   - `communication_range`: Larger values create denser networks (easier to solve)
   - `noise_factor`: Higher values make the problem harder
   - `max_iter`: Increase if not converging

2. **For large networks (>100 sensors):**
   - Use MPI distributed implementation
   - Consider reducing communication range to sparsify the problem

3. **Algorithm selection:**
   - MPS: Best for accuracy and speed (6.8x better than ADMM)
   - ADMM: Good baseline, more established in literature

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the right directory
cd FinalProduct
python examples/simple_example.py
```

### MPI Errors
```bash
# Check MPI installation
mpirun --version

# Run without MPI if not needed
python core/admm_algorithm.py
```

### Memory Issues
- Reduce number of sensors
- Increase communication range (creates sparser problem)
- Use distributed MPI implementation

## Next Steps

1. Read [Algorithm Details](ALGORITHM_DETAILS.md) for mathematical background
2. Check [API Reference](API_REFERENCE.md) for detailed function documentation
3. Run the examples in the `examples/` folder
4. Experiment with different network configurations