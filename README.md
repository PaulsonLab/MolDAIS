# MolDAIS: Molecular Descriptors with Actively Identified Subsets

MolDAIS is a powerful tool for efficient molecular property optimization through adaptive learning of sparse subspaces.

[![PyPI version](https://img.shields.io/pypi/v/MolDAIS.svg)](https://pypi.org/project/MolDAIS/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

MolDAIS provides a framework for accelerating black-box molecular property optimization by intelligently identifying relevant molecular subspaces. This approach significantly reduces the computational cost of exploring chemical space while maintaining optimization effectiveness.

## Installation

```bash
pip install MolDAIS
```

## Quick Start Example

More examples are presented in the jupyter notebook

```python
import torch
from MolDAIS.moldais import MolDAIS

# Step 1: Define your molecular search space
smiles_list = [
    "CCO", "CCC", "CCN", "CCCl", "CCCC", "CCCO", "CCCN", "ClCCCl",
    "CCOCC", "CCON", "CCCNCl", "CCCCCl", "CCCCN", "CCCCCO", "CCNCl",
    "CCNO", "ClCCl", "CCOO", "CCNN", "COCO"
]

# Step 2: Define target property values (e.g., logP)
logP_values = torch.tensor([0.5, 1.2, 0.8, 2.1, 3.0, 1.5, 2.3, 2.8, 0.9, 1.4,
                            2.5, 1.9, 1.1, 2.7, 2.6, 1.6, 1.8, 2.2, 1.3, 1.7], 
                            dtype=torch.float32)

# Step 3: Define the optimization problem
problem = MolDAIS.Problem(smiles_search_space=smiles_list,
                          targets=logP_values.unsqueeze(1),
                          experiment_name="LogP_Optimization_Test")
problem.compute_descriptors()

# Step 4: Configure optimizer parameters
optimizer_parameters = MolDAIS.OptimizerParameters(
    sparsity_method='MI',           # Feature selection method (Mutual Information)
    acq_fun='EI',                   # Acquisition function (Expected Improvement)
    num_sparsity_feats=10,          # Number of features to select
    multi_objective=False,          # Single-objective optimization
    constrained=False,              # No constraints
    total_sample_budget=7,          # Total evaluation budget
    initialization_budget=2,        # Initial points for model building
    seed=123                        # Random seed for reproducibility
)

# Step 5: Create and run the optimization
mol_dais = MolDAIS(problem=problem, optimizer_parameters=optimizer_parameters)
mol_dais.configuration.optimize()

# Step 6: Get results
print("Best molecules found:")
print(mol_dais.results.best_molecules)
print("Best property values:")
print(mol_dais.results.best_values)

# Step 7: Visualize optimization progress
mol_dais.configuration.plot_convergence()
```

## Features

- **Adaptive Feature Selection**: Intelligently identifies relevant molecular descriptors
- **Flexible Acquisition Functions**: Multiple strategies for exploring chemical space
- **Support for Constraints**: Handle constrained optimization problems
- **Multi-objective Optimization**: Optimize multiple molecular properties simultaneously

## Planned Features
- **Custom Callable Functions**: Upcoming support for custom python wrapped functions 
- **Human-in-the-loop**: Upcoming support for interactive optimization

## Usage Options

### With a SMILES List and Callable Function

```python
# Coming soon - Human-in-the-loop optimization capabilities
```

## Citation

Submitted to RSC Digital Discovery. Formal citation coming soon...

Based on previous work:

```bibtex
@misc{sorourifar2024accelerating,
      title={Accelerating Black-Box Molecular Property Optimization by Adaptively Learning Sparse Subspaces}, 
      author={Farshud Sorourifar and Thomas Banker and Joel A. Paulson},
      year={2024},
      eprint={2401.01398},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```

## Contact

For questions, support, or collaboration opportunities:  
**Farshud Sorourifar** - sorourifar.1@osu.edu

## License

This project is licensed under the MIT License - see the LICENSE file for details.
