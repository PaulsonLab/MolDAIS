# MolDAIS (under development)
Molecular Descriptors with Actively Identified Subsets

A code for efficient molecular property optimization. This repository is under development. A cleaner version with tutorials and extended functionality (e.g., constraints, multi-objective, and human-in-the-loop) will be made available upon journal publication. 



## Test the code 
1. Build environment 
2. Run main.py

## Test on your data set
1. Add your <data>.csv file with a SMILES and <property> columns
2. Add src/config/log_p_test_exp/<new_exp_name>.json, e.g., change the "exp_name", "Data_loc", and "y_variable" fields in src/config/log_p_test_exp/log_P_test_exp.json
3. Change line 56 in main to point to <new_exp_name>.json
4. Run main.py


## Test with a smiles list and callable function (or Human in the loop)
- under development


## Citation

Sorourifar, Farshud, Thomas Banker, and Joel A. Paulson. "Accelerating Black-Box Molecular Property Optimization by Adaptively Learning Sparse Subspaces." arXiv preprint arXiv:2401.01398 (2024).

@misc{sorourifar2024accelerating,
      title={Accelerating Black-Box Molecular Property Optimization by Adaptively Learning Sparse Subspaces}, 
      author={Farshud Sorourifar and Thomas Banker and Joel A. Paulson},
      year={2024},
      eprint={2401.01398},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}

## Contact: 
sorourifar.1@osu.edu
