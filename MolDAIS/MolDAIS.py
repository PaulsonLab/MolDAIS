


import os
import time
import random
import numpy as np
import pandas as pd
import torch
import sys
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from botorch.models.model_list_gp_regression import ModelListGP

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll as fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound, ConstrainedExpectedImprovement
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

import pdb

from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement)

from minepy import MINE
from mordred import Calculator, descriptors

import warnings
warnings.filterwarnings("ignore")



class MolDAIS:
    def __init__(self, problem=None, optimizer_parameters=None, configuration=None, results=None):
        self.problem = problem or MolDAIS.Problem()
        self.optimizer_parameters = optimizer_parameters or MolDAIS.OptimizerParameters()
        self.results = results or MolDAIS.Results()  # Initialize the Results class
        self.configuration = configuration or MolDAIS.Configuration(self.problem, self.optimizer_parameters, self.results)
        self.problemname = self.configuration.set_problemname()  # Automatically set the problemname
        self.iteration = 0

    class Results:
        def __init__(self,X_full=torch.tensor([]), X=torch.tensor([]), y=torch.tensor([]), 
                    best_values=None, best_molecules=None):
            self.X_full = X_full
            self.X = X
            self.y = y
            self.best_values = best_values or []  # To store the best objective value at each iteration
            self.best_molecules = best_molecules or []  # To store the best molecule (SMILES) at each iteration







    class Problem:
        def __init__(self, smiles_search_space=None, descriptors_search_space=None, targets=None, experiment_name='None'):
            self.smiles_search_space = smiles_search_space  # pandas series of SMILES
            self.init_smiles_search_space = smiles_search_space  # pandas series of SMILES
            self.descriptors_search_space = descriptors_search_space  # torch.tensor for descriptors
            self.targets = targets  # torch.tensor for targets
            self.init_targets = targets  # torch.tensor for targets
            self.experiment_name = experiment_name

        def compute_descriptors(self):
            calc = Calculator(descriptors, ignore_3D=False)
            # Convert SMILES to RDKit Mol objects
            mols = [Chem.MolFromSmiles(smile) for smile in self.smiles_search_space]
            # Compute Mordred descriptors
            df = calc.pandas(mols,quiet=True,)
            # Remove non-numeric columns and columns with missing values
            df = df.select_dtypes(include=['float64', 'int64']).dropna(axis=1)
            # Convert the descriptors DataFrame to a torch tensor
            self.descriptors_search_space = torch.tensor(df.values, dtype=torch.float32)

    class OptimizerParameters:
        def __init__(self, iterations=100, initialization_budget=10, batch=1, sparsity_method='MIC',
                     num_sparsity_feats=10, frac_sparsity_feats=0.01, multi_objective=False, constrained=False,
                     acq_fun='EI', use_second_var=False, seed=0, custom_acq_fun=None, custom_sparsity_method=None,
                     custom_model_type=None, custom_fit_strategy=None):
            self.iterations = iterations
            self.initialization_budget = initialization_budget
            self.batch = batch
            self.sparsity_method = sparsity_method
            self.num_sparsity_feats = num_sparsity_feats
            self.frac_sparsity_feats = frac_sparsity_feats
            self.multi_objective = multi_objective
            self.constrained = constrained
            self.acq_fun = acq_fun
            self.use_second_var = use_second_var
            self.seed = seed
            self.custom_acq_fun = custom_acq_fun
            self.custom_sparsity_method = custom_sparsity_method
            self.custom_model_type = custom_model_type
            self.custom_fit_strategy = custom_fit_strategy
            if (self.multi_objective or self.constrained):
                self.use_second_var = True
                print('using second variable')

        def set_seed(self, seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    class Configuration:
        def __init__(self, problem, optimizer_parameters, results, model=None, sampled_smiles=None, problemname=None, iteration=0):
            self.results = results  # Access results attributes via self.results
            self.model = model
            self.sampled_smiles = sampled_smiles or []
            self.iteration = iteration
            self.problem = problem  # Store the passed problem
            self.optimizer_parameters = optimizer_parameters  # Store optimizer parameters

            sparsity_method = self.optimizer_parameters.sparsity_method
            if sparsity_method == 'SAAS':
                self.model_type = 'SAAS'
            else:
                self.model_type = self.optimizer_parameters.custom_model_type or 'GP'  # Default model type to GP

        def set_problemname(self):
            """Set the problem name for saving and loading."""
            # Build a descriptive filename that includes the problem name, sparsity method, model type, acquisition function, and iteration
            problem_name = self.problem.experiment_name or 'UnnamedProblem'
            sparsity_method = self.optimizer_parameters.sparsity_method
            model_type = self.model_type
            acq_fun = self.optimizer_parameters.acq_fun
            iteration_str = f"iter_{self.iteration}"
            seed = self.optimizer_parameters.seed
            # Generate descriptive filename
            filename = f"{problem_name}_sparsity-{sparsity_method}_model-{model_type}_acq-{acq_fun}_{iteration_str}_seed-{seed}.pkl"
            return filename

        def apply_sparsity(self):
            """Apply sparsity to the training data."""
            sampled_descriptors = self.results.X_full  # Access X_full from results
            self.results.X = torch.zeros((self.results.y.shape[-1], sampled_descriptors.shape[0], self.optimizer_parameters.num_sparsity_feats))  # Update X in results
            
            if self.optimizer_parameters.sparsity_method == 'Spearman':
                for j in range(self.results.y.shape[-1]):
                    # Compute Spearman correlation between descriptors and targets
                    correlations = []
                    for i in range(sampled_descriptors.shape[1]):
                        try:
                            corr, _ = torch.corrcoef(torch.stack([sampled_descriptors[:, i], self.results.y[:, j].squeeze()]))[0, 1]
                        except:
                            corr = 0
                        correlations.append(corr)

                    # Get absolute values of correlations and sort features by importance
                    correlations = torch.abs(torch.tensor(correlations))
                    top_indices = torch.argsort(correlations, descending=True)[:self.optimizer_parameters.num_sparsity_feats]

                    # Update X with selected features
                    self.results.X[j, :, :] = sampled_descriptors[:, top_indices].unsqueeze(0)

            elif self.optimizer_parameters.sparsity_method == 'MIC':
                # Initialize MIC object
                mine = MINE(alpha=0.6, c=15)
                for j in range(self.results.y.shape[-1]):
                    mic_scores = []
                    for i in range(sampled_descriptors.shape[1]):
                        feature = sampled_descriptors[:, i].numpy()
                        target = self.results.y[:, j].squeeze().numpy()
                        mine.compute_score(feature, target)
                        mic_scores.append(mine.mic())

                    # Convert scores to a tensor and select top features
                    mic_scores = torch.tensor(mic_scores)
                    top_indices = torch.argsort(mic_scores, descending=True)[:self.optimizer_parameters.num_sparsity_feats]

                    # Update X with selected features
                    self.results.X[j, :, :] = sampled_descriptors[:, top_indices].unsqueeze(0)

            elif self.optimizer_parameters.sparsity_method == 'SAAS':
                # Use the SAAS method for sparsity
                self.results.X = torch.zeros((self.results.y.shape[-1], sampled_descriptors.shape[0], sampled_descriptors.shape[1]))  # Update X in results
            
                for j in range(self.results.y.shape[-1]):
                  self.results.X[j,:,:] = sampled_descriptors

            return self.results.X

        def create_model(self):
            """Create and return a model depending on the selected type: GP or SAAS-GP."""
            model_type = self.model_type  # Use the provided model type or default to 'GP'

            if self.optimizer_parameters.use_second_var:
                if model_type == 'GP':
                    model = []
                    mll = []
                    for i in range(self.results.y.shape[-1]):
                        model.append(SingleTaskGP(train_X=self.results.X[i,:, :], train_Y=self.results.y[:, i].unsqueeze(1),
                                                  input_transform=Normalize(d=self.results.X.size(-1)), outcome_transform=Standardize(m=1)))
                        mll.append(ExactMarginalLogLikelihood(likelihood=model[i].likelihood, model=model[i]))
                    self.model = model
                    self.mll = mll
                elif model_type == 'SAAS':
                    model = []
                    for i in range(self.results.y.shape[-1]):
                        model.append(SaasFullyBayesianSingleTaskGP(train_X=self.results.X[i, :, :], train_Y=self.results.y[:, i].unsqueeze(1),
                                                                  input_transform=Normalize(d=self.results.X.size(-1)), outcome_transform=Standardize(m=1)))
                    self.model = model
                    self.mll = None
            else:
                if model_type == 'GP':
                    # Standard GP model
                    self.model = SingleTaskGP(
                        train_X=self.results.X[0, :, :],  # Access X from results
                        train_Y=self.results.y,  # Access y from results
                        input_transform=Normalize(d=self.results.X.shape[-1]),  # Normalize input features
                        outcome_transform=Standardize(m=1)  # Standardize the outputs
                    )
                    self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

                elif model_type == 'SAAS':
                    # SAAS-GP model
                    self.model = SaasFullyBayesianSingleTaskGP(
                        train_X=self.results.X[0, :, :],  # Access X from results
                        train_Y=self.results.y,  # Access y from results
                        input_transform=Normalize(d=self.results.X.shape[-1]),  # Normalize input features
                        outcome_transform=Standardize(m=1)  # Standardize the outputs
                    )
                    self.mll = None  # SAAS-GP uses a fully Bayesian fitting approach, no marginal likelihood
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

        def fit_model(self):
            """Fit the model based on its type."""
            model_type = self.model_type  # Use the provided model type or default to 'GP'

            if model_type == 'GP':
                if not self.optimizer_parameters.use_second_var:
                    fit_gpytorch_model(self.mll)
                else:
                    for mll_i in self.mll:
                        fit_gpytorch_model(mll_i)
                    self.model = ModelListGP(*self.model)

            elif model_type == 'SAAS':
                warmup_steps = 12
                num_samples = 25
                thinning = 16              
                if not self.optimizer_parameters.use_second_var:
                    fit_fully_bayesian_model_nuts(
                        self.model,
                        warmup_steps=warmup_steps,  # Number of warmup steps
                        num_samples=num_samples,   # Number of MCMC samples
                        thinning=thinning  # Thinning rate
                    )
                    self.model = self.model
                else:
                    for model_i in self.model:
                        fit_fully_bayesian_model_nuts(model_i, warmup_steps=warmup_steps, num_samples=num_samples, thinning=thinning, disable_progbar=False)
                    self.model = ModelListGP(*self.model)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        def get_sample(self):
            """Get the next sample based on the selected acquisition function."""
            # Define an acquisition function (EI or UCB)
            if self.optimizer_parameters.use_second_var:
                dim = self.results.X.shape[1]  # Access X from results
                if self.optimizer_parameters.multi_objective:
                    try:
                        partitioning = FastNondominatedPartitioning(
                            ref_point=torch.max(self.results.y, dim=0).values,  # Access y from results
                            Y=self.results.y,  # Access y from results
                        )
                    except:
                        pdb.set_trace()
                    f_acq = qExpectedHypervolumeImprovement(
                        model=self.model,
                        ref_point=torch.max(self.results.y, dim=0).values,
                        partitioning=partitioning)

                    acq = safe_eval(f_acq, self.results.X)  # Access X from results
                    max_acq = torch.max(acq)
                    max_acq_index = torch.argmax(acq)
                    next_sample_index = max_acq_index

                else:  # constrained
                    lwr_bound = 0
                    feas_vals = self.results.y[:, 1] <= lwr_bound  # Access y from results
                    try:
                        max_val = (self.results.y[:, 0] * feas_vals).max()  # Access y from results
                    except:
                        pdb.set_trace()
                    constraints = {1: (lwr_bound, None)}
                    f_acq = ConstrainedExpectedImprovement(self.model, max_val, 0, constraints)

                    acq = safe_eval(f_acq, self.results.X[0, :, :].unsqueeze(1))  # Access X from results
                    max_acq = torch.max(acq)
                    max_acq_index = torch.argmax(acq)
                    next_sample_index = max_acq_index

            else:
                if self.optimizer_parameters.acq_fun == 'EI':
                    acq_function = ExpectedImprovement(self.model, best_f=torch.max(self.results.y))  # Access y from results
                elif self.optimizer_parameters.acq_fun == 'UCB':
                    acq_function = UpperConfidenceBound(self.model, beta=2.0)
                else:
                    raise ValueError("Unknown acquisition function")

                with torch.no_grad():
                    acq_vals = acq_function(self.results.X[0, :, :].unsqueeze(1))  # Access X from results
                    next_sample_index = torch.argmax(acq_vals).item()

            return next_sample_index





        def save_iteration(self):
            """Save the current state of optimization."""
            iteration_data = {
                'X_full': self.results.X_full,
                'X': self.results.X,
                'y': self.results.y,
                'model': self.model,
                'best_values': self.results.best_values,
                'best_molecules': self.results.best_molecules,
                'iteration': self.iteration,
                'smiles_search_space': self.problem.smiles_search_space,
                'descriptors_search_space': self.problem.descriptors_search_space,
                'targets': self.problem.targets,
                'sampled_smiles': self.sampled_smiles
            }
            filename = self.set_problemname()
            with open(filename, 'wb') as f:
                torch.save(iteration_data, f)
            print(f"Iteration {self.iteration} saved as {filename}")

        def load_from_checkpoint(self, checkpoint_filename):
            """Load the state of optimization from a saved checkpoint."""
            with open(checkpoint_filename, 'rb') as f:
                checkpoint_data = torch.load(f)
            self.results.X_full = checkpoint_data['X_full']
            self.results.X = checkpoint_data['X']
            self.results.y = checkpoint_data['y']
            self.model = checkpoint_data['model']
            self.results.best_values = checkpoint_data['best_values']
            self.results.best_molecules = checkpoint_data['best_molecules']
            self.iteration = checkpoint_data['iteration']
            self.problem.smiles_search_space = checkpoint_data['smiles_search_space']
            self.problem.descriptors_search_space = checkpoint_data['descriptors_search_space']
            self.problem.targets = checkpoint_data['targets']
            self.sampled_smiles = checkpoint_data['sampled_smiles']
            print(f"Loaded checkpoint from {checkpoint_filename}")


        def get_max_y(self, y, all=False):
          if self.optimizer_parameters.multi_objective:
            y_ = torch.prod(y, dim=1)
            return max(y_) if not all else y_

          elif self.optimizer_parameters.constrained:
            feas = self.results.y[:,1]<=0
            return max(self.results.y[:,0]*feas)  if not all else self.results.y[:,0]*feas
          else:
            return max(self.results.y) if not all else self.results.y



        def optimize(self):
            """Run the optimization loop."""
            # Set the seed for reproducibility
            seed = self.optimizer_parameters.seed

            total_budget = self.optimizer_parameters.iterations
            init_budget = self.optimizer_parameters.initialization_budget

            # Check if training data is empty
            if self.results.X.numel() == 0:  # Check if X (training data) is empty
                print(f"Initializing with random {init_budget} samples from the search space.")

                # Randomly sample from the search space using initialization budget
                np.random.seed(seed)
                init_indices = np.random.choice(np.arange(len(self.problem.smiles_search_space)), init_budget, replace=False)

                # Add sampled SMILES and corresponding descriptors/targets to the training data
                sampled_smiles = [self.problem.smiles_search_space[idx] for idx in init_indices]
                self.results.X_full = self.problem.descriptors_search_space[init_indices]
                self.results.y = self.problem.targets[init_indices]
                self.results.X = self.apply_sparsity()

                # Remove the sampled data from the search space and targets
                self.problem.smiles_search_space = [smile for i, smile in enumerate(self.problem.smiles_search_space) if i not in init_indices]
                self.problem.descriptors_search_space = torch.cat([self.problem.descriptors_search_space[i].unsqueeze(0) for i in range(len(self.problem.descriptors_search_space)) if i not in init_indices])
                self.problem.targets = torch.cat([self.problem.targets[i].unsqueeze(0) for i in range(len(self.problem.targets)) if i not in init_indices])

                # Store the sampled SMILES for reference
                self.sampled_smiles.extend(sampled_smiles)

            best_value = self.get_max_y(self.results.y).item()
            best_smiles = self.sampled_smiles[torch.argmax(self.get_max_y(self.results.y, all=True))]

            for bo_iter in range(self.iteration, total_budget - init_budget):
                self.iteration = bo_iter
                print(f"Starting BO Iteration: {bo_iter + init_budget} / {total_budget}")
                self.create_model()
                self.fit_model()

                # Get the next sample
                next_sample_idx = self.get_sample()

                next_sample_value = self.problem.targets[next_sample_idx, :]
                if self.optimizer_parameters.use_second_var:
                    next_score = self.get_max_y(next_sample_value.unsqueeze(0))
                else:
                    next_score = next_sample_value

                # Track the best objective value and the corresponding molecule
                if next_score > best_value:
                    best_value = next_score
                    best_smiles = self.problem.smiles_search_space[next_sample_idx]
                try:
                    best_value = best_value.item()
                except:
                    pass
                self.results.best_values.append(best_value)
                self.results.best_molecules.append(best_smiles)

                # Update training data (X, y) with the new sample
                new_X_sample = self.problem.descriptors_search_space[next_sample_idx].unsqueeze(0)
                new_y_sample = self.problem.targets[next_sample_idx].unsqueeze(0)
                self.results.X_full = torch.cat((self.results.X_full, new_X_sample), dim=0)
                self.results.y = torch.cat((self.results.y, new_y_sample), dim=0)
                self.apply_sparsity()

                # Remove the selected sample from the search space and targets
                self.problem.smiles_search_space.pop(next_sample_idx)
                self.problem.descriptors_search_space = torch.cat([self.problem.descriptors_search_space[i].unsqueeze(0) for i in range(len(self.problem.descriptors_search_space)) if i != next_sample_idx], dim=0)
                self.problem.targets = torch.cat([self.problem.targets[i].unsqueeze(0) for i in range(len(self.problem.targets)) if i != next_sample_idx], dim=0)

                # Store the sampled SMILES for reference
                self.sampled_smiles.append(best_smiles)

                self.iteration = bo_iter + 1
                self.save_iteration()


        def plot_convergence(self):
            """Plot the best objective value at each iteration and display the best molecules."""
            zo = 0
            rd_opts = Draw.DrawingOptions()
            rd_opts.bgColor=None

            
            fig, ax = plt.subplots(figsize=(8,5))

            # Plot the convergence curve (best objective values)
            ax.plot(range(len(self.results.best_values)), self.results.best_values, marker='o', label='Best Objective Value')
            #ax.set_title('Convergence Plot of Best Objective Value')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Best Objective Value')
            ax.grid(True)

            top_row = self.get_max_y(self.problem.init_targets)
            bottom_row = min(self.get_max_y(self.problem.init_targets, all =True))
           
            plt.plot([0,len(self.results.best_values)], [top_row, top_row],'k:', label='Dataset min/max') 
            plt.plot([0,len(self.results.best_values)], [bottom_row, bottom_row],'k:') 

            # Draw molecules on the plot at various points --incomplete
            # - does not look good in many cases 
            # - consider changing frequency and size of molecule images
            if 0:
              value_ = -1E12
              for i, (value, smiles) in enumerate(zip(self.results.best_values, self.results.best_molecules)):
                  if value>value_:  
                      mol = Chem.MolFromSmiles(smiles)
                      if mol:
                          img = Draw.MolToImage(mol,options=rd_opts)
                          # Display the molecule image on the plot
                          imgbox = OffsetImage(img, zoom=0.25)
                          ab = AnnotationBbox(imgbox, (i, top_row), frameon=False, zorder=zo)
                          ax.add_artist(ab)
                  value_=value
            
            ax.legend()
            plt.tight_layout()
            plt.ylim(bottom_row*.9,top_row*1.1)
            plt.xlim(0,len(self.results.best_values)-1)
            plt.show()


# Helper function to update the test/train data
def update_test_train_data(next_sample_idx, smiles_search_space, descriptors_search_space):
    # Simulate an update to the test/train data for the sake of testing
    new_smiles_space = smiles_search_space[:next_sample_idx] + smiles_search_space[next_sample_idx + 1:]
    new_descriptors = torch.cat((descriptors_search_space[:next_sample_idx], descriptors_search_space[next_sample_idx + 1:]), dim=0)
    return new_smiles_space, new_descriptors


def safe_eval(f, x):
  n = len(x)
  nm = 1000
  splits = int(np.ceil(n/nm))
  obs = []
  for i in range(splits):
    if i-1 != splits:
      xi = x[nm*i:nm*(i+1)]
    else:
      xi = x[nm*i:]
    obs.append(f(xi).tolist())
  obs = torch.tensor(sum(obs, []))
  return obs


