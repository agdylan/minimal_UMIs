"""
This script provides a command-line interface for running the non-uniform
UMI collision-aware estimator. It includes:

- an argument parser
- a main() function for clean execution
- optional verbose logging
- a simplified UMI probability generator
- a clear workflow: load data → compute probabilities → build estimator
  → generate corrected counts → save output
"""


import math
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
import anndata as ad
import scanpy as sc
import pysam
from tqdm import tqdm
from numba import njit
import argparse


### UMI probability generator
def generate_umi_probabilities(k=5, verbose=False):
    """Generates the non-uniform UMI probability vector for UMI length k"""
    
    if verbose:
        print(f"Generating UMI probabilities for UMI length {k}...")

    ## Set the per-nucleotide probabilities
    nt_probs = {'A': 0.23, 'C': 0.24, 'G': 0.21, 'T': 0.32}
    nt_logs = {nt: np.log(p) for nt, p in nt_probs.items()}

    ## Generate all possible UMIs of length k
    umis = itertools.product(nt_probs.keys(), repeat=k)

    ## Compute log probability of each UMI and exponentiate
    log_probs = []
    for umi in umis:
        log_prob = sum(nt_logs[base] for base in umi)
        log_probs.append(log_prob)

    probs = np.exp(log_probs)

    if verbose:
        print(f"UMI probabilities generated (length={len(probs)})")

    return probs


### Forward model function
@njit
def non_unif_forward_model(n, prob_arr, K):
    """
    Computes the forward model f(n). Jitted for speed.
    """
    return K - np.sum((1 - prob_arr)**n)


### Build lookup table function
@njit
def nonunif_build_lookup_table(Y_max, K, f_hat_values, n_values):
    """
    Builds the lookup table y → n using linear interpolation.
    This is compiled with numba for faster performance.
    """
    n_lookup_table = np.empty(Y_max + 1, dtype=np.float64)
    n_lookup_table[0] = 0.0

    f_hat_idx = 0

    for y in range(1, Y_max + 1):
        ## Move forward through f_hat_values until the interval containing y is found
        while f_hat_idx < len(f_hat_values) - 2 and f_hat_values[f_hat_idx + 1] < y:
            f_hat_idx += 1

        ## Pull the values needed for interpolation
        n_i = n_values[f_hat_idx]
        n_ip1 = n_values[f_hat_idx + 1]
        f_i = f_hat_values[f_hat_idx]
        f_ip1 = f_hat_values[f_hat_idx + 1]

        denom = f_ip1 - f_i
        if denom == 0:
            n_lookup_table[y] = n_i
        else:
            n_lookup_table[y] = n_i + (y - f_i) * (n_ip1 - n_i) / denom

    ## Handle the y = K case with a simple quadratic extrapolation
    if K <= Y_max:
        n1 = n_lookup_table[K - 1]
        n2 = n_lookup_table[K - 2]
        n3 = n_lookup_table[K - 3]
        n_lookup_table[K] = 3 * n1 - 3 * n2 + n3

    return n_lookup_table


### Non-uniform estimator generator
def generate_nonunif_estimator(prob_arr, k, Y_max, verbose=False):
    """
    Builds a fast collision-aware estimator.

    Returns:
        estimator(matrix): a function that maps observed counts to estimates
    """
    if verbose:
        print(f"Building non-uniform estimator (k={k}, Y_max={Y_max})...")

    K = 4**k
    prob_arr = prob_arr.astype(np.float64)

    ## Find an n_max large enough so that f(n_max) exceeds the needed range
    target_y = min(K - 1, Y_max)
    n_max = 2

    while non_unif_forward_model(n_max, prob_arr, K) <= target_y:
        n_max *= 2

    if verbose:
        print(f"Found n_max = {n_max}")

    ## Construct arrays of n-values and corresponding f(n)-values
    n_values = np.arange(1, n_max + 1, dtype=np.float64)
    f_hat_values = np.array([non_unif_forward_model(n, prob_arr, K) for n in n_values])

    ## Sanity check to ensure coverage
    if f_hat_values[-1] < target_y:
        raise ValueError("Interpolation grid insufficient.")

    ## Build lookup table (one-time cost)
    if verbose:
        print(f"Building lookup table...")
    n_lookup_table = nonunif_build_lookup_table(Y_max, K, f_hat_values, n_values)

    if verbose:
        print(f"Estimator lookup table constructed.")

    ## The closure that applies the estimator to matrix inputs
    def estimator(matrix):
        if np.any((matrix < 0) | (matrix > Y_max)):
            raise ValueError(f"Input values must be in [0, {Y_max}]")
        return n_lookup_table[matrix.astype(int)]

    return estimator



### Running estimator on adata 
def predict_with_collision_estimator(chosen_k, adata, verbose=False):
    """
    Full workflow: generate UMI probabilities, build estimator, predict counts.
    """
    if verbose:
        print("Generating UMI probabilities...")

    umi_probs = generate_umi_probabilities(k=chosen_k, verbose=verbose)

    if verbose:
        print("UMI probability vector generated.")

    ## Extract count matrix from AnnData object
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    original_matrix = X.copy()

    ## Build estimator
    Y_max = int(min(original_matrix.max(), 4**chosen_k))
    estimator = generate_nonunif_estimator(
        umi_probs, chosen_k, Y_max, verbose=verbose
    )

    if verbose:
        print("Applying estimator to matrix...")

    predicted_matrix = estimator(original_matrix)

    if verbose:
        print("Matrix estimation complete.")

    ## Build new AnnData object
    predicted_adata = ad.AnnData(X=predicted_matrix)
    predicted_adata.obs_names = adata.obs_names
    predicted_adata.var_names = adata.var_names

    if verbose:
        print("Created new AnnData object with corrected counts.")

    return predicted_adata

### Main funtion with argument parsing
def main():
    parser = argparse.ArgumentParser(
        description="Run UMI collision-aware estimator on an AnnData file."
    )
    parser.add_argument("--adata_path", type=str, required=True,
                        help="Path to input AnnData (.h5ad)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save corrected AnnData (.h5ad)")
    parser.add_argument("--k", type=int, default=5,
                        help="UMI length exponent k (default = 5)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable detailed prints")

    args = parser.parse_args()

    ## Load AnnData
    if args.verbose:
        print("Loading input AnnData...")

    adata = ad.read_h5ad(args.adata_path)

    if args.verbose:
        print(f"Loaded AnnData: shape = {adata.shape}")

    ## Run estimator
    predicted_adata = predict_with_collision_estimator(
        chosen_k=args.k,
        adata=adata,
        verbose=args.verbose
    )

    ## Save output
    predicted_adata.write_h5ad(args.output_path)

    if args.verbose:
        print(f"Saved corrected AnnData to: {args.output_path}")


### Entry point
if __name__ == "__main__":
    main()
