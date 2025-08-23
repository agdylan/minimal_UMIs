import numpy as np
from numba import njit
import math



### Uniform forward model: from umi_utils import f
def f(n_vals, j):
    """Used for the uniform forward model."""
    return 4**j * (1 - (1 - 1 / 4**j)**n_vals)





### Uniform Method of Moments Estimator: from umi_utils import mom_estimator_unif
@njit
def _mom_estimator_unif_scalar(y, K):
    """
    Scalar core (Numba-compiled).
    Uses the same recursive rule for y == K.
    """
    if y == K:
        # one-step recursion: f(K) = f(K-1) + K
        return _mom_estimator_unif_scalar(K - 1.0, K) + K
    # general case
    denom = math.log(1.0 - 1.0 / K)
    return math.log(1.0 - y / K) / denom

@njit
def _mom_estimator_unif_array(y, K):
    """
    Numba-compiled loop over any-shaped array.
    """
    yr = y.ravel()
    out = np.empty(yr.size, np.float64)
    for i in range(yr.size):
        out[i] = _mom_estimator_unif_scalar(yr[i], K)
    return out.reshape(y.shape)

def mom_estimator_unif(y, K):
    """
    Public API unchanged. Accepts scalar/array-like y.
    """
    # cast to float array once
    arr = np.asarray(y, dtype=np.float64)
    K = float(K)
    if arr.ndim == 0:  # scalar y
        return _mom_estimator_unif_scalar(float(arr), K)
    return _mom_estimator_unif_array(arr, K)




### Non-uniform forward model: from umi_utils import non_unif_forward_model 
@njit
def non_unif_forward_model(n, prob_arr, K):
    """
    Calculates the forward model: f(n) = K - sum((1 - p)^n).
    This is our f(n) which is monotonically increasing with n.
    """
    return K - np.sum((1 - prob_arr)**n)




### Non-uniform Method of Moments Estimator, from umi_utils import generate_nonunif_estimator
@njit
def nonunif_build_lookup_table(Y_max, K, f_hat_values, n_values):
    """
    Builds the lookup table for n = f⁻¹(y) for y from 0 to Y_max.
    This is highly optimized by using a linear scan instead of repeated binary searches.
    """
    n_lookup_table = np.empty(Y_max + 1, dtype=np.float64)
    n_lookup_table[0] = 0.0  # Base case: y=0 corresponds to n=0

    f_hat_idx = 0
    # Linearly scan through y values and f_hat_values simultaneously
    for y in range(1, Y_max + 1):
        # Find the interval [f(n_i), f(n_{i+1})] that contains y
        # Since y is always increasing, we can just advance the f_hat_idx pointer
        while f_hat_idx < len(f_hat_values) - 2 and f_hat_values[f_hat_idx + 1] < y:
            f_hat_idx += 1

        # Perform linear interpolation within the found interval
        n_i = n_values[f_hat_idx]
        n_ip1 = n_values[f_hat_idx + 1]
        f_ni = f_hat_values[f_hat_idx]
        f_nip1 = f_hat_values[f_hat_idx + 1]
        
        denominator = f_nip1 - f_ni
        # Avoid division by zero if f_hat values are identical
        if denominator == 0:
            n_lookup_table[y] = n_i
        else:
            n_lookup_table[y] = n_i + (y - f_ni) * (n_ip1 - n_i) / denominator

    # Handle the special quadratic extrapolation case for y = K, if it's within our range
    if K <= Y_max:
        # We use the already interpolated values from the table, which is very fast
        n_k_minus_1 = n_lookup_table[K - 1]
        n_k_minus_2 = n_lookup_table[K - 2]
        n_k_minus_3 = n_lookup_table[K - 3]
        n_lookup_table[K] = 3 * n_k_minus_1 - 3 * n_k_minus_2 + n_k_minus_3
        
    return n_lookup_table

def generate_nonunif_estimator(prob_arr, k, Y_max, verbose=False):
    """
    Factory function to generate a fast Method of Moments estimator.
    Usage: estimator = generate_nonunif_estimator(prob_arr, k, Y_max)
           n_estimates = estimator(counts_matrix)

    Args:
        prob_arr (np.ndarray): Array of probabilities for the forward model.
        k (int): The exponent used to calculate K = 4**k.
        Y_max (int): The maximum count value observed in the data.
        verbose (bool): If True, enables verbose output for debugging.

    Returns:
        function: A fast estimator function that takes a matrix of counts and
                  returns the corresponding estimated n values.
    """
    K = 4**k
    prob_arr = prob_arr.astype(np.float64) # Ensure float64 for precision

    # 1. Find a sufficient n_max using a doubling search
    # We need to find n such that f(n) > min(K-1, Y_max)
    target_y = min(K - 1, Y_max)
    n_max = 2
    # The non_unif_forward_model is jitted, so this search is very fast
    while non_unif_forward_model(n_max, prob_arr, K) <= target_y:
        n_max *= 2
    
    if verbose:
        print(f"Found n_max = {n_max} where f(n_max)= {non_unif_forward_model(n_max, prob_arr, K)} > {target_y}")
        print(f"f(n_max/2)= {non_unif_forward_model(n_max//2, prob_arr, K)} < {target_y}")

    # 2. Create the grid for interpolation up to the found n_max
    n_values = np.arange(1, n_max + 1, dtype=np.float64)
    # This loop can be slow for large n_max, but it's a one-time cost
    f_hat_values = np.array([non_unif_forward_model(n, prob_arr, K) for n in n_values])

    # Check if the interpolation range is valid
    if f_hat_values[-1] < target_y:
        raise ValueError(
            f"Failed to find an n_max where f(n_max) > {target_y}. Error."
            f"f({n_max}) = {f_hat_values[-1]}. Consider increasing search limit."
        )

    # 3. Create the lookup table as a one-time upfront cost
    if verbose:
        print(f"Generating estimator lookup table for Y_max = {Y_max} using n_max = {n_max}...")
    n_lookup_table = nonunif_build_lookup_table(Y_max, K, f_hat_values, n_values)
    if verbose:
        print("Estimator generated successfully.")

    # 4. Define and return the fast estimator function (a closure)
    def estimator(matrix):
        """
        Applies the pre-computed MoM estimator to a matrix of counts.
        Values are clipped to the range [0, Y_max] to prevent indexing errors.
        """
        # Clip values to ensure they are valid indices for the lookup table
        if np.any((matrix < 0) | (matrix > Y_max)):
            raise ValueError(f"Estimator generated with max value {Y_max}. All values in the input matrix must be in the range [0, {Y_max}]. However, values range from {matrix.min()} to {matrix.max()}.")
        return n_lookup_table[matrix.astype(int)]

    return estimator