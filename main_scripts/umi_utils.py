import numpy as np
from numba import njit
import math
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import brentq



### Uniform forward model: from umi_utils import f
def f(n_vals, j):
    """Used for the uniform forward model."""
    return 4**j * (1 - (1 - 1 / 4**j)**n_vals)





### Uniform Method of Moments Estimator
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

### Method of Moments estimator wih uniform distribution
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




### Non-uniform forward model
@njit
def non_unif_forward_model(n, prob_arr, K):
    """
    Calculates the forward model: f(n) = K - sum((1 - p)^n).
    This is our f(n) which is monotonically increasing with n.
    """
    m = prob_arr.size
    return m - np.sum((1 - prob_arr)**n)



### Lookup table construction for the non-uniform estimator
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


### Factory function to generate a fast Method of Moments estimator for the non-uniform case
def generate_mom_estimator(prob_arr, k, Y_max, verbose=False):
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


### Generating the umi_probs for the constant pwm 
def compute_cpwm_probs(max_len=12):
    # nucleotide probabilities (kept local to the function)
    nt_probs = {'A': 0.23, 'C': 0.24, 'G': 0.21, 'T': 0.32}
    nt_logs = {nt: np.log(p) for nt, p in nt_probs.items()}

    umi_dfs = {}
    for length in range(1, max_len + 1):
        # generate all UMIs for this length
        umis = [''.join(p) for p in itertools.product(nt_probs.keys(), repeat=length)]
        
        # compute log-probabilities (sum of logs)
        log_probs = [sum(nt_logs[base] for base in umi) for umi in umis]
        
        # exponentiate to get probabilities
        probs = np.exp(log_probs)
        
        # store in a DataFrame
        df = pd.DataFrame({'UMI': umis, 'prob': probs})
        umi_dfs[length] = df

    return umi_dfs

### Generating the umi_probs for the non-constant pwm
def compute_nonconstant_pwm_probs(pwm, max_len=12):

    umi_prob_dict = {}

    bases = ['A','C','G','T']
    base_to_idx = {b:i for i,b in enumerate(bases)}

    # convert pfm to numpy for fast indexing
    pwm = pwm.loc[bases].values   # shape (4,12)

    for k in range(1, max_len+1):

        umis = []
        probs = []

        for umi_tuple in itertools.product(bases, repeat=k):

            umi = ''.join(umi_tuple)

            prob = 1.0
            for pos, base in enumerate(umi):
                prob *= pwm[base_to_idx[base], pos]

            umis.append(umi)
            probs.append(prob)

        df = pd.DataFrame({
            "UMI": umis,
            "prob": probs
        })

        umi_prob_dict[k] = df

    return umi_prob_dict


### Generating the umi_probs for the synthesis failure model withc constant pwm
def compute_sf_cpwm_probs(s_pos_load, max_k=12):

    bp_probs = [0.23, 0.24, 0.21, 0.32]
    pA, pC, pG, pT = bp_probs

    s_pos_full = np.asarray(s_pos_load, dtype=float)

    distributions = {}

    for k in range(1, max_k + 1):

        df = pd.DataFrame({
            "UMI": ["".join(x) for x in itertools.product("ACGT", repeat=k)]
        })


        df["num_as"] = df["UMI"].str.count("A")
        df["num_cs"] = df["UMI"].str.count("C")
        df["num_gs"] = df["UMI"].str.count("G")
        df["num_ts"] = df["UMI"].str.count("T")

        df["num_trailing_ts"] = (
            df["UMI"]
            .str.extract(r"(T*)$")[0]
            .str.len()
            .astype(int)
        )

        s_pos = s_pos_full[:k]
        L = k

        t = df["num_trailing_ts"].to_numpy(dtype=int)

        nA = df["num_as"].to_numpy()
        nC = df["num_cs"].to_numpy()
        nG = df["num_gs"].to_numpy()
        nT = df["num_ts"].to_numpy()

        nT_nontrail = nT - t

        survival_prefix = np.cumprod(1.0 - s_pos)

        P0_survival_lookup = np.zeros(L + 1)
        P0_survival_lookup[:L] = survival_prefix[::-1]
        P0_survival_lookup[L] = 1.0

        P0_survival = P0_survival_lookup[t]

        P_base = (
            (pA ** nA)
            * (pC ** nC)
            * (pG ** nG)
            * (pT ** nT_nontrail)
        )

        bracket_lookup = np.zeros(L + 1)

        P_fail = s_pos
        P_T = (1.0 - s_pos) * pT

        bracket_lookup[0] = 1.0

        for tt in range(1, L + 1):
            j = L - tt
            bracket_lookup[tt] = P_fail[j] + P_T[j] * bracket_lookup[tt - 1]

        bracket = bracket_lookup[t]

        prob_trunc = P_base * P0_survival * bracket

        distributions[k] = df[["UMI"]].assign(prob=prob_trunc)

    return distributions

### Generating the umi_probs for the synthesis failure model with position-specific pwm
def compute_sf_ncpwm_probs(pwm, s_pos_load, max_k=12):
    """
    Compute UMI probabilities under a synthesis-failure model with
    position-specific base probabilities from a PWM.

    Parameters
    ----------
    pwm : pd.DataFrame
        Position weight matrix with rows A, C, G, T and columns corresponding
        to UMI positions. Entry (base, pos) is the probability of synthesizing
        that base at that position, conditional on synthesis not failing there.
    s_pos_load : array-like
        Position-specific synthesis failure probabilities.
    max_k : int, default=12
        Compute distributions for all UMI lengths k = 1, ..., max_k.

    Returns
    -------
    distributions : dict
        Dictionary where distributions[k] is a DataFrame with columns:
            - UMI  : all possible UMI strings of length k
            - prob : probability of observing that UMI under the model
    """

    # Fix the row order so indexing is consistent everywhere
    bases = ['A', 'C', 'G', 'T']
    base_to_idx = {b: i for i, b in enumerate(bases)}

    # Reorder PWM rows to match bases = ['A', 'C', 'G', 'T']
    pwm = pwm.loc[bases].values

    # Convert synthesis-failure probabilities to a NumPy array
    s_pos_full = np.asarray(s_pos_load, dtype=float)

    # Basic sanity checks: we cannot ask for longer UMIs than the PWM
    # or synthesis-failure vector can support
    if max_k > pwm.shape[1]:
        raise ValueError("max_k exceeds the number of PWM positions")

    if max_k > len(s_pos_full):
        raise ValueError("max_k exceeds the length of s_pos_load")

    distributions = {}

    # Build a separate distribution for each UMI length k
    for k in range(1, max_k + 1):

        umis = []
        probs = []

        # Restrict the synthesis-failure probabilities to the first k positions
        s_pos = s_pos_full[:k]
        L = k

        # survival_prefix[i] = probability that synthesis survives through
        # positions 0, 1, ..., i
        survival_prefix = np.cumprod(1.0 - s_pos)

        # Lookup table for the survival probability of the non-trailing prefix.
        # For a UMI with t trailing T's, the non-trailing prefix has length L - t,
        # so we need product_{i=0}^{L-t-1} (1 - s_i).
        P0_survival_lookup = np.zeros(L + 1)
        P0_survival_lookup[:L] = survival_prefix[::-1]
        P0_survival_lookup[L] = 1.0   # when t = L, the prefix length is 0

        # At position j:
        #   P_fail[j] = probability synthesis fails at j
        #   P_T[j]    = probability synthesis survives at j and emits T there
        P_fail = s_pos
        pT_pos = pwm[base_to_idx['T'], :L]
        P_T = (1.0 - s_pos) * pT_pos

        # bracket_lookup[t] stores the contribution from the trailing region
        # for a tail of exactly t trailing T's
        bracket_lookup = np.zeros(L + 1)
        bracket_lookup[0] = 1.0

        # Build the trailing-region recursion backward from the end
        # of the UMI:
        #
        # bracket_lookup[t] =
        #     P(fail at position j)
        #   + P(survive at j and emit T) * bracket_lookup[t-1]
        #
        # where j = L - t
        for tt in range(1, L + 1):
            j = L - tt
            bracket_lookup[tt] = P_fail[j] + P_T[j] * bracket_lookup[tt - 1]

        # Enumerate every possible UMI of length k
        for umi_tuple in itertools.product(bases, repeat=k):
            umi = ''.join(umi_tuple)

            # Number of trailing T's at the end of the observed UMI
            t = len(umi) - len(umi.rstrip('T'))

            # The prefix before the trailing T run
            prefix_len = L - t

            # Probability of the non-trailing prefix under the PWM:
            # multiply the probability of the observed base at each
            # position in the prefix
            base_prob = 1.0
            for pos in range(prefix_len):
                base = umi[pos]
                base_prob *= pwm[base_to_idx[base], pos]

            # Probability synthesis survives through the non-trailing prefix
            P0_survival = P0_survival_lookup[t]

            # Contribution from the trailing region
            bracket = bracket_lookup[t]

            # Final probability for this observed UMI
            prob = base_prob * P0_survival * bracket

            umis.append(umi)
            probs.append(prob)

        # Store the distribution for this UMI length
        df = pd.DataFrame({
            "UMI": umis,
            "prob": probs
        })

        distributions[k] = df.reset_index(drop=True)

    return distributions


### Generating the empirical umi distribution from the deduplicated data
def compute_emp_dist(dedup_df, max_len=12):
    """
    Takes as input a deduplicated dataframe and and computes empirical UMI distribution for 1 to 12
    """
    total = len(dedup_df)
    
    umi_df = {}
    for k in range(1, 13):
        prefixes = dedup_df['UMI'].str[:k]
        counts = prefixes.value_counts()
        
        probs = counts / total
        
        df_pk = (
        probs
        .rename('prob')
        .reset_index()
        .rename(columns={'index':'umi'})
    )

        umi_df[k] = df_pk
        
    return umi_df

# NOTE: The following code is the function for the MLE Estimator

### Converting the umi distribution dataframe into a dictionary for fast lookup
def load_p_k(df):
    """
    Convert a dataframe with columns ['UMI','prob']
    into a dictionary mapping umi -> probability.
    """

    p_dict = dict(zip(df['UMI'], df['prob']))

    return p_dict


### Generating the MLE for N using a robust Poissonized likelihood approach
def mle_N_fast(observed_umis, p_dict):
    """
    Robust Poissonized MLE for N.
    """
    
    # Edge case: no UMIs observed
    if len(observed_umis) == 0:
        return 0.0
    
    # Extract probabilities
    p_obs = np.array([p_dict[u] for u in observed_umis])
    sum_p_obs = p_obs.sum()
    
    # Stable score function
    def score(N):
        exp_term = np.exp(-p_obs * N)
        denom = 1 - exp_term
        denom = np.maximum(denom, 1e-15) ## Done to prevent division by zero according to floatin point arithmetic
        
        term1 = np.sum(p_obs * exp_term / denom)
        term2 = 1 - sum_p_obs
        
        return term1 - term2
    
    # Lower bound
    lower = 1e-12
    
    # Adaptive upper bound
    upper = max(10 * len(observed_umis), 10.0)
    
    # Expand upper until sign change
    while score(lower) * score(upper) > 0:
        upper *= 2
        if upper > 1e9:
            raise RuntimeError("Failed to bracket root.")
    
    return brentq(score, lower, upper)


    