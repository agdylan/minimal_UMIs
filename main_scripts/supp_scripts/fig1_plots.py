"""
Generate synthetic plots for Figure 1.
Usage: python fig1_plots.py
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def f(n_vals, k):
    """Expected number of unique UMIs given n_vals draws from a k-length UMI."""
    return 4**k * (1 - (1 - 1 / 4**k)**n_vals)


def pred(y_vals, k):
    """Method-of-Moments estimator: invert f to recover true N from observed unique UMI count."""
    predictions = np.zeros(len(y_vals))
    K = 4**k
    mask = y_vals == K

    def estimate(y, K):
        return np.log(1 - y / K) / np.log(1 - 1 / K)

    predictions[~mask] = estimate(y_vals[~mask], K)
    predictions[mask] = estimate(np.array([K - 1]), K) + K
    return predictions


def make_plots(out_dir='figure1_graphics'):
    print('Generating Figure 1 plots...')
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(42)

    k = 5
    K = 4**k
    n_max = 5 * 10**4
    n_values = np.logspace(0, np.log10(n_max), 200)
    n_min = 0.9

    max_n_sim = 5 * k * 4**k
    num_points = 30
    n_arr = np.logspace(0, np.log10(max_n_sim), num_points).astype(int)
    n_arr = np.unique(n_arr)

    # --- Fig 1b: naive UMI data for k=12 ---
    k_full = 12
    umis_full = np.random.choice(range(4**k_full), size=int(np.sum(n_arr)), replace=True)
    cumulative_counts = np.cumsum(n_arr.astype(int))
    starts = np.hstack(([0], cumulative_counts[:-1]))
    y_arr_full = np.array([
        len(set(umis_full[s:e])) for s, e in zip(starts, cumulative_counts)
    ])

    fig1b, ax1b = plt.subplots(figsize=(6, 6))
    ax1b.plot([n_min, n_max], [n_min, n_max], color='black', ls='--')
    sns.scatterplot(ax=ax1b, x=n_arr, y=y_arr_full, alpha=0.8, s=200)
    ax1b.set_xscale('log')
    ax1b.set_yscale('log')
    ax1b.set_xlabel(r'True number of sequences ($N$)', fontsize=14)
    ax1b.set_ylabel(r'Number of unique UMIs ($Y$)', fontsize=14)
    ax1b.tick_params(axis='both', labelsize=12)
    ax1b.set_xlim(n_min, n_max)
    ax1b.set_ylim(n_min, n_max)
    fig1b.savefig(f'{out_dir}/fig1b.pdf', bbox_inches='tight')

    # --- Fig 1d: k=5 UMI data with theoretical curves ---
    umis = np.random.choice(range(K), size=int(np.sum(n_arr)), replace=True)
    cumulative_counts = np.cumsum(n_arr.astype(int))
    starts = np.hstack(([0], cumulative_counts[:-1]))
    y_arr = np.array([
        len(set(umis[s:e])) for s, e in zip(starts, cumulative_counts)
    ])

    fig1d, ax1d = plt.subplots(figsize=(6, 6))
    ax1d.axhline(y=K, color='blue', label=r'$y=4^k$, total # of UMIs', ls='--')
    ax1d.plot([n_min, n_max], [n_min, n_max], color='black', label=r'$y=x$', ls='--')
    ax1d.plot(n_values, f(n_values, k), color='red', label='Expected # of unique UMIs', ls='--')
    sns.scatterplot(ax=ax1d, x=n_arr, y=y_arr, alpha=0.8, s=200,
                    label='Synthetic UMI data', legend=False)
    ax1d.axvline(x=K, label=r'$x=4^k$, total # of UMIs', ls='--', color='orange')
    ax1d.axvline(x=K * np.log(K), label=r'$x=k4^k$', ls='--', color='brown')
    ax1d.plot([max_n_sim, max_n_sim], [max_n_sim, f(max_n_sim, k)], color='green', ls='--')
    ax1d.hlines(max_n_sim, max_n_sim * 0.9, max_n_sim * 1.1, color='green')
    ax1d.hlines(f(max_n_sim, k), max_n_sim * 0.9, max_n_sim * 1.1, color='green')
    ax1d.text(max_n_sim * 0.7, np.sqrt(max_n_sim * K), 'Collisions',
              fontsize=12, color='green', ha='left', va='center', rotation=90)
    ax1d.set_xscale('log')
    ax1d.set_yscale('log')
    ax1d.set_xlabel(r'True number of sequences ($N$)', fontsize=14)
    ax1d.set_ylabel(r'Number of unique UMIs ($Y$)', fontsize=14)
    ax1d.tick_params(axis='both', labelsize=12)
    ax1d.set_xlim(n_min, n_max)
    ax1d.set_ylim(n_min, n_max)
    fig1d.savefig(f'{out_dir}/fig1d.pdf', bbox_inches='tight')

    # --- Fig 1e: Method-of-Moments estimation ---
    optimized_predictions = pred(y_arr, k)

    fig1e, ax1e = plt.subplots(figsize=(6, 6))
    ax1e.plot([n_min, n_max], [n_min, n_max], color='black', ls='--')
    sns.scatterplot(ax=ax1e, x=n_arr, y=optimized_predictions, alpha=0.8, s=200,
                    label='Method-of-Moments estimator', color='green', legend=False)
    ax1e.axvline(x=K, ls='--', color='orange')
    ax1e.axvline(x=K * np.log(K), ls='--', color='brown')
    ax1e.set_xscale('log')
    ax1e.set_yscale('log')
    ax1e.set_xlim(n_min, n_max)
    ax1e.set_ylim(n_min, n_max)
    ax1e.set_xlabel(r'True number of sequences ($N$)', fontsize=14)
    ax1e.set_ylabel(r'Predicted number of sequences ($\hat{N}$)', fontsize=14)
    ax1e.tick_params(axis='both', labelsize=12)
    fig1e.savefig(f'{out_dir}/fig1e.pdf', bbox_inches='tight')

    # --- Combined legend figure ---
    h1d, l1d = ax1d.get_legend_handles_labels()
    h1e, l1e = ax1e.get_legend_handles_labels()
    handles_dict = dict(zip(l1d + l1e, h1d + h1e))

    order = [
        'Synthetic UMI data',
        'Method-of-Moments estimator',
        'Expected # of unique UMIs',
        r'$y=x$',
        r'$y=4^k$, total # of UMIs',
        r'$x=4^k$, total # of UMIs',
        r'$x=k4^k$',
    ]
    handles = [handles_dict[lbl] for lbl in order if lbl in handles_dict]
    labels  = [lbl for lbl in order if lbl in handles_dict]

    fig_leg = plt.figure(figsize=(5, 0.8))
    fig_leg.legend(handles, labels, loc='center', frameon=False, ncol=3)
    fig_leg.savefig(f'{out_dir}/common_legend_fig1.pdf', bbox_inches='tight')

    print(f'Done. Figures saved to {out_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 1 plots.')
    parser.add_argument('--out_dir', default='figure1_graphics',
                        help='Directory to save figures (default: figure1_graphics).')
    args = parser.parse_args()
    make_plots(out_dir=args.out_dir)
