import pysam
from collections import Counter
from typing import Tuple, Dict
import re, gzip, pathlib
import pickle
import dask.dataframe as dd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from dask.diagnostics import ProgressBar
import itertools



gtf_file='YOUR_PATH/cellranger/refdata-gex-GRCh38-2020-A/genes/genes.gtf'
out_fldr='YOUR_ANALYSIS_FOLDER'


def make_ensembl2symbol(gtf_path: str | pathlib.Path,
                        strip_version: bool = True) -> dict[str, str]:
    """
    Return {ENSEMBL_ID → SYMBOL} from a GTF file.

    Parameters
    ----------
    gtf_path : str | Path
        genes/genes.gtf or genes.gtf.gz from the same reference
        used by Cell Ranger.
    strip_version : bool, default True
        Remove trailing ".n" from the Ensembl IDs so that
        ENSG00000129103.16 → ENSG00000129103.
    """
    ens2sym = {}
    open_func = gzip.open if str(gtf_path).endswith((".gz", ".bgz")) else open
    pat = re.compile(r'(\w+) "([^"]+)"')      # parse attributes column

    with open_func(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):          # skip comments
                continue
            cols = line.rstrip().split("\t")
            if cols[2] != "gene":             # keep only 'gene' features
                continue
            attrs = dict(pat.findall(cols[8]))
            eid = attrs["gene_id"]
            if strip_version:
                eid = eid.split(".")[0]
            ens2sym[eid] = attrs["gene_name"]
    return ens2sym



def generate_and_save_mappings(
    gtf_path: str = gtf_file
):
    mapping_fpath = out_fldr+'/ensembl2symbol.pkl'
    map_e2s = make_ensembl2symbol(gtf_path, strip_version=True)
    map_s2e = {v: k for k, v in map_e2s.items()}   # reverse mapping if you need GN→GX

    # save the two dictionaries to a file
    with open(mapping_fpath, 'wb') as f:
        pickle.dump(map_e2s, f)
        pickle.dump(map_s2e, f)
        
        
        
def process_bam_to_tsv(
            tsv_path: str,
            cleaned_tsv_path: str,
            umi_counts_path: str
        ):
    """
    Process a BAM-derived TSV file to remove duplicates, multi-mapping reads,
    and compute UMI counts. Saves cleaned TSV and UMI counts to disk.
    """
    ProgressBar().register()

    # Read the data
    ddf = dd.read_csv(tsv_path, sep='\t')

    # Apply operations
    ddf = ddf.drop_duplicates()
    ddf = ddf.drop(columns=['UR'])
    ddf = ddf[~ddf['GX'].str.contains(';')]  # discard multi-mapping reads

    # Compute and save cleaned TSV
    cleaned_tsv = ddf.compute()
    cleaned_tsv.to_csv(cleaned_tsv_path, sep='\t', index=False)

    # Compute UMI counts and save
    umi_counts = ddf.groupby('UB').size().compute()
    umi_counts.sort_values(ascending=False).reset_index().to_csv(
        umi_counts_path, sep='\t', header=False, index=False
    )
    
    ######## compute UR tsv
    ddf = dd.read_csv(tsv_path, sep='\t')

    # Apply operations
    ddf = ddf.drop_duplicates()
    ddf = ddf[~ddf['GX'].str.contains(';')]  # discard multi-mapping reads

    # Compute and save cleaned TSV
    cleaned_tsv_UR = ddf.compute()
    cleaned_tsv_UR_path = cleaned_tsv_path.replace('cleaned_umis', 'cleaned_umis_UR')
    cleaned_tsv_UR.to_csv(cleaned_tsv_UR_path, sep='\t', index=False)

def run_bam_processing():
    ###### for 10k dataset
    tsv_path = out_fldr+'umis_10k.tsv'
    cleaned_tsv_path = out_fldr+'cleaned_umis_10k.tsv'
    umi_counts_path = out_fldr+'umi_counts_10k_overall.tsv'
    process_bam_to_tsv(tsv_path, cleaned_tsv_path, umi_counts_path)

    ######### for 1k dataset
    tsv_path = out_fldr+'umis_1k.tsv'
    cleaned_tsv_path = out_fldr+'cleaned_umis_1k.tsv'
    umi_counts_path = out_fldr+'umi_counts_1k_overall.tsv'
    process_bam_to_tsv(tsv_path, cleaned_tsv_path, umi_counts_path)


def compute_PWM(df):
    """
    Compute the Position Weight Matrix (PWM) efficiently for a given DataFrame of UMI counts.

    This optimized version avoids creating large intermediate DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ['UMI', 'count'] columns. UMIs must be length 12.

    Returns
    -------
    pd.DataFrame
        A 4x12 DataFrame representing the Position Frequency Matrix (PFM).
    """
    # Ensure we only work with relevant data
    df = df.loc[df['count'] > 0, ['UMI', 'count']]

    # Step A: Initialize an empty Position Count Matrix (PCM)
    # Using all bases ensures we don't miss any that might be absent at certain positions.
    bases = ['A', 'C', 'G', 'T']
    pcm = pd.DataFrame(0, index=bases, columns=range(12))

    # Step B: Loop through each of the 12 positions
    for position in pcm.columns:
        # Step B1: Efficiently get all bases at the current position using vectorized string slicing
        bases_at_pos = df['UMI'].str[position]

        # Step B2: Group the original counts by the bases at this position and sum them up
        position_counts = df['count'].groupby(bases_at_pos).sum()

        # Step B3: Update the PCM with the calculated counts for the current position
        # Using .reindex().fillna() is a robust way to handle bases that might not be present
        pcm[position] = position_counts.reindex(bases).fillna(0)

    # Step C: Normalize the PCM to get the final Position Frequency Matrix (PFM)
    pfm = pcm.div(pcm.sum(axis=0), axis=1)

    return pfm


def process_umi_counts(umi_df, extend_to_full = True, k=12):
    """
    Extend the UMI DataFrame to include all possible UMIs of length k, and add cols for counts and trailing Ts.
    """
    all_possible_umis = umi_df.copy()
    if extend_to_full:
        all_possible_umis = pd.DataFrame({'UMI': [''.join(x) for x in itertools.product('ACGT', repeat=k)]})
        all_possible_umis = all_possible_umis.merge(umi_df[['UMI', 'count']], on='UMI', how='left')
        all_possible_umis['count'] = all_possible_umis['count'].fillna(0)
        all_possible_umis = all_possible_umis.sort_values(by='count', ascending=False)
        all_possible_umis = all_possible_umis[all_possible_umis['UMI'] != 'T' * k]


    all_possible_umis['num_ts'] = all_possible_umis.UMI.str.count('T')
    all_possible_umis['num_as'] = all_possible_umis.UMI.str.count('A')
    all_possible_umis['num_cs'] = all_possible_umis.UMI.str.count('C')
    all_possible_umis['num_gs'] = all_possible_umis.UMI.str.count('G')
    all_possible_umis['prob'] = all_possible_umis['count'] / all_possible_umis['count'].sum()
    all_possible_umis['num_trailing_ts'] = all_possible_umis['UMI'].str.extract(r'(T*)$')[0].str.len()

    all_possible_umis['num_trailing_ts_squared'] = all_possible_umis['num_trailing_ts'] ** 2 / (11**2)*10
    
    return all_possible_umis
