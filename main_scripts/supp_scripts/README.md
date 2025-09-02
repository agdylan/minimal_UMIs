# UMI Analysis - Directory Overview

This README describes several scripts used in the generation of figures for the manuscript. At the top of cells where figures are generated for the paper is the text "Figure generation for paper: fig:***"

## Directory Contents

- **README.md**  
    This file. Describes the purpose and contents of the directory.

- **general_utils.py**
    Utility functions for UMI analysis, including data processing and visualization helpers.

- **bam_to_tsv.sh**  
    Shell script for processing BAM file to extract UMI information.

- **supp_plots.ipynb**
    Majority of figure generation for supplement.

- **hamming_dist_sims.ipynb**
    Compare pairwise hamming distance between UB and UR.

- **MALAT1_anomaly_detection.ipynb**
    Detecting that MALAT1 is an outlier, and MA plot for observed minus expected counts.

- **fig1_plots.ipynb**
    Script for generating Figure 1 synthetic plots.