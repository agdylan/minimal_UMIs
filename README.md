# UMI Collision-Aware Estimator 

## Overview 
RNA sequencing (RNA-seq) is a widespread technique used to quantify gene expression. For accurate estimation, short nucleotide sequences called Unique Molecular Identifiers (UMIs) to ccllaspe PCR duplicates. Sequencing protocols use varying lengths and the relationship between UMI length and quantification accuracy has not be well studied. To address this, we develop a method-of-moments of estimator that accounts for collisions. When applied to single cell RNA-seq data our estimator demonstrates significant improvement in expression accuract and maintains biological insights in downstream tasks.

## Repository Structure 
- `data/` – UMI distributions for each length *k* (1–12)  
- `figures/` – Figures generated from results for each RNA-seq dataset  
- `main_scripts/` – Core scripts required to reproduce experiments  
- `old_scripts/` – Archived scripts retained for reference



 
