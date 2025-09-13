# Unique molecular identifiers don't need to be unique: a collision-aware estimator for RNA-seq quantification

Dylan Agyemang, Rafael A. Irizarry, Tavor Z. Baharav. doi: https://doi.org/10.1101/2025.09.08.674884

## Overview 
RNA-sequencing (RNA-seq) relies on Unique Molecular Identifiers (UMIs) to accurately quantify gene expression after PCR amplification.
Longer UMIs minimize collisions---where two distinct transcripts are assigned the same UMI---at the cost of increased sequencing and synthesis costs. However, it is not clear how long UMIs need to be in practice, especially given the nonuniformity of the empirical UMI distribution.
In this work, we develop a method-of-moments estimator that accounts for UMI collisions, accurately quantifying gene expression and preserving downstream biological insights. We show that UMIs need not be unique: shorter UMIs can be used with a more sophisticated estimator. 

## Repository Structure  
- `figures/` – Figures generated from results for each RNA-seq dataset  
- `main_scripts/` – Core scripts required to reproduce experiments  



 
