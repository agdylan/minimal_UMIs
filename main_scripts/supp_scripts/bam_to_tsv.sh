# Description:
# This script processes the 10x output BAM file, extracting for each read the gene it mapped to (GX), cell barcode (CB), corrected UMI (UB), and raw UMI (UR).

input_bam="PATH_TO_10x_RUN/results_auto/outs/possorted_genome_bam.bam"
output_tsv="PATH_TO_OUTPUT.tsv"

samtools view "$input_bam" | \
awk '
    BEGIN {
        OFS = "\t"
        print "GX","CB","UB","UR"
    }
    {
        gx=""; cb=""; ub=""; ur=""
        for(i=12; i<=NF; i++) {
            if   ($i ~ /^GX:Z:/) gx = substr($i,6)
            else if($i ~ /^CB:Z:/) cb = substr($i,6)
            else if($i ~ /^UB:Z:/) ub = substr($i,6)
            else if($i ~ /^UR:Z:/) ur = substr($i,6)
        }
        if(gx != "" && cb != "" && ub != "")
            print gx, cb, ub, ur
    }
' > "$output_tsv"
