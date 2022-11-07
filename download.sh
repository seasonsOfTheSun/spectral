brew install wget
pip install gseapy
wget https://ndownloader.figshare.com/files/35020903
mkdir data
mkdir data/raw
mv 35020903 data/raw/sample_info.csv
# https://depmap.org/portal/download/
# CRISPR_gene_effect.csv
wget https://ndownloader.figshare.com/files/34990036
mv 34990036 data/raw/CRISPR_gene_effect.csv
