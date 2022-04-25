# To use PUNC++, ask Sigvald.

module load Miniconda3/4.9.2

# Set ${PS1} (needed by conda)
export PS1=\$

# Source the conda environment setup
# ${EBROOTMINICONDA3} comes with the Miniconda module
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh

# Deactivate any spill-over environment from the login node
conda deactivate &> /dev/null

conda activate /cluster/projects/nn9299k/software/conda/envs/punc
