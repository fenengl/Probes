#!/bin/bash

#SBATCH --job-name=600K
#SBATCH --account=nn9299k
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
# #SBATCH --mem-per-cpu=3920M

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

# module --quiet purge
# # module load foss/2020a
# source /cluster/projects/nn9299k/software/punc++.sh

# module load Miniconda3/4.9.2
# # Set ${PS1} (needed by conda)
# export PS1=\$
# # Source the conda environment setup
# # ${EBROOTMINICONDA3} comes with the Miniconda module
# source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
# # Deactivate any spill-over environment from the login node
# conda deactivate &> /dev/null
# conda activate /cluster/projects/nn9299k/software/conda/envs/punc
module list

./interaction setup.ini --time.dt 3.190391E-09
