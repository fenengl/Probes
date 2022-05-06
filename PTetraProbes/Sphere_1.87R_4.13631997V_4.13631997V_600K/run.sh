#!/bin/bash

#SBATCH --job-name=eta80
#SBATCH --account=nn9299k
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16000M
# #SBATCH --mem-per-cpu=3920M

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

module purge
module load foss/2020a

mpirun ./mptetra > mptetra.log
