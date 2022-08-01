#!/bin/bash

#SBATCH --job-name=large
#SBATCH --account=nn9299k
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16000M
# #SBATCH --mem-per-cpu=3920M

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

module purge
module load foss/2020a


rm -f .quit .output .restartfile
mpirun ./mptetra > mptetra.log
