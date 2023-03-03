#!/bin/bash

#SBATCH --job-name=test
#SBATCH --account=nn9299k
#SBATCH --time=24:00:00
#SBATCH --nodes=4 --ntasks-per-node=1 --cpus-per-task=1
# #SBATCH --mem-per-cpu=16000M
# #SBATCH --mem-per-cpu=3920M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sadhi@uio.no

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

module purge
module load foss/2021b


rm -f .quit .output .restartfile
mpirun ./mptetra > mptetra.log
