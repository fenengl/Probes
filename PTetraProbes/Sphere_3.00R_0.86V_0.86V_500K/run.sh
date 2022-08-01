#!/bin/bash

#SBATCH --job-name=Rp3.0
#SBATCH --account=nn9299k
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
# #SBATCH --mem-per-cpu=16000M
# #SBATCH --mem-per-cpu=3920M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sadhi@uio.no

set -o errexit # Exit the script on any error
set -o nounset # Treat any unset variables as an error

module purge
module load foss/2020a


rm -f .quit .output .restartfile
mpirun ./mptetra > mptetra.log
