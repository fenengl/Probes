#!/bin/bash

echo "Data table for PTetra runs"
echo "============================================"


module --quiet purge
module load Anaconda3/2020.11
export PS1=\$
source ${EBROOTANACONDA3}/etc/profile.d/conda.sh
conda deactivate &>/dev/null
conda activate


proberadius=(1.87 1.87 1.87 1.87 1.87 1.87 1.87 1.87 1.87 1.87)  #in terms of debye length

pvolt=(0.51704 1.03407999 1.55111999 2.06815998 2.58519998 3.10223997 3.61927997 4.13631997 4.65335996 5.17039996)

temp=600  #temperature in K


jobname=("eta10" "eta20" "eta30" "eta40" "eta50" "eta60" "eta70" "eta80" "eta90" "eta100")
numjobs=${#pvolt[@]}

for i in `seq 0 $(($numjobs-1))`
do
  projectdir="Sphere_"${proberadius[$i]}"R_"${pvolt[$i]}"V_"${pvolt[$i]}"V_"$temp"K"
  echo "JOB NAME: "${jobname[$i]} >> data_table.log
  echo "============================================" >> data_table.log
  ./plot.py $projectdir --FRE >> data_table.log
  echo "============================================" >> data_table.log
done
