#!/bin/bash

echo "Automating job creation and queue for PTetra"
echo "============================================"

baseproject="demo_project"

temp=(500.00 562.50 625.00 687.50 750.00 812.50 875.00 937.50 1000.00 500.00 562.50 625.00 687.50 750.00 812.50 875.00 937.50 1000.00 500.00 562.50 625.00 687.50 750.00 812.50 875.00 937.50 1000.00 500.00 562.50 625.00 687.50 750.00 812.50 875.00 937.50 1000.00 500.00 562.50 625.00 687.50 750.00 812.50 875.00 937.50 1000.00)  #temperature in K
tempev=(0.0431 0.0485 0.0539 0.0592 0.0646 0.0700 0.0754 0.0808 0.0862 0.0431 0.0485 0.0539 0.0592 0.0646 0.0700 0.0754 0.0808 0.0862 0.0431 0.0485 0.0539 0.0592 0.0646 0.0700 0.0754 0.0808 0.0862 0.0431 0.0485 0.0539 0.0592 0.0646 0.0700 0.0754 0.0808 0.0862 0.0431 0.0485 0.0539 0.0592 0.0646 0.0700 0.0754 0.0808 0.0862)  #temperature in eV

debye=(0.004880 0.005176 0.005456 0.005722 0.005976 0.006220 0.006455 0.006682 0.006901 0.004880 0.005176 0.005456 0.005722 0.005976 0.006220 0.006455 0.006682 0.006901 0.004880 0.005176 0.005456 0.005722 0.005976 0.006220 0.006455 0.006682 0.006901 0.004880 0.005176 0.005456 0.005722 0.005976 0.006220 0.006455 0.006682 0.006901 0.004880 0.005176 0.005456 0.005722 0.005976 0.006220 0.006455 0.006682 0.006901)
proberadius=(0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.30 0.30 0.30 0.30 0.30 0.30 0.30 0.30 0.30 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 0.50 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.87 1.87 1.87 1.87 1.87 1.87 1.87 1.87 1.87)  #in terms of debye length
# pvolt=(4.5 5.5 6.5)
pvolt=(1.29 1.94 2.69 3.55 4.52 5.60 6.79 8.08 9.48 1.29 1.94 2.69 3.55 4.52 5.60 6.79 8.08 9.48 1.29 1.94 2.69 3.55 4.52 5.60 6.79 8.08 9.48 1.29 1.94 2.69 3.55 4.52 5.60 6.79 8.08 9.48 1.29 1.94 2.69 3.55 4.52 5.60 6.79 8.08 9.48)

netotIni=50000000

# jobname=("Rp2.0" "Rp5.0" "Rp10.0" "Rp20.0" "Rp100.0" "Rp120.0")
numjobs=${#pvolt[@]}
usrname="sadhi"
numproc=4
runtime=36

ndir="geometry"
basegeofile="sphere_demo.geo"

host=$(hostname)

module --quiet purge
if [[ $host = *fram.sigma2.no ]]; then
  module load Anaconda3/2019.07
else
  module load Anaconda3/2020.11
fi
export PS1=\$
source ${EBROOTANACONDA3}/etc/profile.d/conda.sh
conda deactivate &>/dev/null
conda activate /cluster/projects/nn9299k/software/conda/envs/punc


if [ -d "$ndir" ]
then
  echo "Directory exists"
  for i in `seq 0 $(($numjobs-1))`
  do
    geofile="sphere_"${proberadius[$i]}"R.geo"
    if [ -f "$ndir"/"$geofile" ]
    then
      echo "Geofile exists"
    else
      cp "$ndir"/"$basegeofile" "$ndir"/"$geofile"
      if [ -f "$ndir"/"$geofile" ]
      then
        echo ".geo file created"
        echo "Modifying .geo file"
        sed -i 's/.*debye = .*/debye = '"${debye[i]}"';/' "$ndir"/"$geofile"
        sed -i 's/.*r = .*/r = '"${proberadius[$i]}"'*debye;/' "$ndir"/"$geofile"
        echo ""
        gmsh -3 -format msh2 -optimize "$ndir"/"$geofile"
      fi
    fi
    echo "Converting .msh to .topo"
    meshfile="sphere_"${proberadius[$i]}"R.msh"
    cp "$ndir"/"msh2topo_bkp.dat" "$ndir"/"msh2topo.dat"
    sed -i 's/.*sphere_.*/'"$meshfile"'/' "$ndir"/"msh2topo.dat"
    cd $ndir
    ./msh2topo
    mv msh2topo.out "sphere_"${proberadius[$i]}"R.topo"
    cd ..
  done
fi

for i in `seq 0 $(($numjobs-1))`
do
  echo "Creating Project Directory"
  projectdir="Sphere_"${proberadius[$i]}"R_"${pvolt[$i]}"V_"${pvolt[$i]}"V_"${temp[$i]}"K"
  echo "Project Directory: "$projectdir
  mkdir $projectdir
  cd $projectdir
  ln -s ../mptetra
  ln -s ../geometry/"sphere_"${proberadius[$i]}"R.topo" meshpic.dat
  cp ../$baseproject/pictetra.dat .
  echo "Modifying pictetra.dat file"
  sed -i 's/.*te=.*/\tte='"${tempev[$i]}"'/' pictetra.dat
  sed -i 's/.*ti=.*/\tti='"${tempev[$i]}"'/' pictetra.dat
  sed -i 's/.*netotIni=.*/\tnetotIni='"$netotIni"'/' pictetra.dat
  sed -i 's/.*3.5 3.5.*/\t\t'"${pvolt[$i]}"'\t'"${pvolt[$i]}"'/' pictetra.dat
  echo "Creating jobscript file"
  if [[ $host = *fram.sigma2.no ]]; then
    cp ../$baseproject/run_fram.sh ./run.sh
  else
    cp ../$baseproject/run_saga.sh ./run.sh
    sed -i 's/.*#SBATCH --ntasks=.*/#SBATCH --ntasks='"$numproc"'/' run.sh
  fi
  sed -i 's/.*#SBATCH --job-name=.*/#SBATCH --job-name='"Ptetjob$i"'/' run.sh
  sed -i 's/.*#SBATCH --time=.*/#SBATCH --time='"$runtime:00:00"'/' run.sh

  echo "Preparation all done!!"
  echo "Putting run on queue"
  sbatch run.sh
  cd ..
done

echo "Checking queue status:"
squeue -u $usrname
