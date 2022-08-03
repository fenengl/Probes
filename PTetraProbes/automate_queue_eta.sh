#!/bin/bash

echo "Automating job creation and queue for PTetra"
echo "============================================"

baserad=1.87
baseproject="Sphere_"$baserad"R_3.5V_3.5V_600K"

temp=500  #temperature in K
tempev=0.0431  #temperature in eV

debye=(0.004880)
proberadius=(2.0 5.0 10.0 20.0 100.0 120.0)  #in terms of debye length
# pvolt=(4.5 5.5 6.5)
pvolt=(1.29 1.29 1.29 1.29 1.29 1.29)

netotIni=50000000

jobname=("Rp2.0" "Rp5.0" "Rp10.0" "Rp20.0" "Rp100.0" "Rp120.0")
numjobs=${#pvolt[@]}
usrname="sadhi"
numproc=4

ndir="geometry"
basegeofile="sphere_"$baserad"R.geo"

module --quiet purge
module load Anaconda3/2020.11
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
        sed -i 's/.*debye = .*/debye = '"${debye[0]}"';/' "$ndir"/"$geofile"
        sed -i 's/.*r = .*/r = '"${proberadius[$i]}"';/' "$ndir"/"$geofile"
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
  projectdir="Sphere_"${proberadius[$i]}"R_"${pvolt[$i]}"V_"${pvolt[$i]}"V_"$temp"K"
  echo "Project Directory: "$projectdir
  mkdir $projectdir
  cd $projectdir
  ln -s ../mptetra
  ln -s ../geometry/"sphere_"${proberadius[$i]}"R.topo" meshpic.dat
  cp ../$baseproject/pictetra.dat .
  echo "Modifying pictetra.dat file"
  sed -i 's/.*te=.*/\tte='"$tempev"'/' pictetra.dat
  sed -i 's/.*ti=.*/\tti='"$tempev"'/' pictetra.dat
  sed -i 's/.*netotIni=.*/\tnetotIni='"$netotIni"'/' pictetra.dat
  sed -i 's/.*3.5 3.5.*/\t\t'"${pvolt[$i]}"'\t'"${pvolt[$i]}"'/' pictetra.dat
  echo "Creating jobscript file"
  cp ../$baseproject/run_bkp.sh ./run.sh
  sed -i 's/.*#SBATCH --job-name=.*/#SBATCH --job-name='"${jobname[$i]}"'/' run.sh
  sed -i 's/.*#SBATCH --ntasks=.*/#SBATCH --ntasks='"$numproc"'/' run.sh
  echo "Preparation all done!!"
  echo "Putting run on queue"
  sbatch run.sh
  cd ..
done

echo "Checking queue status:"
squeue -u $usrname
