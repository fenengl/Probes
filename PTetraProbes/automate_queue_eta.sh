#!/bin/bash

echo "Automating job creation and queue for PTetra"
echo "============================================"

baserad=1.87
baseproject="Sphere_"$baserad"R_3.5V_3.5V_600K"

temp=500  #temperature in K
tempev=0.0431  #temperature in eV

debye=(0.004880)
proberadius=(0.0001 0.20 0.30 0.50 1.00 2.00 3.00 5.00 7.50 10.00 15.00 20.00 50.00 100.00)  #in terms of debye length
# pvolt=(4.5 5.5 6.5)
pvolt=(0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86)

netotIni=60000000

jobname=("Rp.00" "Rp.20" "Rp.30" "Rp.50" "Rp1.0" "Rp2.0" "Rp3.0" "Rp5.0" "Rp7.5" "Rp10.0" "Rp15.0" "Rp20.0" "Rp50.0" "Rp100.0")
numjobs=${#pvolt[@]}
usrname="sadhi"
numproc=4

ndir="geometry"
basegeofile="sphere_"$baserad"R.geo"

module --quiet purge
module load Anaconda3/2019.07
export PS1=\$
source ${EBROOTANACONDA3}/etc/profile.d/conda.sh
conda deactivate &>/dev/null
conda activate /cluster/projects/nn9299k/software/conda/envs/ptetra

if [ -d "$ndir" ]
then
  echo "Directory exists"
  geofile="sphere_"${proberadius[0]}"R.geo"
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
      sed -i 's/.*r = .*/r = '"${proberadius[0]}"';/' "$ndir"/"$geofile"
      echo ""
      gmsh -3 -format msh2 -optimize "$ndir"/"$geofile"
    fi
  fi
  echo "Converting .msh to .topo"
  meshfile="sphere_"${proberadius[0]}"R.msh"
  cp "$ndir"/"msh2topo_bkp.dat" "$ndir"/"msh2topo.dat"
  sed -i 's/.*sphere_.*/'"$meshfile"'/' "$ndir"/"msh2topo.dat"
  cd $ndir
  ./msh2topo
  mv msh2topo.out "sphere_"${proberadius[0]}"R.topo"
  cd ..
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
fi

echo "Checking queue status:"
squeue -u $usrname
