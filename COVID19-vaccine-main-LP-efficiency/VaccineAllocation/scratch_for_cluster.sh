#$ -cwd
#$ -N covid
#$ -o covid.txt
#$ -j y
#$ -S /bin/bash
#$ -pe mpi 200
#$ -l h_rt=48:00:00
#$ -M lindapei2016@gmail.com
#$ -l h_vmem=512g
#$ -V

export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export PATH=/share/apps/linuxbrew/var/homebrew/linked/python/bin:$PATH
export PYTHONPATH="/home/lpei/scratch/09072022/VaccineAllocation":$PYTHONPATH
export PYTHONPATH="/home/klt0112/scratch/09072022":$PYTHONPATH

mpirun -np 200 python3.7 scratch_for_cluster.py