#$ -cwd
#$ -N covid
#$ -o covid.txt
#$ -j y
#$ -S /bin/bash
#$ -pe smp 80
#$ -l h_rt=48:00:00
#$ -M lindapei2016@gmail.com
#$ -l h_vmem=512g

export PYTHONPATH="/home/lpei/scratch/09012022/COVID19-vaccine-main-LP-efficiency/VaccineAllocation":$PYTHONPATH
export PATH="/share/apps/linuxbrew/var/homebrew/linked/python/bin":$PATH

python3.7 -O main_allocation.py