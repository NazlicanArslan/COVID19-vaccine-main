#$ -cwd
#$ -N covid
#$ -o covid.txt
#$ -j y
#$ -S /bin/bash
#$ -pe smp 80
#$ -l h_rt=48:00:00
#$ -M lindapei2016@gmail.com
#$ -l h_vmem=512g
#$ -l h="compute-0-4"

export PYTHONPATH="/home/lpei/scratch/07272022/COVID19-vaccine-main-LP-efficiency/VaccineAllocation":$PYTHONPATH
export PYTHONPATH="/home/klt0112/scratch/07272022/COVID19-vaccine-main-LP-efficiency":$PYTHONPATH
export PATH="/share/apps/linuxbrew/var/homebrew/linked/python/bin":$PATH

# tiers5_opt_Final candidates other than the first threshold have been changed to null
# austin_test_IHT.json grid_size has been changed to 1

mpirun -np 80 python3.7 -W ignore main_allocation.py austin -f=setup_data_Final.json -t=tiers_LP_debug.json -train_reps=2 -test_reps=1 -f_config=austin_test_IHT.json -n_proc=80 -tr=transmission.csv -hos=austin_real_hosp_updated.csv  -v_allocation=vaccine_allocation_fixed.csv -n_policy=7 -v_boost=booster_allocation_fixed.csv -machine=crunch -pub=80
