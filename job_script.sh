#!/bin/sh
#SBATCH -J rep_gen 
#SBATCH -p gpu_edu     
#SBATCH -N 2           
#SBATCH -n 4           
#SBATCH -o %x.o%j     
#SBATCH -e %x.o%j      
#SBATCH --time 24:00:00  
#SBATCH --gres=gpu:1    
#SBATCH --exclusive=user

module purge
module load intel/20.4.9 cuda/10.2 cudampi/openmpi-3.1.6
source conda
conda activate test
python ./main.py
