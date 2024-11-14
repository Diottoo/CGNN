

#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=16G



cd GNN-main/condition

module load gsl
module load gcc
module load python/anaconda-2021.05
python3 main.py 
\

