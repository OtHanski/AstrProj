# AstrProj
Astronomy calc project for GPU programming course. Requires a CUDA compatible GPU  (fuck my laptop)

Calculate three histograms DD, DR, RR (angles between galaxies), D real galaxies, R synthetic
=> For every galaxy, calculate angles to every other galaxy
=> Shit ton of calcs

# Running on Dione
module load CUDA
module load GCC/7.3.0-2.30
nvcc --gpu-architecture=sm_70 --mem=10M -o Hello_world Hello_world.cu
srun -p gpu -c 1 -t 10:00 -e err.txt -o out.txt ./Hello_world
less out.txt

# To check GPUs:
nvidia-smi
# Dione
srun -p gpu nvidia-smi