#!/bin/bash 
#SBATCH --comment=tensorflow
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
##SBATCH --partition=cas_v100nv_8
##SBATCH --partition=cas_v100_4
#SBATCH --time=48:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes 
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=4     # number of cpus per task

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the port and node name
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU 

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

echo "load module-environment"
#module load gcc/10.2.0 cuda/11.6
module load gcc/10.2.0 cuda/12.1 cudampi/openmpi-4.1.1 cmake/3.26.2
#module load gcc/10.2.0 cuda/11.8 cudampi/openmpi-4.1.1 cmake/3.26.2
export CUDA_DIR=${CUDADIR}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDADIR}

echo "execute jupyter"
source ~/.bashrc
#conda activate vision
conda activate vit
cd /scratch/qualis/vit  # the root/work directory of Jupyter lab/notebook
jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --no-browser --NotebookApp.token=${USER} #jupyter token: your account ID 
echo "end of the job"
