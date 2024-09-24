# vision-transformer

This repository provides implementations of vision transformers in Tensorflow, Keras and PyTroch. 

**Contents**
* [KISTI Neuron GPU Cluster](#kisti-neuron-gpu-cluster)
* [Installing Conda](#installing-conda)
* [Creating a Conda Virtual Environment](#creating-a-conda-virtual-environment)
* [Running Jupyter](#running-jupyter)


## KISTI Neuron GPU Cluster
Neuron is a KISTI GPU cluster system consisting of 65 nodes with 260 GPUs (120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>

## Installing Conda
Once logging in to Neuron, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

1. Check the Neuron system specification
```
[glogin01]$ cat /etc/*release*
CentOS Linux release 7.9.2009 (Core)
Derived from Red Hat Enterprise Linux 7.8 (Source)
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"

CentOS Linux release 7.9.2009 (Core)
CentOS Linux release 7.9.2009 (Core)
cpe:/o:centos:centos:7
```

2. Download Anaconda or Miniconda. Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices. 
```
# (option 1) Anaconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate
```
```
# (option 2) Miniconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
```

3. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. You will install and create subsequent conda environments on your scratch directory. 
```
[glogin01]$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
[glogin01]$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here 

Miniconda3 will now be installed into this location:
/home01/qualis/miniconda3        

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home01/qualis/miniconda3] >>> /scratch/$USER/miniconda3  <======== type /scratch/$USER/miniconda3 here
PREFIX=/scratch/qualis/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish to update your shell profile to automatically initialize conda?
This will activate conda on startup and change the command prompt when activated.
If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
no change     /scratch/qualis/miniconda3/etc/profile.d/conda.csh
modified      /home01/qualis/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

Thank you for installing Miniconda3!
```

4. finalize installing Miniconda with environment variables set including conda path

```
[glogin01]$ source ~/.bashrc    # set conda path and environment variables 
[glogin01]$ conda config --set auto_activate_base false
[glogin01]$ which conda
/scratch/$USER/miniconda3/condabin/conda
[glogin01]$ conda --version
conda 23.9.0
```

## Creating a Conda Virtual Environment
You want to create a virtual envrionment with a python version 3.10 for Generative AI Practices.
```
[glogin01]$ conda create -n vit python=3.10
Retrieving notices: ...working... done
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3/envs/vit

  added / updated specs:
    - python=3.10
.
.
.
Proceed ([y]/n)? y    <========== type yes 


Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate vit
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

Now, you need to install Python packages for the `vit` virtual environment.  
```
[glogin01]$ conda activate vit
(vit) [glogin01]$ pip install tensorflow==2.15.0  tensorflow_addons 
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: tensorflow==2.15.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (2.15.0)
Requirement already satisfied: tensorflow_addons in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (0.23.0)
Requirement already satisfied: absl-py>=1.0.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (2.1.0)
Requirement already satisfied: astunparse>=1.6.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (1.6.3)
Requirement already satisfied: flatbuffers>=23.5.26 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (24.3.25)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (0.6.0)
Requirement already satisfied: google-pasta>=0.1.1 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (0.2.0)
Requirement already satisfied: h5py>=2.9.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (3.11.0)
Requirement already satisfied: libclang>=13.0.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (18.1.1)
Requirement already satisfied: ml-dtypes~=0.2.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (0.2.0)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (1.26.4)
Requirement already satisfied: opt-einsum>=2.3.2 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (3.3.0)
Requirement already satisfied: packaging in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (24.1)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (4.25.5)
Requirement already satisfied: setuptools in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (75.1.0)
Requirement already satisfied: six>=1.12.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (4.11.0)
Requirement already satisfied: wrapt<1.15,>=1.11.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (1.14.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (0.37.1)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (1.66.1)
Requirement already satisfied: tensorboard<2.16,>=2.15 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (2.15.2)
Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (2.15.0)
Requirement already satisfied: keras<2.16,>=2.15.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow==2.15.0) (2.15.0)
Requirement already satisfied: typeguard<3.0.0,>=2.7 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorflow_addons) (2.13.3)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow==2.15.0) (0.44.0)
Requirement already satisfied: google-auth<3,>=1.6.3 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow==2.15.0) (2.35.0)
Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow==2.15.0) (1.2.1)
Requirement already satisfied: markdown>=2.6.8 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow==2.15.0) (3.7)
Requirement already satisfied: requests<3,>=2.21.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow==2.15.0) (2.32.3)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow==2.15.0) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from tensorboard<2.16,>=2.15->tensorflow==2.15.0) (3.0.4)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (5.5.0)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (0.4.1)
Requirement already satisfied: rsa<5,>=3.1.4 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (4.9)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (2.0.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (2.2.2)
Requirement already satisfied: certifi>=2017.4.17 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (2024.8.30)
Requirement already satisfied: MarkupSafe>=2.1.1 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (2.1.3)
Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (0.6.1)
Requirement already satisfied: oauthlib>=3.0.0 in /scratch/qualis/miniconda3/envs/vit/lib/python3.10/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow==2.15.0) (3.2.2)
```

## Running Jupyter
[Jupyter](https://jupyter.org/) is free software, open standards, and web services for interactive computing across all programming languages. Jupyterlab is the latest web-based interactive development environment for notebooks, code, and data. The Jupyter Notebook is the original web application for creating and sharing computational documents. You will run a notebook server on a worker node (*not* on a login node), which will be accessed from the browser on your PC or labtop through SSH tunneling. 
<p align="center"><img src="https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod/assets/84169368/34a753fc-ccb7-423e-b0f3-f973b8cd7122"/>
</p>

In order to do so, you need to add the `hf-nlp-course` virtual envrionment that you have created as a python kernel.
1. activate the `hf-nlp-course` virtual environment, if it's not activated:
```
[glogin01]$ conda activate hf-nlp-course
```
2. install Jupyter on the virtual environment:
```
(vit) [glogin01]$ conda install jupyter chardet cchardet 
  
```
3. add the virtual environment as a jupyter kernel:
```
(vit) [glogin01]$ python -m ipykernel install --user --name vit
```
4. check the list of kernels currently installed:
```
(vit) [glogin01]$ jupyter kernelspec list
Available kernels:
python3     /home01/$USER/.local/share/jupyter/kernels/python3
vit         /home01/$USER/.local/share/jupyter/kernels/vit
```
5. launch a jupyter notebook server on a worker node 
- to deactivate the virtual environment
```
(vit) [glogin01]$ conda deactivate
```
- to create a batch script for launching a jupyter notebook server: 
```
[glogin01]$ cat jupyter_run.sh
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

#echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

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
```
- to check the SSH tunneling information generated by the jupyter_run.sh script 
```
[glogin01]$ cat port_forwarding_command
ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
```
6. open a new SSH client (e.g., Putty, MobaXterm, PowerShell, Command Prompt, etc) on your PC or laptop and log in to the Neuron system just by copying and pasting the port_forwarding_command:

![20240123_102609](https://github.com/hwang2006/Generative-AI-with-LLMs/assets/84169368/1f5dd57f-9872-491b-8dd4-0aa99b867789)

7. open a web browser on your PC or laptop to access the jupyter server
```
URL Address: localhost:8888
Password or token: $USER    # your account name on Neuron
```
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218938419-f38c356b-e682-4b1c-9add-6cfc29d53425.png"/></p> 
