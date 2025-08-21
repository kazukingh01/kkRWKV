# kkRWKV

Follow: https://github.com/BlinkDL/RWKV-LM/

# Install

### For Training

##### Prepare docker

```bash
# 0) Check to use GPU
nvidia-smi

# 1) Dependet tools
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# 2) GPG Key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo tee /etc/apt/keyrings/nvidia-container-toolkit.asc

# 3) add nvidia-container repository
echo \
"deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.asc] \
https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" \
| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 4) update & install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 5) setting nvidia runtime to docker
sudo nvidia-ctk runtime configure --runtime=docker

# 6) Restart 
sudo systemctl restart docker
```

##### Start container

```bash
sudo docker pull nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
sudo docker run -itd --gpus all --shm-size=16g --name dev -v /home/share:/home/share nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 /bin/bash --login
sudo docker exec -it dev /bin/bash
```

check.

```bash
nvidia-smi
# Fri Aug  1 10:04:28 2025
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 565.77.01              Driver Version: 566.36         CUDA Version: 12.7     |
# |-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA GeForce RTX 4090 ...    On  |   00000000:01:00.0 Off |                  N/A |
# | N/A   47C    P0             27W /  175W |       0MiB /  16376MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+

# +-----------------------------------------------------------------------------------------+
# | Processes:                                                                              |
# |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
# |        ID   ID                                                               Usage      |
# |=========================================================================================|
# |  No running processes found                                                             |
# +-----------------------------------------------------------------------------------------+
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on Tue_Oct_29_23:50:19_PDT_2024
# Cuda compilation tools, release 12.6, V12.6.85
# Build cuda_12.6.r12.6/compiler.35059454_0
```

##### Install python

```bash
apt-get update
apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev git iputils-ping net-tools vim cron rsyslog
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv/plugins/python-build
bash ./install.sh
INSTALL_PYTHON_VERSION="3.12.10"
/usr/local/bin/python-build -v ${INSTALL_PYTHON_VERSION} ~/local/python-${INSTALL_PYTHON_VERSION}
echo 'export PATH="$HOME/local/python-'${INSTALL_PYTHON_VERSION}'/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

##### Install kkRWKV

```bash
cd ~ && git clone https://github.com/kazukingh01/kkRWKV.git 
cd ~/kkRWKV/ && python -m venv venv && source venv/bin/activate
pip install torch==2.8.0 --upgrade --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

# Test

```bash
python ./tests/test.py
```