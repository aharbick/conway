#!/bin/bash

sudo apt update

# Was needed on a gcloud VM for some classes of GPUs
#sudo apt install -y nvidia-driver-570

sudo apt install -y pkgconf
sudo apt install -y build-essential

# For NVIDIA... on laptop
# wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt install cuda-toolkit

sudo apt install -y libcurl4-openssl-dev
sudo apt install -y libssl-dev
sudo apt install -y clang-format
sudo apt install -y direnv
sudo apt install -y libgtest-dev libgmock-dev
sudo apt install -y nlohmann-json3-dev
