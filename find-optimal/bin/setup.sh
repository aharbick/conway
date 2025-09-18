#!/bin/bash

sudo apt update
sudo apt install -v pkgconf
# Was needed on a gcloud VM
#sudo apt install -y nvidia-driver-570
sudo apt install -y build-essential
sudo apt install -y libcurl4-openssl-dev
sudo apt install -y libssl-dev
sudo apt install -y clang-format
sudo apt install -y nodejs
sudo apt install -y direnv
sudo apt install -y libgtest-dev libgmock-dev
sudo apt install -y nlohmann-json3-dev
