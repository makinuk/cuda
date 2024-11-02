#!/bin/bash

# CUDA Installation Script for Ubuntu

add_to_file_if_missing() {
    local line="$1"
    local file="$2"
    if ! grep -Fxq "$line" "$file"; then
        echo "$line" >> "$file"
    fi
}

# Step 1: Update system
sudo apt update


# Step 4: Download CUDA Toolkit installer
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# Step 5: Install CUDA Toolkit
sudo sh cuda_12.4.0_550.54.14_linux.run

# Step 6: Set up environment variables
add_to_file_if_missing 'export CUDA_HOME=/usr/local/cuda-12.4' ~/.bashrc
add_to_file_if_missing 'export PATH=$CUDA_HOME/bin:$PATH' ~/.bashrc
add_to_file_if_missing 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' ~/.bashrc

source ~/.bashrc

# Step 7: Verify CUDA installation
