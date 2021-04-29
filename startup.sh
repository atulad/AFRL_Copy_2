#!/bin/bash

# this file is used by ec2_cluster.sh
cd ~/afrl
mkdir -p results/tboard
sudo yum install gcc-c++ python3-devel -y  > /tmp/tmp
pip3 install --user -r requirements.txt > /tmp/tmp
echo packages installed
python3 af_sac_old.py