#!/bin/bash

# this file is used by ec2_cluster.sh
idx=$2
mkdir -p results/tboard
sudo pkill PackageKit # else, this program holds the yum lock
sudo yum install gcc-c++ python3-devel -y  > /dev/null
pip3 install --user -r requirements.txt > /dev/null
echo packages installed on $idx