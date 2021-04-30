#!/bin/bash

# this file is used by ec2_cluster.sh
sudo pkill PackageKit # else, this program holds the yum lock
sleep 10 # wait for yum lock
sudo yum install gcc-c++ python3-devel -y  > /dev/null
pip3 install --user -r requirements.txt > /dev/null