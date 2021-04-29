#!/bin/bash

# 1. create n worker instances
# 2. create 1 main (micro) instance
# for each instance...
#    3. start instance
#    4. copy code and redirect stdout to scp
#    5. when experiment is done, send done signal to main
#    6. copy csv/tensorboard files
#    7. shut down and terminate instance
# get main to send results to local computer
# shut down main

# ------------------------------
# -- Manually Set Config      --
# ------------------------------

KEY_NAME=my_kp
KEY_PRIVATE=~/.ssh/aws/my_kp.pem
N_INSTANCES=10
INSTANCE_TYPE="c4.2xlarge"
CMD='chmod +x ~/afrl/startup.sh && ~/afrl/startup.sh'

# ------------------------------
# --    Install packages      --
# ------------------------------
setup_instance() {
    INSTANCE_ID=$1
    idx=$2
    # wait till instance is running
    echo waiting till instance $idx is running...
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID && echo $idx running

    # get instance public DNS
    PUBLIC_DNS=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        | jq -r '.Reservations[].Instances[0].PublicDnsName')

    # wait just a bit before attempting ssh
    while ! nc -z $PUBLIC_DNS 22 ; do sleep 1 ; done

    # test ssh into machine
    ssh -o StrictHostKeyChecking=no -i $KEY_PRIVATE ec2-user@$PUBLIC_DNS -o LogLevel=ERROR exit \
        && echo ssh works on $idx  || echo FAILED TO SSH on $idx

    # copy code to machine
    rsync --exclude results -q -Pa -e "ssh -i $KEY_PRIVATE" -a $(pwd)/ ec2-user@$PUBLIC_DNS:~/afrl
}

# ------------------------------------------
# --        Run code on instance          --
# ------------------------------------------
run_instance() {
    INSTANCE_ID=$1
    idx=$2
    # run command/file
    ssh -i $KEY_PRIVATE ec2-user@$PUBLIC_DNS $CMD &
    # get the PID of the process
    PID=$!

    # copy results back while process is running
    while ps -p $PID > /dev/null; do
        rsync -q -Pa -e "ssh -i $KEY_PRIVATE" ec2-user@${PUBLIC_DNS}:~/afrl/results/tboard/ results/tboard/aws
        sleep 20
    done

    # one more sync at the end
    rsync -q -Pa -e "ssh -i $KEY_PRIVATE" ec2-user@${PUBLIC_DNS}:~/afrl/results/tboard/ results/tboard/aws
    echo $idx is finished
}

# ------------------------------------------
# -- Cleanup and Terminate Instance       --
# ------------------------------------------
kill_instance() {
    INSTANCE_ID=$1
    idx=$2
    # shut down instance
    aws ec2 stop-instances --instance-ids $INSTANCE_ID > /tmp/tmp

    # terminate ec2 instance
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID > /tmp/tmp
    aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID && echo $idx is terminated
}

# ------------------------------
# -- Create Security group    --
# ------------------------------
# create an ec2 security group that would allow SSH access
# aws ec2 create-security-group --description "basic SSH access on port 22" --group-name ssh_only

# allow inbound port 22 traffic
# aws ec2 authorize-security-group-ingress --group-name ssh_only --protocol tcp --port 22 --cidr "0.0.0.0/0"

# remove security group
# aws ec2 delete-security-group --group-name ssh_only

# ------------------------------
# -- Create EC2 Instance(s)   --
# ------------------------------
# start new instances w AMI Linux
INSTANCE_IDS=\
$(aws ec2 run-instances \
    --image-id ami-00f9f4069d04c0c6e \
    --count $N_INSTANCES \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups ssh_only \
    | jq -r '.Instances[].InstanceId')

echo INSTANCE_IDS: $INSTANCE_IDS

# ------------------------------
# --     Run main loop        --
# ------------------------------
idx=1
for INSTANCE_ID in ${INSTANCE_IDS[@]}; do
  setup_instance $INSTANCE_ID $idx && \
    run_instance $INSTANCE_ID $idx && \
    kill_instance $INSTANCE_ID $idx &
  idx=$((idx + 1))
done