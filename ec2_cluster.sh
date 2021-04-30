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
N_INSTANCES=2
INSTANCE_TYPE="c4.2xlarge"
SETUP_CMD="cd ~/afrl && source setup.sh"
CODE_CMD="cd ~/afrl && mkdir -p results/tboard && rm -r results/tboard && mkdir -p results/tboard && python3 af_sac.py"
CREATE_NEW=false # either create new instances or use stopped ones
KILL=false # either terminate the instances at the end or just stop them

# ------------------------------
# --    Install packages      --
# ------------------------------

get_dns() {
    INSTANCE_ID=$1
    # get instance public DNS
    PUBLIC_DNS=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        | jq -r '.Reservations[].Instances[0].PublicDnsName')
}

test_ssh() {
    INSTANCE_ID=$1
    idx=$2

    # wait just a bit before attempting ssh
    while ! nc -z $PUBLIC_DNS 22 ; do sleep 1 ; done

    # test ssh into machine
    ssh -o StrictHostKeyChecking=no -i $KEY_PRIVATE ec2-user@$PUBLIC_DNS -o LogLevel=ERROR exit \
        && echo ssh works on $idx  || echo FAILED TO SSH on $idx
}

copy_cwd() {
    # copy code to machine
    rsync --exclude results -q -Pa -e "ssh -i $KEY_PRIVATE" -a $(pwd)/ ec2-user@$PUBLIC_DNS:~/afrl
}

setup_instance() {
    idx=$2
    ssh -i $KEY_PRIVATE ec2-user@$PUBLIC_DNS $SETUP_CMD
    echo setup completed on $idx
}

# ------------------------------------------
# --        Run code on instance          --
# ------------------------------------------
run_instance() {
    INSTANCE_ID=$1
    idx=$2
    echo running code on $idx
    # run command/file
    ssh -i $KEY_PRIVATE ec2-user@$PUBLIC_DNS $CODE_CMD &
    # get the PID of the process
    PID=$!

    # copy results back while process is running
    while ps -p $PID > /dev/null; do
        rsync -q -Pa -e "ssh -i $KEY_PRIVATE" ec2-user@${PUBLIC_DNS}:~/afrl/results/tboard/ results/tboard/aws
        sleep 20
    done

    # one more sync at the end
    rsync -q -Pa -e "ssh -i $KEY_PRIVATE" ec2-user@${PUBLIC_DNS}:~/afrl/results/tboard/ results/tboard/aws
    echo code is completed on $idx
}

# ------------------------------------------
# --    Stop and Terminate Instance       --
# ------------------------------------------
stop_instance() {
    INSTANCE_ID=$1
    idx=$2
    # shut down instance
    aws ec2 stop-instances --instance-ids $INSTANCE_ID >  /dev/null
    aws ec2 wait instance-stopped --instance-ids $INSTANCE_ID && echo $idx is stopped
}

kill_instance() {
    INSTANCE_ID=$1
    idx=$2
    # terminate ec2 instance
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID >  /dev/null
    aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID && echo $idx is terminated
}

close_instance() {
    INSTANCE_ID=$1
    idx=$2
    if [ "$KILL" = true ]; then
        stop_instance $INSTANCE_ID $idx && \
            kill_instance $INSTANCE_ID $idx
    else
        stop_instance $INSTANCE_ID $idx
    fi
}

# ------------------------------
# -- Create EC2 Instance(s)   --
# ------------------------------
# start new instances w AMI Linux
create_instances() {
    INSTANCE_IDS=( $(aws ec2 run-instances \
        --image-id ami-00f9f4069d04c0c6e \
        --count $N_INSTANCES \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-groups ssh_only \
        | jq -r '.Instances[].InstanceId') )
}

get_stopped_instances() {
    INSTANCE_IDS=( $( aws ec2 describe-instances \
    --filter Name=instance-state-name,Values=stopped \
    | jq -r '.Reservations[].Instances[].InstanceId' ) )
    echo found ${#INSTANCE_IDS[@]} stopped instances
}

start_instances() {
    aws ec2 start-instances --instance-ids $INSTANCE_IDS > /dev/null
    # wait till instance is running
    aws ec2 wait instance-running --instance-ids $INSTANCE_IDS && echo instances running
}


# ------------------------------
# --     Run main loop        --
# ------------------------------
create_main() {
    create_instances && start_instances
    idx=1
    for INSTANCE_ID in ${INSTANCE_IDS[@]}; do
        ( (get_dns $INSTANCE_ID && \
        test_ssh $INSTANCE_ID $idx && \
        copy_cwd && \
        setup_instance $INSTANCE_ID $idx && \
        run_instance $INSTANCE_ID $idx && \
        close_instance $INSTANCE_ID $idx) & )
        idx=$((idx + 1))
    done
}

start_main() {
    get_stopped_instances
    INSTANCE_IDS=(${INSTANCE_IDS[@]:0:$N_INSTANCES})
    echo using ${#INSTANCE_IDS[@]} instances
    echo $INSTANCE_IDS
    start_instances
    idx=1
    for INSTANCE_ID in ${INSTANCE_IDS[@]}; do
        ( (get_dns $INSTANCE_ID && \
        test_ssh $INSTANCE_ID $idx && \
        copy_cwd && \
        run_instance $INSTANCE_ID $idx && \
        close_instance $INSTANCE_ID $idx) & )
        idx=$((idx + 1))
    done
}

if [ "$CREATE_NEW" = true ] ; then
    create_main
else
    start_main
fi