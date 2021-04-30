# use this to stop any instances that are running

INSTANCE_IDS=( $( aws ec2 describe-instances \
--filter Name=instance-state-name,Values=running \
| jq -r '.Reservations[].Instances[].InstanceId' ) )
echo found ${#INSTANCE_IDS[@]} running instances


stop_instances() {
    # shut down instance
    aws ec2 stop-instances --instance-ids $INSTANCE_IDS > /dev/null
    aws ec2 wait instance-stopped --instance-ids $INSTANCE_IDS && echo all stopped
}

kill_instances() {
    # terminate ec2 instance
    aws ec2 terminate-instances --instance-ids $INSTANCE_IDS > /dev/null
    aws ec2 wait instance-terminated --instance-ids $INSTANCE_IDS && echo all terminated
}

stop_instances