## create conda env (must be python3.7)
conda create -n afrl python=3.7 && conda activate afrl

## install packages
pip install -r requirements.txt

## run sanity check 
- (sac should train in ~25 minutes on CPU to get reward > 200)
python sac.py

## run afrl
python af_sac.py

## run on aws cluster
chmod +x ec2_cluster.sh
./ec2_cluster.sh