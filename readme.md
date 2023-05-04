Requirements:
Available in requirements.txt.

conda create -n myenv python=3.8
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

pip install -r requirements.txt


Resource:

resource library is needed for linux to increase the recursion depth. This is not available for windows and the import needs to be commented. Also the corresponding code needs to be commented according to the platform.

job and job_3d are the job files that can be submitted using slurm

prm_3d.py --env 2d --show-animation --save-every 50

prm_3d.py includes both environments 2d as well as 3d and can be switched using argument --env. Help available using -h

--show-animation can be used to plot edges, computationally expensive
(The paths in 3d are plotted using configuration during steer and collision check. Since the number of points can be huge, 10 points are used to reduce time.)

--save-every: save after this many nodes are added to graph

Milestone 1:
Currently only one environment (0) available, we will be adding multiple environments in next milestone

Milestone 2:
Multiple environments added, only graph nodes corresponding to env 0 added to limit the size.


(To find shortest path, we find nearest 5 nodes to start and goal (since the node can be one that is not part of graph), then find shortest paths between these nodes and finally use the shortest of these all)

Data Generation:
data_gen: Submits the job to slurm to generate data samples for a given environment
2d: python dataset_collector.py --env 2d --env-id 0 --save-every 250 --num-iter 5000 --num-nodes 2000
3d: python dataset_collector_3d.py --env 3d --env-id 0 --save-every 250 --num-iter 5000 --num-nodes 2000

Point Cloud Generator:
python pointcloud.py [env_type] [env_folder_path] [num_env] [num_points]


Training: 

job_train: Submits the job to slurm to train the complete model, with 

python train_complete_model.py --N 10 --epochs 10000  --learning-rate 0.01 --decay-step 25 --decay-rate 0.8 --batch-size 1024 --samples 200000


Results obtained by using parameters:
Namespace(N=10, batch_size=1024, decay_rate=0.8, decay_step=25, device=0, env_type='2d', epochs=10000, learning_rate=0.01, model_path='./models/', opt='Adagrad', samples=200000, start_epoch=0, with_start=False, world_size=20.0)