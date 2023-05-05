Requirements:
Available in requirements.txt.

conda create -n myenv python=3.8

conda activate myenv

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

pip install -r requirements.txt

Skip to last section to run RRT* for 2D and 3D environments.


Resource:

resource library is needed for linux to increase the recursion depth. This is not available for windows and the import needs to be commented. Also the corresponding code needs to be commented/uncommented according to the platform.

job and job_3d are the job files that can be submitted using slurm

Generate PRMs for given environments:

prm_3d.py --env 2d --show-animation --save-every 50

prm_3d.py includes both environments 2d as well as 3d and can be switched using argument --env. Help available using -h

--show-animation can be used to plot edges, computationally expensive
(The paths in 3d are plotted using configuration during steer and collision check. Since the number of points can be huge, 10 points are used to reduce time.)

--save-every: save after this many nodes are added to graph

(To find shortest path, we find nearest k nodes to start and goal (since the node can be one that is not part of graph), then find shortest paths between these nodes and finally use the shortest of these all)

Data Generation:
data_gen: Submits the job to slurm to generate data samples for a given environment

2d:
``` 
python dataset_collector.py --env 2d --env-id 0 --save-every 250 --num-iter 5000 --num-nodes 2000
```
3d: 
```
python dataset_collector_3d.py --env 3d --env-id 0 --save-every 250 --num-iter 5000 --num-nodes 2000
```

Point Cloud Generator:
```
python pointcloud.py [env_type] [env_folder_path] [num_env] [num_points]
```

Training: 

job_train: Submits the job to slurm to train the complete model, with 

```
python train.py --N 10 --epochs 10000  --learning-rate 0.01 --batch-size 4096 --samples 50000 --activation relu --dropout 0 --point-cloud --weight-decay 0.001 --env-type 2d
```

RRTStar:

```
python planning_2d.py --sample normal --env [ENV_ID] 
```

--blind flag to disable the animation
--model-path: Path to a model
--iter: Iterations to run the planning
--sample: Which sampling to generate first: 'directed' = guided, 'normal' = Traditional RRT* with rewire.

```
python planning_3d.py --sample normal --iter 100 --show-animation --env-id [ENV_ID]
```
--model-path : Path to the model

if show animation is enabled, the nodes are displayed and finally the path for the sample mentioned in argument is displayed 3 times followed by other sample case.

