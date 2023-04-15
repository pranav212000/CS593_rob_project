Requirements:
Available in requirements.txt, can create conda environment using it.

Resource:
resource library needs to be installed for linux to increase the recursion depth. This is not available for windows and need to be commented. Also the corresponding code needs to be commented.

job and job_3d are the job files that can be submitted using slurm

prm_3d.py includes both environments 2d as well as 3d and can be switched using argument --env. Help available using -h

Currently only one environment (0) available, we will be adding multiple environments in next milestone

--show-animation can be used to plot edges, computationally expensive
(The paths in 3d are plotted using configuration during steer and collision check. Since the number of points can be huge, 10 points are used to reduce time.)

--save-every: save after this many nodes are added to graph

path_util.py: File to find shortest path between two nodes, can be randomly sampled, currently hard coded. (Currently supports 2d only)

(To find shortest path, we find nearest 5 nodes to start and goal (since the node can be one that is not part of graph), then find shortest paths between these nodes and finally use the shortest of these all)