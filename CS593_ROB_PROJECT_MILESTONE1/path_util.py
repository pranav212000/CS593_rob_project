from prm_3d import PRM, Node
import pickle

# Implemented for 2d only for now, 3d is similar but in progress


obstacleList=[
        (-15, 0, 15.0, 5.0),
        (15, -10, 5.0, 10.0),
        (-10, 8, 5.0, 15.0),
        (3, 15, 10.0, 5.0),
        (-10, -10, 10.0, 5.0),
        (5, -5, 5.0, 5.0),
        (10, 10, 5.0, 5.0),
    ]

prm = PRM(obstacleList=obstacleList, randArea=[-20, 20])

with open('2d/graph_env_0_nodes_1200.pkl', 'rb') as f:
    prm.nodeList = pickle.load(f)

# start and goal node can be randomly sampled
path, cost = prm.getShortestPath(start= Node(state=[-20, -20.0]), goal=Node(state=[20.0, 20.0]))

# print(path)
print('cost', cost)

prm.drawGraph(path=path, save=True, filename='shortest_path.png', show_animation=False)
