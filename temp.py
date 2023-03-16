from prm import PRM, Node
import pickle

obstacleList = [
        (-15, 0, 15.0, 5.0),
        (15, -10, 5.0, 10.0),
        (-10, 8, 5.0, 15.0),
        (3, 15, 10.0, 5.0),
        (-10, -10, 10.0, 5.0),
        (5, -5, 5.0, 5.0),
    ]

prm = PRM(obstacleList=obstacleList, randArea=[-20, 20])

with open('graph_1100.pkl', 'rb') as f:
    prm.nodeList = pickle.load(f)

path, cost = prm.getShortestPath(start= Node(state=[-20, -20.0]), goal=Node(state=[20.0, 20.0]))

print(path)
print('cost', cost)
# for node in path:
    # print(node.state)
prm.drawGraph(path=path, save=True, filename='shortest_path.png')
