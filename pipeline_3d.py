"""
Path Planning Sample Code with RRT*

author: Ahmed Qureshi, code adapted from AtsushiSakai(@Atsushi_twi)

"""


from math import sqrt
import argparse
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import torch
import pickle
from Model.complete_model import MLPComplete
import Model.CAE as CAE_2d
import Model.mlp as mlp
from Model.e2e import End2EndMPNet
from tqdm import tqdm
import pybullet as p
import pybullet_data
import os
from prm_3d import Node, set_joint_positions, draw_sphere_marker, draw_line_marker, remove_marker, magnitude, UR5_JOINT_INDICES
from collision_utils import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def diff(v1, v2):
    """
    Computes the difference v1 - v2, assuming v1 and v2 are both vectors
    """
    return [x1 - x2 for x1, x2 in zip(v1, v2)]


def magnitude(v):
    """
    Computes the magnitude of the vector v.
    """
    return math.sqrt(sum([x*x for x in v]))


def dist(p1, p2):
    """
    Computes the Euclidean distance (L2 norm) between two points p1 and p2
    """
    return magnitude(diff(p1[:2], p2[:2]))


def get_pointcloud(obstaclelist):
    point_cloud = []
    num_points = 200
    for point in obstaclelist:
        left_bottom_x = point[0]
        left_bottom_y = point[1]
        size_x = point[2]
        size_y = point[3]
        # print(left_bottom_x, left_bottom_y, size_x, size_y)
        # uniformly sample 100 points in the environment
        for i in range(num_points):
            x = np.random.uniform(left_bottom_x, left_bottom_x + size_x)
            y = np.random.uniform(left_bottom_y, left_bottom_y + size_y)
            point_cloud.append([x, y])
    # convert to numpy array
    # print(point_cloud)
    point_cloud = np.array(point_cloud)
    return point_cloud


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea, alg, geom, pointcloud, dof=2, expandDis=0.05, goalSampleRate=5, maxIter=100, sample='normal', collisionCheck3d=None, env='2d', ur5=None):
        """
        Sets algorithm parameters

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,width,height],...]
        randArea:Ramdom Samping Area [min,max]

        """
        
        if env == '2d':
            self.start = Node(start)
            self.end = Node(goal)
        else:
            self.start = Node(conf=start, ur5=ur5)
            self.end = Node(conf=goal, ur5=ur5)
        self.obstacleList = obstacleList
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.alg = alg
        self.geom = geom
        self.dof = dof
        self.pointcloud = pointcloud
        self.min_cost_to_go = 1000000
        self.sample = sample
        self.cost_to_go = {}
        self.collisionCheck3d = collisionCheck3d
        self.env = env
        self.ur5 = ur5

        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter

        self.cost_to_go[tuple(self.end.conf)] = 0
        self.cost_to_go[tuple(self.start.conf)] = 1000000

        self.goalfound = False
        self.solutionSet = set()

    
    def getDistance(self, node1: Node, node2: Node):        
        return np.linalg.norm(np.array(node1.conf) - np.array(node2.conf))
  

    def getNearNodes(self, newNode: Node, radius: float, k: int):

        nearNodes = []
        GAMMA = 50
        n_nodes = len(self.nodeList)
        radius = GAMMA * (pow((np.log(n_nodes)/n_nodes), (1/self.dof)))

        for i, node in enumerate(self.nodeList):
            if self.getDistance(newNode, node) < radius:
                nearNodes.append(i)

        return nearNodes
    def getEndEffectorPos(self):
        num_joints = p.getNumJoints(self.ur5)
        link_id = num_joints - 1
        link_state = p.getLinkState(
            self.ur5, link_id, computeForwardKinematics=True)
        link_pos = link_state[0]
        link_ori = link_state[1]

        return link_pos


    def steerTo3d(self, rand_node, nearest_node, step_size=0.05, show_animation=True):
        
        
        distance = self.getDistance(rand_node, nearest_node)
        n_steps = round(distance/step_size)
        if n_steps == 0:
            return (self.collisionCheck3d(rand_node.conf), distance)
        unit_step = (np.array(rand_node.conf) -
                     np.array(nearest_node.conf)) / n_steps
        start = np.array(nearest_node.conf)
        link_pos = []
        start_pos = self.getEndEffectorPos()
        for i in range(n_steps):
            # print(i)
            start = start + unit_step
            start = (start[0], start[1], start[2])
            link_pos.append(self.getEndEffectorPos())

            if self.collisionCheck3d(start):
                return (False, None)

        end_pos = self.getEndEffectorPos()

        if show_animation:
            if len(link_pos) > 10:
                link_pos = link_pos[::len(link_pos)//2]

            link_pos.insert(0, start_pos)
            link_pos.append(end_pos)

            # print(len(link_pos))

            for i in range(len(link_pos)-1):
                p.addUserDebugLine(link_pos[i], link_pos[i+1], [1, 0, 0], 1, 0)

        return (True, distance)

    def planning3d(self, show_animation=False, model=None):
        """
        Implements the RTT (or RTT*) algorithm, following the pseudocode in the handout.
        You should read and understand this function, but you don't have to change any of its code - just implement the 3 helper functions.

        animation: flag for animation on or off
        """

        min_time = time.time()
        firstTime = time.time()

        if self.env == '3d':
            print(self.start)
            print(self.end)
            if self.collisionCheck3d(self.start.conf) or self.collisionCheck3d(self.end.conf):
                return [], firstTime, min_time

        self.nodeList = [self.start]
        point_cloud = self.pointcloud
        min_cost = float('inf')

        for i in range(self.maxIter):
            if i% 10 == 0:
                print(i)

            rnd = self.generatesample(point_cloud=point_cloud, model=model)
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            # print(rnd.conf)
            # print(self.nodeList[nind].conf)
            rnd_valid, rnd_cost = self.steerTo3d(rnd, self.nodeList[nind], show_animation=show_animation)
            

            if (rnd_valid):
                newNode = copy.deepcopy(rnd)
                newNode.parent = nind
                newNode.cost = rnd_cost + self.nodeList[nind].cost

                if self.alg == 'rrtstar':
                    # you'll implement this method
                    nearinds = self.find_near_nodes(newNode)
                    # to_add = self.to_be_added(newNode, nearinds, point_cloud)
                    # you'll implement this method
                    newParent = self.choose_parent(newNode, nearinds)
                    # if to_add == True:
                    #     newParent = self.choose_parent(newNode, nearinds) # you'll implement this method
                    # else:
                    #     continue
                else:
                    newParent = None

                # insert newNode into the tree
                if newParent is not None:
                    newNode.parent = newParent
                    newNode.cost = self.getDistance(newNode, self.nodeList[newParent]) + self.nodeList[newParent].cost
                    # newNode.cost = dist(
                    #     newNode.state, self.nodeList[newParent].conf) + self.nodeList[newParent].cost
                else:
                    pass  # nind is already set as newNode's parent
                self.nodeList.append(newNode)
                newNodeIndex = len(self.nodeList) - 1
                self.nodeList[newNode.parent].children.add(newNodeIndex)

                if self.alg == 'rrtstar' and self.sample == 'normal':
                    # you'll implement this method
                    self.rewire(newNode, newNodeIndex, nearinds)

                if self.is_near_goal(newNode):
                    self.solutionSet.add(newNodeIndex)
                    self.goalfound = True
                    firstTime = time.time()
                    cost = self.get_path_len(self.get_path_to_goal())
                    if cost < min_cost:
                        min_time = time.time()

                # if animation:

                #     self.draw_graph(rnd.state)

        return self.get_path_to_goal(), firstTime, min_time

   
    def choose_parent(self, newNode, nearinds):
        """
        Selects the best parent for newNode. This should be the one that results in newNode having the lowest possible cost.

        newNode: the node to be inserted
        nearinds: a list of indices. Contains nodes that are close enough to newNode to be considered as a possible parent.

        Returns: index of the new parent selected
        """
        # your code here
        # find paths for all nodes in neainds and calculate the distance from source
        # find the minimum distance and return the index of the node
        # if no path is found return None
        minCost = float('inf')
        minIndex = None
        for i in nearinds:
            valid, cost = self.steerTo3d(newNode, self.nodeList[i])
            if valid:
                # find cost of reaching to i from source
                cost += self.nodeList[i].cost
                if cost < minCost:
                    minCost = cost
                    minIndex = i

        return minIndex

    
    def generatesample(self, model=None, point_cloud=None):
        """
        Randomly generates a sample, to be used as a new node.
        This sample may be invalid - if so, call generatesample() again.

        You will need to modify this function for question 3 (if self.geom == 'rectangle')

        returns: random c-space vector
        """

        if random.randint(0, 100) > self.goalSampleRate:
            cost_to_go = float('inf')
            # env_data = point_cloud
            # env_data = np.append(env_data, self.end.state)
            env_data = self.end.conf
            point_cloud = np.array(point_cloud).astype(np.float32)
            point_cloud = point_cloud.flatten()
            point_cloud = point_cloud / 20.0

            # make pointcloud a row
            point_cloud = point_cloud.reshape(1, -1)

            # print(point_cloud.shape)
            point_cloud = torch.FloatTensor(point_cloud)
            # print(point_cloud.shape)

            prob_to_rand = np.random.random()

            max_sample = 50

            min_near_neighbor_cost = 10000
            i = 0
            # while cost_to_go > self.min_cost_to_go + 10 or self.sample == 'normal':

            while cost_to_go > min_near_neighbor_cost + 5 or self.sample == 'normal':
                i += 1
                if i > max_sample:
                    break

                while True:

                    if self.env == '2d':
                        sample = []

                        for j in range(0, self.dof):
                            sample.append(random.uniform(
                                self.minrand, self.maxrand))

                        rnd = Node(sample)

                        if self.__CollisionCheck(rnd):
                            break

                    elif self.env == '3d':
                        rnd = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                                                                                           math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))

                        set_joint_positions(self.ur5, UR5_JOINT_INDICES, rnd)
                        rnd = Node(conf=rnd, ur5=self.ur5)

                        if not self.collisionCheck3d(rnd.conf):
                            break

                        

                if self.sample == 'normal':
                    break

                
                
                near_nodes = self.getNearNodes(rnd)
                for node in near_nodes:
                    near_cost = self.cost_to_go[tuple(
                        self.nodeList[node].conf)]
                    # model(torch.FloatTensor(np.append(env_data, self.nodeList[node].state).reshape(1, -1)/20.0), point_cloud) * 20.0
                    if near_cost < min_near_neighbor_cost:
                        min_near_neighbor_cost = near_cost

                node_input = np.append(env_data, rnd.conf)

                node_input = node_input.flatten()
                data = np.array(node_input).astype(np.float32)

                # print(data.shape)

                data = data/20.0
                data = data.reshape(1, -1)
                data = torch.FloatTensor(data)

                cost_to_go = model(data, point_cloud) * 20.0

                
                self.cost_to_go[tuple(rnd.conf)] = cost_to_go

                if prob_to_rand < 0 or len(near_nodes) == 0:
                    break

            self.min_cost_to_go = cost_to_go

        else:
            rnd = self.end

        # print(rnd)
        draw_sphere_marker(position=rnd.state, radius=0.005, color=[1, 1, 0, 1])


        return rnd

    def is_near_goal(self, node):
        """
        node: the location to check

        Returns: True if node is within 5 units of the goal state; False otherwise
        """
        d = dist(node.conf, self.end.conf)
        if d < 5.0:
            return True
        return False

    @staticmethod
    def get_path_len(path, env='2d'):
        """
        path: a list of coordinates

        Returns: total length of the path
        """
        pathLen = 0
        for i in range(1, len(path)):
            if env == '2d':
                pathLen += dist(path[i], path[i-1])
            elif env == '3d':
                pathLen += dist(path[i], path[i-1])
            
            

        return pathLen

    def gen_final_course(self, goalind):
        """
        Traverses up the tree to find the path from start to goal

        goalind: index of the goal node

        Returns: a list of coordinates, representing the path backwards. Traverse this list in reverse order to follow the path from start to end
        """
        path = [self.end.conf]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            path.append(node.conf)
            goalind = node.parent
        path.append(self.start.conf)
        return path

    def find_near_nodes(self, newNode):
        """
        Finds all nodes in the tree that are "near" newNode.
        See the assignment handout for the equation defining the cutoff point (what it means to be "near" newNode)

        newNode: the node to be inserted.

        Returns: a list of indices of nearby nodes.
        """
        # Use this value of gamma
        GAMMA = 50
        # your code here

        n_nodes = len(self.nodeList)

        radius = GAMMA * (pow((np.log(n_nodes)/n_nodes), (1/self.dof)))

        nearinds = []
        for i in range(0, n_nodes):
            if dist(self.nodeList[i].conf, newNode.conf) < radius:
                nearinds.append(i)
        return nearinds

    def to_be_added(self, newNode, nearinds, point_cloud, model=None):
        """
        Finds cost to goal from all nearinds.
        Compares minimum cost to goal with cost to goal from newNode
        if cost to goal from newNode is more, returns False, else true
        """
        # load model entire_model_env_2d_epoch_15000_pc.pt

        # print(point_cloud)
        # print(newNode.state)
        node_input = point_cloud
        node_input = np.append(node_input, self.end.conf)
        node_input = np.append(node_input, newNode.conf)

        node_input = node_input.flatten()
        data = np.array(node_input).astype(np.float32)
        # print(data)
        # normalize data by 20
        data = data/20.0
        data = torch.FloatTensor(data)
        # print(point_cloud)

        # model.load_state_dict(saved_model)
        cost_to_go = model(data)
        # print(cost_to_go)

        # for all nearnodes, find the positive minimum cost to go using model
        # initialize min cost to go from -inf
        min_cost_to_go = cost_to_go
        for i in nearinds:
            node = self.nodeList[i]
            node_input = point_cloud
            node_input = np.append(node_input, node.conf)
            node_input = np.append(node_input, self.end.conf)
            node_input = node_input.flatten()
            data = np.array(node_input).astype(np.float32)
            data = torch.FloatTensor(data)
            near_cost_to_go = model(data)
            if near_cost_to_go > 0 and cost_to_go > 0 and near_cost_to_go < cost_to_go:
                return False

        # print(m_cost_to_go, cost_to_go)
        # cost_to_go < min_cost_to_go  and both are not negative infinity
        if cost_to_go >= 0:
            print(cost_to_go)
            return True
        else:
            return False

    def rewire(self, newNode, newNodeIndex, nearinds):
        """
        Should examine all nodes near newNode, and decide whether to "rewire" them to go through newNode.
        Recall that a node should be rewired if doing so would reduce its cost.

        newNode: the node that was just inserted
        newNodeIndex: the index of newNode
        nearinds: list of indices of nodes near newNode
        """
        # your code here
        for i in nearinds:
            node = self.nodeList[i]
            valid, cost = self.steerTo3d(node, newNode)
            if valid:
                newCost = newNode.cost + cost
                if newCost < node.cost:
                    node.parent = newNodeIndex
                    # update cost of node according to new parent
                    node.cost = newCost
                    # update cost of all children of node
                    self.updateCost(node)

    def updateCost(self, node):
        # get children of node from set in class RRT_Node
        children = node.children
        for child in children:
            self.nodeList[child].cost = node.cost + \
                dist(self.nodeList[child].conf, node.conf)
            # update cost of all children of child
            self.updateCost(self.nodeList[child])

    def GetNearestListIndex(self, nodeList, rnd):
        """
        Searches nodeList for the closest vertex to rnd

        nodeList: list of all nodes currently in the tree
        rnd: node to be added (not currently in the tree)

        Returns: index of nearest node
        """
        dlist = []
        for node in nodeList:
            dlist.append(dist(rnd.conf, node.conf))

        minind = dlist.index(min(dlist))

        return minind

    
    def get_path_to_goal(self):
        """
        Traverses the tree to chart a path between the start state and the goal state.
        There may be multiple paths already discovered - if so, this returns the shortest one

        Returns: a list of coordinates, representing the path backwards; if a path has been found; None otherwise
        """
        if self.goalfound:
            goalind = None
            mincost = float('inf')
            for idx in self.solutionSet:
                cost = self.nodeList[idx].cost + \
                    dist(self.nodeList[idx].conf, self.end.conf)
                if goalind is None or cost < mincost:
                    goalind = idx
                    mincost = cost
            return self.gen_final_course(goalind)
        else:
            return None

    
# class Node():
#     """
#     RRT Node
#     """

#     def __init__(self, state):
#         self.state = state
#         self.cost = 0.0
#         self.parent = None
#         self.children = set()


def main():
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
    parser.add_argument('-g', '--geom', default='point', choices=['point', 'circle', 'rectangle'],
                        help='the geometry of the robot. Choose from "point" (Question 1), "circle" (Question 2), or "rectangle" (Question 3). default: "point"')
    parser.add_argument('--alg', default='rrtstar', choices=['rrt', 'rrtstar'],
                        help='which path-finding algorithm to use. default: "rrt"')
    parser.add_argument('--iter', default=100, type=int,
                        help='number of iterations to run')
    parser.add_argument('--blind', action='store_true',
                        help='set to disable all graphs. Useful for running in a headless session')
    parser.add_argument('--fast', action='store_true',
                        help='set to disable live animation. (the final results will still be shown in a graph). Useful for doing timing analysis')
    parser.add_argument('--env-id', default=1, type=int)
    parser.add_argument('--env-type', default='3d', type=str)
    parser.add_argument('--nn', action='store_true')
    parser.add_argument('--sample', default='directed',
                        type=str, choices=['directed', 'normal'])
    parser.add_argument('--get-results', action='store_true')
    parser.add_argument('--show-animation', action='store_true',
                        help='set to show edges in the graph', default=True)
    

    args = parser.parse_args()
    print(args)

    show_animation = not args.blind and not args.fast

    print("Starting planning algorithm '%s' with '%s' robot geometry" %
          (args.alg, args.geom))
    starttime = time.time()

    env_path = 'envs/{}/env{}.pkl'.format(args.env_type, args.env_id)

    obstacleList = []
    env = pickle.load(open(env_path, 'rb'))
    obstacleList = env

    env_pc_path = 'envs/{}/env{}_pc.pkl'.format(args.env_type, args.env_id)

    pc = pickle.load(open(env_pc_path, 'rb'))

    start = np.random.uniform(-20, 20, 2)
    start = [-18, 10]
      
    goal = np.random.uniform(-20, 20, 2)
    goal = [15, 19]
    goal = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                                                                        math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))

    dof = 3
    
    
    dof = 3
    # TODO uncomment following necessary lines for 3D gui window (doesm't work in ssh)
    if args.show_animation:
        physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))
    else:
        physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setGravity(0, 0, -9.8)

    # load obstacles from pkl file
    with open('envs/3d/env{}.pkl'.format(args.env_id), 'rb') as f:
        env = pickle.load(f)

    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf',
                        basePosition=[0, 0, 0.02], useFixedBase=True)

    obstacles = [plane]

    for i in range(len(env)):
        obstacles.append(p.loadURDF('assets/block.urdf',
                                    basePosition=env[i],
                                    useFixedBase=True))

    collisionCheck3d = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                        attachments=[], self_collisions=True,
                                        disabled_collisions=set())




    start = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)


    # start = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))
    # goal = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))





    set_joint_positions(ur5, UR5_JOINT_INDICES, start)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    goal_marker = draw_sphere_marker(position=start_position, radius=0.02, color=[1, 0, 1, 1])



    rrt = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud=pc, obstacleList=obstacleList,
              dof=dof, alg=args.alg, geom=args.geom, maxIter=args.iter, sample=args.sample, env=args.env_type, ur5=ur5, collisionCheck3d=collisionCheck3d)

    total_input_size = 2806
    output_size = 1

    activation_f = torch.nn.ReLU

    model = MLPComplete(total_input_size, output_size,
                        activation_f=activation_f, dropout=0)
    CAE = CAE_2d
    MLP = mlp.MLP
    total_input_size = 2806
    AE_input_size = 2800
    mlp_input_size = 28+6
    model = End2EndMPNet(total_input_size, AE_input_size, mlp_input_size,
                         output_size, CAE, MLP, activation_f=activation_f, dropout=0.0)

    # model.load('entire_model_env_2d_epoch_15000_pc.pt')
    model = torch.load(
        'entire_model_env_3d_epoch_2850.pt', map_location='cpu')
    model.eval()

    starttime = time.time()
    path, firsttime, minTime = rrt.planning3d(
        show_animation=args.show_animation, model=model)
    endtime = time.time()
    starttime2 = time.time()

    endtime2 = time.time()

    print('Sample Type: ', args.sample)
    print("Time taken: ", endtime - starttime)
    if path is None:
        print("FAILED to find a path in %.2fsec" % (endtime - starttime))
    else:
        print("SUCCESS - found path of cost %.5f in %.2fsec" %
              (rrt.get_path_len(path), endtime - starttime))
        print("First time: ", firsttime - starttime)
        print("Min time: ", minTime - starttime)

    # print('Sample Type: ', 'normal' if args.sample == 'directed' else 'directed')
    # print("Time taken: ", endtime2 - starttime2)

    if not args.blind and path is not None:

        while True:
            for q in path:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(0.5)



if __name__ == '__main__':
    main()
