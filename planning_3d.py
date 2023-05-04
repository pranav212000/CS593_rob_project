"""
File: planning_3d.py

"""


import warnings
from math import sqrt
import argparse
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
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
warnings.filterwarnings("ignore")


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
    print(p1, p2)
    return magnitude(diff(p1, p2))


def get_pointcloud(obstaclelist):
    """
    Given a list of obstacles, returns a point cloud of the obstacles
    """
    point_cloud = []
    num_points = 200
    for point in obstaclelist:
        left_bottom_x = point[0]
        left_bottom_y = point[1]
        size_x = point[2]
        size_y = point[3]

        # uniformly sampled points in the environment
        for i in range(num_points):
            x = np.random.uniform(left_bottom_x, left_bottom_x + size_x)
            y = np.random.uniform(left_bottom_y, left_bottom_y + size_y)
            point_cloud.append([x, y])

    point_cloud = np.array(point_cloud)
    return point_cloud


def getEndEffectorPos(ur5):
    """
    Returns the end effector position of the UR5
    """
    num_joints = p.getNumJoints(ur5)
    link_id = num_joints - 1
    link_state = p.getLinkState(
        ur5, link_id, computeForwardKinematics=True)
    link_pos = link_state[0]
    link_ori = link_state[1]

    return link_pos


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea, alg, pointcloud, dof=2, expandDis=0.05, goalSampleRate=5, maxIter=100, sample='normal', collisionCheck3d=None, env='2d', ur5=None):
        """
        Sets algorithm parameters
        start:Start Position [x,y,z]
        goal:Goal Position [x,y,z]
        obstacleList:obstacle Positions [[x,y,z,size],...]
        pointcloud: point cloud representation of obstacles
        """

        self.start = Node(conf=start, ur5=ur5)
        self.end = Node(conf=goal, ur5=ur5)
        self.obstacleList = obstacleList
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.alg = alg
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
        """
        Returns the distance between two nodes
        """
        if hasattr(node1, 'conf'):
            return np.linalg.norm(np.array(node1.conf) - np.array(node2.conf))
        else:
            return np.linalg.norm(np.array(node1) - np.array(node2))

    def steerTo3d(self, rand_node, nearest_node, step_size=0.05, show_animation=True):
        """
        Steer to function for 3d environment
        """
        distance = self.getDistance(rand_node, nearest_node)
        n_steps = round(distance/step_size)
        if n_steps == 0:
            return (self.collisionCheck3d(rand_node.conf), distance)
        unit_step = (np.array(rand_node.conf) -
                     np.array(nearest_node.conf)) / n_steps
        start = np.array(nearest_node.conf)
        link_pos = []
        start_pos = getEndEffectorPos(self.ur5)
        for i in range(n_steps):
            start = start + unit_step
            start = (start[0], start[1], start[2])
            link_pos.append(getEndEffectorPos(self.ur5))

            if self.collisionCheck3d(start):
                return (False, None)

        end_pos = getEndEffectorPos(self.ur5)

        if show_animation:
            if len(link_pos) > 10:
                link_pos = link_pos[::len(link_pos)//2]

            link_pos.insert(0, start_pos)
            link_pos.append(end_pos)

            # for i in range(len(link_pos)-1):
            #     p.addUserDebugLine(link_pos[i], link_pos[i+1], [1, 0, 0], 1, 0)

        return (True, distance)

    def planning3d(self, show_animation=False, model=None):
        """
        Implements the RTT* algorithm.
        animation: flag for animation on or off
        """

        min_time = time.time()
        firstTime = time.time()

        if self.collisionCheck3d(self.start.conf) or self.collisionCheck3d(self.end.conf):
            return [], firstTime, min_time

        self.nodeList = [self.start]
        point_cloud = self.pointcloud
        min_cost = float('inf')

        for i in range(self.maxIter):

            rnd = self.generatesample(point_cloud=point_cloud, model=model)
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            rnd_valid, rnd_cost = self.steerTo3d(
                rnd, self.nodeList[nind], show_animation=show_animation)

            if (rnd_valid):
                newNode = copy.deepcopy(rnd)
                newNode.parent = nind
                newNode.cost = rnd_cost + self.nodeList[nind].cost

                nearinds = self.find_near_nodes(newNode)
                newParent = self.choose_parent(newNode, nearinds)

                if newParent is not None:
                    newNode.parent = newParent
                    newNode.cost = self.getDistance(
                        newNode, self.nodeList[newParent]) + self.nodeList[newParent].cost

                else:
                    pass  
                self.nodeList.append(newNode)
                newNodeIndex = len(self.nodeList) - 1
                self.nodeList[newNode.parent].children.add(newNodeIndex)

                if self.sample == 'normal':
                    self.rewire(newNode, newNodeIndex, nearinds)

                if self.is_near_goal(newNode):
                    self.solutionSet.add(newNodeIndex)
                    self.goalfound = True
                    firstTime = time.time()
                    cost = self.get_path_len(self.get_path_to_goal())
                    if cost < min_cost:
                        min_time = time.time()

        return self.get_path_to_goal(), firstTime, min_time

    def choose_parent(self, newNode, nearinds):
        """
        Selects the best parent for newNode. This should be the one that results in newNode having the lowest possible cost.

        newNode: the node to be inserted
        nearinds: a list of indices. Contains nodes that are close enough to newNode to be considered as a possible parent.

        Returns: index of the new parent selected
        """

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
        if sample == 'normal' then it samples a random point
        if sample == 'directed' then samples based on the cost-to-go from the model.
        if the cost to go is more than the neighbor's cost to go then sample again. 

        returns: random c-space vector
        """

        if random.randint(0, 100) > self.goalSampleRate:
            cost_to_go = float('inf')

            env_data = self.end.conf
            point_cloud = np.array(point_cloud).astype(np.float32)
            point_cloud = point_cloud.flatten()
            point_cloud = point_cloud / 20.0

            point_cloud = point_cloud.reshape(1, -1)
            point_cloud = torch.FloatTensor(point_cloud)

            prob_to_rand = np.random.random()

            max_sample = 50

            min_near_neighbor_cost = 10000
            i = 0

            while cost_to_go > min_near_neighbor_cost + 5 or self.sample == 'normal':
                i += 1
                if i > max_sample:
                    break

                while True:

                    rnd = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                                                                                       math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))

                    set_joint_positions(self.ur5, UR5_JOINT_INDICES, rnd)
                    rnd = Node(conf=rnd, ur5=self.ur5)

                    if not self.collisionCheck3d(rnd.conf):
                        break

                # if sample == 'normal' just sample one point
                if self.sample == 'normal':
                    break

                # get near nodes and compare the cost to go with current rand node
                near_nodes = self.find_near_nodes(rnd)
                for node in near_nodes:
                    near_cost = self.cost_to_go[tuple(
                        self.nodeList[node].conf)]
                    if near_cost < min_near_neighbor_cost:
                        min_near_neighbor_cost = near_cost

                node_input = np.append(env_data, rnd.conf)

                node_input = node_input.flatten()
                data = np.array(node_input).astype(np.float32)

                # Scaling the data
                data = data/20.0
                data = data.reshape(1, -1)
                data = torch.FloatTensor(data)

                cost_to_go = model(data, point_cloud) * 20.0

                self.cost_to_go[tuple(rnd.conf)] = cost_to_go

                # No near nodes, return the rand node
                if len(near_nodes) == 0:
                    break

            self.min_cost_to_go = cost_to_go

        else:
            rnd = self.end

        draw_sphere_marker(position=rnd.state,
                           radius=0.005, color=[1, 1, 0, 1])

        return rnd

    def is_near_goal(self, node):
        """
        node: the location to check

        Returns: True if node is within 5 units of the goal state; False otherwise
        """
        d = self.getDistance(node.conf, self.end.conf)
        if d < 5:
            return True
        return False

    def get_path_len(self, path, env='2d'):
        """
        path: a list of coordinates

        Returns: total length of the path
        """
        pathLen = 0
        for i in range(1, len(path)):
            pathLen += self.getDistance(path[i], path[i-1])

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
        GAMMA = 5

        n_nodes = len(self.nodeList)

        radius = GAMMA * (pow((np.log(n_nodes)/n_nodes), (1/self.dof)))

        nearinds = []
        for i in range(0, n_nodes):
            if self.getDistance(self.nodeList[i].conf, newNode.conf) < radius:
                nearinds.append(i)
        return nearinds

    def rewire(self, newNode, newNodeIndex, nearinds):
        """
        Examines all nodes near newNode, and decide whether to "rewire" them to go through newNode.

        newNode: the node that was just inserted
        newNodeIndex: the index of newNode
        nearinds: list of indices of nodes near newNode
        """
        for i in nearinds:
            node = self.nodeList[i]
            valid, cost = self.steerTo3d(node, newNode)
            if valid:
                newCost = newNode.cost + cost
                if newCost < node.cost:
                    node.parent = newNodeIndex
                    node.cost = newCost
                    self.updateCost(node)

    def updateCost(self, node):
        """
        Recursively update the cost of all nodes that pass through node
        """
        children = node.children
        for child in children:
            self.nodeList[child].cost = node.cost + \
                self.getDistance(self.nodeList[child].conf, node.conf)
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
            dlist.append(self.getDistance(rnd.conf, node.conf))

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
                    self.getDistance(self.nodeList[idx].conf, self.end.conf)
                if goalind is None or cost < mincost:
                    goalind = idx
                    mincost = cost
            return self.gen_final_course(goalind)
        else:
            return None


def main():
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
    parser.add_argument('--iter', default=100, type=int,
                        help='number of iterations to run')
    parser.add_argument('--env-id', default=1, type=int)
    parser.add_argument('--sample', default='directed',
                        type=str, choices=['directed', 'normal'])
    parser.add_argument('--get-results', action='store_true')
    parser.add_argument('--show-animation', action='store_true',
                        help='set to show edges in the graph', default=False)
    parser.add_argument('--model-path', type=str,
                        default='test_models/entire_model_env_3d_epoch_2850.pt')

    args = parser.parse_args()
    print(args)

    print("Starting planning algorithm RRTStar")


    # TODO: change model path
    model_path = args.model_path
    # model_path = 'entire_model_env_3d_epoch_1700.pt'
    # model_path = 'models/04_032813/entire_model_env_3d_epoch_2300.pt'
    model = torch.load(
        model_path, map_location='cpu')
    print('Using model: ', model_path)
    model.eval()

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

    d_count = 0
    d_time = 0
    n_count = 0
    n_time = 0
    d_path_length = 0
    n_path_length = 0
    d_cost = 0
    n_cost = 0
    both = 0

    test_iterations = 300 if args.get_results else 1

    for test in tqdm(range(1, test_iterations + 1)):

        starttime = time.time()

        if args.get_results:
            env_id = np.random.randint(0, 10)
        else:
            env_id = args.env_id

        env_path = 'envs/3d/env{}.pkl'.format(env_id)

        obstacleList = []
        env = pickle.load(open(env_path, 'rb'))
        obstacleList = env

        env_pc_path = 'envs/3d/env{}_pc.pkl'.format(env_id)

        pc = pickle.load(open(env_pc_path, 'rb'))

        dof = 3

        with open('envs/3d/env{}.pkl'.format(env_id), 'rb') as f:
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

        start = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                 math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))
        set_joint_positions(ur5, UR5_JOINT_INDICES, start)
        while collisionCheck3d(start):
            start = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                     math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))
            set_joint_positions(ur5, UR5_JOINT_INDICES, start)

        goal = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))
        set_joint_positions(ur5, UR5_JOINT_INDICES, goal)
        while collisionCheck3d(goal):
            goal = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                    math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))
            set_joint_positions(ur5, UR5_JOINT_INDICES, goal)

        goal_position = getEndEffectorPos(ur5)

        set_joint_positions(ur5, UR5_JOINT_INDICES, start)
        start_position = getEndEffectorPos(ur5)

        if args.show_animation:
            goal_marker = draw_sphere_marker(
                position=goal_position, radius=0.02, color=[1, 0, 0, 1])
            goal_marker = draw_sphere_marker(
                position=start_position, radius=0.02, color=[1, 0, 1, 1])

        rrt = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud=pc, obstacleList=obstacleList,
                  dof=dof, alg='rrtstar',maxIter=args.iter, sample=args.sample, env='3d', ur5=ur5, collisionCheck3d=collisionCheck3d)

        rrt2 = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud=pc, obstacleList=obstacleList,
                   dof=dof, alg='rrtstar', maxIter=args.iter, sample='normal' if args.sample == 'directed' else 'normal', env='3d', ur5=ur5, collisionCheck3d=collisionCheck3d)

        starttime = time.time()
        path, firsttime, minTime = rrt.planning3d(
            show_animation=args.show_animation, model=model)
        endtime = time.time()
        set_joint_positions(ur5, UR5_JOINT_INDICES, start)
        starttime2 = time.time()
        path2, firsttime2, minTime2 = rrt2.planning3d(
            show_animation=args.show_animation, model=model)

        endtime2 = time.time()

        if path is not None:
            d_count += 1
            d_time += endtime - starttime

        if path2 is not None:
            n_count += 1
            n_time += endtime2 - starttime2

        if path is not None and path2 is not None:
            both += 1
            d_path_length += len(path)
            d_cost += rrt.get_path_len(path)

            n_path_length += len(path2)
            n_cost += rrt2.get_path_len(path2)

        if test % 20 == 0 and d_count != 0 and n_count != 0 and args.get_results:
            print('----------------------------------------------------')
            print('Sample Type: ', args.sample)

            print('Success Rate: ', d_count / test)
            print('Average time: ', d_time / test)
            print('Average path length: ', d_path_length / both)
            print('Average cost: ', d_cost / both)

            print('Sample Type: ', 'normal' if args.sample ==
                  'directed' else 'directed')

            print('Success Rate: ', n_count / test)
            print('Average time: ', n_time / test)
            print('Average path length: ', n_path_length / both)
            print('Average cost: ', n_cost / both, flush=True)

    if args.get_results:
        print('----------------------------------------------------')
        print('Sample Type: ', args.sample)
        print('Final Results')

        print('Success Rate: ', n_count / test_iterations)
        print('Average time: ', d_time / test_iterations)
        print('Average path length: ', d_path_length / both)
        print('Average cost: ', d_cost / both)

        print('Sample Type: ', 'normal' if args.sample ==
              'directed' else 'directed')

        print('Success Rate: ', n_count / test_iterations)
        print('Average time: ', n_time / test_iterations)
        print('Average path length: ', n_path_length / both)
        print('Average cost: ', n_cost / both)

    print('Sample Type: ', args.sample)
    print("Time taken: ", endtime - starttime)

    if path is None:
        print("FAILED to find a path in %.2fsec" % (endtime - starttime))
    else:
        print("SUCCESS - found path of cost %.5f in %.2fsec" %
              (rrt.get_path_len(path), endtime - starttime))
        print("First time: ", firsttime - starttime)
        print("Min time: ", minTime - starttime)
    print('----------------------------------------------------')
    print('Sample Type: ', 'normal' if args.sample == 'directed' else 'directed')
    print("Time taken: ", endtime2 - starttime2)

    if path2 is None:
        print("FAILED to find a path in %.2fsec" % (endtime2 - starttime2))
    else:
        print("SUCCESS - found path of cost %.5f in %.2fsec" %
              (rrt2.get_path_len(path2), endtime2 - starttime2))
        print("First time: ", firsttime2 - starttime2)
        print("Min time: ", minTime2 - starttime2)

    if args.show_animation:

        if path is not None:
            for _ in range(3):
                for q in path:
                    set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                    time.sleep(0.5)

        if path2 is not None:
            for _ in range(3):
                for q in path2:
                    set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                    time.sleep(0.5)


if __name__ == '__main__':
    main()
