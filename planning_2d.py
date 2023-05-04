"""
File: planning_2d.py

"""


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
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from math import sqrt




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

class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea, pointcloud, dof=2, expandDis=0.05, goalSampleRate=5, maxIter=100, sample='normal'):
        """
        Sets algorithm parameters
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,width,height],...]
        pointcloud: point cloud representation of obstacles
        randArea:Ramdom Samping Area [min,max]
        """
        self.start = Node(start)
        self.end = Node(goal)
        self.obstacleList = obstacleList
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.dof = dof
        self.pointcloud = pointcloud
        self.min_cost_to_go = 1000000
        self.sample = sample
        self.cost_to_go = {}

        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter

        self.cost_to_go[tuple(self.end.state)] = 0
        self.cost_to_go[tuple(self.start.state)] = 1000000

        self.goalfound = False
        self.solutionSet = set()

    def planning(self, animation=False, model=None):
        """
        Implements the RTT* algorithm
        animation: flag for animation on or off
        """

        min_time = time.time()
        firstTime = time.time()
        
        if not self.__CollisionCheck(self.end) or not self.__CollisionCheck(self.start):
            return [], firstTime, min_time

        self.nodeList = [self.start]
        point_cloud = self.pointcloud
        min_cost = float('inf')
        
        for i in range(self.maxIter):

            # generating new sample and get nearest node index
            rnd = self.generatesample(point_cloud=point_cloud, model=model)
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            # checking if the sampled node can be connected to nearest nodes
            rnd_valid, rnd_cost = self.steerTo(rnd, self.nodeList[nind])

            if (rnd_valid):
                newNode = copy.deepcopy(rnd)
                newNode.parent = nind
                newNode.cost = rnd_cost + self.nodeList[nind].cost

                nearinds = self.find_near_nodes(newNode)
                newParent = self.choose_parent(newNode, nearinds) 
        
                # inserting newNode into the tree
                if newParent is not None:
                    newNode.parent = newParent
                    newNode.cost = dist(newNode.state, self.nodeList[newParent].state) + self.nodeList[newParent].cost
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

                if animation:
                    self.draw_graph(rnd.state)

        return self.get_path_to_goal(), firstTime, min_time


    def choose_parent(self, newNode, nearinds):
        """
        Selects the best parent for newNode. This should be the one that results in newNode having the lowest possible cost.

        newNode: the node to be inserted
        nearinds: a list of indices. Contains nodes that are close enough to newNode to be considered as a possible parent.

        Returns: index of the new parent selected
        """
        minCost = float('inf')
        minIndex = None
        for i in nearinds:
            valid, cost = self.steerTo(newNode, self.nodeList[i])
            if valid:
                cost += self.nodeList[i].cost
                if cost < minCost:
                    minCost = cost
                    minIndex = i
        
        return minIndex

    def steerTo(self, dest, source):
        """
        Checks whether the route from source to destination is collision-free.
        Discretizes the route into small steps, and checks for a collision at each step.


        dest: destination node
        source: source node

        returns: (success, cost) tuple
            - success is True if the route is collision free; False otherwise.
            - cost is the distance from source to dest, if the route is collision free; or None otherwise.
        """

        newNode = copy.deepcopy(source)

        DISCRETIZATION_STEP=self.expandDis

        dists = np.zeros(self.dof, dtype=np.float32)
        for j in range(0,self.dof):
            dists[j] = dest.state[j] - source.state[j]

        distTotal = magnitude(dists)

        if distTotal>0:
            incrementTotal = distTotal/DISCRETIZATION_STEP
            for j in range(0,self.dof):
                dists[j] =dists[j]/incrementTotal

            numSegments = int(math.floor(incrementTotal))+1

            stateCurr = np.zeros(self.dof,dtype=np.float32)
            for j in range(0,self.dof):
                stateCurr[j] = newNode.state[j]

            stateCurr = Node(stateCurr)

            for i in range(0,numSegments):

                if not self.__CollisionCheck(stateCurr):
                    return (False, None)

                for j in range(0,self.dof):
                    stateCurr.state[j] += dists[j]

            if not self.__CollisionCheck(dest):
                return (False, None)

            return (True, distTotal)
        else:
            return (False, None)

    def generatesample(self, model=None, point_cloud=None):
        """
        Randomly generates a sample, to be used as a new node. Checks if the cost to goal from generated node is less than cost to
        goal from its near nodes. Returns the sample only if the cost-to-go is less.

        model: cost-to-go model
        point_cloud: point cloud of the environment

        returns: random c-space vector
        """
        if random.randint(0, 100) > self.goalSampleRate:
            cost_to_go = float('inf')
            env_data = self.end.state
            point_cloud = np.array(point_cloud).astype(np.float32)
            point_cloud = point_cloud.flatten()
            point_cloud  = point_cloud / 20.0
            point_cloud = point_cloud.reshape(1, -1)
            point_cloud = torch.FloatTensor(point_cloud)

            max_sample = 50
            min_near_neighbor_cost = 100000

            i = 0

            # samples new configuration until cost-to-go is less than cost to go from near nodes
            while cost_to_go > min_near_neighbor_cost + 5  or self.sample == 'normal':
                i += 1
                if i > max_sample:
                    break

                sample=[]
                for j in range(0,self.dof):
                    sample.append(random.uniform(self.minrand, self.maxrand))
                rnd = Node(sample)

                if self.sample == 'normal':
                    break

                # checks minmum cost-to-go from near nodes
                near_nodes = self.find_near_nodes(rnd)
                for node in near_nodes:
                    near_cost = self.cost_to_go[tuple(self.nodeList[node].state)]
                    if near_cost < min_near_neighbor_cost:
                        min_near_neighbor_cost = near_cost

                node_input = np.append(env_data, rnd.state)
                node_input = node_input.flatten()
                data = np.array(node_input).astype(np.float32)
                data = data/20.0
                data = data.reshape(1, -1)
                data = torch.FloatTensor(data)
                cost_to_go = model(data, point_cloud) * 20.0
                self.cost_to_go[tuple(rnd.state)] = cost_to_go
                
                if len(near_nodes) == 0:
                    break
            
            self.min_cost_to_go = cost_to_go

        else:
            rnd = self.end

        return rnd

    def is_near_goal(self, node):
        """
        node: the location to check

        Returns: True if node is within 5 units of the goal state; False otherwise
        """
        d = dist(node.state, self.end.state)
        if d < 5.0:
            return True
        return False

    @staticmethod
    def get_path_len(path):
        """
        path: a list of coordinates

        Returns: total length of the path
        """
        pathLen = 0
        for i in range(1, len(path)):
            pathLen += dist(path[i], path[i-1])

        return pathLen


    def gen_final_course(self, goalind):
        """
        Traverses up the tree to find the path from start to goal

        goalind: index of the goal node

        Returns: a list of coordinates, representing the path backwards. Traverse this list in reverse order to follow the path from start to end
        """
        path = [self.end.state]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            path.append(node.state)
            goalind = node.parent
        path.append(self.start.state)
        return path

    def find_near_nodes(self, newNode):
        """
        Finds all nodes in the tree that are "near" newNode.

        newNode: the node to be inserted.

        Returns: a list of indices of nearby nodes.
        """
        GAMMA = 50
        n_nodes = len(self.nodeList)
        radius = GAMMA * (pow((np.log(n_nodes)/n_nodes), (1/self.dof)))
        nearinds = []
        for i in range(0, n_nodes):
            if dist(self.nodeList[i].state, newNode.state) < radius:
                nearinds.append(i)
        return nearinds


    def rewire(self, newNode, newNodeIndex, nearinds):
        """
        examines all nodes near newNode, and decide whether to "rewire" them to go through newNode.

        newNode: the node that was just inserted
        newNodeIndex: the index of newNode
        nearinds: list of indices of nodes near newNode
        """
        for i in nearinds:
            node = self.nodeList[i]
            valid, cost = self.steerTo(node, newNode)
            if valid:
                newCost = newNode.cost + cost
                if newCost < node.cost:
                    node.parent = newNodeIndex
                    node.cost = newCost
                    self.updateCost(node)

    def updateCost(self, node):
        """
        Updates the cost of all children of node according to the cost of node.
        """
        children = node.children
        for child in children:
            self.nodeList[child].cost = node.cost + dist(self.nodeList[child].state, node.state)
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
            dlist.append(dist(rnd.state, node.state))

        minind = dlist.index(min(dlist))

        return minind

    def __CollisionCheck(self, node):
        """
        Checks whether a given configuration is valid i.e not in obstacles
        """      
        s = np.zeros(2, dtype=np.float32)
        s[0] = node.state[0]
        s[1] = node.state[1]

        for (ox, oy, sizex,sizey) in self.obstacleList:
            obs=[ox+sizex/2.0,oy+sizey/2.0]
            obs_size=[sizex,sizey]
            cf = False
            for j in range(self.dof):
                if abs(obs[j] - s[j])>obs_size[j]/2.0:
                    cf=True
                    break
            if cf == False:
                return False

        return True 


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
                cost = self.nodeList[idx].cost + dist(self.nodeList[idx].state, self.end.state)
                if goalind is None or cost < mincost:
                    goalind = idx
                    mincost = cost
            return self.gen_final_course(goalind)
        else:
            return None

    def draw_graph(self, rnd=None):
        """
        Draws the state space, with the tree, obstacles, and shortest path. Useful for visualization.

        """
        plt.clf()
        # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect(
        #     'key_release_event',
        #     lambda event: [exit(0) if event.key == 'escape' else None])

        for (ox, oy, sizex, sizey) in self.obstacleList:
            rect = mpatches.Rectangle((ox, oy), sizex, sizey, fill=True, color="purple", linewidth=0.1)
            plt.gca().add_patch(rect)


        for node in self.nodeList:
            if node.parent is not None:
                if node.state is not None:
                    plt.plot([node.state[0], self.nodeList[node.parent].state[0]], [node.state[1], self.nodeList[node.parent].state[1]], "-g")
                        

        if self.goalfound:
            path = self.get_path_to_goal()
            x = [p[0] for p in path]
            y = [p[1] for p in path]
            plt.plot(x, y, '-r')

        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")

        plt.plot(self.start.state[0], self.start.state[1], "xr")
        plt.plot(self.end.state[0], self.end.state[1], "xr")
        plt.axis("equal")
        plt.axis([-20, 20, -20, 20])
        plt.grid(True)
        plt.pause(0.01)



class Node():
    """
    RRT Node
    """

    def __init__(self,state):
        self.state =state
        self.cost = 0.0
        self.parent = None
        self.children = set()


def check_collision(obstacles, point):
    """
    Checks if a point is in collision with any obstacles
    """
    x = point[0]
    y = point[1]

    for obstacle in obstacles:
        obs_x = obstacle[0]
        obs_y = obstacle[1]
        obs_sizex = obstacle[2]
        obs_sizey = obstacle[3]

        if x >= obs_x and x <= obs_x + obs_sizex and y >= obs_y and y <= obs_y + obs_sizey:
            return True
        
    return False


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
    parser.add_argument('--iter', default=100, type=int, help='number of iterations to run')
    parser.add_argument('--blind', action='store_true', help='set to disable all graphs. Useful for running in a headless session')
    parser.add_argument('--fast', action='store_true', help='set to disable live animation. (the final results will still be shown in a graph). Useful for doing timing analysis')
    parser.add_argument('--env', default=1, type=int)
    parser.add_argument('--nn', action='store_true')
    parser.add_argument('--sample', default='directed', type=str, choices=['directed', 'normal'])
    parser.add_argument('--get-results', action='store_true')
    parser.add_argument('--model-path', default='test_models/entire_model_env_2d_epoch_2800.pt')


    args = parser.parse_args()

    show_animation = not args.blind and not args.fast

    starttime = time.time()

    # creates environment path from environment id and loads obstacles from the file
    env_path = 'envs/2d/env{}.pkl'.format(args.env)

    obstacleList = []
    env = pickle.load(open(env_path, 'rb'))
    obstacleList = env

    # creates pointcloud file path from p environment id
    env_pc_path = 'envs/2d/env{}_pc.pkl'.format(args.env)

    # loads pointcloud from the file
    pc = pickle.load(open(env_pc_path, 'rb'))

    start = np.random.uniform(-20, 20, 2)
    goal = np.random.uniform(-20, 20, 2)

    # while start and goal are in collision with obstacles, sample new start and goal
    while check_collision(obstacleList, start):
        start = np.random.uniform(-20, 20, 2)

    while check_collision(obstacleList, goal):
        goal = np.random.uniform(-20, 20, 2)
 
    dof = 2

    # model parameters
    total_input_size = 2804
    output_size = 1
    activation_f = torch.nn.ReLU
    model = MLPComplete(total_input_size, output_size, activation_f=activation_f, dropout=0)
    CAE = CAE_2d
    MLP = mlp.MLP
    total_input_size = 2804
    AE_input_size = 2800
    mlp_input_size = 28+4
    model = End2EndMPNet(total_input_size, AE_input_size, mlp_input_size,
                         output_size, CAE, MLP, activation_f=activation_f, dropout=0.0)
    
    # loading model from model_path
    model = torch.load(args.model_path, map_location ='cpu')
    model.eval()

    num_iter = 1
    d_count = 0
    d_time = 0
    n_count = 0
    n_time = 0
    d_path_length = 0
    n_path_length = 0
    d_cost = 0
    n_cost = 0
    both = 0

    if args.get_results:
        num_iter = 300
    

    if args.get_results:
        for it in tqdm(range(num_iter)):
            # randomly generating environment id
            env_id = random.randint(0, 9)
            env_path = 'envs/2d/env{}.pkl'.format(env_id)

            obstacleList = []
            env = pickle.load(open(env_path, 'rb'))
            obstacleList = env

            env_pc_path = 'envs/2d/env{}_pc.pkl'.format(env_id)
            pc = pickle.load(open(env_pc_path, 'rb'))

            start = np.random.uniform(-20, 20, 2)
            goal = np.random.uniform(-20, 20, 2)

            while check_collision(obstacleList, start):
                start = np.random.uniform(-20, 20, 2)

            while check_collision(obstacleList, goal):
                goal = np.random.uniform(-20, 20, 2)
        
            # Initializing 2 rrtstar planners- one for normal RRT*, one for guided (cost-to-go) RRT*
            rrt = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud = pc, obstacleList=obstacleList, dof=dof, maxIter=args.iter, sample=args.sample)
            rrt2 = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud = pc, obstacleList=obstacleList, dof=dof, maxIter=args.iter, sample='normal' if args.sample == 'directed' else 'directed')

            starttime = time.time()
            path, firsttime, minTime = rrt.planning(animation=False, model=model)
            endtime = time.time()
            starttime2 = time.time()
            path2, firsttime2, minTime2 = rrt2.planning(animation=False, model=model)
            endtime2 = time.time()

            if path is not None:
                if len(path) == 0:
                    it -= 1
                    continue

            if path is not None:
                d_count += 1
                d_time += minTime - starttime
                
            if path2 is not None:
                n_count += 1
                n_time += minTime2 - starttime2

            if path is not None and path2 is not None:
                both += 1
                d_path_length += len(path)
                d_cost += rrt.get_path_len(path)
                
                n_path_length += len(path2)
                n_cost += rrt2.get_path_len(path2)

            

            if it % 20 == 0 and it != 0:
                print('directed success rate: ', d_count/it, flush=True)
                print('directed average time: ', d_time/it, flush=True)
                print('directed average path length: ', d_path_length/both, flush=True)
                print('directed average cost: ', d_cost/both, flush=True)
                print('normal success rate: ', n_count/it, flush=True)
                print('normal average time: ', n_time/it, flush=True)
                print('normal average path length: ', n_path_length/both, flush=True)
                print('normal average cost: ', n_cost/both, flush=True)

                    
                    
        print('directed success rate: ', d_count/num_iter)
        print('directed average time: ', d_time/num_iter)
        print('directed average path length: ', d_path_length/both)
        print('directed average cost: ', d_cost/both)
        print('normal success rate: ', n_count/num_iter)
        print('normal average time: ', n_time/num_iter)
        print('normal average path length: ', n_path_length/both)
        print('normal average cost: ', n_cost/both)

    else:
        # Initializing 2 rrtstar planners- one for normal RRT*, one for guided (cost-to-go) RRT*
        rrt = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud = pc, obstacleList=obstacleList, dof=dof, maxIter=args.iter, sample=args.sample)
        rrt2 = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud = pc, obstacleList=obstacleList, dof=dof, maxIter=args.iter, sample='normal' if args.sample == 'directed' else 'directed')

        while check_collision(obstacleList, start):
            start = np.random.uniform(-20, 20, 2)

        while check_collision(obstacleList, goal):
            goal = np.random.uniform(-20, 20, 2)

        starttime = time.time()
        path, firsttime, minTime = rrt.planning(animation=show_animation, model=model)
        endtime = time.time()
        starttime2 = time.time()
        path2, firsttime2, minTime2 = rrt2.planning(animation=show_animation, model=model)
        endtime2 = time.time()

        print('Sample Type: ', args.sample)
        print("Time taken: ", endtime - starttime)
        if path is None:
            print("FAILED to find a path in %.2fsec"%(endtime - starttime))
        else:
            print("SUCCESS - found path of cost %.5f in %.2fsec"%(rrt.get_path_len(path), endtime - starttime))
            print("First time: ", firsttime - starttime)
            print("Min time: ", minTime - starttime)
        
        print('Sample Type: ', 'normal' if args.sample == 'directed' else 'directed')
        print("Time taken: ", endtime2 - starttime2)
        if path2 is None:
            print("FAILED to find a path in %.2fsec"%(endtime2 - starttime2))
        else:
            print("SUCCESS - found path of cost %.5f in %.2fsec"%(rrt2.get_path_len(path2), endtime2 - starttime2))
            print("First time: ", firsttime2 - starttime2)
            print("Min time: ", minTime2 - starttime2)



        if not args.blind:
            
            rrt.draw_graph()    
            plt.show()

           
if __name__ == '__main__':
    main()
