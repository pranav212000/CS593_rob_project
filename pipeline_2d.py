"""
Path Planning Sample Code with RRT*

author: Ahmed Qureshi, code adapted from AtsushiSakai(@Atsushi_twi)

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

    def __init__(self, start, goal, obstacleList, randArea, alg, geom, pointcloud, dof=2, expandDis=0.05, goalSampleRate=5, maxIter=100, sample='normal'):
        """
        Sets algorithm parameters

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,width,height],...]
        randArea:Ramdom Samping Area [min,max]

        """
        self.start = Node(start)
        self.end = Node(goal)
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

        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter

        self.cost_to_go[tuple(self.end.state)] = 0
        self.cost_to_go[tuple(self.start.state)] = 1000000

        self.goalfound = False
        self.solutionSet = set()

    def planning1(self, animation=False, model=None):
        """
        Implements the RTT (or RTT*) algorithm, following the pseudocode in the handout.
        You should read and understand this function, but you don't have to change any of its code - just implement the 3 helper functions.

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
            # if i == self.maxIter - 1 and self.goalfound == False:
            #     self.maxIter += 10

            rnd = self.generatesample(point_cloud=point_cloud, model=model)
            nind = self.GetNearestListIndex(self.nodeList, rnd)


            rnd_valid, rnd_cost = self.steerTo(rnd, self.nodeList[nind])


            if (rnd_valid):
                newNode = copy.deepcopy(rnd)
                newNode.parent = nind
                newNode.cost = rnd_cost + self.nodeList[nind].cost

                if self.alg == 'rrtstar':
                    nearinds = self.find_near_nodes(newNode) # you'll implement this method
                    # to_add = self.to_be_added(newNode, nearinds, point_cloud)
                    newParent = self.choose_parent(newNode, nearinds) # you'll implement this method
                    # if to_add == True:
                    #     newParent = self.choose_parent(newNode, nearinds) # you'll implement this method
                    # else:
                    #     continue
                else:
                    newParent = None

                # insert newNode into the tree
                if newParent is not None:
                    newNode.parent = newParent
                    newNode.cost = dist(newNode.state, self.nodeList[newParent].state) + self.nodeList[newParent].cost
                else:
                    pass # nind is already set as newNode's parent
                self.nodeList.append(newNode)
                newNodeIndex = len(self.nodeList) - 1
                self.nodeList[newNode.parent].children.add(newNodeIndex)

                if self.alg == 'rrtstar' and self.sample == 'normal':
                    self.rewire(newNode, newNodeIndex, nearinds) # you'll implement this method

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

    def planning(self, animation=False, model=None):
        """
        Implements the RTT (or RTT*) algorithm, following the pseudocode in the handout.
        You should read and understand this function, but you don't have to change any of its code - just implement the 3 helper functions.

        animation: flag for animation on or off
        """

        if not self.__CollisionCheck(self.end):
            return None

        self.nodeList = [self.start]
        point_cloud = self.pointcloud
        for i in range(self.maxIter):
            rnd = self.generatesample()
            nind = self.GetNearestListIndex(self.nodeList, rnd)


            rnd_valid, rnd_cost = self.steerTo(rnd, self.nodeList[nind])


            if (rnd_valid):
                newNode = copy.deepcopy(rnd)
                newNode.parent = nind
                newNode.cost = rnd_cost + self.nodeList[nind].cost

                if self.alg == 'rrtstar':
                    nearinds = self.find_near_nodes(newNode) # you'll implement this method
                    to_add = self.to_be_added(newNode, nearinds, point_cloud, model)
                    if to_add == True:
                        newParent = self.choose_parent(newNode, nearinds) # you'll implement this method
                    else:
                        continue
                else:
                    newParent = None

                
                if newParent is not None:
                    newNode.parent = newParent
                    newNode.cost = dist(newNode.state, self.nodeList[newParent].state) + self.nodeList[newParent].cost
                else:
                    pass # nind is already set as newNode's parent
                self.nodeList.append(newNode)
                newNodeIndex = len(self.nodeList) - 1
                self.nodeList[newNode.parent].children.add(newNodeIndex)

                # if self.alg == 'rrtstar':
                #     self.rewire(newNode, newNodeIndex, nearinds) # you'll implement this method

                if self.is_near_goal(newNode):
                    self.solutionSet.add(newNodeIndex)
                    self.goalfound = True

                if animation:
                    self.draw_graph(rnd.state)

        return self.get_path_to_goal()

    def choose_parent(self, newNode, nearinds):
        """
        Selects the best parent for newNode. This should be the one that results in newNode having the lowest possible cost.

        newNode: the node to be inserted
        nearinds: a list of indices. Contains nodes that are close enough to newNode to be considered as a possible parent.

        Returns: index of the new parent selected
        """
        # your code here
        #find paths for all nodes in neainds and calculate the distance from source
        #find the minimum distance and return the index of the node
        #if no path is found return None
        minCost = float('inf')
        minIndex = None
        for i in nearinds:
            valid, cost = self.steerTo(newNode, self.nodeList[i])
            if valid:
                #find cost of reaching to i from source
                cost += self.nodeList[i].cost
                if cost < minCost:
                    minCost = cost
                    minIndex = i
        
        return minIndex

    def steerTo(self, dest, source):
        """
        Charts a route from source to dest, and checks whether the route is collision-free.
        Discretizes the route into small steps, and checks for a collision at each step.

        This function is used in planning() to filter out invalid random samples. You may also find it useful
        for implementing the functions in question 1.

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
        Randomly generates a sample, to be used as a new node.
        This sample may be invalid - if so, call generatesample() again.

        You will need to modify this function for question 3 (if self.geom == 'rectangle')

        returns: random c-space vector
        """
        if self.geom == 'rectangle':
            sample=[]
            sample.append(random.uniform(self.minrand, self.maxrand))
            sample.append(random.uniform(self.minrand, self.maxrand))
            sample.append(random.uniform(-math.pi, math.pi))
            rnd=Node(sample)
            return rnd
        else:
            if random.randint(0, 100) > self.goalSampleRate:
                cost_to_go = float('inf')
                # env_data = point_cloud
                # env_data = np.append(env_data, self.end.state)
                env_data = self.end.state
                point_cloud = np.array(point_cloud).astype(np.float32)
                point_cloud = point_cloud.flatten()
                point_cloud  = point_cloud / 20.0
                
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
                
                while cost_to_go > min_near_neighbor_cost + 5  or self.sample == 'normal':
                    i += 1
                    if i > max_sample:
                        break
                    sample=[]
                    for j in range(0,self.dof):
                        sample.append(random.uniform(self.minrand, self.maxrand))
                    rnd = Node(sample)
                    # print(sample)


                    if self.sample == 'normal':
                        break

                    near_nodes = self.find_near_nodes(rnd)
                    for node in near_nodes:
                        near_cost = self.cost_to_go[tuple(self.nodeList[node].state)]
                        # model(torch.FloatTensor(np.append(env_data, self.nodeList[node].state).reshape(1, -1)/20.0), point_cloud) * 20.0
                        if near_cost < min_near_neighbor_cost:
                            min_near_neighbor_cost = near_cost



                    node_input = np.append(env_data, rnd.state)
                    
                    node_input = node_input.flatten()
                    data = np.array(node_input).astype(np.float32)

                    # print(data.shape)

                    data = data/20.0
                    data = data.reshape(1, -1)
                    data = torch.FloatTensor(data)

            
                    cost_to_go = model(data, point_cloud) * 20.0
                    self.cost_to_go[tuple(rnd.state)] = cost_to_go
                    

                    if prob_to_rand < 0 or len(near_nodes) == 0:
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
            if dist(self.nodeList[i].state, newNode.state) < radius:
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
        node_input = np.append(node_input, self.end.state)
        node_input = np.append(node_input, newNode.state)
        
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
            node_input = np.append(node_input, node.state)
            node_input = np.append(node_input, self.end.state)
            node_input = node_input.flatten()
            data = np.array(node_input).astype(np.float32)
            data = torch.FloatTensor(data)
            near_cost_to_go = model(data)
            if  near_cost_to_go > 0 and cost_to_go > 0 and  near_cost_to_go < cost_to_go :
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
            valid, cost = self.steerTo(node, newNode)
            if valid:
                newCost = newNode.cost + cost
                if newCost < node.cost:
                    node.parent = newNodeIndex
                    #update cost of node according to new parent
                    node.cost = newCost
                    #update cost of all children of node
                    self.updateCost(node)

    def updateCost(self, node):
        # get children of node from set in class RRT_Node
        children = node.children
        for child in children:
            self.nodeList[child].cost = node.cost + dist(self.nodeList[child].state, node.state)
            #update cost of all children of child
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
        Checks whether a given configuration is valid. (collides with obstacles)

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
        """

        if self.geom == 'circle':
            s = np.zeros(2, dtype=np.float32)
            s[0] = node.state[0]
            s[1] = node.state[1]
            
            for (ox, oy, sizex, sizey) in self.obstacleList:
                testX = s[0]
                testY = s[1]
                if s[0] < ox:
                    testX = ox   
                elif s[0] > ox+sizex:
                    testX = ox+sizex  
                if s[1] < oy:
                    testY = oy      
                elif s[1] > oy+sizey: 
                    testY = oy+sizey

                distX = s[0]-testX
                distY = s[1]-testY
                distance = math.sqrt( (distX*distX) + (distY*distY) )

                if distance <= 1:
                    return False
                
            return True

        if self.geom == 'rectangle':
            s = np.zeros(3, dtype=np.float32)
            s[0] = node.state[0]
            s[1] = node.state[1]
            s[2] = node.state[2]

            #get rectangle corners
            x1 = s[0] + 0.75*math.cos(s[2]) - 1.5*math.sin(s[2])
            y1 = s[1] + 0.75*math.sin(s[2]) + 1.5*math.cos(s[2])
            x2 = s[0] + 0.75*math.cos(s[2]) + 1.5*math.sin(s[2])
            y2 = s[1] + 0.75*math.sin(s[2]) - 1.5*math.cos(s[2])
            x3 = s[0] - 0.75*math.cos(s[2]) - 1.5*math.sin(s[2])
            y3 = s[1] - 0.75*math.sin(s[2]) + 1.5*math.cos(s[2])
            x4 = s[0] - 0.75*math.cos(s[2]) + 1.5*math.sin(s[2])
            y4 = s[1] - 0.75*math.sin(s[2]) - 1.5*math.cos(s[2])

            #get list of tuple of vertices of rectangle
            vertices = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]


            #for every obstacle
            for (ox, oy, sizex, sizey) in self.obstacleList:
                # get obstacle corners
                ox1 = ox
                oy1 = oy
                ox2 = ox + sizex
                oy2 = oy
                ox3 = ox
                oy3 = oy + sizey
                ox4 = ox + sizex
                oy4 = oy + sizey

                # get list of tuple of vertices of obstacle
                obs_vertices = [(ox1,oy1), (ox2,oy2), (ox3,oy3), (ox4,oy4)]

                #check if rectangle and obstacle intersect
                if(separating_axis_theorem(obs_vertices, vertices)):
                    return False

            return True


        if self.geom == 'point':
            
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

            return True  # safe'''


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
        Draws the state space, with the tree, obstacles, and shortest path (if found). Useful for visualization.

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
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
                    if self.geom == 'circle':
                        circle = plt.Circle((node.state[0], node.state[1]), 1, color='g', fill=False)
                        plt.gca().add_patch(circle)
                    if self.geom == 'rectangle':
                        x  = node.state[0] 
                        y = node.state[1] 
                        theta = node.state[2]
                        
                        #top right corner
                        tr = np.zeros(2, dtype=np.float32)
                        tr[0] = x + 0.75*math.cos(theta) + 1.5*math.sin(theta)
                        tr[1] = y + 0.75*math.sin(theta) - 1.5*math.cos(theta)

                        #top left corner
                        tl = np.zeros(2, dtype=np.float32)
                        tl[0] = x - 0.75*math.cos(theta) + 1.5*math.sin(theta)

                        tl[1] = y - 0.75*math.sin(theta) - 1.5*math.cos(theta)

                        #bottom right corner
                        br = np.zeros(2, dtype=np.float32)
                        br[0] = x + 0.75*math.cos(theta) - 1.5*math.sin(theta)
                        br[1] = y + 0.75*math.sin(theta) + 1.5*math.cos(theta)

                        #bottom left corner
                        bl = np.zeros(2, dtype=np.float32)
                        bl[0] = x - 0.75*math.cos(theta) - 1.5*math.sin(theta)
                        bl[1] = y - 0.75*math.sin(theta) + 1.5*math.cos(theta)

                        plt.plot([tl[0], tr[0]], [tl[1], tr[1]], color='g')
                        plt.plot([tr[0], br[0]], [tr[1], br[1]], color='g')
                        plt.plot([br[0], bl[0]], [br[1], bl[1]], color='g')
                        plt.plot([bl[0], tl[0]], [bl[1], tl[1]], color='g')

    
                        plt.plot([node.state[0], self.nodeList[node.parent].state[0]], [
                            node.state[1], self.nodeList[node.parent].state[1]], "-g")
                        
                    plt.plot([node.state[0], self.nodeList[node.parent].state[0]], [node.state[1], self.nodeList[node.parent].state[1]], "-g")
                        

        if self.goalfound:
            path = self.get_path_to_goal()
            x = [p[0] for p in path]
            y = [p[1] for p in path]
            plt.plot(x, y, '-r')
            #if geometry is circle
            if self.geom == 'circle':
                for p in path:
                    circle = plt.Circle((p[0], p[1]), 1, color='r', fill=False)
                    plt.gca().add_patch(circle)

            #if geometry is rectangle
            if self.geom == 'rectangle':
                for p in path:
                    x  = p[0] 
                    y = p[1] 
                    theta = p[2]
                    
                    #top right corner
                    tr = np.zeros(2, dtype=np.float32)
                    tr[0] = x + 0.75*math.cos(theta) + 1.5*math.sin(theta)
                    tr[1] = y + 0.75*math.sin(theta) - 1.5*math.cos(theta)

                    #top left corner
                    tl = np.zeros(2, dtype=np.float32)
                    tl[0] = x - 0.75*math.cos(theta) + 1.5*math.sin(theta)

                    tl[1] = y - 0.75*math.sin(theta) - 1.5*math.cos(theta)

                    #bottom right corner
                    br = np.zeros(2, dtype=np.float32)
                    br[0] = x + 0.75*math.cos(theta) - 1.5*math.sin(theta)
                    br[1] = y + 0.75*math.sin(theta) + 1.5*math.cos(theta)

                    #bottom left corner
                    bl = np.zeros(2, dtype=np.float32)
                    bl[0] = x - 0.75*math.cos(theta) - 1.5*math.sin(theta)
                    bl[1] = y - 0.75*math.sin(theta) + 1.5*math.cos(theta)

                    plt.plot([tl[0], tr[0]], [tl[1], tr[1]], color='r')
                    plt.plot([tr[0], br[0]], [tr[1], br[1]], color='r')
                    plt.plot([br[0], bl[0]], [br[1], bl[1]], color='r')
                    plt.plot([bl[0], tl[0]], [bl[1], tl[1]], color='r')


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



def main():
    parser = argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
    parser.add_argument('-g', '--geom', default='point', choices=['point', 'circle', 'rectangle'], \
        help='the geometry of the robot. Choose from "point" (Question 1), "circle" (Question 2), or "rectangle" (Question 3). default: "point"')
    parser.add_argument('--alg', default='rrt', choices=['rrt', 'rrtstar'], \
        help='which path-finding algorithm to use. default: "rrt"')
    parser.add_argument('--iter', default=100, type=int, help='number of iterations to run')
    parser.add_argument('--blind', action='store_true', help='set to disable all graphs. Useful for running in a headless session')
    parser.add_argument('--fast', action='store_true', help='set to disable live animation. (the final results will still be shown in a graph). Useful for doing timing analysis')
    parser.add_argument('--env', default=1, type=int)
    parser.add_argument('--nn', action='store_true')
    parser.add_argument('--sample', default='directed', type=str, choices=['directed', 'normal'])
    parser.add_argument('--get-results', action='store_true')


    args = parser.parse_args()

    show_animation = not args.blind and not args.fast

    print("Starting planning algorithm '%s' with '%s' robot geometry"%(args.alg, args.geom))
    starttime = time.time()

    env_path = 'envs/2d/env{}.pkl'.format(args.env)

    obstacleList = []
    env = pickle.load(open(env_path, 'rb'))
    obstacleList = env

    env_pc_path = 'envs/2d/env{}_pc.pkl'.format(args.env)

    pc = pickle.load(open(env_pc_path, 'rb'))

    start = np.random.uniform(-20, 20, 2)
    start = [-18, 10]
    goal = np.random.uniform(-20, 20, 2)
    goal = [15, 19]
 
    dof = 2

    rrt = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud = pc, obstacleList=obstacleList, dof=dof, alg=args.alg, geom=args.geom, maxIter=args.iter, sample=args.sample)
    rrt2 = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud = pc, obstacleList=obstacleList, dof=dof, alg=args.alg, geom=args.geom, maxIter=args.iter, sample='normal' if args.sample == 'directed' else 'directed')

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
    
    # model.load('entire_model_env_2d_epoch_15000_pc.pt')
    model = torch.load('entire_model_env_2d_epoch_2800.pt', map_location ='cpu')
    model.eval()
    if args.nn:
        
        path = rrt.planning(animation=show_animation, model= model)
        endtime = time.time()
        
        
    else:


        if args.get_results :
            d_count = 0
            d_time = 0
            n_count = 0
            n_time = 0
            d_path_length = 0
            n_path_length = 0
            for it in tqdm(range(1, 10000)):
                env_id = random.randint(0, 9)
                env_path = 'envs/2d/env{}.pkl'.format(env_id)

                obstacleList = []
                env = pickle.load(open(env_path, 'rb'))
                obstacleList = env

                env_pc_path = 'envs/2d/env{}_pc.pkl'.format(env_id)
                pc = pickle.load(open(env_pc_path, 'rb'))


                start = np.random.uniform(-20, 20, 2)
            
                goal = np.random.uniform(-20, 20, 2)

                
            
                rrt = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud = pc, obstacleList=obstacleList, dof=dof, alg=args.alg, geom=args.geom, maxIter=args.iter, sample=args.sample)
                
                rrt2 = RRT(start=start, goal=goal, randArea=[-20, 20], pointcloud = pc, obstacleList=obstacleList, dof=dof, alg=args.alg, geom=args.geom, maxIter=args.iter, sample='normal' if args.sample == 'directed' else 'directed')
                

                starttime = time.time()
                path, firsttime, minTime = rrt.planning1(animation=False, model=model)
                endtime = time.time()
                starttime2 = time.time()
                path2, firsttime2, minTime2 = rrt2.planning1(animation=False, model=model)
                endtime2 = time.time()

                if path is not None:
                    if len(path) == 0:
                        it -= 1
                        continue


                if path is not None:
                    d_count += 1
                    d_time += minTime2 - starttime2
                    d_path_length += len(path)
                if path2 is not None and len(path) != 0:
                    n_count += 1
                    n_time += minTime - starttime
                    n_path_length += len(path2)

                if it % 20 == 0 and it != 0:
                    print('directed success rate: ', d_count/it, flush=True)
                    print('directed average time: ', d_time/d_count, flush=True)
                    print('directed average path length: ', d_path_length/d_count, flush=True)
                    print('normal success rate: ', n_count/it, flush=True)
                    print('normal average time: ', n_time/n_count, flush=True)
                    print('normal average path length: ', n_path_length/n_count, flush=True)
                    
                    
            print('directed success rate: ', d_count/100)
            print('directed average time: ', d_time/d_count)
            print('directed average path length: ', d_path_length/d_count)
            print('normal success rate: ', n_count/100)
            print('normal average time: ', n_time/n_count)
            print('normal average path length: ', n_path_length/n_count)



        starttime = time.time()
        path, firsttime, minTime = rrt.planning1(animation=show_animation, model=model)
        endtime = time.time()
        starttime2 = time.time()
        path2, firsttime2, minTime2 = rrt2.planning1(animation=show_animation, model=model)
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

        if args.nn:
            
            plt.clf()
            starttime = time.time()
            path2 = rrt2.planning1(animation=show_animation)
            endtime = time.time()

            if path2 is None:
                print("FAILED to find a path in %.2fsec"%(endtime - starttime))
            else:
                print("SUCCESS - found path of cost %.5f in %.2fsec"%(RRT.get_path_len(path2), endtime - starttime))
    
            rrt2.draw_graph()
            plt.show()

    


if __name__ == '__main__':
    main()
