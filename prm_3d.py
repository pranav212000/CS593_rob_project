from __future__ import division
from math import sqrt
import argparse
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import time
from typing import List, Dict
import pickle
from tqdm import tqdm
import pickle
import sys
from types import NoneType
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
import math
import os
import random
import sys
from visualizer import visualize
from collision_utils import get_collision_fn
import resource


UR5_JOINT_INDICES = [0, 1, 2]


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


def draw_sphere_marker(position, radius, color):
    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    marker_id = p.createMultiBody(
        basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
    return marker_id

def draw_line_marker(start, end, color):
    # TODO check this function works or not
    vs_id = p.createVisualShape(p.GEOM_LINE, lineWidth=5, lineColor=color)
    marker_id = p.createMultiBody(
        basePosition=start, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
    p.changeVisualShape(marker_id, -1, lineToP2=end)
    return marker_id

def remove_marker(marker_id):
    p.removeBody(marker_id)


def magnitude(v):
    """
    Computes the magnitude of the vector v.
    """
    return math.sqrt(sum([x*x for x in v]))


class Node():
    """
    RRT Node
    """

    def __init__(self, state=None, conf=None, ur5=None):
        self.state = state
        self.conf = conf
        if(conf is not None):
            
            num_joints = p.getNumJoints(ur5)
            link_id = num_joints - 1
            link_state = p.getLinkState(ur5, link_id, computeForwardKinematics=True)
            link_pos = link_state[0]
            self.state = link_pos

        self.cost = 0.0
        self.neighbors = {}

    def add_neighbor(self, node, cost):
        self.neighbors[node] = cost


class PRM():
    """
    Class for PRM
    """

    def __init__(self, obstacleList, randArea, dof=2, expandDis=0.05, maxIter=100, env='2d', collisionCheck3d=None, ur5=None, UR5_JOINT_INDICES=None, env_id=0):
        """
        obstacleList:obstacle Positions [[x,y,width,height],...]
        randArea:Ramdom Samping Area [min,max]
        """

        self.obstacleList = obstacleList
        self.randArea = randArea
        self.dof = dof
        self.expandDis = expandDis
        self.maxIter = maxIter
        self.nodeList = []
        self.minRand = randArea[0]
        self.maxRand = randArea[1]
        self.env = env
        self.collisionCheck3d = collisionCheck3d
        self.ur5 = ur5
        self.UR5_JOINT_INDICES = UR5_JOINT_INDICES
        self.env_id = env_id

    def collisionCheck(self, node):
        """
        Check if the node is in collision.
        """

        if (self.env == '3d'):
            return self.collisionCheck3d(node.conf)

        s = np.zeros(self.dof, dtype=np.float32)
        s[0] = node.state[0]
        s[1] = node.state[1]
        cx = node.state[0]
        cy = node.state[1]
        rx = s[0]
        ry = s[1]

        for (ox, oy, sizex, sizey) in self.obstacleList:
            obs = [ox+sizex/2.0, oy+sizey/2.0]
            obs_size = [sizex, sizey]
            cf = False
            for j in range(self.dof):
                if abs(obs[j] - s[j]) > obs_size[j]/2.0:
                    cf = True
                    break
            if cf == False:
                return False

        return True  # safe'''

    def generateSample(self, iter=100):
        """
        Randomly generates a sample, to be used as a new node.
        This sample may be invalid - if so, call generatesample() again.


        You will need to modify this function for question 3 (if self.geom == 'rectangle')

        returns: random c-space vector
        """

        if (self.env == '2d'):

            while (iter > 0):

                sample = []
                for j in range(0, self.dof):
                    sample.append(random.uniform(self.minRand, self.maxRand))

                rnd = Node(sample)
                if self.collisionCheck(rnd):
                    return rnd
                else:
                    iter = iter - 1

        else:
            rand_state = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                          math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))
            set_joint_positions(self.ur5, UR5_JOINT_INDICES, rand_state)
            while (self.collisionCheck3d(rand_state)):
                rand_state = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                              math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))

            return Node(conf=rand_state, ur5=self.ur5)

    def addNewNode(self, node: Node, newNode: Node, cost: float):

        node.add_neighbor(newNode, cost)
        if node.state != newNode.state:
            newNode.add_neighbor(node, cost)

    def getDistance(self, node1: Node, node2: Node):
        if (self.env == '3d'):
            return np.linalg.norm(np.array(node1.conf) - np.array(node2.conf))
        return np.linalg.norm(np.array(node1.state) - np.array(node2.state))

    def getNearNodes(self, newNode: Node, radius: float, k: int) -> list[Node]:
        nearNodes = []

        for node in self.nodeList:
            if self.getDistance(newNode, node) < radius:
                nearNodes.append(node)
        nearNodes.sort(key=lambda node: self.getDistance(newNode, node))
        return nearNodes[:k]

    def steerTo3d(self, rand_node, nearest_node, step_size=0.05, show_animation=False):

        distance = self.getDistance(rand_node, nearest_node)
        n_steps = round(distance/step_size)
        if n_steps == 0:
            return self.collisionCheck3d(rand_node.conf)
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
                link_pos = link_pos[::len(link_pos)//10]

            link_pos.insert(0, start_pos)
            link_pos.append(end_pos)

            # print(len(link_pos))

            for i in range(len(link_pos)-1):
                p.addUserDebugLine(link_pos[i], link_pos[i+1], [1, 0, 0], 1, 0)

        return (True, distance)

    def steerTo(self, dest, source):
        """
        Charts a route from source to dest, and checks whether the route is collision-free.
        Discretizes the route into small steps, and checks for a collision at each step.

        dest: destination node
        source: source node

        returns: (success, cost) tuple
            - success is True if the route is collision free; False otherwise.
            - cost is the distance from source to dest, if the route is collision free; or None otherwise.
        """

        # newNode = copy.deepcopy(source)
        newNode = copy.copy(source)

        DISCRETIZATION_STEP = self.expandDis

        dists = np.zeros(self.dof, dtype=np.float32)

        for j in range(0, self.dof):
            dists[j] = dest.state[j] - source.state[j]

        distTotal = magnitude(dists)

        if distTotal > 0:
            incrementTotal = distTotal/DISCRETIZATION_STEP
            for j in range(0, self.dof):
                dists[j] = dists[j]/incrementTotal

            numSegments = int(math.floor(incrementTotal))+1

            stateCurr = np.zeros(self.dof, dtype=np.float32)

            for j in range(0, self.dof):
                stateCurr[j] = newNode.state[j]

            stateCurr = Node(stateCurr)

            for i in range(0, numSegments):

                if not self.collisionCheck(stateCurr):
                    return (False, None)

                for j in range(0, self.dof):
                    stateCurr.state[j] += dists[j]

            if not self.collisionCheck(dest):
                return (False, None)

            return (True, distTotal)
        else:
            return (False, None)

    def getEndEffectorPos(self):
        num_joints = p.getNumJoints(self.ur5)
        link_id = num_joints - 1
        link_state = p.getLinkState(
            self.ur5, link_id, computeForwardKinematics=True)
        link_pos = link_state[0]
        link_ori = link_state[1]

        return link_pos

    def planning3d(self, n=1000, radius=10, k=30, show_animation=False, save_every=20):

        for i in tqdm(range(n)):
            newNode = self.generateSample()

            gamma = 500
            radius = gamma * ((math.log(i+1)/math.sqrt(i+1)) ** (1/self.dof))
            # print(newNode)

            # print(newNode.state)
            link_pos = self.getEndEffectorPos()
            # print(link_pos)
            draw_sphere_marker(
                        newNode.state, radius=0.01, color=[1, 0, 0, 1])

            nearNodes = self.getNearNodes(
                newNode, radius=radius, k=len(self.nodeList))

            for node in nearNodes:

                isCollisionFree, cost = self.steerTo3d(
                    node, newNode, show_animation=show_animation)

                if isCollisionFree:
                    if node not in newNode.neighbors.keys():
                        # p.addUserDebugLine(newNode.state, node.state, [0, 0, 1], 1, 0)
                        newNode.neighbors[node] = cost
                        node.neighbors[newNode] = cost

            # print("Neighbors: ", len(newNode.neighbors))

            self.nodeList.append(newNode)

            # if i % 10 == 0:
            #     print("Iteration: ", i, flush=True)

            if i % save_every == 0:
                if self.env == '2d':

                    if not os.path.exists("2d"):
                        os.mkdir("2d")

                    filename = "2d/graph_2d_env_{}_nodes_{}".format(
                        self.env_id, i)
                    self.drawGraph(newNode, save=True, epoch=i,
                                   filename=filename + ".png")
                    self.saveGraph(filename + ".pkl")

                    print("Saved Graph: ", i)
                else:
                    if not os.path.exists("3d"):
                        os.mkdir("3d")

                    filename = "3d/graph_3d_env_{}_nodes_{}".format(
                        self.env_id, i)
                    self.saveGraph(filename + ".pkl")

                    width = 640
                    height = 480

                    view_angles = [
                        [58, -42, 0],
                        [0, -90, 0],
                        [0, -22, 0],
                        [-90, -22, 0]
                    ]

                    for j, view_angle in enumerate(view_angles):
                        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[
                                                                          0, 0, 0], distance=2, yaw=view_angle[0], pitch=view_angle[1], roll=view_angle[2], upAxisIndex=2)
                        projection_matrix = p.computeProjectionMatrixFOV(
                            fov=60, aspect=float(width)/height, nearVal=0.01, farVal=100)
                        img_arr = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix,
                                                   projectionMatrix=projection_matrix, shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)

                        
                        rgb_arr = np.array(img_arr[2])
                        rgb_arr = rgb_arr[:, :, :3]

                        plt.imsave(filename + "_{}.png".format(j), rgb_arr)




                    # view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[
                    #                                                   0, 0, 0], distance=2, yaw=58, pitch=-42, roll=0, upAxisIndex=2)
                    # projection_matrix = p.computeProjectionMatrixFOV(
                    #     fov=60, aspect=float(width)/height, nearVal=0.01, farVal=100)
                    # img_arr = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix,
                    #                            projectionMatrix=projection_matrix, shadow=True, renderer=p.ER_BULLET_HARDWARE_OPENGL)

                    
                    # rgb_arr = np.array(img_arr[2])
                    # rgb_arr = rgb_arr[:, :, :3]  # remove alpha channel
                    

                    # plt.imsave(filename + ".png", rgb_arr)





    def planning(self, n=1000, radius=10, k=30, show_animation=False, save_every=20):

        for i in tqdm(range(n)):
            newNode=self.generateSample()

            # TODO change for PRM*
            # radius = 5.0

            # gamma = 2 * (1 + 1/self.dof)**(1/self.dof)
            gamma=100
            radius=gamma * ((math.log(i+1)/math.sqrt(i+1)) ** (1/self.dof))

            nearNodes=self.getNearNodes(
                newNode, radius=radius, k=len(self.nodeList))

            for node in nearNodes:
                isCollisionFree, cost=self.steerTo(node, newNode)


                if isCollisionFree:
                    if node not in newNode.neighbors.keys():
                        newNode.neighbors[node]=cost
                        node.neighbors[newNode]=cost


            self.nodeList.append(newNode)

            # if i % 10 == 0:
            #     print("Iteration: ", i, flush=True)

            if i % save_every == 0:
                if not os.path.exists("2d"):
                    os.mkdir("2d")
                filename="2d/graph_env_{}_nodes_{}".format(self.env_id, i)
                self.drawGraph(newNode, save=True, epoch=i,
                               filename=filename + ".png", show_animation=show_animation)
                self.saveGraph(filename + ".pkl")
                print("Saved Graph: ", i)


    def saveGraph(self, filename):

        with open(filename, 'wb') as f:
            pickle.dump(self.nodeList, f)

    def getAdjecencyMatrix(self):
        adj=np.zeros((len(self.nodeList), len(self.nodeList)))
        for i in range(len(self.nodeList)):
            for j in range(len(self.nodeList)):
                if self.nodeList[i] in self.nodeList[j].neighbors.keys():
                    adj[i][j]=self.nodeList[j].neighbors[self.nodeList[i]]
        return adj

    def getShortestPath(self, start, goal):
        originalStart=copy.copy(start)
        originalGoal=copy.copy(goal)

        nearNodesToStart=self.getNearNodes(start, radius=50, k=5)
        nearNodesToGoal=self.getNearNodes(goal, radius=50, k=5)

        finalPath=None
        minCost=float('inf')

        for start in nearNodesToStart:
            isCollisionFreeStart, startCost=self.steerTo(
                start, originalStart)

            if isCollisionFreeStart:
                print("Found collision free start")
                for goal in nearNodesToGoal:
                    isCollisionFreeGoal, goalCost=self.steerTo(
                        goal, originalGoal)
                    if isCollisionFreeGoal:
                        path, cost=self.dijkstra(start, goal)
                        if cost + startCost + goalCost < minCost:
                            minCost=cost + startCost + goalCost
                            path.insert(0, originalStart)
                            path.append(originalGoal)
                            finalPath=path
                            # print("Found path with cost: ", minCost)

        return finalPath, minCost

    def dijkstra(self, start, goal):
        """
        Finds the shortest path between start and goal.

        start: start node
        goal: goal node

        returns: list of nodes representing the shortest path from start to goal.
        """


        sptSet=set()
        dist={}
        prev={}

        for node in self.nodeList:
            dist[node]=float('inf')
            prev[node]=None

        dist[start]=0

        while len(sptSet) != len(self.nodeList):
            u=min(dist, key=lambda node: dist[node]
                    if node not in sptSet else float('inf'))
            sptSet.add(u)

            for v in u.neighbors:
                if v not in sptSet:
                    alt=dist[u] + u.neighbors[v]
                    if alt < dist[v]:
                        dist[v]=alt
                        prev[v]=u

        path=[]
        curr=goal

        while curr is not None:
            path.append(curr)
            curr=prev[curr]

        pathcost=0
        for i in range(len(path)-1):
            pathcost += path[i].neighbors[path[i+1]]

        path.reverse()

        return path, pathcost

    def loadGraph(self, filename):
        with open(filename, 'rb') as f:
            self.nodeList=pickle.load(f)

    def drawGraph(self, rnd=None, path=None,  save=False, epoch=0, filename=None, show_animation=True):
        """
        Draws the state space, with the tree, obstacles, and shortest path (if found). Useful for visualization.

        You will need to modify this for question 2 (if self.geom == 'circle') and question 3 (if self.geom == 'rectangle')
        """
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for (ox, oy, sizex, sizey) in self.obstacleList:
            rect=mpatches.Rectangle(
                (ox, oy), sizex, sizey, fill=True, color="purple", linewidth=0.01)
            plt.gca().add_patch(rect)

        visited=[]
        i=0
        for node in self.nodeList:
            visited.append(node)

            plt.scatter(node.state[0], node.state[1], color='b', zorder=1, s=3)

            # if epoch > 400 == 0:
            if not show_animation or epoch > 300:
                continue

            for node2 in node.neighbors:
                if node2 not in visited:
                    i += 1
                    plt.plot([node.state[0], node2.state[0]], [
                             node.state[1], node2.state[1]], color='r', zorder=0, linewidth=0.3)
                    # if i % 200 == 0:
                    #     print('Still Plotting, done {}'. format(i))

        if path is not None:
            for i, node in enumerate(path):
                plt.scatter(node.state[0], node.state[1],
                            color='g', zorder=1, s=10)
                if i != 0:
                    plt.plot([path[i-1].state[0], node.state[0]], [path[i-1].state[1],
                             node.state[1]], color='g', zorder=0, linewidth=1)

        if rnd is not None:
            plt.plot(rnd.state[0], rnd.state[1], "^k")

        if save == True:
            if filename is None:
                plt.savefig("graph_{}.png".format(epoch))
            else:
                plt.savefig(filename)

        # plt.plot(self.start.state[0], self.start.state[1], "xr")
        # plt.plot(self.end.state[0], self.end.state[1], "xr")
        plt.axis("equal")
        plt.axis([-20, 20, -20, 20])
        plt.grid(True)
        plt.pause(0.01)


def main():

    print(resource.getrlimit(resource.RLIMIT_STACK))
    print(sys.getrecursionlimit())

    max_rec=0x100000

    # May segfault without this line. 0x100 is a guess at the size of each stack frame.
    resource.setrlimit(resource.RLIMIT_STACK, [
       0x100 * max_rec, resource.RLIM_INFINITY])
    sys.setrecursionlimit(max_rec)

    parser=argparse.ArgumentParser(description='CS 593-ROB - Assignment 1')
    parser.add_argument('-g', '--geom', default='point', choices=['point', 'circle', 'rectangle'],
                        help='the geometry of the robot. Choose from "point" (Question 1), "circle" (Question 2), or "rectangle" (Question 3). default: "point"')
    parser.add_argument('--alg', default='rrt', choices=['rrt', 'rrtstar'],
                        help='which path-finding algorithm to use. default: "rrt"')
    parser.add_argument('--iter', default=100, type=int,
                        help='number of iterations to run')
    parser.add_argument('--blind', action='store_true',
                        help='set to disable all nodeLists. Useful for running in a headless session')
    parser.add_argument('--fast', action='store_true',
                        help='set to disable live animation. (the final results will still be shown in a nodeList). Useful for doing timing analysis')
    parser.add_argument('--env', default='2d', choices=['2d', '3d'],
                        help='the environment to run in. Choose from "2d" or "3d". default: "2d"')
    parser.add_argument('--env-id', default=0, type=int,
                        choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--show-animation', action='store_true',
                        help='set to show edges in the graph. Useful for debugging')
    parser.add_argument('--save-every', default=20, type=int,
                        help='set to save the graph every n iterations. Useful for debugging')

    args=parser.parse_args()

    show_animation=not args.blind and not args.fast

    print("Starting planning algorithm '%s' with '%s' robot geometry" %
          (args.alg, args.geom))
    starttime=time.time()

    obstacleList=[
        (-15, 0, 15.0, 5.0),
        (15, -10, 5.0, 10.0),
        (-10, 8, 5.0, 15.0),
        (3, 15, 10.0, 5.0),
        (-10, -10, 10.0, 5.0),
        (5, -5, 5.0, 5.0),
        (10, 10, 5.0, 5.0),
    ]


    start=[-10, -17]
    goal=[10, 10]

    dof=2

    if args.env == '3d':
        dof=3
         # TODO uncomment following necessary lines for 3D gui window (doesm't work in ssh)
        physicsClient = p.connect(p.DIRECT)
        # physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setGravity(0, 0, -9.8)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
        # p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
        # p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000,
        #                              cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))

        # load objects
        plane=p.loadURDF("plane.urdf")
        ur5=p.loadURDF('assets/ur5/ur5.urdf',
                         basePosition=[0, 0, 0.02], useFixedBase=True)
        obstacle1=p.loadURDF('assets/block.urdf',
                               basePosition=[1/4, 0, 1/2],
                               useFixedBase=True)
        obstacle2=p.loadURDF('assets/block.urdf',
                               basePosition=[2/4, 0, 2/3],
                               useFixedBase=True)
        obstacle3=p.loadURDF('assets/block.urdf',
                                 basePosition=[-3/4, 0, 1/2],
                                    useFixedBase=True)
        obstacle4=p.loadURDF('assets/block.urdf',
                                    basePosition=[0, 3/5, 1/3],
                                    useFixedBase=True)
        obstacle5=p.loadURDF('assets/block.urdf',
                                    basePosition=[0, -1/8, 2/3],
                                    useFixedBase=True)
        obstacle6=p.loadURDF('assets/block.urdf',
                                 basePosition=[-1/4, 1/3, 3/5],
                                    useFixedBase=True)
        obstacle7=p.loadURDF('assets/block.urdf',
                               basePosition=[2/4, -1/4, 2/3],
                               useFixedBase=True)

        env1=[plane, obstacle1, obstacle2, obstacle3, obstacle4,
            obstacle5, obstacle6, obstacle7, obstacle6]

        envs=[env1]

        obstacles=envs[args.env_id]

        collisionCheck3d=get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                            attachments=[], self_collisions=True,
                                            disabled_collisions=set())

        prm=PRM(obstacleList=obstacles, randArea=[-20, 20], dof=dof, env='3d',
                  collisionCheck3d=collisionCheck3d, ur5=ur5, UR5_JOINT_INDICES=UR5_JOINT_INDICES)
        prm.planning3d(4000, radius=10, k=10,
                       show_animation=args.show_animation, save_every=args.save_every)

    else:
        prm=PRM(obstacleList=obstacleList,
                randArea=[-20, 20], dof=dof, env='2d')

        prm.planning(4000, radius=10, k=10,
                     show_animation=args.show_animation, save_every=args.save_every)

    endtime=time.time()

    # if path is None:
    #     print("FAILED to find a path in %.2fsec" % (endtime - starttime))
    # else:
    #     print("SUCCESS - found path of cost %.5f in %.2fsec" %
    #           (RRT.get_path_len(path), endtime - starttime))
    # Draw final path
    # if not args.blind:
    #     rrt.draw_nodeList()
    #     plt.show()


if __name__ == '__main__':
    main()
