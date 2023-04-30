from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import argparse
import math
import os
import copy
import sys


UR5_JOINT_INDICES = [0, 1, 2]


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)

   
def draw_sphere_marker(position, radius, color):
   vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
   marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
   return marker_id


def remove_marker(marker_id):
   p.removeBody(marker_id)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--birrt', action='store_true', default=False)
    parser.add_argument('--smoothing', action='store_true', default=False)
    args = parser.parse_args()
    return args

#your implementation starts here
#refer to the handout about what the these functions do and their return type
###############################################################################
class RRT_Node:
    def __init__(self, conf):
        self.conf = conf
        self.cost = 0.0
        self.parent = None
        self.node_list = []
        # pass

    def set_parent(self, parent):
        self.parent = parent
        # pass

    def add_child(self, child):
        self.node_list.append(child)
        # pass

    def remove_child(self, child):
        self.node_list.remove(child)
        # pass

def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def sample_conf():
    conf_x = np.random.uniform(-2*math.pi, 2*math.pi)
    conf_y = np.random.uniform(-2*math.pi, 2*math.pi)
    conf_z = np.random.uniform(-1*math.pi, math.pi)
    
    while(collision_fn((conf_x, conf_y, conf_z))):
        conf_x = np.random.uniform(-2*math.pi, 2*math.pi)
        conf_y = np.random.uniform(-2*math.pi, 2*math.pi)
        conf_z = np.random.uniform(-1*math.pi, math.pi)

    if distance((conf_x, conf_y, conf_z), goal_conf) < 0.3:
        return goal_conf, True
        
    return (conf_x, conf_y, conf_z), False
   
def find_nearest(rand_node, node_list):
    min_dist = sys.maxsize
    nearest = None
    for node in node_list:
        dist = distance(rand_node.conf, node.conf)
        if dist  < min_dist:
            min_dist = dist
            nearest = node
    return nearest
        
def steer_to(rand_node, nearest_node):
    dist = distance(rand_node.conf, nearest_node.conf)
    step_size = 0.05
    steps = int(dist/step_size)
    if(steps==0):
        return False, None
    
    unit_step = tuple(map(lambda i, j: (i - j)/steps, rand_node.conf, nearest_node.conf))
    start = nearest_node.conf
    for i in range(steps):
        start = tuple(map(lambda i, j: i + j, start, unit_step))
        if collision_fn(start):
            return False, None
    return True, dist


def steer_to_until(rand_node, nearest_node):
    dist = distance(rand_node.conf, nearest_node.conf)
    step_size = 0.05
    steps = int(dist/step_size)
    if(steps==0):
        return nearest_node

    unit_step = tuple(map(lambda i, j: (i - j)/steps, rand_node.conf, nearest_node.conf))
    prev = nearest_node
    start = nearest_node.conf
    for i in range(steps):
        start = tuple(map(lambda i, j: i + j, start, unit_step))
        if collision_fn(start):
            return prev
        else:
            prev = RRT_Node((start[0], start[1], start[2]))

    return rand_node

def find_near_nodes(newNode, nodelist):
    """
        Finds all nodes in the tree that are "near" newNode.
        See the assignment handout for the equation defining the cutoff point (what it means to be "near" newNode)

        newNode: the node to be inserted.

        Returns: a list of indices of nearby nodes.
    """
    GAMMA = 50

    n_nodes = len(nodelist)
    
    nearest = []

    radius = GAMMA * (pow((np.log(n_nodes)/n_nodes), (1/3)))

    for i in range(n_nodes):
        dist = distance(newNode.conf, nodelist[i].conf)
        if dist <= radius:
            nearest.append(nodelist[i])
    return nearest

def to_be_added(pointcloud, model):
    pass

def choose_parent(newNode, near_nodes):
    minCost = float('inf')
    # print(newNode.conf)
    minIndex = None
    for i in near_nodes:
        valid, cost = steer_to(newNode, i)
        if valid:
            #find cost of reaching to i from source
            cost += i.cost
            if cost < minCost:
                minCost = cost
                minIndex = i
    
    return minIndex

def rewire(newNode, near_nodes):
    for i in near_nodes:
        valid, cost = steer_to(newNode, i)
        if valid:
            #find cost of reaching to i from source
            cost += newNode.cost
            if cost < i.cost:
                i.parent = newNode
                i.cost = cost

def is_near_goal(node):
    if distance(node.conf, goal_conf) < 0.3:
        return True
    return False


def RRTstar(sample):
    #################################################
    # TODO your code to implement the rrtstar algorithm
    #################################################
    tree = []
    path = []
    goal_node = None
    start_node = RRT_Node(start_conf)
    tree.append(start_node)
    while(1):
        conf, is_goal = sample_conf()
        
        q_rand = RRT_Node(conf)
        q_nearest = find_nearest(q_rand, tree)

        valid, cost = steer_to(q_rand, q_nearest)

        if valid:
            newNode = copy.deepcopy(q_rand)
            newNode.parent = q_nearest
            # newNode.cost = rnd_cost + self.nodeList[nind].cost
            q_rand.set_parent(q_nearest)
            q_nearest.add_child(q_rand)
            tree.append(q_rand)

            near_nodes = find_near_nodes(newNode, tree) # you'll implement this method
            to_add = False

            if sample != 'normal':
                to_add = to_be_added(newNode, near_nodes, point_cloud=None, model=None)

                if to_add:
                    newParent = choose_parent(newNode, near_nodes) # you'll implement this method
                else:
                    # newParent = None
                    continue
            else:
                newParent = choose_parent(newNode, near_nodes) # you'll implement this method

        else:
            newParent = None

        if newParent:
            newNode.parent = newParent
            newNode.cost = newNode.parent.cost + distance(newNode.conf, newNode.parent.conf)
            newNode.parent.add_child(newNode)
            tree.append(newNode)
            # Rewire
            for i in near_nodes:
                if i == newParent:
                    continue
                valid, cost = steer_to(i, newNode)
                if valid:
                    #find cost of reaching to i from source
                    cost += newNode.cost
                    if cost < i.cost:
                        i.parent.remove_child(i)
                        i.parent = newNode
                        i.cost = cost
                        newNode.add_child(i)

        if is_near_goal(q_rand):
            goal_node = q_rand
            break
    
    cur = goal_node
    while cur!=None:
        path.append(cur.conf)
        cur = cur.parent

    if len(path) == 0:
        return None
    
    path.reverse()
    # print("Length of RRT path", len(path))
    return path

def RRT():
    ###############################################
    # TODO your code to implement the rrt algorithm
    ###############################################
    tree = []
    path = []
    goal_node = None
    start_node = RRT_Node(start_conf)
    tree.append(start_node)
    while(1):
        conf, is_goal = sample_conf()
        
        q_rand = RRT_Node(conf)
        q_nearest = find_nearest(q_rand, tree)

        if steer_to(q_rand, q_nearest):
            q_rand.set_parent(q_nearest)
            q_nearest.add_child(q_rand)
            tree.append(q_rand)

            if is_goal:
                goal_node = q_rand
                # print("Goal reached at iteration", i)
                break
    
    cur = goal_node
    while cur!=None:
        path.append(cur.conf)
        cur = cur.parent

    if len(path) == 0:
        return None
    
    path.reverse()
    # print("Length of RRT path", len(path))
    return path

def BiRRT():
    #################################################
    # TODO your code to implement the birrt algorithm
    #################################################
    def Connect(q_s, treeB):
        nearest_other = find_nearest(q_s, treeB)
        if steer_to(nearest_other, q_s):
            return True, nearest_other
        return False, None

    def getPath(end):
        path = []
        cur = end
        while cur!=None:
            path.append(cur.conf)
            cur = cur.parent       
        return path

    treeA = []
    treeB = []
    treeA.append(RRT_Node(start_conf))
    treeB.append(RRT_Node(goal_conf))


    while(1):
        conf, is_goal = sample_conf()

        q_rand = RRT_Node(conf)
        q_nearest = find_nearest(q_rand, treeA)
        q_s = steer_to_until(q_rand, q_nearest)
        if q_s != q_nearest:
            q_s.set_parent(q_nearest)
            q_nearest.add_child(q_s)
            treeA.append(q_s)

            connected, nearest_B = Connect(q_s, treeB)
            if connected:
                path1 = getPath(q_s)
                path2 = getPath(nearest_B)
                if path1[-1]==start_conf:
                    path1.reverse()
                    path = path1 + path2
                    # print(path)
                    # print("Goal reached after iterations", i)
                    # print("Length of BiRRT path", len(path))
                    return path
                else :
                    path2.reverse()
                    path = path2 + path1
                    # print("Goal reached after iterations", i)
                    # print("Length of BiRRT path", len(path))
                    return path
        
        temp = treeA
        treeA = treeB
        treeB = temp



def BiRRT_smoothing():
    ################################################################
    # TODO your code to implement the birrt algorithm with smoothing
    ################################################################
    path = BiRRT()
    for i in range(100):
        index1 = random.randint(0,len(path)-1)
        index2 = random.randint(0,len(path)-1)
        while abs(index1-index2) <= 1:
            index1 = random.randint(0,len(path)-1)
            index2 = random.randint(0,len(path)-1)

        if steer_to(RRT_Node(path[index1]), RRT_Node(path[index2])):
            path = path[:min(index1, index2)+1] + path[max(index1, index2):]

    # print("Length of BiRRT Path after smoothing", len(path))  
    return path
###############################################################################
#your implementation ends here

if __name__ == "__main__":
    args = get_args()

    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)
    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
    obstacles = [plane, obstacle1, obstacle2]

    # start and goal
    start_conf = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    goal_marker = draw_sphere_marker(position=start_position, radius=0.02, color=[1, 0, 1, 1])
    set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)

    		# place holder to save the solution path
    path_conf = None

    # get the collision checking function
    from collision_utils import get_collision_fn
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    if args.birrt:
        if args.smoothing:
            # using birrt with smoothing
            path_conf = BiRRT_smoothing()
        else:
            # using birrt without smoothing
            path_conf = BiRRT()
    else:
        # using rrt
        path_conf = RRTstar("normal")

    if path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:
        # execute the path
        while True:
            for q in path_conf:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(0.5)
