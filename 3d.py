from __future__ import division
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
    parser.add_argument('--birrt2', action='store_true', default=False)
    args = parser.parse_args()
    return args

#your implementation starts here
#refer to the handout about what the these functions do and their return type
###############################################################################
class RRT_Node:
    def __init__(self, conf, position = [0,0,0]):
        self.parent = None
        self.children = []
        self.conf = conf
        self.position = position


    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)


def sample_conf(goal_conf):
    # random configuration
    rand_conf = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))

    while(collision_fn(rand_conf)):
        rand_conf = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))
    is_goal = False
   
    epsilon = 0.3

    if compare_conf(rand_conf, goal_conf, epsilon):
        is_goal = True
        rand_conf = goal_conf
        

    return rand_conf, is_goal
    

def get_distance(conf1, conf2):
    # print(conf1)
    # print(conf2)
    # print(type(conf1))
    # print(type(conf2))
    # conf1 = np.array(conf1)
    conf2 = np.array(conf2)
    return math.sqrt((conf1[0] - conf2[0])**2 + (conf1[1] - conf2[1])**2 + (conf1[2] - conf2[2])**2)
   
def find_nearest(rand_node, node_list) -> RRT_Node:
    
    distance = float('inf')

    # distance between points
    for node in node_list:
        # print(rand_node.conf)
        # print(node.conf)
        if get_distance(rand_node.conf, node.conf) < distance:
            distance = get_distance(rand_node.conf, node.conf)
            nearest_node = node

    return nearest_node



        



def steer_to(rand_node, nearest_node, step_size=0.05):

    distance = get_distance(rand_node.conf, nearest_node.conf)
    n_steps = round(distance/step_size)
    if n_steps == 0:
        return collision_fn(rand_node.conf)
    unit_step = (np.array(rand_node.conf) - np.array(nearest_node.conf)) / n_steps
    start = np.array(nearest_node.conf)
    for i in range(n_steps):
        # print(i)
        start = start + unit_step
        start = (start[0], start[1], start[2])
        if collision_fn(start):
            # print('COLLIDED')
            return False

    return True

    

def steer_to_until(rand_node, nearest_node, step_size=0.05):
    
    distance = get_distance(rand_node.conf, nearest_node.conf)
    n_steps = int(distance/0.05)
    if n_steps == 0:
        if collision_fn(rand_node.conf):
            return None
        return rand_node
    unit_step = (np.array(rand_node.conf) - np.array(nearest_node.conf)) / n_steps
    prev_q = nearest_node
    start = np.array(nearest_node.conf)
    for i in range(n_steps):
        # print(i)
        
        start = start + unit_step
        # print('checking: ', start)
        start = (start[0], start[1], start[2])
        if collision_fn(start):
            if prev_q.conf == nearest_node.conf:
                return None
            return prev_q
        else :
            prev_q = RRT_Node((start[0], start[1], start[2]))
    
    return rand_node


def get_path(goal_node):
    path = []
    i = 0

    # while goal node is not None
    while goal_node != None:
        # print('depth: ', i)
    
        path.append(goal_node.conf)
        # print('parent: ', goal_node.conf)
        goal_node = goal_node.parent
        
        # if(i > 50):
        #     break
        i += 1
    
    path.reverse()
    return path


def RRT():
    ###############################################
    # TODO your code to implement the rrt algorithm
    ###############################################

    node_list = []
    start_node = RRT_Node(start_conf)
    goal_node = RRT_Node(goal_conf)
    node_list.append(start_node)
    step_size = 0.05

# CHANGE MAX ITERATIONS AS PER REQUIREMENT
    while True:
        # if(i % 100 == 0):
        #     print(i)
        rand_conf, is_goal = sample_conf(goal_conf)
        rand_node = RRT_Node(rand_conf)
        nearest_node = find_nearest(rand_node, node_list)
        if steer_to(rand_node, nearest_node, step_size):
            rand_node.set_parent(nearest_node)
            nearest_node.add_child(rand_node)
            node_list.append(rand_node)
        
            if is_goal:
                break

    path = get_path(rand_node)
    
    # print('path', path)

    if path == []:
        return None

    return path


def compare_conf(conf1, conf2, epsilon=0.3):
    return abs(conf1[0] - conf2[0]) <= epsilon and abs(conf1[1] - conf2[1]) <= epsilon and abs(conf1[2] - conf2[2]) <= epsilon


def connect(node, node_list):
    q_nearest = find_nearest(node, node_list)
    if steer_to(node, q_nearest):
        
        return True, q_nearest
    return False, None


def swap(list1, list2):
    temp = list1
    list1 = list2
    list2 = temp
    return list1, list2

def BiRRT():

    current_tree = [RRT_Node(start_conf)]
    other_tree = [RRT_Node(goal_conf)]

    while True:
        # if(i % 1000 == 0):
        #     print(i)
        q_rand_conf, _ = sample_conf(goal_conf)
        q_rand = RRT_Node(q_rand_conf)

        q_nearest = find_nearest(q_rand, current_tree)
        q_new = steer_to_until(q_rand, q_nearest)

        if q_new != None :
            # if q_new == q_nearest:
            #     continue
            q_new.set_parent(q_nearest)
            q_nearest.add_child(q_new)
            current_tree.append(q_new)

            is_connect, q_near = connect(q_new, other_tree)
            if is_connect:
                path1 = get_path(q_new)
                path2 = get_path(q_near)
                if path1[-1] == start_conf:
                    path1.reverse()
                    path = path1 + path2
                elif path2[-1] == start_conf:
                    path2.reverse()
                    path = path2 + path1

                
                path = path1 + path2
                
                return path
        else :
            # print('q_new is None')
            continue
        swap(current_tree, other_tree)

    return None




def BiRRT_smoothing():
    ################################################################
    # TODO your code to implement the birrt algorithm with smoothing
    ################################################################

    path = BiRRT()

    for i in range(100):

        # randomly select two non adjecent indices
        rand_index1 = random.randint(0, len(path)-1)
        rand_index2 = random.randint(0, len(path)-1)
        while abs(rand_index1 - rand_index2) <= 1:
            # print('in while loop')
            # if len(path) == 2:
            #     return path
            rand_index1 = random.randint(0, len(path)-1)
            rand_index2 = random.randint(0, len(path)-1)

        
        if steer_to(RRT_Node(path[rand_index1]), RRT_Node(path[rand_index2])):
            path = path[:min(rand_index1, rand_index2)+1] + path[max(rand_index1, rand_index2):]
            

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
    # obstacle3 = p.loadURDF('assets/block.urdf',
    #                        basePosition=[2/4, 0, 1/6],
    #                        useFixedBase=True)
    
    
    # obstacles = [plane, obstacle1, obstacle2, obstacle3]
    obstacles = [plane, obstacle1, obstacle2]

    # start and goal
    start_conf = (-0.813358794499552, -0.37120422397572495, -0.754454729356351)
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443)
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])
    # draw green sphere for start position
    start_marker = draw_sphere_marker(position=start_position, radius=0.02, color=[0, 1, 0, 1])
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
            path_conf = BiRRT()
        

    else:
        # using rrt
        path_conf = RRT()

    if path_conf is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:
        # execute the path
        while True:
            for q in path_conf:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(0.5)
