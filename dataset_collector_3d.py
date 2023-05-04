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
import pybullet as p
import pybullet_data
import numpy as np
import time
import argparse
import math
import os
import random
import sys
from collision_utils import get_collision_fn
import datetime
from prm_3d import PRM, Node, set_joint_positions

UR5_JOINT_INDICES = [0, 1, 2]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=int, default=0)
    parser.add_argument('--env', type=str, default='3d')
    parser.add_argument('--num-nodes', type=int, default=2500)
    parser.add_argument('--num-iter', type=int, default=5000)
    parser.add_argument('--save-every', type=int, default=500)
    args = parser.parse_args()

    print(args)

    dof = 3

    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)

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

    prm = PRM(obstacleList=obstacles, randArea=[-20, 20], dof=dof, env='3d', env_id=args.env_id,
              collisionCheck3d=collisionCheck3d, ur5=ur5, UR5_JOINT_INDICES=UR5_JOINT_INDICES)

    
    with open('{}/{}/graph_{}_env_{}_nodes_{}.pkl'.format(args.env, args.env_id, args.env, args.env_id, args.num_nodes), 'rb') as f:
        nodes = pickle.load(f)

    prm.nodeList = nodes

    dataset = []
    for iter in tqdm(range(args.num_iter)):

        start = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                          math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))
        set_joint_positions(ur5, UR5_JOINT_INDICES, start)
        while (prm.collisionCheck(Node(conf=start, ur5=ur5))):
            # print(start)
            start = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                            math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))

        goal = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                          math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))
            
        while (prm.collisionCheck(Node(conf=goal, ur5=ur5))):
            goal = (np.random.uniform(-2*math.pi, 2*math.pi), np.random.uniform(-2 *
                            math.pi, 2*math.pi), np.random.uniform(-math.pi, math.pi))

        
        path, cost, cost_to_goal = prm.getShortestPath(
            start=Node(conf=start, ur5=ur5), goal=Node(conf=goal, ur5=ur5))

        if path is None:
            continue
        

        data = np.array([])
        start = np.array(start)
        goal = np.array(goal)

        for i in range(len(path)):

            state = np.array(path[i].conf)
            dataset.append(np.array(
                [start[0], start[1], start[2], goal[0], goal[1], goal[2], state[0], state[1], state[2], cost_to_goal[i]]))

        if iter % args.save_every == 0:
            # print('iter', iter)
            
            with open('3d/{}/dataset.pkl'.format(args.env_id), 'wb') as f:
                pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
