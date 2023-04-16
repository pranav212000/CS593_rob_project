# read environment files for 2d from env folder

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import pickle
import os
import sys


args = sys.argv

env_type = args[1]
env_folder_path = args[2]
num_env = int(args[3])
num_points = int(args[4])


def create_point_clouds_2d():
        # for all environment files in the env_type folder of env_folder_path
    for i in range(num_env):
        env_file = 'env' + str(i) + '.pkl'
        # read environment
        point_cloud = []
        env = pickle.load(open(os.path.join(env_folder_path, env_type, env_file), 'rb'))
        # display environment
        # for all points in the environment
        for point in env:
            left_bottom_x = point[0]
            left_bottom_y = point[1]
            size_x = point[2]
            size_y = point[3]
            # uniformly sample 100 points in the environment
            for i in range(num_points):
                x = np.random.uniform(left_bottom_x, left_bottom_x + size_x)
                y = np.random.uniform(left_bottom_y, left_bottom_y + size_y)
                point_cloud.append([x, y])
        # convert to numpy array
        point_cloud = np.array(point_cloud)
        # save point cloud
        pickle.dump(point_cloud, open(os.path.join(env_folder_path, env_type, env_file[:-4] + '_pc.pkl'), 'wb'))


def create_point_clouds_3d():
        # for all environment files in the env_type folder of env_folder_path
    for i in range(num_env):
        env_file = 'env' + str(i) + '.pkl'
        # read environment
        point_cloud = []
        env = pickle.load(open(os.path.join(env_folder_path, env_type, env_file), 'rb'))
        # display environment
        # for all points in the environment
        for point in env:
            center_x = point[0]
            center_y = point[1]
            center_z = point[2]
            size = 0.2
            # uniformly sample 100 points in the environment
            for i in range(num_points):
                x = np.random.uniform(center_x - size, center_x + size)
                y = np.random.uniform(center_y - size, center_y + size)
                z = np.random.uniform(center_z - size, center_z + size)
                point_cloud.append([x, y, z])

        # convert to numpy array
        point_cloud = np.array(point_cloud)
        # save point cloud
        pickle.dump(point_cloud, open(os.path.join(env_folder_path, env_type, env_file[:-4] + '_pc.pkl'), 'wb'))


def plot_point_clouds_2d():
    # for all environment files in the env_type folder of env_folder_path
    for i in range(num_env):
        env_file = 'env' + str(i) + '_pc.pkl'
        # read environment
        point_cloud = pickle.load(open(os.path.join(env_folder_path, env_type, env_file), 'rb'))
        ax = Axes3D(fig) 
        # display environment
        plt.scatter(point_cloud[:, 0], point_cloud[:, 1])
        # save figure in env folder in type folder
        plt.savefig(os.path.join(env_folder_path, env_type, env_file[:-4] + '.png'))
        plt.show()
        # clear plot
        plt.clf()

def plot_point_clouds_3d():
    # for all environment files in the env_type folder of env_folder_path
    for i in range(num_env):
        env_file = 'env' + str(i) + '_pc.pkl'
        # read environment
        point_cloud = pickle.load(open(os.path.join(env_folder_path, env_type, env_file), 'rb'))
        # display environment
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        # save figure in env folder in type folder
        plt.savefig(os.path.join(env_folder_path, env_type, env_file[:-4] + '.png'))
        plt.show()
        # clear plot
        plt.clf()

if __name__ == '__main__':
    if(env_type == '2d'):
        create_point_clouds_2d()
        plot_point_clouds_2d()
    elif(env_type == '3d'):
        create_point_clouds_3d()
        plot_point_clouds_3d()
    