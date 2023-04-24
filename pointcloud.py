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
    for i in range(num_env):
        env_file = 'env' + str(i) + '.pkl'
        point_cloud = []
        env = pickle.load(open(os.path.join(env_folder_path, env_type, env_file), 'rb'))
        for point in env:
            left_bottom_x = point[0]
            left_bottom_y = point[1]
            size_x = point[2]
            size_y = point[3]
            for i in range(num_points):
                x = np.random.uniform(left_bottom_x, left_bottom_x + size_x)
                y = np.random.uniform(left_bottom_y, left_bottom_y + size_y)
                point_cloud.append([x, y])
        point_cloud = np.array(point_cloud)
        pickle.dump(point_cloud, open(os.path.join(env_folder_path, env_type, env_file[:-4] + '_pc.pkl'), 'wb'))


def create_point_clouds_3d():
    for i in range(num_env):
        env_file = 'env' + str(i) + '.pkl'
        point_cloud = []
        env = pickle.load(open(os.path.join(env_folder_path, env_type, env_file), 'rb'))
        for point in env:
            center_x = point[0]
            center_y = point[1]
            center_z = point[2]
            size = 0.2
            for i in range(num_points):
                x = np.random.uniform(center_x - size, center_x + size)
                y = np.random.uniform(center_y - size, center_y + size)
                z = np.random.uniform(center_z - size, center_z + size)
                point_cloud.append([x, y, z])

        point_cloud = np.array(point_cloud)
        pickle.dump(point_cloud, open(os.path.join(env_folder_path, env_type, env_file[:-4] + '_pc.pkl'), 'wb'))


def plot_point_clouds_2d():
    for i in range(num_env):
        env_file = 'env' + str(i) + '_pc.pkl'
        point_cloud = pickle.load(open(os.path.join(env_folder_path, env_type, env_file), 'rb'))
        plt.scatter(point_cloud[:, 0], point_cloud[:, 1])
        plt.savefig(os.path.join(env_folder_path, env_type, env_file[:-4] + '.png'))
        plt.show()
        plt.clf()

def plot_point_clouds_3d():
    for i in range(num_env):
        env_file = 'env' + str(i) + '_pc.pkl'
        point_cloud = pickle.load(open(os.path.join(env_folder_path, env_type, env_file), 'rb'))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
        plt.savefig(os.path.join(env_folder_path, env_type, env_file[:-4] + '.png'))
        plt.show()
        plt.clf()

if __name__ == '__main__':
    if(env_type == '2d'):
        create_point_clouds_2d()
        plot_point_clouds_2d()
    elif(env_type == '3d'):
        create_point_clouds_3d()
        plot_point_clouds_3d()
    