import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import struct
import numpy as np
import argparse
import os
import time
from mpl_toolkits import mplot3d


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time



def plot3DObstacles(points_list, ax):

    for points in points_list:

        points = np.array(points)
        Z = points
        r = [-1, 1]
        X, Y = np.meshgrid(r, r)

        verts = [[Z[0], Z[1], Z[3], Z[2]],
                 [Z[4], Z[5], Z[7], Z[6]],
                 [Z[0], Z[1], Z[5], Z[4]],
                 [Z[2], Z[3], Z[7], Z[6]],
                 [Z[1], Z[3], Z[7], Z[5]],
                 [Z[4], Z[6], Z[2], Z[0]]
                 ]

        ax.add_collection3d(Poly3DCollection(
            verts, alpha=0.8, edgecolors='k', linewidth=0.5))

    return ax

def getPoints(obs, box):
    points = []

    points.append([obs[0] - box[0]/2, obs[1] - box[1]/2, obs[2] - box[2]/2])
    points.append([obs[0] - box[0]/2, obs[1] - box[1]/2, obs[2] + box[2]/2])
    points.append([obs[0] - box[0]/2, obs[1] + box[1]/2, obs[2] - box[2]/2])
    points.append([obs[0] - box[0]/2, obs[1] + box[1]/2, obs[2] + box[2]/2])
    points.append([obs[0] + box[0]/2, obs[1] - box[1]/2, obs[2] - box[2]/2])
    points.append([obs[0] + box[0]/2, obs[1] - box[1]/2, obs[2] + box[2]/2])
    points.append([obs[0] + box[0]/2, obs[1] + box[1]/2, obs[2] - box[2]/2])
    points.append([obs[0] + box[0]/2, obs[1] + box[1]/2, obs[2] + box[2]/2])

    return points

def visualize(obstacle_centers, obstacle_dimensions):

    ax = plt.axes(111, projection='3d')

    boxes = obstacle_dimensions

    # plotted corners of the environment as well so that entire environment is visible 
    corners = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]

    corners = np.array(corners)
    corners = corners * 20

    ax.scatter3D(corners[:, 0], corners[:, 1], corners[:, 2], c='gray', s=0.01)


    points_list = []
    for i, (x, y, z) in enumerate(obstacle_centers):
        points = getPoints((x,y,z), boxes[i])
        points_list.append(points)
        

    ax = plot3DObstacles(points_list, ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()




def main(args):

    print(args.path_file)
    flag = True
    for path_file in args.path_file:
        if os.path.exists(path_file):
            flag = False
            break

    if flag:
        print('No file path found')
        return

    if not os.path.exists('./output/'):
        os.makedirs('./output/')

    output_path = './output/' + 'env_' + str(args.env_id) + '/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig = plt.figure()
    ax = plt.axes(111, projection='3d')


    if args.point_cloud:
        file = args.data_path + f'obs_cloud/obc{args.env_id}.dat'
        obs = []
        temp = np.fromfile(file)
        obs.append(temp)
        obs = np.array(obs).astype(np.float32).reshape(-1, 3)

        ax.scatter3D(obs[:, 0], obs[:, 1], obs[:, 2], c='gray')
        

    else:
        size = 10
        

        boxes = [
            [5, 5, 10], [5, 10, 5],
            [5, 10, 10], [10, 5, 5], [10, 5, 10], [10, 10, 5], [
                10, 10, 10], [5, 5, 5], [10, 10, 10], [5, 5, 5]
        ]

        # plotted corners of the environment as well so that entire environment is visible 
        corners = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                     [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]

        corners = np.array(corners)
        corners = corners * 20

        ax.scatter3D(corners[:, 0], corners[:, 1], corners[:, 2], c='gray', s=0.01)


        points_list = []
        for i, (x, y, z) in enumerate(obc):
            points = getPoints((x,y,z), boxes[i])
            points_list.append(points)
            

        ax = plot3DObstacles(points_list, ax)

       

    for path_file in args.path_file:

        
        if not os.path.exists(path_file):
            print('File path does not exist! Path was not found: ' + path_file)
            print('Continuing')
                
            continue

        distance = 0
        
        path_name = ''

        # visualize path
        if path_file.endswith('.txt'):
            path = np.loadtxt(path_file)
            path_name = path_name + 'MPNet '
        else:
            path = np.fromfile(path_file)
            path_name = path_name + 'RRT* '
        # print(path)

        path = path.reshape(-1, 3)
        path_x = []
        path_y = []
        path_z = []
        for i in range(len(path)):
            path_x.append(path[i][0])
            path_y.append(path[i][1])
            path_z.append(path[i][2])

            if i > 0:
                distance += np.linalg.norm(path[i] - path[i-1])

        print('path: {}, cost: {:.2f}'.format(path_file, distance))
        # print('distance: ' + str(distance))
        path_name = path_name + 'cost {:.2f}'.format(distance)

        ax.plot3D(path_x, path_y, path_z, marker='o', alpha=1, label=path_name)


    ax.plot3D(path_x[0], path_y[0], path_z[0], color='green', marker='^', label='Start', markersize=10)
    ax.plot3D(path_x[-1], path_y[-1], path_z[-1], color='red', marker='^', label='Goal', markersize=10)
    plt.legend(loc="upper left", prop={'size': 6})


    path_file_name = args.path_file[0].split('/')[-1]
    path_file_name = path_file_name.split('.')[0]
    path_file_name = path_file_name.split('_')[-1]
    print('path_file_name: ' + path_file_name)

    time_str = time.strftime("%Y%m%d-%H%M%S")
    path = output_path + str(path_file_name) + '_' + time_str + \
        '_dropout_' + str(args.dropout) + '_lvc_' + str(args.lvc) + '.png'
    print('Saving figure to: ' + path)

    if os.path.exists(path):
        os.remove(path)

    
    plt.title(args.plot_name)

    ax.view_init(elev=30., azim=45)


    plt.savefig(path)


    plt.show()

    print('plotting done!')


# parser = argparse.ArgumentParser()
# parser.add_argument('--data-path', type=str, default='../data/simple/')
# parser.add_argument('--env-id', type=int, default=0)
# parser.add_argument('--point-cloud', default=False, action='store_true')
# parser.add_argument('--path-file', nargs='*', type=str,
#                     default=[], help='path file')
# parser.add_argument('--dropout', default=False, action='store_true')
# parser.add_argument('--lvc', default=False, action='store_true')
# parser.add_argument('--plot-name', type=str, default='figure')
# args = parser.parse_args()
# print(args)
# main(args)
