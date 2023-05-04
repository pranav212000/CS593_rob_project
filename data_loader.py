import numpy as np
import matplotlib.pyplot as plt
import pickle 
from tqdm import tqdm


def data_loader_2d(N=10, with_start = False, samples = 100000, get_together = False, point_cloud = True, env_type='2d'):
    print('Loading data...', flush=True)
    obs = []

    if point_cloud:
        # Load point clouds
        for i in range(0,N):

            temp = pickle.load(open('envs/{}/env{}_pc.pkl'.format(env_type, str(i)), 'rb'))
            temp = np.array(temp)
            temp = temp.flatten()

            obs.append(temp)

        obs = np.array(obs)
        obs = obs / 20.0
    else:
        # Load obstacle points (x,y,sizex,sizey) for 2d and (x,y,z) for 3D
        for i in range(0,N):

            temp = pickle.load(open('envs/{}/env{}_pc.pkl'.format(env_type, str(i)), 'rb'))
            temp = np.array(temp)
            temp = temp.flatten()

            obs.append(temp)

        obs = np.array(obs)
        # Scale the obstacles
        obs = obs / 20.0



    targets = np.array([])
    dataset = None
    env_indices = None
    for i in range(N) :
        env = pickle.load(open('{}/{}/dataset.pkl'.format(env_type, i), 'rb'))
        env = np.array(env)
        target = env[:, -1]
        data = env[:, :-1]
        
        targets = np.concatenate((targets, target))
        if dataset is None:
            dataset = data
        else:
            dataset = np.concatenate((dataset, data), axis=0)

        indices = np.ones(len(target), dtype=int) * i
        if env_indices is None:
            env_indices = indices
        else:
            env_indices = np.concatenate((env_indices, indices), axis=0)


    targets = np.array(targets)
    targets = np.expand_dims(targets, axis=1)
    dataset = np.array(dataset)

    # Drop the start position from the dataset
    if not with_start:
        if env_type == '2d':
            dataset = dataset[:, 2:]
        elif env_type == '3d':
            dataset = dataset[:, 3:]

   
    
    # Scale the dataset and targets
    dataset = dataset / 20.0
    targets = targets / 20.0



    env_indices = np.array(env_indices)

    if get_together:
        dataset = np.concatenate((obs[env_indices], dataset), axis=1)

    # Shuffle the dataset
    zipped = list(zip(dataset, targets, env_indices))
    np.random.shuffle(zipped)
    dataset, targets, env_indices = zip(*zipped)
    dataset = np.array(dataset)
    targets = np.array(targets)
    env_indices = np.array(env_indices)

    if samples > len(dataset):
        samples = len(dataset)


    
    return obs, dataset[:samples], targets[:samples], env_indices[:samples]
