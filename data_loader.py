# data loader for the dataset
# input is obstacle list, start point, goal point and output is the cost

import numpy as np
import matplotlib.pyplot as plt
import pickle 
from tqdm import tqdm


def data_loader_2d(N=10, with_start = False, samples = 100000, get_together = False):
    print('Loading data...', flush=True)
    # load obstacles from pickle file
    obs = []
    for i in range(0,N):
        # load obstacles from pickle file
        temp = pickle.load(open('envs/2d/env'+str(i)+'_pc.pkl', 'rb'))
        # print(temp)
        # make temp a 1d array
        temp = np.array(temp)
        # flatten the array
        temp = temp.flatten()

        obs.append(temp)

    obs = np.array(obs)
    obs = obs / 20.0



    targets = np.array([])
    dataset = None
    env_indices = None
    for i in range(N) :
        env = pickle.load(open('2d/{}/dataset.pkl'.format(i), 'rb'))
        env = np.array(env)
        target = env[:, -1]
        data = env[:, :-1]
        # print(target.shape)
        # print(data.shape)
        
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

    if not with_start:
        dataset = dataset[:, 2:]

    # range -1 to 1
    dataset = dataset / 20.0



    env_indices = np.array(env_indices)

    if get_together:
        dataset = np.concatenate((obs[env_indices], dataset), axis=1)

    # print('targets : ', targets)
    # print('dataset : ', dataset)
    # print('env_indices : ', env_indices)
    zipped = list(zip(dataset, targets, env_indices))
    np.random.shuffle(zipped)
    dataset, targets, env_indices = zip(*zipped)
    dataset = np.array(dataset)
    targets = np.array(targets)
    env_indices = np.array(env_indices)

    if samples > len(dataset):
        samples = len(dataset)


    
    return obs, dataset[:samples], targets[:samples], env_indices[:samples]




def load_train_dataset(N=10):
    obs = []
    for i in range(0,N):
        # load obstacles from pickle file
        temp = pickle.load(open('envs/2d/env'+str(i)+'_pc.pkl', 'rb'))
        # print(temp)
        obs.append(temp)
    obs = np.array(obs)

    # load dataset from 2d folder
    costs = []
    for i in range(0,N):
        # open environment folder from 2d folder
        env = pickle.load(open('2d/'+str(i)+'/dataset.pkl', 'rb'))
        costs.append(env)

    costs = np.array(costs)
    
    data = []
    for i in tqdm(range(0,N)):
        temp = []
        for obstacle in obs[i]:
            temp.append(obstacle[0])
            temp.append(obstacle[1])
        
        for cost in costs[i]:
            temp2 = []
            # add data from temp in temp2
            temp2.extend(temp)
            temp2.append(cost[2])
            temp2.append(cost[3])
            temp2.append(cost[4])
            temp2.append(cost[5])
            temp2.append(cost[6])
            # add temp2 to data
            # print(cost)
            # break
            data.append(temp2)

    print("Data loaded successfully")
    return data
        
        
# define a function to load data in batches
def load_batch(data, batch_size):
    # shuffle the data
    np.random.shuffle(data)
    # get the number of batches
    n_batches = len(data)//batch_size
    # get the data in batches
    batches = np.array_split(data, n_batches)
    return batches

# data = load_train_dataset(10)
# batch = load_batch(data, 32)
