import pickle 
import numpy as np
from prm_3d import PRM, Node
import argparse
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=int, default=0)
    parser.add_argument('--env', type=str, default='2d')
    parser.add_argument('--num-nodes', type=int, default=2500)
    parser.add_argument('--num-iter', type=int, default=5000)
    parser.add_argument('--save-every', type=int, default=500)
    args = parser.parse_args()

    # load env0
    with open('envs/{}/env{}.pkl'.format(args.env, args.env_id), 'rb') as f:
        env0 = pickle.load(f)

    prm = PRM(obstacleList=env0, randArea=[-20, 20])

    
    with open('{}/{}/graph_env_{}_nodes_{}.pkl'.format(args.env, args.env_id, args.env_id, args.num_nodes), 'rb') as f:
        nodes = pickle.load(f)

    prm.nodeList = nodes

    dataset = []
    for iter in tqdm(range(args.num_iter)):

        start = [np.random.uniform(-20, 20), np.random.uniform(-20, 20)]
        while not prm.collisionCheck(Node(state=start)):
            start = [np.random.uniform(-20, 20), np.random.uniform(-20, 20)]


        goal = [np.random.uniform(-20, 20), np.random.uniform(-20, 20)]
        while not prm.collisionCheck(Node(state=goal)):
            goal = [np.random.uniform(-20, 20), np.random.uniform(-20, 20)]


        path, cost, cost_to_goal = prm.getShortestPath(start = Node(state=start), goal = Node(state=goal))


        if path is None:
            continue
        # print(cost)
        # print('cost', cost)

        # print('start', start)
        # print('goal', goal)
        

        data = np.array([])
        start = np.array(start)
        goal = np.array(goal)
        

        # print(len(path))
        # print(len(cost_to_goal))
        # zip and print path and cost to goal
        for i in range(len(path)):
            # print('path', path[i].state, 'cost', cost_to_goal[i])
            state = np.array(path[i].state)
            dataset.append(np.array([start[0], start[1], goal[0], goal[1],state[0], state[1], cost_to_goal[i]]))
                

        

        if iter % args.save_every == 0:
            # save dataset
            with open('2d/{}/dataset.pkl'.format(args.env_id), 'wb') as f:
                pickle.dump(dataset, f)
            
        




if __name__ == '__main__':
    main()