import pickle 
import numpy as np
from prm_3d import PRM, Node
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=int, default=0)
    parser.add_argument('--env', type=str, default='2d')
    parser.add_argument('--num_nodes', type=int, default=2500)
    parser.add_argument('--num_iter', type=int, default=5000)
    args = parser.parse_args()

    # load env0
    with open('envs/{}/env{}.pkl'.format(args.env, args.env_id), 'rb') as f:
        env0 = pickle.load(f)

    prm = PRM(obstacleList=env0, randArea=[-20, 20])

    # open nodelist file
    with open('{}/{}/graph_env_{}_nodes_{}.pkl'.format(args.env, args.env_id, args.env_id, args.num_nodes), 'rb') as f:
        nodes = pickle.load(f)

    prm.nodeList = nodes

    dataset = []
    for iter in range(args.num_iter):

        start = [np.random.uniform(-20, 20), np.random.uniform(-20, 20)]
        while not prm.collisionCheck(Node(state=start)):
            start = [np.random.uniform(-20, 20), np.random.uniform(-20, 20)]


        goal = [np.random.uniform(-20, 20), np.random.uniform(-20, 20)]
        while not prm.collisionCheck(Node(state=goal)):
            goal = [np.random.uniform(-20, 20), np.random.uniform(-20, 20)]


        path, cost, cost_to_goal = prm.getShortestPath(start = Node(state=start), goal = Node(state=goal))
        print('cost', cost)

        print('start', start)
        print('goal', goal)
        

        data = []

        print(len(path))
        print(len(cost_to_goal))
        # zip and print path and cost to goal
        for i in range(len(path)):
            if i != len(path)-1 and i != 0:
                data.append([start, goal, path[i].state, cost_to_goal[i]])
                print(path[i].state, cost_to_goal[i])

        dataset.append(data)

        if iter % 500 == 0:
            print('iter', iter)
            # save dataset
            with open('2d/{}/dataset.pkl'.format(env), 'wb') as f:
                pickle.dump(dataset, f)
            
        




if __name__ == '__main__':
    main()