from __future__ import print_function
from Model.e2e import End2EndMPNet
import Model.mlp as mlp
import Model.CAE as CAE_2d
import numpy as np
import argparse
import os
import torch
from data_loader import data_loader_2d
from utility import *
from torch.autograd import Variable
import copy
import os
import random
import progressbar
import psutil
import datetime


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def main(args):


    timestamp = datetime.datetime.now().strftime("%d_%H%M%S")
    print('timestamp: ', timestamp)

    # set seed
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    # decide dataloader, MLP, AE based on env_type
    if args.env_type == '2d':
        total_input_size = 2800+6
        AE_input_size = 2800
        mlp_input_size = 28+6
        output_size = 1
        load_train_dataset = data_loader_2d

        CAE = CAE_2d
        MLP = mlp.MLP
    elif args.env_type == '3d':
        total_input_size = 6000 + 6
        AE_input_size = 6000
        mlp_input_size = 28+6
        output_size = 3
        # load_train_dataset = data_loader_2d
        CAE = CAE_2d
        MLP = mlp.MLP

    model = End2EndMPNet(total_input_size, AE_input_size, mlp_input_size,
                         output_size, CAE, MLP)

    if args.env_type == '2d' or args.env_type == '3d':
        loss_f = model.loss

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    model_path = 'model_env_{}_epoch_{}.pkl'.format(
        args.env_type, args.start_epoch)

    if args.start_epoch > 0:
        print('Loading model from {}'.format(model_path))
        load_net_state(model, os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(
            os.path.join(args.model_path, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)

    process = psutil.Process(os.getpid())
    print('Physical Memory Usage: ', process.memory_info().rss/1024**3, 'GB')

    print('psutil.virtual_memory().total : ',
          psutil.virtual_memory().total/1024**3, 'GB')

    if torch.cuda.is_available():
        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())
        print('Current cuda device name ',
              torch.cuda.get_device_name(torch.cuda.current_device()))

        model.cuda()

        model.encoder.cuda()

    if args.opt == 'Adagrad':
        model.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
    elif args.opt == 'Adam':
        model.set_opt(torch.optim.Adam, lr=args.learning_rate)
    elif args.opt == 'SGD':
        model.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
    elif args.opt == 'ASGD':
        model.set_opt(torch.optim.ASGD, lr=args.learning_rate)

    if args.start_epoch > 0:
        load_opt_state(model, os.path.join(args.model_path, model_path))

    # load train and test data
    print('loading...')
    obstacles, dataset, targets, env_indices = data_loader_2d(N=args.N)
    print('obstacles.shape', np.array(obstacles).shape, flush=True)
    print('dataset.shape', np.array(dataset).shape, flush=True)
    print('target.shape', np.array(targets).shape, flush=True)
    print('env_indices.shape', np.array(env_indices).shape, flush=True)

    val_size = 1000

    val_dataset = dataset[:-val_size]
    print('val_dataset.shape', np.array(val_dataset).shape, flush=True)
    val_targets = targets[:-val_size]
    val_env_indices = env_indices[:-val_size]
    # Train the Models
    print('training...')
    writer_fname = '%s_%f_%s' % (args.env_type, args.learning_rate, args.opt)
    writer = SummaryWriter('./runs/'+writer_fname)
    record_loss = 0.
    record_i = 0
    val_record_loss = 0.
    val_record_i = 0
    for epoch in range(args.start_epoch, args.epochs+1):

        sum_train_loss = 0.0
        sum_val_loss = 0.0

        widgets = [f'epoch{epoch} ',
                   progressbar.Bar(),
                   ' (batch ', progressbar.SimpleProgress(), ') ',
                   progressbar.ETA(),
                   ' train/val loss ',
                   progressbar.Variable(
                       'train_loss', format='{formatted_value}'), '/',
                   progressbar.Variable(
                       'val_loss', format='{formatted_value}'),
                   ]
        bar = progressbar.ProgressBar(widgets=widgets)
        for i in bar(range(0, len(dataset), args.batch_size)):
            # randomly pick 100 data
            #bi = np.concatenate( (obstacles[env_indices[i:i+100]], dataset[i:i+100]), axis=1).astype(np.float32)
            bi = np.array(dataset[i:i+args.batch_size]).astype(np.float32)

            batch_data = np.array(
                dataset[i:i+args.batch_size]).astype(np.float32)
            batch_target = np.array(
                targets[i:i+args.batch_size]).astype(np.float32)
            batch_env_indices = np.array(
                env_indices[i:i+args.batch_size]).astype(np.float32)
            batch_obstacles = np.array(
                obstacles[env_indices[i:i+args.batch_size]]).astype(np.float32)

            batch_data = torch.FloatTensor(batch_data)
            batch_target = np.array(batch_target).astype(np.float32)
            batch_target = torch.FloatTensor(batch_target)
            batch_env_indices = torch.FloatTensor(batch_env_indices)
            batch_obstacles = torch.FloatTensor(batch_obstacles)

            model.zero_grad()

            batch_data = to_var(batch_data)
            batch_target = to_var(batch_target)
            batch_env_indices = to_var(batch_env_indices)
            batch_obstacles = to_var(batch_obstacles)

            # print('batch_data.shape', batch_data.shape)
            # print('batch_target.shape', batch_target.shape)
            # print('batch_env_indices.shape', batch_env_indices.shape)
            # print('batch_obstacles.shape', batch_obstacles.shape)

            model.step(batch_data, batch_obstacles, batch_target)

            loss = loss_f(model(batch_data, batch_obstacles),
                          batch_target).item()

            sum_train_loss += loss * args.batch_size
            record_loss += loss
            record_i += 1
            if record_i % 100 == 0:
                writer.add_scalar('train_loss', record_loss / 100, record_i)
                record_loss = 0.

            ######################################
            # loss on validation set
            idx = i % val_size

            batch_data = np.array(
                val_dataset[idx:idx+args.batch_size]).astype(np.float32)
            batch_target = np.array(
                val_targets[idx:idx+args.batch_size]).astype(np.float32)
            batch_env_indices = np.array(
                val_env_indices[idx:idx+args.batch_size]).astype(np.float32)
            batch_obstacles = np.array(
                obstacles[val_env_indices[idx:idx+args.batch_size]]).astype(np.float32)

            batch_data = torch.FloatTensor(batch_data)
            batch_target = np.array(batch_target).astype(np.float32)
            batch_target = torch.FloatTensor(batch_target)
            batch_env_indices = torch.FloatTensor(batch_env_indices)
            batch_obstacles = torch.FloatTensor(batch_obstacles)

            model.zero_grad()
            batch_data = to_var(batch_data)
            batch_target = to_var(batch_target)
            batch_env_indices = to_var(batch_env_indices)
            batch_obstacles = to_var(batch_obstacles)

            temp = model(batch_data, batch_obstacles)

            loss = loss_f(model(batch_data, batch_obstacles),
                          batch_target).item()
            sum_val_loss += loss * args.batch_size

            val_record_loss += loss
            val_record_i += 1
            if val_record_i % 100 == 0:
                writer.add_scalar(
                    'val_loss', val_record_loss / 100, val_record_i)
                val_record_loss = 0.

            bar.update(train_loss=sum_train_loss / (i + args.batch_size),
                       val_loss=sum_val_loss / (i + args.batch_size))

        # Save the models every 50 epochs
        if epoch % 50 == 0:
            
            # create a time stamp folder
            if not os.path.exists('models/{}'.format(timestamp)):
                os.makedirs('models/{}'.format(timestamp))


            model_path = "{}/model_env_{}_epoch_{}.pkl".format(timestamp,args.env_type, epoch)
            print(model_path)
            # model_path='mpnet_epoch_%d.pkl' %(epoch)
            save_state(model, torch_seed, np_seed, py_seed,
                       os.path.join(args.model_path, model_path))
            # test
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


parser = argparse.ArgumentParser()

parser.add_argument('--model-path', type=str,
                    default='./models/', help='path for saving trained models')
parser.add_argument('--N', type=int, default=10,
                    help='number of environments')

parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--device', type=int, default=0, help='cuda device')

parser.add_argument('--env-type', type=str, default='2d',
                    help='2d')
parser.add_argument('--world-size', nargs='+', type=float,
                    default=20., help='boundary of world')
parser.add_argument('--opt', type=str, default='Adagrad')
args = parser.parse_args()
print(args)
main(args)
