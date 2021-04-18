import os
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd
from matplotlib import pyplot
from pandas.plotting import lag_plot
import seaborn as sns
import math
import random

dtype = torch.float
device = torch.device("cpu")

time_slice = 4
particles = 2
features = 5
trajectory_length = 50

class Dataset():
    def __init__(self, batch_size=1, time_lag=4):
        self.data = pd.read_pickle('../data/samples/dyari.pkl')
        self.train = self.data.sample(frac=0.8, random_state=200)
        self.test = self.data.drop(self.train.index)
        self.batch_size = batch_size
        self.time_lag = time_lag
        self.trajectory_length = trajectory_length

    def get_batch(self, mode='train'):
        if mode == 'train':
            trajectory_ids = random.sample(range(0, len(self.train)), self.batch_size)
            simulation_samples = self.train.iloc[trajectory_ids]
        else:
            trajectory_ids = random.sample(range(0, len(self.test)), self.batch_size)
            simulation_samples = self.test.iloc[trajectory_ids]

        batch_x = []
        batch_y = []

        def lag_batch(_positions, _velocity, _energy, _edges):
            time_lag = random.randint(self.time_lag, self.trajectory_length)
            _bx = []
            _by = []
            for time_step in range(time_lag-self.time_lag, time_lag):
                frames = [_positions[time_step], _velocity[time_step], _energy[time_step]]
                result = pd.concat(frames)
                _bx.append(result)
                _by.append(_edges[time_step])
            _by = [_by[-1]]
            return np.asarray(_bx), np.asarray(_by)

        for _id in range(0, len(simulation_samples)):
            _positions = simulation_samples.trajectories[_id].positions
            _velocity = simulation_samples.trajectories[_id].velocity
            _energy = simulation_samples.trajectories[_id].total_energy
            _edges = simulation_samples.trajectories[_id].edges
            _x, _y = lag_batch(_positions, _velocity, _energy, _edges)
            batch_x.append(_x)
            batch_y.append(_y)

        return np.asarray(batch_x), np.asarray(batch_y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(particles, 1)
        self.fc2 = torch.nn.Linear(1, features)
        self.cn1 = torch.nn.Conv2d(time_slice, 1, 2, stride=2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.cn1(x)
        return x


class Model:
    def __init__(self):
        self.net = Net()
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=0.001,
                                         momentum=0.9)

    def loss(self, y_prediction, y_real):
        loss = self.criterion(y_prediction, y_real)
        return loss

    def print_params(self, x):
        for param in self.net.parameters():
            print(param)

    def predict(self, x):
        return self.net.forward(torch.from_numpy(x).float())

    def train(self):
        d = Dataset()
        for step in range(10000):
            x_train, target_train = d.get_batch(mode='train')
            x_test, target_test = d.get_batch(mode='test')
            train_loss = self.loss(y_prediction=self.predict(x_train),
                                   y_real=torch.from_numpy(target_train).float())
            test_loss = self.loss(y_prediction=self.predict(x_test),
                                  y_real=torch.from_numpy(target_test).float())
            print(f'step {step}: {train_loss.item()}, {test_loss}')

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()


def main():
    m = Model()
    m.train()


if __name__ == "__main__":
    main()














