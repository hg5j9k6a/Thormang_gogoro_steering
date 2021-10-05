#! /usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Manipulator_Net(torch.nn.Module):
    def __init__(self, n_feature = 2, n_hidden1 = 32,n_hidden2 = 7, n_output = 14):
        super(Manipulator_Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        # self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # hidden layer
        # self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer
        self.predict = torch.nn.Linear(n_hidden1, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        # x = F.relu(self.hidden2(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
    
# if __name__ == "__main__":
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     manipulator = Manipulator_Net(n_feature=2, n_hidden1=32,n_hidden2=7, n_output=14)     # define the network
#     model_path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config/10740000/NN_10740000.pth"
#     manipulator.load_state_dict(torch.load(model_path))
#     # manipulator.to(device)
#     # print(device)
#     # print(manipulator)
    
#     NN_start_time = time.time()
#     target_theta = manipulator(torch.Tensor([0,0]))
#     NN_time_spent = (time.time() - NN_start_time) * 1000
    
#     print("NN_time_spent:",NN_time_spent," ms")
    