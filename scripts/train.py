#! /usr/bin/env python3

from logging import error

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import time
import os
import sys

import torch
import torch.nn.functional as F

import rospy
import rospkg
from pioneer_kinematics.kinematics import Kinematics
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64,Float32MultiArray,Bool

from utils import *
from PSO_IK import *
from Manipulator_NN import *

def target_pos(rotation_angle,right_start_pitch):
    ''' Using `mid_point` and `y_spacing` will compute the left and right
    arm positions and send the commands
    Input rotation_angle is degrees unit
    '''
    
    ###  Parameters  ###
    # X (forward) front of robot is positive
    # Y (sideways) left of robot is positive
    # Z (up-down) up of robot is positive
    rotation_angle = rotation_angle
    
    x_start_pos = 0.41
    x_left_offset = 0.02
            
    y_spacing = 0.248
    z_height = 0.872 # 0.865
    
    left_pitch_offset  = -5
    left_height_offset = -0.02
    left_space_offset  = -0.035
    
    center_offset = 0.1
    sterring_bar_angle = 90 - 63.435
    extend_angle = 18.435

    yaw_offset         = 15
    
    # right_start_pitch = 22.0
    right_pitch = right_start_pitch     # In degrees
    # gasing_offset = 10
    
    roll = 90 # In degrees (for both arms)
    right_roll_offset = -12
    
    mid_point = np.zeros([3])

    mid_point[0] = x_start_pos
    mid_point[1] = 0  #center_offset * np.sin(np.radians(rotation_angle))     # center_offset the center from the steering bar,which is 0.07 m
    mid_point[2] = z_height - 0.1*np.sin(np.radians(sterring_bar_angle))* (1 - np.cos(np.radians(rotation_angle)) )
        

    # Compute new position based on parameters
    l_x = mid_point[0] + x_left_offset + y_spacing * np.sin(np.radians(rotation_angle))   # and use geometry find the kinematic path, which is a ellipse
    l_y = mid_point[1] + (y_spacing+left_space_offset) * ( 1 - 2.1*center_offset * abs(np.sin(np.radians(rotation_angle))) )
    l_z = mid_point[2] + y_spacing * np.sin(np.radians(extend_angle)) *np.cos(np.radians(90 - rotation_angle)) * np.sin(np.radians(sterring_bar_angle))
    
    r_x = mid_point[0] - y_spacing * np.sin(np.radians(rotation_angle))   # and use geometry find the kinematic path, which is a ellipse
    r_y = mid_point[1] - y_spacing * ( 1 - 2.1*center_offset * abs(np.sin(np.radians(rotation_angle))) )
    r_z = mid_point[2] + y_spacing * np.sin(np.radians(extend_angle))* np.cos(np.radians(90 + rotation_angle)) * np.sin(np.radians(sterring_bar_angle))

    left_target  = np.array([l_x,l_y,l_z+left_height_offset, np.radians(roll), np.radians(left_pitch_offset),np.radians( -1.25*rotation_angle + yaw_offset)])
    right_target = np.array([r_x,r_y,r_z,np.radians( -(roll + right_roll_offset )+ 0.5*rotation_angle), np.radians(right_pitch), np.radians( -1.25*rotation_angle - yaw_offset)])
    # print("rotation_angle",rotation_angle)
    return left_target , right_target



if __name__ == "__main__":
    generate_trainin_data = False
    train = True
    load_NN = 142
    
    
    net = Manipulator_Net(n_feature=2, n_hidden1=32,n_hidden2=7, n_output=14)     # define the network
    print(net)  # net architecture

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001) #0.0001
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    rospack = rospkg.RosPack()
    PATH = rospack.get_path('thormang3_gogoro') + "/config/NN"
    
    if load_NN:
        load_path = PATH + "/{}/NN_{}.pth".format(str(load_NN),str(load_NN))
        net.load_state_dict(torch.load(load_path))
        print("Load_modle : NN_{}.pth".format(str(load_NN)))
    # PATH = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config/NN"
    # x = torch.unsqueeze(torch.linspace(-16, 16, 0*28+1+2), dim=1)  # x data (tensor), shape=(100, 1)

    if generate_trainin_data:
        
        rotation_angle = np.linspace(-18,18,20*36+1)
        pitch = np.linspace(-15,25,3*10+1)
        
        right_steer_theta = np.array([ 0.22771684480929189, -1.0913385651906776,-0.2836369083505996, -0.5661194916459493, -1.8383287021792807, -0.4873243771690525, 1.1503494323915078])
        left_steer_theta =  np.array([-0.19468600572796468, 1.0450682422271944, 0.5306074845300532 ,0.29709705728495717 , 2.6494854330479285, 0.11471756843258163, -0.8633519169683135])
        
        steering_command = []
        joint = []
        
        for i in rotation_angle :
            for j in pitch:
                print("---------- steer {} deg & gas {} deg----------".format(round(i,3),round(j,3)))

                left_target_pos , right_target_pos = target_pos(i,j)
                start_time = time.time()
                left_target_theta , left_count , left_error  = IK(left_steer_theta,left_target_pos,"left_arm")
                right_target_theta, right_count, right_error = IK(right_steer_theta,right_target_pos,"right_arm")
                
                right_current_pose = fwd_kinematics(right_target_theta,"right_arm")
                left_current_pose  = fwd_kinematics(left_target_theta,"left_arm")

                steering_command.append(np.array([i,j]))
                joint.append(np.hstack((left_target_theta,right_target_theta)))
            
                end_time = time.time()
                duration = end_time - start_time
                print("right_arm :\nx,y,z",right_current_pose[0:3])
                print("roll,pitch,yaw:",np.degrees(right_current_pose[3:7]))
                print("target",right_target_pos[0:3],"\n",np.degrees(right_target_pos[3:7]))
                print("Error:",right_error)
                print("theta : ",np.degrees(right_target_theta))
                print()
                print("left_arm  :\nx,y,z",left_current_pose[0:3])
                print("roll,pitch,yaw:",np.degrees(left_current_pose[3:7]))
                print("target",left_target_pos[0:3],"\n",np.degrees(left_target_pos[3:7]))
                print("Error:",left_error)
                print("theta : ",np.degrees(left_target_theta))
                
                print()
                print("iteration: ",right_count," (right) + ",left_count," (left) = ",right_count+left_count)
                print("duration",duration)
                
                if right_error > 0.0001 or left_error > 0.0001 :
                    print("IK Fail !")
                    print("IK Fail !")
                    print("IK Fail !")
                    sys.exit()
                    break
                
        np.save(os.path.join(PATH, 'steering_command_input'), np.array(steering_command))
        np.save(os.path.join(PATH, 'joint_output'), np.array(joint))
    
    if train:
        ## Train ##
        
        steering_command = np.load(os.path.join(PATH, 'steering_command_input.npy'))
        joint = np.load(os.path.join(PATH, 'joint_output.npy'))
        
        x = torch.Tensor(steering_command)#.unsqueeze(dim = 1)
        y = torch.Tensor(joint)
        
        ## test shape ##
        # y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
        # print("x",x)
        # print("y",y)    
        ################
        
        # plt.ion()   # plotting iteration
        
        history_loss = []
        NN_err = 2
        t = 0
        if load_NN:
            t = load_NN*10000 + 1
        
        while NN_err > 0.00001 :
            prediction = net(x)     # input x and predict based on x

            loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
            
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
            NN_err = loss.data.numpy()

            if t % 10000 == 0 or NN_err < 0.00001:
                plt.cla()

                history_loss.append(loss.data.numpy())
            #     # plot and show learning process
                # plt.plot(range(len(history_loss[1:])),np.array(history_loss[1:]),'r-', lw=1,label='Loss')
            ##     plt.scatter(x.data.numpy(), y.data.numpy())
            ##     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            ##     plt.text(len(history_loss), 1, 'Loss={:.4}'.format(loss.data.numpy()),fontsize=12,ha='center', va='center')
                # plt.legend()
                # plt.pause(0.1)
                
                path = os.path.join(PATH, '{}'.format(int(t/10000)))
                if not os.path.exists(path):
                    os.mkdir(path)
                filename_1 = "NN_{}.pth".format(int(t/10000))
                filename_2 = "NN_{}.jpg".format(int(t/10000))
                
                # Save matrix
                torch.save(net.state_dict(), os.path.join(path, filename_1))
                # plt.savefig(os.path.join(path, filename_2))
                
                
                print("-------- {} --------".format(int(t/10000)))
                print("Saving NN weights.")
                print("Loss : ",loss.data.numpy())
            t = t + 1
            # print(t)
        
        # plt.ioff()
        # plt.show()
    
