from utils import *
from train import *
from Manipulator_NN import *
from PSO_IK import *

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

import numpy as np
import random
import time
import os
import sys

if __name__ == "__main__":
    # rotation_angle = np.linspace(0,18,1*36+1)
    rotation_angle = [np.degrees(0.15)]
    # rotation_angle = np.array([-18,18])
    left_steer_theta =  np.array([-0.19468600572796468, 1.0450682422271944, 0.5306074845300532 ,0.29709705728495717 , 2.6494854330479285, 0.11471756843258163, -0.8633519169683135])
    right_steer_theta = np.array([ 0.22771684480929189, -0.8913385651906776,-0.2836369083505996, -0.5661194916459493, -1.8383287021792807, -0.4873243771690525, 1.1503494323915078])
    max_joint_change = []
    steering_commands = []
    for i in rotation_angle :
        print("---------- {} deg ----------".format(i))
        start_time = time.time()
        left_target_pos , right_target_pos = target_pos(i,22)

        left_target_theta , left_count , left_error  = IK(left_steer_theta,left_target_pos,"left_arm")
        right_target_theta, right_count, right_error = IK(right_steer_theta,right_target_pos,"right_arm")
        
        right_current_pose = fwd_kinematics(right_target_theta,"right_arm")
        left_current_pose  = fwd_kinematics(left_target_theta,"left_arm")

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
        end_time = time.time()
        print("Computation time : ",end_time-start_time)
        print("----------------------------------------------")
        print("max_joint_change:",np.max(np.abs(np.abs(np.array(right_steer_theta))-np.abs(np.array(right_target_theta)))))
        max_joint_change.append(np.max(np.abs(np.abs(np.array(right_steer_theta))-np.abs(np.array(right_target_theta)))))
        steering_commands.append(i)
    print("----------------------------------------------")
    steering_commands = np.array(steering_commands)
    max_joint_change = np.array(max_joint_change)
    
    
    # fig = plt.figure( figsize=(15,10) )
    # ax = fig.add_subplot(111)

    # ax.plot(np.radians(steering_commands[:]),max_joint_change[:],label='joint_difference',color="b")
    # ax.set_xlabel('Steering_Command (radians)', fontsize=8)
    # ax.set_ylabel('Joint_Difference (radians)', fontsize=8)
    # # print("Max_sim_time_cost : ",np.max(sim_time_cost))
    # print("Av.sim_time_cost : " ,np.mean(sim_time_cost))
    # ss = time.time()
    # time.sleep(1)
    # print(time.time()-ss)
    plt.show()
