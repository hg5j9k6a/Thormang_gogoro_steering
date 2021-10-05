#! /usr/bin/env python3

from utils import *
from train import *
from Manipulator_NN import *

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

import numpy as np
import random
import time
import os
import sys

import torch
import torch.nn.functional as F


def left_forward_kinematics(theta,rend = True):
    # thetas use rad unit
    
    # Draw the world coordinate system
    # plotCoordinateSystem( ax, 0.1, 3.0 )

    # Draw the left shoulder Joint 1 (Lateral)
    leftShoulderTranslation = translate( 0.0, 0.152, 1.138-0.0845 )
    leftShoulderRotation = rotateX( -90.0/180.0 * math.pi )
    
    j1A = leftShoulderTranslation.dot( leftShoulderRotation )
    # plotCoordinateSystem( ax, 0.1, 2.0, j1A )

    # sh_p1 -> sh_r
    j2_d  = 0.060 # m
    j2_ax = 0.057 # m
    j2_ay = 0.039 # m
    j2T = rotateZ( theta[0] ).dot( translate( 0, 0, j2_d) ).dot( translate( j2_ax, j2_ay, 0 ) ).dot( rotateY( -90.0/180.0 * math.pi ) )

    j2A = j1A.dot(j2T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j2A )
    # drawLink(ax, j1A, j2A, width=25 )

    # sh_r -> sh_p2
    j3_d  = 0.057 # m
    j3_ax = 0.033 # m
    j3_ay = 0.000 # m
    j3T = rotateZ( theta[1] ).dot( translate( 0, 0, j3_d) ).dot( translate( j3_ax, j3_ay, 0 ) ).dot( rotateY( 90.0/180.0 * math.pi ) )    
    j3A = j2A.dot(j3T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j3A )
    # drawLink(ax, j2A, j3A, width=25 )
    
    # sh_p2 -> el_y
    j4_d  = 0.187 # m
    j4_ax = 0.030 # m
    j4_ay = -0.057 # m
    j4T = rotateZ( theta[2] ).dot( translate( 0, 0, j4_d) ).dot( translate( j4_ax, j4_ay, 0 ) ).dot( rotateX( -90.0/180.0 * math.pi ) )    
    j4A = j3A.dot(j4T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j4A )
    # drawLink(ax, j3A, j4A, width=25 )

    # el_y -> wr_r
    j5_d  = 0.057 # m
    j5_ax = 0.171 # m
    j5_ay = -0.030 # m
    j5T = rotateZ( theta[3] ).dot( translate( 0, 0, j5_d) ).dot( translate( j5_ax, j5_ay, 0 ) ).dot( rotateY( 90.0/180.0 * math.pi ) )    
    j5A = j4A.dot(j5T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j5A )
    # drawLink(ax, j4A, j5A, width=25 )

    # wr_r -> wr_y
    j6_d  = 0.039 # m
    j6_ax = 0.045 # m
    j6_ay = 0.00 # m
    j6T = rotateZ( theta[4] ).dot( translate( 0, 0, j6_d) ).dot( translate( j6_ax, j6_ay, 0 ) ).dot( rotateY( -90.0/180.0 * math.pi ) )    
    j6A = j5A.dot(j6T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j6A )
    # drawLink(ax, j5A, j6A, width=25 )
    
    # wr_y -> wr_p
    j7_d  = 0.045 # m
    j7_ax = 0.045 # m
    j7_ay = 0.045 # m
    j7T = rotateZ( theta[5] ).dot( translate( 0, 0, j7_d) ).dot( translate( j7_ax, j7_ay, 0 ) ).dot( rotateX( -90.0/180.0 * math.pi ) )    
    j7A = j6A.dot(j7T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j7A )
    # drawLink(ax, j6A, j7A, width=25 )
    
    # wr_p -> grip
    j8_d  = -0.045 # m
    j8_ax = 0.145 # m
    j8_ay = 0.000 # m
    j8T = rotateZ( theta[6] ).dot( translate( 0, 0, j8_d) ).dot( translate( j8_ax, j8_ay, 0 ) ).dot( rotateX( -90.0/180.0 * math.pi ) )    
    j8A = j7A.dot(j8T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j8A )
    # drawLink(ax, j7A, j8A, width=25 )
    
    return j8A

def right_forward_kinematics(theta,rend = True):
    # thetas use rad unit
    
    # Draw the world coordinate system
    # plotCoordinateSystem( ax, 0.1, 3.0 )

    # Draw the left shoulder Joint 1 (Lateral)
    leftShoulderTranslation = translate( 0.0, -0.152, 1.138-0.0845 )
    leftShoulderRotation = rotateX( 90.0/180.0 * math.pi )
    
    j1A = leftShoulderTranslation.dot( leftShoulderRotation )
    # plotCoordinateSystem( ax, 0.1, 2.0, j1A )

    # sh_p1 -> sh_r
    j2_d  = 0.060 # m
    j2_ax = 0.057 # m
    j2_ay = -0.039 # m
    j2T = rotateZ( theta[0] ).dot( translate( 0, 0, j2_d) ).dot( translate( j2_ax, j2_ay, 0 ) ).dot( rotateY( -90.0/180.0 * math.pi ) )

    j2A = j1A.dot(j2T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j2A )
    # drawLink(ax, j1A, j2A, width=25 )

    # sh_r -> sh_p2
    j3_d  = 0.057 # m
    j3_ax = 0.033 # m
    j3_ay = 0.000 # m
    j3T = rotateZ( theta[1] ).dot( translate( 0, 0, j3_d) ).dot( translate( j3_ax, j3_ay, 0 ) ).dot( rotateY( 90.0/180.0 * math.pi ) )    
    j3A = j2A.dot(j3T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j3A )
    # drawLink(ax, j2A, j3A, width=25 )
    
    # sh_p2 -> el_y
    j4_d  = 0.187 # m
    j4_ax = 0.030 # m
    j4_ay = 0.057 # m
    j4T = rotateZ( theta[2] ).dot( translate( 0, 0, j4_d) ).dot( translate( j4_ax, j4_ay, 0 ) ).dot( rotateX( 90.0/180.0 * math.pi ) )    
    j4A = j3A.dot(j4T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j4A )
    # drawLink(ax, j3A, j4A, width=25 )

    # el_y -> wr_r
    j5_d  = 0.057 # m
    j5_ax = 0.171 # m
    j5_ay = 0.030 # m
    j5T = rotateZ( theta[3] ).dot( translate( 0, 0, j5_d) ).dot( translate( j5_ax, j5_ay, 0 ) ).dot( rotateY( 90.0/180.0 * math.pi ) )    
    j5A = j4A.dot(j5T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j5A )
    # drawLink(ax, j4A, j5A, width=25 )

    # wr_r -> wr_y
    j6_d  = 0.039 # m
    j6_ax = 0.045 # m
    j6_ay = 0.00 # m
    j6T = rotateZ( theta[4] ).dot( translate( 0, 0, j6_d) ).dot( translate( j6_ax, j6_ay, 0 ) ).dot( rotateY( -90.0/180.0 * math.pi ) )    
    j6A = j5A.dot(j6T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j6A )
    # drawLink(ax, j5A, j6A, width=25 )
    
    # wr_y -> wr_p
    j7_d  = 0.045 # m
    j7_ax = 0.045 # m
    j7_ay = -0.045 # m
    j7T = rotateZ( theta[5] ).dot( translate( 0, 0, j7_d) ).dot( translate( j7_ax, j7_ay, 0 ) ).dot( rotateX( 90.0/180.0 * math.pi ) )    
    j7A = j6A.dot(j7T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j7A )
    # drawLink(ax, j6A, j7A, width=25 )
    
    # wr_p -> grip
    j8_d  = -0.045 # m
    j8_ax = 0.145 # m
    j8_ay = 0.000 # m
    j8T = rotateZ( theta[6] ).dot( translate( 0, 0, j8_d) ).dot( translate( j8_ax, j8_ay, 0 ) ).dot( rotateX( 90.0/180.0 * math.pi ) )    
    j8A = j7A.dot(j8T)
    # plotCoordinateSystem( ax, 0.1, 2.0, j8A )
    # drawLink(ax, j7A, j8A, width=25 )
    
    return j8A

def fwd_kinematics( thetas ,arm ):
    if arm == "left_arm":
        end_effector = left_forward_kinematics(thetas).dot([0.0, 0.0, 0.0, 1])[0:3]
        roll,pitch,yaw = rxyz(left_forward_kinematics(thetas))
        e = np.array([end_effector[0],end_effector[1],end_effector[2],roll,pitch,yaw])
    elif arm == "right_arm":
        end_effector = right_forward_kinematics(thetas).dot([0.0, 0.0, 0.0, 1])[0:3]
        roll,pitch,yaw = rxyz(right_forward_kinematics(thetas))
        e = np.array([end_effector[0],end_effector[1],end_effector[2],roll,pitch,yaw])
    return e

def Jacobian( thetas, arm ,dt = 20.0/180.0*math.pi):
    J = np.zeros( (6, len(thetas) ) )
    current_pos = fwd_kinematics( thetas ,arm)
    # thetas_d = thetas.copy()
    
    for i,t in enumerate(thetas):
        thetas[i] = thetas[i] + dt
        J[0:6,i] = ( fwd_kinematics(thetas,arm) - current_pos )
        thetas[i] = thetas[i] - dt
    return J

def IK(current_theta,target_pos, arm):
    d_err = target_pos - fwd_kinematics(current_theta,arm)  # fwd_kinematics is to calculate the End effector
    new_theta = current_theta.copy()
    
    ### Momentum Parameter ###
    alpha = 1
    ##########################
    
    ### First Jacobian value ##
    jac = Jacobian(new_theta,arm)
    d_theta = jac.T.dot(d_err)
    new_theta = new_theta + d_theta
    
    ### Clip the joint value in the limited boundary ###
    new_theta = np.hstack((new_theta[0:4].clip(np.radians(-85),np.radians(85)),new_theta[4:7].clip(np.radians(-170),np.radians(170))))
    if arm == "left_arm":
        new_theta[1] = new_theta[1].clip(np.radians(68),np.radians(90))
    else:
        new_theta[1] = new_theta[1].clip(np.radians(-90),np.radians(-68))
    
    ### init the change theta (d_theta) to be first velocity ###
    velocity = d_theta.copy()
    ############################################################
    
    count = 1
    while(np.linalg.norm(d_err) > 0.0001 ):
        ### Calculate Jacobian psedo code ###
        jac = Jacobian(new_theta,arm)
        d_theta = jac.T.dot(d_err)
        new_theta = new_theta + d_theta
        #####################################
        
        ### Clip the joint value in the limited boundary ###
        new_theta = np.hstack((new_theta[0:4].clip(np.radians(-85),np.radians(85)),new_theta[4:7].clip(np.radians(-170),np.radians(170))))
        if arm == "left_arm":
            new_theta[1] = new_theta[1].clip(np.radians(68),np.radians(90))
        else:
            new_theta[1] = new_theta[1].clip(np.radians(-90),np.radians(-68))
        ####################################################
        
        d_err = target_pos - fwd_kinematics(new_theta,arm) # d_err is error from [x,y,z,roll,pitch,yaw]
        
        ### Momentum method ###
        velocity = alpha * ( velocity + d_theta.copy() )
        Momentum = (new_theta + velocity)
        Momentum = np.hstack((Momentum[0:4].clip(np.radians(-85),np.radians(85)),Momentum[4:7].clip(np.radians(-170),np.radians(170))))
        
        if np.linalg.norm(target_pos - fwd_kinematics(Momentum,arm)) < np.linalg.norm(d_err): # Apply momentum if momentum perform better
            new_theta = Momentum
        else: # Reset momentum to now change value If it's not better and don't apply momentum to new_theta
            velocity = d_theta.copy()
        #######################
        count = count + 1
        if (count > 5000 ):
            break
    return new_theta , count , np.linalg.norm(d_err)


def steering_error(steering_command,current_x_pos,x_start_pos=0.41,y_start_pos=0.248):
    acutual_theta = np.degrees(np.arcsin((current_x_pos-x_start_pos)/y_start_pos))
    _steering_error = abs(abs(steering_command) - abs(acutual_theta))
    if steering_command > 0:
        return -_steering_error
    else:
        return _steering_error




# fig = plt.figure()
# ax = fig.gca(projection='3d')

# left_forward_kinematics(ax,[0 for i in range(7)])
# right_forward_kinematics(ax,[0 for i in range(7)])

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# x_major_locator = MultipleLocator(0.1)
# ax.xaxis.set_major_locator(x_major_locator)
# ax.yaxis.set_major_locator(x_major_locator)
# ax.zaxis.set_major_locator(x_major_locator)

# ax.legend()

# plt.show()

if __name__ == "__main__":
    ACCEPTANCE = 0.001
    rotation_angle = np.linspace(-15,15,4*28+1)
    # rotation_angle = np.array([-18,18])
    left_steer_theta =  np.array([-0.19468600572796468, 1.0450682422271944, 0.5306074845300532 ,0.29709705728495717 , 2.6494854330479285, 0.11471756843258163, -0.8633519169683135])
    right_steer_theta = np.array([ 0.22771684480929189, -0.8913385651906776,-0.2836369083505996, -0.5661194916459493, -1.8383287021792807, -0.4873243771690525, 1.1503494323915078])
    
    sim_error_record = []
    sim_time_cost = []
    manipulator = Manipulator_Net(n_feature=2, n_hidden1=32,n_hidden2=7, n_output=14)     # define the network
    model_path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config/NN/123/NN_123.pth"
    manipulator.load_state_dict(torch.load(model_path))
    
    for i in rotation_angle :
        print("---------- {} deg ----------".format(i))
        start_time = time.time()
        ########## NN #########
        # left_target_pos , right_target_pos = target_pos(i,22)
        target_theta = manipulator(torch.Tensor([i,22])).detach().numpy()
        # left_current_pose  = fwd_kinematics(target_theta[0:7],"left_arm")
        # right_current_pose = fwd_kinematics(target_theta[7:],"right_arm")
        
        
        # print("right_arm :\nx,y,z",right_current_pose[0:3])
        # print("roll,pitch,yaw:",np.degrees(right_current_pose[3:7]))
        # print("target",right_target_pos[0:3],"\n",np.degrees(right_target_pos[3:7]))
        # # print("Error:",right_error)
        # print("theta : ",np.degrees(target_theta[7:]))
        # print()
        # print("left_arm  :\nx,y,z",left_current_pose[0:3])
        # print("roll,pitch,yaw:",np.degrees(left_current_pose[3:7]))
        # print("target",left_target_pos[0:3],"\n",np.degrees(left_target_pos[3:7]))
        # # print("Error:",left_error)
        # print("theta : ",np.degrees(target_theta[0:7]))
        
        print()
        # print("iteration: ",right_count," (right) + ",left_count," (left) = ",right_count+left_count)
        end_time = time.time()
        print("Computation time : ",end_time-start_time)
        sim_time_cost.append(end_time-start_time)
        ########## NN #########
        

        # ######################## PSO ######################################
        # # left_target_pos , right_target_pos = [0.5,0.4,0.8,np.rasdians(15),np.radians(-6),np.radians(-10)],[0.4,-0.2,0.9,np.radians(10),np.radians(10),np.radians(5)] #target_pos(i)
        # left_target_pos , right_target_pos = target_pos(i,22)

        # left_target_theta , left_count , left_error  = IK(left_steer_theta,left_target_pos,"left_arm")
        # right_target_theta, right_count, right_error = IK(right_steer_theta,right_target_pos,"right_arm")
        
        # right_current_pose = fwd_kinematics(right_target_theta,"right_arm")
        # left_current_pose  = fwd_kinematics(left_target_theta,"left_arm")

        # # joint.append(np.hstack((left_target_theta,right_target_theta)))
        
        # print("right_arm :\nx,y,z",right_current_pose[0:3])
        # print("roll,pitch,yaw:",np.degrees(right_current_pose[3:7]))
        # print("target",right_target_pos[0:3],"\n",np.degrees(right_target_pos[3:7]))
        # print("Error:",right_error)
        # print("theta : ",np.degrees(right_target_theta))
        # print()
        # print("left_arm  :\nx,y,z",left_current_pose[0:3])
        # print("roll,pitch,yaw:",np.degrees(left_current_pose[3:7]))
        # print("target",left_target_pos[0:3],"\n",np.degrees(left_target_pos[3:7]))
        # print("Error:",left_error)
        # print("theta : ",np.degrees(left_target_theta))
        
        # print()
        # print("iteration: ",right_count," (right) + ",left_count," (left) = ",right_count+left_count)
        # end_time = time.time()
        # print("Computation time : ",end_time-start_time)
        # ######################## PSO ######################################

        # _steering_error = (np.arcsin((left_current_pose[0] - left_target_pos[0])/0.248))
        # print(abs(i))
        # _steering_error = abs( abs(i) - abs(acutual_theta) )
        # sim_error_record.append(_steering_error)
        # print("_steering_error",_steering_error)
        # if right_error > ACCEPTANCE or left_error > ACCEPTANCE :
        #     print("IK Fail !")
        #     print("IK Fail !")
        #     print("IK Fail !")
        #     sys.exit()
        #     break
    print("----------------------------------------------")
    # print(rotation_angle)
    # print("Max steering Eroor (deg) : ",np.max(np.abs(sim_error_record)))
    # print("Max steering Eroor (deg) : ",np.min(sim_error_record))
    
    # sim_error_record = np.array(sim_error_record)
    sim_time_cost = 1000*np.array(sim_time_cost)/2

    fig = plt.figure( figsize=(15,10) )
    ax = fig.add_subplot(111)
    
    # ax.plot(rotation_angle[:],(sim_error_record[:]),label='Steering_Error',color="r")
    # ax.set_xlabel('Steering_Command (degrees)', fontsize=8)
    # ax.set_ylabel('Steering_Error (degrees)', fontsize=8)
    
    ax.plot(np.array([i for i in range(len(sim_time_cost[:]))]),(sim_time_cost[:]),label='Computation Time',color="r")
    ax.set_xlabel('Steering_Command (degrees)', fontsize=8)
    ax.set_ylabel('Computation_time (ms)', fontsize=8)
    print("Max_sim_time_cost : ",np.max(sim_time_cost))
    print("Av.sim_time_cost : " ,np.mean(sim_time_cost))
    # ss = time.time()
    # time.sleep(1)
    # print(time.time()-ss)
    plt.show()


  
    # # LEFT_SHOULDER_LATERAL, LEFT_SHOULDER_FRONTAL,LEFT_SHOULDER_EXTEND ,LEFT_ELBOW_LATERAL,LEFT_ROLL,LEFT_PITCH = range(6)

    # # theta = np.radians([0,0,0,0,0,0,0])
    # # theta = sh_p1 , sh_r , sh_p2 , el_y , wr_r , wr_y , wr_p
    # left_arm_joint = ["l_arm_sh_p1","l_arm_sh_r","l_arm_sh_p2","l_arm_el_y","l_arm_wr_r","l_arm_wr_y","l_arm_wr_p"]
    # # left_arm init_pose
    # # current_theta = [0.5690188117307651,1.031862504786016,-0.11520106285198572,0.49420573480211694,1.0410798907943501,-0.15742867185722087,-0.08522666112220723]
    # current_theta = [0 for i in range(7)]
    # current_pos = fwd_kinematics(current_theta,"left_arm")
    
    # print("left current_pos : \nx,y,z",current_pos[0:3])
    # print("r,p,y",np.degrees(current_pos[3:7]))
    
    # # right_arm init_pose
    # # right_ini_pose = [ -0.5690178546671101, -1.0318633578583105,  0.11520119687434871, ,-0.4942065736290324, -1.04107929650102, 0.1574285966661595 ,-0.08511809326943176]
    # right_ini_pose = [0 for i in range(7)]
    # # left_steering_pose
    # # left_target_theta = [-0.19468623129277507,1.0450681829388877,0.5306073811907908,0.29709720143345564,2.649485610218731,0.11471733970327769,-0.8633519175465221]
    # # left_target_theta = current_theta
    
    # # l_target = fwd_kinematics(left_target_theta,"left_arm")
    # # l_target = fwd_kinematics([0,0,0,0,0,0,0],"left_arm")
    # l_target = [0.32,0.32,0.82,np.radians(0),np.radians(0),np.radians(0)]
    
    # print("left target : \nx,y,z",l_target[0:3])
    # print("r,p,y",np.degrees(l_target[3:7]))
    # # l_target = np.array([0.32782789 ,0.29997311 ,0.78474625,np.radians(90.91553324e-02) , np.radians(10.79511987e+00) , np.radians(8.61942889e-03)])
    
    # start_time = time.time()
    # target_theta,count,err = IK(current_theta,l_target,"left_arm")
    # ik_end = fwd_kinematics(target_theta,"left_arm")
    # end_time = time.time()
    
    # print("IK : \nx,y,z",ik_end[0:3])
    # print("r,p,y",np.degrees(ik_end[3:7]))
    # print("count:",count)
    # print("err:",err)
    # print("target_theta",target_theta)
    # # print("left_target_theta",np.round(left_target_theta,7))
    
    # print("computation time :" , end_time-start_time )
