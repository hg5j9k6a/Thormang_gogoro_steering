#! /usr/bin/env python3

from utils import *
from PSO_IK import *

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
from pioneer_kinematics.kinematics import Kinematics
from sensor_msgs.msg import JointState
from std_msgs.msg import String,Float64,Float32MultiArray,Bool


class Gogoro_IK:
    def __init__(self):
        self.set_joint_pub      = rospy.Publisher('/robotis/set_joint_states',      JointState, queue_size=10)
        self.module_control_pub = rospy.Publisher('/robotis/enable_ctrl_module',    String,     queue_size=10)
        
        
        self.pub_rate       = rospy.Rate(10)
        
        for i in range(4):
            self.module_control_pub.publish("none")
            self.pub_rate.sleep()
            
        rospy.Subscriber("/robotis/present_joint_states",
                         JointState, self._current_pos_callback)

        rospy.Subscriber("/gogoro/ik",
                         JointState, self.IK_func)
        rospy.Subscriber("/gogoro/current_post",
                         String, self.get_IK_pos)

    def get_IK_pos(self,data):
        arm = data.data
        
        left_currents_theta = self.left_arm_joint.copy()
        right_current_theta = self.right_arm_joint.copy()
        
        if arm == "left_arm":
            current_pose = fwd_kinematics(left_currents_theta,"leftt_arm")
        elif arm == "right_arm":
            current_pose = fwd_kinematics(right_current_theta,"right_arm")
        
        print()
        print("Arm :")
        print("x,y,z",current_pose[0:3])
        print("roll,pitch,yaw:",np.degrees(current_pose[3:7]))
        print()
    
    def IK_func(self,data,steering_bool=False):
        
        arm = data.name
        target_pos = data.position
        print(arm[0])
        print(target_pos)
        start_time = time.time()
        
        
        left_arm_joint = ["l_arm_sh_p1","l_arm_sh_r","l_arm_sh_p2","l_arm_el_y","l_arm_wr_r","l_arm_wr_y","l_arm_wr_p"]
        right_arm_joint = ["r_arm_sh_p1","r_arm_sh_r","r_arm_sh_p2","r_arm_el_y","r_arm_wr_r","r_arm_wr_y","r_arm_wr_p"]
        
        left_currents_theta = self.left_arm_joint.copy()
        right_current_theta = self.right_arm_joint.copy()
        
        joint           =   JointState()
        
        if arm[0] == "left_arm":
            joint.name = left_arm_joint
            left_target_theta , left_count , left_error  = IK(left_currents_theta,target_pos,"left_arm")
            joint.position = left_target_theta
            
            count_iteration = left_count
            error_norm = left_error
        elif arm[0] == "right_arm":
            joint.name = right_arm_joint
            right_target_theta, right_count, right_error = IK(right_current_theta,target_pos,"right_arm")
            joint.position = right_target_theta
            
            count_iteration = right_count
            error_norm = right_error
        elif arm[0] == "both_arm":
            joint.name = np.hstack((left_arm_joint,right_arm_joint))

            left_target_pos = target_pos[0:6]
            right_target_pos = target_pos[6:12]

            left_target_theta , left_count , left_error  = IK(left_currents_theta,left_target_pos,"left_arm")
            right_target_theta, right_count, right_error = IK(right_current_theta,right_target_pos,"right_arm")
            target_theta = np.hstack((left_target_theta,right_target_theta))

            joint.position  =   target_theta
            
            count_iteration = right_count
            error_norm = np.linalg.norm(left_error + right_error)
        if steering_bool:
            joint.velocity  =   [ 0 for _ in left_arm_joint ]
            joint.effort    =   [ 0 for _ in left_arm_joint ]
        else:
            joint.velocity  =   [ 0 for _ in left_arm_joint ]
            joint.effort    =   [ 0 for _ in left_arm_joint ]
            
        self.set_joint_pub.publish(joint)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        print("Computation_time : ",round(computation_time,4))
        print("Count_iteration  : ",count_iteration)
        print("Error_norm       : ",error_norm)


    def _current_pos_callback(self,data):
        
        jp = data.position
        # theta = sh_p1 , sh_r , sh_p2 , el_y , wr_r , wr_y , wr_p

        self.left_arm_joint = np.array([jp[4],jp[6],jp[5],jp[2],jp[8],jp[9],jp[7]])
        self.right_arm_joint = np.array([jp[18],jp[20],jp[19],jp[16],jp[22],jp[23],jp[21]])
    
        #   - head_p        0
        #   - head_y        1
        #   - l_arm_el_y    2
        #   - l_arm_grip    3
        #   - l_arm_sh_p1   4
        #   - l_arm_sh_p2   5
        #   - l_arm_sh_r    6
        #   - l_arm_wr_p    7
        #   - l_arm_wr_r    8
        #   - l_arm_wr_y    9
        
        #   - r_arm_el_y    16
        #   - r_arm_grip    17
        #   - r_arm_sh_p1   18
        #   - r_arm_sh_p2   19
        #   - r_arm_sh_r    20
        #   - r_arm_wr_p    21
        #   - r_arm_wr_r    22
        #   - r_arm_wr_y    23
        
        #   - torso_y       30    

if __name__ == "__main__":
    
    _node_name = 'IK_Node'
    rospy.init_node(_node_name, anonymous=True)
    rospy.loginfo('{0} is up!'.format(_node_name))
    
    gogoro_IK = Gogoro_IK()

    gogoro_ik_pub = rospy.Publisher("/gogoro/ik",JointState, queue_size=10)

    time.sleep(0.1)

    msg = JointState()
    
    # msg.name = "left_arm" "right_arm" "both_arm"
    # msg.position = x,y,z,roll,pitch,yaw units are meters to x y z  and radians to roll picth yaw
    
    # msg.name = ["left_arm"]
    # msg.position = [0.32,0.32,0.82,np.radians(0),np.radians(0),np.radians(0)]
    
    # msg.name = ["right_arm"]
    # msg.position = [0.32,-0.32,0.82,np.radians(0),np.radians(0),np.radians(0)]
    
    # gogoro_ik_pub.publish(msg)
        
    # print("Finish")
    
    rospy.spin()
    