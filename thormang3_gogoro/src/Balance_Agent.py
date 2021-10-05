#! /usr/bin/env python3

import rospy
import rospkg
from pioneer_kinematics.kinematics import Kinematics
from pioneer_motor.motor import Motor
from thormang3_manipulation_module_msgs.msg import KinematicsPose
from robotis_controller_msgs.msg import JointCtrlModule, StatusMsg
from sensor_msgs.msg import JointState,MagneticField
from std_msgs.msg import Float64,Float32MultiArray,Bool
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu

from time import sleep,time
import numpy as np
import sys
import os
import random
import math
# from PCL_GRAB import grab_position
from PPO import ActorCritic
from gogoro_env import GogoroEnv
from utils import *
from PSO_IK import *

import torch
import torch.nn as nn

import curses
import threading


Gasing_time = 3
Hold_gasing = 1.2 # hold steering to 0 deg and gas 1 second
Duration_time = 10

TILE_REFER = 0  # degrees
# Refer_offset = -3 #deg

def log(tag, msg):
    rospy.loginfo(tag + ': ' + msg)
    
class Gripper:
    TAG = '[Gripper]'

    MIN_LIMIT = 0.0
    MAX_LIMIT = 1.1
    
    def __init__(self):
        self.left_name = 'l_arm_grip'
        self.right_name = 'r_arm_grip'

        # Joint state publisher
        self.jointStatesPublisher = rospy.Publisher('/robotis/set_joint_states', 
                JointState, queue_size=1)
        self.jointModulePublisher = rospy.Publisher('/robotis/set_joint_ctrl_modules', 
                JointCtrlModule, queue_size=1)

        # Set grippers control module to none
        self._setGrippersNoneModule()

        # Also setup a subscriber and callback for easy debugging
        rospy.Subscriber('/thormang3_gogoro/grippers', Float64, 
                self._grippers_cb)

    def _grippers_cb(self, data):
        self.setGrippers(left=data.data, right=data.data)

    def _setGrippersNoneModule(self):
        msg = JointCtrlModule()
        msg.joint_name = [self.left_name, self.right_name]
        msg.module_name = ['none', 'none']

        self.jointModulePublisher.publish(msg)

    def setGrippers(self, left, right):
        ''' Set grippers as a percentage of maximum and minimum opening

        '''
        if left < 0.0 or left > 1.0 or \
           right < 0.0 or right > 1.0:
            log(self.TAG, "value for `setGrippers` outside of range. [0-1]")

        # Convert percentage to range, take the easy way since `MIN_LIMIT` is 0.0
        right_val = right * self.MAX_LIMIT
        left_val = left * self.MAX_LIMIT
        
        jointStateMsg = JointState()
        jointStateMsg.name = [self.left_name, self.right_name]
        jointStateMsg.position = [left_val, right_val]

        self.jointStatesPublisher.publish(jointStateMsg)
        
        
class Thormang3Steering:
    TAG = '[Thormang3Steering]' 
    def __init__(self,tile_refer = 0,init_bool=True):
        
        ###  Parameters  ###
        # X (forward) front of robot is positive
        # Y (sideways) left of robot is positive
        # Z (up-down) up of robot is positive
        self.x_ini = 0.34
        self.x_start_pos = 0.41
        self.x_left_offset = 0.02
                
        self.y_spacing = 0.248
        self.z_height = 0.872# 0.865
        
        self.left_pitch_offset  = -5
        self.left_height_offset = -0.02
        self.left_space_offset  = -0.035
        
        self.center_offset = 0.1
        self.sterring_bar_angle = 90 - 63.435
        self.yaw_offset         = 15
        
        self.rotation_angle = 0.0   # Around mid_point, In degrees
        
        self.right_start_pitch = 22.0
        self.right_pitch = self.right_start_pitch     # In degrees
        
        self.roll = 90 # In degrees (for both arms)
        self.right_roll_offset = -12
        
        self.mid_point = np.zeros([3])

        self.mid_point[0] = self.x_ini
        self.mid_point[1] = 0.0
        self.mid_point[2] = self.z_height

        self.gogoro_state = np.zeros(9)
        self.gogoro_yaw = 0
        
        
        ## PID paramaters ##
        self.P_gain = 1
        self.I_gain = 0.001
        self.D_gain = 4
        
        # self.refer = np.radians(refer)
        self.i_accum = 0
        self.speed_x = 0
        self.last_error = None
        
        ## Tile ##
        self.tile_P = 0.2
        self.tile_I = 0.001
        self.tile_D = 0.25

        self.tile_refer = np.radians(tile_refer)
        self.tile_last_error = None
        self.tile_i_accum = 0
        
        self.action_angle = 0
    
        self.dt = None
        self.break_signal = False
        self.start_signal = False
        
        ### Load position_npy ###
        position_path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config"
        
        self.right_non_position = np.load(os.path.join(position_path,"right_non_gasing_position.npy"))
        self.left_non_position  = np.load(os.path.join(position_path,"left_non_gasing_position.npy"))
        
        self.right_gasing_position = np.load(os.path.join(position_path,"right_gasing_position.npy"))
        self.left_gasing_position  = np.load(os.path.join(position_path,"left_gasing_position.npy"))
        
        self.right_break_position  = np.load(os.path.join(position_path,"right_break.npy"))
        self.left_break_position   = np.load(os.path.join(position_path,"left_break.npy"))
        
        # laft_arm then right_arm
        self.arm_joint = ["l_arm_el_y","l_arm_grip","l_arm_sh_p1","l_arm_sh_p2","l_arm_sh_r","l_arm_wr_p","l_arm_wr_r","l_arm_wr_y",
                          "r_arm_el_y","r_arm_grip","r_arm_sh_p1","r_arm_sh_p2","r_arm_sh_r","r_arm_wr_p","r_arm_wr_r","r_arm_wr_y"]
        
        #########################

        # Kinematics control object
        self.kinematics = Kinematics()
        self.kinematics.publisher_(self.kinematics.module_control_pub, "manipulation_module", latch=True)  

        if init_bool:
            # Send `ini_pose`
            log(self.TAG, "Moving to ini pose...")
            self.kinematics.publisher_(self.kinematics.send_ini_pose_msg_pub, "ini_pose", latch=True)
            sleep(3)
            input("FINISH INIT")

            # Set ready position
            self.ReadyPos()
            
        # Present all the params value
        print("x_fornt:{:.3f} y_spacing:{:.3f} z_height:{:.3f} left_arm_offset:{:.3f}".format(self.mid_point[0], self.y_spacing , self.mid_point[2], self.left_height_offset))
        
        self.data_pub = rospy.Publisher('/thormang3_gogoro/steering/data', Float32MultiArray, queue_size = 5)
        
        
        rospy.Subscriber('/thormang3_gogoro/steering/x_front', 
                Float64, self._x_fornt_cb)
        rospy.Subscriber('/thormang3_gogoro/steering/y_spacing', 
                Float64, self._y_spacing_cb)
        rospy.Subscriber('/thormang3_gogoro/steering/z_height', 
                Float64, self._z_height_cb)
        rospy.Subscriber('/thormang3_gogoro/steering/left_height_offset', 
                Float64, self._left_height_offset_cb)
        
        rospy.Subscriber('/thormang3_gogoro/steering/steering_angle', 
                Float64, self._rotation_angle_cb)
        rospy.Subscriber('/thormang3_gogoro/steering/right_pitch', 
                Float64, self._right_gripper_pitch_cb)
        
        rospy.Subscriber("/robotis/sensor/imu/imu",
                         Imu, self._imu_callback)

        rospy.Subscriber("/robotis/present_joint_states",
                    JointState, self._check_move)
        
        rospy.Subscriber("/thormang3_gogoro/steering/break", 
                         Bool, self._break_callback)
        rospy.Subscriber("/thormang3_gogoro/steering/start", 
                         Bool, self._start_callback)
        
        rospy.Subscriber("/imu/magnetic_field", 
                         MagneticField, self._magnetic_field)
        
        self.mutex  = threading.Lock()
        
    ### Ready Pos after FINISH INIT POS  ###
    def ReadyPos(self):
        log(self.TAG, "Ready_for_Grabbing_Position")
        
        self.kinematics.set_kinematics_pose("left_arm" , 3.0, **{ 'x': self.x_ini, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.z_height + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 3.0, **{ 'x': self.x_ini, 'y':  self.mid_point[1]-self.y_spacing, 'z': self.z_height, 'roll': -(90.00+ self.right_roll_offset), 'pitch': self.right_start_pitch, 'yaw': -self.yaw_offset })
        sleep(1.5)
        
    ########################################
    ###     Modify Pos after Ready Pos   ###
    def _x_fornt_cb(self,data):
        self.x_start_pos = data.data
        self.mid_point[0] = self.x_start_pos 
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(90.00+ self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Moving X_fornt to {:.3f}".format(self.mid_point[0]))
        
    def _y_spacing_cb(self, data):
        log(self.TAG, "new `y_spacing`: {0}".format(data.data))
        self.y_spacing = data.data
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(90.00+ self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Moving Y_Spacing to {:.3f}",format(self.y_spacing))
        
    def _z_height_cb(self,data):
        self.mid_point[2] = data.data 
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(90.00+ self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Moving Z_height to {:.3f}".format(self.mid_point[2]))

    def _left_height_offset_cb(self,data):
        self.left_height_offset = data.data
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(90.00+ self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Change Left_Arm_Offset to {:.3f}".format(self.left_height_offset))
        
    def _right_gripper_pitch_cb(self, data):
        # log(self.TAG, "new `right_pitch`: {0}".format(data.data))
        self.right_pitch = data.data
        self.kinematics.set_kinematics_pose("left_arm" , 1.0, **{ 'x': self.mid_point[0], 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 1.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(90.00+ self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Moving Gripper_pitch to {:.3f}".format(self.right_pitch))

    ########################################
    ###         Go to Steering Pos       ###
    def Go_Steering_pos(self):
                
        self.mid_point[0] = self.x_start_pos
        self.mid_point[1] = 0.0
        self.mid_point[2] = self.z_height
        
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0]+self.x_left_offset, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(self.roll + self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        sleep(1.5)
    ########################################
    
    ###         Agent Control _cb        ###
    def _rotation_angle_cb(self, data):
        log(self.TAG, "new `rotation_angle`: {0}".format(data.data))
        self.rotation_angle = data.data
        self.move()
    ########################################
    
    ###           Player Control         ###
    def setRotation(self, rotation_angle):
        self.rotation_angle = rotation_angle
        if self.rotation_angle >= 18:
            self.rotation_angle = 18
        elif self.rotation_angle <= -18:
            self.rotation_angle = -18
        self.gogoro_state[8] = rotation_angle
        self.move(1)

    def setRightGripperPitch(self, pitch):
        self.right_pitch = pitch
        if self.right_pitch >= 25:
            self.right_pitch = 25
        elif self.right_pitch <= -15:
            self.right_pitch = -15
        self.move(1)
    
    ########################################

    def move(self, time=None):
        ''' Using `mid_point` and `y_spacing` will compute the left and right
        arm positions and send the commands

        '''
        if time != None:
            time = time
        else:
            time = 1

        # Compute new position based on parameters
        self.mid_point[0] = self.x_start_pos
        self.mid_point[1] = 0  #self.center_offset * np.sin(np.radians(self.rotation_angle))     # center_offset the center from the steering bar,which is 0.07 m
        self.mid_point[2] = self.z_height - 0.1*np.sin(np.radians(self.sterring_bar_angle))* (1 - np.cos(np.radians(self.rotation_angle)) )
        
        l_x = self.mid_point[0] + self.x_left_offset + self.y_spacing * np.sin(np.radians(self.rotation_angle))   # and use geometry find the kinematic path, which is a ellipse
        l_y = self.mid_point[1] + (self.y_spacing+self.left_space_offset) * ( 1 - 2.1*self.center_offset * abs(np.sin(np.radians(self.rotation_angle))) )
        l_z = self.mid_point[2] + self.y_spacing * np.sin(np.radians(self.extend_angle)) *np.cos(np.radians(90 - self.rotation_angle)) * np.sin(np.radians(self.sterring_bar_angle))
        
        r_x = self.mid_point[0] - self.y_spacing * np.sin(np.radians(self.rotation_angle))   # and use geometry find the kinematic path, which is a ellipse
        r_y = self.mid_point[1] - self.y_spacing * ( 1 - 2.1*self.center_offset * abs(np.sin(np.radians(self.rotation_angle))) )
        r_z = self.mid_point[2] + self.y_spacing * np.sin(np.radians(self.extend_angle))* np.cos(np.radians(90 + self.rotation_angle)) * np.sin(np.radians(self.sterring_bar_angle))
        
        # Send new positions
        self.kinematics.set_kinematics_pose("left_arm", time,
                **{'x': l_x, 'y': l_y, 'z': l_z + self.left_height_offset,
                   'roll': self.roll , 'pitch': self.left_pitch_offset, 'yaw': -1.25*self.rotation_angle + self.yaw_offset})
        # Note the mirrored signal for the roll in the right arm
        self.kinematics.set_kinematics_pose("right_arm", time,
                **{'x': r_x, 'y': r_y, 'z': r_z,
                    'roll': -(self.roll + self.right_roll_offset )+ 0.5*self.rotation_angle, 'pitch': self.right_pitch, 'yaw': -1.25*self.rotation_angle - self.yaw_offset})
    ########################################
        
    def End_Pos(self):
        self.kinematics.set_kinematics_pose("left_arm" , 3.0, **{ 'x': self.x_ini, 'y':   self.mid_point[1]+self.y_spacing, 'z': self.z_height + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': 0.00 })
        self.kinematics.set_kinematics_pose("right_arm" , 3.0, **{ 'x': self.x_ini, 'y':  self.mid_point[1]-self.y_spacing, 'z': self.z_height, 'roll': -90.00, 'pitch': self.right_pitch, 'yaw': 0.00 })
        sleep(1.5)
        
    def _imu_callback(self, msg):
        ### calculate observations, units in radians ###
        
        #print("[IMU_CALLBACK]")
        quaternion = [msg.orientation.x, 
                      msg.orientation.y, 
                      msg.orientation.z,
                      msg.orientation.w]
                                
        euler = quaternion_to_euler(*quaternion)
        # print(euler)
        
        ## unit is radians
        if euler[0] > 0:
            roll = euler[0] - math.pi
        elif euler[0] < 0:
            roll = euler[0] + math.pi
        else :
            roll = euler[0]
            
        # pitch = euler[1]
        # yaw = euler[2]
        
        angvel_x = msg.angular_velocity.x
        angvel_y = msg.angular_velocity.y
        angvel_z = msg.angular_velocity.z
        
        linear_x = msg.linear_acceleration.x
        linear_y = msg.linear_acceleration.y
        linear_z = msg.linear_acceleration.z
        
        self.gogoro_state[0] = -roll
        
        self.gogoro_state[1] = angvel_x
        self.gogoro_state[2] = angvel_y
        self.gogoro_state[3] = angvel_z
        
        self.gogoro_state[4] = linear_x
        self.gogoro_state[5] = linear_y
        self.gogoro_state[6] = linear_z - 9.81
        
        self.gogoro_state[7] = 0.0
        
            
        # self.gogoro_state[8] = curr steering
        
    ########################################
    
    def sample_position_control(self,predict,last_time, refer = 0,gasing_bool= False,get_started = False):

        # if last_time == None:
        #     pass
        # else:
        #     self.dt = time() - last_time
        #     if self.gogoro_state[4] < -1:
        #         self.speed_x += (self.gogoro_state[4] + 0.45)*self.dt
        #     elif self.gogoro_state[4] > -0.3:
        #         self.speed_x += (self.gogoro_state[4] + 0.45)*self.dt

        ## PID Balance ##
        if get_started:
            error = 0
        else:
            error = (predict - refer)
        
        self.i_accum += error
        
        if self.last_error == None:
            error_dot = 0
        else:
            error_dot = error - self.last_error
        
        # P_scale = 12
        # if abs(self.speed_x) < 10 :
        #     P_offset = 9
        # else:
        #     P_offset = 2 * abs(self.speed_x)
        
        P_signal = self.P_gain * error #(self.P_gain * P_scale/P_offset )* error 
        I_signal = self.I_gain * self.i_accum
        D_signal = self.D_gain * error_dot
        
        angle    = P_signal # + I_signal + D_signal
        
        self.last_error = error
        
        # ## Tile ##
        
        # if self.tile_refer > np.radians(180):
        #     self.tile_refer = self.tile_refer - np.radians(360)
        # elif self.tile_refer < np.radians(-180):
        #     self.tile_refer = self.tile_refer + np.radians(360)
        
        # tile_error = self.tile_refer - self.gogoro_yaw  # rad/s
        # if abs(tile_error + np.radians(360)) < abs(tile_error) :
        #     tile_error = tile_error + np.radians(360)
        # tile_error = -tile_error
        
        # # if abs(tile_error) > np.radians(30):
        # #     self.break_signal = True 
            
        # self.tile_i_accum += tile_error
        
        # if self.tile_last_error == None:
        #     tile_error_dot = 0
        # else:
        #     tile_error_dot = tile_error - self.tile_last_error
            
        # tile_P_signal = self.tile_P * tile_error #/ P_offset 
        # tile_I_signal = self.tile_I * self.tile_i_accum
        # tile_D_signal = self.tile_D * tile_error_dot
        
        # self.tile_last_error = tile_error
        
        # tile_angle    = tile_P_signal + tile_I_signal + tile_D_signal
        # tile_angle    = 0 
        
        # action =  np.degrees(angle) # imu use radians!
        action =  angle
        
        if action >= 18:
            action = 17.999
        elif action <= -18:
            action = -17.999
                    
        ## pub command ##
        joint           =   JointState()
        joint.name      =   self.arm_joint

        self.mutex.acquire()
        idx = int(action // 2)

        m = action % 2
        n = 2 - m
        
        if not gasing_bool :
            left_goal   = ( m*self.left_non_position[9+idx+1] + n*self.left_non_position[9+idx] ) / 2
            right_goal  = ( m*self.right_non_position[9+idx+1] + n*self.right_non_position[9+idx] ) / 2
        else:
            left_goal   = ( m*self.left_gasing_position[9+idx+1] + n*self.left_gasing_position[9+idx] ) / 2
            right_goal  = ( m*self.right_gasing_position[9+idx+1] + n*self.right_gasing_position[9+idx] ) / 2
            
        arm_goal    = np.hstack([left_goal,right_goal])
        
        joint.position  =   arm_goal
        self.mutex.release()
        
        joint.velocity  =   [ 0 for _ in self.arm_joint ]
        joint.effort    =   [ 0 for _ in self.arm_joint ]

        self.kinematics.publisher_(self.kinematics.set_joint_pub, joint, latch=False)
        
        self.action_angle = action
        
        gogoro_data = Float32MultiArray()
        gogoro_data.data = [self.speed_x,self.gogoro_state[3],np.degrees(self.gogoro_state[0]),np.degrees(self.gogoro_yaw),self.action_angle]
        self.data_pub.publish(gogoro_data)
        
        # print("P_signal : ",np.degrees(P_signal))
        # print("I_signal : ",np.degrees(I_signal))
        # print("D_signal : ",np.degrees(D_signal))
        # print("Angle(rad/s) : ",angle)
        # print("")
        # print("P_gain : ",(self.P_gain * P_scale/P_offset ))
        # print("speed_x (m/s): ",self.speed_x ) 
        # print("linear_x (rad) : ",self.gogoro_state[4])
        # print("gogoro roll (deg) : ", np.degrees(self.gogoro_state[0]))
        # print("gogoro yaw (deg) : ", np.degrees(self.gogoro_yaw))
        # print("gogoro tile_error (deg) : ", np.degrees(tile_error))
        # print("steering angle (deg) : ", action)
        # print("")
        # print("--------------------\n")

        return action,error
        # self.gogoro_state[8] = predict
    
    def breaking(self):
        right_break = self.right_break_position
        left_break  = self.left_break_position
        
        arm_break   = np.hstack([left_break,right_break])
        
        joint           =   JointState()
        joint.name      =   self.arm_joint
        
        joint.position  =   arm_break
        
        joint.velocity  =   [ 0 for _ in self.arm_joint ]
        joint.effort    =   [ 0 for _ in self.arm_joint ]

        self.kinematics.publisher_(self.kinematics.set_joint_pub, joint, latch=False)
        
    def _magnetic_field(self,msg):
        self.gogoro_yaw = np.radians(math.atan2(msg.magnetic_field.y ,msg.magnetic_field.x) * 180/math.pi + 180)
        
    def _break_callback(self,msg):
        self.break_signal = msg.data
        if self.break_signal : 
            self.breaking()
        
    def _start_callback(self,msg):
        self.start_signal = msg.data
                
    def _check_move(self,data):
        self.joint_name = data.name
        self.joint_position = data.position
        
        self.left_check = self.joint_position[2:10]
        self.right_check = self.joint_position[16:24]


class Manipulator_Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1,n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        x = F.relu(self.hidden2(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
    
if __name__ == "__main__":
    _node_name = 'Balance_Agent'
    rospy.init_node(_node_name, anonymous=True)
    rospy.loginfo('{0} is up!'.format(_node_name))

    ### Load PPO Agent ###
    # agent = ActorCritic()    
    # model_path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/scripts/model_30000_20.pt"
    # agent.load_state_dict(torch.load(model_path))
    ######################
    
    ### Load NN manipulator ###
    manipulator = Manipulator_Net(n_feature=1, n_hidden1=32,n_hidden2=7, n_output=14)     # define the network
    model_path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config/NN/10340000/NN_10340000.pth"
    manipulator.load_state_dict(torch.load(model_path))
    ###########################
    
    steering = Thormang3Steering(tile_refer = TILE_REFER)
    
    input("FINISH ReadyPos, Press to Init Gripper")
    
    gripper = Gripper()
    
    ### init ####
    gripper.setGrippers(left=0, right=0)
    sleep(1.5)
    input("FINISH Open Gripper, Press to Steering Pos")
    
    steering.Go_Steering_pos()
    input("Ready to Start! Press will Grib tight steering bar .")
    
    gripper.setGrippers(left=0.5, right=0.5)
    sleep(1.5)
    
    # input("Gasing .")
    # steering.setRightGripperPitch(0)
    ########
    
    action_times = 10

    # rospack = rospkg.RosPack()
    PKG_PATH = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro" #rospack.get_path('thormang_gogoro')
    EXPERIMENT_NAME = 0
    
    input("Set Refer_offset.")
    Refer_offset = steering.gogoro_state[0]
    print("Refer_offset:",np.degrees(Refer_offset))
    
    input("READY TO START!!")
    steering.kinematics.publisher_(steering.kinematics.module_control_pub, "none", latch=True) 
    steering.sample_position_control(0,None,refer = Refer_offset,get_started=True)
    
    # print("Wait for call")
    # while not(steering.start_signal):
    #     sleep(0.33)
    
    
    ### NN manipulator ###
    input("START!!")
    joint           =   JointState()

    left_arm_joint = ["l_arm_sh_p1","l_arm_sh_r","l_arm_sh_p2","l_arm_el_y","l_arm_wr_r","l_arm_wr_y","l_arm_wr_p"]
    right_arm_joint = ["r_arm_sh_p1","r_arm_sh_r","r_arm_sh_p2","r_arm_el_y","r_arm_wr_r","r_arm_wr_y","r_arm_wr_p"]
    arm_joint = np.hstack((left_arm_joint,right_arm_joint))
    
    joint.name      =   arm_joint
    # joint.position  =   target_theta

    joint.velocity  =   [ 0 for _ in arm_joint ]
    joint.effort    =   [ 0 for _ in arm_joint ]
    
    def _manipulator_callback(data):
        angle = data.data
        print("Recived! target_angle : ", data.data)
        if angle > 18:
            angle = 18
        elif angle < -18:
            angle = -18
        target_theta = manipulator(torch.Tensor([angle]))
        joint.position  =   target_theta.data.numpy()
        steering.kinematics.publisher_(steering.kinematics.set_joint_pub, joint, latch=False)
        
        # for i in range(2):
        #     print(i)
        #     sleep(1)
        
        # print(steering.sample_position_control(angle,None,refer = Refer_offset,gasing_bool= False))

    sub = rospy.Subscriber("/gogoro_manipulator_steering",Float64,_manipulator_callback)
    rospy.spin()
    #######################
    
    
    # last_time = None
    # star_time = time()
    # time_count = 0
    
    # control_record = []
    
    # # Prepare a curses window control
    # stdscr = curses.initscr()
    # curses.noecho()
    # curses.cbreak()
    # stdscr.keypad(True)
    # stdscr.nodelay(True)
    
    # while not rospy.is_shutdown():
        
        ### PPO ###        
        # predict_angle = agent.act_inference(torch.Tensor(steering.gogoro_state)).item() * np.radians(15)
        # steering.sample_position_control(predict_angle)
        
        # roll_state = np.degrees(steering.gogoro_state[0])
        ###########

        ## NN manipulator ##

        # predict_joint = manipulator(torch.Tensor([angle_command]))
        
        # l_t , r_t = target_pos(angle_command)
        # l_p = fwd_kinematics(predict_joint.data.numpy()[0:7],"left_arm")
        # r_p = fwd_kinematics(predict_joint.data.numpy()[7:14],"right_arm")
    
        
        ####################
    #     if time_count < Hold_gasing:
    #         angle,error = steering.sample_position_control(steering.gogoro_state[0],last_time,refer = Refer_offset,gasing_bool= True,get_started=True)
    #     elif time_count < Gasing_time:
    #         angle,error = steering.sample_position_control(steering.gogoro_state[0],last_time,refer = Refer_offset,gasing_bool= True)
    #     else:    
    #         angle,error = steering.sample_position_control(steering.gogoro_state[0],last_time,refer = Refer_offset,gasing_bool= False)
        
    #     control_record.append([angle,np.degrees(error)]) # record with deg
        
    #     sleep(1/action_times)
        
    #     last_time = time()
        
    #     time_count = ( last_time - star_time ) 
        
    #     stdscr.clear()
    #     gogoro_yaw = steering.gogoro_yaw
    #     if gogoro_yaw > np.radians(180):
    #         gogoro_yaw = gogoro_yaw - np.radians(360)
        
    #     stdscr.addstr(f"Gogoro State (right side is positive value) \n")
    #     stdscr.addstr(f"Time    : {time_count:.2f} s \n")
        
    #     stdscr.addstr(f"Speed       : {-steering.speed_x:.2f} m/s \n")
    #     stdscr.addstr(f"Speed       : {-steering.speed_x*3.6:.2f} km/h \n")
    #     stdscr.addstr(f"Yaw_Speed   : {-steering.gogoro_state[3]:.2f} rad/s \n\n")

    #     stdscr.addstr(f"Roll  : {np.degrees(steering.gogoro_state[0]):.2f} deg\n")
    #     stdscr.addstr(f"Yaw   : {np.degrees(gogoro_yaw):.2f} deg\n")
    #     stdscr.addstr(f"Steer : {steering.action_angle:.2f} deg\n\n")

    #     stdscr.addstr(f"PRESS 'q' TO BREAK! \n")

    #     c = stdscr.getch()
        
    #     if c == ord('q') or (time_count > Duration_time) : 
    #         steering.breaking()
    #         break
    #     if (steering.break_signal):
    #         break
        
    # curses.nocbreak()
    # stdscr.keypad(False)
    # curses.echo()
    # curses.endwin()
        
    # control_record = np.array(control_record)

    # PATH = os.path.join(PKG_PATH, f'data/{EXPERIMENT_NAME}')
    # while os.path.exists(PATH):
    #     EXPERIMENT_NAME += 1
    #     PATH = os.path.join(PKG_PATH, f'data/{EXPERIMENT_NAME}')
    # os.mkdir(PATH)
    
    # np.save(os.path.join(PATH, 'control_record'),  control_record)
    
    # print("Finished.")