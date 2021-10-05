#! /usr/bin/env python3

import rospy
from pioneer_kinematics.kinematics import Kinematics
from thormang3_manipulation_module_msgs.msg import KinematicsPose
from robotis_controller_msgs.msg import JointCtrlModule, StatusMsg
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu,JointState

from time import sleep
import numpy as np
import sys
import os

# from PCL_GRAB import grab_position
from PPO import ActorCritic
from gogoro_env import GogoroEnv
import torch
import torch.nn as nn

import curses
import threading
from utils import *

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
    def __init__(self,init_bool = True):
        
        ###  Parameters  ###
        # X (forward) front of robot is positive
        # Y (sideways) left of robot is positive
        # Z (up-down) up of robot is positive
        self.x_ini = 0.34
        self.x_start_pos = 0.41
        self.x_left_offset = 0.02
                
        self.y_spacing = 0.248
        self.z_height = 0.872 # 0.865
        
        self.left_pitch_offset  = -5
        self.left_height_offset = -0.02
        self.left_space_offset  = -0.035
        
        self.center_offset = 0.1
        self.sterring_bar_angle = 90 - 63.435
        self.extend_angle = 18.435

        self.yaw_offset         = 15
        
        self.rotation_angle = 0.0   # Around mid_point, In degrees
        
        self.right_start_pitch = 22.0
        self.right_pitch = self.right_start_pitch     # In degrees
        # self.gasing_offset = 10
        
        self.roll = 90 # In degrees (for both arms)
        self.right_roll_offset = -12
        
        self.mid_point = np.zeros([3])

        self.mid_point[0] = self.x_ini
        self.mid_point[1] = 0.0
        self.mid_point[2] = self.z_height

        self.gogoro_state = np.zeros(9)

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
        
        # rospy.Subscriber("/robotis/sensor/imu/imu",
        #                  Imu, self._imu_callback)
        rospy.Subscriber('/robotis/status', StatusMsg,
                         self._kinematic_status_cb)
        
        rospy.Subscriber("/robotis/present_joint_states",
                         JointState, self._record)
        
    ### Ready Pos after FINISH INIT POS  ###
    def ReadyPos(self):
        log(self.TAG, "Ready_for_Grabbing_Position")
        
        self.kinematics.set_kinematics_pose("left_arm" , 3.0, **{ 'x': self.x_ini, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.z_height + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 3.0, **{ 'x': self.x_ini, 'y':  self.mid_point[1]-self.y_spacing, 'z': self.z_height, 'roll': -(90.00+self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        sleep(1.5)
        
    ########################################
    ###     Modify Pos after Ready Pos   ###
    def _x_fornt_cb(self,data):
        self.x_start_pos = data.data
        self.mid_point[0] = self.x_start_pos 
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0]+self.x_left_offset, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(90.00+self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Moving X_fornt to {:.3f}".format(self.mid_point[0]))
        
    def _y_spacing_cb(self, data):
        log(self.TAG, "new `y_spacing`: {0}".format(data.data))
        self.y_spacing = data.data
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0]+self.x_left_offset, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(90.00+self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Moving Y_Spacing to {:.3f}",format(self.y_spacing))
        
    def _z_height_cb(self,data):
        self.mid_point[2] = data.data 
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0]+self.x_left_offset, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(90.00+self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Moving Z_height to {:.3f}".format(self.mid_point[2]))

    def _left_height_offset_cb(self,data):
        self.left_height_offset = data.data
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0]+self.x_left_offset, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(90.00+self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Change Left_Arm_Offset to {:.3f}".format(self.left_height_offset))
        
    def _right_gripper_pitch_cb(self, data):
        # log(self.TAG, "new `right_pitch`: {0}".format(data.data))
        self.right_pitch = data.data
        self.kinematics.set_kinematics_pose("left_arm" , 1.0, **{ 'x': self.mid_point[0]+self.x_left_offset, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 1.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(90.00+self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
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
        if self.rotation_angle >= 20:
            self.rotation_angle = 20
        elif self.rotation_angle <= -20:
            self.rotation_angle = -20
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
        # if self.rotation_angle >= 12:
        #     self.kinematics.set_kinematics_pose("right_arm", time,
        #             **{'x': r_x, 'y': r_y, 'z': r_z,
        #                'roll': -(self.roll + self.right_roll_offset )+ 0.5*self.rotation_angle, 'pitch': self.right_pitch + 0.22*(self.rotation_angle) , 'yaw': -1.25*self.rotation_angle - self.yaw_offset})
        # elif self.rotation_angle >= 8:
        #     self.kinematics.set_kinematics_pose("right_arm", time,
        #             **{'x': r_x, 'y': r_y, 'z': r_z,
        #                'roll': -(self.roll + self.right_roll_offset )+ 0.5*self.rotation_angle, 'pitch': self.right_pitch + 0.2*(self.rotation_angle) , 'yaw': -1.25*self.rotation_angle - self.yaw_offset})
        # elif self.rotation_angle >= 0:
        #     self.kinematics.set_kinematics_pose("right_arm", time,
        #             **{'x': r_x, 'y': r_y, 'z': r_z,
        #                'roll': -(self.roll + self.right_roll_offset )+ 0.5*self.rotation_angle, 'pitch': self.right_pitch + 0.12*(self.rotation_angle) , 'yaw': -1.25*self.rotation_angle - self.yaw_offset})
        # elif self.rotation_angle >= -10:
        #     self.kinematics.set_kinematics_pose("right_arm", time,
        #             **{'x': r_x, 'y': r_y, 'z': r_z,
        #                'roll': -(self.roll + self.right_roll_offset )+ 0.5*self.rotation_angle, 'pitch': self.right_pitch + 0.05*(self.rotation_angle) , 'yaw': -1.25*self.rotation_angle - self.yaw_offset})
        # else:
        #     self.kinematics.set_kinematics_pose("right_arm", time,
        #             **{'x': r_x, 'y': r_y, 'z': r_z,
        #                'roll': -(self.roll + self.right_roll_offset )+ 0.5*self.rotation_angle, 'pitch': self.right_pitch + 0.05*(self.rotation_angle) , 'yaw': -1.25*self.rotation_angle - self.yaw_offset})
        
            
    def End_Pos(self):
        
        self.kinematics.set_kinematics_pose("left_arm" , 1.0, **{ 'x': self.x_ini - 0.1, 'y':   self.mid_point[1]+0.47, 'z': self.z_height + self.left_height_offset + 0.02, 'roll': self.roll, 'pitch': self.left_pitch_offset, 'yaw': -1.25*self.rotation_angle + self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 1.0, **{ 'x': self.x_ini- 0.1, 'y':  self.mid_point[1] -self.y_spacing - 0.05, 'z': self.z_height + 0.15, 'roll': -(90.00+self.right_roll_offset), 'pitch': self.right_pitch+5, 'yaw': -self.yaw_offset })
        
        sleep(1.5)
        
    def _kinematic_status_cb(self,msg):
        # self.mutex.acquire()
        self.module_name = msg.module_name
        self.status_msg  = msg.status_msg
        if self.status_msg == "End Left Arm Trajectory":
            self.left_ready = True
        elif self.status_msg == "Start Left Arm Trajectory":
            self.left_ready = False
        
        if self.status_msg == "End Right Arm Trajectory":
            self.right_ready = True
        elif self.status_msg == "Start Right Arm Trajectory":
            self.right_ready = False
        
        # self.mutex.release()
    # def _imu_callback(self, msg):
    #     ### calculate observations, units in radians ###
        
    #     #print("[IMU_CALLBACK]")
    #     quaternion = [msg.orientation.x, 
    #                   msg.orientation.y, 
    #                   msg.orientation.z,
    #                   msg.orientation.w]
                                
    #     euler = quaternion_to_euler(*quaternion)
    #     # print(euler)
        
    #     ## unit is radians
    #     if euler[0] > 0:
    #         roll = euler[0] - math.pi
    #     elif euler[0] < 0:
    #         roll = euler[0] + math.pi
    #     else :
    #         roll = 0

    #     angvel_x = msg.angular_velocity.x
    #     angvel_y = msg.angular_velocity.y
    #     angvel_z = msg.angular_velocity.z
        
    #     linear_x = msg.linear_acceleration.x
    #     linear_y = msg.linear_acceleration.y
    #     linear_z = msg.linear_acceleration.z
        
    #     self.gogoro_state[0] = roll
        
    #     self.gogoro_state[1] = angvel_x
    #     self.gogoro_state[2] = angvel_y
    #     self.gogoro_state[3] = angvel_z
        
    #     self.gogoro_state[4] = linear_x
    #     self.gogoro_state[5] = linear_y
    #     self.gogoro_state[6] = linear_z - 9.828
        
    #     self.gogoro_state[7] = 0.0
    #     # self.gogoro_state[8] = curr steering
    
    def _record(self,data):
        self.joint_name = data.name
        self.joint_position = data.position
        self.left_arm_position = self.joint_position[2:10]
        self.right_arm_position = self.joint_position[16:24]
        
        
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
        #   - l_leg_an_p    10
        #   - l_leg_an_r    11
        #   - l_leg_hip_p   12
        #   - l_leg_hip_r   13
        #   - l_leg_hip_y   14
        #   - l_leg_kn_p    15
        #   - r_arm_el_y    16
        #   - r_arm_grip    17
        #   - r_arm_sh_p1   18
        #   - r_arm_sh_p2   19
        #   - r_arm_sh_r    20
        #   - r_arm_wr_p    21
        #   - r_arm_wr_r    22
        #   - r_arm_wr_y    23
        #   - r_leg_an_p    24 
        #   - r_leg_an_r    25
        #   - r_leg_hip_p   26
        #   - r_leg_hip_r   27
        #   - r_leg_hip_y   28
        #   - r_leg_kn_p    29
        #   - torso_y       30

        

if __name__ == "__main__":
    _node_name = 'Balance_Agent'
    rospy.init_node(_node_name, anonymous=True)
    rospy.loginfo('{0} is up!'.format(_node_name))

    steering = Thormang3Steering(init_bool= True)
    input("FINISH ReadyPos, Press to Init Gripper")
    
    gripper = Gripper()
    gripper.setGrippers(left=0.0, right=0.0)
    sleep(1.5)
    input("FINISH Open Gripper, Press to Steering Pos")
    
    steering.Go_Steering_pos()
    input("Ready to Start! Press will Grib tight steering bar .")
    
    gripper.setGrippers(left=0.5, right=0.5)
    sleep(1.5)
    
    # input("Gasing .")
    # steering.setRightGripperPitch(20)
    
    
    
    # angle_list = [-14 , -12 , -10 , -8 , -6 , -4 , -2 , 0 , 2 , 4 , 6 , 8, 10 , 12 , 14]
    # angle_list = [0, -2, -4 , -6, -8 , -10, -12 ,-14 , -12 , -10 , -8 , -6 , -4 , -2 , 0 , 2 , 4 , 6 , 8, 10 , 12 , 14, 12 , 10, 8 , 6, 4, 2, 0 ]
    angle_list = [-18,-16,-14 , -12 , -10 , -8 , -6 , -4 , -2 , 0 , 2 , 4 , 6 , 8, 10 , 12 , 14,16,18]
    
    # record_right_list = np.zeros((len(angle_list),8))
    # record_left_list  = np.zeros((len(angle_list),8))
    
    # record_right_gas  =np.zeros((len(angle_list),8))
    # record_left_gas   =np.zeros((len(angle_list),8))
    
    ############# -14 , -12 , -10 , -8 , -6 , -4 , -2 , 0 , 2 , 4 , 6 , 8, 10 , 12 , 14
    # r_arm_el_y
    # r_arm_grip
    # r_arm_sh_p1
    # r_arm_sh_p2
    # r_arm_sh_r
    # r_arm_wr_p
    # r_arm_wr_r
    # r_arm_wr_y

    
    ############# -14 , -12 , -10 , -8 , -6 , -4 , -2 , 0 , 2 , 4 , 6 , 8, 10 , 12 , 14
    # l_arm_el_y
    # l_arm_grip
    # l_arm_sh_p1
    # l_arm_sh_p2
    # l_arm_sh_r
    # l_arm_wr_p
    # l_arm_wr_r
    # l_arm_wr_y
    
    # input("START Record Non-Gasing Mode!!")
    
    # for i,angle in enumerate(angle_list):
    #     input("angle : {}".format(angle))
    #     steering.setRotation(angle)
    #     sleep(1.2)
    #     for k in range(8):
    #         record_left_list[i,k]  = steering.left_arm_position[k]
    #         record_right_list[i,k] = steering.right_arm_position[k]
        
    #     # sleep(0.5)
    
    # ## save npy file ##
    
    PATH = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config"
    
    # np.save(os.path.join(PATH, 'left_non_gasing_position'),  record_left_list)
    # np.save(os.path.join(PATH, 'right_non_gasing_position'), record_right_list)
    
    # steering.setRotation(0)
    # sleep(2)
        
    # steering.setRightGripperPitch(-12)
    # input("START Record Gasing Mode!!")
    
    # for i,angle in enumerate(angle_list):
    #     input("angle : {}".format(angle))
    #     steering.setRotation(angle)
    #     sleep(1.5)
    #     for k in range(8):
    #         record_left_gas[i,k]  = steering.left_arm_position[k]
    #         record_right_gas[i,k] = steering.right_arm_position[k]
    
    #     # sleep(0.5)
    
    # np.save(os.path.join(PATH, 'left_gasing_position'),  record_left_gas)
    # np.save(os.path.join(PATH, 'right_gasing_position'), record_right_gas)
    
    # steering.setRotation(0)
    
    input("Break!")
    steering.setRightGripperPitch(25)
    sleep(1.5)
    steering.setRotation(0)
    sleep(1.5)
    gripper.setGrippers(left=0.0, right=0.0)
    sleep(1.5)
    steering.End_Pos()
    
    left_break  = np.zeros(8)
    right_break = np.zeros(8)
    input("Record!")
    
    for j in range(8):
        left_break[j]  = steering.left_arm_position[j]
        right_break[j] = steering.right_arm_position[j]
        
    np.save(os.path.join(PATH, 'left_break'),  left_break)
    np.save(os.path.join(PATH, 'right_break'), right_break)

        
    print("Finished.")
    rospy.spin()