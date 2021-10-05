#! /usr/bin/env python3

import rospy
from pioneer_kinematics.kinematics import Kinematics
from pioneer_motor.motor import Motor
from thormang3_manipulation_module_msgs.msg import KinematicsPose
from robotis_controller_msgs.msg import JointCtrlModule, StatusMsg
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu

from time import sleep,time
import numpy as np
import sys
import os
import random
import matplotlib.pyplot as plt

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
    def __init__(self):
        
        ###  Parameters  ###
        # X (forward) front of robot is positive
        # Y (sideways) left of robot is positive
        # Z (up-down) up of robot is positive
        self.x_ini = 0.34
        self.x_start_pos = 0.508
        self.x_left_offset = 0.012
                
        self.y_spacing = 0.235
        self.z_height = 0.81# 0.865
        
        self.left_pitch_offset  = -5
        self.left_height_offset = 0.00
        self.left_space_offset  = -0.00
        
        self.center_offset = 0.1
        self.sterring_bar_angle = 30
        self.yaw_offset         = 15
        
        self.rotation_angle = 0.0   # Around mid_point, In degrees
        
        self.right_start_pitch = 18.0
        self.right_pitch = self.right_start_pitch     # In degrees
        
        self.roll = 90 # In degrees (for both arms)
        self.right_roll_offset = -7
        
        self.mid_point = np.zeros([3])

        self.mid_point[0] = self.x_ini
        self.mid_point[1] = 0.0
        self.mid_point[2] = self.z_height

        self.gogoro_state = np.zeros(9)
        
        ## PID paramaters ##
        self.P_gain = 6
        self.I_gain = 0.0008
        self.D_gain = 2
        
        self.refer = 0
        self.i_accum = 0
        self.speed_x = 0
        self.last_error = None
        
        ### Load position_npy ###
        position_path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config"
        
        self.right_position = np.load(os.path.join(position_path,"right_non_gasing_position.npy"))
        self.left_position  = np.load(os.path.join(position_path,"left_non_gasing_position.npy"))
        
        print("Right_Position",self.right_position)
        print("Leftt_Position",self.left_position)
        
        # laft_arm then right_arm
        self.arm_joint = ["l_arm_el_y","l_arm_grip","l_arm_sh_p1","l_arm_sh_p2","l_arm_sh_r","l_arm_wr_p","l_arm_wr_r","l_arm_wr_y",
                          "r_arm_el_y","r_arm_grip","r_arm_sh_p1","r_arm_sh_p2","r_arm_sh_r","r_arm_wr_p","r_arm_wr_r","r_arm_wr_y"]
        
        #########################

        # Kinematics control object
        self.kinematics = Kinematics()
        self.kinematics.publisher_(self.kinematics.module_control_pub, "manipulation_module", latch=True)  

        # Send `ini_pose`
        # log(self.TAG, "Moving to ini pose...")
        # self.kinematics.publisher_(self.kinematics.send_ini_pose_msg_pub, "ini_pose", latch=True)
        # sleep(3)
        # input("FINISH INIT")

        # Set ready position
        # self.ReadyPos()
        
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
        
        rospy.Subscriber("/robotis/sensor/imu/imu",
                         Imu, self._imu_callback)

        rospy.Subscriber("/robotis/present_joint_states",
                    JointState, self._check_move)
        
        self.mutex  = threading.Lock()
        
    ### Ready Pos after FINISH INIT POS  ###
    def ReadyPos(self):
        log(self.TAG, "Ready_for_Grabbing_Position")
        
        self.kinematics.set_kinematics_pose("left_arm" , 3.0, **{ 'x': self.x_ini, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.z_height + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 3.0, **{ 'x': self.x_ini, 'y':  self.mid_point[1]-self.y_spacing, 'z': self.z_height, 'roll': -90.00, 'pitch': self.right_start_pitch, 'yaw': -self.yaw_offset })
        sleep(1.5)
        
    ########################################
    ###     Modify Pos after Ready Pos   ###
    def _x_fornt_cb(self,data):
        self.x_start_pos = data.data
        self.mid_point[0] = self.x_start_pos 
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -90.00, 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Moving X_fornt to {:.3f}".format(self.mid_point[0]))
        
    def _y_spacing_cb(self, data):
        log(self.TAG, "new `y_spacing`: {0}".format(data.data))
        self.y_spacing = data.data
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -90.00, 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Moving Y_Spacing to {:.3f}",format(self.y_spacing))
        
    def _z_height_cb(self,data):
        self.mid_point[2] = data.data 
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -90.00, 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Moving Z_height to {:.3f}".format(self.mid_point[2]))

    def _left_height_offset_cb(self,data):
        self.left_height_offset = data.data
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -90.00, 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        log(self.TAG,"Change Left_Arm_Offset to {:.3f}".format(self.left_height_offset))
        
    def _right_gripper_pitch_cb(self, data):
        # log(self.TAG, "new `right_pitch`: {0}".format(data.data))
        self.right_pitch = data.data
        self.kinematics.set_kinematics_pose("left_arm" , 1.0, **{ 'x': self.mid_point[0], 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 1.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -90.00, 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
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
        if self.rotation_angle >= 14:
            self.rotation_angle = 13.99
        elif self.rotation_angle <= -14:
            self.rotation_angle = -13.99
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
        self.mid_point[1] = self.center_offset * np.sin(np.radians(self.rotation_angle))     # center_offset the center from the steering bar,which is 0.07 m
        self.mid_point[2] = self.z_height - 0.07*np.sin(np.radians(self.sterring_bar_angle))* (1 - np.cos(np.radians(self.rotation_angle)) )
        
        l_x = self.mid_point[0] + self.y_spacing * np.sin(np.radians(self.rotation_angle))   # and use geometry find the kinematic path, which is a ellipse
        l_y = self.mid_point[1] + (self.y_spacing+self.left_space_offset) * np.cos(np.radians(self.rotation_angle))
        l_z = self.mid_point[2] + self.y_spacing * np.cos(np.radians(90 - self.rotation_angle)) * np.sin(np.radians(self.sterring_bar_angle))
        
        r_x = self.mid_point[0] - self.y_spacing * np.sin(np.radians(self.rotation_angle))   # and use geometry find the kinematic path, which is a ellipse
        r_y = self.mid_point[1] - self.y_spacing * np.cos(np.radians(self.rotation_angle))
        r_z = self.mid_point[2] + self.y_spacing * np.cos(np.radians(90 + self.rotation_angle)) * np.sin(np.radians(self.sterring_bar_angle))
        
        # Send new positions
        self.kinematics.set_kinematics_pose("left_arm", time,
                **{'x': l_x, 'y': l_y, 'z': l_z + self.left_height_offset,
                   'roll': self.roll, 'pitch': self.left_pitch_offset, 'yaw': -self.rotation_angle + self.yaw_offset})
        # Note the mirrored signal for the roll in the right arm
        self.kinematics.set_kinematics_pose("right_arm", time,
                **{'x': r_x, 'y': r_y, 'z': r_z,
                   'roll': -self.roll, 'pitch': self.right_pitch, 'yaw': -(self.rotation_angle+self.yaw_offset)})
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
            roll = 0

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
    
    def sample_position_control(self,predict):
        
        action =  np.degrees(predict)
        if action >= 14:
            action = 13.99
        elif action <= -14:
            action = -13.99
                    
        ## pub command ##
        joint           =   JointState()
        joint.name      =   self.arm_joint

        self.mutex.acquire()
        idx = int(action // 2)

        m = action % 2
        n = 2 - m

        left_goal   = ( m*self.left_position[7+idx+1] + n*self.left_position[7+idx] ) / 2
        right_goal  = ( m*self.right_position[7+idx+1] + n*self.right_position[7+idx] ) / 2
        arm_goal    = np.hstack([left_goal,right_goal])
        
        joint.position  =   arm_goal
        self.mutex.release()
        
        joint.velocity  =   [ 0 for _ in self.arm_joint ]
        joint.effort    =   [ 0 for _ in self.arm_joint ]

        self.kinematics.publisher_(self.kinematics.set_joint_pub, joint, latch=False)      

        
        # self.gogoro_state[8] = predict
                

    def _check_move(self,data):
        self.joint_name = data.name
        self.joint_position = data.position
        
        self.left_check = self.joint_position[2:10]
        self.right_check = self.joint_position[16:24]

if __name__ == "__main__":
    _node_name = 'Balance_Agent'
    rospy.init_node(_node_name, anonymous=True)
    rospy.loginfo('{0} is up!'.format(_node_name))

    ### Load PPO Agent ###
    # agent = ActorCritic()    
    # model_path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/scripts/model_30000_20.pt"
    # agent.load_state_dict(torch.load(model_path))
    ######################
    
    steering = Thormang3Steering()
    input("FINISH ReadyPos, Press to Init Gripper")
    
    gripper = Gripper()
    
    ### init ####
    # gripper.setGrippers(left=0, right=0)
    # sleep(1.5)
    # input("FINISH Open Gripper, Press to Steering Pos")
    
    # steering.Go_Steering_pos()
    # input("Ready to Start! Press will Grib tight steering bar .")
    
    # gripper.setGrippers(left=0.5, right=0.5)
    # sleep(1.5)
    
    # input("Gasing .")
    # steering.setRightGripperPitch(0)
    ########
    
    action_times = 10
    
    input("START!!")
        
    steering.kinematics.publisher_(steering.kinematics.module_control_pub, "none", latch=True) 
    steering.sample_position_control(0)
    input("START!")
    
    last_value = 0
    x = []
    y = []
    
    for i in range(300):
        
        # predict_angle = agent.act_inference(torch.Tensor(steering.gogoro_state)).item() * np.radians(15)
        # steering.sample_position_control(predict_angle)
        
        # roll_state = np.degrees(steering.gogoro_state[0])
        rand_value = 28*np.random.rand()-14
        steering.sample_position_control(rand_value)
        sleep(1/action_times)
        
        delt = rand_value - last_value     
        last_value = rand_value
        
        x.append(delt)
        y.append(steering.gogoro_state[4])

        # for test_angle in range(5):
        #     # pub_test_angle = -15 + random.randint(0, 15)
        #     pub_test_angle = 2* (7 - random.randint(0,14))

        #     # if test_angle % 2 == 0:
        #     #     pub_test_angle = - pub_test_angle
        #     # print(pub_test_angle)
        #     steering.sample_position_control(pub_test_angle)
        #     sleep(1/action_times)
        #     # input("pub_test_angle : {}".format(pub_test_angle))
        
        # print("predict_angle",predict_angle)
        # print("curr roll",np.degrees(steering.gogoro_state[0]))
        # sleep(1/action_times)
                  
        # break
    x = np.array(x)
    y = np.array(y)
    
    PATH = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config"
    
    np.save(os.path.join(PATH, 'modify_x'),  x)
    np.save(os.path.join(PATH, 'modify_y'),  y)
    
    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca()
    ax = fig.add_subplot(111)
    dot, = ax.plot(x, y, color='blue', marker='o', markersize=4, markeredgecolor='black', linestyle='')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    plt.show()
    
    print("Finished.")
    rospy.spin()
    