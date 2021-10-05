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

import time
import numpy as np
import sys
import os
import random
import math
# from PCL_GRAB import grab_position
# from PPO import ActorCritic
# from gogoro_env import GogoroEnv
from utils import *
from PSO_IK import *
from gripper import *
from Manipulator_NN import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import curses
import threading


Hold_gasing = 3.5 # hold steering to 0 deg and gas 1 second
Duration_time = 8

rospack = rospkg.RosPack()
# PKG_PATH = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro" #rospack.get_path('thormang_gogoro')
PKG_PATH = rospack.get_path('thormang3_gogoro')

EXPERIMENT_TIME = 0
EXPERIMENT_NAME = "0706_PID_13"

TILE_REFER = 0  # degrees
        
class Thormang3Steering:
    TAG = '[Thormang3Steering]' 
    def __init__(self,tile_refer = 0,init_bool=True):
        
        ## PID paramaters ##
        
        # self.P_gain = 0
        # self.I_gain = 0
        # self.D_gain = 0
        
        self.P_gain = 1.2
        self.I_gain = 0.001
        self.D_gain = 2
        
        ### Manipulator Parameters  ###
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
        #######################
        
        ### gogoro state ###  (imu sensor data)
        self.gogoro_state = np.zeros(9)
        self.gogoro_yaw = 0
        self.Refer_offset = 0
        ####################
        
        # self.refer = np.radians(refer)
        self.i_accum = 0
        self.speed_x = 0
        self.last_error = None

        self.action_angle = 0
    
        self.dt = None
        self.break_signal = False
        self.start_signal = False
        
        #########################NN_10340000

        self.senario = ""
        self.NN_time_spent = None

        ### Load NN manipulator ###
        self.manipulator = Manipulator_Net(n_feature=2, n_hidden1=32,n_hidden2=7, n_output=14)     # define the network
        
        model_path = rospack.get_path('thormang3_gogoro') + "/config/NN/60/NN_60.pth"
        # model_path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config/NN/123/NN_123.pth"
        self.manipulator.load_state_dict(torch.load(model_path))
        ###########################

        # Kinematics control object
        self.kinematics = Kinematics()
        for i in range(3):
            self.kinematics.publisher_(self.kinematics.module_control_pub, "manipulation_module", latch=True)  
            time.sleep(0.1)

        self.break_arm_joint = ["l_arm_el_y","l_arm_grip","l_arm_sh_p1","l_arm_sh_p2","l_arm_sh_r","l_arm_wr_p","l_arm_wr_r","l_arm_wr_y",
                                "r_arm_el_y","r_arm_grip","r_arm_sh_p1","r_arm_sh_p2","r_arm_sh_r","r_arm_wr_p","r_arm_wr_r","r_arm_wr_y"]
        
        position_path = rospack.get_path('thormang3_gogoro') + "/config"
        
        # position_path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/config"
        
        self.right_break_position  = np.load(os.path.join(position_path,"right_break.npy"))
        self.left_break_position   = np.load(os.path.join(position_path,"left_break.npy"))
        
        if init_bool:
            # Send `ini_pose`
            log(self.TAG, "Moving to ini pose...")
            self.kinematics.publisher_(self.kinematics.send_ini_pose_msg_pub, "ini_pose", latch=True)
            time.sleep(4)
            # input("FINISH INIT")

            # Set ready position
            self.ReadyPos()
            
        # Present all the params value
        print("x_fornt:{:.3f} y_spacing:{:.3f} z_height:{:.3f} left_arm_offset:{:.3f}".format(self.mid_point[0], self.y_spacing , self.mid_point[2], self.left_height_offset))
        
        self.data_pub = rospy.Publisher('/thormang3_gogoro/steering/data', Float32MultiArray, queue_size = 5)
        
        rospy.Subscriber("/robotis/sensor/imu/imu",
                         Imu, self._imu_callback)

        ## phone udp msgs from mpc ##
        rospy.Subscriber("/thormang3_gogoro/steering/break", 
                         Bool, self._break_callback)
        rospy.Subscriber("/thormang3_gogoro/steering/start", 
                         Bool, self._start_callback)
        
        rospy.Subscriber("/imu/magnetic_field", 
                         MagneticField, self._magnetic_field)
        rospy.Subscriber("/thormang3_gogoro/manipulator_steering",
                         Float32MultiArray,self._manipulator_callback)
        
        
    ### Ready Pos after FINISH INIT POS  ###
    def ReadyPos(self):
        log(self.TAG, "Ready_for_Grabbing_Position")
        
        self.kinematics.set_kinematics_pose("left_arm" , 3.0, **{ 'x': self.x_ini, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.z_height + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 3.0, **{ 'x': self.x_ini, 'y':  self.mid_point[1]-self.y_spacing, 'z': self.z_height, 'roll': -(90.00+ self.right_roll_offset), 'pitch': self.right_start_pitch, 'yaw': -self.yaw_offset })
        time.sleep(1.5)
        
    ########################################

    ###         Go to Steering Pos       ###
    def Go_Steering_pos(self):
        
        # init_steering_pos = self.manipulator(torch.Tensor([0,25]))
        # joint           =   JointState()

        # left_arm_joint = ["l_arm_sh_p1","l_arm_sh_r","l_arm_sh_p2","l_arm_el_y","l_arm_wr_r","l_arm_wr_y","l_arm_wr_p"]
        # right_arm_joint = ["r_arm_sh_p1","r_arm_sh_r","r_arm_sh_p2","r_arm_el_y","r_arm_wr_r","r_arm_wr_y","r_arm_wr_p"]
        # arm_joint = np.hstack((left_arm_joint,right_arm_joint))
        
        # joint.name      =   arm_joint
        # # joint.position  =   target_theta

        # joint.velocity  =   [ 0 for _ in arm_joint ] #max is 3.5
        # joint.effort    =   [ 0 for _ in arm_joint ]
        
        # final_steering_pos = init_steering_pos.data.numpy()
        
        # joint.position  =   init_steering_pos.data.numpy()
        # self.kinematics.publisher_(self.kinematics.set_joint_pub, joint, latch=False)
        
        self.mid_point[0] = self.x_start_pos
        self.mid_point[1] = 0.0
        self.mid_point[2] = self.z_height
        
        self.kinematics.set_kinematics_pose("left_arm" , 2.0, **{ 'x': self.mid_point[0]+self.x_left_offset, 'y':   self.mid_point[1]+self.y_spacing+self.left_space_offset, 'z': self.mid_point[2] + self.left_height_offset, 'roll': 90.00, 'pitch': self.left_pitch_offset, 'yaw': self.yaw_offset })
        self.kinematics.set_kinematics_pose("right_arm" , 2.0, **{ 'x': self.mid_point[0], 'y':  self.mid_point[1]-self.y_spacing, 'z': self.mid_point[2], 'roll': -(self.roll + self.right_roll_offset), 'pitch': self.right_pitch, 'yaw': -self.yaw_offset })
        time.sleep(1.5)
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
        time.sleep(1.5)
        
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
        
        self.gogoro_state[0] = -roll    # radian unit
        
        self.gogoro_state[1] = angvel_x
        self.gogoro_state[2] = angvel_y
        self.gogoro_state[3] = angvel_z
        
        self.gogoro_state[4] = linear_x
        self.gogoro_state[5] = linear_y
        self.gogoro_state[6] = linear_z - 9.81
        
        self.gogoro_state[7] = 0.0
        
            
        # self.gogoro_state[8] = curr steering
        
        if self.senario == "Start_Gasing":
            ## PID control ##
            error = (self.gogoro_state[0] - self.Refer_offset)
            
            self.i_accum += error
            
            if self.last_error == None:
                error_dot = 0
            else:
                error_dot = error - self.last_error
            
            P_signal = self.P_gain * error #(self.P_gain * P_scale/P_offset )* error 
            I_signal = self.I_gain * self.i_accum
            D_signal = self.D_gain * error_dot
            
            steering_angle_deg    = np.degrees( P_signal  + I_signal + D_signal)
            
            self.last_error = error
            #################
            self.action_angle = steering_angle_deg
            self.steering_control(steering_angle_deg,100)
        elif self.senario == "Steering_Control":
            ## PID control ##
            error = (self.gogoro_state[0] - self.Refer_offset)
            
            self.i_accum += error
            
            if self.last_error == None:
                error_dot = 0
            else:
                error_dot = error - self.last_error
            
            P_signal = self.P_gain * error #(self.P_gain * P_scale/P_offset )* error 
            I_signal = self.I_gain * self.i_accum
            D_signal = self.D_gain * error_dot
            
            steering_angle_deg    = np.degrees( P_signal  + I_signal + D_signal)
            
            self.last_error = error
            #################
            self.action_angle = steering_angle_deg
            self.steering_control(steering_angle_deg,0)
        elif self.senario == "breaking" :
            self.breaking()
        
    ########################################
    
    def steering_control(self,predict,gas):

        angle = predict
        gas_pitch = np.interp(gas,[i for i in np.linspace(0,100,100)],[j for j in np.linspace(25,-15,100)])

        joint           =   JointState()

        left_arm_joint = ["l_arm_sh_p1","l_arm_sh_r","l_arm_sh_p2","l_arm_el_y","l_arm_wr_r","l_arm_wr_y","l_arm_wr_p"]
        right_arm_joint = ["r_arm_sh_p1","r_arm_sh_r","r_arm_sh_p2","r_arm_el_y","r_arm_wr_r","r_arm_wr_y","r_arm_wr_p"]
        arm_joint = np.hstack((left_arm_joint,right_arm_joint))
        
        joint.name      =   arm_joint
        # joint.position  =   target_theta

        joint.velocity  =   [ 0 for _ in arm_joint ]
        joint.effort    =   [ 0 for _ in arm_joint ]

        if angle > 18:
            angle = 18
        elif angle < -18:
            angle = -18
        # NN_start_time = time.time()
        target_theta = self.manipulator(torch.Tensor([angle,gas_pitch]))
        # self.NN_time_spent = (time.time() - NN_start_time) * 1000

        joint.position  =   target_theta.data.numpy()
        self.kinematics.publisher_(self.kinematics.set_joint_pub, joint, latch=False)
    
    def breaking(self):
        # laft_arm then right_arm break joint
        
        right_break = self.right_break_position
        left_break  = self.left_break_position
        
        break_pos   = np.hstack([left_break,right_break])
        
        break_joint           =   JointState()
        break_joint.name      =   self.break_arm_joint
        
        break_joint.position  =   break_pos
        
        break_joint.velocity  =   [ 0 for _ in self.break_arm_joint ]
        break_joint.effort    =   [ 0 for _ in self.break_arm_joint ]

        self.kinematics.publisher_(self.kinematics.set_joint_pub, break_joint, latch=False)
    
    def _manipulator_callback(self,data):
        callback_data = data.data
        # print("callback_data",callback_data)
        # print("Len of callback_data",len(callback_data))
        if len(callback_data) == 2:
            manual_angle = callback_data[0]
            manual_gas   = callback_data[1]
        elif len(callback_data) == 1:
            manual_angle = callback_data[0]
            manual_gas   = 0.0
        else:
            manual_angle = 0.0
            manual_gas   = 0.0
        self.steering_control(manual_angle,manual_gas)
    
    def _magnetic_field(self,msg):
        self.gogoro_yaw = np.radians(math.atan2(msg.magnetic_field.y ,msg.magnetic_field.x) * 180/math.pi + 180)
        
    def _break_callback(self,msg):
        self.break_signal = msg.data
        if self.break_signal : 
            self.senario = "breaking"
            self.breaking()
        
    def _start_callback(self,msg):
        self.start_signal = msg.data


    
if __name__ == "__main__":
    _node_name = 'Balance_Agent'
    rospy.init_node(_node_name, anonymous=True)
    rospy.loginfo('{0} is up!'.format(_node_name))

    ### Load PPO Agent ###
    # agent = ActorCritic()    
    # model_path = "/home/hg5j9k6a/thor_ws/src/thormang3_gogoro/scripts/model_30000_20.pt"
    # agent.load_state_dict(torch.load(model_path))
    ######################
    
    steering = Thormang3Steering(tile_refer = TILE_REFER)
    
    print("FINISH ReadyPos")
    time.sleep(1.5)
    gripper = Gripper()
    
    ### init ####
    gripper.setGrippers(left=0, right=0)
    time.sleep(1.5)
    print("FINISH  Init Gripper")

    #########
    # If you want to test the grab postion, pause the program here
    # and use topic to find the position
    # rospy.spin()
    #########
    
    input("Moving to Steering Pos")
    steering.Go_Steering_pos()
    time.sleep(1)
    
    input("Ready to Start! Press will Grib tight the steering bar .")
    gripper.setGrippers(left=0.5, right=0.5)
    time.sleep(0.5)
    steering.kinematics.publisher_(steering.kinematics.module_control_pub, "none", latch=True)
    
    input("Set Refer_offset.")
    steering.Refer_offset = steering.gogoro_state[0]
    print("Refer_offset:",np.degrees(steering.Refer_offset))

    steering.steering_control(0,0)
    
    input("Press ENTER to START!!")

    print("Wait for call")
    while not(steering.start_signal):
        time.sleep(0.33)
    
    # last_time = None
    star_time = time.time()
    time_count = 0
    control_record = []
    
    # Prepare a curses window control
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)
    
    while not rospy.is_shutdown():
        # last_time = time.time()
        time_count = ( time.time() - star_time ) 
        if time_count < Hold_gasing:
            steering.senario = "Start_Gasing"
        else:
            steering.senario = "Steering_Control"
        
        control_record.append([steering.action_angle,np.degrees(steering.gogoro_state[0]-steering.Refer_offset)]) # record with action deg & error deg
        
        stdscr.clear()
        # gogoro_yaw = steering.gogoro_yaw
        # if gogoro_yaw > np.radians(180):
        #     gogoro_yaw = gogoro_yaw - np.radians(360)
        
        stdscr.addstr(f"Gogoro State (right side is positive value) \n")
        stdscr.addstr(f"Time    : {time_count:.2f} s \n")
        
        # stdscr.addstr(f"Speed       : {-steering.speed_x:.2f} m/s \n")
        # stdscr.addstr(f"Speed       : {-steering.speed_x*3.6:.2f} km/h \n")
        # stdscr.addstr(f"Yaw_Speed   : {-steering.gogoro_state[3]:.2f} rad/s \n\n")

        # stdscr.addstr(f"NN_time_cost: {steering.NN_time_spent:.2f} ms \n\n")

        stdscr.addstr(f"Roll  : {np.degrees(steering.gogoro_state[0]):.2f} deg\n")
        # stdscr.addstr(f"Yaw   : {np.degrees(gogoro_yaw):.2f} deg\n")
        stdscr.addstr(f"Steer : {steering.action_angle:.2f} deg\n\n")

        stdscr.addstr(f"PRESS 'q' TO BREAK! \n")

        c = stdscr.getch()
        if c == ord('q') or (time_count > Duration_time) :
            for _ in range(3):
                steering.senario = "breaking" 
                steering.breaking()
                time.sleep(0.1)
            break
        if (steering.break_signal):
            for _ in range(3):
                steering.senario = "breaking" 
            break
        time.sleep(0.1)

    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()
        
    control_record = np.array(control_record)

    PATH = os.path.join(PKG_PATH, f'data/{EXPERIMENT_NAME}'+ "_{}".format(EXPERIMENT_TIME))
    while os.path.exists(PATH):
        EXPERIMENT_TIME += 1
        PATH = os.path.join(PKG_PATH, f'data/{EXPERIMENT_NAME}'+ "_{}".format(EXPERIMENT_TIME))
    os.mkdir(PATH)
    
    # np.save(os.path.join(PATH, 'control_record'),  control_record)
    
    print("Finished.")