#! /usr/bin/env python3

import rospy
import numpy as np
import time
import math
from time import sleep
from std_msgs.msg import Float64
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty

from utils import *

class GogoroEnv:
    def __init__(self, action_scale=1, speed=25):
        # Create Publishers and Subscribers
        self.imu_sub = rospy.Subscriber("/robotis/sensor/imu",
                Imu, self._imu_callback)
        self.gas_pub = rospy.Publisher("/gogoro/gas/command", 
                Float64, queue_size=1)
        #self.reward_pub = rospy.Publisher("/gogoro/reward", Float64, queue_size=5)
        self.steering_pub = rospy.Publisher("/gogoro/steering/command", 
                Float64, queue_size=1)

        # Create Services
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_srv = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        # Gogoro parameters
        self.wheel_radius = 0.25 # In meters
        self.upright_tolerance = np.radians(1) # Degrees of tolerance
        self.limit_angle = np.radians(50) # Maximum angle of bike roll 
        self.bike_constant_speed = speed # Rad/s
        self.action_scale = action_scale
        # self.action_scale = 2 

        self.action_num = 0
        self.firstGasCmd = False

        # State of the Gogoro from IMU data
        # self.gogoro_state = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0,"delta_yaw": 0.0,
        #                      "curr_steering_angle": 0.0, "target_delta_yaw": 0.0, 
        #                      "angvel_x":0.0,"angvel_y":0.0,"angvel_z":0.0,
        #                      "linear_x":0.0,"linear_y":0.0,"linear_z":0.0}
        self.gogoro_state = np.zeros(9)

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
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        angvel_x = msg.angular_velocity.x
        angvel_y = msg.angular_velocity.y
        angvel_z = msg.angular_velocity.z
        
        linear_x = msg.linear_acceleration.x
        linear_y = msg.linear_acceleration.y
        linear_z = msg.linear_acceleration.z
        
        self.gogoro_state[0] = -roll
        
        self.gogoro_state[1] = angvel_x
        self.gogoro_state[2] = -angvel_y
        self.gogoro_state[3] = angvel_z
        
        self.gogoro_state[4] = linear_x
        self.gogoro_state[5] = -linear_y
        self.gogoro_state[6] = linear_z - 9.8
        
        self.gogoro_state[7] = 0.0
        
        ### Calculate Speed ###
        # self.count_speed_x = self.count_speed_x + linear_x * 0.1
        # self.count_speed_y = self.count_speed_y + linear_y * 0.1
        # self.count_speed_z = self.count_speed_z + linear_z * 0.1
        
        # self.gogoro_state[8] = curr steering
        # self.step(0)
        
        ### If done : ###
        if abs(self.gogoro_state[0]) > self.limit_angle:
            self.reset()
        #################
        
    def reset(self):
        # Reset simulation, i.e. move scooter back to initial position
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_srv()
        except rospy.ServiceException as e:
            print("[ERROR]: Failed to reset the world!")
            
        # print("Move {} times".format(self.action_num))
        self.action_num = 0
        
        self.firstGasCmd = False
        self.steering_pub.publish(0)
        self.gogoro_state[8] = 0
        
        sleep(0.05)

        # State_Value Called from imu_cb #
        
        ##################################


    def step(self, action):

        # if not self.firstGasCmd:
            # Set bike speed
        self.gas_pub.publish(self.bike_constant_speed )#+ (5 * np.random.random() - 2.5))
            # self.firstGasCmd = True
        
        self.action_num += 1
        
        # Perform action on the bike
        steering_angle = action #* np.radians(15)
        # print("steering_angle",steering_angle)
        # Publish new steering command message
        self.steering_pub.publish(np.radians(steering_angle * self.action_scale))
        # self.steering_pub.publish(-steering_angle)
        
        
        ## state : "curr_steering" = action
        self.gogoro_state[8] = action
        # print("")


if __name__ == "__main__":
    _node_name = 'Gogoro_env'
    rospy.init_node(_node_name, anonymous=True)
    rospy.loginfo('{0} is up!'.format(_node_name))
    
    env = GogoroEnv()
    
    rospy.spin()